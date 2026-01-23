//! Forward pass implementation for training.
//!
//! Contains methods for forward pass with activation caching needed for backward pass.

use crate::nn::TransformerBlock;
use crate::ops::{embedding, flash_attention, rmsnorm, swiglu, transpose_for_attention, transpose_from_attention};
use crate::profile::Profiler;
use crate::tensor::Tensor;

use super::cache::{ForwardCache, LayerCache};
use super::helpers::{ensure_fp32, linear_forward, repeat_kv};
use super::trainer::Trainer;
use crate::ops::rope;

impl Trainer {
    /// Forward pass with activation caching for backward.
    ///
    /// Returns a `ForwardCache` containing all intermediate activations needed
    /// for the backward pass.
    pub(super) fn forward_with_cache(
        &self,
        input_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> ForwardCache {
        let hidden_dim = self.model.config.hidden_dim;
        let n = batch_size * seq_len;

        // Embedding lookup (convert BF16 weights to FP32 if needed)
        let embed_fp32 = ensure_fp32(&self.model.embed_tokens);
        let embedded = embedding(&embed_fp32, input_ids);
        let mut hidden = embedded.view(&[batch_size, seq_len, hidden_dim]);

        // Process each transformer layer
        let mut layer_caches = Vec::new();
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            Profiler::set_layer(Some(layer_idx));
            let (new_hidden, cache) =
                self.forward_layer_with_cache(&hidden, layer, batch_size, seq_len);
            layer_caches.push(cache);
            hidden = new_hidden;
        }
        Profiler::set_layer(None);

        // Store hidden state before final norm (for backward pass through RMSNorm)
        let pre_final_norm = hidden.clone();

        // Final norm (convert BF16 gamma to FP32 if needed)
        let final_norm_fp32 = ensure_fp32(&self.model.final_norm);
        let final_hidden = rmsnorm(&hidden, &final_norm_fp32, self.model.config.norm_eps).unwrap();
        let final_hidden_2d = final_hidden.view(&[n, hidden_dim]);

        ForwardCache {
            embedded,
            layers: layer_caches,
            pre_final_norm,
            final_hidden: final_hidden_2d,
        }
    }

    /// Forward pass through a single transformer layer with caching.
    ///
    /// Returns the output hidden states and a `LayerCache` containing all
    /// intermediate activations for that layer.
    pub(super) fn forward_layer_with_cache(
        &self,
        x: &Tensor,
        layer: &TransformerBlock,
        batch_size: usize,
        seq_len: usize,
    ) -> (Tensor, LayerCache) {
        let hidden_dim = layer.hidden_dim;
        let num_heads = layer.attention.num_heads;
        let num_kv_heads = layer.attention.num_kv_heads;
        let head_dim = layer.attention.head_dim;
        let n = batch_size * seq_len;

        // Store input
        let input = x.clone();

        // Attention norm (convert BF16 gamma to FP32 if needed)
        let attn_norm_fp32 = ensure_fp32(&layer.attn_norm);
        let normed_attn = rmsnorm(x, &attn_norm_fp32, layer.norm_eps).unwrap();

        // Q, K, V projections
        let normed_2d = normed_attn.view(&[n, hidden_dim]);
        let q_proj = linear_forward(&normed_2d, &layer.attention.wq.weight);
        let k_proj = linear_forward(&normed_2d, &layer.attention.wk.weight);
        let v_proj = linear_forward(&normed_2d, &layer.attention.wv.weight);

        // Reshape for attention
        let q = q_proj.view(&[batch_size, seq_len, num_heads, head_dim]);
        let k = k_proj.view(&[batch_size, seq_len, num_kv_heads, head_dim]);
        let v = v_proj.view(&[batch_size, seq_len, num_kv_heads, head_dim]);

        // Apply RoPE
        let q_rope = rope(&q, layer.attention.rope_base, 0);
        let k_rope = rope(&k, layer.attention.rope_base, 0);

        // Expand KV heads if GQA
        let (k_expanded, v_expanded) = if num_kv_heads != num_heads {
            (
                repeat_kv(&k_rope, batch_size, seq_len, num_heads, num_kv_heads, head_dim),
                repeat_kv(&v, batch_size, seq_len, num_heads, num_kv_heads, head_dim),
            )
        } else {
            (k_rope.clone(), v.clone())
        };

        // Transpose for attention: [batch, heads, seq, head_dim]
        let q_t = transpose_for_attention(&q_rope);
        let k_t = transpose_for_attention(&k_expanded);
        let v_t = transpose_for_attention(&v_expanded);

        // Cache Q, K, V for backward pass (recomputing attention backward)
        // q_t, k_t, v_t: [batch, heads, seq, head_dim]
        let q_for_attn = q_t.clone();
        let k_for_attn = k_t.clone();
        let v_for_attn = v_t.clone();

        // Use FlashAttention for fused, memory-efficient attention
        // FlashAttention computes: softmax(Q @ K^T / sqrt(d) + causal_mask) @ V
        // Input/output: [batch, heads, seq, head_dim]
        let attn_out_4d = flash_attention(&q_t, &k_t, &v_t, true);

        // Transpose back: [batch, seq, heads, head_dim]
        let attn_out = transpose_from_attention(&attn_out_4d, batch_size, seq_len, num_heads, head_dim);
        let attn_out_2d = attn_out.view(&[n, num_heads * head_dim]);

        // Output projection
        let attn_out_pre_wo = attn_out_2d.clone();
        let attn_projected = linear_forward(&attn_out_2d, &layer.attention.wo.weight);
        let attn_projected = attn_projected.view(&[batch_size, seq_len, hidden_dim]);

        // Residual connection
        let post_attn = crate::ops::add(x, &attn_projected).unwrap();

        // FFN norm (convert BF16 gamma to FP32 if needed)
        let ffn_norm_fp32 = ensure_fp32(&layer.ffn_norm);
        let normed_ffn = rmsnorm(&post_attn, &ffn_norm_fp32, layer.norm_eps).unwrap();

        // FFN: gate and up projections
        let normed_ffn_2d = normed_ffn.view(&[n, hidden_dim]);
        let gate = linear_forward(&normed_ffn_2d, &layer.ffn.w_gate.weight);
        let up = linear_forward(&normed_ffn_2d, &layer.ffn.w_up.weight);

        // SwiGLU
        let swiglu_out = swiglu(&gate, &up).unwrap();

        // Down projection
        let ffn_out = linear_forward(&swiglu_out, &layer.ffn.w_down.weight);
        let ffn_out = ffn_out.view(&[batch_size, seq_len, hidden_dim]);

        // Residual connection
        let output = crate::ops::add(&post_attn, &ffn_out).unwrap();

        let cache = LayerCache {
            input,
            normed_attn,
            q_proj,
            k_proj,
            v_proj,
            q_rope,
            k_rope,
            q_for_attn,
            k_for_attn,
            v_for_attn,
            attn_out_pre_wo,
            post_attn,
            normed_ffn,
            gate,
            up,
            swiglu_out,
        };

        (output, cache)
    }
}
