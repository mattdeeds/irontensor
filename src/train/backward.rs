//! Backward pass implementation for training.
//!
//! Contains methods for computing gradients through transformer layers.

use crate::nn::TransformerBlock;
use crate::ops::{dropout_backward, rmsnorm_backward, rope_backward, swiglu_backward, transpose_for_attention, transpose_for_attention_backward};
use crate::tensor::Tensor;

use super::cache::{LayerCache, LayerGradients};
use super::helpers::{
    add_tensors, add3_tensors, attention_backward_recompute, ensure_fp32, linear_backward, repeat_kv_backward,
};
use super::trainer::Trainer;

impl Trainer {
    /// Backward pass through a transformer layer.
    ///
    /// Computes gradients for all parameters in the layer given the gradient
    /// of the layer's output. Returns gradients for both the layer input and
    /// all learnable parameters.
    pub(super) fn backward_transformer_layer(
        &self,
        grad_output: &Tensor,
        cache: &LayerCache,
        layer: &TransformerBlock,
        batch_size: usize,
        seq_len: usize,
    ) -> LayerGradients {
        let hidden_dim = layer.hidden_dim;
        let n = batch_size * seq_len;

        // Reshape grad_output to 3D if needed
        let grad_out_3d = if grad_output.shape().len() == 2 {
            grad_output.view(&[batch_size, seq_len, hidden_dim])
        } else {
            grad_output.clone()
        };

        // ===== FFN Backward =====
        // Gradient flows through residual
        let grad_ffn_out = grad_out_3d.clone();
        let grad_post_attn_from_ffn = grad_out_3d.clone();

        // Backward through FFN dropout (if applied in forward)
        let grad_ffn_out = if cache.ffn_dropout_seed != 0 {
            dropout_backward(&grad_ffn_out, self.model.config.ffn_dropout, cache.ffn_dropout_seed).unwrap()
        } else {
            grad_ffn_out
        };

        // Backward through down projection
        let grad_ffn_out_2d = grad_ffn_out.view(&[n, hidden_dim]);
        let (grad_swiglu, grad_w_down) =
            linear_backward(&grad_ffn_out_2d, &cache.swiglu_out, &layer.ffn.w_down.weight);

        // Backward through SwiGLU
        let (grad_gate, grad_up) = swiglu_backward(&grad_swiglu, &cache.gate, &cache.up);

        // Backward through gate and up projections
        let normed_ffn_2d = cache.normed_ffn.view(&[n, hidden_dim]);
        let (grad_normed_ffn_from_gate, grad_w_gate) =
            linear_backward(&grad_gate, &normed_ffn_2d, &layer.ffn.w_gate.weight);
        let (grad_normed_ffn_from_up, grad_w_up) =
            linear_backward(&grad_up, &normed_ffn_2d, &layer.ffn.w_up.weight);
        let grad_normed_ffn = add_tensors(&grad_normed_ffn_from_gate, &grad_normed_ffn_from_up);

        // Backward through FFN norm (convert BF16 gamma to FP32 if needed)
        let grad_normed_ffn_3d = grad_normed_ffn.view(&[batch_size, seq_len, hidden_dim]);
        let ffn_norm_fp32 = ensure_fp32(&layer.ffn_norm);
        let (grad_post_attn_from_norm, grad_ffn_norm) =
            rmsnorm_backward(&grad_normed_ffn_3d, &cache.post_attn, &ffn_norm_fp32, layer.norm_eps);

        // Combine gradients for post_attn
        let grad_post_attn = add_tensors(&grad_post_attn_from_ffn, &grad_post_attn_from_norm);

        // ===== Attention Backward =====
        // Gradient flows through residual
        let grad_attn_out = grad_post_attn.clone();
        let grad_input_from_attn_residual = grad_post_attn.clone();

        // Backward through attention dropout (if applied in forward)
        let grad_attn_out = if cache.attn_dropout_seed != 0 {
            dropout_backward(&grad_attn_out, self.model.config.attn_dropout, cache.attn_dropout_seed).unwrap()
        } else {
            grad_attn_out
        };

        // Backward through output projection
        let grad_attn_out_2d = grad_attn_out.view(&[n, hidden_dim]);
        let (grad_attn_pre_wo, grad_wo) =
            linear_backward(&grad_attn_out_2d, &cache.attn_out_pre_wo, &layer.attention.wo.weight);

        // ===== Proper Attention Backward =====
        let num_heads = layer.attention.num_heads;
        let num_kv_heads = layer.attention.num_kv_heads;
        let head_dim = layer.attention.head_dim;

        // Reshape grad_attn_pre_wo from [n, hidden] to [batch, seq, heads, head_dim]
        let grad_attn_4d = grad_attn_pre_wo.view(&[batch_size, seq_len, num_heads, head_dim]);

        // Transpose to attention layout: [batch, heads, seq, head_dim]
        let grad_attn_transposed = transpose_for_attention(&grad_attn_4d);

        // Call attention backward (recomputes attention weights from Q, K, V)
        // This trades compute for memory - no need to cache O(seq^2) attention weights
        let (grad_q_attn, grad_k_attn, grad_v_attn) = attention_backward_recompute(
            &grad_attn_transposed,
            &cache.q_for_attn,
            &cache.k_for_attn,
            &cache.v_for_attn,
        );

        // Transpose gradients back: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        let grad_q_4d = transpose_for_attention_backward(&grad_q_attn, batch_size, seq_len, num_heads, head_dim);
        let grad_k_expanded = transpose_for_attention_backward(&grad_k_attn, batch_size, seq_len, num_heads, head_dim);
        let grad_v_expanded = transpose_for_attention_backward(&grad_v_attn, batch_size, seq_len, num_heads, head_dim);

        // If GQA, reduce K and V gradients back to KV heads
        let (grad_k_4d, grad_v_4d) = if num_kv_heads != num_heads {
            (
                repeat_kv_backward(&grad_k_expanded, batch_size, seq_len, num_heads, num_kv_heads, head_dim),
                repeat_kv_backward(&grad_v_expanded, batch_size, seq_len, num_heads, num_kv_heads, head_dim),
            )
        } else {
            (grad_k_expanded, grad_v_expanded)
        };

        // Backward through RoPE for Q
        let grad_q_pre_rope = rope_backward(&grad_q_4d, layer.attention.rope_base, 0);

        // Backward through RoPE for K
        let grad_k_pre_rope = rope_backward(&grad_k_4d, layer.attention.rope_base, 0);

        // Reshape to 2D for linear backward: [n, qkv_dim]
        let grad_q_proj = grad_q_pre_rope.view(&[n, num_heads * head_dim]);
        let grad_k_proj = grad_k_pre_rope.view(&[n, num_kv_heads * head_dim]);
        let grad_v_proj = grad_v_4d.view(&[n, num_kv_heads * head_dim]);

        // Backward through Q, K, V projections
        let normed_attn_2d = cache.normed_attn.view(&[n, hidden_dim]);

        let (grad_normed_attn_from_q, grad_wq) =
            linear_backward(&grad_q_proj, &normed_attn_2d, &layer.attention.wq.weight);
        let (grad_normed_attn_from_k, grad_wk) =
            linear_backward(&grad_k_proj, &normed_attn_2d, &layer.attention.wk.weight);
        let (grad_normed_attn_from_v, grad_wv) =
            linear_backward(&grad_v_proj, &normed_attn_2d, &layer.attention.wv.weight);

        // Sum gradients from Q, K, V paths (fused 3-way add)
        let grad_normed_attn = add3_tensors(
            &grad_normed_attn_from_q,
            &grad_normed_attn_from_k,
            &grad_normed_attn_from_v,
        );

        // Backward through attention norm (convert BF16 gamma to FP32 if needed)
        let grad_normed_attn_3d = grad_normed_attn.view(&[batch_size, seq_len, hidden_dim]);
        let attn_norm_fp32 = ensure_fp32(&layer.attn_norm);
        let (grad_input_from_norm, grad_attn_norm) =
            rmsnorm_backward(&grad_normed_attn_3d, &cache.input, &attn_norm_fp32, layer.norm_eps);

        // Combine gradients for layer input
        let grad_input = add_tensors(&grad_input_from_attn_residual, &grad_input_from_norm);

        LayerGradients {
            grad_input,
            grad_attn_norm,
            grad_ffn_norm,
            grad_wq,
            grad_wk,
            grad_wv,
            grad_wo,
            grad_w_gate,
            grad_w_up,
            grad_w_down,
        }
    }
}
