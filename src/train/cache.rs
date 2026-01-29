use crate::precision::Precision;
use crate::tensor::Tensor;

/// Cached activations for a single transformer layer
pub(crate) struct LayerCache {
    /// Input to the layer [batch, seq, hidden]
    pub input: Tensor,
    /// After attention norm [batch, seq, hidden]
    pub normed_attn: Tensor,
    /// Q transposed for attention [batch, heads, seq, head_dim]
    pub q_for_attn: Tensor,
    /// K transposed for attention (expanded if GQA) [batch, heads, seq, head_dim]
    pub k_for_attn: Tensor,
    /// V transposed for attention (expanded if GQA) [batch, heads, seq, head_dim]
    pub v_for_attn: Tensor,
    /// Attention output before wo [batch*seq, hidden]
    pub attn_out_pre_wo: Tensor,
    /// After attention + residual [batch, seq, hidden]
    pub post_attn: Tensor,
    /// After FFN norm [batch, seq, hidden]
    pub normed_ffn: Tensor,
    /// Gate projection output [batch*seq, intermediate]
    pub gate: Tensor,
    /// Up projection output [batch*seq, intermediate]
    pub up: Tensor,
    /// After SwiGLU [batch*seq, intermediate]
    pub swiglu_out: Tensor,
    /// Dropout seed for attention output (0 if no dropout)
    pub attn_dropout_seed: u64,
    /// Dropout seed for FFN output (0 if no dropout)
    pub ffn_dropout_seed: u64,
}

/// Minimal data for checkpointed layers (activation checkpointing).
///
/// Instead of storing all intermediate activations (~112 MB per layer),
/// we store only the layer input and dropout seeds (~8 MB per layer).
/// During backward pass, activations are recomputed from this checkpoint.
pub(crate) struct CheckpointedLayerData {
    /// Layer input [batch, seq, hidden] - used to recompute all activations
    pub input: Tensor,
    /// Dropout seed for attention output (for deterministic replay)
    pub attn_dropout_seed: u64,
    /// Dropout seed for FFN output (for deterministic replay)
    pub ffn_dropout_seed: u64,
}

/// Either full cache or checkpoint for a layer.
///
/// Used during forward pass with activation checkpointing enabled.
/// Checkpointed layers store only minimal data and recompute activations during backward.
pub(crate) enum LayerCacheVariant {
    /// Full cache with all intermediate activations
    Full(LayerCache),
    /// Minimal checkpoint data for recomputation
    Checkpointed(CheckpointedLayerData),
}


/// Cached activations for the full forward pass
pub(crate) struct ForwardCache {
    /// Per-layer caches
    pub layers: Vec<LayerCache>,
    /// Hidden state before final norm [batch, seq, hidden] (output of last layer)
    pub pre_final_norm: Tensor,
    /// After final norm [batch*seq, hidden]
    pub final_hidden: Tensor,
}

/// Cached activations for forward pass with activation checkpointing.
///
/// Some layers store full cache, others store only checkpoint data.
pub(crate) struct ForwardCacheCheckpointed {
    /// Per-layer caches (either full or checkpointed)
    pub layers: Vec<LayerCacheVariant>,
    /// Hidden state before final norm [batch, seq, hidden] (output of last layer)
    pub pre_final_norm: Tensor,
    /// After final norm [batch*seq, hidden]
    pub final_hidden: Tensor,
}

/// Gradients for a single transformer layer
#[derive(Clone)]
pub(crate) struct LayerGradients {
    pub grad_input: Tensor,
    pub grad_attn_norm: Tensor,
    pub grad_ffn_norm: Tensor,
    pub grad_wq: Tensor,
    pub grad_wk: Tensor,
    pub grad_wv: Tensor,
    pub grad_wo: Tensor,
    pub grad_w_gate: Tensor,
    pub grad_w_up: Tensor,
    pub grad_w_down: Tensor,
}

/// Accumulated gradients for a single transformer layer (used for gradient accumulation)
pub(crate) struct AccumulatedLayerGradients {
    pub grad_attn_norm: Tensor,
    pub grad_ffn_norm: Tensor,
    pub grad_wq: Tensor,
    pub grad_wk: Tensor,
    pub grad_wv: Tensor,
    pub grad_wo: Tensor,
    pub grad_w_gate: Tensor,
    pub grad_w_up: Tensor,
    pub grad_w_down: Tensor,
}

impl AccumulatedLayerGradients {
    /// Create zero-initialized accumulated gradients matching model shapes
    pub fn zeros(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        Self {
            grad_attn_norm: Tensor::zeros(&[hidden_dim], Precision::FP32),
            grad_ffn_norm: Tensor::zeros(&[hidden_dim], Precision::FP32),
            grad_wq: Tensor::zeros(&[q_dim, hidden_dim], Precision::FP32),
            grad_wk: Tensor::zeros(&[kv_dim, hidden_dim], Precision::FP32),
            grad_wv: Tensor::zeros(&[kv_dim, hidden_dim], Precision::FP32),
            grad_wo: Tensor::zeros(&[hidden_dim, q_dim], Precision::FP32),
            grad_w_gate: Tensor::zeros(&[intermediate_dim, hidden_dim], Precision::FP32),
            grad_w_up: Tensor::zeros(&[intermediate_dim, hidden_dim], Precision::FP32),
            grad_w_down: Tensor::zeros(&[hidden_dim, intermediate_dim], Precision::FP32),
        }
    }
}

/// Accumulated gradients for the full model (used for gradient accumulation)
pub(crate) struct AccumulatedGradients {
    pub grad_embed: Tensor,
    pub grad_final_norm: Tensor,
    pub layer_grads: Vec<AccumulatedLayerGradients>,
}

impl AccumulatedGradients {
    /// Create zero-initialized accumulated gradients matching model shapes
    pub fn zeros(
        vocab_size: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
    ) -> Self {
        Self {
            grad_embed: Tensor::zeros(&[vocab_size, hidden_dim], Precision::FP32),
            grad_final_norm: Tensor::zeros(&[hidden_dim], Precision::FP32),
            layer_grads: (0..num_layers)
                .map(|_| {
                    AccumulatedLayerGradients::zeros(
                        hidden_dim,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        intermediate_dim,
                    )
                })
                .collect(),
        }
    }
}
