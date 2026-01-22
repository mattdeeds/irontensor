use crate::tensor::Tensor;

/// Cached activations for a single transformer layer
pub(crate) struct LayerCache {
    /// Input to the layer [batch, seq, hidden]
    pub input: Tensor,
    /// After attention norm [batch, seq, hidden]
    pub normed_attn: Tensor,
    /// Q after projection [batch*seq, qkv_dim]
    pub q_proj: Tensor,
    /// K after projection [batch*seq, kv_dim]
    pub k_proj: Tensor,
    /// V after projection [batch*seq, kv_dim]
    pub v_proj: Tensor,
    /// Q after RoPE [batch, heads, seq, head_dim]
    pub q_rope: Tensor,
    /// K after RoPE [batch, kv_heads, seq, head_dim]
    pub k_rope: Tensor,
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
}

/// Cached activations for the full forward pass
pub(crate) struct ForwardCache {
    /// Embedded tokens [batch*seq, hidden]
    pub embedded: Tensor,
    /// Per-layer caches
    pub layers: Vec<LayerCache>,
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
