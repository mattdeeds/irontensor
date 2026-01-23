use crate::ops::{attention, rope};
use crate::optim::ParamState;
use crate::profile::context;
use crate::tensor::Tensor;

use super::linear::Linear;

/// Multi-head attention with RoPE positional encoding
///
/// Implements the attention mechanism used in Llama-style models:
/// - Separate Q, K, V projections
/// - Rotary Position Embeddings (RoPE)
/// - Grouped Query Attention (GQA) support
/// - Causal masking for autoregressive generation
pub struct MultiHeadAttention {
    /// Query projection [hidden_dim, num_heads * head_dim]
    pub wq: Linear,
    /// Key projection [hidden_dim, num_kv_heads * head_dim]
    pub wk: Linear,
    /// Value projection [hidden_dim, num_kv_heads * head_dim]
    pub wv: Linear,
    /// Output projection [num_heads * head_dim, hidden_dim]
    pub wo: Linear,

    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// RoPE base frequency
    pub rope_base: f32,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    ///
    /// - `hidden_dim`: Model hidden dimension
    /// - `num_heads`: Number of query attention heads
    /// - `num_kv_heads`: Number of key-value heads (for GQA, typically num_heads or num_heads/8)
    /// - `rope_base`: RoPE base frequency (typically 10000.0)
    pub fn new(hidden_dim: usize, num_heads: usize, num_kv_heads: usize, rope_base: f32) -> Self {
        assert!(
            num_heads % num_kv_heads == 0,
            "num_heads must be divisible by num_kv_heads"
        );

        let head_dim = hidden_dim / num_heads;

        let wq = Linear::new(hidden_dim, num_heads * head_dim, false);
        let wk = Linear::new(hidden_dim, num_kv_heads * head_dim, false);
        let wv = Linear::new(hidden_dim, num_kv_heads * head_dim, false);
        let wo = Linear::new(num_heads * head_dim, hidden_dim, false);

        Self {
            wq,
            wk,
            wv,
            wo,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            rope_base,
        }
    }

    /// Forward pass
    ///
    /// Input shape: [batch, seq_len, hidden_dim]
    /// Output shape: [batch, seq_len, hidden_dim]
    ///
    /// - `position_offset`: Starting position for RoPE (for KV cache continuation)
    /// - `causal`: Whether to apply causal masking
    pub fn forward(&self, x: &Tensor, position_offset: usize, causal: bool) -> Tensor {
        let _ctx = context("attn");

        let shape = x.shape();
        assert_eq!(shape.len(), 3, "Input must be [batch, seq_len, hidden_dim]");

        let batch = shape[0];
        let seq_len = shape[1];
        let hidden = shape[2];
        assert_eq!(hidden, self.hidden_dim);

        // Project to Q, K, V
        let (q, k, v) = {
            let _ctx = context("qkv");
            let q = self.wq.forward(x); // [batch, seq_len, num_heads * head_dim]
            let k = self.wk.forward(x); // [batch, seq_len, num_kv_heads * head_dim]
            let v = self.wv.forward(x); // [batch, seq_len, num_kv_heads * head_dim]
            (q, k, v)
        };

        // Reshape to [batch, seq_len, num_heads, head_dim]
        let q = self.reshape_for_attention(&q, batch, seq_len, self.num_heads);
        let k = self.reshape_for_attention(&k, batch, seq_len, self.num_kv_heads);
        let v = self.reshape_for_attention(&v, batch, seq_len, self.num_kv_heads);

        // Apply RoPE to Q and K
        let (q, k) = {
            let _ctx = context("rope");
            let q = rope(&q, self.rope_base, position_offset);
            let k = rope(&k, self.rope_base, position_offset);
            (q, k)
        };

        // Expand KV heads if using GQA
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let k = self.repeat_kv(&k, batch, seq_len);
            let v = self.repeat_kv(&v, batch, seq_len);
            (k, v)
        } else {
            (k, v)
        };

        // Transpose to [batch, num_heads, seq_len, head_dim] for attention
        let q = self.transpose_for_scores(&q, batch, seq_len, self.num_heads);
        let k = self.transpose_for_scores(&k, batch, seq_len, self.num_heads);
        let v = self.transpose_for_scores(&v, batch, seq_len, self.num_heads);

        // Compute attention
        let attn_output = {
            let _ctx = context("scores");
            attention(&q, &k, &v, causal)
        };

        // Transpose back to [batch, seq_len, num_heads, head_dim]
        let attn_output = self.transpose_from_scores(&attn_output, batch, seq_len, self.num_heads);

        // Reshape to [batch, seq_len, hidden_dim]
        let attn_output = self.reshape_from_attention(&attn_output, batch, seq_len);

        // Output projection
        {
            let _ctx = context("proj");
            self.wo.forward(&attn_output)
        }
    }

    /// Reshape from [batch, seq, heads * head_dim] to [batch, seq, heads, head_dim]
    fn reshape_for_attention(
        &self,
        x: &Tensor,
        batch: usize,
        seq_len: usize,
        num_heads: usize,
    ) -> Tensor {
        let data = x.as_f32_slice().to_vec();
        Tensor::from_f32_slice(&data, &[batch, seq_len, num_heads, self.head_dim])
    }

    /// Reshape from [batch, seq, heads, head_dim] to [batch, seq, heads * head_dim]
    fn reshape_from_attention(&self, x: &Tensor, batch: usize, seq_len: usize) -> Tensor {
        let data = x.as_f32_slice().to_vec();
        Tensor::from_f32_slice(&data, &[batch, seq_len, self.num_heads * self.head_dim])
    }

    /// Transpose from [batch, seq, heads, head_dim] to [batch, heads, seq, head_dim]
    fn transpose_for_scores(
        &self,
        x: &Tensor,
        batch: usize,
        seq_len: usize,
        num_heads: usize,
    ) -> Tensor {
        let data = x.as_f32_slice();
        let mut transposed = vec![0.0f32; data.len()];

        for b in 0..batch {
            for s in 0..seq_len {
                for h in 0..num_heads {
                    for d in 0..self.head_dim {
                        let src_idx =
                            b * seq_len * num_heads * self.head_dim + s * num_heads * self.head_dim
                                + h * self.head_dim
                                + d;
                        let dst_idx =
                            b * num_heads * seq_len * self.head_dim + h * seq_len * self.head_dim
                                + s * self.head_dim
                                + d;
                        transposed[dst_idx] = data[src_idx];
                    }
                }
            }
        }

        Tensor::from_f32_slice(&transposed, &[batch, num_heads, seq_len, self.head_dim])
    }

    /// Transpose from [batch, heads, seq, head_dim] to [batch, seq, heads, head_dim]
    fn transpose_from_scores(
        &self,
        x: &Tensor,
        batch: usize,
        seq_len: usize,
        num_heads: usize,
    ) -> Tensor {
        let data = x.as_f32_slice();
        let mut transposed = vec![0.0f32; data.len()];

        for b in 0..batch {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    for d in 0..self.head_dim {
                        let src_idx =
                            b * num_heads * seq_len * self.head_dim + h * seq_len * self.head_dim
                                + s * self.head_dim
                                + d;
                        let dst_idx =
                            b * seq_len * num_heads * self.head_dim + s * num_heads * self.head_dim
                                + h * self.head_dim
                                + d;
                        transposed[dst_idx] = data[src_idx];
                    }
                }
            }
        }

        Tensor::from_f32_slice(&transposed, &[batch, seq_len, num_heads, self.head_dim])
    }

    /// Repeat KV heads for GQA: [batch, seq, kv_heads, head_dim] -> [batch, seq, num_heads, head_dim]
    fn repeat_kv(&self, x: &Tensor, batch: usize, seq_len: usize) -> Tensor {
        let repeats = self.num_heads / self.num_kv_heads;
        let data = x.as_f32_slice();
        let mut expanded = vec![0.0f32; batch * seq_len * self.num_heads * self.head_dim];

        for b in 0..batch {
            for s in 0..seq_len {
                for kv_h in 0..self.num_kv_heads {
                    for r in 0..repeats {
                        let h = kv_h * repeats + r;
                        for d in 0..self.head_dim {
                            let src_idx = b * seq_len * self.num_kv_heads * self.head_dim
                                + s * self.num_kv_heads * self.head_dim
                                + kv_h * self.head_dim
                                + d;
                            let dst_idx = b * seq_len * self.num_heads * self.head_dim
                                + s * self.num_heads * self.head_dim
                                + h * self.head_dim
                                + d;
                            expanded[dst_idx] = data[src_idx];
                        }
                    }
                }
            }
        }

        Tensor::from_f32_slice(&expanded, &[batch, seq_len, self.num_heads, self.head_dim])
    }

    /// Get total parameter count
    pub fn num_params(&self) -> usize {
        self.wq.num_params() + self.wk.num_params() + self.wv.num_params() + self.wo.num_params()
    }

    /// Convert all weights to BF16
    pub fn to_bf16(&mut self) {
        self.wq.to_bf16();
        self.wk.to_bf16();
        self.wv.to_bf16();
        self.wo.to_bf16();
    }

    /// Convert all weights to FP32
    pub fn to_f32(&mut self) {
        self.wq.to_f32();
        self.wk.to_f32();
        self.wv.to_f32();
        self.wo.to_f32();
    }
}

/// Optimizer state for MultiHeadAttention
pub struct MultiHeadAttentionState {
    pub wq_state: ParamState,
    pub wk_state: ParamState,
    pub wv_state: ParamState,
    pub wo_state: ParamState,
}

impl MultiHeadAttentionState {
    pub fn new(layer: &MultiHeadAttention) -> Self {
        Self {
            wq_state: ParamState::new(layer.wq.weight.shape()),
            wk_state: ParamState::new(layer.wk.weight.shape()),
            wv_state: ParamState::new(layer.wv.weight.shape()),
            wo_state: ParamState::new(layer.wo.weight.shape()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_forward() {
        let hidden_dim = 64;
        let num_heads = 4;
        let batch = 2;
        let seq_len = 8;

        let attn = MultiHeadAttention::new(hidden_dim, num_heads, num_heads, 10000.0);

        let input_data: Vec<f32> = (0..batch * seq_len * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, hidden_dim]);

        let output = attn.forward(&input, 0, true);

        assert_eq!(output.shape(), &[batch, seq_len, hidden_dim]);
    }

    #[test]
    fn test_attention_gqa() {
        // Test Grouped Query Attention (4 heads, 2 KV heads)
        let hidden_dim = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let batch = 1;
        let seq_len = 4;

        let attn = MultiHeadAttention::new(hidden_dim, num_heads, num_kv_heads, 10000.0);

        let input_data: Vec<f32> = (0..batch * seq_len * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, hidden_dim]);

        let output = attn.forward(&input, 0, true);

        assert_eq!(output.shape(), &[batch, seq_len, hidden_dim]);
    }

    #[test]
    fn test_attention_num_params() {
        let attn = MultiHeadAttention::new(512, 8, 8, 10000.0);
        // Q, K, V, O projections: 4 * 512 * 512 = 1,048,576
        assert_eq!(attn.num_params(), 4 * 512 * 512);
    }
}
