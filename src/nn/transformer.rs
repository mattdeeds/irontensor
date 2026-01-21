use crate::ops::{add, rmsnorm};
use crate::optim::ParamState;
use crate::tensor::Tensor;

use super::attention::{MultiHeadAttention, MultiHeadAttentionState};
use super::ffn::{FeedForward, FeedForwardState};

/// Transformer block (Llama-style)
///
/// Architecture:
/// ```text
/// x -> RMSNorm -> Attention -> + residual -> RMSNorm -> FFN -> + residual -> output
/// ```
pub struct TransformerBlock {
    /// Attention layer
    pub attention: MultiHeadAttention,
    /// Feed-forward network
    pub ffn: FeedForward,
    /// Pre-attention normalization
    pub attn_norm: Tensor,
    /// Pre-FFN normalization
    pub ffn_norm: Tensor,
    /// RMSNorm epsilon
    pub norm_eps: f32,
    /// Hidden dimension
    pub hidden_dim: usize,
}

impl TransformerBlock {
    /// Create a new transformer block
    ///
    /// - `hidden_dim`: Model hidden dimension
    /// - `num_heads`: Number of attention heads
    /// - `num_kv_heads`: Number of KV heads (for GQA)
    /// - `intermediate_dim`: FFN intermediate dimension
    /// - `rope_base`: RoPE base frequency
    /// - `norm_eps`: RMSNorm epsilon
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
        rope_base: f32,
        norm_eps: f32,
    ) -> Self {
        let attention = MultiHeadAttention::new(hidden_dim, num_heads, num_kv_heads, rope_base);
        let ffn = FeedForward::new(hidden_dim, intermediate_dim);

        // Initialize norm weights to 1.0
        let attn_norm = Tensor::from_f32_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]);
        let ffn_norm = Tensor::from_f32_slice(&vec![1.0f32; hidden_dim], &[hidden_dim]);

        Self {
            attention,
            ffn,
            attn_norm,
            ffn_norm,
            norm_eps,
            hidden_dim,
        }
    }

    /// Forward pass
    ///
    /// Input shape: [batch, seq_len, hidden_dim]
    /// Output shape: [batch, seq_len, hidden_dim]
    pub fn forward(&self, x: &Tensor, position_offset: usize, causal: bool) -> Tensor {
        // Pre-attention norm + attention + residual
        let normed = rmsnorm(x, &self.attn_norm, self.norm_eps);
        let attn_out = self.attention.forward(&normed, position_offset, causal);
        let x = add(x, &attn_out);

        // Pre-FFN norm + FFN + residual
        let normed = rmsnorm(&x, &self.ffn_norm, self.norm_eps);
        let ffn_out = self.ffn.forward(&normed);
        add(&x, &ffn_out)
    }

    /// Get total parameter count
    pub fn num_params(&self) -> usize {
        self.attention.num_params() + self.ffn.num_params() + 2 * self.hidden_dim // norm weights
    }
}

/// Optimizer state for TransformerBlock
pub struct TransformerBlockState {
    pub attention_state: MultiHeadAttentionState,
    pub ffn_state: FeedForwardState,
    pub attn_norm_state: ParamState,
    pub ffn_norm_state: ParamState,
}

impl TransformerBlockState {
    pub fn new(block: &TransformerBlock) -> Self {
        Self {
            attention_state: MultiHeadAttentionState::new(&block.attention),
            ffn_state: FeedForwardState::new(&block.ffn),
            attn_norm_state: ParamState::new(block.attn_norm.shape()),
            ffn_norm_state: ParamState::new(block.ffn_norm.shape()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_block_forward() {
        let hidden_dim = 64;
        let num_heads = 4;
        let intermediate_dim = 128;
        let batch = 2;
        let seq_len = 8;

        let block = TransformerBlock::new(hidden_dim, num_heads, num_heads, intermediate_dim, 10000.0, 1e-5);

        let input_data: Vec<f32> = (0..batch * seq_len * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, hidden_dim]);

        let output = block.forward(&input, 0, true);

        assert_eq!(output.shape(), &[batch, seq_len, hidden_dim]);

        // Output should be different from input (due to attention and FFN)
        let input_sum: f32 = input.as_f32_slice().iter().sum();
        let output_sum: f32 = output.as_f32_slice().iter().sum();
        assert!((input_sum - output_sum).abs() > 1e-3);
    }

    #[test]
    fn test_transformer_block_num_params() {
        let block = TransformerBlock::new(512, 8, 8, 1024, 10000.0, 1e-5);

        // Attention: 4 * 512 * 512 = 1,048,576
        // FFN: 3 * 512 * 1024 = 1,572,864
        // Norms: 2 * 512 = 1,024
        // Total: 2,622,464
        let expected = 4 * 512 * 512 + 3 * 512 * 1024 + 2 * 512;
        assert_eq!(block.num_params(), expected);
    }
}
