use crate::ops::{embedding, matmul, rmsnorm};
use crate::optim::ParamState;
use crate::tensor::Tensor;

use super::transformer::{TransformerBlock, TransformerBlockState};

/// Model configuration
#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RoPE base frequency
    pub rope_base: f32,
    /// RMSNorm epsilon
    pub norm_eps: f32,
    /// Whether to tie embedding and output weights
    pub tie_weights: bool,
}

impl ModelConfig {
    /// Create a small model config (good for testing)
    pub fn tiny() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 4,
            intermediate_dim: 512,
            max_seq_len: 512,
            rope_base: 10000.0,
            norm_eps: 1e-5,
            tie_weights: true,
        }
    }

    /// Create a small model (similar to Llama 7B architecture but scaled down)
    pub fn small() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 512,
            num_layers: 8,
            num_heads: 8,
            num_kv_heads: 8,
            intermediate_dim: 1024,
            max_seq_len: 2048,
            rope_base: 10000.0,
            norm_eps: 1e-5,
            tie_weights: true,
        }
    }

    /// Create a medium model
    pub fn medium() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 1024,
            num_layers: 16,
            num_heads: 16,
            num_kv_heads: 16,
            intermediate_dim: 2816,
            max_seq_len: 2048,
            rope_base: 10000.0,
            norm_eps: 1e-5,
            tie_weights: true,
        }
    }

    /// Compute the number of parameters for this config
    pub fn num_params(&self) -> usize {
        // Embedding
        let embed_params = self.vocab_size * self.hidden_dim;

        // Per-layer params
        let attn_params = 4 * self.hidden_dim * self.hidden_dim; // Q, K, V, O
        let ffn_params = 3 * self.hidden_dim * self.intermediate_dim; // gate, up, down
        let norm_params = 2 * self.hidden_dim; // attn_norm, ffn_norm
        let layer_params = attn_params + ffn_params + norm_params;
        let all_layers = layer_params * self.num_layers;

        // Final norm
        let final_norm = self.hidden_dim;

        // Output projection (if not tied)
        let output_proj = if self.tie_weights {
            0
        } else {
            self.vocab_size * self.hidden_dim
        };

        embed_params + all_layers + final_norm + output_proj
    }
}

/// GPT-style language model (Llama architecture)
///
/// Architecture:
/// ```text
/// tokens -> Embedding -> [N x TransformerBlock] -> RMSNorm -> Output projection -> logits
/// ```
pub struct GPTModel {
    /// Model configuration
    pub config: ModelConfig,
    /// Token embedding weights [vocab_size, hidden_dim]
    pub embed_tokens: Tensor,
    /// Transformer blocks
    pub layers: Vec<TransformerBlock>,
    /// Final layer normalization
    pub final_norm: Tensor,
    /// Output projection weights [vocab_size, hidden_dim] (may be tied to embed_tokens)
    output_weight: Option<Tensor>,
}

impl GPTModel {
    /// Create a new GPT model
    pub fn new(config: ModelConfig) -> Self {
        // Initialize embedding with small random values
        let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_dim)
            .map(|i| {
                let x = ((i as f32 * 0.618033988749895) % 1.0) * 2.0 - 1.0;
                x * 0.02 // Small initialization
            })
            .collect();
        let embed_tokens = Tensor::from_f32_slice(&embed_data, &[config.vocab_size, config.hidden_dim]);

        // Create transformer layers
        let layers: Vec<TransformerBlock> = (0..config.num_layers)
            .map(|_| {
                TransformerBlock::new(
                    config.hidden_dim,
                    config.num_heads,
                    config.num_kv_heads,
                    config.intermediate_dim,
                    config.rope_base,
                    config.norm_eps,
                )
            })
            .collect();

        // Final norm
        let final_norm = Tensor::from_f32_slice(&vec![1.0f32; config.hidden_dim], &[config.hidden_dim]);

        // Output weights (None if tied)
        let output_weight = if config.tie_weights {
            None
        } else {
            let data: Vec<f32> = (0..config.vocab_size * config.hidden_dim)
                .map(|i| {
                    let x = ((i as f32 * 0.618033988749895) % 1.0) * 2.0 - 1.0;
                    x * 0.02
                })
                .collect();
            Some(Tensor::from_f32_slice(&data, &[config.vocab_size, config.hidden_dim]))
        };

        Self {
            config,
            embed_tokens,
            layers,
            final_norm,
            output_weight,
        }
    }

    /// Forward pass
    ///
    /// Input: token indices [batch, seq_len]
    /// Output: logits [batch, seq_len, vocab_size]
    ///
    /// - `position_offset`: Starting position for RoPE (for KV cache)
    pub fn forward(&self, input_ids: &[u32], batch_size: usize, seq_len: usize, position_offset: usize) -> Tensor {
        assert_eq!(input_ids.len(), batch_size * seq_len);

        // Token embedding
        let mut hidden = embedding(&self.embed_tokens, input_ids);
        // Reshape to [batch, seq_len, hidden_dim]
        let hidden_data = hidden.as_f32_slice().to_vec();
        hidden = Tensor::from_f32_slice(&hidden_data, &[batch_size, seq_len, self.config.hidden_dim]);

        // Pass through transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, position_offset, true);
        }

        // Final norm
        hidden = rmsnorm(&hidden, &self.final_norm, self.config.norm_eps);

        // Output projection (compute logits)
        self.compute_logits(&hidden, batch_size, seq_len)
    }

    /// Compute logits from hidden states
    fn compute_logits(&self, hidden: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        let output_weight = self.output_weight.as_ref().unwrap_or(&self.embed_tokens);

        // hidden: [batch, seq_len, hidden_dim]
        // output_weight: [vocab_size, hidden_dim]
        // We want: [batch, seq_len, vocab_size]

        // Reshape hidden to [batch * seq_len, hidden_dim]
        let hidden_data = hidden.as_f32_slice();
        let hidden_2d = Tensor::from_f32_slice(
            hidden_data,
            &[batch_size * seq_len, self.config.hidden_dim],
        );

        // Transpose output weights: [hidden_dim, vocab_size]
        let w = output_weight.as_f32_slice();
        let mut wt = vec![0.0f32; self.config.hidden_dim * self.config.vocab_size];
        for i in 0..self.config.vocab_size {
            for j in 0..self.config.hidden_dim {
                wt[j * self.config.vocab_size + i] = w[i * self.config.hidden_dim + j];
            }
        }
        let weight_t = Tensor::from_f32_slice(&wt, &[self.config.hidden_dim, self.config.vocab_size]);

        // Compute logits: [batch * seq_len, vocab_size]
        let logits = matmul(&hidden_2d, &weight_t);

        // Reshape to [batch, seq_len, vocab_size]
        let logits_data = logits.as_f32_slice().to_vec();
        Tensor::from_f32_slice(&logits_data, &[batch_size, seq_len, self.config.vocab_size])
    }

    /// Get the total number of parameters
    pub fn num_params(&self) -> usize {
        self.config.num_params()
    }

    /// Get a summary of the model
    pub fn summary(&self) -> String {
        let params = self.num_params();
        let params_m = params as f64 / 1_000_000.0;
        format!(
            "GPTModel(\n  vocab_size={},\n  hidden_dim={},\n  num_layers={},\n  num_heads={},\n  intermediate_dim={},\n  params={:.2}M\n)",
            self.config.vocab_size,
            self.config.hidden_dim,
            self.config.num_layers,
            self.config.num_heads,
            self.config.intermediate_dim,
            params_m
        )
    }
}

/// Optimizer state for GPTModel
pub struct GPTModelState {
    pub embed_state: ParamState,
    pub layer_states: Vec<TransformerBlockState>,
    pub final_norm_state: ParamState,
    pub output_weight_state: Option<ParamState>,
}

impl GPTModelState {
    pub fn new(model: &GPTModel) -> Self {
        Self {
            embed_state: ParamState::new(model.embed_tokens.shape()),
            layer_states: model
                .layers
                .iter()
                .map(|l| TransformerBlockState::new(l))
                .collect(),
            final_norm_state: ParamState::new(model.final_norm.shape()),
            output_weight_state: model.output_weight.as_ref().map(|w| ParamState::new(w.shape())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_tiny_forward() {
        let config = ModelConfig::tiny();
        let model = GPTModel::new(config.clone());

        let batch = 2;
        let seq_len = 8;
        let input_ids: Vec<u32> = (0..batch * seq_len).map(|i| (i % 100) as u32).collect();

        let logits = model.forward(&input_ids, batch, seq_len, 0);

        assert_eq!(logits.shape(), &[batch, seq_len, config.vocab_size]);
    }

    #[test]
    fn test_model_config_params() {
        let tiny = ModelConfig::tiny();
        println!("Tiny model params: {:.2}M", tiny.num_params() as f64 / 1e6);

        let small = ModelConfig::small();
        println!("Small model params: {:.2}M", small.num_params() as f64 / 1e6);

        // Tiny should have fewer params than small
        assert!(tiny.num_params() < small.num_params());
    }

    #[test]
    fn test_model_summary() {
        let config = ModelConfig::tiny();
        let model = GPTModel::new(config);
        let summary = model.summary();
        assert!(summary.contains("GPTModel"));
        assert!(summary.contains("vocab_size=32000"));
    }

    #[test]
    fn test_model_state() {
        let config = ModelConfig::tiny();
        let model = GPTModel::new(config);
        let state = GPTModelState::new(&model);

        assert_eq!(state.layer_states.len(), model.layers.len());
    }
}
