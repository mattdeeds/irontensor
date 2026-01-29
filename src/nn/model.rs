use crate::ops::{embedding, matmul, rmsnorm, to_f32_gpu};
use crate::optim::ParamState;
use crate::precision::Precision;
use crate::profile::context;
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
    /// Weight precision (FP32 or BF16)
    pub precision: Precision,
    /// Dropout rate after embedding (default 0.0)
    pub embed_dropout: f32,
    /// Dropout rate after attention output (default 0.1)
    pub attn_dropout: f32,
    /// Dropout rate after FFN output (default 0.1)
    pub ffn_dropout: f32,
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
            precision: Precision::FP32,
            embed_dropout: 0.0,
            attn_dropout: 0.1,
            ffn_dropout: 0.1,
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
            precision: Precision::FP32,
            embed_dropout: 0.0,
            attn_dropout: 0.1,
            ffn_dropout: 0.1,
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
            precision: Precision::FP32,
            embed_dropout: 0.0,
            attn_dropout: 0.1,
            ffn_dropout: 0.1,
        }
    }

    /// Create a Shakespeare-optimized config for training on TinyShakespeare
    /// with BPE tokenization (~2048 vocab, ~5.5M params)
    pub fn shakespeare() -> Self {
        Self {
            vocab_size: 2048,
            hidden_dim: 256,
            num_layers: 6,
            num_heads: 8,
            num_kv_heads: 8,
            intermediate_dim: 512,
            max_seq_len: 512,
            rope_base: 10000.0,
            norm_eps: 1e-5,
            tie_weights: true,
            precision: Precision::FP32,
            embed_dropout: 0.0,
            attn_dropout: 0.1,
            ffn_dropout: 0.1,
        }
    }

    /// Return a new config with BF16 precision
    pub fn with_bf16(mut self) -> Self {
        self.precision = Precision::BF16;
        self
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
    pub fn new(config: &ModelConfig) -> Self {
        // Initialize embedding with small random values
        let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_dim)
            .map(|i| {
                let x = ((i as f32 * crate::PHI_FRAC) % 1.0) * 2.0 - 1.0;
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
                    let x = ((i as f32 * crate::PHI_FRAC) % 1.0) * 2.0 - 1.0;
                    x * 0.02
                })
                .collect();
            Some(Tensor::from_f32_slice(&data, &[config.vocab_size, config.hidden_dim]))
        };

        Self {
            config: config.clone(),
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
        let (logits, _) = self.forward_with_hidden(input_ids, batch_size, seq_len, position_offset);
        logits
    }

    /// Forward pass returning both logits and final hidden states
    ///
    /// Returns: (logits, final_hidden_before_output)
    /// - logits: [batch, seq_len, vocab_size]
    /// - final_hidden: [batch * seq_len, hidden_dim] (after final norm, before output projection)
    pub fn forward_with_hidden(&self, input_ids: &[u32], batch_size: usize, seq_len: usize, position_offset: usize) -> (Tensor, Tensor) {
        assert_eq!(input_ids.len(), batch_size * seq_len);

        // Token embedding
        let mut hidden = {
            let _ctx = context("embed");
            let h = embedding(&self.embed_tokens, input_ids);
            // Reshape to [batch, seq_len, hidden_dim]
            let hidden_data = h.as_f32_slice().to_vec();
            Tensor::from_f32_slice(&hidden_data, &[batch_size, seq_len, self.config.hidden_dim])
        };

        // Pass through transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, position_offset, true);
        }

        // Final norm
        hidden = {
            let _ctx = context("final_norm");
            rmsnorm(&hidden, &self.final_norm, self.config.norm_eps).unwrap()
        };

        // Save hidden states for backward pass (reshape to 2D)
        let final_hidden = Tensor::from_f32_slice(
            hidden.as_f32_slice(),
            &[batch_size * seq_len, self.config.hidden_dim],
        );

        // Output projection (compute logits)
        let logits = {
            let _ctx = context("output_proj");
            self.compute_logits(&hidden, batch_size, seq_len)
        };

        (logits, final_hidden)
    }

    /// Compute logits from hidden states
    fn compute_logits(&self, hidden: &Tensor, batch_size: usize, seq_len: usize) -> Tensor {
        let output_weight = self.output_weight.as_ref().unwrap_or(&self.embed_tokens);

        // Convert BF16 weights to FP32 if needed (mixed precision support)
        let output_weight = if output_weight.precision() == Precision::BF16 {
            to_f32_gpu(output_weight)
        } else {
            output_weight.clone()
        };

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
        let logits = matmul(&hidden_2d, &weight_t).unwrap();

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
            "GPTModel(\n  vocab_size={},\n  hidden_dim={},\n  num_layers={},\n  num_heads={},\n  intermediate_dim={},\n  precision={:?},\n  params={:.2}M\n)",
            self.config.vocab_size,
            self.config.hidden_dim,
            self.config.num_layers,
            self.config.num_heads,
            self.config.intermediate_dim,
            self.config.precision,
            params_m
        )
    }

    /// Get the model's precision
    pub fn precision(&self) -> Precision {
        self.config.precision
    }

    /// Check if the model uses BF16 precision
    pub fn is_bf16(&self) -> bool {
        self.config.precision == Precision::BF16
    }

    /// Convert model weights to BF16
    ///
    /// Returns the memory savings ratio (BF16 size / FP32 size)
    pub fn to_bf16(&mut self) {
        if self.config.precision == Precision::BF16 {
            return; // Already BF16
        }

        // Convert embedding weights
        self.embed_tokens = self.embed_tokens.to_bf16();

        // Convert transformer layer weights
        for layer in &mut self.layers {
            layer.to_bf16();
        }

        // Convert final norm
        self.final_norm = self.final_norm.to_bf16();

        // Convert output weights if not tied
        if let Some(ref mut w) = self.output_weight {
            *w = w.to_bf16();
        }

        self.config.precision = Precision::BF16;
    }

    /// Convert model weights to FP32
    pub fn to_f32(&mut self) {
        if self.config.precision == Precision::FP32 {
            return; // Already FP32
        }

        // Convert embedding weights
        self.embed_tokens = self.embed_tokens.to_f32();

        // Convert transformer layer weights
        for layer in &mut self.layers {
            layer.to_f32();
        }

        // Convert final norm
        self.final_norm = self.final_norm.to_f32();

        // Convert output weights if not tied
        if let Some(ref mut w) = self.output_weight {
            *w = w.to_f32();
        }

        self.config.precision = Precision::FP32;
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let bytes_per_param = self.config.precision.byte_size();
        self.num_params() * bytes_per_param
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
                .map(TransformerBlockState::new)
                .collect(),
            final_norm_state: ParamState::new(model.final_norm.shape()),
            output_weight_state: model.output_weight.as_ref().map(|w| ParamState::new(w.shape())),
        }
    }

    /// Save the model optimizer state to a writer.
    pub fn save<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Save embedding state
        self.embed_state.save(writer)?;

        // Save number of layers
        let num_layers = self.layer_states.len() as u32;
        writer.write_all(&num_layers.to_le_bytes())?;

        // Save layer states
        for layer_state in &self.layer_states {
            layer_state.save(writer)?;
        }

        // Save final norm state
        self.final_norm_state.save(writer)?;

        // Save output weight state (if present)
        let has_output_state = self.output_weight_state.is_some();
        writer.write_all(&[if has_output_state { 1u8 } else { 0u8 }])?;
        if let Some(ref state) = self.output_weight_state {
            state.save(writer)?;
        }

        Ok(())
    }

    /// Load the model optimizer state from a reader.
    pub fn load<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        // Load embedding state
        let embed_state = ParamState::load(reader)?;

        // Load number of layers
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let num_layers = u32::from_le_bytes(buf4) as usize;

        // Load layer states
        let mut layer_states = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layer_states.push(TransformerBlockState::load(reader)?);
        }

        // Load final norm state
        let final_norm_state = ParamState::load(reader)?;

        // Load output weight state (if present)
        let mut buf1 = [0u8; 1];
        reader.read_exact(&mut buf1)?;
        let output_weight_state = if buf1[0] == 1 {
            Some(ParamState::load(reader)?)
        } else {
            None
        };

        Ok(Self {
            embed_state,
            layer_states,
            final_norm_state,
            output_weight_state,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_tiny_forward() {
        let config = ModelConfig::tiny();
        let model = GPTModel::new(&config);

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
        let model = GPTModel::new(&config);
        let summary = model.summary();
        assert!(summary.contains("GPTModel"));
        assert!(summary.contains("vocab_size=32000"));
    }

    #[test]
    fn test_model_state() {
        let config = ModelConfig::tiny();
        let model = GPTModel::new(&config);
        let state = GPTModelState::new(&model);

        assert_eq!(state.layer_states.len(), model.layers.len());
    }

    #[test]
    fn test_model_bf16_conversion() {
        let config = ModelConfig::tiny();
        let mut model = GPTModel::new(&config);

        // Start as FP32
        assert!(!model.is_bf16());
        assert_eq!(model.precision(), Precision::FP32);
        let fp32_memory = model.memory_bytes();

        // Convert to BF16
        model.to_bf16();
        assert!(model.is_bf16());
        assert_eq!(model.precision(), Precision::BF16);
        let bf16_memory = model.memory_bytes();

        // BF16 should use half the memory
        assert_eq!(bf16_memory, fp32_memory / 2);

        // Convert back to FP32
        model.to_f32();
        assert!(!model.is_bf16());
        assert_eq!(model.precision(), Precision::FP32);
    }

    #[test]
    fn test_model_config_with_bf16() {
        let config = ModelConfig::tiny().with_bf16();
        assert_eq!(config.precision, Precision::BF16);
    }
}
