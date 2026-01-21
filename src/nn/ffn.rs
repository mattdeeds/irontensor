use crate::ops::swiglu;
use crate::optim::ParamState;
use crate::tensor::Tensor;

use super::linear::Linear;

/// Feed-Forward Network with SwiGLU activation (Llama-style)
///
/// SwiGLU FFN: output = down(swiglu(gate(x), up(x)))
///
/// Where swiglu(gate, up) = silu(gate) * up
///
/// The intermediate dimension is typically 8/3 * hidden_dim (rounded to multiple of 256)
pub struct FeedForward {
    /// Gate projection [hidden_dim, intermediate_dim]
    pub w_gate: Linear,
    /// Up projection [hidden_dim, intermediate_dim]
    pub w_up: Linear,
    /// Down projection [intermediate_dim, hidden_dim]
    pub w_down: Linear,

    /// Hidden dimension
    pub hidden_dim: usize,
    /// Intermediate dimension
    pub intermediate_dim: usize,
}

impl FeedForward {
    /// Create a new FFN layer
    ///
    /// - `hidden_dim`: Model hidden dimension
    /// - `intermediate_dim`: FFN intermediate dimension (typically ~2.67 * hidden_dim)
    pub fn new(hidden_dim: usize, intermediate_dim: usize) -> Self {
        let w_gate = Linear::new(hidden_dim, intermediate_dim, false);
        let w_up = Linear::new(hidden_dim, intermediate_dim, false);
        let w_down = Linear::new(intermediate_dim, hidden_dim, false);

        Self {
            w_gate,
            w_up,
            w_down,
            hidden_dim,
            intermediate_dim,
        }
    }

    /// Create FFN with default intermediate dimension (8/3 * hidden_dim, rounded)
    pub fn new_default(hidden_dim: usize) -> Self {
        // Llama uses 8/3 * hidden_dim, rounded to nearest multiple of 256
        let intermediate = (hidden_dim * 8 / 3 + 255) / 256 * 256;
        Self::new(hidden_dim, intermediate)
    }

    /// Forward pass
    ///
    /// Input shape: [batch, seq_len, hidden_dim]
    /// Output shape: [batch, seq_len, hidden_dim]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert!(shape.len() >= 2);
        assert_eq!(shape[shape.len() - 1], self.hidden_dim);

        // Gate and up projections
        let gate = self.w_gate.forward(x); // [batch, seq, intermediate]
        let up = self.w_up.forward(x); // [batch, seq, intermediate]

        // SwiGLU activation: silu(gate) * up
        let hidden = swiglu(&gate, &up);

        // Down projection
        self.w_down.forward(&hidden)
    }

    /// Get total parameter count
    pub fn num_params(&self) -> usize {
        self.w_gate.num_params() + self.w_up.num_params() + self.w_down.num_params()
    }
}

/// Optimizer state for FeedForward
pub struct FeedForwardState {
    pub w_gate_state: ParamState,
    pub w_up_state: ParamState,
    pub w_down_state: ParamState,
}

impl FeedForwardState {
    pub fn new(layer: &FeedForward) -> Self {
        Self {
            w_gate_state: ParamState::new(layer.w_gate.weight.shape()),
            w_up_state: ParamState::new(layer.w_up.weight.shape()),
            w_down_state: ParamState::new(layer.w_down.weight.shape()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffn_forward() {
        let hidden_dim = 64;
        let intermediate_dim = 128;
        let batch = 2;
        let seq_len = 8;

        let ffn = FeedForward::new(hidden_dim, intermediate_dim);

        let input_data: Vec<f32> = (0..batch * seq_len * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, hidden_dim]);

        let output = ffn.forward(&input);

        assert_eq!(output.shape(), &[batch, seq_len, hidden_dim]);
    }

    #[test]
    fn test_ffn_default_intermediate() {
        let ffn = FeedForward::new_default(768);
        // 768 * 8 / 3 = 2048, already multiple of 256
        assert_eq!(ffn.intermediate_dim, 2048);
    }

    #[test]
    fn test_ffn_num_params() {
        let ffn = FeedForward::new(512, 1024);
        // gate: 512*1024, up: 512*1024, down: 1024*512 = 3 * 512 * 1024
        assert_eq!(ffn.num_params(), 3 * 512 * 1024);
    }
}
