//! Lion optimizer configuration and parameter state.

use crate::precision::Precision;
use crate::tensor::Tensor;

/// Momentum state for a parameter tensor.
pub struct ParamState {
    /// Momentum tensor (same shape as parameter).
    pub momentum: Tensor,
}

impl ParamState {
    /// Create a new parameter state with zeroed momentum.
    pub fn new(shape: &[usize]) -> Self {
        Self {
            momentum: Tensor::zeros(shape, Precision::FP32),
        }
    }
}

/// Lion optimizer configuration.
#[derive(Clone, Debug)]
pub struct LionConfig {
    /// Learning rate (default: 1e-4).
    pub lr: f32,
    /// Momentum decay for update computation (default: 0.9).
    pub beta1: f32,
    /// Momentum decay for momentum update (default: 0.99).
    pub beta2: f32,
    /// Weight decay coefficient (default: 0.0).
    pub weight_decay: f32,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }
}

impl LionConfig {
    /// Create a new Lion config with the given learning rate.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    /// Set weight decay.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set beta parameters.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}
