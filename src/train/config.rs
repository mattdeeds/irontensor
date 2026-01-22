/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Base learning rate
    pub learning_rate: f32,
    /// Weight decay coefficient
    pub weight_decay: f32,
    /// Lion optimizer beta1
    pub beta1: f32,
    /// Lion optimizer beta2
    pub beta2: f32,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
    /// Number of warmup steps
    pub warmup_steps: usize,
    /// Total number of training steps
    pub total_steps: usize,
    /// Log every N steps
    pub log_interval: usize,
    /// Save checkpoint every N steps
    pub save_interval: usize,
    /// Evaluate every N steps
    pub eval_interval: usize,
    /// Checkpoint save directory
    pub checkpoint_dir: String,
    /// Use BF16 mixed precision training
    /// When enabled, model weights are stored in BF16 for ~50% memory reduction.
    /// Forward/backward passes compute in FP32 for numerical stability.
    pub use_bf16: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            max_grad_norm: 1.0,
            warmup_steps: 100,
            total_steps: 10000,
            log_interval: 10,
            save_interval: 1000,
            eval_interval: 100,
            checkpoint_dir: "checkpoints".to_string(),
            use_bf16: false,
        }
    }
}

/// Training metrics for a single step
#[derive(Debug, Clone)]
pub struct TrainMetrics {
    pub step: usize,
    pub loss: f32,
    pub grad_norm: f32,
    pub learning_rate: f32,
    pub tokens_per_sec: f32,
}
