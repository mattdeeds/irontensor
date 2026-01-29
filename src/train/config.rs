use super::checkpoint_grad::CheckpointConfig;
use crate::gpu_trace::GpuTraceConfig;

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
    /// Use async GPU command submission for CPU/GPU overlap.
    /// When enabled, the optimizer step commits asynchronously, allowing
    /// the CPU to start preparing the next batch while the GPU finishes.
    pub async_gpu: bool,
    /// Enable dropout during training.
    /// When false, dropout is disabled (useful for evaluation/testing).
    pub dropout_enabled: bool,
    /// Number of micro-batches to accumulate before optimizer step.
    /// Effective batch size = batch_size * accumulation_steps.
    /// Default is 1 (no accumulation).
    pub accumulation_steps: usize,
    /// Early stopping patience: stop training if validation loss doesn't improve
    /// for this many consecutive evaluations. None = disabled.
    pub early_stopping_patience: Option<usize>,
    /// Minimum improvement in validation loss required to reset patience counter.
    /// Default is 0.0 (any improvement resets counter).
    pub early_stopping_min_delta: f32,
    /// Activation checkpointing configuration.
    /// When enabled, stores only layer inputs at checkpoint boundaries and recomputes
    /// activations during backward pass. Trades compute for ~90% activation memory reduction.
    pub checkpoint_config: CheckpointConfig,
    /// GPU trace capture configuration.
    /// When enabled via IRONTENSOR_GPU_TRACE=1, captures GPU traces for shader profiling.
    /// Optional step to capture can be set via IRONTENSOR_GPU_TRACE_STEP=N.
    pub gpu_trace_config: Option<GpuTraceConfig>,
    /// Specific step to capture GPU trace for (if gpu_trace_config is Some).
    /// When None and gpu_trace is enabled, captures every step (not recommended).
    pub gpu_trace_step: Option<usize>,
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
            async_gpu: true, // Enabled by default for better performance
            dropout_enabled: true, // Enabled by default for training
            accumulation_steps: 1, // No accumulation by default
            early_stopping_patience: None, // Disabled by default
            early_stopping_min_delta: 0.0, // Any improvement resets counter
            checkpoint_config: CheckpointConfig::default(), // Disabled by default
            gpu_trace_config: if GpuTraceConfig::is_enabled_via_env() {
                Some(GpuTraceConfig::from_env())
            } else {
                None
            },
            gpu_trace_step: GpuTraceConfig::capture_step_from_env(),
        }
    }
}

impl TrainingConfig {
    /// Create a config with GPU trace capture enabled for a specific step.
    ///
    /// This is useful for programmatic control over GPU trace capture.
    pub fn with_gpu_trace(mut self, output_dir: &str, step: usize) -> Self {
        self.gpu_trace_config = Some(GpuTraceConfig {
            output_dir: output_dir.to_string(),
            timestamp_files: true,
        });
        self.gpu_trace_step = Some(step);
        self
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
