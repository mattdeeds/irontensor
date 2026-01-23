//! Serializable record formats for logging.
//!
//! All records use serde for JSON serialization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete run log containing all metrics for a single training/inference run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunLog {
    /// Unique identifier for this run
    pub run_id: String,
    /// Model name/config identifier
    pub model_name: String,
    /// Training configuration
    pub config: TrainConfigSnapshot,
    /// Training metrics and summary
    pub training: TrainingLog,
    /// Inference metrics
    pub inference: Vec<InferenceRecord>,
}

/// Training-specific logging data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingLog {
    /// Total training time in seconds
    pub total_time_sec: f32,
    /// Total training steps completed
    pub total_steps: usize,
    /// Total epochs completed
    pub epochs_completed: usize,
    /// Final training loss
    pub final_loss: f32,
    /// Best validation loss achieved
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_val_loss: Option<f32>,
    /// Average tokens processed per second
    pub avg_tokens_per_sec: f32,
    /// Per-step training records
    pub steps: Vec<TrainStepRecord>,
    /// Profiler report (if profiling was enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiler_report: Option<ProfileReportRecord>,
}

impl Default for TrainingLog {
    fn default() -> Self {
        Self {
            total_time_sec: 0.0,
            total_steps: 0,
            epochs_completed: 0,
            final_loss: 0.0,
            best_val_loss: None,
            avg_tokens_per_sec: 0.0,
            steps: Vec::new(),
            profiler_report: None,
        }
    }
}

/// Per-step training metrics record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainStepRecord {
    /// Unix timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Current training step
    pub step: usize,
    /// Current epoch
    pub epoch: usize,
    /// Training loss for this step
    pub loss: f32,
    /// Gradient norm before clipping
    pub grad_norm: f32,
    /// Current learning rate
    pub learning_rate: f32,
    /// Tokens processed per second
    pub tokens_per_sec: f32,
    /// Time for this step in milliseconds
    pub step_time_ms: f32,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Validation loss (if evaluated this step)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub val_loss: Option<f32>,
    /// Phase timing breakdown in milliseconds (Forward, Backward, Optimizer)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_times_ms: Option<HashMap<String, f32>>,
    /// Detailed operation breakdown
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op_breakdown: Option<Vec<OpTimingRecord>>,
}

/// Timing for a single operation type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpTimingRecord {
    /// Operation name (e.g., "Matmul", "RmsNorm")
    pub op: String,
    /// Total time in milliseconds
    pub time_ms: f32,
    /// Number of invocations
    pub count: usize,
    /// Average time per invocation in milliseconds
    pub avg_ms: f32,
}

/// Snapshot of training configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainConfigSnapshot {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub batch_size: usize,
    pub seq_len: usize,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub use_bf16: bool,
}

/// Serializable version of ProfileReport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReportRecord {
    /// Total profiled time in milliseconds
    pub total_time_ms: f32,
    /// Number of steps recorded
    pub steps_recorded: usize,
    /// Average time per step in milliseconds
    pub avg_step_ms: f32,
    /// Phase timing breakdown (phase name -> ms)
    pub phase_breakdown: HashMap<String, f32>,
    /// Per-layer timing breakdown
    pub layer_breakdown: Vec<LayerTimingRecord>,
    /// Top operations by time
    pub top_operations: Vec<OpTimingRecord>,
}

/// Per-layer timing record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTimingRecord {
    pub layer: usize,
    pub forward_ms: f32,
    pub backward_ms: f32,
    pub total_ms: f32,
}

/// Inference metrics record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRecord {
    /// Unix timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Number of tokens in the prompt
    pub prompt_tokens: usize,
    /// Number of tokens generated
    pub generated_tokens: usize,
    /// Time to first token in milliseconds
    pub time_to_first_token_ms: f32,
    /// Total generation time in milliseconds
    pub total_time_ms: f32,
    /// Average inter-token latency (excluding first token) in milliseconds
    pub inter_token_latency_ms: f32,
    /// Tokens generated per second
    pub tokens_per_sec: f32,
    /// Sampling temperature used
    pub temperature: f32,
}

impl TrainStepRecord {
    /// Create a new training step record with current timestamp.
    pub fn new(
        step: usize,
        epoch: usize,
        loss: f32,
        grad_norm: f32,
        learning_rate: f32,
        tokens_per_sec: f32,
        step_time_ms: f32,
        batch_size: usize,
        seq_len: usize,
    ) -> Self {
        Self {
            timestamp_ms: current_timestamp_ms(),
            step,
            epoch,
            loss,
            grad_norm,
            learning_rate,
            tokens_per_sec,
            step_time_ms,
            batch_size,
            seq_len,
            val_loss: None,
            phase_times_ms: None,
            op_breakdown: None,
        }
    }

    /// Set validation loss.
    pub fn with_val_loss(mut self, val_loss: f32) -> Self {
        self.val_loss = Some(val_loss);
        self
    }

    /// Set phase timing breakdown.
    pub fn with_phase_times(mut self, phase_times: HashMap<String, f32>) -> Self {
        self.phase_times_ms = Some(phase_times);
        self
    }

    /// Set operation breakdown.
    pub fn with_op_breakdown(mut self, ops: Vec<OpTimingRecord>) -> Self {
        self.op_breakdown = Some(ops);
        self
    }
}

impl InferenceRecord {
    /// Create a new inference record with current timestamp.
    pub fn new(
        prompt_tokens: usize,
        generated_tokens: usize,
        time_to_first_token_ms: f32,
        total_time_ms: f32,
        temperature: f32,
    ) -> Self {
        let inter_token_latency_ms = if generated_tokens > 1 {
            (total_time_ms - time_to_first_token_ms) / (generated_tokens - 1) as f32
        } else {
            0.0
        };
        let tokens_per_sec = if total_time_ms > 0.0 {
            generated_tokens as f32 / (total_time_ms / 1000.0)
        } else {
            0.0
        };

        Self {
            timestamp_ms: current_timestamp_ms(),
            prompt_tokens,
            generated_tokens,
            time_to_first_token_ms,
            total_time_ms,
            inter_token_latency_ms,
            tokens_per_sec,
            temperature,
        }
    }
}

impl RunLog {
    /// Create a new run log.
    pub fn new(run_id: String, model_name: String, config: TrainConfigSnapshot) -> Self {
        Self {
            run_id,
            model_name,
            config,
            training: TrainingLog::default(),
            inference: Vec::new(),
        }
    }
}

/// Get current Unix timestamp in milliseconds.
pub fn current_timestamp_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
