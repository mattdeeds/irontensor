//! File-based performance logging for training and inference.
//!
//! This module provides structured logging to a single JSON file per run containing:
//! - Training configuration
//! - Per-step training metrics
//! - Training summary with total time
//! - Per-generation inference metrics (TTFT, tokens/sec)
//!
//! # Usage
//!
//! Enable logging via environment variable:
//! ```bash
//! IRONTENSOR_LOG=1 cargo run --release
//! IRONTENSOR_LOG_DIR=./my_logs cargo run  # Custom directory
//! IRONTENSOR_LOG_OPS=1 cargo run          # Include op breakdown
//! ```
//!
//! Or programmatically:
//! ```ignore
//! use irontensor::logging::{Logger, LogConfig};
//!
//! Logger::init(LogConfig {
//!     enabled: true,
//!     log_dir: "logs".to_string(),
//!     ..Default::default()
//! });
//!
//! // Log training steps
//! Logger::log_train_step(&record);
//!
//! // Log inference
//! Logger::log_inference(&inference_record);
//!
//! // Finalize training summary
//! Logger::finalize_training(total_time_sec, final_loss, best_val_loss, profiler_report);
//!
//! // Shutdown writes the complete log file
//! Logger::shutdown();
//! ```
//!
//! # Output
//!
//! Single JSON file per run: `logs/run_{run_id}.json`

pub mod formats;
pub mod inference_logger;
pub mod train_logger;
pub mod writer;

pub use formats::{
    current_timestamp_ms, InferenceRecord, LayerTimingRecord, MatmulShapeRecord, OpTimingRecord,
    ProfileReportRecord, RunLog, TrainConfigSnapshot, TrainStepRecord, TrainingLog,
};
pub use inference_logger::InferenceTimer;
pub use writer::write_json_file;

use std::cell::RefCell;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Global flag indicating if logging is enabled.
static LOGGING_ENABLED: AtomicBool = AtomicBool::new(false);

// Thread-local logger state.
thread_local! {
    static LOGGER_STATE: RefCell<Option<LoggerState>> = const { RefCell::new(None) };
}

/// Configuration for the logging system.
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Whether logging is enabled
    pub enabled: bool,
    /// Directory for log files (default: "logs")
    pub log_dir: String,
    /// Run ID (auto-generated if None)
    pub run_id: Option<String>,
    /// Model name for the log (default: "model")
    pub model_name: String,
    /// Training configuration snapshot
    pub config: TrainConfigSnapshot,
    /// Include detailed operation breakdown from profiler
    pub include_op_breakdown: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            log_dir: "logs".to_string(),
            run_id: None,
            model_name: "model".to_string(),
            config: TrainConfigSnapshot::default(),
            include_op_breakdown: false,
        }
    }
}

impl LogConfig {
    /// Create config from environment variables.
    ///
    /// - `IRONTENSOR_LOG=1` - Enable logging
    /// - `IRONTENSOR_LOG_DIR=path` - Set log directory
    /// - `IRONTENSOR_LOG_OPS=1` - Include operation breakdown
    pub fn from_env() -> Self {
        let enabled = std::env::var("IRONTENSOR_LOG").is_ok();
        let log_dir = std::env::var("IRONTENSOR_LOG_DIR").unwrap_or_else(|_| "logs".to_string());
        let include_op_breakdown = std::env::var("IRONTENSOR_LOG_OPS").is_ok();

        Self {
            enabled,
            log_dir,
            include_op_breakdown,
            ..Default::default()
        }
    }
}

/// Internal state for the logger.
struct LoggerState {
    config: LogConfig,
    run_id: String,
    run_log: RunLog,
    training_start: Option<Instant>,
    total_tokens: usize,
}

impl LoggerState {
    fn new(config: LogConfig) -> std::io::Result<Self> {
        // Create log directory
        fs::create_dir_all(&config.log_dir)?;

        // Generate run ID
        let run_id = config.run_id.clone().unwrap_or_else(|| {
            let timestamp = chrono_lite_timestamp();
            format!("{}_{}", timestamp, config.model_name)
        });

        let run_log = RunLog::new(
            run_id.clone(),
            config.model_name.clone(),
            config.config.clone(),
        );

        Ok(Self {
            config,
            run_id,
            run_log,
            training_start: None,
            total_tokens: 0,
        })
    }

    fn log_path(&self) -> PathBuf {
        PathBuf::from(&self.config.log_dir).join(format!("run_{}.json", self.run_id))
    }
}

/// Global logger for performance metrics.
pub struct Logger;

impl Logger {
    /// Initialize the logging system.
    ///
    /// Call this once at the start of your program.
    /// If initialization fails (e.g., can't create directory), logging is silently disabled.
    pub fn init(config: LogConfig) {
        if !config.enabled {
            LOGGING_ENABLED.store(false, Ordering::SeqCst);
            return;
        }

        LOGGER_STATE.with(|state| {
            match LoggerState::new(config) {
                Ok(s) => {
                    *state.borrow_mut() = Some(s);
                    LOGGING_ENABLED.store(true, Ordering::SeqCst);
                }
                Err(e) => {
                    eprintln!("[Logger] Failed to initialize: {}", e);
                    LOGGING_ENABLED.store(false, Ordering::SeqCst);
                }
            }
        });
    }

    /// Initialize from environment variables.
    pub fn init_from_env() {
        Self::init(LogConfig::from_env());
    }

    /// Shutdown the logging system and write the complete log file.
    pub fn shutdown() {
        if !Self::is_enabled() {
            return;
        }

        LOGGER_STATE.with(|state| {
            if let Some(s) = state.borrow_mut().take() {
                let log_path = s.log_path();

                // Write the complete run log
                if let Err(e) = write_json_file(&log_path, &s.run_log) {
                    eprintln!("Failed to write log: {}", e);
                } else {
                    println!("Log saved to {}", log_path.display());
                }
            }
        });

        LOGGING_ENABLED.store(false, Ordering::SeqCst);
    }

    /// Check if logging is enabled.
    #[inline]
    pub fn is_enabled() -> bool {
        LOGGING_ENABLED.load(Ordering::Relaxed)
    }

    /// Get the current run ID.
    pub fn run_id() -> Option<String> {
        if !Self::is_enabled() {
            return None;
        }

        LOGGER_STATE.with(|state| state.borrow().as_ref().map(|s| s.run_id.clone()))
    }

    /// Get the log file path.
    pub fn log_path() -> Option<PathBuf> {
        if !Self::is_enabled() {
            return None;
        }

        LOGGER_STATE.with(|state| state.borrow().as_ref().map(|s| s.log_path()))
    }

    /// Start training timer.
    pub fn start_training() {
        if !Self::is_enabled() {
            return;
        }

        LOGGER_STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.training_start = Some(Instant::now());
                s.total_tokens = 0;
            }
        });
    }

    /// Log a training step record.
    pub fn log_train_step(record: &TrainStepRecord) {
        if !Self::is_enabled() {
            return;
        }

        LOGGER_STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.run_log.training.steps.push(record.clone());
                s.total_tokens += record.batch_size * record.seq_len;
            }
        });
    }

    /// Finalize training summary.
    pub fn finalize_training(
        total_steps: usize,
        final_loss: f32,
        best_val_loss: Option<f32>,
        epochs_completed: usize,
        profiler_report: Option<ProfileReportRecord>,
    ) {
        if !Self::is_enabled() {
            return;
        }

        LOGGER_STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                let total_time_sec = s
                    .training_start
                    .map(|t| t.elapsed().as_secs_f32())
                    .unwrap_or(0.0);

                let avg_tokens_per_sec = if total_time_sec > 0.0 {
                    s.total_tokens as f32 / total_time_sec
                } else {
                    0.0
                };

                s.run_log.training.total_time_sec = total_time_sec;
                s.run_log.training.total_steps = total_steps;
                s.run_log.training.epochs_completed = epochs_completed;
                s.run_log.training.final_loss = final_loss;
                s.run_log.training.best_val_loss = best_val_loss;
                s.run_log.training.avg_tokens_per_sec = avg_tokens_per_sec;
                s.run_log.training.profiler_report = profiler_report;
            }
        });
    }

    /// Log an inference record.
    pub fn log_inference(record: &InferenceRecord) {
        if !Self::is_enabled() {
            return;
        }

        LOGGER_STATE.with(|state| {
            if let Some(ref mut s) = *state.borrow_mut() {
                s.run_log.inference.push(record.clone());
            }
        });
    }

    /// Check if operation breakdown should be included.
    pub fn include_op_breakdown() -> bool {
        if !Self::is_enabled() {
            return false;
        }

        LOGGER_STATE.with(|state| {
            state
                .borrow()
                .as_ref()
                .is_some_and(|s| s.config.include_op_breakdown)
        })
    }
}

/// Generate a timestamp string without external chrono dependency.
/// Format: YYYYMMDD_HHMMSS
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Convert Unix timestamp to date/time components
    let secs_per_min = 60;
    let secs_per_hour = 3600;
    let secs_per_day = 86400;

    let days_since_epoch = now / secs_per_day;
    let time_of_day = now % secs_per_day;

    let hours = time_of_day / secs_per_hour;
    let minutes = (time_of_day % secs_per_hour) / secs_per_min;
    let seconds = time_of_day % secs_per_min;

    // Approximate year/month/day from days since epoch (1970-01-01)
    let years_since_epoch = days_since_epoch / 365;
    let year = 1970 + years_since_epoch;

    // Approximate day of year
    let day_of_year = days_since_epoch - (years_since_epoch * 365) - (years_since_epoch / 4);

    // Very rough month/day approximation (30 days per month)
    let month = (day_of_year / 30).min(11) + 1;
    let day = (day_of_year % 30) + 1;

    format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}",
        year, month, day, hours, minutes, seconds
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = LogConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.log_dir, "logs");
        assert!(config.run_id.is_none());
    }

    #[test]
    fn test_chrono_lite_timestamp() {
        let ts = chrono_lite_timestamp();
        // Should be in format YYYYMMDD_HHMMSS (15 chars)
        assert_eq!(ts.len(), 15);
        assert!(ts.contains('_'));
    }
}
