//! Profiling module for IronTensor.
//!
//! Provides opt-in profiling to identify performance bottlenecks in training.
//! Uses thread-local storage for zero overhead when disabled and no mutex contention.
//!
//! # Example
//! ```ignore
//! use irontensor::profile::{Profiler, ProfilerConfig, Phase};
//!
//! // Initialize profiling
//! Profiler::init(ProfilerConfig {
//!     enabled: true,
//!     warmup_steps: 5,
//!     report_interval: 100,
//! });
//!
//! // Training loop
//! for step in 0..1000 {
//!     Profiler::begin_step();
//!     Profiler::set_phase(Phase::Forward);
//!     // ... forward pass ...
//!     Profiler::set_phase(Phase::Backward);
//!     // ... backward pass ...
//!     Profiler::set_phase(Phase::Optimizer);
//!     // ... optimizer step ...
//!     Profiler::end_step();
//! }
//!
//! // Get report
//! let report = Profiler::report();
//! report.print();
//! ```

pub mod categories;
pub mod report;
pub mod timer;

pub use categories::{OpCategory, Phase};
pub use report::{LayerTiming, OpStat, ProfileReport};
pub use timer::{is_profiling_enabled, timed, TimedOp};

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{Duration, Instant};

thread_local! {
    /// Thread-local profiler instance.
    pub(crate) static PROFILER: RefCell<Option<ProfilerState>> = const { RefCell::new(None) };
}

/// Profiler configuration.
#[derive(Clone, Debug)]
pub struct ProfilerConfig {
    /// Whether profiling is enabled.
    pub enabled: bool,
    /// Number of warmup steps to skip before recording.
    pub warmup_steps: usize,
    /// Interval for automatic report printing (0 = disabled).
    pub report_interval: usize,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            warmup_steps: 0,
            report_interval: 0,
        }
    }
}

/// Internal profiler state.
pub struct ProfilerState {
    config: ProfilerConfig,
    step: usize,
    step_start: Option<Instant>,
    current_phase: Phase,
    current_layer: Option<usize>,
    total_time: Duration,
    steps_recorded: usize,
    phase_times: HashMap<Phase, Duration>,
    layer_times: Vec<LayerTiming>,
    op_stats: HashMap<OpCategory, OpStat>,
}

impl ProfilerState {
    fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            step: 0,
            step_start: None,
            current_phase: Phase::Forward,
            current_layer: None,
            total_time: Duration::ZERO,
            steps_recorded: 0,
            phase_times: HashMap::new(),
            layer_times: Vec::new(),
            op_stats: HashMap::new(),
        }
    }

    fn begin_step(&mut self) {
        if self.step >= self.config.warmup_steps {
            self.step_start = Some(Instant::now());
        }
        self.current_phase = Phase::Forward;
        self.current_layer = None;
    }

    fn end_step(&mut self) {
        if let Some(start) = self.step_start.take() {
            let duration = start.elapsed();
            self.total_time += duration;
            self.steps_recorded += 1;

            // Auto-report if configured
            if self.config.report_interval > 0
                && self.steps_recorded % self.config.report_interval == 0
            {
                self.generate_report().print();
            }
        }
        self.step += 1;
    }

    fn set_phase(&mut self, phase: Phase) {
        self.current_phase = phase;
    }

    fn set_layer(&mut self, layer: Option<usize>) {
        self.current_layer = layer;
    }

    pub fn record(&mut self, category: OpCategory, duration: Duration, elements: usize) {
        // Skip if still in warmup
        if self.step < self.config.warmup_steps {
            return;
        }

        // Record phase time
        *self.phase_times.entry(self.current_phase).or_insert(Duration::ZERO) += duration;

        // Record layer time if set
        if let Some(layer) = self.current_layer {
            // Ensure we have enough layer entries
            while self.layer_times.len() <= layer {
                self.layer_times.push(LayerTiming::default());
            }
            match self.current_phase {
                Phase::Forward => self.layer_times[layer].forward += duration,
                Phase::Backward => self.layer_times[layer].backward += duration,
                Phase::Optimizer => {} // Optimizer doesn't have per-layer timing
            }
        }

        // Record operation stats
        self.op_stats
            .entry(category.clone())
            .or_insert_with(|| OpStat::new(category))
            .record(duration, elements);
    }

    fn generate_report(&self) -> ProfileReport {
        ProfileReport {
            total_time: self.total_time,
            steps_recorded: self.steps_recorded,
            phase_breakdown: self.phase_times.clone(),
            layer_breakdown: self.layer_times.clone(),
            op_stats: self.op_stats.values().cloned().collect(),
        }
    }
}

/// Main profiler interface.
///
/// All methods are static and thread-local, providing zero overhead
/// when profiling is disabled.
pub struct Profiler;

impl Profiler {
    /// Initialize the profiler with the given configuration.
    ///
    /// If `config.enabled` is false, the profiler will not be initialized
    /// and all profiling calls will be no-ops.
    pub fn init(config: ProfilerConfig) {
        if config.enabled {
            PROFILER.with(|p| {
                *p.borrow_mut() = Some(ProfilerState::new(config));
            });
        }
    }

    /// Shut down the profiler and release resources.
    pub fn shutdown() {
        PROFILER.with(|p| {
            *p.borrow_mut() = None;
        });
    }

    /// Check if profiling is enabled.
    pub fn is_enabled() -> bool {
        is_profiling_enabled()
    }

    /// Mark the beginning of a training step.
    pub fn begin_step() {
        PROFILER.with(|p| {
            if let Some(ref mut profiler) = *p.borrow_mut() {
                profiler.begin_step();
            }
        });
    }

    /// Mark the end of a training step.
    pub fn end_step() {
        PROFILER.with(|p| {
            if let Some(ref mut profiler) = *p.borrow_mut() {
                profiler.end_step();
            }
        });
    }

    /// Set the current training phase.
    pub fn set_phase(phase: Phase) {
        PROFILER.with(|p| {
            if let Some(ref mut profiler) = *p.borrow_mut() {
                profiler.set_phase(phase);
            }
        });
    }

    /// Set the current layer index (None for operations outside layers).
    pub fn set_layer(layer: Option<usize>) {
        PROFILER.with(|p| {
            if let Some(ref mut profiler) = *p.borrow_mut() {
                profiler.set_layer(layer);
            }
        });
    }

    /// Generate and return a profiling report.
    pub fn report() -> ProfileReport {
        PROFILER.with(|p| {
            if let Some(ref profiler) = *p.borrow() {
                profiler.generate_report()
            } else {
                ProfileReport {
                    total_time: Duration::ZERO,
                    steps_recorded: 0,
                    phase_breakdown: HashMap::new(),
                    layer_breakdown: Vec::new(),
                    op_stats: Vec::new(),
                }
            }
        })
    }

    /// Get the number of steps recorded so far.
    pub fn steps_recorded() -> usize {
        PROFILER.with(|p| {
            if let Some(ref profiler) = *p.borrow() {
                profiler.steps_recorded
            } else {
                0
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_disabled_by_default() {
        assert!(!Profiler::is_enabled());
    }

    #[test]
    fn test_profiler_init_and_shutdown() {
        // Shutdown any existing profiler first
        Profiler::shutdown();
        assert!(!Profiler::is_enabled());

        // Initialize
        Profiler::init(ProfilerConfig {
            enabled: true,
            warmup_steps: 0,
            report_interval: 0,
        });
        assert!(Profiler::is_enabled());

        // Shutdown
        Profiler::shutdown();
        assert!(!Profiler::is_enabled());
    }

    #[test]
    fn test_profiler_recording() {
        Profiler::shutdown();
        Profiler::init(ProfilerConfig {
            enabled: true,
            warmup_steps: 0,
            report_interval: 0,
        });

        Profiler::begin_step();
        Profiler::set_phase(Phase::Forward);

        // Simulate some operation timing
        let _timer = timed(OpCategory::Matmul, 1000);
        std::thread::sleep(std::time::Duration::from_millis(1));
        drop(_timer);

        Profiler::end_step();

        let report = Profiler::report();
        assert_eq!(report.steps_recorded, 1);
        assert!(!report.op_stats.is_empty());

        Profiler::shutdown();
    }

    #[test]
    fn test_profiler_warmup() {
        Profiler::shutdown();
        Profiler::init(ProfilerConfig {
            enabled: true,
            warmup_steps: 2,
            report_interval: 0,
        });

        // First two steps should be skipped
        for _ in 0..2 {
            Profiler::begin_step();
            let _timer = timed(OpCategory::Matmul, 1000);
            drop(_timer);
            Profiler::end_step();
        }

        let report = Profiler::report();
        assert_eq!(report.steps_recorded, 0);

        // Third step should be recorded
        Profiler::begin_step();
        let _timer = timed(OpCategory::Matmul, 1000);
        drop(_timer);
        Profiler::end_step();

        let report = Profiler::report();
        assert_eq!(report.steps_recorded, 1);

        Profiler::shutdown();
    }
}
