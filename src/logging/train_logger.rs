//! Training-specific logging helpers.

use std::collections::HashMap;

use super::formats::{OpTimingRecord, TrainStepRecord};

/// Create a TrainStepRecord with optional phase timing and operation breakdown.
#[allow(clippy::too_many_arguments)]
pub fn create_step_record(
    step: usize,
    epoch: usize,
    loss: f32,
    grad_norm: f32,
    learning_rate: f32,
    tokens_per_sec: f32,
    step_time_ms: f32,
    batch_size: usize,
    seq_len: usize,
    phase_times: Option<HashMap<String, f32>>,
    op_breakdown: Option<Vec<OpTimingRecord>>,
) -> TrainStepRecord {
    let mut record = TrainStepRecord::new(
        step,
        epoch,
        loss,
        grad_norm,
        learning_rate,
        tokens_per_sec,
        step_time_ms,
        batch_size,
        seq_len,
    );

    if let Some(times) = phase_times {
        record = record.with_phase_times(times);
    }

    if let Some(ops) = op_breakdown {
        record = record.with_op_breakdown(ops);
    }

    record
}
