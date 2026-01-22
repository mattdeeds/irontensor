//! Gradient utilities for Lion optimizer.

use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

use crate::command_batch::CommandBatch;
use crate::device::MetalContext;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

use super::pipelines::{create_buffer, get_pipelines};

/// Zero out gradients in a tensor.
pub fn zero_gradients(gradients: &Tensor) {
    let _timer = timed(OpCategory::ZeroGradients, gradients.numel());
    assert_eq!(gradients.precision(), Precision::FP32);

    let count = gradients.numel();
    if count == 0 {
        return;
    }

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let count_u32: u32 = count as u32;
    let count_buffer = create_buffer(ctx, &count_u32);

    let gradients_buf = gradients.buffer();

    let thread_width = pipelines.zero_gradients.threadExecutionWidth();
    let grid_size = MTLSize {
        width: count,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: thread_width.min(count),
        height: 1,
        depth: 1,
    };

    CommandBatch::dispatch(
        &pipelines.zero_gradients,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(gradients_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 1);
        },
        grid_size,
        threadgroup_size,
    );
}

/// Compute the global L2 norm of gradients.
pub fn grad_norm(gradients: &Tensor) -> f32 {
    let _timer = timed(OpCategory::GradientNorm, gradients.numel());
    assert_eq!(gradients.precision(), Precision::FP32);

    let count = gradients.numel();
    if count == 0 {
        return 0.0;
    }

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    // Create buffer for atomic sum
    let zero: f32 = 0.0;
    let sum_sq_buffer = create_buffer(ctx, &zero);

    let count_u32: u32 = count as u32;
    let count_buffer = create_buffer(ctx, &count_u32);

    let gradients_buf = gradients.buffer();

    let thread_width = pipelines.grad_norm_squared.threadExecutionWidth();
    let grid_size = MTLSize {
        width: count,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: thread_width.min(count),
        height: 1,
        depth: 1,
    };

    CommandBatch::dispatch(
        &pipelines.grad_norm_squared,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(gradients_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&sum_sq_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 2);
        },
        grid_size,
        threadgroup_size,
    );

    // Sync before reading result
    CommandBatch::sync();

    // Read back sum of squares
    let sum_sq = unsafe { *(sum_sq_buffer.contents().as_ptr() as *const f32) };
    sum_sq.sqrt()
}

/// Clip gradients by global norm.
///
/// If the global norm exceeds `max_norm`, scales all gradients by `max_norm / actual_norm`.
/// Returns the actual norm before clipping.
pub fn clip_grad_norm(gradients: &Tensor, max_norm: f32) -> f32 {
    let _timer = timed(OpCategory::GradientClip, gradients.numel());
    let actual_norm = grad_norm(gradients);

    if actual_norm <= max_norm || actual_norm == 0.0 {
        return actual_norm;
    }

    let clip_scale = max_norm / actual_norm;

    let count = gradients.numel();
    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let scale_buffer = create_buffer(ctx, &clip_scale);

    let count_u32: u32 = count as u32;
    let count_buffer = create_buffer(ctx, &count_u32);

    let gradients_buf = gradients.buffer();

    let thread_width = pipelines.grad_clip.threadExecutionWidth();
    let grid_size = MTLSize {
        width: count,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: thread_width.min(count),
        height: 1,
        depth: 1,
    };

    CommandBatch::dispatch(
        &pipelines.grad_clip,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(gradients_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&scale_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 2);
        },
        grid_size,
        threadgroup_size,
    );

    actual_norm
}
