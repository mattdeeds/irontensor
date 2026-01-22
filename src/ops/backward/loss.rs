use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::command_batch::CommandBatch;
use crate::device::MetalContext;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const LOSS_SHADER: &str = include_str!("../../shaders/loss.metal");

#[repr(C)]
struct CrossEntropyParams {
    batch_size: u32,
    vocab_size: u32,
}

struct LossPipelines {
    cross_entropy_forward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    cross_entropy_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    cross_entropy_fused: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    reduce_mean: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static LOSS_PIPELINES: OnceLock<LossPipelines> = OnceLock::new();

fn get_pipelines() -> &'static LossPipelines {
    LOSS_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(LOSS_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile loss shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        LossPipelines {
            cross_entropy_forward: make_pipeline("cross_entropy_forward_f32"),
            cross_entropy_backward: make_pipeline("cross_entropy_backward_f32"),
            cross_entropy_fused: make_pipeline("cross_entropy_fused_f32"),
            reduce_mean: make_pipeline("reduce_mean_f32"),
        }
    })
}

/// Cross-entropy loss (forward only)
/// logits: [batch, vocab_size]
/// targets: [batch] - target class indices
/// Returns: (mean_loss, per_sample_losses)
pub fn cross_entropy(logits: &Tensor, targets: &[u32]) -> (f32, Tensor) {
    let _timer = timed(OpCategory::CrossEntropyBackward, logits.numel());
    assert_eq!(logits.precision(), Precision::FP32);

    let shape = logits.shape();
    assert_eq!(shape.len(), 2);

    let batch_size = shape[0];
    let vocab_size = shape[1];

    assert_eq!(targets.len(), batch_size);

    let losses = Tensor::zeros(&[batch_size], Precision::FP32);
    let mean_loss = Tensor::zeros(&[1], Precision::FP32);

    if batch_size == 0 {
        return (0.0, losses);
    }

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let targets_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(targets.as_ptr() as *mut _).unwrap(),
            targets.len() * std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create targets buffer");

    let params = CrossEntropyParams {
        batch_size: batch_size as u32,
        vocab_size: vocab_size as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<CrossEntropyParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    // Forward pass
    let logits_buf = logits.buffer();
    let losses_buf = losses.buffer();

    let thread_width = pipelines.cross_entropy_forward.threadExecutionWidth();
    let grid_size = MTLSize { width: batch_size, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: thread_width.min(batch_size), height: 1, depth: 1 };

    CommandBatch::dispatch(
        &pipelines.cross_entropy_forward,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&targets_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(losses_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        },
        grid_size,
        threadgroup_size,
    );

    // Sync before reduce (depends on losses being computed)
    CommandBatch::sync();

    // Reduce to mean
    let count_u32: u32 = batch_size as u32;
    let count_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&count_u32 as *const _ as *mut _).unwrap(),
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create count buffer");

    let mean_loss_buf = mean_loss.buffer();

    let grid_size = MTLSize { width: 1, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: 1, height: 1, depth: 1 };

    CommandBatch::dispatch(
        &pipelines.reduce_mean,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(losses_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(mean_loss_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 2);
        },
        grid_size,
        threadgroup_size,
    );

    // Sync before reading result
    CommandBatch::sync();

    let mean = mean_loss.as_f32_slice()[0];
    (mean, losses)
}

/// Cross-entropy backward (gradient of softmax + cross-entropy w.r.t. logits)
/// This computes: grad_logits = softmax(logits) - one_hot(targets)
pub fn cross_entropy_backward(logits: &Tensor, targets: &[u32]) -> Tensor {
    let _timer = timed(OpCategory::CrossEntropyBackward, logits.numel());
    assert_eq!(logits.precision(), Precision::FP32);

    let shape = logits.shape();
    assert_eq!(shape.len(), 2);

    let batch_size = shape[0];
    let vocab_size = shape[1];

    assert_eq!(targets.len(), batch_size);

    let grad_logits = Tensor::zeros(shape, Precision::FP32);

    if batch_size == 0 {
        return grad_logits;
    }

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let targets_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(targets.as_ptr() as *mut _).unwrap(),
            targets.len() * std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create targets buffer");

    let params = CrossEntropyParams {
        batch_size: batch_size as u32,
        vocab_size: vocab_size as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<CrossEntropyParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    // grad_scale = 1/batch_size for mean reduction
    let grad_scale: f32 = 1.0 / batch_size as f32;
    let grad_scale_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&grad_scale as *const _ as *mut _).unwrap(),
            std::mem::size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create grad_scale buffer");

    let logits_buf = logits.buffer();
    let grad_logits_buf = grad_logits.buffer();

    let thread_width = pipelines.cross_entropy_backward.threadExecutionWidth();
    let grid_size = MTLSize { width: batch_size, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: thread_width.min(batch_size), height: 1, depth: 1 };

    CommandBatch::dispatch(
        &pipelines.cross_entropy_backward,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&targets_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(grad_logits_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&grad_scale_buffer), 0, 4);
        },
        grid_size,
        threadgroup_size,
    );

    grad_logits
}

/// Fused cross-entropy: computes both loss and gradients in one pass
/// Returns (mean_loss, per_sample_losses, grad_logits)
pub fn cross_entropy_fused(logits: &Tensor, targets: &[u32]) -> (f32, Tensor, Tensor) {
    let _timer = timed(OpCategory::CrossEntropyBackward, logits.numel());
    assert_eq!(logits.precision(), Precision::FP32);

    let shape = logits.shape();
    assert_eq!(shape.len(), 2);

    let batch_size = shape[0];
    let vocab_size = shape[1];

    assert_eq!(targets.len(), batch_size);

    let losses = Tensor::zeros(&[batch_size], Precision::FP32);
    let grad_logits = Tensor::zeros(shape, Precision::FP32);
    let mean_loss = Tensor::zeros(&[1], Precision::FP32);

    if batch_size == 0 {
        return (0.0, losses, grad_logits);
    }

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let targets_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(targets.as_ptr() as *mut _).unwrap(),
            targets.len() * std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create targets buffer");

    let params = CrossEntropyParams {
        batch_size: batch_size as u32,
        vocab_size: vocab_size as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<CrossEntropyParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let grad_scale: f32 = 1.0 / batch_size as f32;
    let grad_scale_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&grad_scale as *const _ as *mut _).unwrap(),
            std::mem::size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create grad_scale buffer");

    // Fused forward + backward
    let logits_buf = logits.buffer();
    let losses_buf = losses.buffer();
    let grad_logits_buf = grad_logits.buffer();

    let thread_width = pipelines.cross_entropy_fused.threadExecutionWidth();
    let grid_size = MTLSize { width: batch_size, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: thread_width.min(batch_size), height: 1, depth: 1 };

    CommandBatch::dispatch(
        &pipelines.cross_entropy_fused,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(logits_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&targets_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(losses_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(grad_logits_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&grad_scale_buffer), 0, 5);
        },
        grid_size,
        threadgroup_size,
    );

    // Sync before reduce (depends on losses being computed)
    CommandBatch::sync();

    // Reduce to mean
    let count_u32: u32 = batch_size as u32;
    let count_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&count_u32 as *const _ as *mut _).unwrap(),
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create count buffer");

    let mean_loss_buf = mean_loss.buffer();

    let grid_size = MTLSize { width: 1, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: 1, height: 1, depth: 1 };

    CommandBatch::dispatch(
        &pipelines.reduce_mean,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(losses_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(mean_loss_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 2);
        },
        grid_size,
        threadgroup_size,
    );

    // Sync before reading result
    CommandBatch::sync();

    let mean = mean_loss.as_f32_slice()[0];
    (mean, losses, grad_logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_cross_entropy(logits: &[f32], targets: &[u32], vocab_size: usize) -> (f32, Vec<f32>) {
        let batch_size = targets.len();
        let mut losses = vec![0.0f32; batch_size];

        for b in 0..batch_size {
            let offset = b * vocab_size;
            let row = &logits[offset..offset + vocab_size];
            let target = targets[b] as usize;

            // Find max for stability
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Log-sum-exp
            let sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
            let log_sum_exp = max_val + sum_exp.ln();

            losses[b] = -row[target] + log_sum_exp;
        }

        let mean = losses.iter().sum::<f32>() / batch_size as f32;
        (mean, losses)
    }

    fn reference_cross_entropy_backward(logits: &[f32], targets: &[u32], vocab_size: usize) -> Vec<f32> {
        let batch_size = targets.len();
        let mut grad = vec![0.0f32; logits.len()];
        let grad_scale = 1.0 / batch_size as f32;

        for b in 0..batch_size {
            let offset = b * vocab_size;
            let row = &logits[offset..offset + vocab_size];
            let target = targets[b] as usize;

            // Softmax
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_vals.iter().sum();

            for i in 0..vocab_size {
                let softmax_val = exp_vals[i] / sum_exp;
                let target_val = if i == target { 1.0 } else { 0.0 };
                grad[offset + i] = (softmax_val - target_val) * grad_scale;
            }
        }

        grad
    }

    #[test]
    fn test_cross_entropy_simple() {
        let batch_size = 2;
        let vocab_size = 5;

        let logits_data: Vec<f32> = (0..(batch_size * vocab_size))
            .map(|i| (i as f32 - 5.0) * 0.5)
            .collect();
        let targets = vec![2u32, 4];

        let logits = Tensor::from_f32_slice(&logits_data, &[batch_size, vocab_size]);

        let (mean_loss, losses) = cross_entropy(&logits, &targets);
        let (expected_mean, expected_losses) = reference_cross_entropy(&logits_data, &targets, vocab_size);

        assert!(
            (mean_loss - expected_mean).abs() < 1e-5,
            "Mean loss mismatch: expected {}, got {}",
            expected_mean, mean_loss
        );

        let loss_result = losses.as_f32_slice();
        for (i, (r, e)) in loss_result.iter().zip(expected_losses.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Loss mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }

    #[test]
    fn test_cross_entropy_backward_simple() {
        let batch_size = 2;
        let vocab_size = 5;

        let logits_data: Vec<f32> = (0..(batch_size * vocab_size))
            .map(|i| (i as f32 - 5.0) * 0.5)
            .collect();
        let targets = vec![2u32, 4];

        let logits = Tensor::from_f32_slice(&logits_data, &[batch_size, vocab_size]);

        let grad_logits = cross_entropy_backward(&logits, &targets);
        let expected_grad = reference_cross_entropy_backward(&logits_data, &targets, vocab_size);

        let result = grad_logits.as_f32_slice();
        for (i, (r, e)) in result.iter().zip(expected_grad.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Grad mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }

    #[test]
    fn test_cross_entropy_fused() {
        let batch_size = 4;
        let vocab_size = 10;

        let logits_data: Vec<f32> = (0..(batch_size * vocab_size))
            .map(|i| ((i % 20) as f32 - 10.0) * 0.3)
            .collect();
        let targets: Vec<u32> = (0..batch_size).map(|i| (i * 3 % vocab_size) as u32).collect();

        let logits = Tensor::from_f32_slice(&logits_data, &[batch_size, vocab_size]);

        let (mean_loss, losses, grad_logits) = cross_entropy_fused(&logits, &targets);

        // Verify against separate forward/backward
        let (expected_mean, expected_losses) = reference_cross_entropy(&logits_data, &targets, vocab_size);
        let expected_grad = reference_cross_entropy_backward(&logits_data, &targets, vocab_size);

        assert!(
            (mean_loss - expected_mean).abs() < 1e-4,
            "Mean loss mismatch: expected {}, got {}",
            expected_mean, mean_loss
        );

        let loss_result = losses.as_f32_slice();
        for (i, (r, e)) in loss_result.iter().zip(expected_losses.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "Loss mismatch at {}", i);
        }

        let grad_result = grad_logits.as_f32_slice();
        for (i, (r, e)) in grad_result.iter().zip(expected_grad.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "Grad mismatch at {}", i);
        }
    }
}
