use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::device::MetalContext;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const SOFTMAX_SHADER: &str = include_str!("../shaders/softmax.metal");
const SOFTMAX_THREADS: usize = 256;

#[repr(C)]
struct SoftmaxParams {
    batch_seq: u32,
    dim: u32,
}

struct SoftmaxPipelines {
    softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    softmax_fast: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static SOFTMAX_PIPELINES: OnceLock<SoftmaxPipelines> = OnceLock::new();

fn get_pipelines() -> &'static SoftmaxPipelines {
    SOFTMAX_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(SOFTMAX_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile softmax shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        SoftmaxPipelines {
            softmax: make_pipeline("softmax_f32"),
            softmax_fast: make_pipeline("softmax_fast_f32"),
        }
    })
}

/// Softmax: output_i = exp(input_i - max) / sum(exp(input - max))
///
/// Applies softmax along the last dimension.
///
/// Input shapes:
/// - input: [..., dim] - softmax is applied over the last dimension
///
/// Returns tensor with same shape as input
pub fn softmax(input: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::Softmax, input.numel());
    assert_eq!(input.precision(), Precision::FP32);

    let shape = input.shape();
    assert!(!shape.is_empty(), "Input must have at least 1 dimension");

    let dim = shape[shape.len() - 1];
    let batch_seq: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);

    let output = Tensor::zeros(shape, Precision::FP32);

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = SoftmaxParams {
        batch_seq: batch_seq as u32,
        dim: dim as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<SoftmaxParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    let use_fast = dim >= SOFTMAX_THREADS;

    if use_fast {
        encoder.setComputePipelineState(&pipelines.softmax_fast);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
        }

        let threadgroup_count = MTLSize {
            width: batch_seq,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: SOFTMAX_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroup_count, threadgroup_size);
    } else {
        encoder.setComputePipelineState(&pipelines.softmax);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
        }

        let grid_size = MTLSize {
            width: batch_seq,
            height: 1,
            depth: 1,
        };
        let max_threads = pipelines.softmax.threadExecutionWidth();
        let threadgroup_size = MTLSize {
            width: max_threads.min(batch_seq),
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    }

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_softmax(input: &[f32], dim: usize) -> Vec<f32> {
        let batch_seq = input.len() / dim;
        let mut output = vec![0.0f32; input.len()];

        for b in 0..batch_seq {
            let offset = b * dim;
            let row = &input[offset..offset + dim];

            // Find max
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Compute exp and sum
            let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exp_vals.iter().sum();

            // Normalize
            for (i, &exp_val) in exp_vals.iter().enumerate() {
                output[offset + i] = exp_val / sum;
            }
        }

        output
    }

    #[test]
    fn test_softmax_simple() {
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = Tensor::from_f32_slice(&input_data, &[4]);

        let output = softmax(&input);
        let result = output.as_f32_slice();

        let expected = reference_softmax(&input_data, 4);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }

        // Verify sum is 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Sum should be 1, got {}", sum);
    }

    #[test]
    fn test_softmax_batch() {
        let batch = 3;
        let dim = 5;
        let input_data: Vec<f32> = (0..(batch * dim))
            .map(|i| (i as f32 - 7.0) * 0.5)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch, dim]);

        let output = softmax(&input);
        let result = output.as_f32_slice();

        let expected = reference_softmax(&input_data, dim);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }

        // Verify each row sums to 1
        for b in 0..batch {
            let row_sum: f32 = result[b * dim..(b + 1) * dim].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "Row {} sum should be 1, got {}",
                b, row_sum
            );
        }
    }

    #[test]
    fn test_softmax_large() {
        // Test with dim >= SOFTMAX_THREADS to use fast kernel
        let batch = 4;
        let seq_len = 8;
        let dim = 512;
        let input_data: Vec<f32> = (0..(batch * seq_len * dim))
            .map(|i| ((i % 100) as f32 - 50.0) * 0.1)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, dim]);

        let output = softmax(&input);
        let result = output.as_f32_slice();

        let expected = reference_softmax(&input_data, dim);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Test with large values to verify numerical stability
        let input_data = vec![1000.0f32, 1001.0, 1002.0, 1003.0];
        let input = Tensor::from_f32_slice(&input_data, &[4]);

        let output = softmax(&input);
        let result = output.as_f32_slice();

        // Should not have NaN or Inf
        for (i, &val) in result.iter().enumerate() {
            assert!(val.is_finite(), "Result {} is not finite: {}", i, val);
            assert!(val >= 0.0 && val <= 1.0, "Result {} out of range: {}", i, val);
        }

        // Verify sum is 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Sum should be 1, got {}", sum);
    }
}
