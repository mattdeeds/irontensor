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
use crate::tensor::Tensor;

const NORM_SHADER: &str = include_str!("../shaders/norm.metal");
const RMSNORM_THREADS: usize = 256;

#[repr(C)]
struct RMSNormParams {
    batch_seq: u32,
    hidden_dim: u32,
    eps: f32,
}

struct NormPipelines {
    rmsnorm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rmsnorm_fast: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static NORM_PIPELINES: OnceLock<NormPipelines> = OnceLock::new();

fn get_pipelines() -> &'static NormPipelines {
    NORM_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(NORM_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile norm shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        NormPipelines {
            rmsnorm: make_pipeline("rmsnorm_f32"),
            rmsnorm_fast: make_pipeline("rmsnorm_fast_f32"),
        }
    })
}

/// RMSNorm: output = (x / sqrt(mean(x^2) + eps)) * gamma
///
/// Input shapes:
/// - input: [..., hidden_dim] - last dimension is normalized
/// - gamma: [hidden_dim] - learnable scale parameter
///
/// Returns tensor with same shape as input
pub fn rmsnorm(input: &Tensor, gamma: &Tensor, eps: f32) -> Tensor {
    assert_eq!(input.precision(), Precision::FP32);
    assert_eq!(gamma.precision(), Precision::FP32);

    let shape = input.shape();
    assert!(shape.len() >= 1, "Input must have at least 1 dimension");

    let hidden_dim = shape[shape.len() - 1];
    assert_eq!(
        gamma.shape(),
        &[hidden_dim],
        "Gamma must have shape [hidden_dim]"
    );

    // Compute batch_seq (product of all dims except last)
    let batch_seq: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);

    let output = Tensor::zeros(shape, Precision::FP32);

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = RMSNormParams {
        batch_seq: batch_seq as u32,
        hidden_dim: hidden_dim as u32,
        eps,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<RMSNormParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    // Use fast kernel for larger hidden dimensions
    let use_fast = hidden_dim >= RMSNORM_THREADS;

    if use_fast {
        encoder.setComputePipelineState(&pipelines.rmsnorm_fast);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gamma.buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        }

        // One threadgroup per row, using dispatchThreadgroups
        let threadgroup_count = MTLSize {
            width: batch_seq,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: RMSNORM_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroup_count, threadgroup_size);
    } else {
        encoder.setComputePipelineState(&pipelines.rmsnorm);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(gamma.buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        }

        // One thread per row
        let grid_size = MTLSize {
            width: batch_seq,
            height: 1,
            depth: 1,
        };
        let max_threads = pipelines.rmsnorm.threadExecutionWidth();
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

    fn reference_rmsnorm(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let hidden_dim = gamma.len();
        let batch_seq = input.len() / hidden_dim;
        let mut output = vec![0.0f32; input.len()];

        for b in 0..batch_seq {
            let offset = b * hidden_dim;
            let row = &input[offset..offset + hidden_dim];

            // Compute sum of squares
            let sum_sq: f32 = row.iter().map(|x| x * x).sum();
            let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;

            // Normalize and scale
            for i in 0..hidden_dim {
                output[offset + i] = row[i] * inv_rms * gamma[i];
            }
        }

        output
    }

    #[test]
    fn test_rmsnorm_simple() {
        let hidden_dim = 4;
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma_data = vec![1.0f32; hidden_dim];
        let eps = 1e-5;

        let input = Tensor::from_f32_slice(&input_data, &[hidden_dim]);
        let gamma = Tensor::from_f32_slice(&gamma_data, &[hidden_dim]);

        let output = rmsnorm(&input, &gamma, eps);
        let result = output.as_f32_slice();

        let expected = reference_rmsnorm(&input_data, &gamma_data, eps);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }

    #[test]
    fn test_rmsnorm_batch() {
        let batch = 3;
        let hidden_dim = 8;
        let input_data: Vec<f32> = (0..(batch * hidden_dim))
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let gamma_data: Vec<f32> = (0..hidden_dim).map(|i| 1.0 + i as f32 * 0.1).collect();
        let eps = 1e-5;

        let input = Tensor::from_f32_slice(&input_data, &[batch, hidden_dim]);
        let gamma = Tensor::from_f32_slice(&gamma_data, &[hidden_dim]);

        let output = rmsnorm(&input, &gamma, eps);
        let result = output.as_f32_slice();

        let expected = reference_rmsnorm(&input_data, &gamma_data, eps);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }

    #[test]
    fn test_rmsnorm_large() {
        // Test with hidden_dim >= RMSNORM_THREADS to use fast kernel
        let batch = 2;
        let seq_len = 4;
        let hidden_dim = 512;
        let input_data: Vec<f32> = (0..(batch * seq_len * hidden_dim))
            .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
            .collect();
        let gamma_data: Vec<f32> = (0..hidden_dim).map(|i| 1.0 + (i as f32).sin() * 0.1).collect();
        let eps = 1e-5;

        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, hidden_dim]);
        let gamma = Tensor::from_f32_slice(&gamma_data, &[hidden_dim]);

        let output = rmsnorm(&input, &gamma, eps);
        let result = output.as_f32_slice();

        let expected = reference_rmsnorm(&input_data, &gamma_data, eps);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-3,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }
}
