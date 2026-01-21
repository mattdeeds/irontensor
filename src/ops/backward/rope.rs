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

const BACKWARD_ROPE_SHADER: &str = include_str!("../../shaders/backward/rope.metal");

#[repr(C)]
struct RoPEParams {
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    base: f32,
    position_offset: u32,
}

struct RoPEBackwardPipelines {
    rope_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static ROPE_BACKWARD_PIPELINES: OnceLock<RoPEBackwardPipelines> = OnceLock::new();

fn get_pipelines() -> &'static RoPEBackwardPipelines {
    ROPE_BACKWARD_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(BACKWARD_ROPE_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile backward rope shader: {e}"));

        let func = library
            .newFunctionWithName(&objc2_foundation::NSString::from_str("rope_backward_f32"))
            .expect("rope_backward_f32 function not found");

        let rope_backward = device
            .newComputePipelineStateWithFunction_error(&func)
            .expect("Failed to create rope_backward pipeline");

        RoPEBackwardPipelines { rope_backward }
    })
}

/// RoPE backward pass
/// The backward of rotation is the inverse rotation (transpose)
pub fn rope_backward(grad_output: &Tensor, base: f32, position_offset: usize) -> Tensor {
    assert_eq!(grad_output.precision(), Precision::FP32);

    let shape = grad_output.shape();
    assert_eq!(shape.len(), 4, "Input must be 4D [batch, seq, heads, dim]");

    let batch_size = shape[0];
    let seq_len = shape[1];
    let num_heads = shape[2];
    let head_dim = shape[3];

    assert!(head_dim % 2 == 0);

    let grad_input = Tensor::zeros(shape, Precision::FP32);

    if batch_size == 0 || seq_len == 0 {
        return grad_input;
    }

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = RoPEParams {
        batch_size: batch_size as u32,
        seq_len: seq_len as u32,
        num_heads: num_heads as u32,
        head_dim: head_dim as u32,
        base,
        position_offset: position_offset as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<RoPEParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.rope_backward);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(grad_output.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(grad_input.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
    }

    let grid_size = MTLSize {
        width: head_dim / 2,
        height: seq_len,
        depth: batch_size * num_heads,
    };
    let thread_width = pipelines.rope_backward.threadExecutionWidth();
    let max_threads = pipelines.rope_backward.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize {
        width: thread_width.min(head_dim / 2),
        height: (max_threads / thread_width).min(seq_len).max(1),
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    grad_input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::rope;

    fn reference_rope_backward(
        grad_output: &[f32],
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        base: f32,
        position_offset: usize,
    ) -> Vec<f32> {
        let mut grad_input = vec![0.0f32; grad_output.len()];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..num_heads {
                    for p in 0..head_dim / 2 {
                        let position = s + position_offset;
                        let dim_idx = (p * 2) as f32;
                        let theta = 1.0 / base.powf(dim_idx / head_dim as f32);
                        let angle = position as f32 * theta;

                        let cos_angle = angle.cos();
                        let sin_angle = angle.sin();

                        let offset = b * seq_len * num_heads * head_dim
                            + s * num_heads * head_dim
                            + h * head_dim
                            + p * 2;

                        let go0 = grad_output[offset];
                        let go1 = grad_output[offset + 1];

                        // Inverse rotation (transpose)
                        grad_input[offset] = go0 * cos_angle + go1 * sin_angle;
                        grad_input[offset + 1] = -go0 * sin_angle + go1 * cos_angle;
                    }
                }
            }
        }

        grad_input
    }

    #[test]
    fn test_rope_backward_simple() {
        let batch = 1;
        let seq_len = 2;
        let num_heads = 1;
        let head_dim = 4;

        let grad_out_data: Vec<f32> = (0..(batch * seq_len * num_heads * head_dim))
            .map(|i| i as f32 + 1.0)
            .collect();

        let grad_out = Tensor::from_f32_slice(&grad_out_data, &[batch, seq_len, num_heads, head_dim]);
        let grad_input = rope_backward(&grad_out, 10000.0, 0);

        let result = grad_input.as_f32_slice();
        let expected = reference_rope_backward(&grad_out_data, batch, seq_len, num_heads, head_dim, 10000.0, 0);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }

    #[test]
    fn test_rope_backward_roundtrip() {
        // Apply forward then backward should give identity (rotation is orthogonal)
        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 8;

        let input_data: Vec<f32> = (0..(batch * seq_len * num_heads * head_dim))
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, num_heads, head_dim]);

        // Forward
        let output = rope(&input, 10000.0, 0);

        // Backward with identity gradient
        let grad_out = output.as_f32_slice().to_vec();
        let grad_out_tensor = Tensor::from_f32_slice(&grad_out, &[batch, seq_len, num_heads, head_dim]);
        let recovered = rope_backward(&grad_out_tensor, 10000.0, 0);

        // Should recover original input
        let result = recovered.as_f32_slice();
        for (i, (r, e)) in result.iter().zip(input_data.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "Roundtrip mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }
}
