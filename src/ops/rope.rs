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

const ROPE_SHADER: &str = include_str!("../shaders/rope.metal");

#[repr(C)]
struct RoPEParams {
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    base: f32,
    position_offset: u32,
}

struct RoPEPipelines {
    rope: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static ROPE_PIPELINES: OnceLock<RoPEPipelines> = OnceLock::new();

fn get_pipelines() -> &'static RoPEPipelines {
    ROPE_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(ROPE_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile RoPE shader: {e}"));

        let func = library
            .newFunctionWithName(&objc2_foundation::NSString::from_str("rope_f32"))
            .expect("rope_f32 function not found");

        let rope = device
            .newComputePipelineStateWithFunction_error(&func)
            .expect("Failed to create RoPE pipeline");

        RoPEPipelines { rope }
    })
}

/// Apply RoPE (Rotary Position Embedding) to input tensor
///
/// Input shape: [batch, seq_len, num_heads, head_dim]
/// - head_dim must be even (rotations work on pairs)
/// - base: typically 10000.0
/// - position_offset: for KV cache continuation (default 0)
///
/// Returns tensor with same shape as input
pub fn rope(input: &Tensor, base: f32, position_offset: usize) -> Tensor {
    assert_eq!(input.precision(), Precision::FP32);

    let shape = input.shape();
    assert_eq!(shape.len(), 4, "Input must be 4D [batch, seq_len, num_heads, head_dim]");

    let batch_size = shape[0];
    let seq_len = shape[1];
    let num_heads = shape[2];
    let head_dim = shape[3];

    assert!(head_dim % 2 == 0, "head_dim must be even, got {}", head_dim);

    let output = Tensor::zeros(shape, Precision::FP32);

    if batch_size == 0 || seq_len == 0 {
        return output;
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

    encoder.setComputePipelineState(&pipelines.rope);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
    }

    let grid_size = MTLSize {
        width: head_dim / 2,
        height: seq_len,
        depth: batch_size * num_heads,
    };
    let thread_width = pipelines.rope.threadExecutionWidth();
    let max_threads = pipelines.rope.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize {
        width: thread_width.min(head_dim / 2),
        height: (max_threads / thread_width).min(seq_len).max(1),
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

/// Convenience function with default base=10000.0 and position_offset=0
pub fn rope_default(input: &Tensor) -> Tensor {
    rope(input, 10000.0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_rope(
        input: &[f32],
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        base: f32,
        position_offset: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; input.len()];

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

                        let x0 = input[offset];
                        let x1 = input[offset + 1];

                        output[offset] = x0 * cos_angle - x1 * sin_angle;
                        output[offset + 1] = x0 * sin_angle + x1 * cos_angle;
                    }
                }
            }
        }

        output
    }

    #[test]
    fn test_rope_simple() {
        let batch = 1;
        let seq_len = 2;
        let num_heads = 1;
        let head_dim = 4;

        let input_data: Vec<f32> = (0..(batch * seq_len * num_heads * head_dim))
            .map(|i| i as f32 + 1.0)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, num_heads, head_dim]);
        let output = rope(&input, 10000.0, 0);

        let result = output.as_f32_slice();
        let expected = reference_rope(&input_data, batch, seq_len, num_heads, head_dim, 10000.0, 0);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }

    #[test]
    fn test_rope_position_zero() {
        // At position 0, all angles are 0, so cos=1, sin=0
        // Output should equal input
        let batch = 1;
        let seq_len = 1;
        let num_heads = 2;
        let head_dim = 8;

        let input_data: Vec<f32> = (0..(batch * seq_len * num_heads * head_dim))
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, num_heads, head_dim]);
        let output = rope(&input, 10000.0, 0);

        let result = output.as_f32_slice();

        // At position 0, cos(0)=1, sin(0)=0, so output = input
        for (i, (r, e)) in result.iter().zip(input_data.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }

    #[test]
    fn test_rope_larger() {
        let batch = 2;
        let seq_len = 16;
        let num_heads = 4;
        let head_dim = 64;

        let input_data: Vec<f32> = (0..(batch * seq_len * num_heads * head_dim))
            .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, num_heads, head_dim]);
        let output = rope(&input, 10000.0, 0);

        let result = output.as_f32_slice();
        let expected = reference_rope(&input_data, batch, seq_len, num_heads, head_dim, 10000.0, 0);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }

    #[test]
    fn test_rope_with_offset() {
        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 8;
        let position_offset = 10;

        let input_data: Vec<f32> = (0..(batch * seq_len * num_heads * head_dim))
            .map(|i| i as f32 * 0.1)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, num_heads, head_dim]);
        let output = rope(&input, 10000.0, position_offset);

        let result = output.as_f32_slice();
        let expected = reference_rope(&input_data, batch, seq_len, num_heads, head_dim, 10000.0, position_offset);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }
}
