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

const TRANSPOSE_SHADER: &str = include_str!("../shaders/transpose.metal");

#[repr(C)]
struct TransposeParams {
    rows: u32,
    cols: u32,
}

#[repr(C)]
struct BatchedTransposeParams {
    batch: u32,
    dim1: u32,
    dim2: u32,
    dim3: u32,
}

struct TransposePipelines {
    transpose_2d: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    transpose_2d_tiled: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    transpose_0213: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    transpose_0213_inverse: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static TRANSPOSE_PIPELINES: OnceLock<TransposePipelines> = OnceLock::new();

fn get_pipelines() -> &'static TransposePipelines {
    TRANSPOSE_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(TRANSPOSE_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile transpose shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        TransposePipelines {
            transpose_2d: make_pipeline("transpose_2d_f32"),
            transpose_2d_tiled: make_pipeline("transpose_2d_tiled_f32"),
            transpose_0213: make_pipeline("transpose_0213_f32"),
            transpose_0213_inverse: make_pipeline("transpose_0213_inverse_f32"),
        }
    })
}

/// Transpose a 2D tensor: [rows, cols] -> [cols, rows]
pub fn transpose_2d(input: &Tensor) -> Tensor {
    assert_eq!(input.precision(), Precision::FP32);
    let shape = input.shape();
    assert_eq!(shape.len(), 2, "transpose_2d requires 2D tensor");

    let rows = shape[0];
    let cols = shape[1];

    if rows == 0 || cols == 0 {
        return Tensor::zeros(&[cols, rows], Precision::FP32);
    }

    let output = Tensor::zeros(&[cols, rows], Precision::FP32);
    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    // Use tiled version for larger matrices
    let use_tiled = rows >= 32 && cols >= 32;

    let params = TransposeParams {
        rows: rows as u32,
        cols: cols as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<TransposeParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    if use_tiled {
        const TILE_SIZE: usize = 16;

        encoder.setComputePipelineState(&pipelines.transpose_2d_tiled);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
        }

        // Set threadgroup memory size for the tile
        let tile_memory = TILE_SIZE * TILE_SIZE * std::mem::size_of::<f32>();
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(tile_memory, 0);
        }

        // Dispatch in tiles
        let grid_x = (cols + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
        let grid_y = (rows + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
        let grid_size = MTLSize { width: grid_x, height: grid_y, depth: 1 };
        let threadgroup_size = MTLSize { width: TILE_SIZE, height: TILE_SIZE, depth: 1 };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    } else {
        encoder.setComputePipelineState(&pipelines.transpose_2d);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
        }

        let thread_width = pipelines.transpose_2d.threadExecutionWidth();
        let grid_size = MTLSize { width: cols, height: rows, depth: 1 };
        let tg_width = thread_width.min(cols);
        let tg_height = (256 / tg_width).min(rows).max(1);
        let threadgroup_size = MTLSize { width: tg_width, height: tg_height, depth: 1 };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    }

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

/// Transpose dimensions 1 and 2 of a 4D tensor: [batch, dim1, dim2, dim3] -> [batch, dim2, dim1, dim3]
/// Used for attention: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
pub fn transpose_for_attention(input: &Tensor) -> Tensor {
    assert_eq!(input.precision(), Precision::FP32);
    let shape = input.shape();
    assert_eq!(shape.len(), 4, "transpose_for_attention requires 4D tensor");

    let batch = shape[0];
    let dim1 = shape[1];  // seq
    let dim2 = shape[2];  // heads
    let dim3 = shape[3];  // head_dim

    if batch == 0 || dim1 == 0 || dim2 == 0 || dim3 == 0 {
        return Tensor::zeros(&[batch, dim2, dim1, dim3], Precision::FP32);
    }

    let output = Tensor::zeros(&[batch, dim2, dim1, dim3], Precision::FP32);
    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = BatchedTransposeParams {
        batch: batch as u32,
        dim1: dim1 as u32,
        dim2: dim2 as u32,
        dim3: dim3 as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<BatchedTransposeParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.transpose_0213);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
    }

    let thread_width = pipelines.transpose_0213.threadExecutionWidth();
    let grid_size = MTLSize {
        width: dim2 * dim3,  // Flatten dim2 and dim3
        height: dim1,
        depth: batch
    };
    let tg_width = thread_width.min(dim2 * dim3);
    let tg_height = (256 / tg_width).min(dim1).max(1);
    let threadgroup_size = MTLSize { width: tg_width, height: tg_height, depth: 1 };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

/// Transpose dimensions 1 and 2 back: [batch, dim2, dim1, dim3] -> [batch, dim1, dim2, dim3]
/// Inverse of transpose_for_attention
pub fn transpose_from_attention(input: &Tensor, batch: usize, dim1: usize, dim2: usize, dim3: usize) -> Tensor {
    assert_eq!(input.precision(), Precision::FP32);
    let shape = input.shape();
    assert_eq!(shape.len(), 4, "transpose_from_attention requires 4D tensor");
    assert_eq!(shape, &[batch, dim2, dim1, dim3], "Input shape mismatch");

    if batch == 0 || dim1 == 0 || dim2 == 0 || dim3 == 0 {
        return Tensor::zeros(&[batch, dim1, dim2, dim3], Precision::FP32);
    }

    let output = Tensor::zeros(&[batch, dim1, dim2, dim3], Precision::FP32);
    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = BatchedTransposeParams {
        batch: batch as u32,
        dim1: dim1 as u32,
        dim2: dim2 as u32,
        dim3: dim3 as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<BatchedTransposeParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.transpose_0213_inverse);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
    }

    let thread_width = pipelines.transpose_0213_inverse.threadExecutionWidth();
    let grid_size = MTLSize {
        width: dim1 * dim3,  // Flatten dim1 and dim3
        height: dim2,
        depth: batch
    };
    let tg_width = thread_width.min(dim1 * dim3);
    let tg_height = (256 / tg_width).min(dim2).max(1);
    let threadgroup_size = MTLSize { width: tg_width, height: tg_height, depth: 1 };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_2d_small() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::from_f32_slice(&data, &[2, 3]);

        let output = transpose_2d(&input);

        assert_eq!(output.shape(), &[3, 2]);
        let result = output.as_f32_slice();
        // Expected: [[1, 4], [2, 5], [3, 6]]
        assert_eq!(result, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_2d_large() {
        let rows = 64;
        let cols = 128;
        let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32).collect();
        let input = Tensor::from_f32_slice(&data, &[rows, cols]);

        let output = transpose_2d(&input);

        assert_eq!(output.shape(), &[cols, rows]);
        let result = output.as_f32_slice();

        // Check a few elements
        for i in 0..rows.min(10) {
            for j in 0..cols.min(10) {
                let orig_val = (i * cols + j) as f32;
                let transposed_val = result[j * rows + i];
                assert_eq!(transposed_val, orig_val, "Mismatch at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_transpose_for_attention() {
        let batch = 2;
        let seq = 4;
        let heads = 3;
        let head_dim = 2;

        let data: Vec<f32> = (0..(batch * seq * heads * head_dim)).map(|i| i as f32).collect();
        let input = Tensor::from_f32_slice(&data, &[batch, seq, heads, head_dim]);

        let output = transpose_for_attention(&input);

        assert_eq!(output.shape(), &[batch, heads, seq, head_dim]);
        let result = output.as_f32_slice();

        // Verify: input[b, s, h, d] should equal output[b, h, s, d]
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..heads {
                    for d in 0..head_dim {
                        let input_idx = b * seq * heads * head_dim + s * heads * head_dim + h * head_dim + d;
                        let output_idx = b * heads * seq * head_dim + h * seq * head_dim + s * head_dim + d;
                        assert_eq!(
                            result[output_idx], data[input_idx],
                            "Mismatch at b={}, s={}, h={}, d={}", b, s, h, d
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_transpose_roundtrip() {
        let batch = 2;
        let seq = 4;
        let heads = 3;
        let head_dim = 2;

        let data: Vec<f32> = (0..(batch * seq * heads * head_dim)).map(|i| i as f32).collect();
        let input = Tensor::from_f32_slice(&data, &[batch, seq, heads, head_dim]);

        let transposed = transpose_for_attention(&input);
        let restored = transpose_from_attention(&transposed, batch, seq, heads, head_dim);

        assert_eq!(restored.shape(), input.shape());
        let result = restored.as_f32_slice();

        for (i, (&r, &e)) in result.iter().zip(data.iter()).enumerate() {
            assert_eq!(r, e, "Mismatch at index {}", i);
        }
    }
}
