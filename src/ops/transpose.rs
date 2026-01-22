use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2::AllocAnyThread;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixCopy, MPSMatrixCopyDescriptor, MPSMatrixCopyOffsets,
    MPSMatrixDescriptor,
};

use crate::command_batch::CommandBatch;
use crate::device::MetalContext;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
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
///
/// Note: MPS transpose is 1.05-2.09x faster in isolation benchmarks, but requires CommandBatch::sync()
/// which breaks GPU pipelining. Custom shader is faster in practice during training.
pub fn transpose_2d(input: &Tensor) -> Tensor {
    transpose_2d_custom(input)
}

/// MPS-based 2D transpose using MPSMatrixCopy with transpose flags.
/// 1.05-2.09x faster than custom shader, with larger speedups for bigger matrices.
fn transpose_2d_mps(input: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::Transpose, input.numel());
    assert_eq!(input.precision(), Precision::FP32);
    let shape = input.shape();
    assert_eq!(shape.len(), 2, "transpose_2d requires 2D tensor");

    let rows = shape[0];
    let cols = shape[1];

    if rows == 0 || cols == 0 {
        return Tensor::zeros(&[cols, rows], Precision::FP32);
    }

    let output = Tensor::zeros(&[cols, rows], Precision::FP32);

    // Need to sync before MPS operations since MPS uses its own command buffer
    CommandBatch::sync();

    autoreleasepool(|_| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        // Source: [rows, cols] in row-major
        let src_row_bytes = cols * std::mem::size_of::<f32>();
        let src_desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                rows,
                cols,
                src_row_bytes,
                MPSDataType::Float32,
            )
        };

        // Destination: [cols, rows] in row-major
        let dst_row_bytes = rows * std::mem::size_of::<f32>();
        let dst_desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                cols,
                rows,
                dst_row_bytes,
                MPSDataType::Float32,
            )
        };

        let src_matrix = unsafe {
            MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), input.buffer(), &src_desc)
        };

        let dst_matrix = unsafe {
            MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), output.buffer(), &dst_desc)
        };

        // Create copy kernel with transpose
        // sourcesAreTransposed=true interprets source as transposed (col-major view of row-major data)
        // This effectively performs the transpose during copy
        let kernel = unsafe {
            MPSMatrixCopy::initWithDevice_copyRows_copyColumns_sourcesAreTransposed_destinationsAreTransposed(
                MPSMatrixCopy::alloc(),
                device,
                cols,  // copy cols rows from transposed source view
                rows,  // copy rows columns from transposed source view
                true,  // source is transposed
                false, // destination is not transposed
            )
        };

        let offsets = MPSMatrixCopyOffsets {
            sourceRowOffset: 0,
            sourceColumnOffset: 0,
            destinationRowOffset: 0,
            destinationColumnOffset: 0,
        };

        let copy_desc = unsafe {
            MPSMatrixCopyDescriptor::descriptorWithSourceMatrix_destinationMatrix_offsets(
                &src_matrix,
                &dst_matrix,
                offsets,
            )
        };

        let command_buffer = ctx
            .command_queue()
            .commandBuffer()
            .expect("Failed to create command buffer for MPS");

        unsafe {
            kernel.encodeToCommandBuffer_copyDescriptor(&command_buffer, &copy_desc);
        }

        command_buffer.commit();
        command_buffer.waitUntilCompleted();
    });

    output
}

/// Custom shader-based 2D transpose (kept for reference)
#[allow(dead_code)]
fn transpose_2d_custom(input: &Tensor) -> Tensor {
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

    let input_buf = input.buffer();
    let output_buf = output.buffer();

    if use_tiled {
        const TILE_SIZE: usize = 16;

        // Dispatch in tiles
        let grid_x = (cols + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
        let grid_y = (rows + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
        let grid_size = MTLSize { width: grid_x, height: grid_y, depth: 1 };
        let threadgroup_size = MTLSize { width: TILE_SIZE, height: TILE_SIZE, depth: 1 };

        // Note: For tiled transpose with threadgroup memory, we need to fall back to immediate mode
        // because CommandBatch::dispatch doesn't support setThreadgroupMemoryLength
        let ctx = MetalContext::global();
        let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
        let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

        encoder.setComputePipelineState(&pipelines.transpose_2d_tiled);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
        }

        // Set threadgroup memory size for the tile
        let tile_memory = TILE_SIZE * TILE_SIZE * std::mem::size_of::<f32>();
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(tile_memory, 0);
        }

        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
    } else {
        let thread_width = pipelines.transpose_2d.threadExecutionWidth();
        let grid_size = MTLSize { width: cols, height: rows, depth: 1 };
        let tg_width = thread_width.min(cols);
        let tg_height = (256 / tg_width).min(rows).max(1);
        let threadgroup_size = MTLSize { width: tg_width, height: tg_height, depth: 1 };

        CommandBatch::dispatch(
            &pipelines.transpose_2d,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
            },
            grid_size,
            threadgroup_size,
        );
    }

    output
}

/// Transpose dimensions 1 and 2 of a 4D tensor: [batch, dim1, dim2, dim3] -> [batch, dim2, dim1, dim3]
/// Used for attention: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
pub fn transpose_for_attention(input: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::Transpose, input.numel());
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

    let input_buf = input.buffer();
    let output_buf = output.buffer();

    let thread_width = pipelines.transpose_0213.threadExecutionWidth();
    let grid_size = MTLSize {
        width: dim2 * dim3,  // Flatten dim2 and dim3
        height: dim1,
        depth: batch
    };
    let tg_width = thread_width.min(dim2 * dim3);
    let tg_height = (256 / tg_width).min(dim1).max(1);
    let threadgroup_size = MTLSize { width: tg_width, height: tg_height, depth: 1 };

    CommandBatch::dispatch(
        &pipelines.transpose_0213,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
        },
        grid_size,
        threadgroup_size,
    );

    output
}

/// Transpose dimensions 1 and 2 back: [batch, dim2, dim1, dim3] -> [batch, dim1, dim2, dim3]
/// Inverse of transpose_for_attention
pub fn transpose_from_attention(input: &Tensor, batch: usize, dim1: usize, dim2: usize, dim3: usize) -> Tensor {
    let _timer = timed(OpCategory::Transpose, input.numel());
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

    let input_buf = input.buffer();
    let output_buf = output.buffer();

    let thread_width = pipelines.transpose_0213_inverse.threadExecutionWidth();
    let grid_size = MTLSize {
        width: dim1 * dim3,  // Flatten dim1 and dim3
        height: dim2,
        depth: batch
    };
    let tg_width = thread_width.min(dim1 * dim3);
    let tg_height = (256 / tg_width).min(dim2).max(1);
    let threadgroup_size = MTLSize { width: tg_width, height: tg_height, depth: 1 };

    CommandBatch::dispatch(
        &pipelines.transpose_0213_inverse,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
        },
        grid_size,
        threadgroup_size,
    );

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

    /// Benchmark comparing MPS transpose (via MPSMatrixCopy) vs custom shader
    /// Run with: cargo test benchmark_mps_transpose --release -- --nocapture --ignored
    #[test]
    #[ignore]
    fn benchmark_mps_transpose() {
        use std::time::Instant;
        use objc2::rc::autoreleasepool;
        use objc2::AllocAnyThread;
        use objc2_metal::MTLCommandQueue;
        use objc2_metal_performance_shaders::{
            MPSDataType, MPSMatrix, MPSMatrixCopy, MPSMatrixCopyDescriptor,
            MPSMatrixDescriptor,
        };

        let test_cases = [
            (256, 512),      // Small
            (512, 512),      // Square
            (1024, 512),     // Tall
            (512, 1024),     // Wide
            (2048, 512),     // Larger
            (4096, 512),     // Large (batch*seq, hidden)
        ];

        println!("\n{}", "=".repeat(80));
        println!("Transpose Benchmark: MPS (MPSMatrixCopy) vs Custom Shader (FP32)");
        println!("{}", "=".repeat(80));
        println!("{:>10} {:>10} | {:>12} {:>12} | {:>12}",
                 "rows", "cols", "Custom", "MPS", "Winner");
        println!("{}", "-".repeat(80));

        let warmup_iters = 10;
        let bench_iters = 100;

        for (rows, cols) in test_cases {
            let input_data: Vec<f32> = (0..(rows * cols))
                .map(|i| i as f32)
                .collect();

            let input = Tensor::from_f32_slice(&input_data, &[rows, cols]);

            // Warmup custom shader
            for _ in 0..warmup_iters {
                let _ = transpose_2d(&input);
            }
            CommandBatch::sync();

            // Benchmark custom shader
            let start = Instant::now();
            for _ in 0..bench_iters {
                let _ = transpose_2d(&input);
            }
            CommandBatch::sync();
            let custom_time = start.elapsed().as_secs_f64() / bench_iters as f64 * 1000.0;

            // MPS transpose using MPSMatrixCopy
            // MPSMatrixCopy with sourcesAreTransposed interprets the source as transposed
            // So if we have [rows, cols] and want [cols, rows], we set up the destination
            // as [cols, rows] and tell MPS the source is transposed (row-major -> col-major)
            let output_mps = Tensor::zeros(&[cols, rows], Precision::FP32);

            // Warmup MPS
            for _ in 0..warmup_iters {
                autoreleasepool(|_| {
                    let ctx = MetalContext::global();
                    let device = ctx.device();

                    // Source: [rows, cols] in row-major
                    let src_row_bytes = cols * std::mem::size_of::<f32>();
                    let src_desc = unsafe {
                        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                            rows, cols, src_row_bytes, MPSDataType::Float32,
                        )
                    };

                    // Destination: [cols, rows] in row-major
                    let dst_row_bytes = rows * std::mem::size_of::<f32>();
                    let dst_desc = unsafe {
                        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                            cols, rows, dst_row_bytes, MPSDataType::Float32,
                        )
                    };

                    let src_matrix = unsafe {
                        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), input.buffer(), &src_desc)
                    };

                    let dst_matrix = unsafe {
                        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), output_mps.buffer(), &dst_desc)
                    };

                    // Create copy kernel with transpose
                    // sourcesAreTransposed=true means source is stored transposed (col-major)
                    // destinationsAreTransposed=false means destination is row-major
                    // This should copy rows->cols, cols->rows effectively
                    let kernel = unsafe {
                        MPSMatrixCopy::initWithDevice_copyRows_copyColumns_sourcesAreTransposed_destinationsAreTransposed(
                            MPSMatrixCopy::alloc(),
                            device,
                            cols,  // copy cols rows from source (transposed view)
                            rows,  // copy rows columns from source (transposed view)
                            true,  // source is transposed
                            false, // destination is not transposed
                        )
                    };

                    // Create copy descriptor
                    let offsets = objc2_metal_performance_shaders::MPSMatrixCopyOffsets {
                        sourceRowOffset: 0,
                        sourceColumnOffset: 0,
                        destinationRowOffset: 0,
                        destinationColumnOffset: 0,
                    };

                    let copy_desc = unsafe {
                        MPSMatrixCopyDescriptor::descriptorWithSourceMatrix_destinationMatrix_offsets(
                            &src_matrix,
                            &dst_matrix,
                            offsets,
                        )
                    };

                    let command_buffer = ctx.command_queue()
                        .commandBuffer()
                        .expect("Failed to create command buffer");

                    unsafe {
                        kernel.encodeToCommandBuffer_copyDescriptor(&command_buffer, &copy_desc);
                    }

                    command_buffer.commit();
                    command_buffer.waitUntilCompleted();
                });
            }

            // Benchmark MPS
            let start = Instant::now();
            for _ in 0..bench_iters {
                autoreleasepool(|_| {
                    let ctx = MetalContext::global();
                    let device = ctx.device();

                    let src_row_bytes = cols * std::mem::size_of::<f32>();
                    let src_desc = unsafe {
                        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                            rows, cols, src_row_bytes, MPSDataType::Float32,
                        )
                    };

                    let dst_row_bytes = rows * std::mem::size_of::<f32>();
                    let dst_desc = unsafe {
                        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                            cols, rows, dst_row_bytes, MPSDataType::Float32,
                        )
                    };

                    let src_matrix = unsafe {
                        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), input.buffer(), &src_desc)
                    };

                    let dst_matrix = unsafe {
                        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), output_mps.buffer(), &dst_desc)
                    };

                    let kernel = unsafe {
                        MPSMatrixCopy::initWithDevice_copyRows_copyColumns_sourcesAreTransposed_destinationsAreTransposed(
                            MPSMatrixCopy::alloc(),
                            device,
                            cols,
                            rows,
                            true,
                            false,
                        )
                    };

                    let offsets = objc2_metal_performance_shaders::MPSMatrixCopyOffsets {
                        sourceRowOffset: 0,
                        sourceColumnOffset: 0,
                        destinationRowOffset: 0,
                        destinationColumnOffset: 0,
                    };

                    let copy_desc = unsafe {
                        MPSMatrixCopyDescriptor::descriptorWithSourceMatrix_destinationMatrix_offsets(
                            &src_matrix,
                            &dst_matrix,
                            offsets,
                        )
                    };

                    let command_buffer = ctx.command_queue()
                        .commandBuffer()
                        .expect("Failed to create command buffer");

                    unsafe {
                        kernel.encodeToCommandBuffer_copyDescriptor(&command_buffer, &copy_desc);
                    }

                    command_buffer.commit();
                    command_buffer.waitUntilCompleted();
                });
            }
            let mps_time = start.elapsed().as_secs_f64() / bench_iters as f64 * 1000.0;

            // Verify correctness
            let custom_result = transpose_2d(&input);
            CommandBatch::sync();
            let custom_slice = custom_result.as_f32_slice();
            let mps_slice = output_mps.as_f32_slice();

            let mut max_diff = 0.0f32;
            let mut mismatch_count = 0;
            for (i, (c, m)) in custom_slice.iter().zip(mps_slice.iter()).enumerate() {
                let diff = (c - m).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > 1e-5 {
                    mismatch_count += 1;
                    if mismatch_count <= 3 {
                        // Print first few mismatches for debugging
                        let row = i / rows;
                        let col = i % rows;
                        eprintln!("  Mismatch at [{},{}]: custom={}, mps={}", row, col, c, m);
                    }
                }
            }

            let speedup = custom_time / mps_time;
            let winner = if speedup > 1.0 { "MPS" } else { "Custom" };
            let ratio = if speedup > 1.0 { speedup } else { 1.0 / speedup };

            let correctness = if mismatch_count == 0 { "✓" } else { "✗" };

            println!("{:>10} {:>10} | {:>10.3}ms {:>10.3}ms | {:>8} ({:.2}x) {}",
                     rows, cols, custom_time, mps_time, winner, ratio, correctness);
        }

        println!("{}", "=".repeat(80));
    }
}
