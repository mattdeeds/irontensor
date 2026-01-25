use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2::AllocAnyThread;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixSoftMax};

use crate::command_batch::CommandBatch;
use crate::device::MetalContext;
use crate::error::{TensorError, TensorResult};
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

/// Softmax: output_i = exp(input_i - max) / sum(exp(input - max)) (fallible version)
///
/// Applies softmax along the last dimension.
///
/// Input shapes:
/// - input: [..., dim] - softmax is applied over the last dimension
///
/// Returns tensor with same shape as input
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if input is not FP32
/// - `TensorError::EmptyTensor` if input has no dimensions
pub fn softmax(input: &Tensor) -> TensorResult<Tensor> {
    let _timer = timed(OpCategory::Softmax, input.numel());

    if input.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: "softmax",
            expected: "FP32",
            got: if input.precision() == Precision::BF16 { "BF16" } else { "unknown" },
        });
    }

    let shape = input.shape();
    if shape.is_empty() {
        return Err(TensorError::EmptyTensor {
            operation: "softmax",
        });
    }

    // Use custom shader (avoids CommandBatch::sync overhead)
    Ok(softmax_custom_inner(input, shape))
}

/// MPS-based softmax using MPSMatrixSoftMax.
/// 1.8-5.4x faster than custom shader across typical LLM dimensions.
/// Kept as reference implementation per PERF_OPTIMIZATION.md.
#[allow(dead_code)]
fn softmax_mps(input: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::Softmax, input.numel());
    assert_eq!(input.precision(), Precision::FP32);

    let shape = input.shape();
    assert!(!shape.is_empty(), "Input must have at least 1 dimension");

    let dim = shape[shape.len() - 1];
    let batch_seq: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);

    let output = Tensor::zeros(shape, Precision::FP32);

    // Need to sync before MPS operations since MPS uses its own command buffer
    CommandBatch::sync();

    autoreleasepool(|_| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let row_bytes = dim * std::mem::size_of::<f32>();

        let desc = unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                batch_seq,
                dim,
                row_bytes,
                MPSDataType::Float32,
            )
        };

        let matrix_in = unsafe {
            MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), input.buffer(), &desc)
        };

        let matrix_out = unsafe {
            MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), output.buffer(), &desc)
        };

        let kernel = unsafe {
            MPSMatrixSoftMax::initWithDevice(MPSMatrixSoftMax::alloc(), device)
        };

        let command_buffer = ctx
            .command_queue()
            .commandBuffer()
            .expect("Failed to create command buffer for MPS");

        unsafe {
            kernel.encodeToCommandBuffer_inputMatrix_resultMatrix(
                &command_buffer,
                &matrix_in,
                &matrix_out,
            );
        }

        command_buffer.commit();
        command_buffer.waitUntilCompleted();
    });

    output
}

/// Custom shader-based softmax (kept for reference and edge cases)
#[allow(dead_code)]
fn softmax_custom(input: &Tensor) -> Tensor {
    assert_eq!(input.precision(), Precision::FP32);

    let shape = input.shape();
    assert!(!shape.is_empty(), "Input must have at least 1 dimension");

    softmax_custom_inner(input, shape)
}

/// Inner implementation of softmax custom shader (no validation)
fn softmax_custom_inner(input: &Tensor, shape: &[usize]) -> Tensor {
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

    let use_fast = dim >= SOFTMAX_THREADS;

    let input_buf = input.buffer();
    let output_buf = output.buffer();

    if use_fast {
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

        CommandBatch::dispatch_threadgroups(
            &pipelines.softmax_fast,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
            },
            threadgroup_count,
            threadgroup_size,
        );
    } else {
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

        CommandBatch::dispatch(
            &pipelines.softmax,
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

        let output = softmax(&input).unwrap();
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

        let output = softmax(&input).unwrap();
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

        let output = softmax(&input).unwrap();
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

        let output = softmax(&input).unwrap();
        let result = output.as_f32_slice();

        // Should not have NaN or Inf
        for (i, &val) in result.iter().enumerate() {
            assert!(val.is_finite(), "Result {} is not finite: {}", i, val);
            assert!((0.0..=1.0).contains(&val), "Result {} out of range: {}", i, val);
        }

        // Verify sum is 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Sum should be 1, got {}", sum);
    }

    /// Benchmark comparing MPS softmax vs custom shader
    /// Run with: cargo test benchmark_mps_softmax --release -- --nocapture --ignored
    #[test]
    #[ignore]
    fn benchmark_mps_softmax() {
        use std::time::Instant;
        use objc2::rc::autoreleasepool;
        use objc2::AllocAnyThread;
        use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
        use objc2_metal_performance_shaders::{MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixSoftMax};

        let test_cases = [
            (256, 512),      // Small batch, typical hidden dim
            (1024, 512),     // Medium batch
            (4096, 512),     // Large batch (batch * seq for training)
            (256, 2048),     // Larger hidden dim
            (4096, 2048),    // Large batch + large dim
            (16384, 256),    // Very large batch, attention-style (batch*heads*seq, seq)
        ];

        println!("\n{}", "=".repeat(80));
        println!("Softmax Benchmark: MPS vs Custom Shader (FP32)");
        println!("{}", "=".repeat(80));
        println!("{:>10} {:>10} | {:>12} {:>12} | {:>12}",
                 "batch_seq", "dim", "Custom", "MPS", "Winner");
        println!("{}", "-".repeat(80));

        let warmup_iters = 10;
        let bench_iters = 100;

        for (batch_seq, dim) in test_cases {
            let input_data: Vec<f32> = (0..(batch_seq * dim))
                .map(|i| ((i % 100) as f32 - 50.0) * 0.1)
                .collect();

            let input = Tensor::from_f32_slice(&input_data, &[batch_seq, dim]);

            // Warmup custom shader
            for _ in 0..warmup_iters {
                let _ = softmax(&input);
            }
            CommandBatch::sync();

            // Benchmark custom shader
            let start = Instant::now();
            for _ in 0..bench_iters {
                let _ = softmax(&input).unwrap();
            }
            CommandBatch::sync();
            let custom_time = start.elapsed().as_secs_f64() / bench_iters as f64 * 1000.0;

            // MPS softmax
            let output_mps = Tensor::zeros(&[batch_seq, dim], Precision::FP32);

            // Warmup MPS
            for _ in 0..warmup_iters {
                autoreleasepool(|_| {
                    let ctx = MetalContext::global();
                    let device = ctx.device();

                    let row_bytes = dim * std::mem::size_of::<f32>();

                    let desc = unsafe {
                        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                            batch_seq, dim, row_bytes, MPSDataType::Float32,
                        )
                    };

                    let matrix_in = unsafe {
                        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), input.buffer(), &desc)
                    };

                    let matrix_out = unsafe {
                        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), output_mps.buffer(), &desc)
                    };

                    let kernel = unsafe {
                        MPSMatrixSoftMax::initWithDevice(MPSMatrixSoftMax::alloc(), device)
                    };

                    let command_buffer = ctx.command_queue()
                        .commandBuffer()
                        .expect("Failed to create command buffer");

                    unsafe {
                        kernel.encodeToCommandBuffer_inputMatrix_resultMatrix(
                            &command_buffer,
                            &matrix_in,
                            &matrix_out,
                        );
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

                    let row_bytes = dim * std::mem::size_of::<f32>();

                    let desc = unsafe {
                        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                            batch_seq, dim, row_bytes, MPSDataType::Float32,
                        )
                    };

                    let matrix_in = unsafe {
                        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), input.buffer(), &desc)
                    };

                    let matrix_out = unsafe {
                        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), output_mps.buffer(), &desc)
                    };

                    let kernel = unsafe {
                        MPSMatrixSoftMax::initWithDevice(MPSMatrixSoftMax::alloc(), device)
                    };

                    let command_buffer = ctx.command_queue()
                        .commandBuffer()
                        .expect("Failed to create command buffer");

                    unsafe {
                        kernel.encodeToCommandBuffer_inputMatrix_resultMatrix(
                            &command_buffer,
                            &matrix_in,
                            &matrix_out,
                        );
                    }

                    command_buffer.commit();
                    command_buffer.waitUntilCompleted();
                });
            }
            let mps_time = start.elapsed().as_secs_f64() / bench_iters as f64 * 1000.0;

            // Verify correctness (spot check)
            let custom_result = softmax(&input).unwrap();
            CommandBatch::sync();
            let custom_slice = custom_result.as_f32_slice();
            let mps_slice = output_mps.as_f32_slice();

            let mut max_diff = 0.0f32;
            for (c, m) in custom_slice.iter().zip(mps_slice.iter()) {
                max_diff = max_diff.max((c - m).abs());
            }

            let speedup = custom_time / mps_time;
            let winner = if speedup > 1.0 { "MPS" } else { "Custom" };
            let ratio = if speedup > 1.0 { speedup } else { 1.0 / speedup };

            println!("{:>10} {:>10} | {:>10.3}ms {:>10.3}ms | {:>8} ({:.2}x) [diff={:.2e}]",
                     batch_seq, dim, custom_time, mps_time, winner, ratio, max_diff);
        }

        println!("{}", "=".repeat(80));
    }
}
