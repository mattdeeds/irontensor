//! MPS-based GEMM using Apple's Metal Performance Shaders.
//!
//! Uses MPSMatrixMultiplication which leverages the AMX (Apple Matrix coprocessor)
//! for highly optimized matrix multiplication on Apple Silicon.

use objc2::AllocAnyThread;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
use objc2_metal_performance_shaders::{MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication};

use crate::command_batch::CommandBatch;
use crate::device::MetalContext;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

/// MPS-based matrix multiplication: C = A @ B
///
/// Uses Apple's Metal Performance Shaders for optimized GEMM.
/// Falls back to custom kernel for unsupported cases.
///
/// Supports:
/// - 2D tensors: [M, K] @ [K, N] -> [M, N]
/// - 3D tensors (batched): [B, M, K] @ [B, K, N] -> [B, M, N]
pub fn matmul_mps(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.precision(), Precision::FP32, "MPS matmul currently only supports FP32");
    assert_eq!(b.precision(), Precision::FP32, "MPS matmul currently only supports FP32");

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Compute output elements for profiling
    let output_elements = match (a_shape.len(), b_shape.len()) {
        (2, 2) => a_shape[0] * b_shape[1],
        (3, 3) => a_shape[0] * a_shape[1] * b_shape[2],
        _ => 0,
    };
    let _timer = timed(OpCategory::Matmul, output_elements);

    match (a_shape.len(), b_shape.len()) {
        (2, 2) => matmul_mps_2d(a, b),
        (3, 3) => matmul_mps_batched(a, b),
        _ => panic!(
            "MPS matmul requires 2D or 3D tensors, got shapes {:?} and {:?}",
            a_shape, b_shape
        ),
    }
}

fn matmul_mps_2d(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let m = a_shape[0];
    let k = a_shape[1];
    let k2 = b_shape[0];
    let n = b_shape[1];

    assert_eq!(k, k2, "Inner dimensions must match: A[{}, {}] @ B[{}, {}]", m, k, k2, n);

    let c = Tensor::zeros(&[m, n], Precision::FP32);

    // Need to sync before MPS operations since MPS uses its own command buffer
    CommandBatch::sync();

    let ctx = MetalContext::global();
    let device = ctx.device();

    // Create matrix descriptors
    // Row-major layout: rowBytes = columns * sizeof(float)
    let row_bytes_a = k * std::mem::size_of::<f32>();
    let row_bytes_b = n * std::mem::size_of::<f32>();
    let row_bytes_c = n * std::mem::size_of::<f32>();

    let desc_a = unsafe {
        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            m,
            k,
            row_bytes_a,
            MPSDataType::Float32,
        )
    };

    let desc_b = unsafe {
        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            k,
            n,
            row_bytes_b,
            MPSDataType::Float32,
        )
    };

    let desc_c = unsafe {
        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            m,
            n,
            row_bytes_c,
            MPSDataType::Float32,
        )
    };

    // Create MPS matrices from existing buffers
    let matrix_a = unsafe {
        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), a.buffer(), &desc_a)
    };

    let matrix_b = unsafe {
        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), b.buffer(), &desc_b)
    };

    let matrix_c = unsafe {
        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), c.buffer(), &desc_c)
    };

    // Create the multiplication kernel
    // C = alpha * A @ B + beta * C
    // We want C = A @ B, so alpha = 1.0, beta = 0.0
    let kernel = unsafe {
        MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
            MPSMatrixMultiplication::alloc(),
            device,
            false,  // don't transpose A
            false,  // don't transpose B
            m,      // result rows
            n,      // result columns
            k,      // interior columns (shared dimension)
            1.0,    // alpha
            0.0,    // beta
        )
    };

    // Create command buffer and encode
    let command_buffer = ctx.command_queue()
        .commandBuffer()
        .expect("Failed to create command buffer for MPS");

    unsafe {
        kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
            &command_buffer,
            &matrix_a,
            &matrix_b,
            &matrix_c,
        );
    }

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    c
}

fn matmul_mps_batched(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let batch2 = b_shape[0];
    let k2 = b_shape[1];
    let n = b_shape[2];

    assert_eq!(batch, batch2, "Batch sizes must match: {} vs {}", batch, batch2);
    assert_eq!(k, k2, "Inner dimensions must match");

    let c = Tensor::zeros(&[batch, m, n], Precision::FP32);

    // Need to sync before MPS operations
    CommandBatch::sync();

    let ctx = MetalContext::global();
    let device = ctx.device();

    // For batched matmul, we need to set up matrix descriptors with matrixBytes
    let row_bytes_a = k * std::mem::size_of::<f32>();
    let row_bytes_b = n * std::mem::size_of::<f32>();
    let row_bytes_c = n * std::mem::size_of::<f32>();

    let matrix_bytes_a = m * row_bytes_a;
    let matrix_bytes_b = k * row_bytes_b;
    let matrix_bytes_c = m * row_bytes_c;

    let desc_a = unsafe {
        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType(
            m,
            k,
            batch,
            row_bytes_a,
            matrix_bytes_a,
            MPSDataType::Float32,
        )
    };

    let desc_b = unsafe {
        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType(
            k,
            n,
            batch,
            row_bytes_b,
            matrix_bytes_b,
            MPSDataType::Float32,
        )
    };

    let desc_c = unsafe {
        MPSMatrixDescriptor::matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType(
            m,
            n,
            batch,
            row_bytes_c,
            matrix_bytes_c,
            MPSDataType::Float32,
        )
    };

    let matrix_a = unsafe {
        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), a.buffer(), &desc_a)
    };

    let matrix_b = unsafe {
        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), b.buffer(), &desc_b)
    };

    let matrix_c = unsafe {
        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), c.buffer(), &desc_c)
    };

    // Create kernel - MPS handles batching automatically based on descriptor
    let kernel = unsafe {
        MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
            MPSMatrixMultiplication::alloc(),
            device,
            false,
            false,
            m,
            n,
            k,
            1.0,
            0.0,
        )
    };

    // Set batch parameters
    unsafe {
        kernel.setBatchStart(0);
        kernel.setBatchSize(batch);
    }

    let command_buffer = ctx.command_queue()
        .commandBuffer()
        .expect("Failed to create command buffer for MPS");

    unsafe {
        kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
            &command_buffer,
            &matrix_a,
            &matrix_b,
            &matrix_c,
        );
    }

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::matmul;

    #[test]
    fn test_mps_matmul_2x2() {
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A @ B = [[19, 22], [43, 50]]
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_f32_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul_mps(&a, &b);

        let result = c.as_f32_slice();
        assert_eq!(result.len(), 4);
        assert!((result[0] - 19.0).abs() < 1e-5);
        assert!((result[1] - 22.0).abs() < 1e-5);
        assert!((result[2] - 43.0).abs() < 1e-5);
        assert!((result[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_mps_matmul_larger() {
        let m = 64;
        let k = 32;
        let n = 48;

        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

        let a = Tensor::from_f32_slice(&a_data, &[m, k]);
        let b = Tensor::from_f32_slice(&b_data, &[k, n]);
        let c = matmul_mps(&a, &b);

        assert_eq!(c.shape(), &[m, n]);

        // Verify a few values manually
        let result = c.as_f32_slice();

        // C[0,0] = sum(A[0,:] * B[:,0])
        let mut expected_00 = 0.0f32;
        for i in 0..k {
            expected_00 += a_data[i] * b_data[i * n];
        }
        assert!((result[0] - expected_00).abs() < 1e-3, "C[0,0]: expected {}, got {}", expected_00, result[0]);
    }

    #[test]
    fn test_mps_matmul_batched() {
        let batch = 2;
        let m = 3;
        let k = 4;
        let n = 5;

        let a_data: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.1).collect();

        let a = Tensor::from_f32_slice(&a_data, &[batch, m, k]);
        let b = Tensor::from_f32_slice(&b_data, &[batch, k, n]);
        let c = matmul_mps(&a, &b);

        assert_eq!(c.shape(), &[batch, m, n]);

        // Verify first element of first batch
        let result = c.as_f32_slice();
        let mut expected = 0.0f32;
        for i in 0..k {
            expected += a_data[i] * b_data[i * n];
        }
        assert!((result[0] - expected).abs() < 1e-3);
    }

    #[test]
    fn test_mps_matches_reference() {
        // Compare MPS result with simple reference implementation
        let m = 16;
        let k = 8;
        let n = 12;

        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) - 3.0).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) - 2.0).collect();

        let a = Tensor::from_f32_slice(&a_data, &[m, k]);
        let b = Tensor::from_f32_slice(&b_data, &[k, n]);
        let c = matmul_mps(&a, &b);

        let result = c.as_f32_slice();

        // Reference implementation
        for i in 0..m {
            for j in 0..n {
                let mut expected = 0.0f32;
                for p in 0..k {
                    expected += a_data[i * k + p] * b_data[p * n + j];
                }
                let actual = result[i * n + j];
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "Mismatch at [{}, {}]: expected {}, got {}",
                    i, j, expected, actual
                );
            }
        }
    }

    /// Benchmark comparing MPS GEMM vs custom kernel
    /// Run with: cargo test benchmark_mps_vs_custom --release -- --nocapture --ignored
    #[test]
    #[ignore] // Only run when explicitly requested
    fn benchmark_mps_vs_custom() {
        use std::time::Instant;

        // Sync before benchmarking
        CommandBatch::sync();

        // Test sizes relevant to LLM training:
        // (M, K, N) - typical dimensions
        let test_sizes = [
            (256, 512, 512),    // Small
            (512, 512, 512),    // Medium
            (1024, 512, 512),   // Larger M (batch * seq)
            (256, 512, 2048),   // FFN intermediate
            (256, 2048, 512),   // FFN output
            (4096, 512, 512),   // Large batch
            (4096, 512, 2048),  // Large batch + FFN
        ];

        println!("\n{}", "=".repeat(80));
        println!("GEMM Benchmark: MPS vs Custom Kernel (FP32)");
        println!("{}", "=".repeat(80));
        println!("{:>10} {:>10} {:>10} | {:>12} {:>12} | {:>10}",
                 "M", "K", "N", "Custom(ms)", "MPS(ms)", "Speedup");
        println!("{}", "-".repeat(80));

        let warmup_iters = 5;
        let bench_iters = 20;

        for (m, k, n) in test_sizes {
            // Create random-ish test data
            let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
            let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();

            let a = Tensor::from_f32_slice(&a_data, &[m, k]);
            let b = Tensor::from_f32_slice(&b_data, &[k, n]);

            // Warmup custom kernel
            for _ in 0..warmup_iters {
                let _ = matmul(&a, &b);
            }
            CommandBatch::sync();

            // Benchmark custom kernel
            let start = Instant::now();
            for _ in 0..bench_iters {
                let _ = matmul(&a, &b);
            }
            CommandBatch::sync();
            let custom_time = start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;

            // Warmup MPS kernel
            for _ in 0..warmup_iters {
                let _ = matmul_mps(&a, &b);
            }

            // Benchmark MPS kernel
            let start = Instant::now();
            for _ in 0..bench_iters {
                let _ = matmul_mps(&a, &b);
            }
            let mps_time = start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;

            let speedup = custom_time / mps_time;

            println!("{:>10} {:>10} {:>10} | {:>12.3} {:>12.3} | {:>10.2}x",
                     m, k, n, custom_time, mps_time, speedup);
        }

        println!("{}", "=".repeat(80));
        println!();

        // Also test batched matmul
        println!("\n{}", "=".repeat(80));
        println!("Batched GEMM Benchmark: MPS vs Custom Kernel (FP32)");
        println!("{}", "=".repeat(80));
        println!("{:>8} {:>8} {:>8} {:>8} | {:>12} {:>12} | {:>10}",
                 "Batch", "M", "K", "N", "Custom(ms)", "MPS(ms)", "Speedup");
        println!("{}", "-".repeat(80));

        let batched_sizes = [
            (16, 256, 64, 64),   // Attention heads
            (16, 256, 64, 256),  // Q @ K^T
            (16, 256, 256, 64),  // Attn @ V
            (1, 4096, 512, 512), // Large single batch
        ];

        for (batch, m, k, n) in batched_sizes {
            let a_data: Vec<f32> = (0..batch * m * k).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
            let b_data: Vec<f32> = (0..batch * k * n).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();

            let a = Tensor::from_f32_slice(&a_data, &[batch, m, k]);
            let b = Tensor::from_f32_slice(&b_data, &[batch, k, n]);

            // Warmup and benchmark custom
            for _ in 0..warmup_iters {
                let _ = matmul(&a, &b);
            }
            CommandBatch::sync();

            let start = Instant::now();
            for _ in 0..bench_iters {
                let _ = matmul(&a, &b);
            }
            CommandBatch::sync();
            let custom_time = start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;

            // Warmup and benchmark MPS
            for _ in 0..warmup_iters {
                let _ = matmul_mps(&a, &b);
            }

            let start = Instant::now();
            for _ in 0..bench_iters {
                let _ = matmul_mps(&a, &b);
            }
            let mps_time = start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;

            let speedup = custom_time / mps_time;

            println!("{:>8} {:>8} {:>8} {:>8} | {:>12.3} {:>12.3} | {:>10.2}x",
                     batch, m, k, n, custom_time, mps_time, speedup);
        }

        println!("{}", "=".repeat(80));
    }
}
