//! BF16 (Brain Float 16) operations for mixed precision training.
//!
//! These operations read/write BF16 tensors but compute in FP32 for numerical stability.
//! This provides ~2x memory savings compared to FP32 while maintaining training quality.

mod elementwise;
mod gemm;
mod norm;
mod pipelines;
mod precision;
mod softmax;

// Re-export all public functions
pub use elementwise::{add_bf16, mul_bf16, scale_bf16, silu_bf16, swiglu_bf16};
pub use gemm::{matmul_bf16, matmul_bf16_batched};
pub use norm::rmsnorm_bf16;
pub use precision::{to_bf16_gpu, to_f32_gpu};
pub use softmax::softmax_bf16;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::{bf16_to_f32, Precision};
    use crate::tensor::Tensor;

    fn bf16_tensor_from_f32(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_f32_as_bf16(data, shape)
    }

    fn bf16_tensor_to_f32_vec(tensor: &Tensor) -> Vec<f32> {
        tensor.as_bf16_slice().iter().map(|&x| bf16_to_f32(x)).collect()
    }

    #[test]
    fn test_matmul_bf16() {
        let a = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let c = matmul_bf16(&a, &b);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.precision(), Precision::BF16);

        let result = bf16_tensor_to_f32_vec(&c);
        // [1,2,3] @ [1,2; 3,4; 5,6] = [22, 28; 49, 64]
        assert!((result[0] - 22.0).abs() < 0.5);
        assert!((result[1] - 28.0).abs() < 0.5);
        assert!((result[2] - 49.0).abs() < 0.5);
        assert!((result[3] - 64.0).abs() < 0.5);
    }

    #[test]
    fn test_add_bf16() {
        let a = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = bf16_tensor_from_f32(&[0.5, 1.0, 1.5, 2.0], &[4]);
        let c = add_bf16(&a, &b);

        let result = bf16_tensor_to_f32_vec(&c);
        assert!((result[0] - 1.5).abs() < 0.01);
        assert!((result[1] - 3.0).abs() < 0.01);
        assert!((result[2] - 4.5).abs() < 0.01);
        assert!((result[3] - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_mul_bf16() {
        let a = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = bf16_tensor_from_f32(&[2.0, 3.0, 4.0, 5.0], &[4]);
        let c = mul_bf16(&a, &b);

        let result = bf16_tensor_to_f32_vec(&c);
        assert!((result[0] - 2.0).abs() < 0.01);
        assert!((result[1] - 6.0).abs() < 0.01);
        assert!((result[2] - 12.0).abs() < 0.01);
        assert!((result[3] - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_scale_bf16() {
        let a = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let c = scale_bf16(&a, 2.5);

        let result = bf16_tensor_to_f32_vec(&c);
        assert!((result[0] - 2.5).abs() < 0.01);
        assert!((result[1] - 5.0).abs() < 0.01);
        assert!((result[2] - 7.5).abs() < 0.01);
        assert!((result[3] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_silu_bf16() {
        let a = bf16_tensor_from_f32(&[0.0, 1.0, 2.0, -1.0], &[4]);
        let c = silu_bf16(&a);

        let result = bf16_tensor_to_f32_vec(&c);
        // SiLU(0) = 0
        assert!(result[0].abs() < 0.01);
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.731
        assert!((result[1] - 0.731).abs() < 0.05);
        // SiLU(2) ≈ 1.762
        assert!((result[2] - 1.762).abs() < 0.05);
    }

    #[test]
    fn test_rmsnorm_bf16() {
        let input = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let gamma = bf16_tensor_from_f32(&[1.0, 1.0, 1.0, 1.0], &[4]);
        let output = rmsnorm_bf16(&input, &gamma, 1e-5);

        assert_eq!(output.shape(), &[2, 4]);
        assert_eq!(output.precision(), Precision::BF16);

        let result = bf16_tensor_to_f32_vec(&output);
        // Check that outputs are normalized (roughly mean of squares ≈ 1)
        let row0_sq_sum: f32 = result[0..4].iter().map(|x| x * x).sum();
        let row0_mean_sq = row0_sq_sum / 4.0;
        assert!((row0_mean_sq - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_softmax_bf16() {
        let input = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let output = softmax_bf16(&input);

        let result = bf16_tensor_to_f32_vec(&output);
        // Check that sum is approximately 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
        // Check that values are in increasing order
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
        assert!(result[2] < result[3]);
    }

    #[test]
    fn test_gpu_precision_conversion() {
        let f32_tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let bf16_tensor = to_bf16_gpu(&f32_tensor);

        assert_eq!(bf16_tensor.precision(), Precision::BF16);

        let back_to_f32 = to_f32_gpu(&bf16_tensor);
        assert_eq!(back_to_f32.precision(), Precision::FP32);

        let result = back_to_f32.as_f32_slice();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
        assert!((result[2] - 3.0).abs() < 0.01);
        assert!((result[3] - 4.0).abs() < 0.01);
    }

    /// Benchmark comparing custom BF16 GEMM shader vs MPS FP32 (with conversion)
    /// Run with: cargo test benchmark_bf16_gemm --release -- --nocapture --ignored
    #[test]
    #[ignore]
    fn benchmark_bf16_gemm_vs_mps() {
        use std::time::Instant;
        use crate::command_batch::CommandBatch;
        use crate::ops::matmul_mps_nt;

        CommandBatch::sync();

        let test_sizes = [
            (256, 512, 512),
            (512, 512, 512),
            (1024, 512, 512),
            (256, 512, 2048),
            (4096, 512, 512),
        ];

        println!("\n{}", "=".repeat(90));
        println!("BF16 GEMM Benchmark: Custom BF16 Shader vs MPS FP32 (with BF16→FP32 conversion)");
        println!("{}", "=".repeat(90));
        println!("{:>10} {:>10} {:>10} | {:>14} {:>14} | {:>10}",
                 "M", "K", "N", "BF16 Custom", "MPS+Convert", "Winner");
        println!("{}", "-".repeat(90));

        let warmup_iters = 5;
        let bench_iters = 20;

        for (m, k, n) in test_sizes {
            let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
            let b_data: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();

            // BF16 tensors for custom shader (weight shape for NT matmul: [N, K])
            let a_bf16 = Tensor::from_f32_as_bf16(&a_data, &[m, k]);
            let b_bf16 = Tensor::from_f32_as_bf16(&b_data, &[n, k]);

            // FP32 input, BF16 weight for MPS approach
            let a_fp32 = Tensor::from_f32_slice(&a_data, &[m, k]);

            // Warmup custom BF16 GEMM (need to transpose B for fair comparison)
            // Our custom matmul_bf16 does A @ B, but linear_forward does A @ B^T
            // Let's compare just the conversion + MPS overhead
            let b_bf16_transposed = Tensor::from_f32_as_bf16(
                &(0..k * n).map(|i| {
                    let row = i / n;
                    let col = i % n;
                    b_data[col * k + row]
                }).collect::<Vec<_>>(),
                &[k, n]
            );

            for _ in 0..warmup_iters {
                let _ = matmul_bf16(&a_bf16, &b_bf16_transposed);
            }
            CommandBatch::sync();

            // Benchmark custom BF16 GEMM
            let start = Instant::now();
            for _ in 0..bench_iters {
                let _ = matmul_bf16(&a_bf16, &b_bf16_transposed);
            }
            CommandBatch::sync();
            let bf16_time = start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;

            // Warmup MPS (with BF16→FP32 conversion)
            for _ in 0..warmup_iters {
                let b_fp32 = to_f32_gpu(&b_bf16);
                let _ = matmul_mps_nt(&a_fp32, &b_fp32);
            }
            CommandBatch::sync();

            // Benchmark MPS with conversion
            let start = Instant::now();
            for _ in 0..bench_iters {
                let b_fp32 = to_f32_gpu(&b_bf16);
                let _ = matmul_mps_nt(&a_fp32, &b_fp32);
            }
            CommandBatch::sync();
            let mps_time = start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;

            let winner = if bf16_time < mps_time { "BF16" } else { "MPS" };
            let ratio = if bf16_time < mps_time {
                format!("{:.1}x faster", mps_time / bf16_time)
            } else {
                format!("{:.1}x faster", bf16_time / mps_time)
            };

            println!("{:>10} {:>10} {:>10} | {:>12.3}ms {:>12.3}ms | {:>10} ({})",
                     m, k, n, bf16_time, mps_time, winner, ratio);
        }

        println!("{}", "=".repeat(90));
    }
}
