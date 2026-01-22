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

const FUSED_SHADER: &str = include_str!("../shaders/fused_rmsnorm_linear.metal");
const FUSED_THREADS: usize = 256;

#[repr(C)]
struct FusedRMSNormLinearParams {
    batch_seq: u32,
    hidden_dim: u32,
    out_features: u32,
    eps: f32,
}

struct FusedPipelines {
    basic: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    tiled: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static FUSED_PIPELINES: OnceLock<FusedPipelines> = OnceLock::new();

fn get_pipelines() -> &'static FusedPipelines {
    FUSED_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(FUSED_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile fused RMSNorm+Linear shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        FusedPipelines {
            basic: make_pipeline("fused_rmsnorm_linear_f32"),
            tiled: make_pipeline("fused_rmsnorm_linear_tiled_f32"),
        }
    })
}

/// Fused RMSNorm + Linear projection
///
/// Computes: output = rmsnorm(input, gamma, eps) @ weight^T
///
/// This fusion eliminates the intermediate normalized tensor, reducing memory bandwidth.
///
/// Input shapes:
/// - input: [batch_seq, hidden_dim] - 2D tensor (already flattened batch*seq)
/// - gamma: [hidden_dim] - RMSNorm scale parameter
/// - weight: [out_features, hidden_dim] - Linear weight (transposed during matmul)
///
/// Returns: [batch_seq, out_features]
pub fn fused_rmsnorm_linear(
    input: &Tensor,
    gamma: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Tensor {
    assert_eq!(input.precision(), Precision::FP32, "Input must be FP32");

    // Convert BF16 parameters to FP32 if needed
    let gamma = if gamma.precision() == Precision::BF16 {
        crate::ops::to_f32_gpu(gamma)
    } else {
        gamma.clone()
    };
    let gamma = &gamma;

    let weight = if weight.precision() == Precision::BF16 {
        crate::ops::to_f32_gpu(weight)
    } else {
        weight.clone()
    };
    let weight = &weight;

    let input_shape = input.shape();
    let weight_shape = weight.shape();

    assert_eq!(input_shape.len(), 2, "Input must be 2D [batch_seq, hidden_dim]");
    assert_eq!(weight_shape.len(), 2, "Weight must be 2D [out_features, hidden_dim]");

    let batch_seq = input_shape[0];
    let hidden_dim = input_shape[1];
    let out_features = weight_shape[0];
    let weight_hidden = weight_shape[1];

    assert_eq!(
        hidden_dim, weight_hidden,
        "Hidden dimensions must match: input has {}, weight has {}",
        hidden_dim, weight_hidden
    );
    assert_eq!(
        gamma.shape(),
        &[hidden_dim],
        "Gamma must have shape [hidden_dim]"
    );

    // Profile as FusedRMSNormLinear
    let _timer = timed(OpCategory::FusedRMSNormLinear, batch_seq * out_features);

    let output = Tensor::zeros(&[batch_seq, out_features], Precision::FP32);

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = FusedRMSNormLinearParams {
        batch_seq: batch_seq as u32,
        hidden_dim: hidden_dim as u32,
        out_features: out_features as u32,
        eps,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<FusedRMSNormLinearParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let input_buf = input.buffer();
    let gamma_buf = gamma.buffer();
    let weight_buf = weight.buffer();
    let output_buf = output.buffer();

    // Use basic kernel for small hidden_dim (can cache in shared memory)
    // Use tiled kernel for larger dimensions
    let use_tiled = hidden_dim > 1024 || out_features > FUSED_THREADS;

    if use_tiled {
        // Tiled: each threadgroup handles one row and a tile of outputs
        let num_out_tiles = (out_features + FUSED_THREADS - 1) / FUSED_THREADS;

        let threadgroup_count = MTLSize {
            width: num_out_tiles,
            height: batch_seq,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: FUSED_THREADS,
            height: 1,
            depth: 1,
        };

        CommandBatch::dispatch_threadgroups(
            &pipelines.tiled,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(gamma_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(weight_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 4);
            },
            threadgroup_count,
            threadgroup_size,
        );
    } else {
        // Basic: each threadgroup handles one row, all outputs
        let threadgroup_count = MTLSize {
            width: batch_seq,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: FUSED_THREADS,
            height: 1,
            depth: 1,
        };

        CommandBatch::dispatch_threadgroups(
            &pipelines.basic,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(gamma_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(weight_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 4);
            },
            threadgroup_count,
            threadgroup_size,
        );
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{rmsnorm, matmul_mps_nt};

    fn reference_fused(input: &[f32], gamma: &[f32], weight: &[f32], eps: f32, batch_seq: usize, hidden_dim: usize, out_features: usize) -> Vec<f32> {
        // Step 1: RMSNorm
        let mut normed = vec![0.0f32; batch_seq * hidden_dim];
        for b in 0..batch_seq {
            let offset = b * hidden_dim;
            let row = &input[offset..offset + hidden_dim];

            let sum_sq: f32 = row.iter().map(|x| x * x).sum();
            let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;

            for i in 0..hidden_dim {
                normed[offset + i] = row[i] * inv_rms * gamma[i];
            }
        }

        // Step 2: Linear (normed @ weight^T)
        let mut output = vec![0.0f32; batch_seq * out_features];
        for b in 0..batch_seq {
            for o in 0..out_features {
                let mut sum = 0.0f32;
                for k in 0..hidden_dim {
                    sum += normed[b * hidden_dim + k] * weight[o * hidden_dim + k];
                }
                output[b * out_features + o] = sum;
            }
        }

        output
    }

    #[test]
    fn test_fused_rmsnorm_linear_basic() {
        let batch_seq = 4;
        let hidden_dim = 64;
        let out_features = 32;
        let eps = 1e-5;

        let input_data: Vec<f32> = (0..(batch_seq * hidden_dim))
            .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
            .collect();
        let gamma_data: Vec<f32> = (0..hidden_dim)
            .map(|i| 1.0 + (i as f32).sin() * 0.1)
            .collect();
        let weight_data: Vec<f32> = (0..(out_features * hidden_dim))
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch_seq, hidden_dim]);
        let gamma = Tensor::from_f32_slice(&gamma_data, &[hidden_dim]);
        let weight = Tensor::from_f32_slice(&weight_data, &[out_features, hidden_dim]);

        let output = fused_rmsnorm_linear(&input, &gamma, &weight, eps);
        let result = output.as_f32_slice();

        let expected = reference_fused(&input_data, &gamma_data, &weight_data, eps, batch_seq, hidden_dim, out_features);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-3,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }

    #[test]
    fn test_fused_matches_separate_ops() {
        let batch_seq = 8;
        let hidden_dim = 128;
        let out_features = 64;
        let eps = 1e-5;

        let input_data: Vec<f32> = (0..(batch_seq * hidden_dim))
            .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
            .collect();
        let gamma_data: Vec<f32> = (0..hidden_dim)
            .map(|i| 1.0 + (i as f32).sin() * 0.1)
            .collect();
        let weight_data: Vec<f32> = (0..(out_features * hidden_dim))
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch_seq, hidden_dim]);
        let gamma = Tensor::from_f32_slice(&gamma_data, &[hidden_dim]);
        let weight = Tensor::from_f32_slice(&weight_data, &[out_features, hidden_dim]);

        // Fused version
        let fused_output = fused_rmsnorm_linear(&input, &gamma, &weight, eps);

        // Separate ops version
        let normed = rmsnorm(&input, &gamma, eps);
        let separate_output = matmul_mps_nt(&normed, &weight);

        let fused_result = fused_output.as_f32_slice();
        let separate_result = separate_output.as_f32_slice();

        for (i, (f, s)) in fused_result.iter().zip(separate_result.iter()).enumerate() {
            assert!(
                (f - s).abs() < 1e-3,
                "Mismatch at {}: fused={}, separate={}",
                i, f, s
            );
        }
    }

    #[test]
    fn test_fused_large_hidden_dim() {
        // Test tiled kernel path (hidden_dim > 1024)
        let batch_seq = 2;
        let hidden_dim = 512;  // Use smaller for faster test, tiled triggers at out_features > FUSED_THREADS
        let out_features = 512;  // > FUSED_THREADS triggers tiled
        let eps = 1e-5;

        let input_data: Vec<f32> = (0..(batch_seq * hidden_dim))
            .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
            .collect();
        let gamma_data: Vec<f32> = (0..hidden_dim)
            .map(|i| 1.0 + (i as f32).sin() * 0.1)
            .collect();
        let weight_data: Vec<f32> = (0..(out_features * hidden_dim))
            .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch_seq, hidden_dim]);
        let gamma = Tensor::from_f32_slice(&gamma_data, &[hidden_dim]);
        let weight = Tensor::from_f32_slice(&weight_data, &[out_features, hidden_dim]);

        let output = fused_rmsnorm_linear(&input, &gamma, &weight, eps);
        let result = output.as_f32_slice();

        let expected = reference_fused(&input_data, &gamma_data, &weight_data, eps, batch_seq, hidden_dim, out_features);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-2,  // Slightly looser tolerance for larger dimensions
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }

    /// Benchmark fused vs separate ops
    /// Run with: cargo test benchmark_fused_rmsnorm_linear --release -- --nocapture --ignored
    #[test]
    #[ignore]
    fn benchmark_fused_rmsnorm_linear() {
        use std::time::Instant;

        let test_cases = [
            (256, 512, 512),    // Small
            (256, 512, 2048),   // Q/K/V projection size (hidden -> 4*hidden)
            (256, 2048, 512),   // Output projection (4*hidden -> hidden)
            (1024, 512, 512),   // Larger batch
            (4096, 512, 2048),  // Typical attention layer
        ];

        println!("\n{:>10} {:>10} {:>10} | {:>12} {:>12} | {:>12}",
                 "batch_seq", "hidden", "out_feat", "Separate", "Fused", "Speedup");
        println!("{}", "-".repeat(80));

        for (batch_seq, hidden_dim, out_features) in test_cases {
            let eps = 1e-5;
            let iterations = 100;

            let input_data: Vec<f32> = (0..(batch_seq * hidden_dim))
                .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
                .collect();
            let gamma_data: Vec<f32> = (0..hidden_dim)
                .map(|_| 1.0)
                .collect();
            let weight_data: Vec<f32> = (0..(out_features * hidden_dim))
                .map(|i| ((i % 200) as f32 - 100.0) * 0.001)
                .collect();

            let input = Tensor::from_f32_slice(&input_data, &[batch_seq, hidden_dim]);
            let gamma = Tensor::from_f32_slice(&gamma_data, &[hidden_dim]);
            let weight = Tensor::from_f32_slice(&weight_data, &[out_features, hidden_dim]);

            // Warmup
            for _ in 0..10 {
                let _ = fused_rmsnorm_linear(&input, &gamma, &weight, eps);
                let normed = rmsnorm(&input, &gamma, eps);
                let _ = matmul_mps_nt(&normed, &weight);
            }

            // Benchmark separate
            CommandBatch::sync();
            let start = Instant::now();
            for _ in 0..iterations {
                let normed = rmsnorm(&input, &gamma, eps);
                let _ = matmul_mps_nt(&normed, &weight);
            }
            CommandBatch::sync();
            let separate_time = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            // Benchmark fused
            CommandBatch::sync();
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = fused_rmsnorm_linear(&input, &gamma, &weight, eps);
            }
            CommandBatch::sync();
            let fused_time = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;

            let speedup = separate_time / fused_time;
            let winner = if speedup > 1.0 { "Fused" } else { "Separate" };

            println!("{:>10} {:>10} {:>10} | {:>10.3}ms {:>10.3}ms | {:>10} ({:.2}x)",
                     batch_seq, hidden_dim, out_features,
                     separate_time, fused_time,
                     winner, if speedup > 1.0 { speedup } else { 1.0 / speedup });
        }
    }
}
