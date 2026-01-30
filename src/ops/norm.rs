use objc2_metal::{MTLComputePipelineState, MTLSize};

use crate::define_pipelines;
use crate::ops::kernel::{dispatch, dispatch_threadgroups, params_buffer, BufferBinding};
use crate::ops::params::RMSNormParams;
use crate::error::{TensorError, TensorResult};
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const SHADER: &str = include_str!("../shaders/norm.metal");
const RMSNORM_THREADS: usize = 256;

define_pipelines!(Pipelines, SHADER, "norm", {
    rmsnorm => "rmsnorm_f32",
    rmsnorm_fast => "rmsnorm_fast_f32",
});

/// RMSNorm: output = (x / sqrt(mean(x^2) + eps)) * gamma
///
/// Input shapes:
/// - input: [..., hidden_dim] - last dimension is normalized
/// - gamma: [hidden_dim] - learnable scale parameter (FP32 or BF16 - BF16 is converted)
///
/// Returns tensor with same shape as input (always FP32)
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if input is not FP32
/// - `TensorError::EmptyTensor` if input has no dimensions
/// - `TensorError::ShapeMismatch` if gamma shape doesn't match hidden_dim
pub fn rmsnorm(input: &Tensor, gamma: &Tensor, eps: f32) -> TensorResult<Tensor> {
    let _timer = timed(OpCategory::RmsNorm, input.numel());

    if input.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: "rmsnorm",
            expected: "FP32",
            got: if input.precision() == Precision::BF16 {
                "BF16"
            } else {
                "unknown"
            },
        });
    }

    // Convert BF16 gamma to FP32 if needed (mixed precision support)
    let gamma = if gamma.precision() == Precision::BF16 {
        crate::ops::to_f32_gpu(gamma)
    } else {
        gamma.clone()
    };
    let gamma = &gamma;

    let shape = input.shape();
    if shape.is_empty() {
        return Err(TensorError::EmptyTensor {
            operation: "rmsnorm",
        });
    }

    let hidden_dim = shape[shape.len() - 1];
    if gamma.shape() != [hidden_dim] {
        return Err(TensorError::ShapeMismatch {
            operation: "rmsnorm",
            expected: format!("[{}]", hidden_dim),
            got: format!("{:?}", gamma.shape()),
        });
    }

    // Compute batch_seq (product of all dims except last)
    let batch_seq: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);

    let output = Tensor::zeros(shape, Precision::FP32);

    let pipelines = get_pipelines();
    let params_buf = params_buffer(&RMSNormParams {
        batch_seq: batch_seq as u32,
        hidden_dim: hidden_dim as u32,
        eps,
    });

    // Use fast kernel for larger hidden dimensions
    let use_fast = hidden_dim >= RMSNORM_THREADS;

    if use_fast {
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

        dispatch_threadgroups(
            &pipelines.rmsnorm_fast,
            [
                BufferBinding::from(input),
                BufferBinding::from(gamma),
                BufferBinding::from(&output),
                BufferBinding::from(&params_buf),
            ],
            threadgroup_count,
            threadgroup_size,
        );
    } else {
        let max_threads = pipelines.rmsnorm.threadExecutionWidth() as usize;
        let grid_size = MTLSize {
            width: batch_seq,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: max_threads.min(batch_seq),
            height: 1,
            depth: 1,
        };

        dispatch(
            &pipelines.rmsnorm,
            [
                BufferBinding::from(input),
                BufferBinding::from(gamma),
                BufferBinding::from(&output),
                BufferBinding::from(&params_buf),
            ],
            grid_size,
            threadgroup_size,
        );
    }

    Ok(output)
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

        let output = rmsnorm(&input, &gamma, eps).unwrap();
        let result = output.as_f32_slice();

        let expected = reference_rmsnorm(&input_data, &gamma_data, eps);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at {}: expected {}, got {}",
                i,
                e,
                r
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

        let output = rmsnorm(&input, &gamma, eps).unwrap();
        let result = output.as_f32_slice();

        let expected = reference_rmsnorm(&input_data, &gamma_data, eps);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "Mismatch at {}: expected {}, got {}",
                i,
                e,
                r
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
        let gamma_data: Vec<f32> = (0..hidden_dim)
            .map(|i| 1.0 + (i as f32).sin() * 0.1)
            .collect();
        let eps = 1e-5;

        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, hidden_dim]);
        let gamma = Tensor::from_f32_slice(&gamma_data, &[hidden_dim]);

        let output = rmsnorm(&input, &gamma, eps).unwrap();
        let result = output.as_f32_slice();

        let expected = reference_rmsnorm(&input_data, &gamma_data, eps);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-3,
                "Mismatch at {}: expected {}, got {}",
                i,
                e,
                r
            );
        }
    }
}
