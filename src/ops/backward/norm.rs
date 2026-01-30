use objc2_metal::{MTLComputePipelineState, MTLSize};

use crate::command_batch::CommandBatch;
use crate::define_pipelines;
use crate::ops::kernel::{dispatch, params_buffer, slice_buffer, BufferBinding};
use crate::ops::params::RMSNormParams;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const SHADER: &str = include_str!("../../shaders/backward/norm.metal");

define_pipelines!(Pipelines, SHADER, "backward/norm", {
    rmsnorm_backward => "rmsnorm_backward_f32",
    zero_buffer => "zero_buffer_f32",
});

/// RMSNorm backward pass
/// Returns (grad_input, grad_gamma)
pub fn rmsnorm_backward(
    grad_output: &Tensor,
    input: &Tensor,
    gamma: &Tensor,
    eps: f32,
) -> (Tensor, Tensor) {
    let _timer = timed(OpCategory::RmsNormBackward, grad_output.numel());
    assert_eq!(grad_output.precision(), Precision::FP32);
    assert_eq!(input.precision(), Precision::FP32);
    assert_eq!(gamma.precision(), Precision::FP32);
    assert_eq!(grad_output.shape(), input.shape());

    let shape = input.shape();
    let hidden_dim = shape[shape.len() - 1];
    assert_eq!(gamma.shape(), &[hidden_dim]);

    let batch_seq: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);

    let grad_input = Tensor::zeros(shape, Precision::FP32);
    let grad_gamma = Tensor::zeros(&[hidden_dim], Precision::FP32);

    if batch_seq == 0 {
        return (grad_input, grad_gamma);
    }

    let pipelines = get_pipelines();

    // First, zero the grad_gamma buffer (it will be accumulated with atomics)
    let count_buf = slice_buffer(&[hidden_dim as u32]);

    let thread_width = pipelines.zero_buffer.threadExecutionWidth() as usize;
    let grid_size = MTLSize {
        width: hidden_dim,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: thread_width.min(hidden_dim),
        height: 1,
        depth: 1,
    };

    dispatch(
        &pipelines.zero_buffer,
        [
            BufferBinding::from(&grad_gamma),
            BufferBinding::from(&count_buf),
        ],
        grid_size,
        threadgroup_size,
    );

    // Sync before backward pass (depends on zeroed grad_gamma)
    CommandBatch::sync();

    // Now compute the backward pass
    let params_buf = params_buffer(&RMSNormParams {
        batch_seq: batch_seq as u32,
        hidden_dim: hidden_dim as u32,
        eps,
    });

    // One thread per row
    let thread_width = pipelines.rmsnorm_backward.threadExecutionWidth() as usize;
    let grid_size = MTLSize {
        width: batch_seq,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: thread_width.min(batch_seq),
        height: 1,
        depth: 1,
    };

    dispatch(
        &pipelines.rmsnorm_backward,
        [
            BufferBinding::from(grad_output),
            BufferBinding::from(input),
            BufferBinding::from(gamma),
            BufferBinding::from(&grad_input),
            BufferBinding::from(&grad_gamma),
            BufferBinding::from(&params_buf),
        ],
        grid_size,
        threadgroup_size,
    );

    (grad_input, grad_gamma)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_rmsnorm_backward(
        grad_output: &[f32],
        input: &[f32],
        gamma: &[f32],
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let hidden_dim = gamma.len();
        let batch_seq = input.len() / hidden_dim;
        let n = hidden_dim as f32;

        let mut grad_input = vec![0.0f32; input.len()];
        let mut grad_gamma = vec![0.0f32; hidden_dim];

        for b in 0..batch_seq {
            let offset = b * hidden_dim;
            let x = &input[offset..offset + hidden_dim];
            let go = &grad_output[offset..offset + hidden_dim];

            // Compute rms
            let sum_sq: f32 = x.iter().map(|xi| xi * xi).sum();
            let rms = (sum_sq / n + eps).sqrt();
            let s = 1.0 / rms;
            let s3 = s * s * s;

            // Compute dot_sum = sum_j(go_j * gamma_j * x_j)
            let dot_sum: f32 = (0..hidden_dim).map(|i| go[i] * gamma[i] * x[i]).sum();

            for i in 0..hidden_dim {
                // grad_x_i = go_i * gamma_i * s - (1/n) * s^3 * x_i * dot_sum
                grad_input[offset + i] = go[i] * gamma[i] * s - (1.0 / n) * s3 * x[i] * dot_sum;

                // grad_gamma_i += go_i * x_i * s
                grad_gamma[i] += go[i] * x[i] * s;
            }
        }

        (grad_input, grad_gamma)
    }

    #[test]
    fn test_rmsnorm_backward_simple() {
        let hidden_dim = 4;
        let batch = 2;
        let eps = 1e-5;

        let input_data: Vec<f32> = (0..(batch * hidden_dim))
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let gamma_data = vec![1.0f32; hidden_dim];
        let grad_out_data = vec![1.0f32; batch * hidden_dim];

        let input = Tensor::from_f32_slice(&input_data, &[batch, hidden_dim]);
        let gamma = Tensor::from_f32_slice(&gamma_data, &[hidden_dim]);
        let grad_out = Tensor::from_f32_slice(&grad_out_data, &[batch, hidden_dim]);

        let (grad_input, grad_gamma) = rmsnorm_backward(&grad_out, &input, &gamma, eps);

        let (expected_grad_input, expected_grad_gamma) =
            reference_rmsnorm_backward(&grad_out_data, &input_data, &gamma_data, eps);

        let gi_result = grad_input.as_f32_slice();
        let gg_result = grad_gamma.as_f32_slice();

        for (i, (r, e)) in gi_result.iter().zip(expected_grad_input.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "grad_input mismatch at {}: expected {}, got {}",
                i,
                e,
                r
            );
        }

        for (i, (r, e)) in gg_result.iter().zip(expected_grad_gamma.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "grad_gamma mismatch at {}: expected {}, got {}",
                i,
                e,
                r
            );
        }
    }
}
