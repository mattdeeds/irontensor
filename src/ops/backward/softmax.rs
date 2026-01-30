use objc2_metal::{MTLComputePipelineState, MTLSize};

use crate::define_pipelines;
use crate::ops::kernel::{dispatch, dispatch_threadgroups, params_buffer, BufferBinding};
use crate::ops::params::SoftmaxParams;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const SHADER: &str = include_str!("../../shaders/backward/softmax.metal");
const SOFTMAX_THREADS: usize = 256;

define_pipelines!(Pipelines, SHADER, "backward/softmax", {
    softmax_backward => "softmax_backward_f32",
    softmax_backward_fast => "softmax_backward_fast_f32",
});

/// Softmax backward pass
/// grad_x = y * (grad_y - dot(grad_y, y))
/// where y is the softmax output
pub fn softmax_backward(grad_output: &Tensor, output: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::SoftmaxBackward, grad_output.numel());
    assert_eq!(grad_output.precision(), Precision::FP32);
    assert_eq!(output.precision(), Precision::FP32);
    assert_eq!(grad_output.shape(), output.shape());

    let shape = grad_output.shape();
    assert!(!shape.is_empty());

    let dim = shape[shape.len() - 1];
    let batch_seq: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);

    let grad_input = Tensor::zeros(shape, Precision::FP32);

    if batch_seq == 0 {
        return grad_input;
    }

    let pipelines = get_pipelines();
    let params_buf = params_buffer(&SoftmaxParams {
        batch_seq: batch_seq as u32,
        dim: dim as u32,
    });

    let use_fast = dim >= SOFTMAX_THREADS;

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

        dispatch_threadgroups(
            &pipelines.softmax_backward_fast,
            [
                BufferBinding::from(grad_output),
                BufferBinding::from(output),
                BufferBinding::from(&grad_input),
                BufferBinding::from(&params_buf),
            ],
            threadgroup_count,
            threadgroup_size,
        );
    } else {
        let thread_width = pipelines.softmax_backward.threadExecutionWidth() as usize;
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
            &pipelines.softmax_backward,
            [
                BufferBinding::from(grad_output),
                BufferBinding::from(output),
                BufferBinding::from(&grad_input),
                BufferBinding::from(&params_buf),
            ],
            grid_size,
            threadgroup_size,
        );
    }

    grad_input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::softmax;

    fn reference_softmax_backward(grad_output: &[f32], output: &[f32], dim: usize) -> Vec<f32> {
        let batch_seq = output.len() / dim;
        let mut grad_input = vec![0.0f32; output.len()];

        for b in 0..batch_seq {
            let offset = b * dim;
            let y = &output[offset..offset + dim];
            let go = &grad_output[offset..offset + dim];

            // dot(grad_y, y)
            let dot_sum: f32 = (0..dim).map(|i| go[i] * y[i]).sum();

            // grad_x = y * (grad_y - dot_sum)
            for i in 0..dim {
                grad_input[offset + i] = y[i] * (go[i] - dot_sum);
            }
        }

        grad_input
    }

    #[test]
    fn test_softmax_backward_simple() {
        let dim = 4;
        let batch = 2;

        // First compute forward softmax
        let input_data: Vec<f32> = (0..(batch * dim)).map(|i| i as f32 * 0.5 - 2.0).collect();
        let input = Tensor::from_f32_slice(&input_data, &[batch, dim]);
        let output = softmax(&input).unwrap();

        // Then backward
        let grad_out_data = vec![1.0f32; batch * dim];
        let grad_out = Tensor::from_f32_slice(&grad_out_data, &[batch, dim]);

        let grad_input = softmax_backward(&grad_out, &output);

        let expected = reference_softmax_backward(&grad_out_data, output.as_f32_slice(), dim);

        let result = grad_input.as_f32_slice();
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
    fn test_softmax_backward_large() {
        // Test fast kernel
        let dim = 512;
        let batch = 4;

        let input_data: Vec<f32> = (0..(batch * dim))
            .map(|i| ((i % 100) as f32 - 50.0) * 0.1)
            .collect();
        let input = Tensor::from_f32_slice(&input_data, &[batch, dim]);
        let output = softmax(&input).unwrap();

        let grad_out_data: Vec<f32> = (0..(batch * dim))
            .map(|i| ((i % 50) as f32 - 25.0) * 0.01)
            .collect();
        let grad_out = Tensor::from_f32_slice(&grad_out_data, &[batch, dim]);

        let grad_input = softmax_backward(&grad_out, &output);

        let expected = reference_softmax_backward(&grad_out_data, output.as_f32_slice(), dim);

        let result = grad_input.as_f32_slice();
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
}
