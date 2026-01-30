use crate::define_pipelines;
use crate::ops::kernel::{dispatch, params_buffer, threadgroup_3d, BufferBinding};
use crate::ops::params::RoPEParams;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const SHADER: &str = include_str!("../../shaders/backward/rope.metal");

define_pipelines!(Pipelines, SHADER, "backward/rope", {
    rope_backward => "rope_backward_f32",
});

/// RoPE backward pass
/// The backward of rotation is the inverse rotation (transpose)
pub fn rope_backward(grad_output: &Tensor, base: f32, position_offset: usize) -> Tensor {
    let _timer = timed(OpCategory::RoPEBackward, grad_output.numel());
    assert_eq!(grad_output.precision(), Precision::FP32);

    let shape = grad_output.shape();
    assert_eq!(shape.len(), 4, "Input must be 4D [batch, seq, heads, dim]");

    let batch_size = shape[0];
    let seq_len = shape[1];
    let num_heads = shape[2];
    let head_dim = shape[3];

    assert!(head_dim.is_multiple_of(2));

    let grad_input = Tensor::zeros(shape, Precision::FP32);

    if batch_size == 0 || seq_len == 0 {
        return grad_input;
    }

    let pipelines = get_pipelines();
    let params_buf = params_buffer(&RoPEParams {
        batch_size: batch_size as u32,
        seq_len: seq_len as u32,
        num_heads: num_heads as u32,
        head_dim: head_dim as u32,
        base,
        position_offset: position_offset as u32,
    });

    let (grid, threadgroup) = threadgroup_3d(
        &pipelines.rope_backward,
        head_dim / 2,
        seq_len,
        batch_size * num_heads,
    );

    dispatch(
        &pipelines.rope_backward,
        [
            BufferBinding::from(grad_output),
            BufferBinding::from(&grad_input),
            BufferBinding::from(&params_buf),
        ],
        grid,
        threadgroup,
    );

    grad_input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::rope;

    fn reference_rope_backward(
        grad_output: &[f32],
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        base: f32,
        position_offset: usize,
    ) -> Vec<f32> {
        let mut grad_input = vec![0.0f32; grad_output.len()];

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

                        let go0 = grad_output[offset];
                        let go1 = grad_output[offset + 1];

                        // Inverse rotation (transpose)
                        grad_input[offset] = go0 * cos_angle + go1 * sin_angle;
                        grad_input[offset + 1] = -go0 * sin_angle + go1 * cos_angle;
                    }
                }
            }
        }

        grad_input
    }

    #[test]
    fn test_rope_backward_simple() {
        let batch = 1;
        let seq_len = 2;
        let num_heads = 1;
        let head_dim = 4;

        let grad_out_data: Vec<f32> = (0..(batch * seq_len * num_heads * head_dim))
            .map(|i| i as f32 + 1.0)
            .collect();

        let grad_out =
            Tensor::from_f32_slice(&grad_out_data, &[batch, seq_len, num_heads, head_dim]);
        let grad_input = rope_backward(&grad_out, 10000.0, 0);

        let result = grad_input.as_f32_slice();
        let expected = reference_rope_backward(
            &grad_out_data,
            batch,
            seq_len,
            num_heads,
            head_dim,
            10000.0,
            0,
        );

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
    fn test_rope_backward_roundtrip() {
        // Apply forward then backward should give identity (rotation is orthogonal)
        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 8;

        let input_data: Vec<f32> = (0..(batch * seq_len * num_heads * head_dim))
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, num_heads, head_dim]);

        // Forward
        let output = rope(&input, 10000.0, 0);

        // Backward with identity gradient
        let grad_out = output.as_f32_slice().to_vec();
        let grad_out_tensor =
            Tensor::from_f32_slice(&grad_out, &[batch, seq_len, num_heads, head_dim]);
        let recovered = rope_backward(&grad_out_tensor, 10000.0, 0);

        // Should recover original input
        let result = recovered.as_f32_slice();
        for (i, (r, e)) in result.iter().zip(input_data.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "Roundtrip mismatch at {}: expected {}, got {}",
                i,
                e,
                r
            );
        }
    }
}
