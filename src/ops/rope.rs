use crate::define_pipelines;
use crate::ops::kernel::{dispatch, params_buffer, threadgroup_3d, BufferBinding};
use crate::ops::params::RoPEParams;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const SHADER: &str = include_str!("../shaders/rope.metal");

define_pipelines!(Pipelines, SHADER, "rope", {
    rope => "rope_f32",
});

/// Apply RoPE (Rotary Position Embedding) to input tensor
///
/// Input shape: [batch, seq_len, num_heads, head_dim]
/// - head_dim must be even (rotations work on pairs)
/// - base: typically 10000.0
/// - position_offset: for KV cache continuation (default 0)
///
/// Returns tensor with same shape as input
pub fn rope(input: &Tensor, base: f32, position_offset: usize) -> Tensor {
    let _timer = timed(OpCategory::RoPE, input.numel());
    assert_eq!(input.precision(), Precision::FP32);

    let shape = input.shape();
    assert_eq!(
        shape.len(),
        4,
        "Input must be 4D [batch, seq_len, num_heads, head_dim]"
    );

    let batch_size = shape[0];
    let seq_len = shape[1];
    let num_heads = shape[2];
    let head_dim = shape[3];

    assert!(
        head_dim.is_multiple_of(2),
        "head_dim must be even, got {}",
        head_dim
    );

    let output = Tensor::zeros(shape, Precision::FP32);

    if batch_size == 0 || seq_len == 0 {
        return output;
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

    let (grid, threadgroup) =
        threadgroup_3d(&pipelines.rope, head_dim / 2, seq_len, batch_size * num_heads);

    dispatch(
        &pipelines.rope,
        [
            BufferBinding::from(input),
            BufferBinding::from(&output),
            BufferBinding::from(&params_buf),
        ],
        grid,
        threadgroup,
    );

    output
}

/// Convenience function with default base=10000.0 and position_offset=0
pub fn rope_default(input: &Tensor) -> Tensor {
    rope(input, 10000.0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_rope(
        input: &[f32],
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        base: f32,
        position_offset: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; input.len()];

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

                        let x0 = input[offset];
                        let x1 = input[offset + 1];

                        output[offset] = x0 * cos_angle - x1 * sin_angle;
                        output[offset + 1] = x0 * sin_angle + x1 * cos_angle;
                    }
                }
            }
        }

        output
    }

    #[test]
    fn test_rope_simple() {
        let batch = 1;
        let seq_len = 2;
        let num_heads = 1;
        let head_dim = 4;

        let input_data: Vec<f32> = (0..(batch * seq_len * num_heads * head_dim))
            .map(|i| i as f32 + 1.0)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, num_heads, head_dim]);
        let output = rope(&input, 10000.0, 0);

        let result = output.as_f32_slice();
        let expected = reference_rope(&input_data, batch, seq_len, num_heads, head_dim, 10000.0, 0);

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
    fn test_rope_position_zero() {
        // At position 0, all angles are 0, so cos=1, sin=0
        // Output should equal input
        let batch = 1;
        let seq_len = 1;
        let num_heads = 2;
        let head_dim = 8;

        let input_data: Vec<f32> = (0..(batch * seq_len * num_heads * head_dim))
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, num_heads, head_dim]);
        let output = rope(&input, 10000.0, 0);

        let result = output.as_f32_slice();

        // At position 0, cos(0)=1, sin(0)=0, so output = input
        for (i, (r, e)) in result.iter().zip(input_data.iter()).enumerate() {
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
    fn test_rope_larger() {
        let batch = 2;
        let seq_len = 16;
        let num_heads = 4;
        let head_dim = 64;

        let input_data: Vec<f32> = (0..(batch * seq_len * num_heads * head_dim))
            .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, num_heads, head_dim]);
        let output = rope(&input, 10000.0, 0);

        let result = output.as_f32_slice();
        let expected = reference_rope(&input_data, batch, seq_len, num_heads, head_dim, 10000.0, 0);

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
    fn test_rope_with_offset() {
        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 8;
        let position_offset = 10;

        let input_data: Vec<f32> = (0..(batch * seq_len * num_heads * head_dim))
            .map(|i| i as f32 * 0.1)
            .collect();

        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, num_heads, head_dim]);
        let output = rope(&input, 10000.0, position_offset);

        let result = output.as_f32_slice();
        let expected = reference_rope(
            &input_data,
            batch,
            seq_len,
            num_heads,
            head_dim,
            10000.0,
            position_offset,
        );

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
