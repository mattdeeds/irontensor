use crate::define_pipelines;
use crate::ops::kernel::{dispatch, params_buffer, slice_buffer, threadgroup_2d, BufferBinding};
use crate::ops::params::EmbeddingParams;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const SHADER: &str = include_str!("../../shaders/backward/embedding.metal");

define_pipelines!(Pipelines, SHADER, "backward/embedding", {
    embedding_backward => "embedding_backward_f32",
});

/// Embedding backward pass
/// Accumulates gradients into grad_weights at positions specified by indices
///
/// grad_output: [num_indices, embed_dim]
/// indices: [num_indices]
/// vocab_size: size of vocabulary
/// Returns: grad_weights [vocab_size, embed_dim]
pub fn embedding_backward(grad_output: &Tensor, indices: &[u32], vocab_size: usize) -> Tensor {
    let _timer = timed(OpCategory::EmbeddingBackward, grad_output.numel());
    assert_eq!(grad_output.precision(), Precision::FP32);

    let grad_shape = grad_output.shape();
    assert_eq!(grad_shape.len(), 2);

    let num_indices = grad_shape[0];
    let embed_dim = grad_shape[1];

    assert_eq!(indices.len(), num_indices);

    // Initialize grad_weights to zero
    let grad_weights = Tensor::zeros(&[vocab_size, embed_dim], Precision::FP32);

    if num_indices == 0 {
        return grad_weights;
    }

    let pipelines = get_pipelines();
    let indices_buf = slice_buffer(indices);
    let params_buf = params_buffer(&EmbeddingParams {
        num_indices: num_indices as u32,
        embed_dim: embed_dim as u32,
    });

    let (grid, threadgroup) = threadgroup_2d(&pipelines.embedding_backward, embed_dim, num_indices);

    dispatch(
        &pipelines.embedding_backward,
        [
            BufferBinding::from(grad_output),
            BufferBinding::from(&indices_buf),
            BufferBinding::from(&grad_weights),
            BufferBinding::from(&params_buf),
        ],
        grid,
        threadgroup,
    );

    grad_weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_backward_simple() {
        let vocab_size = 5;
        let embed_dim = 4;

        // grad_output for 3 tokens
        let grad_out_data = vec![
            1.0, 2.0, 3.0, 4.0, // token at pos 0
            5.0, 6.0, 7.0, 8.0, // token at pos 1
            9.0, 10.0, 11.0, 12.0, // token at pos 2
        ];
        let grad_out = Tensor::from_f32_slice(&grad_out_data, &[3, embed_dim]);

        let indices = vec![0u32, 2, 0]; // token 0 appears twice

        let grad_weights = embedding_backward(&grad_out, &indices, vocab_size);

        let result = grad_weights.as_f32_slice();

        // Token 0 should have gradient from positions 0 and 2 (accumulated)
        // grad_weights[0] = grad_out[0] + grad_out[2] = [1,2,3,4] + [9,10,11,12] = [10,12,14,16]
        assert_eq!(&result[0..4], &[10.0, 12.0, 14.0, 16.0]);

        // Token 1 should have zero gradient (not used)
        assert_eq!(&result[4..8], &[0.0, 0.0, 0.0, 0.0]);

        // Token 2 should have gradient from position 1
        assert_eq!(&result[8..12], &[5.0, 6.0, 7.0, 8.0]);

        // Tokens 3, 4 should have zero gradient
        assert_eq!(&result[12..16], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&result[16..20], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_embedding_backward_larger() {
        let vocab_size = 100;
        let embed_dim = 32;
        let seq_len = 16;

        let grad_out_data: Vec<f32> = (0..(seq_len * embed_dim))
            .map(|i| i as f32 * 0.1)
            .collect();
        let grad_out = Tensor::from_f32_slice(&grad_out_data, &[seq_len, embed_dim]);

        let indices: Vec<u32> = (0..seq_len).map(|i| (i * 7 % vocab_size) as u32).collect();

        let grad_weights = embedding_backward(&grad_out, &indices, vocab_size);

        assert_eq!(grad_weights.shape(), &[vocab_size, embed_dim]);

        // Verify a few entries against CPU reference
        let result = grad_weights.as_f32_slice();
        let mut expected = vec![0.0f32; vocab_size * embed_dim];
        for (pos, &token) in indices.iter().enumerate() {
            for d in 0..embed_dim {
                expected[token as usize * embed_dim + d] += grad_out_data[pos * embed_dim + d];
            }
        }

        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < 1e-4,
                "Mismatch at {}: expected {}, got {}",
                i,
                expected[i],
                result[i]
            );
        }
    }
}
