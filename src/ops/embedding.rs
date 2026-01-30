use crate::define_pipelines;
use crate::ops::kernel::{dispatch, params_buffer, slice_buffer, threadgroup_2d, BufferBinding};
use crate::ops::params::EmbeddingParams;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const SHADER: &str = include_str!("../shaders/embedding.metal");

define_pipelines!(Pipelines, SHADER, "embedding", {
    embedding => "embedding_f32",
});

/// Embedding lookup: output[i] = weights[indices[i]]
///
/// Input shapes:
/// - weights: [vocab_size, embed_dim] (FP32 or BF16 - BF16 is converted to FP32)
/// - indices: [num_indices] - each value is a token ID in [0, vocab_size)
///
/// Returns: [num_indices, embed_dim] (always FP32)
pub fn embedding(weights: &Tensor, indices: &[u32]) -> Tensor {
    let _timer = timed(OpCategory::Embedding, indices.len() * weights.shape()[1]);

    // Convert BF16 weights to FP32 if needed (mixed precision support)
    let weights = if weights.precision() == Precision::BF16 {
        crate::ops::to_f32_gpu(weights)
    } else {
        weights.clone()
    };

    let weights_shape = weights.shape();
    assert_eq!(
        weights_shape.len(),
        2,
        "Weights must be 2D [vocab_size, embed_dim]"
    );

    let vocab_size = weights_shape[0];
    let embed_dim = weights_shape[1];

    // Validate indices
    for (i, &idx) in indices.iter().enumerate() {
        assert!(
            (idx as usize) < vocab_size,
            "Index {} at position {} is out of bounds for vocab_size {}",
            idx,
            i,
            vocab_size
        );
    }

    let num_indices = indices.len();
    let output = Tensor::zeros(&[num_indices, embed_dim], Precision::FP32);

    if num_indices == 0 {
        return output;
    }

    let pipelines = get_pipelines();
    let indices_buf = slice_buffer(indices);
    let params_buf = params_buffer(&EmbeddingParams {
        num_indices: num_indices as u32,
        embed_dim: embed_dim as u32,
    });

    let (grid, threadgroup) = threadgroup_2d(&pipelines.embedding, embed_dim, num_indices);

    dispatch(
        &pipelines.embedding,
        [
            BufferBinding::from(&weights),
            BufferBinding::from(&indices_buf),
            BufferBinding::from(&output),
            BufferBinding::from(&params_buf),
        ],
        grid,
        threadgroup,
    );

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_simple() {
        // weights: 3 tokens, 4 dimensions each
        let weights_data = vec![
            1.0, 2.0, 3.0, 4.0, // token 0
            5.0, 6.0, 7.0, 8.0, // token 1
            9.0, 10.0, 11.0, 12.0, // token 2
        ];
        let weights = Tensor::from_f32_slice(&weights_data, &[3, 4]);

        let indices = vec![0, 2, 1];
        let output = embedding(&weights, &indices);

        assert_eq!(output.shape(), &[3, 4]);

        let result = output.as_f32_slice();
        // token 0
        assert_eq!(&result[0..4], &[1.0, 2.0, 3.0, 4.0]);
        // token 2
        assert_eq!(&result[4..8], &[9.0, 10.0, 11.0, 12.0]);
        // token 1
        assert_eq!(&result[8..12], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_embedding_repeated_indices() {
        let weights_data = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
        ];
        let weights = Tensor::from_f32_slice(&weights_data, &[2, 2]);

        let indices = vec![0, 0, 1, 0];
        let output = embedding(&weights, &indices);

        assert_eq!(output.shape(), &[4, 2]);

        let result = output.as_f32_slice();
        assert_eq!(&result[0..2], &[1.0, 2.0]);
        assert_eq!(&result[2..4], &[1.0, 2.0]);
        assert_eq!(&result[4..6], &[3.0, 4.0]);
        assert_eq!(&result[6..8], &[1.0, 2.0]);
    }

    #[test]
    fn test_embedding_larger() {
        let vocab_size = 1000;
        let embed_dim = 256;
        let seq_len = 128;

        // Create weights with predictable pattern
        let weights_data: Vec<f32> = (0..(vocab_size * embed_dim))
            .map(|i| i as f32 * 0.001)
            .collect();
        let weights = Tensor::from_f32_slice(&weights_data, &[vocab_size, embed_dim]);

        // Create indices
        let indices: Vec<u32> = (0..seq_len).map(|i| (i * 7 % vocab_size) as u32).collect();
        let output = embedding(&weights, &indices);

        assert_eq!(output.shape(), &[seq_len, embed_dim]);

        // Verify results
        let result = output.as_f32_slice();
        for (seq_pos, &token_id) in indices.iter().enumerate() {
            for dim in 0..embed_dim {
                let expected = (token_id as usize * embed_dim + dim) as f32 * 0.001;
                let actual = result[seq_pos * embed_dim + dim];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "Mismatch at seq_pos={}, dim={}: expected {}, got {}",
                    seq_pos,
                    dim,
                    expected,
                    actual
                );
            }
        }
    }

    #[test]
    fn test_embedding_empty() {
        let weights_data = vec![1.0, 2.0, 3.0, 4.0];
        let weights = Tensor::from_f32_slice(&weights_data, &[2, 2]);

        let indices: Vec<u32> = vec![];
        let output = embedding(&weights, &indices);

        assert_eq!(output.shape(), &[0, 2]);
    }
}
