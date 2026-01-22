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

const EMBEDDING_SHADER: &str = include_str!("../shaders/embedding.metal");

#[repr(C)]
struct EmbeddingParams {
    num_indices: u32,
    embed_dim: u32,
}

struct EmbeddingPipelines {
    embedding: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static EMBEDDING_PIPELINES: OnceLock<EmbeddingPipelines> = OnceLock::new();

fn get_pipelines() -> &'static EmbeddingPipelines {
    EMBEDDING_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(EMBEDDING_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile embedding shader: {e}"));

        let func = library
            .newFunctionWithName(&objc2_foundation::NSString::from_str("embedding_f32"))
            .expect("embedding_f32 function not found");

        let embedding = device
            .newComputePipelineStateWithFunction_error(&func)
            .expect("Failed to create embedding pipeline");

        EmbeddingPipelines { embedding }
    })
}

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
    let weights = &weights;

    let weights_shape = weights.shape();
    assert_eq!(weights_shape.len(), 2, "Weights must be 2D [vocab_size, embed_dim]");

    let vocab_size = weights_shape[0];
    let embed_dim = weights_shape[1];

    // Validate indices
    for (i, &idx) in indices.iter().enumerate() {
        assert!(
            (idx as usize) < vocab_size,
            "Index {} at position {} is out of bounds for vocab_size {}",
            idx, i, vocab_size
        );
    }

    let num_indices = indices.len();
    let output = Tensor::zeros(&[num_indices, embed_dim], Precision::FP32);

    if num_indices == 0 {
        return output;
    }

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    // Create buffer for indices
    let indices_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(indices.as_ptr() as *mut _).unwrap(),
            indices.len() * std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create indices buffer");

    let params = EmbeddingParams {
        num_indices: num_indices as u32,
        embed_dim: embed_dim as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<EmbeddingParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let weights_buf = weights.buffer();
    let output_buf = output.buffer();

    let grid_size = MTLSize {
        width: embed_dim,
        height: num_indices,
        depth: 1,
    };
    let thread_width = pipelines.embedding.threadExecutionWidth();
    let max_threads = pipelines.embedding.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize {
        width: thread_width.min(embed_dim),
        height: (max_threads / thread_width).min(num_indices),
        depth: 1,
    };

    CommandBatch::dispatch(
        &pipelines.embedding,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(weights_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&indices_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        },
        grid_size,
        threadgroup_size,
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
            1.0, 2.0, 3.0, 4.0,     // token 0
            5.0, 6.0, 7.0, 8.0,     // token 1
            9.0, 10.0, 11.0, 12.0,  // token 2
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
            1.0, 2.0,  // token 0
            3.0, 4.0,  // token 1
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
                    seq_pos, dim, expected, actual
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
