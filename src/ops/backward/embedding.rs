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

const BACKWARD_EMBEDDING_SHADER: &str = include_str!("../../shaders/backward/embedding.metal");

#[repr(C)]
struct EmbeddingParams {
    num_indices: u32,
    embed_dim: u32,
}

struct EmbeddingBackwardPipelines {
    embedding_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static EMBEDDING_BACKWARD_PIPELINES: OnceLock<EmbeddingBackwardPipelines> = OnceLock::new();

fn get_pipelines() -> &'static EmbeddingBackwardPipelines {
    EMBEDDING_BACKWARD_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(BACKWARD_EMBEDDING_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile backward embedding shader: {e}"));

        let func = library
            .newFunctionWithName(&objc2_foundation::NSString::from_str("embedding_backward_f32"))
            .expect("embedding_backward_f32 function not found");

        let embedding_backward = device
            .newComputePipelineStateWithFunction_error(&func)
            .expect("Failed to create embedding_backward pipeline");

        EmbeddingBackwardPipelines { embedding_backward }
    })
}

/// Embedding backward pass
/// Accumulates gradients into grad_weights at positions specified by indices
///
/// grad_output: [num_indices, embed_dim]
/// indices: [num_indices]
/// vocab_size: size of vocabulary
/// Returns: grad_weights [vocab_size, embed_dim]
pub fn embedding_backward(
    grad_output: &Tensor,
    indices: &[u32],
    vocab_size: usize,
) -> Tensor {
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

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    // Create indices buffer
    let indices_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(indices.as_ptr() as *mut _).unwrap(),
            std::mem::size_of_val(indices),
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

    let grad_output_buf = grad_output.buffer();
    let grad_weights_buf = grad_weights.buffer();

    let grid_size = MTLSize {
        width: embed_dim,
        height: num_indices,
        depth: 1,
    };
    let thread_width = pipelines.embedding_backward.threadExecutionWidth();
    let max_threads = pipelines.embedding_backward.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize {
        width: thread_width.min(embed_dim),
        height: (max_threads / thread_width).min(num_indices).max(1),
        depth: 1,
    };

    CommandBatch::dispatch(
        &pipelines.embedding_backward,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(grad_output_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&indices_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(grad_weights_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        },
        grid_size,
        threadgroup_size,
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
            1.0, 2.0, 3.0, 4.0,  // token at pos 0
            5.0, 6.0, 7.0, 8.0,  // token at pos 1
            9.0, 10.0, 11.0, 12.0,  // token at pos 2
        ];
        let grad_out = Tensor::from_f32_slice(&grad_out_data, &[3, embed_dim]);

        let indices = vec![0u32, 2, 0];  // token 0 appears twice

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
                i, expected[i], result[i]
            );
        }
    }
}
