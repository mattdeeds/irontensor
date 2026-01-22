use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::device::MetalContext;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const FLASH_ATTENTION_SHADER: &str = include_str!("../shaders/flash_attention.metal");

#[repr(C)]
struct FlashAttentionParams {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
    causal: u32,
}

struct FlashAttentionPipelines {
    flash_forward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    flash_small: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static FLASH_ATTENTION_PIPELINES: OnceLock<FlashAttentionPipelines> = OnceLock::new();

fn get_pipelines() -> &'static FlashAttentionPipelines {
    FLASH_ATTENTION_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(FLASH_ATTENTION_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile flash attention shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        FlashAttentionPipelines {
            flash_forward: make_pipeline("flash_attention_forward_f32"),
            flash_small: make_pipeline("flash_attention_small_f32"),
        }
    })
}

/// FlashAttention - Memory-efficient attention
///
/// Computes attention without materializing the full N×N attention matrix,
/// reducing memory usage from O(N²) to O(N).
///
/// Input shapes:
/// - Q: [batch, num_heads, seq_len, head_dim]
/// - K: [batch, num_heads, seq_len, head_dim]
/// - V: [batch, num_heads, seq_len, head_dim]
///
/// Output shape: [batch, num_heads, seq_len, head_dim]
///
/// Arguments:
/// - `causal`: If true, applies causal masking (each position can only attend to previous positions)
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Tensor {
    let _timer = timed(OpCategory::FlashAttention, q.numel());
    assert_eq!(q.precision(), Precision::FP32);
    assert_eq!(k.precision(), Precision::FP32);
    assert_eq!(v.precision(), Precision::FP32);

    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    assert_eq!(q_shape.len(), 4, "Q must be 4D [batch, heads, seq, head_dim]");
    assert_eq!(k_shape.len(), 4, "K must be 4D [batch, heads, seq, head_dim]");
    assert_eq!(v_shape.len(), 4, "V must be 4D [batch, heads, seq, head_dim]");

    assert_eq!(q_shape, k_shape, "Q and K must have same shape");
    assert_eq!(q_shape, v_shape, "Q and V must have same shape");

    let batch_size = q_shape[0];
    let num_heads = q_shape[1];
    let seq_len = q_shape[2];
    let head_dim = q_shape[3];

    assert!(head_dim <= 128, "head_dim must be <= 128");

    let output = Tensor::zeros(&[batch_size, num_heads, seq_len, head_dim], Precision::FP32);

    if seq_len == 0 {
        return output;
    }

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let scale = 1.0 / (head_dim as f32).sqrt();

    let params = FlashAttentionParams {
        batch_size: batch_size as u32,
        num_heads: num_heads as u32,
        seq_len: seq_len as u32,
        head_dim: head_dim as u32,
        scale,
        causal: if causal { 1 } else { 0 },
    };

    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<FlashAttentionParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let command_buffer = ctx
        .command_queue()
        .commandBuffer()
        .expect("Failed to create command buffer");
    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    // Use the small kernel for short sequences, tiled kernel for longer ones
    let use_small_kernel = seq_len <= 256;

    if use_small_kernel {
        encoder.setComputePipelineState(&pipelines.flash_small);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 4);
        }

        let grid_size = MTLSize {
            width: seq_len,
            height: 1,
            depth: batch_size * num_heads,
        };
        let thread_width = pipelines.flash_small.threadExecutionWidth();
        let threadgroup_size = MTLSize {
            width: thread_width.min(seq_len),
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    } else {
        encoder.setComputePipelineState(&pipelines.flash_forward);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(q.buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(k.buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(v.buffer()), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 4);
        }

        // Each threadgroup processes one query block
        let block_m = 32;
        let num_q_blocks = (seq_len + block_m - 1) / block_m;

        let grid_size = MTLSize {
            width: num_q_blocks,
            height: 1,
            depth: batch_size * num_heads,
        };
        let threadgroup_size = MTLSize {
            width: block_m,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
    }

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::attention;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, |acc, x| acc.max(x))
    }

    /// Transpose from [batch, heads, seq, dim] to [batch, seq, heads, dim]
    fn transpose_to_standard(data: &[f32], batch: usize, heads: usize, seq: usize, dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; data.len()];
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq {
                    for d in 0..dim {
                        let flash_idx = b * heads * seq * dim + h * seq * dim + s * dim + d;
                        let std_idx = b * seq * heads * dim + s * heads * dim + h * dim + d;
                        out[std_idx] = data[flash_idx];
                    }
                }
            }
        }
        out
    }

    /// Transpose from [batch, seq, heads, dim] to [batch, heads, seq, dim]
    fn transpose_from_standard(data: &[f32], batch: usize, heads: usize, seq: usize, dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; data.len()];
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq {
                    for d in 0..dim {
                        let std_idx = b * seq * heads * dim + s * heads * dim + h * dim + d;
                        let flash_idx = b * heads * seq * dim + h * seq * dim + s * dim + d;
                        out[flash_idx] = data[std_idx];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn test_flash_attention_matches_standard() {
        let batch = 2;
        let heads = 4;
        let seq_len = 32;
        let head_dim = 16;

        // Create data in flash attention format [batch, heads, seq, dim]
        let qkv_data: Vec<f32> = (0..batch * heads * seq_len * head_dim)
            .map(|i| (i as f32 * 0.1).sin() * 0.5)
            .collect();

        // Flash attention tensors (native format)
        let q_flash = Tensor::from_f32_slice(&qkv_data, &[batch, heads, seq_len, head_dim]);
        let k_flash = Tensor::from_f32_slice(&qkv_data, &[batch, heads, seq_len, head_dim]);
        let v_flash = Tensor::from_f32_slice(&qkv_data, &[batch, heads, seq_len, head_dim]);

        // Standard attention tensors (transposed format)
        let qkv_std = transpose_to_standard(&qkv_data, batch, heads, seq_len, head_dim);
        let q_std = Tensor::from_f32_slice(&qkv_std, &[batch, seq_len, heads, head_dim]);
        let k_std = Tensor::from_f32_slice(&qkv_std, &[batch, seq_len, heads, head_dim]);
        let v_std = Tensor::from_f32_slice(&qkv_std, &[batch, seq_len, heads, head_dim]);

        // Test non-causal attention
        let standard_out = attention(&q_std, &k_std, &v_std, false);
        let flash_out = flash_attention(&q_flash, &k_flash, &v_flash, false);

        // Transpose standard output to flash format for comparison
        let standard_out_transposed = transpose_from_standard(
            standard_out.as_f32_slice(), batch, heads, seq_len, head_dim
        );

        let diff = max_abs_diff(&standard_out_transposed, flash_out.as_f32_slice());
        assert!(
            diff < 1e-3,
            "Non-causal attention mismatch: max diff = {}",
            diff
        );

        // Test causal attention
        let standard_causal = attention(&q_std, &k_std, &v_std, true);
        let flash_causal = flash_attention(&q_flash, &k_flash, &v_flash, true);

        let standard_causal_transposed = transpose_from_standard(
            standard_causal.as_f32_slice(), batch, heads, seq_len, head_dim
        );

        let diff_causal = max_abs_diff(&standard_causal_transposed, flash_causal.as_f32_slice());
        assert!(
            diff_causal < 1e-3,
            "Causal attention mismatch: max diff = {}",
            diff_causal
        );
    }

    #[test]
    fn test_flash_attention_longer_sequence() {
        let batch = 1;
        let heads = 2;
        let seq_len = 128;
        let head_dim = 32;

        let qkv_data: Vec<f32> = (0..batch * heads * seq_len * head_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.3)
            .collect();

        let q_flash = Tensor::from_f32_slice(&qkv_data, &[batch, heads, seq_len, head_dim]);
        let k_flash = Tensor::from_f32_slice(&qkv_data, &[batch, heads, seq_len, head_dim]);
        let v_flash = Tensor::from_f32_slice(&qkv_data, &[batch, heads, seq_len, head_dim]);

        let qkv_std = transpose_to_standard(&qkv_data, batch, heads, seq_len, head_dim);
        let q_std = Tensor::from_f32_slice(&qkv_std, &[batch, seq_len, heads, head_dim]);
        let k_std = Tensor::from_f32_slice(&qkv_std, &[batch, seq_len, heads, head_dim]);
        let v_std = Tensor::from_f32_slice(&qkv_std, &[batch, seq_len, heads, head_dim]);

        let standard_out = attention(&q_std, &k_std, &v_std, true);
        let flash_out = flash_attention(&q_flash, &k_flash, &v_flash, true);

        let standard_out_transposed = transpose_from_standard(
            standard_out.as_f32_slice(), batch, heads, seq_len, head_dim
        );

        let diff = max_abs_diff(&standard_out_transposed, flash_out.as_f32_slice());
        assert!(
            diff < 1e-3,
            "Longer sequence attention mismatch: max diff = {}",
            diff
        );
    }

    #[test]
    fn test_flash_attention_output_shape() {
        let batch = 2;
        let heads = 8;
        let seq_len = 64;
        let head_dim = 64;

        let qkv_data: Vec<f32> = (0..batch * heads * seq_len * head_dim)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        let q = Tensor::from_f32_slice(&qkv_data, &[batch, heads, seq_len, head_dim]);
        let k = Tensor::from_f32_slice(&qkv_data, &[batch, heads, seq_len, head_dim]);
        let v = Tensor::from_f32_slice(&qkv_data, &[batch, heads, seq_len, head_dim]);

        let output = flash_attention(&q, &k, &v, true);

        assert_eq!(output.shape(), &[batch, heads, seq_len, head_dim]);
    }
}
