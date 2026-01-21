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
use crate::tensor::Tensor;

const ATTENTION_SHADER: &str = include_str!("../shaders/attention.metal");

#[repr(C)]
struct AttentionParams {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
}

struct AttentionPipelines {
    causal_mask: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    transpose_qkv: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    transpose_output: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static ATTENTION_PIPELINES: OnceLock<AttentionPipelines> = OnceLock::new();

fn get_pipelines() -> &'static AttentionPipelines {
    ATTENTION_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(ATTENTION_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile attention shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        AttentionPipelines {
            causal_mask: make_pipeline("causal_mask_f32"),
            transpose_qkv: make_pipeline("transpose_qkv_f32"),
            transpose_output: make_pipeline("transpose_output_f32"),
        }
    })
}

/// Transpose tensor from [batch, seq, heads, dim] to [batch, heads, seq, dim]
fn transpose_to_attention_layout(input: &Tensor) -> Tensor {
    let shape = input.shape();
    assert_eq!(shape.len(), 4);

    let batch_size = shape[0];
    let seq_len = shape[1];
    let num_heads = shape[2];
    let head_dim = shape[3];

    let output = Tensor::zeros(&[batch_size, num_heads, seq_len, head_dim], Precision::FP32);

    if batch_size == 0 || seq_len == 0 {
        return output;
    }

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = AttentionParams {
        batch_size: batch_size as u32,
        num_heads: num_heads as u32,
        seq_len: seq_len as u32,
        head_dim: head_dim as u32,
        scale: 0.0, // Not used for transpose
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<AttentionParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.transpose_qkv);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
    }

    let grid_size = MTLSize {
        width: head_dim,
        height: seq_len,
        depth: batch_size * num_heads,
    };
    let thread_width = pipelines.transpose_qkv.threadExecutionWidth();
    let max_threads = pipelines.transpose_qkv.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize {
        width: thread_width.min(head_dim),
        height: (max_threads / thread_width).min(seq_len).max(1),
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

/// Transpose tensor from [batch, heads, seq, dim] to [batch, seq, heads, dim]
fn transpose_from_attention_layout(input: &Tensor) -> Tensor {
    let shape = input.shape();
    assert_eq!(shape.len(), 4);

    let batch_size = shape[0];
    let num_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    let output = Tensor::zeros(&[batch_size, seq_len, num_heads, head_dim], Precision::FP32);

    if batch_size == 0 || seq_len == 0 {
        return output;
    }

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = AttentionParams {
        batch_size: batch_size as u32,
        num_heads: num_heads as u32,
        seq_len: seq_len as u32,
        head_dim: head_dim as u32,
        scale: 0.0,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<AttentionParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.transpose_output);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
    }

    let grid_size = MTLSize {
        width: head_dim,
        height: seq_len,
        depth: batch_size * num_heads,
    };
    let thread_width = pipelines.transpose_output.threadExecutionWidth();
    let max_threads = pipelines.transpose_output.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize {
        width: thread_width.min(head_dim),
        height: (max_threads / thread_width).min(seq_len).max(1),
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

/// Apply causal mask to attention scores (in-place conceptually, returns new tensor)
fn apply_causal_mask(scores: &Tensor) -> Tensor {
    let shape = scores.shape();
    assert_eq!(shape.len(), 4); // [batch, heads, seq, seq]

    let batch_size = shape[0];
    let num_heads = shape[1];
    let seq_len = shape[2];
    assert_eq!(shape[3], seq_len, "Attention scores must be square in last two dims");

    // Copy scores to output (we'll modify in place)
    let output = Tensor::zeros(shape, Precision::FP32);

    // Copy data
    let src = scores.as_f32_slice();
    let dst = output.as_f32_slice();
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr() as *mut f32, src.len());
    }

    if batch_size == 0 || seq_len == 0 {
        return output;
    }

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = AttentionParams {
        batch_size: batch_size as u32,
        num_heads: num_heads as u32,
        seq_len: seq_len as u32,
        head_dim: 0, // Not used for mask
        scale: 0.0,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<AttentionParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.causal_mask);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 1);
    }

    let grid_size = MTLSize {
        width: seq_len,
        height: seq_len,
        depth: batch_size * num_heads,
    };
    let thread_width = pipelines.causal_mask.threadExecutionWidth();
    let max_threads = pipelines.causal_mask.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize {
        width: thread_width.min(seq_len),
        height: (max_threads / thread_width).min(seq_len).max(1),
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

/// Scaled Dot-Product Attention
///
/// Computes: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// Input shapes:
/// - q: [batch, seq_len, num_heads, head_dim]
/// - k: [batch, seq_len, num_heads, head_dim]  (can have different seq_len for KV cache)
/// - v: [batch, seq_len, num_heads, head_dim]
/// - causal: if true, apply causal mask (for autoregressive models)
///
/// Returns: [batch, seq_len, num_heads, head_dim]
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Tensor {
    assert_eq!(q.precision(), Precision::FP32);
    assert_eq!(k.precision(), Precision::FP32);
    assert_eq!(v.precision(), Precision::FP32);

    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    assert_eq!(q_shape.len(), 4, "Q must be 4D [batch, seq, heads, dim]");
    assert_eq!(k_shape.len(), 4, "K must be 4D [batch, seq, heads, dim]");
    assert_eq!(v_shape.len(), 4, "V must be 4D [batch, seq, heads, dim]");

    let batch_size = q_shape[0];
    let q_seq_len = q_shape[1];
    let num_heads = q_shape[2];
    let head_dim = q_shape[3];
    let kv_seq_len = k_shape[1];

    assert_eq!(k_shape[0], batch_size, "Batch size mismatch");
    assert_eq!(v_shape[0], batch_size, "Batch size mismatch");
    assert_eq!(k_shape[2], num_heads, "Num heads mismatch");
    assert_eq!(v_shape[2], num_heads, "Num heads mismatch");
    assert_eq!(k_shape[3], head_dim, "Head dim mismatch");
    assert_eq!(v_shape[3], head_dim, "Head dim mismatch");
    assert_eq!(v_shape[1], kv_seq_len, "K and V seq_len must match");

    if batch_size == 0 || q_seq_len == 0 || kv_seq_len == 0 {
        return Tensor::zeros(&[batch_size, q_seq_len, num_heads, head_dim], Precision::FP32);
    }

    // Step 1: Transpose Q, K, V to [batch, heads, seq, dim]
    let q_t = transpose_to_attention_layout(q);
    let k_t = transpose_to_attention_layout(k);
    let v_t = transpose_to_attention_layout(v);

    // Step 2: Compute Q @ K^T
    // q_t: [batch, heads, q_seq, dim]
    // k_t: [batch, heads, kv_seq, dim]
    // We need k_t transposed on last two dims: [batch, heads, dim, kv_seq]
    // Result: [batch, heads, q_seq, kv_seq]

    // For batched matmul, we reshape to [batch*heads, seq, dim]
    let q_flat = reshape_for_batched_matmul(&q_t, batch_size * num_heads, q_seq_len, head_dim);
    let k_flat = reshape_for_batched_matmul(&k_t, batch_size * num_heads, kv_seq_len, head_dim);

    // Transpose K for matmul: we need K^T
    let k_t_flat = transpose_last_two_dims(&k_flat, batch_size * num_heads, kv_seq_len, head_dim);

    // scores = Q @ K^T: [batch*heads, q_seq, kv_seq]
    let scores_flat = super::matmul(&q_flat, &k_t_flat);

    // Reshape scores to [batch, heads, q_seq, kv_seq]
    let mut scores = scores_flat;
    scores.reshape(&[batch_size, num_heads, q_seq_len, kv_seq_len]);

    // Step 3: Scale by 1/sqrt(d_k)
    let scale = 1.0 / (head_dim as f32).sqrt();
    let scores_scaled = super::scale(&scores, scale);

    // Step 4: Apply causal mask if needed
    let scores_masked = if causal {
        assert_eq!(
            q_seq_len, kv_seq_len,
            "Causal attention requires Q and K to have same seq_len"
        );
        apply_causal_mask(&scores_scaled)
    } else {
        scores_scaled
    };

    // Step 5: Softmax over last dimension (kv_seq)
    let attn_weights = softmax_4d(&scores_masked);

    // Step 6: attn_weights @ V
    // attn_weights: [batch, heads, q_seq, kv_seq]
    // v_t: [batch, heads, kv_seq, dim]
    // result: [batch, heads, q_seq, dim]

    let attn_flat = reshape_for_batched_matmul(&attn_weights, batch_size * num_heads, q_seq_len, kv_seq_len);
    let v_flat = reshape_for_batched_matmul(&v_t, batch_size * num_heads, kv_seq_len, head_dim);

    let output_flat = super::matmul(&attn_flat, &v_flat);

    // Reshape back to [batch, heads, q_seq, dim]
    let mut output_heads = output_flat;
    output_heads.reshape(&[batch_size, num_heads, q_seq_len, head_dim]);

    // Step 7: Transpose back to [batch, seq, heads, dim]
    transpose_from_attention_layout(&output_heads)
}

/// Helper to reshape tensor for batched matmul
fn reshape_for_batched_matmul(t: &Tensor, batch: usize, m: usize, n: usize) -> Tensor {
    assert_eq!(t.numel(), batch * m * n);
    let data = t.as_f32_slice();
    Tensor::from_f32_slice(data, &[batch, m, n])
}

/// Transpose last two dimensions: [batch, m, n] -> [batch, n, m]
fn transpose_last_two_dims(t: &Tensor, batch: usize, m: usize, n: usize) -> Tensor {
    let input = t.as_f32_slice();
    let mut output = vec![0.0f32; batch * n * m];

    for b in 0..batch {
        for i in 0..m {
            for j in 0..n {
                output[b * n * m + j * m + i] = input[b * m * n + i * n + j];
            }
        }
    }

    Tensor::from_f32_slice(&output, &[batch, n, m])
}

/// Softmax for 4D tensor, applied over the last dimension
fn softmax_4d(t: &Tensor) -> Tensor {
    let shape = t.shape();
    assert_eq!(shape.len(), 4);

    let total_rows: usize = shape[..3].iter().product();
    let dim = shape[3];

    // Reshape to 2D [total_rows, dim]
    let data = t.as_f32_slice();
    let flat = Tensor::from_f32_slice(data, &[total_rows, dim]);

    // Apply softmax
    let result = super::softmax(&flat);

    // Reshape back
    let result_data = result.as_f32_slice();
    Tensor::from_f32_slice(result_data, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        causal: bool,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; batch * seq_len * num_heads * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        for b in 0..batch {
            for h in 0..num_heads {
                // Compute attention scores for this batch/head
                let mut scores = vec![0.0f32; seq_len * seq_len];

                for q_pos in 0..seq_len {
                    for k_pos in 0..seq_len {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            let q_idx = b * seq_len * num_heads * head_dim
                                + q_pos * num_heads * head_dim
                                + h * head_dim
                                + d;
                            let k_idx = b * seq_len * num_heads * head_dim
                                + k_pos * num_heads * head_dim
                                + h * head_dim
                                + d;
                            dot += q[q_idx] * k[k_idx];
                        }
                        scores[q_pos * seq_len + k_pos] = dot * scale;
                    }
                }

                // Apply causal mask
                if causal {
                    for q_pos in 0..seq_len {
                        for k_pos in (q_pos + 1)..seq_len {
                            scores[q_pos * seq_len + k_pos] = f32::NEG_INFINITY;
                        }
                    }
                }

                // Softmax per row
                for q_pos in 0..seq_len {
                    let row = &mut scores[q_pos * seq_len..(q_pos + 1) * seq_len];
                    let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
                    for val in row.iter_mut() {
                        *val = (*val - max_val).exp() / exp_sum;
                    }
                }

                // Apply to values
                for q_pos in 0..seq_len {
                    for d in 0..head_dim {
                        let mut sum = 0.0f32;
                        for k_pos in 0..seq_len {
                            let v_idx = b * seq_len * num_heads * head_dim
                                + k_pos * num_heads * head_dim
                                + h * head_dim
                                + d;
                            sum += scores[q_pos * seq_len + k_pos] * v[v_idx];
                        }
                        let out_idx = b * seq_len * num_heads * head_dim
                            + q_pos * num_heads * head_dim
                            + h * head_dim
                            + d;
                        output[out_idx] = sum;
                    }
                }
            }
        }

        output
    }

    #[test]
    fn test_attention_simple() {
        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 8;

        let numel = batch * seq_len * num_heads * head_dim;
        let q_data: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.1) - 3.0).collect();
        let k_data: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.1) - 2.0).collect();
        let v_data: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.1) - 1.0).collect();

        let q = Tensor::from_f32_slice(&q_data, &[batch, seq_len, num_heads, head_dim]);
        let k = Tensor::from_f32_slice(&k_data, &[batch, seq_len, num_heads, head_dim]);
        let v = Tensor::from_f32_slice(&v_data, &[batch, seq_len, num_heads, head_dim]);

        let output = attention(&q, &k, &v, false);
        let result = output.as_f32_slice();

        let expected = reference_attention(&q_data, &k_data, &v_data, batch, seq_len, num_heads, head_dim, false);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-3,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }

    #[test]
    fn test_attention_causal() {
        let batch = 1;
        let seq_len = 4;
        let num_heads = 1;
        let head_dim = 4;

        let numel = batch * seq_len * num_heads * head_dim;
        let q_data: Vec<f32> = (0..numel).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let k_data: Vec<f32> = (0..numel).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let v_data: Vec<f32> = (0..numel).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let q = Tensor::from_f32_slice(&q_data, &[batch, seq_len, num_heads, head_dim]);
        let k = Tensor::from_f32_slice(&k_data, &[batch, seq_len, num_heads, head_dim]);
        let v = Tensor::from_f32_slice(&v_data, &[batch, seq_len, num_heads, head_dim]);

        let output = attention(&q, &k, &v, true);
        let result = output.as_f32_slice();

        let expected = reference_attention(&q_data, &k_data, &v_data, batch, seq_len, num_heads, head_dim, true);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-3,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }

    #[test]
    fn test_attention_larger() {
        let batch = 2;
        let seq_len = 16;
        let num_heads = 4;
        let head_dim = 32;

        let numel = batch * seq_len * num_heads * head_dim;
        let q_data: Vec<f32> = (0..numel).map(|i| ((i % 100) as f32 - 50.0) * 0.01).collect();
        let k_data: Vec<f32> = (0..numel).map(|i| ((i % 100) as f32 - 50.0) * 0.01).collect();
        let v_data: Vec<f32> = (0..numel).map(|i| ((i % 100) as f32 - 50.0) * 0.01).collect();

        let q = Tensor::from_f32_slice(&q_data, &[batch, seq_len, num_heads, head_dim]);
        let k = Tensor::from_f32_slice(&k_data, &[batch, seq_len, num_heads, head_dim]);
        let v = Tensor::from_f32_slice(&v_data, &[batch, seq_len, num_heads, head_dim]);

        let output = attention(&q, &k, &v, true);
        let result = output.as_f32_slice();

        let expected = reference_attention(&q_data, &k_data, &v_data, batch, seq_len, num_heads, head_dim, true);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-2,
                "Mismatch at {}: expected {}, got {}",
                i, e, r
            );
        }
    }
}
