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

const REPEAT_KV_SHADER: &str = include_str!("../shaders/repeat_kv.metal");

#[repr(C)]
struct RepeatKVParams {
    batch: u32,
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    repeats: u32,
}

struct RepeatKVPipelines {
    repeat_kv: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    repeat_kv_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static REPEAT_KV_PIPELINES: OnceLock<RepeatKVPipelines> = OnceLock::new();

fn get_pipelines() -> &'static RepeatKVPipelines {
    REPEAT_KV_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(REPEAT_KV_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile repeat_kv shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        RepeatKVPipelines {
            repeat_kv: make_pipeline("repeat_kv_f32"),
            repeat_kv_backward: make_pipeline("repeat_kv_backward_f32"),
        }
    })
}

/// Repeat KV heads for GQA (Grouped Query Attention) - GPU accelerated
///
/// Input:  [batch, seq_len, num_kv_heads, head_dim]
/// Output: [batch, seq_len, num_heads, head_dim]
///
/// Each KV head is repeated `num_heads / num_kv_heads` times.
pub fn repeat_kv_gpu(
    x: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Tensor {
    let _timer = timed(OpCategory::RepeatKV, batch * seq_len * num_heads * head_dim);
    assert_eq!(x.precision(), Precision::FP32);

    let repeats = num_heads / num_kv_heads;
    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "num_heads ({}) must be divisible by num_kv_heads ({})",
        num_heads,
        num_kv_heads
    );

    // If no repetition needed, return a view
    if repeats == 1 {
        return x.view(&[batch, seq_len, num_heads, head_dim]);
    }

    let output = Tensor::zeros(&[batch, seq_len, num_heads, head_dim], Precision::FP32);
    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = RepeatKVParams {
        batch: batch as u32,
        seq_len: seq_len as u32,
        num_heads: num_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        head_dim: head_dim as u32,
        repeats: repeats as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<RepeatKVParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let input_buf = x.buffer();
    let output_buf = output.buffer();

    let thread_width = pipelines.repeat_kv.threadExecutionWidth();
    let batch_seq = batch * seq_len;

    // Grid: (head_dim, num_heads, batch*seq_len)
    let grid_size = MTLSize {
        width: head_dim,
        height: num_heads,
        depth: batch_seq,
    };
    let tg_width = thread_width.min(head_dim);
    let tg_height = (256 / tg_width).min(num_heads).max(1);
    let threadgroup_size = MTLSize {
        width: tg_width,
        height: tg_height,
        depth: 1,
    };

    CommandBatch::dispatch(
        &pipelines.repeat_kv,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
        },
        grid_size,
        threadgroup_size,
    );

    output
}

/// Backward for repeat_kv - sums gradients from expanded heads back to KV heads (GPU accelerated)
///
/// Input (grad_expanded):  [batch, seq_len, num_heads, head_dim]
/// Output (grad_kv):       [batch, seq_len, num_kv_heads, head_dim]
pub fn repeat_kv_backward_gpu(
    grad_expanded: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Tensor {
    let _timer = timed(
        OpCategory::RepeatKVBackward,
        batch * seq_len * num_kv_heads * head_dim,
    );
    assert_eq!(grad_expanded.precision(), Precision::FP32);

    let repeats = num_heads / num_kv_heads;
    assert!(
        num_heads.is_multiple_of(num_kv_heads),
        "num_heads ({}) must be divisible by num_kv_heads ({})",
        num_heads,
        num_kv_heads
    );

    // If no repetition, gradient passes through unchanged
    if repeats == 1 {
        return grad_expanded.view(&[batch, seq_len, num_kv_heads, head_dim]);
    }

    let output = Tensor::zeros(&[batch, seq_len, num_kv_heads, head_dim], Precision::FP32);
    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = RepeatKVParams {
        batch: batch as u32,
        seq_len: seq_len as u32,
        num_heads: num_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        head_dim: head_dim as u32,
        repeats: repeats as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<RepeatKVParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let input_buf = grad_expanded.buffer();
    let output_buf = output.buffer();

    let thread_width = pipelines.repeat_kv_backward.threadExecutionWidth();
    let batch_seq = batch * seq_len;

    // Grid: (head_dim, num_kv_heads, batch*seq_len)
    let grid_size = MTLSize {
        width: head_dim,
        height: num_kv_heads,
        depth: batch_seq,
    };
    let tg_width = thread_width.min(head_dim);
    let tg_height = (256 / tg_width).min(num_kv_heads).max(1);
    let threadgroup_size = MTLSize {
        width: tg_width,
        height: tg_height,
        depth: 1,
    };

    CommandBatch::dispatch(
        &pipelines.repeat_kv_backward,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
        },
        grid_size,
        threadgroup_size,
    );

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeat_kv_cpu(
        x: &Tensor,
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let repeats = num_heads / num_kv_heads;
        let data = x.as_f32_slice();
        let mut expanded = vec![0.0f32; batch * seq_len * num_heads * head_dim];

        for b in 0..batch {
            for s in 0..seq_len {
                for kv_h in 0..num_kv_heads {
                    for r in 0..repeats {
                        let h = kv_h * repeats + r;
                        for d in 0..head_dim {
                            let src = b * seq_len * num_kv_heads * head_dim
                                + s * num_kv_heads * head_dim
                                + kv_h * head_dim
                                + d;
                            let dst = b * seq_len * num_heads * head_dim
                                + s * num_heads * head_dim
                                + h * head_dim
                                + d;
                            expanded[dst] = data[src];
                        }
                    }
                }
            }
        }
        expanded
    }

    fn repeat_kv_backward_cpu(
        grad_expanded: &Tensor,
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let repeats = num_heads / num_kv_heads;
        let data = grad_expanded.as_f32_slice();
        let mut grad_kv = vec![0.0f32; batch * seq_len * num_kv_heads * head_dim];

        for b in 0..batch {
            for s in 0..seq_len {
                for kv_h in 0..num_kv_heads {
                    for d in 0..head_dim {
                        let dst = b * seq_len * num_kv_heads * head_dim
                            + s * num_kv_heads * head_dim
                            + kv_h * head_dim
                            + d;
                        for r in 0..repeats {
                            let h = kv_h * repeats + r;
                            let src = b * seq_len * num_heads * head_dim
                                + s * num_heads * head_dim
                                + h * head_dim
                                + d;
                            grad_kv[dst] += data[src];
                        }
                    }
                }
            }
        }
        grad_kv
    }

    #[test]
    fn test_repeat_kv_simple() {
        let batch = 2;
        let seq_len = 4;
        let num_heads = 8;
        let num_kv_heads = 2;
        let head_dim = 4;

        let input_size = batch * seq_len * num_kv_heads * head_dim;
        let input_data: Vec<f32> = (0..input_size).map(|i| i as f32).collect();
        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, num_kv_heads, head_dim]);

        let output = repeat_kv_gpu(&input, batch, seq_len, num_heads, num_kv_heads, head_dim);
        crate::command_batch::CommandBatch::sync();

        let result = output.as_f32_slice();
        let expected = repeat_kv_cpu(&input, batch, seq_len, num_heads, num_kv_heads, head_dim);

        assert_eq!(output.shape(), &[batch, seq_len, num_heads, head_dim]);
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Forward mismatch at {}: expected {}, got {}",
                i,
                e,
                r
            );
        }
    }

    #[test]
    fn test_repeat_kv_backward_simple() {
        let batch = 2;
        let seq_len = 4;
        let num_heads = 8;
        let num_kv_heads = 2;
        let head_dim = 4;

        let grad_size = batch * seq_len * num_heads * head_dim;
        let grad_data: Vec<f32> = (0..grad_size).map(|i| i as f32 * 0.1).collect();
        let grad_expanded =
            Tensor::from_f32_slice(&grad_data, &[batch, seq_len, num_heads, head_dim]);

        let output = repeat_kv_backward_gpu(
            &grad_expanded,
            batch,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        );
        crate::command_batch::CommandBatch::sync();

        let result = output.as_f32_slice();
        let expected = repeat_kv_backward_cpu(
            &grad_expanded,
            batch,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        assert_eq!(output.shape(), &[batch, seq_len, num_kv_heads, head_dim]);
        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "Backward mismatch at {}: expected {}, got {}",
                i,
                e,
                r
            );
        }
    }

    #[test]
    fn test_repeat_kv_no_repeat() {
        // When num_heads == num_kv_heads, should return a view
        let batch = 2;
        let seq_len = 4;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = 8;

        let input_size = batch * seq_len * num_kv_heads * head_dim;
        let input_data: Vec<f32> = (0..input_size).map(|i| i as f32).collect();
        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, num_kv_heads, head_dim]);

        let output = repeat_kv_gpu(&input, batch, seq_len, num_heads, num_kv_heads, head_dim);
        crate::command_batch::CommandBatch::sync();

        let result = output.as_f32_slice();
        assert_eq!(output.shape(), &[batch, seq_len, num_heads, head_dim]);
        for (i, (r, e)) in result.iter().zip(input_data.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "No-repeat mismatch at {}: expected {}, got {}",
                i,
                e,
                r
            );
        }
    }

    #[test]
    fn test_repeat_kv_roundtrip() {
        // Test that forward -> backward produces expected gradient scale
        let batch = 1;
        let seq_len = 2;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 2;
        let repeats = num_heads / num_kv_heads;

        let input_size = batch * seq_len * num_kv_heads * head_dim;
        let input_data: Vec<f32> = (0..input_size).map(|i| i as f32 + 1.0).collect();
        let input = Tensor::from_f32_slice(&input_data, &[batch, seq_len, num_kv_heads, head_dim]);

        // Forward
        let _expanded = repeat_kv_gpu(&input, batch, seq_len, num_heads, num_kv_heads, head_dim);
        crate::command_batch::CommandBatch::sync();

        // Gradient of ones for output
        let grad_size = batch * seq_len * num_heads * head_dim;
        let grad_ones: Vec<f32> = vec![1.0; grad_size];
        let grad_expanded =
            Tensor::from_f32_slice(&grad_ones, &[batch, seq_len, num_heads, head_dim]);

        // Backward
        let grad_input = repeat_kv_backward_gpu(
            &grad_expanded,
            batch,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        );
        crate::command_batch::CommandBatch::sync();

        // Each KV head gradient should be the sum of `repeats` ones = repeats
        let result = grad_input.as_f32_slice();
        for v in result.iter() {
            assert!(
                (*v - repeats as f32).abs() < 1e-5,
                "Expected gradient to be {}, got {}",
                repeats,
                v
            );
        }
    }
}
