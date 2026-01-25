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

const FUSED_LINEAR_CE_SHADER: &str = include_str!("../shaders/fused_linear_cross_entropy.metal");

#[repr(C)]
struct FusedLinearCEParams {
    batch_seq: u32,
    hidden_dim: u32,
    vocab_size: u32,
    ignore_index: f32,
    grad_scale: f32,
}

struct FusedLinearCEPipelines {
    chunked: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    weight_grad: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static FUSED_LINEAR_CE_PIPELINES: OnceLock<FusedLinearCEPipelines> = OnceLock::new();

fn get_pipelines() -> &'static FusedLinearCEPipelines {
    FUSED_LINEAR_CE_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(FUSED_LINEAR_CE_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile fused linear CE shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        FusedLinearCEPipelines {
            chunked: make_pipeline("fused_linear_cross_entropy_chunked_f32"),
            weight_grad: make_pipeline("fused_linear_cross_entropy_weight_grad_f32"),
        }
    })
}

/// FusedLinearCrossEntropy - Memory-efficient output projection + loss computation
///
/// Combines the final linear layer (vocab projection) with cross-entropy loss,
/// avoiding the need to materialize the full [batch*seq, vocab_size] logits tensor.
///
/// For a vocabulary of 50k tokens and batch*seq of 1024, this saves:
/// - Standard: 1024 * 50000 * 4 bytes = 200MB
/// - Fused: 1024 * (hidden_dim + 1) * 4 bytes = ~4MB (for hidden_dim=1024)
///
/// Arguments:
/// - `hidden`: Hidden states [batch*seq, hidden_dim]
/// - `weight`: Vocabulary projection weights [vocab_size, hidden_dim]
/// - `targets`: Target token indices [batch*seq] (-100 for ignore)
///
/// Returns:
/// - `loss`: Scalar mean cross-entropy loss
/// - `grad_hidden`: Gradient w.r.t. hidden states [batch*seq, hidden_dim]
/// - `grad_weight`: Gradient w.r.t. weight matrix [vocab_size, hidden_dim]
pub fn fused_linear_cross_entropy(
    hidden: &Tensor,
    weight: &Tensor,
    targets: &[i32],
) -> (f32, Tensor, Tensor) {
    let _timer = timed(OpCategory::FusedLinearCE, hidden.numel() + weight.numel());
    assert_eq!(hidden.precision(), Precision::FP32);
    assert_eq!(weight.precision(), Precision::FP32);

    let hidden_shape = hidden.shape();
    let weight_shape = weight.shape();

    assert_eq!(
        hidden_shape.len(),
        2,
        "hidden must be 2D [batch*seq, hidden_dim]"
    );
    assert_eq!(
        weight_shape.len(),
        2,
        "weight must be 2D [vocab_size, hidden_dim]"
    );

    let batch_seq = hidden_shape[0];
    let hidden_dim = hidden_shape[1];
    let vocab_size = weight_shape[0];

    assert_eq!(
        weight_shape[1], hidden_dim,
        "weight hidden_dim must match hidden"
    );
    assert_eq!(
        targets.len(),
        batch_seq,
        "targets length must match batch*seq"
    );
    assert!(
        hidden_dim <= 256,
        "hidden_dim must be <= 256 for this kernel"
    );

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    // Create output tensors
    let losses = Tensor::zeros(&[batch_seq], Precision::FP32);
    let grad_hidden = Tensor::zeros(&[batch_seq, hidden_dim], Precision::FP32);
    let grad_weight = Tensor::zeros(&[vocab_size, hidden_dim], Precision::FP32);

    if batch_seq == 0 {
        return (0.0, grad_hidden, grad_weight);
    }

    // Count valid tokens for gradient scaling
    let valid_count = targets.iter().filter(|&&t| t >= 0 && (t as usize) < vocab_size).count();
    let grad_scale = if valid_count > 0 { 1.0 / valid_count as f32 } else { 0.0 };

    let params = FusedLinearCEParams {
        batch_seq: batch_seq as u32,
        hidden_dim: hidden_dim as u32,
        vocab_size: vocab_size as u32,
        ignore_index: -100.0,
        grad_scale,
    };

    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<FusedLinearCEParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    // Create targets buffer
    let targets_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(targets.as_ptr() as *mut _).unwrap(),
            std::mem::size_of_val(targets),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create targets buffer");

    // Run chunked forward pass to compute losses and grad_hidden
    let hidden_buf = hidden.buffer();
    let weight_buf = weight.buffer();
    let losses_buf = losses.buffer();
    let grad_hidden_buf = grad_hidden.buffer();

    let grid_size = MTLSize {
        width: batch_seq,
        height: 1,
        depth: 1,
    };
    let thread_width = pipelines.chunked.threadExecutionWidth() as usize;
    let threadgroup_size = MTLSize {
        width: thread_width.min(batch_seq),
        height: 1,
        depth: 1,
    };

    CommandBatch::dispatch(
        &pipelines.chunked,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(hidden_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&targets_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(losses_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(grad_hidden_buf), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 5);
        },
        grid_size,
        threadgroup_size,
    );

    // Need to sync before reading losses
    CommandBatch::sync();

    // Compute mean loss from per-token losses
    let losses_data = losses.as_f32_slice();
    let mut total_loss = 0.0f32;
    let mut count = 0usize;
    for (i, &loss) in losses_data.iter().enumerate() {
        if targets[i] >= 0 && (targets[i] as usize) < vocab_size {
            total_loss += loss;
            count += 1;
        }
    }
    let mean_loss = if count > 0 {
        total_loss / count as f32
    } else {
        0.0
    };

    // Run weight gradient computation
    let grad_weight_buf = grad_weight.buffer();

    let grid_size2 = MTLSize {
        width: vocab_size,
        height: hidden_dim,
        depth: 1,
    };
    let threadgroup_size2 = MTLSize {
        width: thread_width.min(vocab_size),
        height: 1,
        depth: 1,
    };

    CommandBatch::dispatch(
        &pipelines.weight_grad,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(hidden_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&targets_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(grad_weight_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 4);
        },
        grid_size2,
        threadgroup_size2,
    );

    (mean_loss, grad_hidden, grad_weight)
}

/// Simplified version that only returns loss and grad_hidden (more memory efficient)
/// Use this when you don't need weight gradients (e.g., inference or frozen weights)
pub fn fused_linear_cross_entropy_forward_only(
    hidden: &Tensor,
    weight: &Tensor,
    targets: &[i32],
) -> (f32, Tensor) {
    let _timer = timed(OpCategory::FusedLinearCE, hidden.numel() + weight.numel());
    assert_eq!(hidden.precision(), Precision::FP32);
    assert_eq!(weight.precision(), Precision::FP32);

    let hidden_shape = hidden.shape();
    let weight_shape = weight.shape();

    assert_eq!(
        hidden_shape.len(),
        2,
        "hidden must be 2D [batch*seq, hidden_dim]"
    );
    assert_eq!(
        weight_shape.len(),
        2,
        "weight must be 2D [vocab_size, hidden_dim]"
    );

    let batch_seq = hidden_shape[0];
    let hidden_dim = hidden_shape[1];
    let vocab_size = weight_shape[0];

    assert_eq!(
        weight_shape[1], hidden_dim,
        "weight hidden_dim must match hidden"
    );
    assert_eq!(
        targets.len(),
        batch_seq,
        "targets length must match batch*seq"
    );

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let losses = Tensor::zeros(&[batch_seq], Precision::FP32);
    let grad_hidden = Tensor::zeros(&[batch_seq, hidden_dim], Precision::FP32);

    if batch_seq == 0 {
        return (0.0, grad_hidden);
    }

    // Count valid tokens for gradient scaling
    let valid_count = targets.iter().filter(|&&t| t >= 0 && (t as usize) < vocab_size).count();
    let grad_scale = if valid_count > 0 { 1.0 / valid_count as f32 } else { 0.0 };

    let params = FusedLinearCEParams {
        batch_seq: batch_seq as u32,
        hidden_dim: hidden_dim as u32,
        vocab_size: vocab_size as u32,
        ignore_index: -100.0,
        grad_scale,
    };

    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<FusedLinearCEParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let targets_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(targets.as_ptr() as *mut _).unwrap(),
            std::mem::size_of_val(targets),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create targets buffer");

    let hidden_buf = hidden.buffer();
    let weight_buf = weight.buffer();
    let losses_buf = losses.buffer();
    let grad_hidden_buf = grad_hidden.buffer();

    let grid_size = MTLSize {
        width: batch_seq,
        height: 1,
        depth: 1,
    };
    let thread_width = pipelines.chunked.threadExecutionWidth() as usize;
    let threadgroup_size = MTLSize {
        width: thread_width.min(batch_seq),
        height: 1,
        depth: 1,
    };

    CommandBatch::dispatch(
        &pipelines.chunked,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(hidden_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(weight_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&targets_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(losses_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(grad_hidden_buf), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 5);
        },
        grid_size,
        threadgroup_size,
    );

    // Need to sync before reading losses
    CommandBatch::sync();

    // Compute mean loss
    let losses_data = losses.as_f32_slice();
    let mut total_loss = 0.0f32;
    let mut count = 0usize;
    for (i, &loss) in losses_data.iter().enumerate() {
        if targets[i] >= 0 && (targets[i] as usize) < vocab_size {
            total_loss += loss;
            count += 1;
        }
    }
    let mean_loss = if count > 0 {
        total_loss / count as f32
    } else {
        0.0
    };

    (mean_loss, grad_hidden)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{cross_entropy_fused, matmul};

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, |acc, x| acc.max(x))
    }

    #[test]
    fn test_fused_linear_cross_entropy_matches_separate() {
        let batch_seq = 8;
        let hidden_dim = 32;
        let vocab_size = 64;

        // Create test data
        let hidden_data: Vec<f32> = (0..batch_seq * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.02).cos() * 0.3)
            .collect();
        let targets: Vec<i32> = (0..batch_seq).map(|i| (i * 7) as i32 % vocab_size as i32).collect();

        let hidden = Tensor::from_f32_slice(&hidden_data, &[batch_seq, hidden_dim]);
        let weight = Tensor::from_f32_slice(&weight_data, &[vocab_size, hidden_dim]);

        // Compute with fused kernel
        let (fused_loss, fused_grad_hidden, _fused_grad_weight) =
            fused_linear_cross_entropy(&hidden, &weight, &targets);

        // Compute with separate operations
        // First, compute logits = hidden @ weight.T
        let weight_t = {
            let w = weight.as_f32_slice();
            let mut t = vec![0.0f32; vocab_size * hidden_dim];
            for v in 0..vocab_size {
                for d in 0..hidden_dim {
                    t[d * vocab_size + v] = w[v * hidden_dim + d];
                }
            }
            Tensor::from_f32_slice(&t, &[hidden_dim, vocab_size])
        };
        let logits = matmul(&hidden, &weight_t).unwrap();

        // Compute cross-entropy
        let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
        let (separate_loss, _probs, grad_logits) = cross_entropy_fused(&logits, &targets_u32);

        // Compute grad_hidden = grad_logits @ weight
        let grad_hidden_separate = matmul(&grad_logits, &weight).unwrap();

        // Compare loss
        let loss_diff = (fused_loss - separate_loss).abs();
        assert!(
            loss_diff < 1e-3,
            "Loss mismatch: fused={}, separate={}, diff={}",
            fused_loss,
            separate_loss,
            loss_diff
        );

        // Compare grad_hidden
        let grad_diff = max_abs_diff(
            fused_grad_hidden.as_f32_slice(),
            grad_hidden_separate.as_f32_slice(),
        );
        assert!(
            grad_diff < 1e-3,
            "grad_hidden mismatch: max diff = {}",
            grad_diff
        );
    }

    #[test]
    fn test_fused_linear_cross_entropy_ignore_index() {
        let batch_seq = 4;
        let hidden_dim = 16;
        let vocab_size = 32;

        let hidden_data: Vec<f32> = (0..batch_seq * hidden_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let weight_data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();

        let hidden = Tensor::from_f32_slice(&hidden_data, &[batch_seq, hidden_dim]);
        let weight = Tensor::from_f32_slice(&weight_data, &[vocab_size, hidden_dim]);

        // Test with some ignored positions
        let targets = vec![5, -100, 10, -100]; // -100 is ignore_index

        let (loss, grad_hidden, _) = fused_linear_cross_entropy(&hidden, &weight, &targets);

        // Loss should be computed only from non-ignored positions
        assert!(loss > 0.0, "Loss should be positive");

        // Gradients for ignored positions should be zero
        let grad_h = grad_hidden.as_f32_slice();
        for d in 0..hidden_dim {
            assert_eq!(
                grad_h[hidden_dim + d],
                0.0,
                "Gradient for ignored position 1 should be zero"
            );
            assert_eq!(
                grad_h[3 * hidden_dim + d], 0.0,
                "Gradient for ignored position 3 should be zero"
            );
        }
    }

    #[test]
    fn test_fused_linear_cross_entropy_forward_only() {
        let batch_seq = 8;
        let hidden_dim = 32;
        let vocab_size = 64;

        let hidden_data: Vec<f32> = (0..batch_seq * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.02).cos() * 0.3)
            .collect();
        let targets: Vec<i32> = (0..batch_seq).map(|i| (i * 7) as i32 % vocab_size as i32).collect();

        let hidden = Tensor::from_f32_slice(&hidden_data, &[batch_seq, hidden_dim]);
        let weight = Tensor::from_f32_slice(&weight_data, &[vocab_size, hidden_dim]);

        // Compare full version with forward-only version
        let (full_loss, full_grad_hidden, _) =
            fused_linear_cross_entropy(&hidden, &weight, &targets);
        let (forward_loss, forward_grad_hidden) =
            fused_linear_cross_entropy_forward_only(&hidden, &weight, &targets);

        assert!(
            (full_loss - forward_loss).abs() < 1e-5,
            "Loss mismatch between full and forward-only"
        );

        let grad_diff = max_abs_diff(
            full_grad_hidden.as_f32_slice(),
            forward_grad_hidden.as_f32_slice(),
        );
        assert!(
            grad_diff < 1e-5,
            "grad_hidden mismatch between full and forward-only"
        );
    }

    #[test]
    fn test_fused_linear_cross_entropy_shapes() {
        let batch_seq = 16;
        let hidden_dim = 64;
        let vocab_size = 128;

        let hidden = Tensor::zeros(&[batch_seq, hidden_dim], Precision::FP32);
        let weight = Tensor::zeros(&[vocab_size, hidden_dim], Precision::FP32);
        let targets: Vec<i32> = (0..batch_seq).map(|i| i as i32 % vocab_size as i32).collect();

        let (_, grad_hidden, grad_weight) =
            fused_linear_cross_entropy(&hidden, &weight, &targets);

        assert_eq!(grad_hidden.shape(), &[batch_seq, hidden_dim]);
        assert_eq!(grad_weight.shape(), &[vocab_size, hidden_dim]);
    }
}
