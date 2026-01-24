//! Dropout backward operation.
//!
//! Regenerates the same mask using the cached seed from the forward pass
//! and applies it to the incoming gradients.

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
use crate::error::{TensorError, TensorResult};
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const DROPOUT_BACKWARD_SHADER: &str = include_str!("../../shaders/backward/dropout.metal");

struct DropoutBackwardPipelines {
    backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static DROPOUT_BACKWARD_PIPELINES: OnceLock<DropoutBackwardPipelines> = OnceLock::new();

fn get_pipelines() -> &'static DropoutBackwardPipelines {
    DROPOUT_BACKWARD_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(DROPOUT_BACKWARD_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile dropout backward shader: {e}"));

        let func = library
            .newFunctionWithName(ns_string!("dropout_backward_f32"))
            .expect("dropout_backward_f32 function not found");
        let backward = device
            .newComputePipelineStateWithFunction_error(&func)
            .expect("Failed to create dropout backward pipeline");

        DropoutBackwardPipelines { backward }
    })
}

/// Backward pass for dropout.
///
/// Regenerates the same mask using the seed from the forward pass and
/// applies it to the gradients.
///
/// # Arguments
/// * `grad_output` - Gradient of the loss with respect to dropout output
/// * `dropout_rate` - Probability of dropping each element (must match forward)
/// * `seed` - The seed returned by `dropout()` in the forward pass
///
/// # Returns
/// * `grad_input` - Gradient with respect to dropout input
///
/// # Notes
/// If `seed == 0`, the dropout was a passthrough (inference mode or rate=0),
/// so we return grad_output unchanged.
///
/// # Errors
/// * `TensorError::PrecisionMismatch` if grad_output is not FP32
pub fn dropout_backward(
    grad_output: &Tensor,
    dropout_rate: f32,
    seed: u64,
) -> TensorResult<Tensor> {
    let _timer = timed(OpCategory::DropoutBackward, grad_output.numel());

    // If seed is 0, dropout was a passthrough (inference or rate=0)
    if seed == 0 {
        return Ok(grad_output.clone());
    }

    // Validate precision
    if grad_output.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: "dropout_backward",
            expected: "FP32",
            got: if grad_output.precision() == Precision::BF16 {
                "BF16"
            } else {
                "unknown"
            },
        });
    }

    let count = grad_output.numel();
    let grad_input = Tensor::zeros(grad_output.shape(), Precision::FP32);
    let scale = 1.0 / (1.0 - dropout_rate);

    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;

    let ctx = MetalContext::global();
    let pipeline = &get_pipelines().backward;

    // Create parameter buffers
    let dropout_prob_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&dropout_rate as *const _ as *mut _).unwrap(),
            std::mem::size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create dropout_prob buffer");

    let scale_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&scale as *const _ as *mut _).unwrap(),
            std::mem::size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create scale buffer");

    let seed_lo_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&seed_lo as *const _ as *mut _).unwrap(),
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create seed_lo buffer");

    let seed_hi_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&seed_hi as *const _ as *mut _).unwrap(),
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create seed_hi buffer");

    let count_u32: u32 = count as u32;
    let count_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&count_u32 as *const _ as *mut _).unwrap(),
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create count buffer");

    let thread_width = pipeline.threadExecutionWidth();
    let grid_size = MTLSize {
        width: count,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: thread_width.min(count),
        height: 1,
        depth: 1,
    };

    let grad_output_buf = grad_output.buffer();
    let grad_input_buf = grad_input.buffer();

    CommandBatch::dispatch(
        pipeline,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(grad_output_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(grad_input_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&dropout_prob_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&scale_buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&seed_lo_buffer), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&seed_hi_buffer), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 6);
        },
        grid_size,
        threadgroup_size,
    );

    Ok(grad_input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_batch::CommandBatch;
    use crate::ops::dropout::dropout;
    use crate::rng::set_dropout_seed;

    #[test]
    fn test_dropout_backward_passthrough() {
        let grad = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        // seed=0 means dropout was passthrough
        let grad_input = dropout_backward(&grad, 0.5, 0).unwrap();
        assert_eq!(grad_input.as_f32_slice(), grad.as_f32_slice());
    }

    #[test]
    fn test_dropout_backward_mask_consistency() {
        set_dropout_seed(42);

        // Forward pass
        let input = Tensor::from_f32_slice(&[1.0; 1000], &[1000]);
        CommandBatch::begin();
        let (output, seed) = dropout(&input, 0.5, true).unwrap();
        CommandBatch::sync();
        let forward_result = output.as_f32_slice().to_vec();

        // Count zeros in forward pass
        let forward_zeros: Vec<usize> = forward_result
            .iter()
            .enumerate()
            .filter(|&(_, x)| *x == 0.0)
            .map(|(i, _)| i)
            .collect();

        // Backward pass with same seed
        let grad_output = Tensor::from_f32_slice(&[1.0; 1000], &[1000]);
        let grad_input = dropout_backward(&grad_output, 0.5, seed).unwrap();
        CommandBatch::sync();
        CommandBatch::end();

        let backward_result = grad_input.as_f32_slice();

        // Check that zeros are in the same positions
        let backward_zeros: Vec<usize> = backward_result
            .iter()
            .enumerate()
            .filter(|&(_, x)| *x == 0.0)
            .map(|(i, _)| i)
            .collect();

        assert_eq!(
            forward_zeros, backward_zeros,
            "Forward and backward should have zeros at the same positions"
        );
    }
}
