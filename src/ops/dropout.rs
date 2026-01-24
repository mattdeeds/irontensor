//! Dropout operation with GPU-accelerated Philox RNG.
//!
//! Dropout randomly zeroes elements during training with probability `p`,
//! and scales remaining elements by `1/(1-p)` to maintain expected value.
//! Uses Philox counter-based RNG for deterministic mask regeneration in backward pass.

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
use crate::rng::next_dropout_seed;
use crate::tensor::Tensor;

const DROPOUT_SHADER: &str = include_str!("../shaders/dropout.metal");

struct DropoutPipelines {
    forward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static DROPOUT_PIPELINES: OnceLock<DropoutPipelines> = OnceLock::new();

fn get_pipelines() -> &'static DropoutPipelines {
    DROPOUT_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(DROPOUT_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile dropout shader: {e}"));

        let func = library
            .newFunctionWithName(ns_string!("dropout_forward_f32"))
            .expect("dropout_forward_f32 function not found");
        let forward = device
            .newComputePipelineStateWithFunction_error(&func)
            .expect("Failed to create dropout forward pipeline");

        DropoutPipelines { forward }
    })
}

/// Apply dropout during training.
///
/// # Arguments
/// * `input` - Input tensor
/// * `dropout_rate` - Probability of dropping each element (0.0 to 1.0)
/// * `training` - If false, returns input unchanged (inference mode)
///
/// # Returns
/// * `(output, seed)` - Output tensor and the seed used for mask generation.
///   The seed should be cached for use in `dropout_backward`.
///
/// # Errors
/// * `TensorError::PrecisionMismatch` if input is not FP32
/// * `TensorError::InvalidValue` if dropout_rate is not in [0, 1]
pub fn dropout(input: &Tensor, dropout_rate: f32, training: bool) -> TensorResult<(Tensor, u64)> {
    let _timer = timed(OpCategory::Dropout, input.numel());

    // Inference mode or zero dropout: return input unchanged
    if !training || dropout_rate == 0.0 {
        return Ok((input.clone(), 0));
    }

    // Validate dropout rate
    if !(0.0..1.0).contains(&dropout_rate) {
        return Err(TensorError::InvalidValue {
            operation: "dropout",
            message: format!("dropout_rate must be in [0, 1), got {}", dropout_rate),
        });
    }

    // Validate precision
    if input.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: "dropout",
            expected: "FP32",
            got: if input.precision() == Precision::BF16 {
                "BF16"
            } else {
                "unknown"
            },
        });
    }

    let count = input.numel();
    let output = Tensor::zeros(input.shape(), Precision::FP32);
    let scale = 1.0 / (1.0 - dropout_rate);

    // Get unique seed for this dropout call
    let seed = next_dropout_seed();
    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;

    let ctx = MetalContext::global();
    let pipeline = &get_pipelines().forward;

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

    let input_buf = input.buffer();
    let output_buf = output.buffer();

    CommandBatch::dispatch(
        pipeline,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&dropout_prob_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&scale_buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&seed_lo_buffer), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&seed_hi_buffer), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 6);
        },
        grid_size,
        threadgroup_size,
    );

    Ok((output, seed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_batch::CommandBatch;
    use crate::rng::set_dropout_seed;

    #[test]
    fn test_dropout_shape_preserved() {
        let input = Tensor::from_f32_slice(&[1.0; 100], &[10, 10]);
        CommandBatch::begin();
        let (output, seed) = dropout(&input, 0.5, true).unwrap();
        CommandBatch::sync();
        CommandBatch::end();

        assert_eq!(output.shape(), input.shape());
        assert_ne!(seed, 0);
    }

    #[test]
    fn test_dropout_inference_passthrough() {
        let input = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let (output, seed) = dropout(&input, 0.5, false).unwrap();

        assert_eq!(seed, 0);
        assert_eq!(output.as_f32_slice(), input.as_f32_slice());
    }

    #[test]
    fn test_dropout_zero_rate_passthrough() {
        let input = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let (output, seed) = dropout(&input, 0.0, true).unwrap();

        assert_eq!(seed, 0);
        assert_eq!(output.as_f32_slice(), input.as_f32_slice());
    }

    #[test]
    fn test_dropout_zeros_some_elements() {
        set_dropout_seed(42);
        let input = Tensor::from_f32_slice(&[1.0; 1000], &[1000]);
        CommandBatch::begin();
        let (output, _) = dropout(&input, 0.5, true).unwrap();
        CommandBatch::sync();
        CommandBatch::end();

        let result = output.as_f32_slice();
        let num_zeros = result.iter().filter(|&&x| x == 0.0).count();
        let num_nonzero = result.iter().filter(|&&x| x != 0.0).count();

        // With p=0.5, expect roughly half zeros (with some variance)
        // Allow for statistical variance: 35% to 65%
        assert!(
            num_zeros >= 350 && num_zeros <= 650,
            "Expected ~50% zeros, got {}%",
            num_zeros * 100 / 1000
        );
        assert_eq!(num_zeros + num_nonzero, 1000);
    }

    #[test]
    fn test_dropout_scaling() {
        set_dropout_seed(12345);
        let input = Tensor::from_f32_slice(&[1.0; 100], &[100]);
        CommandBatch::begin();
        let (output, _) = dropout(&input, 0.5, true).unwrap();
        CommandBatch::sync();
        CommandBatch::end();

        let result = output.as_f32_slice();
        // Non-zero elements should be scaled by 1/(1-0.5) = 2.0
        for &val in result.iter() {
            if val != 0.0 {
                assert!(
                    (val - 2.0).abs() < 1e-5,
                    "Non-zero element should be scaled to 2.0, got {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_dropout_invalid_rate() {
        let input = Tensor::from_f32_slice(&[1.0; 10], &[10]);
        assert!(dropout(&input, -0.1, true).is_err());
        assert!(dropout(&input, 1.0, true).is_err());
        assert!(dropout(&input, 1.5, true).is_err());
    }
}
