//! Lion optimizer implementation.

use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

use crate::command_batch::CommandBatch;
use crate::device::MetalContext;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

use super::config::{LionConfig, ParamState};
use super::pipelines::{create_buffer, get_pipelines, LionParams, LionParamsBF16};

/// Lion optimizer.
///
/// Lion (EvoLved Sign Momentum) is a simple and memory-efficient optimizer
/// that uses sign-based updates. It typically works well for language models
/// and requires less memory than Adam (only stores momentum, not variance).
///
/// Update rule:
/// ```text
/// update = sign(beta2 * m + (1 - beta2) * g)
/// m_new = beta1 * m + (1 - beta1) * g
/// w_new = w - lr * update - lr * weight_decay * w
/// ```
pub struct Lion {
    config: LionConfig,
}

impl Lion {
    /// Create a new Lion optimizer with the given configuration.
    pub fn new(config: LionConfig) -> Self {
        Self { config }
    }

    /// Create a new Lion optimizer with default settings and given learning rate.
    pub fn with_lr(lr: f32) -> Self {
        Self::new(LionConfig::new(lr))
    }

    /// Get the current learning rate.
    pub fn lr(&self) -> f32 {
        self.config.lr
    }

    /// Set the learning rate.
    pub fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }

    /// Perform one optimization step.
    ///
    /// Updates the weights in-place using the gradients and momentum state.
    pub fn step(&self, weights: &Tensor, gradients: &Tensor, state: &mut ParamState) {
        let _timer = timed(OpCategory::LionStep, weights.numel());
        assert_eq!(weights.precision(), Precision::FP32);
        assert_eq!(gradients.precision(), Precision::FP32);
        assert_eq!(weights.shape(), gradients.shape());
        assert_eq!(weights.shape(), state.momentum.shape());

        let count = weights.numel();
        if count == 0 {
            return;
        }

        let ctx = MetalContext::global();
        let pipelines = get_pipelines();

        let params = LionParams {
            count: count as u32,
            lr: self.config.lr,
            beta1: self.config.beta1,
            beta2: self.config.beta2,
            weight_decay: self.config.weight_decay,
        };

        let params_buffer = create_buffer(ctx, &params);

        let weights_buf = weights.buffer();
        let momentum_buf = state.momentum.buffer();
        let gradients_buf = gradients.buffer();

        let thread_width = pipelines.lion_step.threadExecutionWidth();
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

        CommandBatch::dispatch(
            &pipelines.lion_step,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(weights_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(momentum_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(gradients_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
            },
            grid_size,
            threadgroup_size,
        );
    }

    /// Perform one optimization step for BF16 weights with FP32 gradients.
    ///
    /// This is the mixed-precision variant: weights are stored in BF16 for memory
    /// efficiency, but gradients and momentum remain in FP32 for numerical stability.
    pub fn step_bf16(&self, weights: &Tensor, gradients: &Tensor, state: &mut ParamState) {
        let _timer = timed(OpCategory::LionStep, weights.numel());
        assert_eq!(weights.precision(), Precision::BF16, "step_bf16 requires BF16 weights");
        assert_eq!(gradients.precision(), Precision::FP32, "step_bf16 requires FP32 gradients");
        assert_eq!(weights.shape(), gradients.shape());
        assert_eq!(weights.shape(), state.momentum.shape());

        let count = weights.numel();
        if count == 0 {
            return;
        }

        let ctx = MetalContext::global();
        let pipelines = get_pipelines();

        let params = LionParamsBF16 {
            lr: self.config.lr,
            beta1: self.config.beta1,
            beta2: self.config.beta2,
            weight_decay: self.config.weight_decay,
            lr_scale: 1.0,
        };

        let params_buffer = create_buffer(ctx, &params);

        let numel = count as u32;
        let numel_buffer = create_buffer(ctx, &numel);

        let weights_buf = weights.buffer();
        let momentum_buf = state.momentum.buffer();
        let gradients_buf = gradients.buffer();

        let thread_width = pipelines.lion_step_bf16.threadExecutionWidth();
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

        CommandBatch::dispatch(
            &pipelines.lion_step_bf16,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(weights_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(gradients_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(momentum_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&numel_buffer), 0, 4);
            },
            grid_size,
            threadgroup_size,
        );
    }

    /// Perform one optimization step with per-parameter learning rate scaling.
    pub fn step_scaled(
        &self,
        weights: &Tensor,
        gradients: &Tensor,
        lr_scale: &Tensor,
        state: &mut ParamState,
    ) {
        assert_eq!(weights.precision(), Precision::FP32);
        assert_eq!(gradients.precision(), Precision::FP32);
        assert_eq!(lr_scale.precision(), Precision::FP32);
        assert_eq!(weights.shape(), gradients.shape());
        assert_eq!(weights.shape(), lr_scale.shape());
        assert_eq!(weights.shape(), state.momentum.shape());

        let count = weights.numel();
        if count == 0 {
            return;
        }

        let ctx = MetalContext::global();
        let pipelines = get_pipelines();

        let params = LionParams {
            count: count as u32,
            lr: self.config.lr,
            beta1: self.config.beta1,
            beta2: self.config.beta2,
            weight_decay: self.config.weight_decay,
        };

        let params_buffer = create_buffer(ctx, &params);

        let weights_buf = weights.buffer();
        let momentum_buf = state.momentum.buffer();
        let gradients_buf = gradients.buffer();
        let lr_scale_buf = lr_scale.buffer();

        let thread_width = pipelines.lion_step_scaled.threadExecutionWidth();
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

        CommandBatch::dispatch(
            &pipelines.lion_step_scaled,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(weights_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(momentum_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(gradients_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(lr_scale_buf), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 4);
            },
            grid_size,
            threadgroup_size,
        );
    }
}
