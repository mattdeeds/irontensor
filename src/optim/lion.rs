use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::command_batch::CommandBatch;
use crate::device::MetalContext;
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const LION_SHADER: &str = include_str!("../shaders/lion.metal");

#[repr(C)]
struct LionParams {
    count: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
}

struct LionPipelines {
    lion_step: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    lion_step_scaled: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    zero_gradients: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    grad_norm_squared: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    grad_clip: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static LION_PIPELINES: OnceLock<LionPipelines> = OnceLock::new();

fn get_pipelines() -> &'static LionPipelines {
    LION_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(LION_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile Lion shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        LionPipelines {
            lion_step: make_pipeline("lion_step_f32"),
            lion_step_scaled: make_pipeline("lion_step_scaled_f32"),
            zero_gradients: make_pipeline("zero_gradients_f32"),
            grad_norm_squared: make_pipeline("grad_norm_squared_f32"),
            grad_clip: make_pipeline("grad_clip_f32"),
        }
    })
}

/// Momentum state for a parameter tensor
pub struct ParamState {
    /// Momentum tensor (same shape as parameter)
    pub momentum: Tensor,
}

impl ParamState {
    /// Create a new parameter state with zeroed momentum
    pub fn new(shape: &[usize]) -> Self {
        Self {
            momentum: Tensor::zeros(shape, Precision::FP32),
        }
    }
}

/// Lion optimizer configuration
#[derive(Clone, Debug)]
pub struct LionConfig {
    /// Learning rate (default: 1e-4)
    pub lr: f32,
    /// Momentum decay for update computation (default: 0.9)
    pub beta1: f32,
    /// Momentum decay for momentum update (default: 0.99)
    pub beta2: f32,
    /// Weight decay coefficient (default: 0.0)
    pub weight_decay: f32,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }
}

impl LionConfig {
    /// Create a new Lion config with the given learning rate
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set beta parameters
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

/// Lion optimizer
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
    /// Create a new Lion optimizer with the given configuration
    pub fn new(config: LionConfig) -> Self {
        Self { config }
    }

    /// Create a new Lion optimizer with default settings and given learning rate
    pub fn with_lr(lr: f32) -> Self {
        Self::new(LionConfig::new(lr))
    }

    /// Get the current learning rate
    pub fn lr(&self) -> f32 {
        self.config.lr
    }

    /// Set the learning rate
    pub fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }

    /// Perform one optimization step
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

        let params_buffer = unsafe {
            ctx.device().newBufferWithBytes_length_options(
                NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<LionParams>(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("Failed to create params buffer");

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

    /// Perform one optimization step with per-parameter learning rate scaling
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

        let params_buffer = unsafe {
            ctx.device().newBufferWithBytes_length_options(
                NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<LionParams>(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("Failed to create params buffer");

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

/// Zero out gradients in a tensor
pub fn zero_gradients(gradients: &Tensor) {
    let _timer = timed(OpCategory::ZeroGradients, gradients.numel());
    assert_eq!(gradients.precision(), Precision::FP32);

    let count = gradients.numel();
    if count == 0 {
        return;
    }

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let count_u32: u32 = count as u32;
    let count_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&count_u32 as *const _ as *mut _).unwrap(),
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create count buffer");

    let gradients_buf = gradients.buffer();

    let thread_width = pipelines.zero_gradients.threadExecutionWidth();
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
        &pipelines.zero_gradients,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(gradients_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 1);
        },
        grid_size,
        threadgroup_size,
    );
}

/// Compute the global L2 norm of gradients
pub fn grad_norm(gradients: &Tensor) -> f32 {
    let _timer = timed(OpCategory::GradientNorm, gradients.numel());
    assert_eq!(gradients.precision(), Precision::FP32);

    let count = gradients.numel();
    if count == 0 {
        return 0.0;
    }

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    // Create buffer for atomic sum
    let sum_sq_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&0.0f32 as *const _ as *mut _).unwrap(),
            std::mem::size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create sum buffer");

    let count_u32: u32 = count as u32;
    let count_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&count_u32 as *const _ as *mut _).unwrap(),
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create count buffer");

    let gradients_buf = gradients.buffer();

    let thread_width = pipelines.grad_norm_squared.threadExecutionWidth();
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
        &pipelines.grad_norm_squared,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(gradients_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&sum_sq_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 2);
        },
        grid_size,
        threadgroup_size,
    );

    // Sync before reading result
    CommandBatch::sync();

    // Read back sum of squares
    let sum_sq = unsafe { *(sum_sq_buffer.contents().as_ptr() as *const f32) };
    sum_sq.sqrt()
}

/// Clip gradients by global norm
///
/// If the global norm exceeds `max_norm`, scales all gradients by `max_norm / actual_norm`.
/// Returns the actual norm before clipping.
pub fn clip_grad_norm(gradients: &Tensor, max_norm: f32) -> f32 {
    let _timer = timed(OpCategory::GradientClip, gradients.numel());
    let actual_norm = grad_norm(gradients);

    if actual_norm <= max_norm || actual_norm == 0.0 {
        return actual_norm;
    }

    let clip_scale = max_norm / actual_norm;

    let count = gradients.numel();
    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let scale_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&clip_scale as *const _ as *mut _).unwrap(),
            std::mem::size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create scale buffer");

    let count_u32: u32 = count as u32;
    let count_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&count_u32 as *const _ as *mut _).unwrap(),
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create count buffer");

    let gradients_buf = gradients.buffer();

    let thread_width = pipelines.grad_clip.threadExecutionWidth();
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
        &pipelines.grad_clip,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(gradients_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&scale_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 2);
        },
        grid_size,
        threadgroup_size,
    );

    actual_norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lion_step_basic() {
        // Initial weights
        let weights = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        // Gradients (all positive)
        let gradients = Tensor::from_f32_slice(&[0.1, 0.2, 0.3, 0.4], &[4]);
        // Initial momentum (zeros)
        let mut state = ParamState::new(&[4]);

        let optimizer = Lion::new(LionConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        });

        optimizer.step(&weights, &gradients, &mut state);

        let w = weights.as_f32_slice();
        let m = state.momentum.as_f32_slice();

        // With zero initial momentum and positive gradients:
        // update = sign(0.99 * 0 + 0.01 * g) = sign(0.01 * g) = 1 for all
        // w_new = w - 0.1 * 1 = w - 0.1
        for i in 0..4 {
            let expected_w = (i + 1) as f32 - 0.1;
            assert!(
                (w[i] - expected_w).abs() < 1e-5,
                "Weight mismatch at {}: expected {}, got {}",
                i,
                expected_w,
                w[i]
            );
        }

        // Momentum should be updated: m_new = 0.9 * 0 + 0.1 * g = 0.1 * g
        for i in 0..4 {
            let expected_m = 0.1 * (i + 1) as f32 * 0.1;
            assert!(
                (m[i] - expected_m).abs() < 1e-5,
                "Momentum mismatch at {}: expected {}, got {}",
                i,
                expected_m,
                m[i]
            );
        }
    }

    #[test]
    fn test_lion_step_with_momentum() {
        // Test that momentum affects the update direction
        let weights = Tensor::from_f32_slice(&[1.0, 1.0, 1.0, 1.0], &[4]);
        // Gradients: some positive, some negative
        let gradients = Tensor::from_f32_slice(&[1.0, -1.0, 0.5, -0.5], &[4]);
        let mut state = ParamState::new(&[4]);

        let optimizer = Lion::with_lr(0.01);

        // First step
        optimizer.step(&weights, &gradients, &mut state);
        let w1 = weights.as_f32_slice().to_vec();

        // Second step with same gradients
        optimizer.step(&weights, &gradients, &mut state);
        let w2 = weights.as_f32_slice().to_vec();

        // Weights should continue to decrease/increase based on sign
        assert!(w2[0] < w1[0], "Positive gradient should decrease weight");
        assert!(w2[1] > w1[1], "Negative gradient should increase weight");
    }

    #[test]
    fn test_lion_weight_decay() {
        let weights = Tensor::from_f32_slice(&[10.0, -10.0, 5.0, -5.0], &[4]);
        let gradients = Tensor::from_f32_slice(&[0.0, 0.0, 0.0, 0.0], &[4]); // Zero gradients
        let mut state = ParamState::new(&[4]);

        let optimizer = Lion::new(LionConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.1,
        });

        let w_before = weights.as_f32_slice().to_vec();
        optimizer.step(&weights, &gradients, &mut state);
        let w_after = weights.as_f32_slice();

        // With zero gradients, only weight decay affects weights
        // update = sign(0) = 0, so w_new = w - lr * wd * w = w * (1 - lr * wd)
        // w_new = w * (1 - 0.1 * 0.1) = w * 0.99
        for i in 0..4 {
            let expected = w_before[i] * 0.99;
            assert!(
                (w_after[i] - expected).abs() < 1e-4,
                "Weight decay mismatch at {}: expected {}, got {}",
                i,
                expected,
                w_after[i]
            );
        }
    }

    #[test]
    fn test_zero_gradients() {
        let gradients = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);

        zero_gradients(&gradients);

        let g = gradients.as_f32_slice();
        for i in 0..4 {
            assert_eq!(g[i], 0.0, "Gradient at {} should be zero", i);
        }
    }

    #[test]
    fn test_grad_norm() {
        // Test with known values: [3, 4] has norm 5
        let gradients = Tensor::from_f32_slice(&[3.0, 4.0], &[2]);
        let norm = grad_norm(&gradients);
        assert!(
            (norm - 5.0).abs() < 1e-4,
            "Norm mismatch: expected 5.0, got {}",
            norm
        );
    }

    #[test]
    fn test_clip_grad_norm() {
        let gradients = Tensor::from_f32_slice(&[3.0, 4.0], &[2]);

        // Clip to max_norm=2.5 (original norm is 5.0)
        let actual_norm = clip_grad_norm(&gradients, 2.5);

        assert!(
            (actual_norm - 5.0).abs() < 1e-4,
            "Returned norm mismatch: expected 5.0, got {}",
            actual_norm
        );

        // After clipping, gradients should be scaled by 2.5/5.0 = 0.5
        let g = gradients.as_f32_slice();
        assert!(
            (g[0] - 1.5).abs() < 1e-4,
            "Clipped grad[0] mismatch: expected 1.5, got {}",
            g[0]
        );
        assert!(
            (g[1] - 2.0).abs() < 1e-4,
            "Clipped grad[1] mismatch: expected 2.0, got {}",
            g[1]
        );

        // Verify new norm
        let new_norm = grad_norm(&gradients);
        assert!(
            (new_norm - 2.5).abs() < 1e-4,
            "New norm mismatch: expected 2.5, got {}",
            new_norm
        );
    }

    #[test]
    fn test_clip_grad_norm_no_clip() {
        let gradients = Tensor::from_f32_slice(&[3.0, 4.0], &[2]);
        let original = gradients.as_f32_slice().to_vec();

        // Clip to max_norm=10.0 (original norm is 5.0, so no clipping needed)
        let actual_norm = clip_grad_norm(&gradients, 10.0);

        assert!(
            (actual_norm - 5.0).abs() < 1e-4,
            "Returned norm mismatch: expected 5.0, got {}",
            actual_norm
        );

        // Gradients should be unchanged
        let g = gradients.as_f32_slice();
        assert_eq!(g[0], original[0], "Grad[0] should be unchanged");
        assert_eq!(g[1], original[1], "Grad[1] should be unchanged");
    }

    #[test]
    fn test_lion_multiple_steps() {
        // Test convergence towards minimum of f(x) = x^2
        // Gradient of x^2 is 2x
        let weights = Tensor::from_f32_slice(&[5.0], &[1]);
        let mut state = ParamState::new(&[1]);

        let optimizer = Lion::with_lr(0.1);

        // Run several steps
        for _ in 0..50 {
            let w = weights.as_f32_slice()[0];
            let grad = 2.0 * w; // gradient of x^2
            let gradients = Tensor::from_f32_slice(&[grad], &[1]);
            optimizer.step(&weights, &gradients, &mut state);
        }

        let final_w = weights.as_f32_slice()[0];
        // Should have moved towards 0
        assert!(
            final_w.abs() < 1.0,
            "Weight should be close to 0, got {}",
            final_w
        );
    }
}
