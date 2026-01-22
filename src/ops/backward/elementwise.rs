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

const BACKWARD_ELEMENTWISE_SHADER: &str = include_str!("../../shaders/backward/elementwise.metal");

struct ElementwiseBackwardPipelines {
    mul_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    scale_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    silu_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gelu_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    relu_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    swiglu_backward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static ELEMENTWISE_BACKWARD_PIPELINES: OnceLock<ElementwiseBackwardPipelines> = OnceLock::new();

fn get_pipelines() -> &'static ElementwiseBackwardPipelines {
    ELEMENTWISE_BACKWARD_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(BACKWARD_ELEMENTWISE_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile backward elementwise shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        ElementwiseBackwardPipelines {
            mul_backward: make_pipeline("mul_backward_f32"),
            scale_backward: make_pipeline("scale_backward_f32"),
            silu_backward: make_pipeline("silu_backward_f32"),
            gelu_backward: make_pipeline("gelu_backward_f32"),
            relu_backward: make_pipeline("relu_backward_f32"),
            swiglu_backward: make_pipeline("swiglu_backward_f32"),
        }
    })
}

/// Backward pass for element-wise multiplication
/// Returns (grad_a, grad_b)
pub fn mul_backward(grad_output: &Tensor, a: &Tensor, b: &Tensor) -> (Tensor, Tensor) {
    let _timer = timed(OpCategory::ElementwiseBackward("mul".to_string()), grad_output.numel());
    assert_eq!(grad_output.precision(), Precision::FP32);
    assert_eq!(a.precision(), Precision::FP32);
    assert_eq!(b.precision(), Precision::FP32);
    assert_eq!(grad_output.shape(), a.shape());
    assert_eq!(a.shape(), b.shape());

    let count = a.numel();
    let grad_a = Tensor::zeros(a.shape(), Precision::FP32);
    let grad_b = Tensor::zeros(b.shape(), Precision::FP32);

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

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.mul_backward);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(grad_output.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(a.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(b.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(grad_a.buffer()), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(grad_b.buffer()), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 5);
    }

    let thread_width = pipelines.mul_backward.threadExecutionWidth();
    let grid_size = MTLSize { width: count, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: thread_width.min(count), height: 1, depth: 1 };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    (grad_a, grad_b)
}

/// Backward pass for scale operation
pub fn scale_backward(grad_output: &Tensor, scalar: f32) -> Tensor {
    let _timer = timed(OpCategory::ElementwiseBackward("scale".to_string()), grad_output.numel());
    assert_eq!(grad_output.precision(), Precision::FP32);

    let count = grad_output.numel();
    let grad_input = Tensor::zeros(grad_output.shape(), Precision::FP32);

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let scalar_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&scalar as *const _ as *mut _).unwrap(),
            std::mem::size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create scalar buffer");

    let count_u32: u32 = count as u32;
    let count_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&count_u32 as *const _ as *mut _).unwrap(),
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create count buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.scale_backward);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(grad_output.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(grad_input.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&scalar_buffer), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 3);
    }

    let thread_width = pipelines.scale_backward.threadExecutionWidth();
    let grid_size = MTLSize { width: count, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: thread_width.min(count), height: 1, depth: 1 };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    grad_input
}

fn dispatch_unary_backward(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    grad_output: &Tensor,
    input: &Tensor,
) -> Tensor {
    assert_eq!(grad_output.precision(), Precision::FP32);
    assert_eq!(input.precision(), Precision::FP32);
    assert_eq!(grad_output.shape(), input.shape());

    let count = input.numel();
    let grad_input = Tensor::zeros(input.shape(), Precision::FP32);

    let ctx = MetalContext::global();

    let count_u32: u32 = count as u32;
    let count_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&count_u32 as *const _ as *mut _).unwrap(),
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create count buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(grad_output.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(grad_input.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 3);
    }

    let thread_width = pipeline.threadExecutionWidth();
    let grid_size = MTLSize { width: count, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: thread_width.min(count), height: 1, depth: 1 };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    grad_input
}

/// Backward pass for SiLU activation
pub fn silu_backward(grad_output: &Tensor, input: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::ElementwiseBackward("silu".to_string()), grad_output.numel());
    dispatch_unary_backward(&get_pipelines().silu_backward, grad_output, input)
}

/// Backward pass for GELU activation
pub fn gelu_backward(grad_output: &Tensor, input: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::ElementwiseBackward("gelu".to_string()), grad_output.numel());
    dispatch_unary_backward(&get_pipelines().gelu_backward, grad_output, input)
}

/// Backward pass for ReLU activation
pub fn relu_backward(grad_output: &Tensor, input: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::ElementwiseBackward("relu".to_string()), grad_output.numel());
    dispatch_unary_backward(&get_pipelines().relu_backward, grad_output, input)
}

/// Backward pass for SwiGLU
/// Returns (grad_gate, grad_up)
pub fn swiglu_backward(grad_output: &Tensor, gate: &Tensor, up: &Tensor) -> (Tensor, Tensor) {
    let _timer = timed(OpCategory::ElementwiseBackward("swiglu".to_string()), grad_output.numel());
    assert_eq!(grad_output.precision(), Precision::FP32);
    assert_eq!(gate.precision(), Precision::FP32);
    assert_eq!(up.precision(), Precision::FP32);
    assert_eq!(grad_output.shape(), gate.shape());
    assert_eq!(gate.shape(), up.shape());

    let count = gate.numel();
    let grad_gate = Tensor::zeros(gate.shape(), Precision::FP32);
    let grad_up = Tensor::zeros(up.shape(), Precision::FP32);

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

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.swiglu_backward);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(grad_output.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(gate.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(up.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(grad_gate.buffer()), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(grad_up.buffer()), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 5);
    }

    let thread_width = pipelines.swiglu_backward.threadExecutionWidth();
    let grid_size = MTLSize { width: count, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: thread_width.min(count), height: 1, depth: 1 };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    (grad_gate, grad_up)
}

#[cfg(test)]
mod tests {
    use super::*;
    

    fn numerical_gradient<F>(f: F, x: &[f32], eps: f32) -> Vec<f32>
    where
        F: Fn(&[f32]) -> f32,
    {
        let mut grad = vec![0.0f32; x.len()];
        for i in 0..x.len() {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * eps);
        }
        grad
    }

    #[test]
    fn test_silu_backward() {
        let x_data = vec![-1.0f32, 0.0, 1.0, 2.0];
        let grad_out_data = vec![1.0f32; 4];

        let x = Tensor::from_f32_slice(&x_data, &[4]);
        let grad_out = Tensor::from_f32_slice(&grad_out_data, &[4]);

        let grad_x = silu_backward(&grad_out, &x);
        let result = grad_x.as_f32_slice();

        // Numerical gradient check
        for i in 0..4 {
            let numerical = numerical_gradient(
                |xv| {
                    let xi = xv[i];
                    let sig = 1.0 / (1.0 + (-xi).exp());
                    xi * sig
                },
                &x_data,
                1e-4,
            )[i];
            assert!(
                (result[i] - numerical).abs() < 1e-3,
                "SiLU backward mismatch at {}: analytical={}, numerical={}",
                i, result[i], numerical
            );
        }
    }

    #[test]
    fn test_relu_backward() {
        let x_data = vec![-1.0f32, 0.0, 1.0, 2.0];
        let grad_out_data = vec![1.0f32, 1.0, 1.0, 1.0];

        let x = Tensor::from_f32_slice(&x_data, &[4]);
        let grad_out = Tensor::from_f32_slice(&grad_out_data, &[4]);

        let grad_x = relu_backward(&grad_out, &x);
        let result = grad_x.as_f32_slice();

        // ReLU gradient: 1 if x > 0, else 0
        assert_eq!(result[0], 0.0); // x = -1
        assert_eq!(result[1], 0.0); // x = 0
        assert_eq!(result[2], 1.0); // x = 1
        assert_eq!(result[3], 1.0); // x = 2
    }

    #[test]
    fn test_mul_backward() {
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let grad_out_data = vec![1.0f32; 4];

        let a = Tensor::from_f32_slice(&a_data, &[4]);
        let b = Tensor::from_f32_slice(&b_data, &[4]);
        let grad_out = Tensor::from_f32_slice(&grad_out_data, &[4]);

        let (grad_a, grad_b) = mul_backward(&grad_out, &a, &b);

        // grad_a = grad_out * b
        assert_eq!(grad_a.as_f32_slice(), &[5.0, 6.0, 7.0, 8.0]);
        // grad_b = grad_out * a
        assert_eq!(grad_b.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scale_backward() {
        let grad_out_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let scalar = 2.5f32;

        let grad_out = Tensor::from_f32_slice(&grad_out_data, &[4]);
        let grad_x = scale_backward(&grad_out, scalar);

        // grad_x = grad_out * scalar
        assert_eq!(grad_x.as_f32_slice(), &[2.5, 5.0, 7.5, 10.0]);
    }
}
