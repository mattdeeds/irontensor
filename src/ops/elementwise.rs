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

const ELEMENTWISE_SHADER: &str = include_str!("../shaders/elementwise.metal");

struct ElementwisePipelines {
    add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    mul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    scale: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    add_scalar: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    silu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gelu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    relu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    swiglu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static ELEMENTWISE_PIPELINES: OnceLock<ElementwisePipelines> = OnceLock::new();

fn get_pipelines() -> &'static ElementwisePipelines {
    ELEMENTWISE_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(ELEMENTWISE_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile elementwise shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        ElementwisePipelines {
            add: make_pipeline("add_f32"),
            mul: make_pipeline("mul_f32"),
            scale: make_pipeline("scale_f32"),
            add_scalar: make_pipeline("add_scalar_f32"),
            silu: make_pipeline("silu_f32"),
            gelu: make_pipeline("gelu_f32"),
            relu: make_pipeline("relu_f32"),
            swiglu: make_pipeline("swiglu_f32"),
        }
    })
}

fn dispatch_binary_op(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
) -> Tensor {
    assert_eq!(a.precision(), Precision::FP32);
    assert_eq!(b.precision(), Precision::FP32);
    assert_eq!(a.shape(), b.shape(), "Tensors must have the same shape");

    let count = a.numel();
    let c = Tensor::zeros(a.shape(), Precision::FP32);

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

    let thread_width = pipeline.threadExecutionWidth();
    let grid_size = MTLSize { width: count, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: thread_width.min(count), height: 1, depth: 1 };

    let a_buf = a.buffer();
    let b_buf = b.buffer();
    let c_buf = c.buffer();

    CommandBatch::dispatch(
        pipeline,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(c_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 3);
        },
        grid_size,
        threadgroup_size,
    );

    c
}

fn dispatch_unary_op(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    input: &Tensor,
) -> Tensor {
    assert_eq!(input.precision(), Precision::FP32);

    let count = input.numel();
    let output = Tensor::zeros(input.shape(), Precision::FP32);

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

    let thread_width = pipeline.threadExecutionWidth();
    let grid_size = MTLSize { width: count, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: thread_width.min(count), height: 1, depth: 1 };

    let input_buf = input.buffer();
    let output_buf = output.buffer();

    CommandBatch::dispatch(
        pipeline,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 2);
        },
        grid_size,
        threadgroup_size,
    );

    output
}

fn dispatch_scalar_op(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    input: &Tensor,
    scalar: f32,
) -> Tensor {
    assert_eq!(input.precision(), Precision::FP32);

    let count = input.numel();
    let output = Tensor::zeros(input.shape(), Precision::FP32);

    let ctx = MetalContext::global();

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

    let thread_width = pipeline.threadExecutionWidth();
    let grid_size = MTLSize { width: count, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: thread_width.min(count), height: 1, depth: 1 };

    let input_buf = input.buffer();
    let output_buf = output.buffer();

    CommandBatch::dispatch(
        pipeline,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(input_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&scalar_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 3);
        },
        grid_size,
        threadgroup_size,
    );

    output
}

/// Element-wise addition: C = A + B
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::Elementwise("add".to_string()), a.numel());
    dispatch_binary_op(&get_pipelines().add, a, b)
}

/// Element-wise multiplication: C = A * B
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::Elementwise("mul".to_string()), a.numel());
    dispatch_binary_op(&get_pipelines().mul, a, b)
}

/// Scale tensor by scalar: B = A * scalar
pub fn scale(a: &Tensor, scalar: f32) -> Tensor {
    let _timer = timed(OpCategory::Elementwise("scale".to_string()), a.numel());
    dispatch_scalar_op(&get_pipelines().scale, a, scalar)
}

/// Add scalar to tensor: B = A + scalar
pub fn add_scalar(a: &Tensor, scalar: f32) -> Tensor {
    let _timer = timed(OpCategory::Elementwise("add_scalar".to_string()), a.numel());
    dispatch_scalar_op(&get_pipelines().add_scalar, a, scalar)
}

/// SiLU (Swish) activation: y = x * sigmoid(x)
pub fn silu(input: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::Elementwise("silu".to_string()), input.numel());
    dispatch_unary_op(&get_pipelines().silu, input)
}

/// GELU (Gaussian Error Linear Unit) activation using tanh approximation.
///
/// Uses the fast tanh approximation formula:
/// ```text
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
/// ```
///
/// This approximation is commonly used in transformer models (GPT-2, BERT)
/// as it is faster to compute than the exact GELU while maintaining
/// similar accuracy. The maximum error vs exact GELU is ~0.004.
pub fn gelu(input: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::Elementwise("gelu".to_string()), input.numel());
    dispatch_unary_op(&get_pipelines().gelu, input)
}

/// ReLU activation: y = max(0, x)
pub fn relu(input: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::Elementwise("relu".to_string()), input.numel());
    dispatch_unary_op(&get_pipelines().relu, input)
}

/// SwiGLU: output = silu(gate) * up
/// Used in Llama-style FFN
pub fn swiglu(gate: &Tensor, up: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::Elementwise("swiglu".to_string()), gate.numel());
    dispatch_binary_op(&get_pipelines().swiglu, gate, up)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Tensor::from_f32_slice(&[5.0, 6.0, 7.0, 8.0], &[4]);
        let c = add(&a, &b);
        let result = c.as_f32_slice();
        assert_eq!(result, &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Tensor::from_f32_slice(&[2.0, 3.0, 4.0, 5.0], &[4]);
        let c = mul(&a, &b);
        let result = c.as_f32_slice();
        assert_eq!(result, &[2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_scale() {
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = scale(&a, 2.5);
        let result = b.as_f32_slice();
        assert_eq!(result, &[2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_silu() {
        let input = Tensor::from_f32_slice(&[0.0, 1.0, -1.0, 2.0], &[4]);
        let output = silu(&input);
        let result = output.as_f32_slice();

        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!((result[0] - 0.0).abs() < 1e-5);
        // silu(1) = 1 * sigmoid(1) ≈ 0.731
        assert!((result[1] - 0.7310586).abs() < 1e-5);
        // silu(-1) = -1 * sigmoid(-1) ≈ -0.269
        assert!((result[2] - (-0.2689414)).abs() < 1e-5);
    }

    #[test]
    fn test_relu() {
        let input = Tensor::from_f32_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]);
        let output = relu(&input);
        let result = output.as_f32_slice();
        assert_eq!(result, &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_swiglu() {
        let gate = Tensor::from_f32_slice(&[0.0, 1.0, 2.0], &[3]);
        let up = Tensor::from_f32_slice(&[1.0, 2.0, 3.0], &[3]);
        let output = swiglu(&gate, &up);
        let result = output.as_f32_slice();

        // swiglu(0, 1) = silu(0) * 1 = 0
        assert!((result[0] - 0.0).abs() < 1e-5);
        // swiglu(1, 2) = silu(1) * 2 ≈ 1.462
        assert!((result[1] - 1.4621172).abs() < 1e-5);
    }
}
