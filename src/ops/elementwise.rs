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

const ELEMENTWISE_SHADER: &str = include_str!("../shaders/elementwise.metal");

struct ElementwisePipelines {
    add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    mul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    scale: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    scale_inplace: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    add_scalar: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    add3: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    silu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gelu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    relu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    swiglu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    axpy_inplace: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    zero: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
            scale_inplace: make_pipeline("scale_inplace_f32"),
            add_scalar: make_pipeline("add_scalar_f32"),
            add3: make_pipeline("add3_f32"),
            silu: make_pipeline("silu_f32"),
            gelu: make_pipeline("gelu_f32"),
            relu: make_pipeline("relu_f32"),
            swiglu: make_pipeline("swiglu_f32"),
            axpy_inplace: make_pipeline("axpy_inplace_f32"),
            zero: make_pipeline("zero_f32"),
        }
    })
}

fn try_dispatch_binary_op(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
    op_name: &'static str,
) -> TensorResult<Tensor> {
    if a.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: op_name,
            expected: "FP32",
            got: if a.precision() == Precision::BF16 { "BF16" } else { "unknown" },
        });
    }
    if b.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: op_name,
            expected: "FP32",
            got: if b.precision() == Precision::BF16 { "BF16" } else { "unknown" },
        });
    }
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            operation: op_name,
            expected: format!("{:?}", a.shape()),
            got: format!("{:?}", b.shape()),
        });
    }

    Ok(dispatch_binary_op_inner(pipeline, a, b))
}

fn dispatch_binary_op_inner(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    a: &Tensor,
    b: &Tensor,
) -> Tensor {
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

fn try_dispatch_unary_op(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    input: &Tensor,
    op_name: &'static str,
) -> TensorResult<Tensor> {
    if input.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: op_name,
            expected: "FP32",
            got: if input.precision() == Precision::BF16 { "BF16" } else { "unknown" },
        });
    }

    Ok(dispatch_unary_op_inner(pipeline, input))
}

fn dispatch_unary_op_inner(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    input: &Tensor,
) -> Tensor {
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

fn try_dispatch_scalar_op(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    input: &Tensor,
    scalar: f32,
    op_name: &'static str,
) -> TensorResult<Tensor> {
    if input.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: op_name,
            expected: "FP32",
            got: if input.precision() == Precision::BF16 { "BF16" } else { "unknown" },
        });
    }

    Ok(dispatch_scalar_op_inner(pipeline, input, scalar))
}

fn dispatch_scalar_op_inner(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    input: &Tensor,
    scalar: f32,
) -> Tensor {
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
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if tensors are not FP32
/// - `TensorError::ShapeMismatch` if tensor shapes don't match
pub fn add(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    let _timer = timed(OpCategory::Elementwise("add".to_string()), a.numel());
    try_dispatch_binary_op(&get_pipelines().add, a, b, "add")
}

/// Element-wise multiplication: C = A * B
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if tensors are not FP32
/// - `TensorError::ShapeMismatch` if tensor shapes don't match
pub fn mul(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    let _timer = timed(OpCategory::Elementwise("mul".to_string()), a.numel());
    try_dispatch_binary_op(&get_pipelines().mul, a, b, "mul")
}

/// Scale tensor by scalar: B = A * scalar
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if tensor is not FP32
pub fn scale(a: &Tensor, scalar: f32) -> TensorResult<Tensor> {
    let _timer = timed(OpCategory::Elementwise("scale".to_string()), a.numel());
    try_dispatch_scalar_op(&get_pipelines().scale, a, scalar, "scale")
}

/// Add scalar to tensor: B = A + scalar
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if tensor is not FP32
pub fn add_scalar(a: &Tensor, scalar: f32) -> TensorResult<Tensor> {
    let _timer = timed(OpCategory::Elementwise("add_scalar".to_string()), a.numel());
    try_dispatch_scalar_op(&get_pipelines().add_scalar, a, scalar, "add_scalar")
}

/// SiLU (Swish) activation: y = x * sigmoid(x)
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if tensor is not FP32
pub fn silu(input: &Tensor) -> TensorResult<Tensor> {
    let _timer = timed(OpCategory::Elementwise("silu".to_string()), input.numel());
    try_dispatch_unary_op(&get_pipelines().silu, input, "silu")
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
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if tensor is not FP32
pub fn gelu(input: &Tensor) -> TensorResult<Tensor> {
    let _timer = timed(OpCategory::Elementwise("gelu".to_string()), input.numel());
    try_dispatch_unary_op(&get_pipelines().gelu, input, "gelu")
}

/// ReLU activation: y = max(0, x)
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if tensor is not FP32
pub fn relu(input: &Tensor) -> TensorResult<Tensor> {
    let _timer = timed(OpCategory::Elementwise("relu".to_string()), input.numel());
    try_dispatch_unary_op(&get_pipelines().relu, input, "relu")
}

/// SwiGLU: output = silu(gate) * up
/// Used in Llama-style FFN
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if tensors are not FP32
/// - `TensorError::ShapeMismatch` if tensor shapes don't match
pub fn swiglu(gate: &Tensor, up: &Tensor) -> TensorResult<Tensor> {
    let _timer = timed(OpCategory::Elementwise("swiglu".to_string()), gate.numel());
    try_dispatch_binary_op(&get_pipelines().swiglu, gate, up, "swiglu")
}

/// Element-wise addition of three tensors: D = A + B + C
///
/// Fused kernel to reduce kernel launch overhead when combining gradients.
/// Equivalent to `add(add(a, b), c)` but with a single kernel dispatch.
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if tensors are not FP32
/// - `TensorError::ShapeMismatch` if tensor shapes don't match
pub fn add3(a: &Tensor, b: &Tensor, c: &Tensor) -> TensorResult<Tensor> {
    let _timer = timed(OpCategory::Elementwise("add3".to_string()), a.numel());

    if a.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: "add3",
            expected: "FP32",
            got: if a.precision() == Precision::BF16 { "BF16" } else { "unknown" },
        });
    }
    if b.precision() != Precision::FP32 || c.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: "add3",
            expected: "FP32",
            got: "non-FP32",
        });
    }
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(TensorError::ShapeMismatch {
            operation: "add3",
            expected: format!("{:?}", a.shape()),
            got: format!("{:?} or {:?}", b.shape(), c.shape()),
        });
    }

    let count = a.numel();
    let d = Tensor::zeros(a.shape(), Precision::FP32);

    let ctx = MetalContext::global();
    let pipeline = &get_pipelines().add3;

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
    let d_buf = d.buffer();

    CommandBatch::dispatch(
        pipeline,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(c_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(d_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 4);
        },
        grid_size,
        threadgroup_size,
    );

    Ok(d)
}

/// Scale tensor in-place: tensor = tensor * scalar
///
/// Modifies the tensor in-place without allocating a new buffer.
/// Useful for gradient clipping to avoid allocation overhead.
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if tensor is not FP32
pub fn scale_inplace(tensor: &Tensor, scalar: f32) -> TensorResult<()> {
    let _timer = timed(OpCategory::Elementwise("scale_inplace".to_string()), tensor.numel());

    if tensor.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: "scale_inplace",
            expected: "FP32",
            got: if tensor.precision() == Precision::BF16 { "BF16" } else { "unknown" },
        });
    }

    let count = tensor.numel();
    let ctx = MetalContext::global();
    let pipeline = &get_pipelines().scale_inplace;

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

    let tensor_buf = tensor.buffer();

    CommandBatch::dispatch(
        pipeline,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(tensor_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&scalar_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 2);
        },
        grid_size,
        threadgroup_size,
    );

    Ok(())
}

/// Scale multiple tensors in-place with the same scalar
///
/// Useful for gradient clipping where all gradients need the same scale factor.
/// More efficient than calling scale_inplace repeatedly as it reuses the scalar buffer.
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if any tensor is not FP32
pub fn scale_tensors_inplace(tensors: &[&Tensor], scalar: f32) -> TensorResult<()> {
    let _timer = timed(
        OpCategory::Elementwise("scale_batch".to_string()),
        tensors.iter().map(|t| t.numel()).sum(),
    );

    for tensor in tensors.iter() {
        if tensor.precision() != Precision::FP32 {
            return Err(TensorError::PrecisionMismatch {
                operation: "scale_tensors_inplace",
                expected: "FP32",
                got: if tensor.precision() == Precision::BF16 { "BF16" } else { "unknown" },
            });
        }
    }

    if tensors.is_empty() || scalar == 1.0 {
        return Ok(());
    }

    let ctx = MetalContext::global();
    let pipeline = &get_pipelines().scale_inplace;

    // Create scalar buffer once for all tensors
    let scalar_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&scalar as *const _ as *mut _).unwrap(),
            std::mem::size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create scalar buffer");

    let thread_width = pipeline.threadExecutionWidth();

    // Dispatch all scale operations with the same scalar buffer
    for tensor in tensors {
        let count = tensor.numel();
        let count_u32: u32 = count as u32;
        let count_buffer = unsafe {
            ctx.device().newBufferWithBytes_length_options(
                NonNull::new(&count_u32 as *const _ as *mut _).unwrap(),
                std::mem::size_of::<u32>(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("Failed to create count buffer");

        let grid_size = MTLSize { width: count, height: 1, depth: 1 };
        let threadgroup_size = MTLSize { width: thread_width.min(count), height: 1, depth: 1 };
        let tensor_buf = tensor.buffer();

        CommandBatch::dispatch(
            pipeline,
            |encoder| unsafe {
                encoder.setBuffer_offset_atIndex(Some(tensor_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&scalar_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 2);
            },
            grid_size,
            threadgroup_size,
        );
    }

    Ok(())
}

/// In-place scaled addition (AXPY): A = A + scale * B
///
/// Efficient for gradient accumulation: acc = acc + (1/N) * grad
/// where N is the number of gradient accumulation steps.
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if tensors are not FP32
/// - `TensorError::ShapeMismatch` if tensor shapes don't match
pub fn axpy_inplace(a: &Tensor, b: &Tensor, scale: f32) -> TensorResult<()> {
    let _timer = timed(OpCategory::Elementwise("axpy".to_string()), a.numel());

    if a.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: "axpy_inplace",
            expected: "FP32",
            got: if a.precision() == Precision::BF16 { "BF16" } else { "unknown" },
        });
    }
    if b.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: "axpy_inplace",
            expected: "FP32",
            got: if b.precision() == Precision::BF16 { "BF16" } else { "unknown" },
        });
    }
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            operation: "axpy_inplace",
            expected: format!("{:?}", a.shape()),
            got: format!("{:?}", b.shape()),
        });
    }

    let count = a.numel();
    let ctx = MetalContext::global();
    let pipeline = &get_pipelines().axpy_inplace;

    let scale_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&scale as *const _ as *mut _).unwrap(),
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

    let thread_width = pipeline.threadExecutionWidth();
    let grid_size = MTLSize { width: count, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: thread_width.min(count), height: 1, depth: 1 };

    let a_buf = a.buffer();
    let b_buf = b.buffer();

    CommandBatch::dispatch(
        pipeline,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&scale_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 3);
        },
        grid_size,
        threadgroup_size,
    );

    Ok(())
}

/// Zero tensor in-place
///
/// Sets all elements to zero. Useful for resetting accumulated gradients.
///
/// # Errors
/// - `TensorError::PrecisionMismatch` if tensor is not FP32
pub fn zero_tensor(tensor: &Tensor) -> TensorResult<()> {
    let _timer = timed(OpCategory::Elementwise("zero".to_string()), tensor.numel());

    if tensor.precision() != Precision::FP32 {
        return Err(TensorError::PrecisionMismatch {
            operation: "zero_tensor",
            expected: "FP32",
            got: if tensor.precision() == Precision::BF16 { "BF16" } else { "unknown" },
        });
    }

    let count = tensor.numel();
    let ctx = MetalContext::global();
    let pipeline = &get_pipelines().zero;

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

    let tensor_buf = tensor.buffer();

    CommandBatch::dispatch(
        pipeline,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(tensor_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 1);
        },
        grid_size,
        threadgroup_size,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Tensor::from_f32_slice(&[5.0, 6.0, 7.0, 8.0], &[4]);
        let c = add(&a, &b).unwrap();
        let result = c.as_f32_slice();
        assert_eq!(result, &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Tensor::from_f32_slice(&[2.0, 3.0, 4.0, 5.0], &[4]);
        let c = mul(&a, &b).unwrap();
        let result = c.as_f32_slice();
        assert_eq!(result, &[2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_scale() {
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = scale(&a, 2.5).unwrap();
        let result = b.as_f32_slice();
        assert_eq!(result, &[2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_silu() {
        let input = Tensor::from_f32_slice(&[0.0, 1.0, -1.0, 2.0], &[4]);
        let output = silu(&input).unwrap();
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
        let output = relu(&input).unwrap();
        let result = output.as_f32_slice();
        assert_eq!(result, &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_swiglu() {
        let gate = Tensor::from_f32_slice(&[0.0, 1.0, 2.0], &[3]);
        let up = Tensor::from_f32_slice(&[1.0, 2.0, 3.0], &[3]);
        let output = swiglu(&gate, &up).unwrap();
        let result = output.as_f32_slice();

        // swiglu(0, 1) = silu(0) * 1 = 0
        assert!((result[0] - 0.0).abs() < 1e-5);
        // swiglu(1, 2) = silu(1) * 2 ≈ 1.462
        assert!((result[1] - 1.4621172).abs() < 1e-5);
    }

    #[test]
    fn test_add3() {
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Tensor::from_f32_slice(&[5.0, 6.0, 7.0, 8.0], &[4]);
        let c = Tensor::from_f32_slice(&[10.0, 20.0, 30.0, 40.0], &[4]);
        let d = add3(&a, &b, &c).unwrap();
        let result = d.as_f32_slice();
        assert_eq!(result, &[16.0, 28.0, 40.0, 52.0]);
    }

    #[test]
    fn test_scale_inplace() {
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        scale_inplace(&a, 2.5).unwrap();
        let result = a.as_f32_slice();
        assert_eq!(result, &[2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_scale_tensors_inplace() {
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Tensor::from_f32_slice(&[10.0, 20.0], &[2]);
        scale_tensors_inplace(&[&a, &b], 0.5).unwrap();
        assert_eq!(a.as_f32_slice(), &[0.5, 1.0, 1.5, 2.0]);
        assert_eq!(b.as_f32_slice(), &[5.0, 10.0]);
    }

    #[test]
    fn test_scale_tensors_inplace_noop() {
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        // scale by 1.0 should be a no-op
        scale_tensors_inplace(&[&a], 1.0).unwrap();
        assert_eq!(a.as_f32_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
