//! BF16 (Brain Float 16) operations for mixed precision training
//!
//! These operations read/write BF16 tensors but compute in FP32 for numerical stability.
//! This provides ~2x memory savings compared to FP32 while maintaining training quality.

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

const BF16_SHADER: &str = include_str!("../shaders/bf16_ops.metal");
const TILE_SIZE: usize = 16;

// ============================================================================
// Pipeline Management
// ============================================================================

struct BF16Pipelines {
    gemm_tiled: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    gemm_batched: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    mul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    scale: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    silu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    swiglu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rmsnorm_fast: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    f32_to_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    bf16_to_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static BF16_PIPELINES: OnceLock<BF16Pipelines> = OnceLock::new();

fn get_pipelines() -> &'static BF16Pipelines {
    BF16_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(BF16_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile BF16 shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        BF16Pipelines {
            gemm_tiled: make_pipeline("gemm_bf16_tiled"),
            gemm_batched: make_pipeline("gemm_bf16_batched"),
            add: make_pipeline("add_bf16"),
            mul: make_pipeline("mul_bf16"),
            scale: make_pipeline("scale_bf16"),
            silu: make_pipeline("silu_bf16"),
            swiglu: make_pipeline("swiglu_bf16"),
            rmsnorm_fast: make_pipeline("rmsnorm_bf16_fast"),
            softmax: make_pipeline("softmax_bf16"),
            f32_to_bf16: make_pipeline("f32_to_bf16"),
            bf16_to_f32: make_pipeline("bf16_to_f32"),
        }
    })
}

// ============================================================================
// Helper: Create buffer from data
// ============================================================================

fn create_buffer<T>(ctx: &MetalContext, data: &T) -> Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>> {
    unsafe {
        ctx.device()
            .newBufferWithBytes_length_options(
                NonNull::new(data as *const T as *mut _).unwrap(),
                std::mem::size_of::<T>(),
                MTLResourceOptions::StorageModeShared,
            )
    }
    .expect("Failed to create buffer")
}

// ============================================================================
// BF16 Matrix Multiplication
// ============================================================================

#[repr(C)]
struct GemmParams {
    m: u32,
    n: u32,
    k: u32,
}

/// BF16 matrix multiplication: C = A @ B
///
/// A: [M, K], B: [K, N] -> C: [M, N]
/// Computes in FP32, stores result in BF16.
pub fn matmul_bf16(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.precision(), Precision::BF16, "matmul_bf16 requires BF16 input A");
    assert_eq!(b.precision(), Precision::BF16, "matmul_bf16 requires BF16 input B");

    let a_shape = a.shape();
    let b_shape = b.shape();

    assert_eq!(a_shape.len(), 2, "matmul_bf16 requires 2D tensors");
    assert_eq!(b_shape.len(), 2, "matmul_bf16 requires 2D tensors");
    assert_eq!(a_shape[1], b_shape[0], "Inner dimensions must match");

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    let output = Tensor::zeros(&[m, n], Precision::BF16);

    let pipelines = get_pipelines();
    let ctx = MetalContext::global();

    let params = GemmParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let params_buffer = create_buffer(ctx, &params);

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create encoder");

    encoder.setComputePipelineState(&pipelines.gemm_tiled);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
    }

    let grid_size = MTLSize {
        width: (n + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
        height: (m + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: TILE_SIZE,
        height: TILE_SIZE,
        depth: 1,
    };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

/// BF16 batched matrix multiplication: C[b] = A[b] @ B[b]
///
/// A: [batch, M, K], B: [batch, K, N] -> C: [batch, M, N]
pub fn matmul_bf16_batched(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.precision(), Precision::BF16);
    assert_eq!(b.precision(), Precision::BF16);

    let a_shape = a.shape();
    let b_shape = b.shape();

    assert_eq!(a_shape.len(), 3, "batched matmul requires 3D tensors");
    assert_eq!(b_shape.len(), 3, "batched matmul requires 3D tensors");
    assert_eq!(a_shape[0], b_shape[0], "Batch dimensions must match");
    assert_eq!(a_shape[2], b_shape[1], "Inner dimensions must match");

    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];

    let output = Tensor::zeros(&[batch, m, n], Precision::BF16);

    let pipelines = get_pipelines();
    let ctx = MetalContext::global();

    let params = GemmParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let params_buffer = create_buffer(ctx, &params);
    let batch_val = batch as u32;
    let batch_buffer = create_buffer(ctx, &batch_val);

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create encoder");

    encoder.setComputePipelineState(&pipelines.gemm_batched);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&batch_buffer), 0, 4);
    }

    let grid_size = MTLSize {
        width: n,
        height: m,
        depth: batch,
    };
    let max_threads = pipelines.gemm_batched.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize {
        width: max_threads.min(n).max(1),
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

// ============================================================================
// BF16 Element-wise Operations
// ============================================================================

/// BF16 element-wise addition
pub fn add_bf16(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.precision(), Precision::BF16);
    assert_eq!(b.precision(), Precision::BF16);
    assert_eq!(a.shape(), b.shape(), "Shapes must match for add");

    let output = Tensor::zeros(a.shape(), Precision::BF16);
    let numel = a.numel() as u32;

    let pipelines = get_pipelines();
    let ctx = MetalContext::global();
    let numel_buffer = create_buffer(ctx, &numel);

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create encoder");

    encoder.setComputePipelineState(&pipelines.add);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&numel_buffer), 0, 3);
    }

    let grid_size = MTLSize { width: numel as usize, height: 1, depth: 1 };
    let max_threads = pipelines.add.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize { width: max_threads.min(256), height: 1, depth: 1 };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

/// BF16 element-wise multiplication
pub fn mul_bf16(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.precision(), Precision::BF16);
    assert_eq!(b.precision(), Precision::BF16);
    assert_eq!(a.shape(), b.shape(), "Shapes must match for mul");

    let output = Tensor::zeros(a.shape(), Precision::BF16);
    let numel = a.numel() as u32;

    let pipelines = get_pipelines();
    let ctx = MetalContext::global();
    let numel_buffer = create_buffer(ctx, &numel);

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create encoder");

    encoder.setComputePipelineState(&pipelines.mul);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&numel_buffer), 0, 3);
    }

    let grid_size = MTLSize { width: numel as usize, height: 1, depth: 1 };
    let max_threads = pipelines.mul.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize { width: max_threads.min(256), height: 1, depth: 1 };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

/// BF16 scalar multiplication
pub fn scale_bf16(input: &Tensor, scalar: f32) -> Tensor {
    assert_eq!(input.precision(), Precision::BF16);

    let output = Tensor::zeros(input.shape(), Precision::BF16);
    let numel = input.numel() as u32;

    let pipelines = get_pipelines();
    let ctx = MetalContext::global();
    let scalar_buffer = create_buffer(ctx, &scalar);
    let numel_buffer = create_buffer(ctx, &numel);

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create encoder");

    encoder.setComputePipelineState(&pipelines.scale);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&scalar_buffer), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&numel_buffer), 0, 3);
    }

    let grid_size = MTLSize { width: numel as usize, height: 1, depth: 1 };
    let max_threads = pipelines.scale.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize { width: max_threads.min(256), height: 1, depth: 1 };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

/// BF16 SiLU activation
pub fn silu_bf16(input: &Tensor) -> Tensor {
    assert_eq!(input.precision(), Precision::BF16);

    let output = Tensor::zeros(input.shape(), Precision::BF16);
    let numel = input.numel() as u32;

    let pipelines = get_pipelines();
    let ctx = MetalContext::global();
    let numel_buffer = create_buffer(ctx, &numel);

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create encoder");

    encoder.setComputePipelineState(&pipelines.silu);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&numel_buffer), 0, 2);
    }

    let grid_size = MTLSize { width: numel as usize, height: 1, depth: 1 };
    let max_threads = pipelines.silu.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize { width: max_threads.min(256), height: 1, depth: 1 };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

/// BF16 SwiGLU activation
pub fn swiglu_bf16(gate: &Tensor, up: &Tensor) -> Tensor {
    assert_eq!(gate.precision(), Precision::BF16);
    assert_eq!(up.precision(), Precision::BF16);
    assert_eq!(gate.shape(), up.shape());

    let output = Tensor::zeros(gate.shape(), Precision::BF16);
    let numel = gate.numel() as u32;

    let pipelines = get_pipelines();
    let ctx = MetalContext::global();
    let numel_buffer = create_buffer(ctx, &numel);

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create encoder");

    encoder.setComputePipelineState(&pipelines.swiglu);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(gate.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(up.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&numel_buffer), 0, 3);
    }

    let grid_size = MTLSize { width: numel as usize, height: 1, depth: 1 };
    let max_threads = pipelines.swiglu.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize { width: max_threads.min(256), height: 1, depth: 1 };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

// ============================================================================
// BF16 RMSNorm
// ============================================================================

#[repr(C)]
struct RMSNormParams {
    batch_size: u32,
    hidden_dim: u32,
    eps: f32,
}

/// BF16 RMSNorm
///
/// input: [batch, hidden_dim], gamma: [hidden_dim] -> output: [batch, hidden_dim]
pub fn rmsnorm_bf16(input: &Tensor, gamma: &Tensor, eps: f32) -> Tensor {
    assert_eq!(input.precision(), Precision::BF16);
    assert_eq!(gamma.precision(), Precision::BF16);
    assert_eq!(input.shape().len(), 2);
    assert_eq!(gamma.shape().len(), 1);
    assert_eq!(input.shape()[1], gamma.shape()[0]);

    let batch_size = input.shape()[0];
    let hidden_dim = input.shape()[1];

    let output = Tensor::zeros(input.shape(), Precision::BF16);

    let pipelines = get_pipelines();
    let ctx = MetalContext::global();

    let params = RMSNormParams {
        batch_size: batch_size as u32,
        hidden_dim: hidden_dim as u32,
        eps,
    };
    let params_buffer = create_buffer(ctx, &params);

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create encoder");

    let threads_per_group: u32 = 256.min(hidden_dim as u32);
    let tg_size_buffer = create_buffer(ctx, &threads_per_group);

    encoder.setComputePipelineState(&pipelines.rmsnorm_fast);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(gamma.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&tg_size_buffer), 0, 4);
    }

    // 1D dispatch: one threadgroup per batch element
    let threadgroups = MTLSize { width: batch_size, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: threads_per_group as usize, height: 1, depth: 1 };

    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threadgroup_size);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

// ============================================================================
// BF16 Softmax
// ============================================================================

#[repr(C)]
struct SoftmaxParams {
    batch_size: u32,
    seq_len: u32,
}

/// BF16 Softmax over last dimension
///
/// input: [batch, seq_len] -> output: [batch, seq_len]
pub fn softmax_bf16(input: &Tensor) -> Tensor {
    assert_eq!(input.precision(), Precision::BF16);
    assert_eq!(input.shape().len(), 2);

    let batch_size = input.shape()[0];
    let seq_len = input.shape()[1];

    let output = Tensor::zeros(input.shape(), Precision::BF16);

    let pipelines = get_pipelines();
    let ctx = MetalContext::global();

    let params = SoftmaxParams {
        batch_size: batch_size as u32,
        seq_len: seq_len as u32,
    };
    let params_buffer = create_buffer(ctx, &params);

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create encoder");

    encoder.setComputePipelineState(&pipelines.softmax);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 2);
    }

    let grid_size = MTLSize { width: batch_size, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: 1, height: 1, depth: 1 };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

// ============================================================================
// GPU Precision Conversion
// ============================================================================

/// Convert FP32 tensor to BF16 on GPU
pub fn to_bf16_gpu(input: &Tensor) -> Tensor {
    assert_eq!(input.precision(), Precision::FP32);

    let output = Tensor::zeros(input.shape(), Precision::BF16);
    let numel = input.numel() as u32;

    let pipelines = get_pipelines();
    let ctx = MetalContext::global();
    let numel_buffer = create_buffer(ctx, &numel);

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create encoder");

    encoder.setComputePipelineState(&pipelines.f32_to_bf16);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&numel_buffer), 0, 2);
    }

    let grid_size = MTLSize { width: numel as usize, height: 1, depth: 1 };
    let max_threads = pipelines.f32_to_bf16.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize { width: max_threads.min(256), height: 1, depth: 1 };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

/// Convert BF16 tensor to FP32 on GPU
pub fn to_f32_gpu(input: &Tensor) -> Tensor {
    assert_eq!(input.precision(), Precision::BF16);

    let output = Tensor::zeros(input.shape(), Precision::FP32);
    let numel = input.numel() as u32;

    let pipelines = get_pipelines();
    let ctx = MetalContext::global();
    let numel_buffer = create_buffer(ctx, &numel);

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create encoder");

    encoder.setComputePipelineState(&pipelines.bf16_to_f32);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&numel_buffer), 0, 2);
    }

    let grid_size = MTLSize { width: numel as usize, height: 1, depth: 1 };
    let max_threads = pipelines.bf16_to_f32.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize { width: max_threads.min(256), height: 1, depth: 1 };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::bf16_to_f32;

    fn bf16_tensor_from_f32(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_f32_as_bf16(data, shape)
    }

    fn bf16_tensor_to_f32_vec(tensor: &Tensor) -> Vec<f32> {
        tensor.as_bf16_slice().iter().map(|&x| bf16_to_f32(x)).collect()
    }

    #[test]
    fn test_matmul_bf16() {
        let a = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let c = matmul_bf16(&a, &b);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.precision(), Precision::BF16);

        let result = bf16_tensor_to_f32_vec(&c);
        // [1,2,3] @ [1,2; 3,4; 5,6] = [22, 28; 49, 64]
        assert!((result[0] - 22.0).abs() < 0.5);
        assert!((result[1] - 28.0).abs() < 0.5);
        assert!((result[2] - 49.0).abs() < 0.5);
        assert!((result[3] - 64.0).abs() < 0.5);
    }

    #[test]
    fn test_add_bf16() {
        let a = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = bf16_tensor_from_f32(&[0.5, 1.0, 1.5, 2.0], &[4]);
        let c = add_bf16(&a, &b);

        let result = bf16_tensor_to_f32_vec(&c);
        assert!((result[0] - 1.5).abs() < 0.01);
        assert!((result[1] - 3.0).abs() < 0.01);
        assert!((result[2] - 4.5).abs() < 0.01);
        assert!((result[3] - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_mul_bf16() {
        let a = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = bf16_tensor_from_f32(&[2.0, 3.0, 4.0, 5.0], &[4]);
        let c = mul_bf16(&a, &b);

        let result = bf16_tensor_to_f32_vec(&c);
        assert!((result[0] - 2.0).abs() < 0.01);
        assert!((result[1] - 6.0).abs() < 0.01);
        assert!((result[2] - 12.0).abs() < 0.01);
        assert!((result[3] - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_scale_bf16() {
        let a = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let c = scale_bf16(&a, 2.5);

        let result = bf16_tensor_to_f32_vec(&c);
        assert!((result[0] - 2.5).abs() < 0.01);
        assert!((result[1] - 5.0).abs() < 0.01);
        assert!((result[2] - 7.5).abs() < 0.01);
        assert!((result[3] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_silu_bf16() {
        let a = bf16_tensor_from_f32(&[0.0, 1.0, 2.0, -1.0], &[4]);
        let c = silu_bf16(&a);

        let result = bf16_tensor_to_f32_vec(&c);
        // SiLU(0) = 0
        assert!(result[0].abs() < 0.01);
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.731
        assert!((result[1] - 0.731).abs() < 0.05);
        // SiLU(2) ≈ 1.762
        assert!((result[2] - 1.762).abs() < 0.05);
    }

    #[test]
    fn test_rmsnorm_bf16() {
        let input = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let gamma = bf16_tensor_from_f32(&[1.0, 1.0, 1.0, 1.0], &[4]);
        let output = rmsnorm_bf16(&input, &gamma, 1e-5);

        assert_eq!(output.shape(), &[2, 4]);
        assert_eq!(output.precision(), Precision::BF16);

        let result = bf16_tensor_to_f32_vec(&output);
        // Check that outputs are normalized (roughly mean of squares ≈ 1)
        let row0_sq_sum: f32 = result[0..4].iter().map(|x| x * x).sum();
        let row0_mean_sq = row0_sq_sum / 4.0;
        assert!((row0_mean_sq - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_softmax_bf16() {
        let input = bf16_tensor_from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let output = softmax_bf16(&input);

        let result = bf16_tensor_to_f32_vec(&output);
        // Check that sum is approximately 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
        // Check that values are in increasing order
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
        assert!(result[2] < result[3]);
    }

    #[test]
    fn test_gpu_precision_conversion() {
        let f32_tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let bf16_tensor = to_bf16_gpu(&f32_tensor);

        assert_eq!(bf16_tensor.precision(), Precision::BF16);

        let back_to_f32 = to_f32_gpu(&bf16_tensor);
        assert_eq!(back_to_f32.precision(), Precision::FP32);

        let result = back_to_f32.as_f32_slice();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
        assert!((result[2] - 3.0).abs() < 0.01);
        assert!((result[3] - 4.0).abs() < 0.01);
    }
}
