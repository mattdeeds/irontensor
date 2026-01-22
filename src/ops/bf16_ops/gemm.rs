//! BF16 matrix multiplication operations.

use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

use crate::device::MetalContext;
use crate::precision::Precision;
use crate::tensor::Tensor;

use super::pipelines::{create_buffer, get_pipelines, TILE_SIZE};

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
