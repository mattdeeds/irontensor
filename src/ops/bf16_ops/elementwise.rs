//! BF16 element-wise operations.

use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

use crate::device::MetalContext;
use crate::precision::Precision;
use crate::tensor::Tensor;

use super::pipelines::{create_buffer, get_pipelines};

/// BF16 element-wise addition.
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

/// BF16 element-wise multiplication.
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

/// BF16 scalar multiplication.
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

/// BF16 SiLU activation.
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

/// BF16 SwiGLU activation.
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
