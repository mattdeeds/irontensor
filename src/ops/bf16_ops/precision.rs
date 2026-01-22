//! GPU precision conversion between FP32 and BF16.

use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

use crate::device::MetalContext;
use crate::precision::Precision;
use crate::tensor::Tensor;

use super::pipelines::{create_buffer, get_pipelines};

/// Convert FP32 tensor to BF16 on GPU.
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

/// Convert BF16 tensor to FP32 on GPU.
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
