//! BF16 RMSNorm operation.

use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

use crate::device::MetalContext;
use crate::precision::Precision;
use crate::tensor::Tensor;

use super::pipelines::{create_buffer, get_pipelines};

#[repr(C)]
struct RMSNormParams {
    batch_size: u32,
    hidden_dim: u32,
    eps: f32,
}

/// BF16 RMSNorm.
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
