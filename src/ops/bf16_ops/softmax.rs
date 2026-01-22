//! BF16 Softmax operation.

use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

use crate::device::MetalContext;
use crate::precision::Precision;
use crate::tensor::Tensor;

use super::pipelines::{create_buffer, get_pipelines};

#[repr(C)]
struct SoftmaxParams {
    batch_size: u32,
    seq_len: u32,
}

/// BF16 Softmax over last dimension.
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
