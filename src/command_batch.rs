//! Command buffer batching for reduced GPU synchronization overhead.
//!
//! Instead of creating a new command buffer and waiting after each operation,
//! this module allows batching multiple compute operations into a single
//! command buffer that is only committed at explicit sync points.
//!
//! # Usage
//! ```ignore
//! // Start a batch (operations will be accumulated)
//! CommandBatch::begin();
//!
//! // ... perform multiple GPU operations ...
//! // Each op uses CommandBatch::with_encoder() instead of creating its own buffer
//!
//! // Commit and wait for all accumulated operations
//! CommandBatch::sync();
//! ```

use std::cell::RefCell;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

use crate::device::MetalContext;

thread_local! {
    static BATCH: RefCell<Option<BatchState>> = const { RefCell::new(None) };
}

struct BatchState {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    op_count: usize,
}

impl BatchState {
    fn new() -> Self {
        let ctx = MetalContext::global();
        let command_buffer = ctx
            .command_queue()
            .commandBuffer()
            .expect("Failed to create command buffer");

        Self {
            command_buffer,
            encoder: None,
            op_count: 0,
        }
    }

    fn get_encoder(&mut self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
        if self.encoder.is_none() {
            self.encoder = Some(
                self.command_buffer
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder"),
            );
        }
        self.encoder.as_ref().unwrap()
    }

    fn end_current_encoder(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
    }

    fn commit_and_wait(&mut self) {
        self.end_current_encoder();
        self.command_buffer.commit();
        self.command_buffer.waitUntilCompleted();
    }
}

/// Command batch manager for reducing GPU synchronization overhead.
pub struct CommandBatch;

impl CommandBatch {
    /// Begin a new command batch.
    ///
    /// All subsequent GPU operations will be accumulated into a single
    /// command buffer until `sync()` is called.
    pub fn begin() {
        BATCH.with(|b| {
            let mut batch = b.borrow_mut();
            if batch.is_none() {
                *batch = Some(BatchState::new());
            }
        });
    }

    /// Check if batching is currently active.
    pub fn is_active() -> bool {
        BATCH.with(|b| b.borrow().is_some())
    }

    /// Synchronize: commit the current batch and wait for completion.
    ///
    /// This must be called before reading any GPU buffer contents.
    pub fn sync() {
        BATCH.with(|b| {
            let needs_reset = {
                let mut batch_ref = b.borrow_mut();
                if let Some(ref mut batch) = *batch_ref {
                    batch.commit_and_wait();
                    true
                } else {
                    false
                }
            };
            // Reset for next batch if we were active
            if needs_reset {
                *b.borrow_mut() = Some(BatchState::new());
            }
        });
    }

    /// End batching mode entirely.
    pub fn end() {
        BATCH.with(|b| {
            {
                let mut batch_ref = b.borrow_mut();
                if let Some(ref mut batch) = *batch_ref {
                    batch.commit_and_wait();
                }
            }
            *b.borrow_mut() = None;
        });
    }

    /// Execute a compute operation, using the batch if active.
    ///
    /// If batching is active, the operation is added to the current batch.
    /// If batching is not active, the operation executes immediately with its own buffer.
    ///
    /// # Arguments
    /// * `pipeline` - The compute pipeline to use
    /// * `setup` - Closure that sets up the encoder (buffers, etc.)
    /// * `grid_size` - The dispatch grid size
    /// * `threadgroup_size` - The threadgroup size
    pub fn dispatch<F>(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        setup: F,
        grid_size: MTLSize,
        threadgroup_size: MTLSize,
    ) where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    {
        BATCH.with(|b| {
            let mut batch_ref = b.borrow_mut();

            if let Some(ref mut batch) = *batch_ref {
                // Batched mode: use shared encoder
                let encoder = batch.get_encoder();
                encoder.setComputePipelineState(pipeline);
                setup(encoder);
                encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
                batch.op_count += 1;
            } else {
                // Immediate mode: create dedicated command buffer
                drop(batch_ref); // Release borrow before calling MetalContext

                let ctx = MetalContext::global();
                let command_buffer = ctx
                    .command_queue()
                    .commandBuffer()
                    .expect("Failed to create command buffer");
                let encoder = command_buffer
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");

                encoder.setComputePipelineState(pipeline);
                setup(&encoder);
                encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

                encoder.endEncoding();
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
            }
        });
    }

    /// Execute a compute operation using threadgroups dispatch (for kernels using threadgroup memory).
    pub fn dispatch_threadgroups<F>(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        setup: F,
        threadgroup_count: MTLSize,
        threadgroup_size: MTLSize,
    ) where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    {
        BATCH.with(|b| {
            let mut batch_ref = b.borrow_mut();

            if let Some(ref mut batch) = *batch_ref {
                // Batched mode
                let encoder = batch.get_encoder();
                encoder.setComputePipelineState(pipeline);
                setup(encoder);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroup_count, threadgroup_size);
                batch.op_count += 1;
            } else {
                // Immediate mode
                drop(batch_ref);

                let ctx = MetalContext::global();
                let command_buffer = ctx
                    .command_queue()
                    .commandBuffer()
                    .expect("Failed to create command buffer");
                let encoder = command_buffer
                    .computeCommandEncoder()
                    .expect("Failed to create compute encoder");

                encoder.setComputePipelineState(pipeline);
                setup(&encoder);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroup_count, threadgroup_size);

                encoder.endEncoding();
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
            }
        });
    }

    /// Get the number of operations in the current batch.
    pub fn op_count() -> usize {
        BATCH.with(|b| {
            if let Some(ref batch) = *b.borrow() {
                batch.op_count
            } else {
                0
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::add;
    use crate::tensor::Tensor;

    #[test]
    fn test_batch_inactive_by_default() {
        assert!(!CommandBatch::is_active());
    }

    #[test]
    fn test_batch_begin_end() {
        assert!(!CommandBatch::is_active());
        CommandBatch::begin();
        assert!(CommandBatch::is_active());
        CommandBatch::end();
        assert!(!CommandBatch::is_active());
    }

    #[test]
    fn test_immediate_mode_works() {
        // Without batching, operations should still work
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Tensor::from_f32_slice(&[5.0, 6.0, 7.0, 8.0], &[4]);
        let c = add(&a, &b);
        let result = c.as_f32_slice();
        assert_eq!(result, &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_batched_mode_works() {
        CommandBatch::begin();

        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Tensor::from_f32_slice(&[5.0, 6.0, 7.0, 8.0], &[4]);
        let c = add(&a, &b);

        // Must sync before reading results
        CommandBatch::sync();

        let result = c.as_f32_slice();
        assert_eq!(result, &[6.0, 8.0, 10.0, 12.0]);

        CommandBatch::end();
    }
}
