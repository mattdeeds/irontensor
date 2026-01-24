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
//!
//! # Async Mode (Phase 6)
//! ```ignore
//! // For CPU/GPU overlap, use async commit:
//! CommandBatch::begin();
//! // ... GPU operations ...
//! CommandBatch::commit_async(); // Returns immediately
//! // ... CPU work (prepare next batch) ...
//! CommandBatch::wait_for_completion(); // Wait before reading results
//! ```

use std::cell::RefCell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

use crate::device::MetalContext;
use crate::profile::OpCategory;

thread_local! {
    static BATCH: RefCell<Option<BatchState>> = const { RefCell::new(None) };
    /// Tracks the last async command buffer for wait_for_completion
    static PENDING_BUFFER: RefCell<Option<PendingBuffer>> = const { RefCell::new(None) };
}

/// Tracks an asynchronously submitted command buffer
struct PendingBuffer {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    completed: Arc<AtomicBool>,
    /// Keep the completion block alive until command buffer completes
    _block: RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLCommandBuffer>>)>,
}

impl PendingBuffer {
    /// Wait for the command buffer to complete
    fn wait(&self) {
        // First check the atomic flag (fast path)
        if self.completed.load(Ordering::Acquire) {
            return;
        }
        // Fall back to blocking wait
        self.command_buffer.waitUntilCompleted();
    }

    /// Check if the command buffer has completed without blocking
    fn is_completed(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }
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

        // Time the GPU wait to identify sync overhead
        let start = Instant::now();
        self.command_buffer.waitUntilCompleted();
        let wait_duration = start.elapsed();

        // Record sync wait time if profiling is enabled
        // Note: We use manual recording here because we're inside a RefCell borrow
        // and timed() would try to borrow PROFILER which may cause issues
        crate::profile::PROFILER.with(|p| {
            if let Some(ref mut profiler) = *p.borrow_mut() {
                profiler.record(OpCategory::SyncWait, wait_duration, self.op_count);
            }
        });
    }

    /// Commit the command buffer asynchronously and return a completion tracker
    fn commit_async(mut self) -> PendingBuffer {
        self.end_current_encoder();

        let completed = Arc::new(AtomicBool::new(false));
        let completed_clone = completed.clone();

        // Create an RcBlock that takes NonNull<ProtocolObject<dyn MTLCommandBuffer>>
        // SAFETY: The block is invoked on a Metal internal thread when the command buffer completes
        let block = RcBlock::new(move |_: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            completed_clone.store(true, Ordering::Release);
        });

        unsafe {
            // Convert RcBlock to the raw pointer expected by addCompletedHandler
            self.command_buffer
                .addCompletedHandler(&*block as *const _ as *mut _);
        }
        self.command_buffer.commit();

        PendingBuffer {
            command_buffer: self.command_buffer,
            completed,
            // Keep the block alive until command buffer completes
            _block: block,
        }
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

    /// Commit the current batch asynchronously without waiting.
    ///
    /// This allows CPU work to proceed while the GPU executes.
    /// Call `wait_for_completion()` before reading any results.
    ///
    /// After calling this, a new batch is automatically started for subsequent operations.
    pub fn commit_async() {
        BATCH.with(|b| {
            let pending = {
                let mut batch_ref = b.borrow_mut();
                batch_ref.take().map(|batch| batch.commit_async())
            };

            // Store the pending buffer for later wait
            if let Some(pending) = pending {
                PENDING_BUFFER.with(|p| {
                    // Wait for any previous pending buffer first
                    if let Some(old) = p.borrow_mut().take() {
                        old.wait();
                    }
                    *p.borrow_mut() = Some(pending);
                });
            }

            // Start a new batch for subsequent operations
            *b.borrow_mut() = Some(BatchState::new());
        });
    }

    /// Wait for the last async-committed batch to complete.
    ///
    /// This must be called before reading any GPU buffer contents after `commit_async()`.
    pub fn wait_for_completion() {
        PENDING_BUFFER.with(|p| {
            if let Some(pending) = p.borrow().as_ref() {
                pending.wait();
            }
        });
    }

    /// Check if the last async-committed batch has completed.
    ///
    /// Returns true if there's no pending async work or if it has completed.
    pub fn is_async_complete() -> bool {
        PENDING_BUFFER.with(|p| {
            p.borrow()
                .as_ref()
                .map(|pending| pending.is_completed())
                .unwrap_or(true)
        })
    }

    /// End async mode and clean up pending buffer.
    ///
    /// Waits for any pending async work and clears the pending buffer.
    pub fn end_async() {
        PENDING_BUFFER.with(|p| {
            if let Some(pending) = p.borrow_mut().take() {
                pending.wait();
            }
        });
        Self::end();
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
        let c = add(&a, &b).unwrap();
        let result = c.as_f32_slice();
        assert_eq!(result, &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_batched_mode_works() {
        CommandBatch::begin();

        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Tensor::from_f32_slice(&[5.0, 6.0, 7.0, 8.0], &[4]);
        let c = add(&a, &b).unwrap();

        // Must sync before reading results
        CommandBatch::sync();

        let result = c.as_f32_slice();
        assert_eq!(result, &[6.0, 8.0, 10.0, 12.0]);

        CommandBatch::end();
    }

    #[test]
    fn test_async_commit() {
        CommandBatch::begin();

        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Tensor::from_f32_slice(&[5.0, 6.0, 7.0, 8.0], &[4]);
        let c = add(&a, &b).unwrap();

        // Commit async - returns immediately
        CommandBatch::commit_async();

        // Do some "CPU work" while GPU executes
        let _cpu_result: i32 = (0..100).sum();

        // Wait before reading results
        CommandBatch::wait_for_completion();

        let result = c.as_f32_slice();
        assert_eq!(result, &[6.0, 8.0, 10.0, 12.0]);

        CommandBatch::end_async();
    }

    #[test]
    fn test_is_async_complete() {
        CommandBatch::begin();

        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Tensor::from_f32_slice(&[5.0, 6.0, 7.0, 8.0], &[4]);
        let _c = add(&a, &b).unwrap();

        CommandBatch::commit_async();

        // Poll for completion
        while !CommandBatch::is_async_complete() {
            std::thread::yield_now();
        }

        assert!(CommandBatch::is_async_complete());
        CommandBatch::end_async();
    }

    #[test]
    fn test_multiple_async_batches() {
        // First batch
        CommandBatch::begin();
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = Tensor::from_f32_slice(&[5.0, 6.0, 7.0, 8.0], &[4]);
        let c1 = add(&a, &b).unwrap();
        CommandBatch::commit_async();

        // Second batch (starts immediately, waits for first internally)
        let d = Tensor::from_f32_slice(&[10.0, 20.0, 30.0, 40.0], &[4]);
        let e = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let c2 = add(&d, &e).unwrap();
        CommandBatch::commit_async();

        // Wait for all
        CommandBatch::wait_for_completion();

        let r1 = c1.as_f32_slice();
        let r2 = c2.as_f32_slice();
        assert_eq!(r1, &[6.0, 8.0, 10.0, 12.0]);
        assert_eq!(r2, &[11.0, 22.0, 33.0, 44.0]);

        CommandBatch::end_async();
    }
}
