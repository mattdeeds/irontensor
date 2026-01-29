use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCaptureDestination, MTLCaptureManager, MTLCommandQueue, MTLCreateSystemDefaultDevice,
    MTLDevice,
};

pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

static GLOBAL_CONTEXT: OnceLock<MetalContext> = OnceLock::new();

impl MetalContext {
    fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");
        let command_queue = device.newCommandQueue().expect("Failed to create command queue");
        Self {
            device,
            command_queue,
        }
    }

    pub fn global() -> &'static MetalContext {
        GLOBAL_CONTEXT.get_or_init(MetalContext::new)
    }

    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    pub fn command_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.command_queue
    }

    /// Returns the current total GPU memory allocated by Metal (in bytes).
    ///
    /// This is useful for profiling memory usage during training.
    /// On Apple Silicon with unified memory, this tracks all Metal buffer allocations.
    pub fn current_allocated_size(&self) -> usize {
        self.device.currentAllocatedSize()
    }

    /// Get the shared capture manager singleton.
    ///
    /// The capture manager allows programmatic control over GPU trace capture
    /// for shader profiling in Xcode.
    pub fn capture_manager(&self) -> Retained<MTLCaptureManager> {
        // SAFETY: sharedCaptureManager returns a singleton that is always valid
        unsafe { MTLCaptureManager::sharedCaptureManager() }
    }

    /// Check if GPU trace capture to file is supported on this system.
    ///
    /// Returns true if the system supports capturing to GPU trace documents
    /// that can be opened in Xcode.
    pub fn supports_gpu_trace(&self) -> bool {
        self.capture_manager()
            .supportsDestination(MTLCaptureDestination::GPUTraceDocument)
    }
}

/// Returns the current total GPU memory allocated by Metal (in bytes).
///
/// Convenience function that calls `MetalContext::global().current_allocated_size()`.
pub fn gpu_memory_allocated() -> usize {
    MetalContext::global().current_allocated_size()
}

/// Format bytes as human-readable string (e.g., "123.4 MB")
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * 1024;
    const GB: usize = 1024 * 1024 * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
