//! GPU trace capture for shader profiling.
//!
//! This module provides programmatic GPU trace capture using Metal's `MTLCaptureManager` API.
//! Captured traces can be opened in Xcode for detailed shader analysis, including:
//! - Execution time per kernel
//! - Memory access patterns
//! - Pipeline statistics and occupancy
//! - GPU timeline visualization
//!
//! # Usage
//!
//! ## Manual capture
//!
//! ```rust,no_run
//! use irontensor::{GpuTrace, CommandBatch};
//!
//! // Start capturing GPU commands
//! GpuTrace::start("/tmp/my_trace.gputrace").unwrap();
//!
//! // Perform GPU operations
//! CommandBatch::begin();
//! // ... tensor operations ...
//! CommandBatch::sync();
//!
//! // Stop capturing and finalize the trace file
//! GpuTrace::stop().unwrap();
//! // Open /tmp/my_trace.gputrace in Xcode
//! ```
//!
//! ## Block capture
//!
//! ```rust,no_run
//! use irontensor::GpuTrace;
//!
//! let result = GpuTrace::capture("/tmp/forward_pass.gputrace", || {
//!     // GPU operations are captured
//!     trainer.compute_loss(&input_ids, &target_ids, batch_size, seq_len)
//! }).unwrap();
//! ```
//!
//! # Environment Variables
//!
//! GPU trace capture can also be enabled via environment variables:
//!
//! - `METAL_CAPTURE_ENABLED=1` - **Required** by Metal for programmatic capture
//! - `IRONTENSOR_GPU_TRACE=1` - Enable GPU trace capture
//! - `IRONTENSOR_GPU_TRACE_DIR=./traces` - Output directory (default: current directory)
//! - `IRONTENSOR_GPU_TRACE_STEP=10` - Only capture step N (optional)
//!
//! Example:
//! ```bash
//! METAL_CAPTURE_ENABLED=1 IRONTENSOR_GPU_TRACE=1 IRONTENSOR_GPU_TRACE_STEP=10 cargo run --release
//! ```

use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use objc2::rc::Retained;
use objc2_foundation::NSURL;
use objc2_metal::{
    MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureError, MTLCaptureManager,
};

use crate::device::MetalContext;

/// Error type for GPU trace operations.
#[derive(Debug, Clone)]
pub enum GpuTraceError {
    /// GPU trace capture is not supported on this system.
    NotSupported,
    /// A capture is already in progress.
    AlreadyCapturing,
    /// The capture descriptor contains invalid parameters.
    InvalidDescriptor(String),
    /// The output path could not be converted to a URL.
    InvalidPath(String),
    /// No capture is currently in progress.
    NotCapturing,
    /// An unexpected error occurred.
    Other(String),
}

impl fmt::Display for GpuTraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuTraceError::NotSupported => {
                write!(f, "GPU trace capture is not supported on this system")
            }
            GpuTraceError::AlreadyCapturing => {
                write!(f, "A capture is already in progress")
            }
            GpuTraceError::InvalidDescriptor(msg) => {
                write!(f, "Invalid capture descriptor: {}", msg)
            }
            GpuTraceError::InvalidPath(path) => {
                write!(f, "Invalid output path: {}", path)
            }
            GpuTraceError::NotCapturing => {
                write!(f, "No capture is currently in progress")
            }
            GpuTraceError::Other(msg) => {
                write!(f, "GPU trace error: {}", msg)
            }
        }
    }
}

impl std::error::Error for GpuTraceError {}

/// Configuration for GPU trace capture.
#[derive(Debug, Clone)]
pub struct GpuTraceConfig {
    /// Output directory for .gputrace files.
    pub output_dir: String,
    /// Whether to automatically timestamp filenames.
    pub timestamp_files: bool,
}

impl Default for GpuTraceConfig {
    fn default() -> Self {
        Self {
            output_dir: ".".to_string(),
            timestamp_files: true,
        }
    }
}

impl GpuTraceConfig {
    /// Create configuration from environment variables.
    ///
    /// Reads:
    /// - `IRONTENSOR_GPU_TRACE_DIR` - Output directory (default: ".")
    pub fn from_env() -> Self {
        let output_dir = std::env::var("IRONTENSOR_GPU_TRACE_DIR").unwrap_or_else(|_| ".".to_string());
        Self {
            output_dir,
            timestamp_files: true,
        }
    }

    /// Check if GPU tracing is enabled via environment variable.
    pub fn is_enabled_via_env() -> bool {
        std::env::var("IRONTENSOR_GPU_TRACE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false)
    }

    /// Get the step to capture from environment variable.
    ///
    /// Returns Some(step) if `IRONTENSOR_GPU_TRACE_STEP` is set, None otherwise.
    pub fn capture_step_from_env() -> Option<usize> {
        std::env::var("IRONTENSOR_GPU_TRACE_STEP")
            .ok()
            .and_then(|v| v.parse().ok())
    }
}

/// Track whether a capture is in progress.
static CAPTURE_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// GPU trace capture for shader profiling.
///
/// Provides programmatic control over Metal's GPU trace capture feature.
/// Captured traces can be opened in Xcode for detailed shader analysis.
pub struct GpuTrace;

impl GpuTrace {
    /// Check if GPU trace capture is supported on this system.
    ///
    /// Returns true if the system supports capturing to GPU trace documents.
    pub fn is_supported() -> bool {
        let manager = Self::capture_manager();
        manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument)
    }

    /// Check if a capture is currently in progress.
    pub fn is_capturing() -> bool {
        CAPTURE_IN_PROGRESS.load(Ordering::Acquire)
    }

    /// Start capturing GPU commands to a .gputrace file.
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path where the .gputrace file will be written.
    ///   The file must not exist (will fail if it does).
    ///
    /// # Returns
    ///
    /// Returns an error if:
    /// - GPU trace capture is not supported
    /// - A capture is already in progress
    /// - The output path is invalid
    /// - The output file already exists
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use irontensor::GpuTrace;
    ///
    /// GpuTrace::start("/tmp/my_trace.gputrace")?;
    /// // ... GPU operations ...
    /// GpuTrace::stop()?;
    /// ```
    pub fn start<P: AsRef<Path>>(output_path: P) -> Result<(), GpuTraceError> {
        let path = output_path.as_ref();

        // Check if already capturing
        if CAPTURE_IN_PROGRESS.load(Ordering::Acquire) {
            return Err(GpuTraceError::AlreadyCapturing);
        }

        // Get capture manager
        let manager = Self::capture_manager();

        // Check if supported
        if !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
            return Err(GpuTraceError::NotSupported);
        }

        // Create capture descriptor
        let descriptor = MTLCaptureDescriptor::new();

        // Set destination to GPU trace document
        descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);

        // Set output URL
        let url = Self::path_to_url(path)?;
        descriptor.setOutputURL(Some(&url));

        // Set capture object to the device (captures all command queues)
        let device = MetalContext::global().device();
        descriptor.set_capture_device(device);

        // Start capture
        match manager.startCaptureWithDescriptor_error(&descriptor) {
            Ok(()) => {
                CAPTURE_IN_PROGRESS.store(true, Ordering::Release);
                Ok(())
            }
            Err(ns_error) => {
                // Map NSError to our error type
                let description = ns_error.localizedDescription().to_string();

                // Check the error code
                let code = ns_error.code();
                if code == MTLCaptureError::NotSupported.0 {
                    Err(GpuTraceError::NotSupported)
                } else if code == MTLCaptureError::AlreadyCapturing.0 {
                    Err(GpuTraceError::AlreadyCapturing)
                } else if code == MTLCaptureError::InvalidDescriptor.0 {
                    Err(GpuTraceError::InvalidDescriptor(description))
                } else {
                    Err(GpuTraceError::Other(description))
                }
            }
        }
    }

    /// Stop the current capture session.
    ///
    /// The .gputrace file is finalized and can be opened in Xcode.
    ///
    /// # Returns
    ///
    /// Returns an error if no capture is currently in progress.
    pub fn stop() -> Result<(), GpuTraceError> {
        if !CAPTURE_IN_PROGRESS.load(Ordering::Acquire) {
            return Err(GpuTraceError::NotCapturing);
        }

        let manager = Self::capture_manager();
        manager.stopCapture();
        CAPTURE_IN_PROGRESS.store(false, Ordering::Release);

        Ok(())
    }

    /// Capture a single operation or block of operations.
    ///
    /// This is a convenience method that starts capture, executes the closure,
    /// and stops capture, returning the result of the closure.
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path where the .gputrace file will be written
    /// * `f` - Closure containing GPU operations to capture
    ///
    /// # Returns
    ///
    /// Returns the result of the closure, or an error if capture fails.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use irontensor::GpuTrace;
    ///
    /// let loss = GpuTrace::capture("/tmp/forward.gputrace", || {
    ///     trainer.compute_loss(&input_ids, &target_ids, batch_size, seq_len)
    /// })?;
    /// ```
    pub fn capture<P, F, R>(output_path: P, f: F) -> Result<R, GpuTraceError>
    where
        P: AsRef<Path>,
        F: FnOnce() -> R,
    {
        Self::start(output_path)?;
        let result = f();
        Self::stop()?;
        Ok(result)
    }

    /// Get the shared capture manager.
    fn capture_manager() -> Retained<MTLCaptureManager> {
        // SAFETY: sharedCaptureManager returns a singleton that is always valid
        unsafe { MTLCaptureManager::sharedCaptureManager() }
    }

    /// Convert a Path to an NSURL.
    fn path_to_url(path: &Path) -> Result<Retained<NSURL>, GpuTraceError> {
        // Get absolute path
        let absolute_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|e| GpuTraceError::InvalidPath(e.to_string()))?
                .join(path)
        };

        // Convert to string
        let path_str = absolute_path
            .to_str()
            .ok_or_else(|| GpuTraceError::InvalidPath(format!("{:?}", path)))?;

        // Create NSURL using fileURLWithPath
        let ns_string = objc2_foundation::NSString::from_str(path_str);
        // SAFETY: fileURLWithPath is always safe to call with a valid NSString
        let url = NSURL::fileURLWithPath(&ns_string);

        Ok(url)
    }

    /// Generate a timestamped output path.
    ///
    /// # Arguments
    ///
    /// * `base_dir` - Directory for the output file
    /// * `prefix` - Prefix for the filename (e.g., "train_step")
    ///
    /// # Returns
    ///
    /// A path like `{base_dir}/{prefix}_20240115_143022.gputrace`
    pub fn timestamped_path(base_dir: &str, prefix: &str) -> String {
        use std::time::SystemTime;

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = now.as_secs();

        // Simple timestamp formatting
        let hours = (secs / 3600) % 24;
        let minutes = (secs / 60) % 60;
        let seconds = secs % 60;
        let days = secs / 86400;

        format!(
            "{}/{}_{:05}_{:02}{:02}{:02}.gputrace",
            base_dir, prefix, days, hours, minutes, seconds
        )
    }
}

/// RAII guard for GPU trace capture.
///
/// Automatically stops capture when dropped, ensuring the trace file is finalized
/// even if an error occurs during the captured operations.
pub struct GpuTraceGuard {
    active: bool,
}

impl GpuTraceGuard {
    /// Start a new GPU trace capture that will be stopped when the guard is dropped.
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path where the .gputrace file will be written
    ///
    /// # Returns
    ///
    /// A guard that stops capture when dropped, or an error if capture fails to start.
    pub fn start<P: AsRef<Path>>(output_path: P) -> Result<Self, GpuTraceError> {
        GpuTrace::start(output_path)?;
        Ok(Self { active: true })
    }

    /// Stop capture early (before the guard is dropped).
    ///
    /// This is useful if you want to handle stop errors explicitly.
    pub fn stop(mut self) -> Result<(), GpuTraceError> {
        self.active = false;
        GpuTrace::stop()
    }
}

impl Drop for GpuTraceGuard {
    fn drop(&mut self) {
        if self.active {
            // Best effort - ignore errors on drop
            let _ = GpuTrace::stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_supported() {
        // Just check that the function doesn't panic
        let supported = GpuTrace::is_supported();
        println!("GPU trace capture supported: {}", supported);
    }

    #[test]
    fn test_not_capturing_initially() {
        assert!(!GpuTrace::is_capturing());
    }

    #[test]
    fn test_stop_without_start_fails() {
        let result = GpuTrace::stop();
        assert!(matches!(result, Err(GpuTraceError::NotCapturing)));
    }

    #[test]
    fn test_timestamped_path() {
        let path = GpuTrace::timestamped_path("/tmp", "test");
        assert!(path.starts_with("/tmp/test_"));
        assert!(path.ends_with(".gputrace"));
    }

    #[test]
    fn test_config_defaults() {
        let config = GpuTraceConfig::default();
        assert_eq!(config.output_dir, ".");
        assert!(config.timestamp_files);
    }

    #[test]
    #[ignore] // Requires GPU and may conflict with other tests
    fn test_gpu_trace_capture() {
        use crate::precision::Precision;
        use crate::tensor::Tensor;

        let path = "/tmp/irontensor_test_capture.gputrace";

        // Clean up any existing file
        let _ = std::fs::remove_file(path);
        // Also remove any directory with the same name (Metal sometimes creates a bundle)
        let _ = std::fs::remove_dir_all(path);

        if !GpuTrace::is_supported() {
            println!("GPU trace capture not supported, skipping test");
            return;
        }

        assert!(!GpuTrace::is_capturing());

        // Start capture
        GpuTrace::start(path).expect("Failed to start capture");
        assert!(GpuTrace::is_capturing());

        // Do some GPU work using zeros (randn doesn't exist)
        let a = Tensor::zeros(&[64, 64], Precision::FP32);
        let b = Tensor::zeros(&[64, 64], Precision::FP32);
        let _ = crate::ops::matmul(&a, &b);

        // Stop capture
        GpuTrace::stop().expect("Failed to stop capture");
        assert!(!GpuTrace::is_capturing());

        // Check that the trace file was created
        assert!(
            std::path::Path::new(path).exists(),
            "Trace file should exist at {}",
            path
        );

        // Clean up
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_dir_all(path);
    }
}
