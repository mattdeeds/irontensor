//! GPU trace capture for debugging and profiling.

use super::Trainer;

impl Trainer {
    /// Check if GPU trace capture should be enabled for the next step.
    ///
    /// Returns true if:
    /// - GPU trace is configured (gpu_trace_config is Some)
    /// - GPU trace is supported on this system
    /// - The next step (self.step + 1) matches the configured capture step
    ///
    /// Note: This is called *before* train_step(), and self.step is incremented
    /// at the *end* of train_step(). So we check for self.step + 1.
    pub fn should_capture_gpu_trace(&self) -> bool {
        if self.config.gpu_trace_config.is_none() {
            return false;
        }

        // Check if supported
        if !crate::gpu_trace::GpuTrace::is_supported() {
            return false;
        }

        // Check if we should capture the next step
        // self.step is the *current* step count, and train_step will increment it
        // So to capture "step 10", we check when self.step == 9 (before increment)
        match self.config.gpu_trace_step {
            Some(target_step) => self.step + 1 == target_step,
            None => false, // Don't capture every step by default
        }
    }

    /// Execute a training step with GPU trace capture.
    ///
    /// Wraps the normal train_step with GPU trace capture, saving the trace
    /// to the configured output directory.
    pub fn train_step_with_gpu_trace(
        &mut self,
        input_ids: &[u32],
        target_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> (f32, f32) {
        use crate::gpu_trace::GpuTrace;

        let config = self.config.gpu_trace_config.as_ref().unwrap();
        // Use step + 1 because this is called before train_step increments self.step
        let step_num = self.step + 1;
        let output_path = format!(
            "{}/train_step_{}.gputrace",
            config.output_dir, step_num
        );

        // Ensure output directory exists
        if let Err(e) = std::fs::create_dir_all(&config.output_dir) {
            eprintln!(
                "Warning: Failed to create GPU trace directory '{}': {}",
                config.output_dir, e
            );
            // Fall back to normal train_step
            return self.train_step(input_ids, target_ids, batch_size, seq_len);
        }

        // Remove existing file if it exists (capture fails if file exists)
        let _ = std::fs::remove_file(&output_path);
        let _ = std::fs::remove_dir_all(&output_path);

        println!("Capturing GPU trace: {}", output_path);

        match GpuTrace::start(&output_path) {
            Ok(()) => {
                let result = self.train_step(input_ids, target_ids, batch_size, seq_len);

                if let Err(e) = GpuTrace::stop() {
                    eprintln!("Warning: Failed to stop GPU trace capture: {}", e);
                } else {
                    println!("GPU trace saved: {}", output_path);
                }

                result
            }
            Err(e) => {
                eprintln!("Warning: Failed to start GPU trace capture: {}", e);
                // Fall back to normal train_step
                self.train_step(input_ids, target_ids, batch_size, seq_len)
            }
        }
    }
}
