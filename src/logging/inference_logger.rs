//! Inference timing and logging helpers.

use std::time::Instant;

use super::formats::InferenceRecord;
use super::Logger;

/// RAII helper for timing inference/text generation.
///
/// Tracks time to first token (TTFT), total time, and tokens generated.
///
/// # Example
///
/// ```ignore
/// let mut timer = InferenceTimer::new(prompt_tokens, temperature);
///
/// for i in 0..max_tokens {
///     let next_token = sample_token(...);
///     if i == 0 {
///         timer.mark_first_token();
///     } else {
///         timer.token_generated();
///     }
/// }
///
/// let record = timer.finish();
/// Logger::log_inference(&record);
/// ```
pub struct InferenceTimer {
    /// Start time of generation
    start: Instant,
    /// Time when first token was generated
    first_token_time: Option<Instant>,
    /// Number of tokens generated (including first)
    tokens_generated: usize,
    /// Number of tokens in the prompt
    prompt_tokens: usize,
    /// Sampling temperature
    temperature: f32,
}

impl InferenceTimer {
    /// Create a new inference timer.
    ///
    /// Call this just before starting token generation.
    pub fn new(prompt_tokens: usize, temperature: f32) -> Self {
        Self {
            start: Instant::now(),
            first_token_time: None,
            tokens_generated: 0,
            prompt_tokens,
            temperature,
        }
    }

    /// Mark that the first token has been generated.
    ///
    /// This records the Time To First Token (TTFT).
    pub fn mark_first_token(&mut self) {
        if self.first_token_time.is_none() {
            self.first_token_time = Some(Instant::now());
            self.tokens_generated = 1;
        }
    }

    /// Record that a token has been generated (after the first).
    pub fn token_generated(&mut self) {
        self.tokens_generated += 1;
    }

    /// Set the total number of tokens generated.
    ///
    /// Use this if you're not calling token_generated() for each token.
    pub fn set_tokens_generated(&mut self, count: usize) {
        self.tokens_generated = count;
    }

    /// Finish timing and create the inference record.
    ///
    /// Automatically logs the record if logging is enabled.
    pub fn finish(self) -> InferenceRecord {
        let total_time_ms = self.start.elapsed().as_secs_f32() * 1000.0;

        let time_to_first_token_ms = self
            .first_token_time
            .map(|t| (t - self.start).as_secs_f32() * 1000.0)
            .unwrap_or(total_time_ms); // If TTFT not marked, use total time

        let record = InferenceRecord::new(
            self.prompt_tokens,
            self.tokens_generated,
            time_to_first_token_ms,
            total_time_ms,
            self.temperature,
        );

        Logger::log_inference(&record);
        record
    }

    /// Finish timing and return the record without logging.
    pub fn finish_no_log(self) -> InferenceRecord {
        let total_time_ms = self.start.elapsed().as_secs_f32() * 1000.0;

        let time_to_first_token_ms = self
            .first_token_time
            .map(|t| (t - self.start).as_secs_f32() * 1000.0)
            .unwrap_or(total_time_ms);

        InferenceRecord::new(
            self.prompt_tokens,
            self.tokens_generated,
            time_to_first_token_ms,
            total_time_ms,
            self.temperature,
        )
    }

    /// Get elapsed time since start in milliseconds.
    pub fn elapsed_ms(&self) -> f32 {
        self.start.elapsed().as_secs_f32() * 1000.0
    }

    /// Get time to first token in milliseconds (if marked).
    pub fn ttft_ms(&self) -> Option<f32> {
        self.first_token_time
            .map(|t| (t - self.start).as_secs_f32() * 1000.0)
    }

    /// Get number of tokens generated so far.
    pub fn tokens(&self) -> usize {
        self.tokens_generated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_inference_timer_basic() {
        let mut timer = InferenceTimer::new(10, 0.8);

        // Simulate some work
        sleep(Duration::from_millis(5));
        timer.mark_first_token();

        sleep(Duration::from_millis(5));
        timer.token_generated();
        timer.token_generated();

        let record = timer.finish_no_log();

        assert_eq!(record.prompt_tokens, 10);
        assert_eq!(record.generated_tokens, 3);
        assert!(record.time_to_first_token_ms > 0.0);
        assert!(record.total_time_ms >= record.time_to_first_token_ms);
        assert_eq!(record.temperature, 0.8);
    }

    #[test]
    fn test_inference_timer_set_tokens() {
        let mut timer = InferenceTimer::new(5, 1.0);
        timer.mark_first_token();
        timer.set_tokens_generated(50);

        let record = timer.finish_no_log();
        assert_eq!(record.generated_tokens, 50);
    }
}
