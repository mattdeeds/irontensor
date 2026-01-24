//! Global dropout seed management.
//!
//! Uses atomic counters to generate unique seeds for each dropout operation.
//! The forward pass generates a seed, and the backward pass uses the same seed
//! to regenerate the identical mask (Philox RNG on GPU).

use std::sync::atomic::{AtomicU64, Ordering};

/// Base seed for dropout RNG (can be set for reproducibility)
static DROPOUT_SEED: AtomicU64 = AtomicU64::new(42);

/// Counter that increments for each dropout call
static DROPOUT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Set the global dropout seed for reproducibility.
///
/// Call this at the start of training with a fixed seed for reproducible results.
pub fn set_dropout_seed(seed: u64) {
    DROPOUT_SEED.store(seed, Ordering::SeqCst);
    DROPOUT_COUNTER.store(0, Ordering::SeqCst);
}

/// Get the next unique dropout seed.
///
/// Each call returns a different seed by combining the base seed with an
/// incrementing counter. The returned seed should be passed to the GPU
/// kernel and cached for use in the backward pass.
pub fn next_dropout_seed() -> u64 {
    let base = DROPOUT_SEED.load(Ordering::Relaxed);
    let counter = DROPOUT_COUNTER.fetch_add(1, Ordering::Relaxed);
    // Combine base seed and counter using a simple hash
    // This ensures different seeds even with the same base
    base.wrapping_mul(0x517cc1b727220a95).wrapping_add(counter)
}

/// Reset the dropout counter (e.g., at the start of each epoch for reproducibility).
pub fn reset_dropout_counter() {
    DROPOUT_COUNTER.store(0, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_uniqueness() {
        set_dropout_seed(12345);
        let s1 = next_dropout_seed();
        let s2 = next_dropout_seed();
        let s3 = next_dropout_seed();
        assert_ne!(s1, s2);
        assert_ne!(s2, s3);
        assert_ne!(s1, s3);
    }

    #[test]
    fn test_seed_reproducibility() {
        set_dropout_seed(12345);
        let seeds1: Vec<u64> = (0..5).map(|_| next_dropout_seed()).collect();

        set_dropout_seed(12345);
        let seeds2: Vec<u64> = (0..5).map(|_| next_dropout_seed()).collect();

        assert_eq!(seeds1, seeds2);
    }
}
