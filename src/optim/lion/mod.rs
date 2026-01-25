//! Lion optimizer implementation.
//!
//! Lion (EvoLved Sign Momentum) is a simple and memory-efficient optimizer
//! that uses sign-based updates. It typically works well for language models
//! and requires less memory than Adam (only stores momentum, not variance).

mod config;
mod gradients;
mod optimizer;
mod pipelines;

// Re-export public API
pub use config::{LionConfig, ParamState};
pub use gradients::{clip_grad_norm, grad_norm, zero_gradients};
pub use optimizer::Lion;

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_lion_step_basic() {
        // Initial weights
        let weights = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        // Gradients (all positive)
        let gradients = Tensor::from_f32_slice(&[0.1, 0.2, 0.3, 0.4], &[4]);
        // Initial momentum (zeros)
        let mut state = ParamState::new(&[4]);

        let optimizer = Lion::new(LionConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        });

        optimizer.step(&weights, &gradients, &mut state);

        let w = weights.as_f32_slice();
        let m = state.momentum.as_f32_slice();

        // With zero initial momentum and positive gradients:
        // update = sign(0.99 * 0 + 0.01 * g) = sign(0.01 * g) = 1 for all
        // w_new = w - 0.1 * 1 = w - 0.1
        for i in 0..4 {
            let expected_w = (i + 1) as f32 - 0.1;
            assert!(
                (w[i] - expected_w).abs() < 1e-5,
                "Weight mismatch at {}: expected {}, got {}",
                i,
                expected_w,
                w[i]
            );
        }

        // Momentum should be updated: m_new = 0.9 * 0 + 0.1 * g = 0.1 * g
        for i in 0..4 {
            let expected_m = 0.1 * (i + 1) as f32 * 0.1;
            assert!(
                (m[i] - expected_m).abs() < 1e-5,
                "Momentum mismatch at {}: expected {}, got {}",
                i,
                expected_m,
                m[i]
            );
        }
    }

    #[test]
    fn test_lion_step_with_momentum() {
        // Test that momentum affects the update direction
        let weights = Tensor::from_f32_slice(&[1.0, 1.0, 1.0, 1.0], &[4]);
        // Gradients: some positive, some negative
        let gradients = Tensor::from_f32_slice(&[1.0, -1.0, 0.5, -0.5], &[4]);
        let mut state = ParamState::new(&[4]);

        let optimizer = Lion::with_lr(0.01);

        // First step
        optimizer.step(&weights, &gradients, &mut state);
        let w1 = weights.as_f32_slice().to_vec();

        // Second step with same gradients
        optimizer.step(&weights, &gradients, &mut state);
        let w2 = weights.as_f32_slice().to_vec();

        // Weights should continue to decrease/increase based on sign
        assert!(w2[0] < w1[0], "Positive gradient should decrease weight");
        assert!(w2[1] > w1[1], "Negative gradient should increase weight");
    }

    #[test]
    fn test_lion_weight_decay() {
        let weights = Tensor::from_f32_slice(&[10.0, -10.0, 5.0, -5.0], &[4]);
        let gradients = Tensor::from_f32_slice(&[0.0, 0.0, 0.0, 0.0], &[4]); // Zero gradients
        let mut state = ParamState::new(&[4]);

        let optimizer = Lion::new(LionConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.1,
        });

        let w_before = weights.as_f32_slice().to_vec();
        optimizer.step(&weights, &gradients, &mut state);
        let w_after = weights.as_f32_slice();

        // With zero gradients, only weight decay affects weights
        // update = sign(0) = 0, so w_new = w - lr * wd * w = w * (1 - lr * wd)
        // w_new = w * (1 - 0.1 * 0.1) = w * 0.99
        for i in 0..4 {
            let expected = w_before[i] * 0.99;
            assert!(
                (w_after[i] - expected).abs() < 1e-4,
                "Weight decay mismatch at {}: expected {}, got {}",
                i,
                expected,
                w_after[i]
            );
        }
    }

    #[test]
    fn test_zero_gradients() {
        let gradients = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);

        zero_gradients(&gradients);

        let g = gradients.as_f32_slice();
        for i in 0..4 {
            assert_eq!(g[i], 0.0, "Gradient at {} should be zero", i);
        }
    }

    #[test]
    fn test_grad_norm() {
        // Test with known values: [3, 4] has norm 5
        let gradients = Tensor::from_f32_slice(&[3.0, 4.0], &[2]);
        let norm = grad_norm(&gradients);
        assert!(
            (norm - 5.0).abs() < 1e-4,
            "Norm mismatch: expected 5.0, got {}",
            norm
        );
    }

    #[test]
    fn test_clip_grad_norm() {
        let gradients = Tensor::from_f32_slice(&[3.0, 4.0], &[2]);

        // Clip to max_norm=2.5 (original norm is 5.0)
        let actual_norm = clip_grad_norm(&gradients, 2.5);

        assert!(
            (actual_norm - 5.0).abs() < 1e-4,
            "Returned norm mismatch: expected 5.0, got {}",
            actual_norm
        );

        // After clipping, gradients should be scaled by 2.5/5.0 = 0.5
        let g = gradients.as_f32_slice();
        assert!(
            (g[0] - 1.5).abs() < 1e-4,
            "Clipped grad[0] mismatch: expected 1.5, got {}",
            g[0]
        );
        assert!(
            (g[1] - 2.0).abs() < 1e-4,
            "Clipped grad[1] mismatch: expected 2.0, got {}",
            g[1]
        );

        // Verify new norm
        let new_norm = grad_norm(&gradients);
        assert!(
            (new_norm - 2.5).abs() < 1e-4,
            "New norm mismatch: expected 2.5, got {}",
            new_norm
        );
    }

    #[test]
    fn test_clip_grad_norm_no_clip() {
        let gradients = Tensor::from_f32_slice(&[3.0, 4.0], &[2]);
        let original = gradients.as_f32_slice().to_vec();

        // Clip to max_norm=10.0 (original norm is 5.0, so no clipping needed)
        let actual_norm = clip_grad_norm(&gradients, 10.0);

        assert!(
            (actual_norm - 5.0).abs() < 1e-4,
            "Returned norm mismatch: expected 5.0, got {}",
            actual_norm
        );

        // Gradients should be unchanged
        let g = gradients.as_f32_slice();
        assert_eq!(g[0], original[0], "Grad[0] should be unchanged");
        assert_eq!(g[1], original[1], "Grad[1] should be unchanged");
    }

    #[test]
    fn test_lion_multiple_steps() {
        // Test convergence towards minimum of f(x) = x^2
        // Gradient of x^2 is 2x
        let weights = Tensor::from_f32_slice(&[5.0], &[1]);
        let mut state = ParamState::new(&[1]);

        let optimizer = Lion::with_lr(0.1);

        // Run several steps
        for _ in 0..50 {
            let w = weights.as_f32_slice()[0];
            let grad = 2.0 * w; // gradient of x^2
            let gradients = Tensor::from_f32_slice(&[grad], &[1]);
            optimizer.step(&weights, &gradients, &mut state);
        }

        let final_w = weights.as_f32_slice()[0];
        // Should have moved towards 0
        assert!(
            final_w.abs() < 1.0,
            "Weight should be close to 0, got {}",
            final_w
        );
    }
}
