/// Learning rate schedulers for training

/// Learning rate scheduler trait
pub trait LRScheduler {
    /// Get the learning rate for the current step
    fn get_lr(&self, step: usize) -> f32;

    /// Get the base learning rate
    fn base_lr(&self) -> f32;
}

/// Constant learning rate (no scheduling)
pub struct ConstantLR {
    lr: f32,
}

impl ConstantLR {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLR {
    fn get_lr(&self, _step: usize) -> f32 {
        self.lr
    }

    fn base_lr(&self) -> f32 {
        self.lr
    }
}

/// Linear warmup followed by constant learning rate
pub struct WarmupConstantLR {
    base_lr: f32,
    warmup_steps: usize,
}

impl WarmupConstantLR {
    pub fn new(base_lr: f32, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
        }
    }
}

impl LRScheduler for WarmupConstantLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step as f32 + 1.0) / self.warmup_steps as f32
        } else {
            self.base_lr
        }
    }

    fn base_lr(&self) -> f32 {
        self.base_lr
    }
}

/// Cosine annealing with optional warmup
///
/// lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * (step - warmup) / (total - warmup)))
pub struct CosineAnnealingLR {
    max_lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
}

impl CosineAnnealingLR {
    pub fn new(max_lr: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        assert!(total_steps > warmup_steps, "total_steps must be > warmup_steps");
        Self {
            max_lr,
            min_lr,
            warmup_steps,
            total_steps,
        }
    }

    /// Create with default min_lr = 0.1 * max_lr
    pub fn with_warmup(max_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self::new(max_lr, max_lr * 0.1, warmup_steps, total_steps)
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.max_lr * (step as f32 + 1.0) / self.warmup_steps as f32
        } else if step >= self.total_steps {
            // After total steps, return min_lr
            self.min_lr
        } else {
            // Cosine annealing
            let progress = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps) as f32;
            self.min_lr
                + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }

    fn base_lr(&self) -> f32 {
        self.max_lr
    }
}

/// Linear decay with optional warmup
///
/// After warmup, lr decreases linearly from max_lr to min_lr
pub struct LinearDecayLR {
    max_lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    decay_steps: usize,
}

impl LinearDecayLR {
    pub fn new(max_lr: f32, min_lr: f32, warmup_steps: usize, decay_steps: usize) -> Self {
        Self {
            max_lr,
            min_lr,
            warmup_steps,
            decay_steps,
        }
    }
}

impl LRScheduler for LinearDecayLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.max_lr * (step as f32 + 1.0) / self.warmup_steps as f32
        } else {
            let decay_step = step - self.warmup_steps;
            if decay_step >= self.decay_steps {
                self.min_lr
            } else {
                let progress = decay_step as f32 / self.decay_steps as f32;
                self.max_lr - (self.max_lr - self.min_lr) * progress
            }
        }
    }

    fn base_lr(&self) -> f32 {
        self.max_lr
    }
}

/// 1/sqrt(step) decay - used in original Transformer paper
///
/// lr = base_lr * min(1/sqrt(step), step * warmup^{-1.5})
pub struct InverseSqrtLR {
    base_lr: f32,
    warmup_steps: usize,
}

impl InverseSqrtLR {
    pub fn new(base_lr: f32, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
        }
    }
}

impl LRScheduler for InverseSqrtLR {
    fn get_lr(&self, step: usize) -> f32 {
        let step = step.max(1) as f32;
        let warmup = self.warmup_steps as f32;

        let inv_sqrt = 1.0 / step.sqrt();
        let warmup_factor = step * warmup.powf(-1.5);

        self.base_lr * inv_sqrt.min(warmup_factor)
    }

    fn base_lr(&self) -> f32 {
        self.base_lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_lr() {
        let scheduler = ConstantLR::new(0.001);
        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(100), 0.001);
        assert_eq!(scheduler.get_lr(10000), 0.001);
    }

    #[test]
    fn test_warmup_constant_lr() {
        let scheduler = WarmupConstantLR::new(0.001, 100);

        // During warmup
        assert!((scheduler.get_lr(0) - 0.00001).abs() < 1e-7);
        assert!((scheduler.get_lr(49) - 0.0005).abs() < 1e-7);

        // After warmup
        assert_eq!(scheduler.get_lr(100), 0.001);
        assert_eq!(scheduler.get_lr(1000), 0.001);
    }

    #[test]
    fn test_cosine_annealing_lr() {
        let scheduler = CosineAnnealingLR::new(0.001, 0.0001, 100, 1000);

        // During warmup (linear increase)
        assert!(scheduler.get_lr(0) < scheduler.get_lr(50));
        assert!(scheduler.get_lr(50) < scheduler.get_lr(99));

        // At peak (just after warmup)
        let peak = scheduler.get_lr(100);
        assert!((peak - 0.001).abs() < 1e-5);

        // Decreasing during annealing
        assert!(scheduler.get_lr(200) < scheduler.get_lr(150));
        assert!(scheduler.get_lr(500) < scheduler.get_lr(300));

        // At end (should be close to min_lr)
        let end = scheduler.get_lr(999);
        assert!(end > 0.0001);
        assert!(end < 0.0002);
    }

    #[test]
    fn test_linear_decay_lr() {
        let scheduler = LinearDecayLR::new(0.001, 0.0001, 100, 900);

        // During warmup
        assert!(scheduler.get_lr(0) < scheduler.get_lr(50));

        // At peak
        assert!((scheduler.get_lr(100) - 0.001).abs() < 1e-7);

        // Midway through decay
        let mid = scheduler.get_lr(550); // 450 steps into decay
        assert!(mid > 0.0001);
        assert!(mid < 0.001);

        // At end
        assert_eq!(scheduler.get_lr(1000), 0.0001);
        assert_eq!(scheduler.get_lr(2000), 0.0001);
    }

    #[test]
    fn test_inverse_sqrt_lr() {
        let scheduler = InverseSqrtLR::new(0.001, 100);

        // During warmup (linear increase)
        assert!(scheduler.get_lr(1) < scheduler.get_lr(50));

        // After warmup (1/sqrt decay)
        let lr_100 = scheduler.get_lr(100);
        let lr_400 = scheduler.get_lr(400);

        // lr_400 should be roughly lr_100 / 2 (since sqrt(400) = 2 * sqrt(100))
        let ratio = lr_100 / lr_400;
        assert!((ratio - 2.0).abs() < 0.1);
    }
}
