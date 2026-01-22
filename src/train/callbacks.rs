use super::config::TrainMetrics;

/// Callback for training events
pub trait TrainCallback {
    /// Called after each training step
    fn on_step(&mut self, metrics: &TrainMetrics);

    /// Called after evaluation
    fn on_eval(&mut self, step: usize, val_loss: f32);

    /// Called when saving checkpoint
    fn on_save(&mut self, step: usize, path: &str);
}

/// Simple callback that prints to stdout
pub struct PrintCallback;

impl TrainCallback for PrintCallback {
    fn on_step(&mut self, metrics: &TrainMetrics) {
        println!(
            "Step {:>6} | Loss: {:.4} | Grad norm: {:.4} | LR: {:.2e} | {:.0} tok/s",
            metrics.step,
            metrics.loss,
            metrics.grad_norm,
            metrics.learning_rate,
            metrics.tokens_per_sec
        );
    }

    fn on_eval(&mut self, step: usize, val_loss: f32) {
        println!("Step {:>6} | Validation loss: {:.4}", step, val_loss);
    }

    fn on_save(&mut self, step: usize, path: &str) {
        println!("Step {:>6} | Saved checkpoint to: {}", step, path);
    }
}
