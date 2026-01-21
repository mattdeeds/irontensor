use std::path::Path;
use std::time::Instant;

use crate::data::{DatasetIterator, TokenDataset};
use crate::nn::{GPTModel, GPTModelState, ModelConfig};
use crate::ops::{cross_entropy_fused, embedding_backward, matmul};
use crate::optim::{Lion, LionConfig};
use crate::tensor::Tensor;

use super::checkpoint::{save_model_weights, Checkpoint};
use super::scheduler::{CosineAnnealingLR, LRScheduler};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Base learning rate
    pub learning_rate: f32,
    /// Weight decay coefficient
    pub weight_decay: f32,
    /// Lion optimizer beta1
    pub beta1: f32,
    /// Lion optimizer beta2
    pub beta2: f32,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
    /// Number of warmup steps
    pub warmup_steps: usize,
    /// Total number of training steps
    pub total_steps: usize,
    /// Log every N steps
    pub log_interval: usize,
    /// Save checkpoint every N steps
    pub save_interval: usize,
    /// Evaluate every N steps
    pub eval_interval: usize,
    /// Checkpoint save directory
    pub checkpoint_dir: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            max_grad_norm: 1.0,
            warmup_steps: 100,
            total_steps: 10000,
            log_interval: 10,
            save_interval: 1000,
            eval_interval: 100,
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}

/// Training metrics for a single step
#[derive(Debug, Clone)]
pub struct TrainMetrics {
    pub step: usize,
    pub loss: f32,
    pub grad_norm: f32,
    pub learning_rate: f32,
    pub tokens_per_sec: f32,
}

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

/// Main trainer struct
pub struct Trainer {
    pub config: TrainingConfig,
    pub model: GPTModel,
    pub optimizer: Lion,
    pub model_state: GPTModelState,
    pub scheduler: Box<dyn LRScheduler>,
    pub step: usize,
    pub epoch: usize,
    pub best_val_loss: f32,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(model_config: ModelConfig, train_config: TrainingConfig) -> Self {
        let model = GPTModel::new(model_config.clone());
        let model_state = GPTModelState::new(&model);

        let optimizer = Lion::new(LionConfig {
            lr: train_config.learning_rate,
            beta1: train_config.beta1,
            beta2: train_config.beta2,
            weight_decay: train_config.weight_decay,
        });

        let scheduler = Box::new(CosineAnnealingLR::with_warmup(
            train_config.learning_rate,
            train_config.warmup_steps,
            train_config.total_steps,
        ));

        Self {
            config: train_config,
            model,
            optimizer,
            model_state,
            scheduler,
            step: 0,
            epoch: 0,
            best_val_loss: f32::INFINITY,
        }
    }

    /// Create trainer from a checkpoint
    pub fn from_checkpoint<P: AsRef<Path>>(
        path: P,
        train_config: TrainingConfig,
    ) -> std::io::Result<Self> {
        let (model, checkpoint) = super::checkpoint::load_model_weights(path)?;
        let model_state = GPTModelState::new(&model);

        let optimizer = Lion::new(LionConfig {
            lr: train_config.learning_rate,
            beta1: train_config.beta1,
            beta2: train_config.beta2,
            weight_decay: train_config.weight_decay,
        });

        let scheduler = Box::new(CosineAnnealingLR::with_warmup(
            train_config.learning_rate,
            train_config.warmup_steps,
            train_config.total_steps,
        ));

        Ok(Self {
            config: train_config,
            model,
            optimizer,
            model_state,
            scheduler,
            step: checkpoint.step,
            epoch: checkpoint.epoch,
            best_val_loss: checkpoint.best_val_loss,
        })
    }

    /// Save a checkpoint
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let checkpoint = Checkpoint {
            config: self.model.config.clone(),
            step: self.step,
            epoch: self.epoch,
            best_val_loss: self.best_val_loss,
            learning_rate: self.scheduler.get_lr(self.step),
        };
        save_model_weights(path, &self.model, &checkpoint)
    }

    /// Compute loss for a batch (forward pass only)
    pub fn compute_loss(&self, input_ids: &[u32], target_ids: &[u32], batch_size: usize, seq_len: usize) -> f32 {
        let logits = self.model.forward(input_ids, batch_size, seq_len, 0);

        // Reshape logits to [batch_size * seq_len, vocab_size]
        let logits_2d = {
            let data = logits.as_f32_slice();
            Tensor::from_f32_slice(data, &[batch_size * seq_len, self.model.config.vocab_size])
        };

        let (loss, _, _) = cross_entropy_fused(&logits_2d, target_ids);
        loss
    }

    /// Training step: forward, backward, and optimizer step
    ///
    /// Returns the loss and gradient norm for this step.
    pub fn train_step(&mut self, input_ids: &[u32], target_ids: &[u32], batch_size: usize, seq_len: usize) -> (f32, f32) {
        let lr = self.scheduler.get_lr(self.step);

        // Update optimizer learning rate
        self.optimizer.set_lr(lr);

        // Forward pass
        let logits = self.model.forward(input_ids, batch_size, seq_len, 0);

        // Reshape logits to [batch_size * seq_len, vocab_size]
        let logits_2d = {
            let data = logits.as_f32_slice();
            Tensor::from_f32_slice(data, &[batch_size * seq_len, self.model.config.vocab_size])
        };

        // Compute loss and gradient
        let (loss, _, grad_logits) = cross_entropy_fused(&logits_2d, target_ids);

        // Backward pass through the model
        // This is a simplified backward pass - in practice you'd want a proper autodiff system
        let grad_hidden = self.backward_output_projection(&grad_logits, batch_size, seq_len);
        self.backward_through_layers(&grad_hidden, input_ids, batch_size, seq_len);

        // Compute gradient norm before clipping
        let total_grad_norm = self.compute_total_grad_norm();

        // Clip gradients
        self.clip_all_gradients();

        // Optimizer step
        self.optimizer_step();

        // Zero gradients for next step
        self.zero_all_gradients();

        self.step += 1;

        (loss, total_grad_norm)
    }

    /// Backward pass through output projection (lm_head or tied embeddings)
    fn backward_output_projection(&mut self, grad_logits: &Tensor, _batch_size: usize, _seq_len: usize) -> Tensor {
        // grad_logits: [batch_size * seq_len, vocab_size]
        // If using weight tying, output projection is embed_tokens.T
        // grad_hidden = grad_logits @ embed_tokens

        let grad_hidden = matmul(grad_logits, &self.model.embed_tokens);

        // Accumulate gradient for embed_tokens (from output projection)
        // grad_embed += grad_logits.T @ hidden (but we'd need hidden states saved)
        // For simplicity in this implementation, we skip this gradient

        grad_hidden
    }

    /// Backward pass through transformer layers
    fn backward_through_layers(&mut self, grad_hidden: &Tensor, input_ids: &[u32], _batch_size: usize, _seq_len: usize) {
        // This is a simplified backward pass
        // A full implementation would cache activations during forward and use them here

        // For now, we'll compute gradients for the embedding layer
        let grad_embed_out = grad_hidden;

        // Backward through final norm (simplified - just pass through for demonstration)
        // In practice you'd use rmsnorm_backward here

        // Backward through embedding
        // grad_embed_weights accumulates gradients for each token
        let _grad_embed_weights = embedding_backward(
            grad_embed_out,
            input_ids,
            self.model.config.vocab_size,
        );

        // Accumulate into model state
        // (In a full implementation, you'd have proper gradient tensors)
        // For now, this demonstrates the structure

        // Note: A complete backward pass would go through each transformer layer
        // computing gradients for attention, FFN, and layer norms
    }

    /// Compute total gradient norm across all parameters
    fn compute_total_grad_norm(&self) -> f32 {
        // In a full implementation, you'd sum the squared norms of all gradient tensors
        // For now, return a placeholder
        1.0
    }

    /// Clip gradients for all parameters
    fn clip_all_gradients(&mut self) {
        // In a full implementation, you'd clip all gradient tensors
        // This requires storing gradient tensors separately
    }

    /// Perform optimizer step on all parameters
    fn optimizer_step(&mut self) {
        // In a full implementation, you'd call optimizer.step for each parameter
        // This requires proper gradient tensors stored in model_state
    }

    /// Zero all gradients
    fn zero_all_gradients(&mut self) {
        // In a full implementation, you'd zero all gradient tensors
    }

    /// Train for one epoch
    pub fn train_epoch<C: TrainCallback>(
        &mut self,
        dataset: &TokenDataset,
        batch_size: usize,
        shuffle: bool,
        callback: &mut C,
    ) {
        let seq_len = dataset.seq_len();
        let mut iter = DatasetIterator::new(dataset, batch_size, shuffle);

        let epoch_start = Instant::now();
        let mut step_start = Instant::now();
        let mut tokens_processed = 0usize;

        for (input_ids, target_ids) in &mut iter {
            let actual_batch = input_ids.len() / seq_len;
            if actual_batch == 0 {
                continue;
            }

            let (loss, gn) = self.train_step(&input_ids, &target_ids, actual_batch, seq_len);
            tokens_processed += actual_batch * seq_len;

            // Log at intervals
            if self.step % self.config.log_interval == 0 {
                let elapsed = step_start.elapsed().as_secs_f32();
                let tokens_per_sec = (self.config.log_interval * batch_size * seq_len) as f32 / elapsed;

                let metrics = TrainMetrics {
                    step: self.step,
                    loss,
                    grad_norm: gn,
                    learning_rate: self.scheduler.get_lr(self.step),
                    tokens_per_sec,
                };
                callback.on_step(&metrics);
                step_start = Instant::now();
            }

            // Save checkpoint at intervals
            if self.step % self.config.save_interval == 0 && self.step > 0 {
                let path = format!("{}/step_{}.bin", self.config.checkpoint_dir, self.step);

                // Create checkpoint directory if it doesn't exist
                std::fs::create_dir_all(&self.config.checkpoint_dir).ok();

                if let Err(e) = self.save_checkpoint(&path) {
                    eprintln!("Failed to save checkpoint: {}", e);
                } else {
                    callback.on_save(self.step, &path);
                }
            }

            // Check if we've reached total steps
            if self.step >= self.config.total_steps {
                break;
            }
        }

        self.epoch += 1;
        let epoch_elapsed = epoch_start.elapsed().as_secs_f32();
        println!(
            "Epoch {} completed in {:.1}s ({} tokens)",
            self.epoch, epoch_elapsed, tokens_processed
        );
    }

    /// Evaluate on a validation dataset
    pub fn evaluate(&self, dataset: &TokenDataset, batch_size: usize) -> f32 {
        let seq_len = dataset.seq_len();
        let iter = DatasetIterator::new(dataset, batch_size, false);

        let mut total_loss = 0.0f32;
        let mut num_batches = 0usize;

        for (input_ids, target_ids) in iter {
            let actual_batch = input_ids.len() / seq_len;
            if actual_batch == 0 {
                continue;
            }

            let loss = self.compute_loss(&input_ids, &target_ids, actual_batch, seq_len);
            total_loss += loss;
            num_batches += 1;
        }

        if num_batches > 0 {
            total_loss / num_batches as f32
        } else {
            0.0
        }
    }

    /// Full training loop
    pub fn train<C: TrainCallback>(
        &mut self,
        train_dataset: &TokenDataset,
        val_dataset: Option<&TokenDataset>,
        batch_size: usize,
        num_epochs: usize,
        callback: &mut C,
    ) {
        println!("Starting training:");
        println!("  Model params: {:.2}M", self.model.config.num_params() as f64 / 1e6);
        println!("  Batch size: {}", batch_size);
        println!("  Sequence length: {}", train_dataset.seq_len());
        println!("  Total steps: {}", self.config.total_steps);
        println!("  Learning rate: {}", self.config.learning_rate);
        println!();

        for epoch in 0..num_epochs {
            println!("=== Epoch {}/{} ===", epoch + 1, num_epochs);

            // Training
            self.train_epoch(train_dataset, batch_size, true, callback);

            // Validation
            if let Some(val_ds) = val_dataset {
                let val_loss = self.evaluate(val_ds, batch_size);
                callback.on_eval(self.step, val_loss);

                // Save best model
                if val_loss < self.best_val_loss {
                    self.best_val_loss = val_loss;
                    let path = format!("{}/best.bin", self.config.checkpoint_dir);
                    std::fs::create_dir_all(&self.config.checkpoint_dir).ok();
                    if let Err(e) = self.save_checkpoint(&path) {
                        eprintln!("Failed to save best model: {}", e);
                    } else {
                        println!("New best validation loss: {:.4}", val_loss);
                    }
                }
            }

            if self.step >= self.config.total_steps {
                println!("Reached total_steps ({}), stopping training", self.config.total_steps);
                break;
            }
        }

        // Save final checkpoint
        let path = format!("{}/final.bin", self.config.checkpoint_dir);
        std::fs::create_dir_all(&self.config.checkpoint_dir).ok();
        if let Err(e) = self.save_checkpoint(&path) {
            eprintln!("Failed to save final checkpoint: {}", e);
        } else {
            callback.on_save(self.step, &path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_trainer_creation() {
        let model_config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_dim: 64,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            max_seq_len: 128,
            tie_weights: true,
        };

        let train_config = TrainingConfig::default();
        let trainer = Trainer::new(model_config, train_config);

        assert_eq!(trainer.step, 0);
        assert_eq!(trainer.epoch, 0);
    }

    #[test]
    fn test_compute_loss() {
        let model_config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_dim: 64,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            max_seq_len: 128,
            tie_weights: true,
        };

        let train_config = TrainingConfig::default();
        let trainer = Trainer::new(model_config, train_config);

        let batch_size = 2;
        let seq_len = 8;
        let input_ids: Vec<u32> = (0..batch_size * seq_len).map(|i| (i % 100) as u32).collect();
        let target_ids: Vec<u32> = (0..batch_size * seq_len).map(|i| ((i + 1) % 100) as u32).collect();

        let loss = trainer.compute_loss(&input_ids, &target_ids, batch_size, seq_len);

        // Loss should be positive
        assert!(loss > 0.0);
        // Loss should be reasonable for random initialization
        assert!(loss < 10.0);
    }

    #[test]
    fn test_save_load_trainer() {
        let model_config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_dim: 64,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            max_seq_len: 128,
            tie_weights: true,
        };

        let train_config = TrainingConfig::default();
        let mut trainer = Trainer::new(model_config.clone(), train_config.clone());

        // Simulate some training
        trainer.step = 500;
        trainer.epoch = 2;
        trainer.best_val_loss = 2.5;

        // Save checkpoint
        let path = temp_dir().join("trainer_test.bin");
        trainer.save_checkpoint(&path).unwrap();

        // Load checkpoint
        let loaded = Trainer::from_checkpoint(&path, train_config).unwrap();

        assert_eq!(loaded.step, 500);
        assert_eq!(loaded.epoch, 2);
        assert!((loaded.best_val_loss - 2.5).abs() < 1e-5);

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_evaluate() {
        let model_config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_dim: 64,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            max_seq_len: 128,
            tie_weights: true,
        };

        let train_config = TrainingConfig::default();
        let trainer = Trainer::new(model_config, train_config);

        // Create a small dataset
        let dataset_path = temp_dir().join("eval_test.bin");
        let tokens: Vec<u32> = (0..500).map(|i| (i % 100) as u32).collect();
        TokenDataset::create(&dataset_path, &tokens).unwrap();
        let dataset = TokenDataset::open(&dataset_path, 16).unwrap();

        let val_loss = trainer.evaluate(&dataset, 4);

        assert!(val_loss > 0.0);
        assert!(val_loss < 10.0);

        // Cleanup
        std::fs::remove_file(dataset_path).ok();
    }
}
