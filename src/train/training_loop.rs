//! Training loop implementation.
//!
//! Contains the main training and evaluation orchestration methods.

use std::time::Instant;

use crate::command_batch::CommandBatch;
use crate::data::{DatasetIterator, TokenDataset};

use super::callbacks::TrainCallback;
use super::config::TrainMetrics;
use super::trainer::Trainer;

impl Trainer {
    /// Train for one epoch.
    ///
    /// Uses async command submission to overlap CPU data preparation with GPU execution.
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

        // Prefetch first batch
        let mut next_batch = iter.next();

        while let Some((input_ids, target_ids)) = next_batch {
            let actual_batch = input_ids.len() / seq_len;
            if actual_batch == 0 {
                next_batch = iter.next();
                continue;
            }

            // Prefetch next batch while GPU executes current batch
            // This overlaps CPU data loading with GPU computation
            let prefetch_batch = iter.next();

            let (loss, gn) = self.train_step(&input_ids, &target_ids, actual_batch, seq_len);
            tokens_processed += actual_batch * seq_len;

            // Use prefetched batch for next iteration
            next_batch = prefetch_batch;

            // Log at intervals
            if self.step.is_multiple_of(self.config.log_interval) {
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
            if self.step.is_multiple_of(self.config.save_interval) && self.step > 0 {
                let path = format!("{}/step_{}.bin", self.config.checkpoint_dir, self.step);

                // Create checkpoint directory if it doesn't exist
                if let Err(e) = std::fs::create_dir_all(&self.config.checkpoint_dir) {
                    eprintln!("Failed to create checkpoint directory '{}': {}", self.config.checkpoint_dir, e);
                }

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

    /// Evaluate on a validation dataset.
    ///
    /// Returns the average loss across all batches in the dataset.
    /// Automatically disables dropout during evaluation.
    pub fn evaluate(&mut self, dataset: &TokenDataset, batch_size: usize) -> f32 {
        let seq_len = dataset.seq_len();
        let iter = DatasetIterator::new(dataset, batch_size, false);

        // Set model to evaluation mode (disables dropout)
        let was_training = self.model.is_training();
        self.model.set_training(false);

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

        // Restore training mode
        self.model.set_training(was_training);

        if num_batches > 0 {
            total_loss / num_batches as f32
        } else {
            0.0
        }
    }

    /// Full training loop.
    ///
    /// Runs for the specified number of epochs, with optional validation after each epoch.
    /// Saves checkpoints at configured intervals and tracks the best validation loss.
    pub fn train<C: TrainCallback>(
        &mut self,
        train_dataset: &TokenDataset,
        val_dataset: Option<&TokenDataset>,
        batch_size: usize,
        num_epochs: usize,
        callback: &mut C,
    ) {
        println!("Starting training:");
        println!(
            "  Model params: {:.2}M",
            self.model.config.num_params() as f64 / 1e6
        );
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
            let mut should_early_stop = false;
            if let Some(val_ds) = val_dataset {
                let val_loss = self.evaluate(val_ds, batch_size);
                callback.on_eval(self.step, val_loss);

                // Check for improvement (with min_delta threshold)
                let min_delta = self.config.early_stopping_min_delta;
                if val_loss < self.best_val_loss - min_delta {
                    // Improvement: save best model and reset patience
                    self.best_val_loss = val_loss;
                    self.patience_counter = 0;
                    let path = format!("{}/best.bin", self.config.checkpoint_dir);
                    if let Err(e) = std::fs::create_dir_all(&self.config.checkpoint_dir) {
                        eprintln!("Failed to create checkpoint directory '{}': {}", self.config.checkpoint_dir, e);
                    }
                    if let Err(e) = self.save_checkpoint(&path) {
                        eprintln!("Failed to save best model: {}", e);
                    } else {
                        println!("New best validation loss: {:.4}", val_loss);
                    }
                } else {
                    // No improvement: increment patience counter
                    self.patience_counter += 1;
                    if let Some(patience) = self.config.early_stopping_patience
                        && self.patience_counter >= patience
                    {
                        println!(
                            "Early stopping: no improvement for {} evaluations (best val_loss: {:.4})",
                            patience, self.best_val_loss
                        );
                        should_early_stop = true;
                    }
                }
            }

            if should_early_stop {
                break;
            }

            if self.step >= self.config.total_steps {
                println!(
                    "Reached total_steps ({}), stopping training",
                    self.config.total_steps
                );
                break;
            }
        }

        // Ensure all async GPU work is complete before saving
        if self.config.async_gpu {
            CommandBatch::wait_for_completion();
            CommandBatch::end_async();
        }

        // Save final checkpoint
        let path = format!("{}/final.bin", self.config.checkpoint_dir);
        if let Err(e) = std::fs::create_dir_all(&self.config.checkpoint_dir) {
            eprintln!("Failed to create checkpoint directory '{}': {}", self.config.checkpoint_dir, e);
        }
        if let Err(e) = self.save_checkpoint(&path) {
            eprintln!("Failed to save final checkpoint: {}", e);
        } else {
            callback.on_save(self.step, &path);
        }
    }
}
