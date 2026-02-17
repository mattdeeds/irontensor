//! Core trainer struct and training step implementation.
//!
//! The forward pass is in `forward.rs`, backward pass in `backward.rs`,
//! and training loop orchestration in `training_loop.rs`.

mod gpu_trace;
mod train_step;

use std::path::Path;

use crate::nn::{GPTModel, GPTModelState, ModelConfig};
use crate::optim::{Lion, LionConfig};
use crate::tensor::Tensor;

use super::cache::{AccumulatedGradients, ForwardCache};
use super::checkpoint::{save_model_weights, Checkpoint};
use super::config::TrainingConfig;
use super::helpers::ensure_fp32;
use super::scheduler::{CosineAnnealingLR, LRScheduler};

/// Main trainer struct.
///
/// Holds the model, optimizer, scheduler, and training state.
pub struct Trainer {
    pub config: TrainingConfig,
    pub model: GPTModel,
    pub optimizer: Lion,
    pub model_state: GPTModelState,
    pub scheduler: Box<dyn LRScheduler>,
    pub step: usize,
    pub epoch: usize,
    pub best_val_loss: f32,
    /// Micro-step counter for gradient accumulation (0 to accumulation_steps-1)
    pub(crate) micro_step: usize,
    /// Accumulated gradients (None if accumulation_steps == 1)
    pub(crate) accumulated_grads: Option<AccumulatedGradients>,
    /// Accumulated loss for averaging
    pub(crate) accumulated_loss: f32,
    /// Counter for early stopping patience (tracks consecutive evaluations without improvement)
    pub(super) patience_counter: usize,
}

impl Trainer {
    /// Create a new trainer.
    pub fn new(model_config: &ModelConfig, train_config: &TrainingConfig) -> Self {
        let mut model = GPTModel::new(model_config);

        // Convert model to BF16 for mixed precision training if enabled
        if train_config.use_bf16 {
            model.to_bf16();
        }

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

        // Initialize accumulated gradients if using gradient accumulation
        let accumulated_grads = if train_config.accumulation_steps > 1 {
            let head_dim = model_config.hidden_dim / model_config.num_heads;
            Some(AccumulatedGradients::zeros(
                model_config.vocab_size,
                model_config.hidden_dim,
                model_config.num_layers,
                model_config.num_heads,
                model_config.num_kv_heads,
                head_dim,
                model_config.intermediate_dim,
            ))
        } else {
            None
        };

        Self {
            config: train_config.clone(),
            model,
            optimizer,
            model_state,
            scheduler,
            step: 0,
            epoch: 0,
            best_val_loss: f32::INFINITY,
            micro_step: 0,
            accumulated_grads,
            accumulated_loss: 0.0,
            patience_counter: 0,
        }
    }

    /// Create trainer from a checkpoint.
    ///
    /// If the checkpoint contains optimizer state, it will be restored automatically.
    /// Otherwise, fresh optimizer state (zeroed momentum) will be created.
    pub fn from_checkpoint<P: AsRef<Path>>(
        path: P,
        train_config: TrainingConfig,
    ) -> std::io::Result<Self> {
        let (model, checkpoint, opt_state) =
            super::checkpoint::load_model_weights_with_optimizer(path)?;

        // Use loaded optimizer state if available, otherwise create fresh state
        let model_state = opt_state.unwrap_or_else(|| GPTModelState::new(&model));

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

        if checkpoint.has_optimizer_state {
            eprintln!(
                "Loaded checkpoint with optimizer state (step {}, epoch {})",
                checkpoint.step, checkpoint.epoch
            );
        } else {
            eprintln!(
                "Loaded checkpoint without optimizer state (step {}, epoch {}) - momentum will restart from zero",
                checkpoint.step, checkpoint.epoch
            );
        }

        // Initialize accumulated gradients if using gradient accumulation
        let accumulated_grads = if train_config.accumulation_steps > 1 {
            let model_config = &checkpoint.config;
            let head_dim = model_config.hidden_dim / model_config.num_heads;
            Some(AccumulatedGradients::zeros(
                model_config.vocab_size,
                model_config.hidden_dim,
                model_config.num_layers,
                model_config.num_heads,
                model_config.num_kv_heads,
                head_dim,
                model_config.intermediate_dim,
            ))
        } else {
            None
        };

        Ok(Self {
            config: train_config,
            model,
            optimizer,
            model_state,
            scheduler,
            step: checkpoint.step,
            epoch: checkpoint.epoch,
            best_val_loss: checkpoint.best_val_loss,
            micro_step: 0,
            accumulated_grads,
            accumulated_loss: 0.0,
            patience_counter: 0, // Reset patience when loading from checkpoint
        })
    }

    /// Save a checkpoint (without optimizer state).
    ///
    /// For training resumption with stable optimizer momentum, use
    /// `save_checkpoint_with_optimizer` instead.
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let checkpoint = Checkpoint {
            config: self.model.config.clone(),
            step: self.step,
            epoch: self.epoch,
            best_val_loss: self.best_val_loss,
            learning_rate: self.scheduler.get_lr(self.step),
            has_optimizer_state: false,
        };
        save_model_weights(path, &self.model, &checkpoint)
    }

    /// Save a checkpoint with optimizer state.
    ///
    /// This preserves the optimizer momentum tensors, allowing training to resume
    /// without the "warmup" period that occurs when momentum is reset to zero.
    pub fn save_checkpoint_with_optimizer<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let checkpoint = Checkpoint {
            config: self.model.config.clone(),
            step: self.step,
            epoch: self.epoch,
            best_val_loss: self.best_val_loss,
            learning_rate: self.scheduler.get_lr(self.step),
            has_optimizer_state: true,
        };
        super::checkpoint::save_model_weights_with_optimizer(
            path,
            &self.model,
            &checkpoint,
            &self.model_state,
        )
    }

    /// Compute loss for a batch (forward pass only, no gradients).
    pub fn compute_loss(
        &self,
        input_ids: &[u32],
        target_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> f32 {
        use crate::ops::cross_entropy_fused;

        let logits = self.model.forward(input_ids, batch_size, seq_len, 0);

        // Reshape logits to [batch_size * seq_len, vocab_size]
        let logits_2d = logits.view(&[batch_size * seq_len, self.model.config.vocab_size]);

        let (loss, _, _) = cross_entropy_fused(&logits_2d, target_ids);
        loss
    }

    /// Compute logits from hidden states.
    pub(crate) fn compute_logits_from_hidden(&self, hidden: &Tensor) -> Tensor {
        // hidden: [n, hidden_dim], embed: [vocab, hidden_dim]
        // logits = hidden @ embed.T = [n, vocab]
        // Use MPS's native transpose support (convert BF16 weights to FP32 if needed)
        let embed_fp32 = ensure_fp32(&self.model.embed_tokens);
        crate::ops::matmul_mps_nt(hidden, &embed_fp32)
    }

    /// Get hidden states before final norm (from last layer cache).
    pub(crate) fn get_pre_final_norm_hidden(
        &self,
        cache: &ForwardCache,
        _batch_size: usize,
        _seq_len: usize,
    ) -> Tensor {
        // Return the cached pre-final-norm hidden state (output of last transformer layer)
        cache.pre_final_norm.clone()
    }
}

#[cfg(test)]
mod tests;
