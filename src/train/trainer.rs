//! Core trainer struct and training step implementation.
//!
//! The forward pass is in `forward.rs`, backward pass in `backward.rs`,
//! and training loop orchestration in `training_loop.rs`.

use std::path::Path;

use crate::command_batch::CommandBatch;
use crate::nn::{GPTModel, GPTModelState, ModelConfig};
use crate::ops::{
    cross_entropy_fused, embedding_backward, matmul, rmsnorm_backward,
    total_l2_norm_gpu,
};
use crate::optim::{Lion, LionConfig};
use crate::profile::{Phase, Profiler};
use crate::tensor::Tensor;

use super::cache::ForwardCache;
use super::checkpoint::{save_model_weights, Checkpoint};
use super::config::TrainingConfig;
use super::helpers::{
    add_tensors, ensure_fp32, scale_gradients_inplace,
};
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

        Self {
            config: train_config.clone(),
            model,
            optimizer,
            model_state,
            scheduler,
            step: 0,
            epoch: 0,
            best_val_loss: f32::INFINITY,
        }
    }

    /// Create trainer from a checkpoint.
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

    /// Save a checkpoint.
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

    /// Compute loss for a batch (forward pass only, no gradients).
    pub fn compute_loss(
        &self,
        input_ids: &[u32],
        target_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> f32 {
        let logits = self.model.forward(input_ids, batch_size, seq_len, 0);

        // Reshape logits to [batch_size * seq_len, vocab_size]
        let logits_2d = logits.view(&[batch_size * seq_len, self.model.config.vocab_size]);

        let (loss, _, _) = cross_entropy_fused(&logits_2d, target_ids);
        loss
    }

    /// Training step with full backward pass through all layers.
    ///
    /// Returns (loss, gradient_norm).
    pub fn train_step(
        &mut self,
        input_ids: &[u32],
        target_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> (f32, f32) {
        // Wait for any pending async work from previous step
        // This ensures optimizer updates are complete before we read weights
        if self.config.async_gpu {
            CommandBatch::wait_for_completion();
        }

        // Enable command buffer batching for reduced GPU synchronization
        CommandBatch::begin();

        Profiler::begin_step();

        let lr = self.scheduler.get_lr(self.step);
        self.optimizer.set_lr(lr);

        let vocab_size = self.model.config.vocab_size;
        let n = batch_size * seq_len;

        // ========== Forward pass with activation caching ==========
        Profiler::set_phase(Phase::Forward);
        let cache = self.forward_with_cache(input_ids, batch_size, seq_len);

        // Compute logits: final_hidden @ embed_tokens.T
        let logits = self.compute_logits_from_hidden(&cache.final_hidden);

        // Sync before operations that need completed results
        CommandBatch::sync();
        let logits_2d = logits.view(&[n, vocab_size]);

        // Compute loss and gradient w.r.t. logits (has internal syncs for loss read)
        let (loss, _, grad_logits) = cross_entropy_fused(&logits_2d, target_ids);

        // ========== Backward pass ==========
        Profiler::set_phase(Phase::Backward);

        // Gradient for embedding from output projection
        // logits = final_hidden @ embed.T, so grad_embed_out = grad_logits.T @ final_hidden
        let grad_embed_out = matmul_tn(&grad_logits, &cache.final_hidden);

        // Gradient flowing back through output projection
        // grad_final_hidden = grad_logits @ embed (convert BF16 weights to FP32 if needed)
        let embed_fp32 = ensure_fp32(&self.model.embed_tokens);
        let grad_hidden_2d = matmul(&grad_logits, &embed_fp32).unwrap();
        // Reshape to 3D for norm backward
        let hidden_dim = self.model.config.hidden_dim;
        let grad_hidden_3d = grad_hidden_2d.view(&[batch_size, seq_len, hidden_dim]);

        // Backward through final norm (convert BF16 gamma to FP32 if needed)
        let final_norm_fp32 = ensure_fp32(&self.model.final_norm);
        let (grad_pre_norm, grad_final_norm) = rmsnorm_backward(
            &grad_hidden_3d,
            &self.get_pre_final_norm_hidden(&cache, batch_size, seq_len),
            &final_norm_fp32,
            self.model.config.norm_eps,
        );
        let mut grad_hidden = grad_pre_norm;

        // Backward through transformer layers (in reverse order)
        let mut layer_grads = Vec::new();
        for (layer_idx, layer_cache) in cache.layers.iter().enumerate().rev() {
            Profiler::set_layer(Some(layer_idx));
            let layer = &self.model.layers[layer_idx];
            let grads =
                self.backward_transformer_layer(&grad_hidden, layer_cache, layer, batch_size, seq_len);
            grad_hidden = grads.grad_input.clone();
            layer_grads.push(grads);
        }
        Profiler::set_layer(None);
        layer_grads.reverse();

        // Backward through embedding lookup
        // Reshape grad_hidden to 2D for embedding_backward
        let grad_hidden_2d_for_embed = grad_hidden.view(&[n, hidden_dim]);
        let grad_embed_in = embedding_backward(&grad_hidden_2d_for_embed, input_ids, vocab_size);

        // Sum embedding gradients (input + output if tied)
        let grad_embed = if self.model.config.tie_weights {
            add_tensors(&grad_embed_out, &grad_embed_in)
        } else {
            grad_embed_in
        };

        // ========== Compute gradient norm and clip ==========
        // Use GPU-based gradient norm computation - dispatches reduction kernels
        // to GPU, syncs once, and reads only partial sums (much faster than CPU)
        let mut all_grads: Vec<&Tensor> = vec![&grad_embed, &grad_final_norm];
        for grads in &layer_grads {
            all_grads.push(&grads.grad_attn_norm);
            all_grads.push(&grads.grad_ffn_norm);
            all_grads.push(&grads.grad_wq);
            all_grads.push(&grads.grad_wk);
            all_grads.push(&grads.grad_wv);
            all_grads.push(&grads.grad_wo);
            all_grads.push(&grads.grad_w_gate);
            all_grads.push(&grads.grad_w_up);
            all_grads.push(&grads.grad_w_down);
        }

        let total_grad_norm = total_l2_norm_gpu(&all_grads);
        let clip_scale = if total_grad_norm > self.config.max_grad_norm {
            self.config.max_grad_norm / total_grad_norm
        } else {
            1.0
        };

        // Scale all gradients in-place with a single batched operation
        // This avoids allocating 56 new tensors and reduces kernel launch overhead
        if clip_scale != 1.0 {
            scale_gradients_inplace(&all_grads, clip_scale);
        }

        // ========== Apply optimizer to all parameters ==========
        Profiler::set_phase(Phase::Optimizer);
        let use_bf16 = self.config.use_bf16;

        // Helper macro to handle precision-aware optimizer step
        macro_rules! opt_step {
            ($weights:expr, $grads:expr, $state:expr) => {
                if use_bf16 {
                    self.optimizer.step_bf16($weights, $grads, $state);
                } else {
                    self.optimizer.step($weights, $grads, $state);
                }
            };
        }

        // Embedding (gradients are already scaled in-place)
        opt_step!(
            &self.model.embed_tokens,
            &grad_embed,
            &mut self.model_state.embed_state
        );

        // Final norm
        opt_step!(
            &self.model.final_norm,
            &grad_final_norm,
            &mut self.model_state.final_norm_state
        );

        // Transformer layers
        for (layer_idx, grads) in layer_grads.iter().enumerate() {
            let layer = &self.model.layers[layer_idx];
            let state = &mut self.model_state.layer_states[layer_idx];

            // Attention norms
            opt_step!(&layer.attn_norm, &grads.grad_attn_norm, &mut state.attn_norm_state);
            opt_step!(&layer.ffn_norm, &grads.grad_ffn_norm, &mut state.ffn_norm_state);

            // Attention weights
            opt_step!(&layer.attention.wq.weight, &grads.grad_wq, &mut state.attention_state.wq_state);
            opt_step!(&layer.attention.wk.weight, &grads.grad_wk, &mut state.attention_state.wk_state);
            opt_step!(&layer.attention.wv.weight, &grads.grad_wv, &mut state.attention_state.wv_state);
            opt_step!(&layer.attention.wo.weight, &grads.grad_wo, &mut state.attention_state.wo_state);

            // FFN weights
            opt_step!(&layer.ffn.w_gate.weight, &grads.grad_w_gate, &mut state.ffn_state.w_gate_state);
            opt_step!(&layer.ffn.w_up.weight, &grads.grad_w_up, &mut state.ffn_state.w_up_state);
            opt_step!(&layer.ffn.w_down.weight, &grads.grad_w_down, &mut state.ffn_state.w_down_state);
        }

        self.step += 1;

        // End command batching
        if self.config.async_gpu {
            // Async mode: commit without waiting, allowing CPU to prepare next batch
            // The wait happens at the start of the next train_step
            CommandBatch::commit_async();
        } else {
            // Sync mode: commit and wait for all optimizer steps
            CommandBatch::end();
        }

        Profiler::end_step();
        (loss, total_grad_norm)
    }

    /// Compute logits from hidden states.
    fn compute_logits_from_hidden(&self, hidden: &Tensor) -> Tensor {
        // hidden: [n, hidden_dim], embed: [vocab, hidden_dim]
        // logits = hidden @ embed.T = [n, vocab]
        // Use MPS's native transpose support (convert BF16 weights to FP32 if needed)
        let embed_fp32 = ensure_fp32(&self.model.embed_tokens);
        crate::ops::matmul_mps_nt(hidden, &embed_fp32)
    }

    /// Get hidden states before final norm (from last layer cache).
    fn get_pre_final_norm_hidden(
        &self,
        cache: &ForwardCache,
        _batch_size: usize,
        _seq_len: usize,
    ) -> Tensor {
        // Return the cached pre-final-norm hidden state (output of last transformer layer)
        cache.pre_final_norm.clone()
    }
}

/// Matrix multiply with first operand transposed: A.T @ B
///
/// Uses MPS's native transpose support to avoid explicit transposition.
fn matmul_tn(a: &Tensor, b: &Tensor) -> Tensor {
    crate::ops::matmul_mps_tn(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::Precision;

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
            precision: Precision::FP32,
            embed_dropout: 0.0,
            attn_dropout: 0.1,
            ffn_dropout: 0.1,
        };

        let train_config = TrainingConfig::default();
        let trainer = Trainer::new(&model_config, &train_config);

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
            precision: Precision::FP32,
            embed_dropout: 0.0,
            attn_dropout: 0.1,
            ffn_dropout: 0.1,
        };

        let train_config = TrainingConfig::default();
        let trainer = Trainer::new(&model_config, &train_config);

        let batch_size = 2;
        let seq_len = 8;
        let input_ids: Vec<u32> = (0..batch_size * seq_len).map(|i| (i % 100) as u32).collect();
        let target_ids: Vec<u32> = (0..batch_size * seq_len)
            .map(|i| ((i + 1) % 100) as u32)
            .collect();

        let loss = trainer.compute_loss(&input_ids, &target_ids, batch_size, seq_len);

        // Loss should be positive
        assert!(loss > 0.0);
        // Loss should be reasonable for random initialization
        assert!(loss < 10.0);
    }

    #[test]
    fn test_train_step_produces_gradients() {
        use crate::ops::cross_entropy_fused;

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
            precision: Precision::FP32,
            embed_dropout: 0.0,
            attn_dropout: 0.0,  // Disable dropout for determinism
            ffn_dropout: 0.0,
        };

        let train_config = TrainingConfig {
            dropout_enabled: false, // Disable dropout
            async_gpu: false,       // Use synchronous mode
            ..Default::default()
        };

        let trainer = Trainer::new(&model_config, &train_config);

        let batch_size = 2;
        let seq_len = 8;
        let vocab_size = model_config.vocab_size;
        let n = batch_size * seq_len;

        let input_ids: Vec<u32> = (0..n).map(|i| (i % 100) as u32).collect();
        let target_ids: Vec<u32> = (0..n).map(|i| ((i + 1) % 100) as u32).collect();

        // Step 1: Forward pass (model.forward reads GPU data, so don't use batching)
        // NOTE: model.forward() has internal as_f32_slice() calls which require sync
        let logits = trainer.model.forward(&input_ids, batch_size, seq_len, 0);

        let logits_data = logits.as_f32_slice();
        let logits_nonzero = logits_data.iter().any(|&x| x != 0.0);
        println!("Logits shape: {:?}, non-zero: {}", logits.shape(), logits_nonzero);
        assert!(logits_nonzero, "Logits should be non-zero");

        // Step 2: Cross-entropy gradient (cross_entropy_fused has internal syncs)
        let logits_2d = logits.view(&[n, vocab_size]);
        let (loss, _, grad_logits) = cross_entropy_fused(&logits_2d, &target_ids);

        let grad_logits_data = grad_logits.as_f32_slice();
        let grad_logits_nonzero = grad_logits_data.iter().any(|&x| x != 0.0);
        let grad_logits_sum: f32 = grad_logits_data.iter().map(|x| x.abs()).sum();
        println!(
            "Loss: {}, grad_logits shape: {:?}, non-zero: {}, abs_sum: {}",
            loss,
            grad_logits.shape(),
            grad_logits_nonzero,
            grad_logits_sum
        );
        assert!(grad_logits_nonzero, "grad_logits should be non-zero");

        // Step 3: Test a simple matmul with grad_logits (matmul syncs internally)
        let embed_fp32 = ensure_fp32(&trainer.model.embed_tokens);
        println!("embed_fp32 shape: {:?}", embed_fp32.shape());

        let grad_hidden = matmul(&grad_logits, &embed_fp32).unwrap();

        let grad_hidden_data = grad_hidden.as_f32_slice();
        let grad_hidden_nonzero = grad_hidden_data.iter().any(|&x| x != 0.0);
        let grad_hidden_sum: f32 = grad_hidden_data.iter().map(|x| x.abs()).sum();
        println!(
            "grad_hidden shape: {:?}, non-zero: {}, abs_sum: {}",
            grad_hidden.shape(),
            grad_hidden_nonzero,
            grad_hidden_sum
        );
        assert!(grad_hidden_nonzero, "grad_hidden should be non-zero");

        // Loss should be positive
        assert!(loss > 0.0, "Loss should be positive, got {}", loss);
    }

    #[test]
    fn test_full_train_step_gradients() {
        use crate::command_batch::CommandBatch;
        use crate::ops::cross_entropy_fused;

        // Test that trainer forward_with_cache + backward produces non-zero gradients
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
            precision: Precision::FP32,
            embed_dropout: 0.0,
            attn_dropout: 0.0,  // Disable dropout for determinism
            ffn_dropout: 0.0,
        };

        let train_config = TrainingConfig {
            dropout_enabled: false, // Disable dropout
            async_gpu: false,       // Use synchronous mode
            ..Default::default()
        };

        let trainer = Trainer::new(&model_config, &train_config);

        let batch_size = 2;
        let seq_len = 8;
        let vocab_size = model_config.vocab_size;
        let n = batch_size * seq_len;
        let input_ids: Vec<u32> = (0..n).map(|i| (i % 100) as u32).collect();
        let target_ids: Vec<u32> = (0..n).map(|i| ((i + 1) % 100) as u32).collect();

        // Test 1: Forward pass WITHOUT CommandBatch (immediate mode)
        println!("\n=== Test 1: Forward without batching ===");
        let cache = trainer.forward_with_cache(&input_ids, batch_size, seq_len);

        let fh_data = cache.final_hidden.as_f32_slice();
        let fh_nonzero = fh_data.iter().any(|&x| x != 0.0);
        let fh_sum: f32 = fh_data.iter().map(|x| x.abs()).sum();
        println!("Final hidden shape: {:?}, non-zero: {}, abs_sum: {}",
                 cache.final_hidden.shape(), fh_nonzero, fh_sum);
        assert!(fh_nonzero, "Final hidden should be non-zero (immediate mode)");

        // Test 2: Forward pass WITH CommandBatch (batched mode)
        println!("\n=== Test 2: Forward with batching ===");
        CommandBatch::begin();
        let cache2 = trainer.forward_with_cache(&input_ids, batch_size, seq_len);
        CommandBatch::sync();

        let fh_data2 = cache2.final_hidden.as_f32_slice();
        let fh_nonzero2 = fh_data2.iter().any(|&x| x != 0.0);
        let fh_sum2: f32 = fh_data2.iter().map(|x| x.abs()).sum();
        println!("Final hidden shape: {:?}, non-zero: {}, abs_sum: {}",
                 cache2.final_hidden.shape(), fh_nonzero2, fh_sum2);
        CommandBatch::end();
        assert!(fh_nonzero2, "Final hidden should be non-zero (batched mode)");

        // Test 3: Logits computation
        println!("\n=== Test 3: Logits computation ===");
        let logits = trainer.compute_logits_from_hidden(&cache.final_hidden);
        let logits_data = logits.as_f32_slice();
        let logits_nonzero = logits_data.iter().any(|&x| x != 0.0);
        let logits_sum: f32 = logits_data.iter().map(|x| x.abs()).sum();
        println!("Logits shape: {:?}, non-zero: {}, abs_sum: {}",
                 logits.shape(), logits_nonzero, logits_sum);
        assert!(logits_nonzero, "Logits should be non-zero");

        // Test 4: Cross-entropy gradient
        println!("\n=== Test 4: Cross-entropy gradient ===");
        let logits_2d = logits.view(&[n, vocab_size]);
        let (loss, _, grad_logits) = cross_entropy_fused(&logits_2d, &target_ids);
        let gl_data = grad_logits.as_f32_slice();
        let gl_nonzero = gl_data.iter().any(|&x| x != 0.0);
        let gl_sum: f32 = gl_data.iter().map(|x| x.abs()).sum();
        println!("Loss: {}, grad_logits shape: {:?}, non-zero: {}, abs_sum: {}",
                 loss, grad_logits.shape(), gl_nonzero, gl_sum);
        assert!(gl_nonzero, "grad_logits should be non-zero");

        // Test 5: Backward matmul (grad_embed_out)
        println!("\n=== Test 5: Backward matmul ===");
        let grad_embed_out = matmul_tn(&grad_logits, &cache.final_hidden);
        let geo_data = grad_embed_out.as_f32_slice();
        let geo_nonzero = geo_data.iter().any(|&x| x != 0.0);
        let geo_sum: f32 = geo_data.iter().map(|x| x.abs()).sum();
        println!("grad_embed_out shape: {:?}, non-zero: {}, abs_sum: {}",
                 grad_embed_out.shape(), geo_nonzero, geo_sum);
        assert!(geo_nonzero, "grad_embed_out should be non-zero");

        println!("\n=== All checks passed! ===");
    }
}
