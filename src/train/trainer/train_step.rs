//! Training step implementation with gradient computation and optimizer updates.

use crate::command_batch::CommandBatch;
use crate::ops::{
    cross_entropy_fused, embedding_backward, matmul, rmsnorm_backward,
    total_l2_norm_gpu,
};
use crate::profile::{Phase, Profiler};
use crate::tensor::Tensor;

use super::super::cache::{LayerCacheVariant, LayerGradients};
use super::super::helpers::{
    add_tensors, ensure_fp32, scale_gradients_inplace,
};
use super::Trainer;

/// Matrix multiply with first operand transposed: A.T @ B
///
/// Uses MPS's native transpose support to avoid explicit transposition.
fn matmul_tn(a: &Tensor, b: &Tensor) -> Tensor {
    crate::ops::matmul_mps_tn(a, b)
}

impl Trainer {
    /// Training step with full backward pass through all layers.
    ///
    /// When `accumulation_steps > 1`, this method accumulates gradients over multiple
    /// micro-batches before applying the optimizer. The effective batch size becomes
    /// `batch_size * accumulation_steps`.
    ///
    /// Returns (loss, gradient_norm, optimizer_stepped):
    /// - `loss`: The loss for this micro-batch
    /// - `gradient_norm`: The gradient norm (only meaningful when optimizer stepped)
    /// - For backwards compatibility, returns (loss, grad_norm) tuple
    ///
    /// When using gradient accumulation:
    /// - Loss is the micro-batch loss (not averaged)
    /// - Gradient norm is 0.0 except on the final micro-batch when optimizer steps
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

        let vocab_size = self.model.config.vocab_size;
        let n = batch_size * seq_len;
        let hidden_dim = self.model.config.hidden_dim;
        let accumulation_steps = self.config.accumulation_steps;

        // ========== Forward pass with activation caching ==========
        Profiler::set_phase(Phase::Forward);

        // Use checkpointing forward pass if enabled
        let use_checkpointing = self.config.checkpoint_config.enabled;
        let (loss, layer_grads, grad_embed_out, grad_final_norm, grad_hidden) = if use_checkpointing {
            self.forward_backward_with_checkpointing(
                input_ids, target_ids, batch_size, seq_len, n, vocab_size, hidden_dim
            )
        } else {
            self.forward_backward_standard(
                input_ids, target_ids, batch_size, seq_len, n, vocab_size, hidden_dim
            )
        };

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

        // ========== Gradient Accumulation ==========
        if accumulation_steps > 1 {
            return self.apply_gradient_accumulation(
                loss, layer_grads, grad_embed, grad_final_norm, accumulation_steps
            );
        }

        // ========== No accumulation: original behavior ==========
        self.apply_optimizer_direct(loss, layer_grads, grad_embed, grad_final_norm)
    }

    /// Forward and backward pass with activation checkpointing.
    fn forward_backward_with_checkpointing(
        &self,
        input_ids: &[u32],
        target_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
        n: usize,
        vocab_size: usize,
        hidden_dim: usize,
    ) -> (f32, Vec<LayerGradients>, Tensor, Tensor, Tensor) {
        use LayerGradients;

        // Forward with checkpointing (stores only minimal data for some layers)
        let cache = self.forward_with_checkpointing(input_ids, batch_size, seq_len);

        // Compute logits: final_hidden @ embed_tokens.T
        let logits = self.compute_logits_from_hidden(&cache.final_hidden);

        // Sync before operations that need completed results
        CommandBatch::sync();
        let logits_2d = logits.view(&[n, vocab_size]);

        // Compute loss and gradient w.r.t. logits (has internal syncs for loss read)
        let (loss_val, _, grad_logits) = cross_entropy_fused(&logits_2d, target_ids);

        // ========== Backward pass ==========
        Profiler::set_phase(Phase::Backward);

        // Gradient for embedding from output projection
        let grad_embed_out = matmul_tn(&grad_logits, &cache.final_hidden);

        // Gradient flowing back through output projection
        let embed_fp32 = ensure_fp32(&self.model.embed_tokens);
        let grad_hidden_2d = matmul(&grad_logits, &embed_fp32).unwrap();
        let grad_hidden_3d = grad_hidden_2d.view(&[batch_size, seq_len, hidden_dim]);

        // Backward through final norm
        let final_norm_fp32 = ensure_fp32(&self.model.final_norm);
        let (grad_pre_norm, grad_final_norm) = rmsnorm_backward(
            &grad_hidden_3d,
            &cache.pre_final_norm,
            &final_norm_fp32,
            self.model.config.norm_eps,
        );
        let mut grad_hidden = grad_pre_norm;

        // Backward through transformer layers (in reverse order)
        // For checkpointed layers, recompute the full cache first
        let mut layer_grads: Vec<LayerGradients> = Vec::new();
        let num_layers = cache.layers.len();
        for layer_idx in (0..num_layers).rev() {
            Profiler::set_layer(Some(layer_idx));
            let layer = &self.model.layers[layer_idx];

            // Get or recompute full cache
            let grads = match &cache.layers[layer_idx] {
                LayerCacheVariant::Full(c) => {
                    self.backward_transformer_layer(&grad_hidden, c, layer, batch_size, seq_len)
                }
                LayerCacheVariant::Checkpointed(cp) => {
                    // Recompute full cache then compute gradients
                    let recomputed_cache = self.recompute_layer_forward(cp, layer, batch_size, seq_len);
                    self.backward_transformer_layer(&grad_hidden, &recomputed_cache, layer, batch_size, seq_len)
                }
            };

            grad_hidden = grads.grad_input.clone();
            layer_grads.push(grads);
        }
        Profiler::set_layer(None);
        layer_grads.reverse();

        (loss_val, layer_grads, grad_embed_out, grad_final_norm, grad_hidden)
    }

    /// Standard forward and backward pass (stores all activations).
    fn forward_backward_standard(
        &self,
        input_ids: &[u32],
        target_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
        n: usize,
        vocab_size: usize,
        hidden_dim: usize,
    ) -> (f32, Vec<LayerGradients>, Tensor, Tensor, Tensor) {
        use LayerGradients;

        // Standard forward pass (stores all activations)
        let cache = self.forward_with_cache(input_ids, batch_size, seq_len);

        // Compute logits: final_hidden @ embed_tokens.T
        let logits = self.compute_logits_from_hidden(&cache.final_hidden);

        // Sync before operations that need completed results
        CommandBatch::sync();
        let logits_2d = logits.view(&[n, vocab_size]);

        // Compute loss and gradient w.r.t. logits (has internal syncs for loss read)
        let (loss_val, _, grad_logits) = cross_entropy_fused(&logits_2d, target_ids);

        // ========== Backward pass ==========
        Profiler::set_phase(Phase::Backward);

        // Gradient for embedding from output projection
        let grad_embed_out = matmul_tn(&grad_logits, &cache.final_hidden);

        // Gradient flowing back through output projection
        let embed_fp32 = ensure_fp32(&self.model.embed_tokens);
        let grad_hidden_2d = matmul(&grad_logits, &embed_fp32).unwrap();
        let grad_hidden_3d = grad_hidden_2d.view(&[batch_size, seq_len, hidden_dim]);

        // Backward through final norm
        let final_norm_fp32 = ensure_fp32(&self.model.final_norm);
        let (grad_pre_norm, grad_final_norm) = rmsnorm_backward(
            &grad_hidden_3d,
            &self.get_pre_final_norm_hidden(&cache, batch_size, seq_len),
            &final_norm_fp32,
            self.model.config.norm_eps,
        );
        let mut grad_hidden = grad_pre_norm;

        // Backward through transformer layers (in reverse order)
        let mut layer_grads: Vec<LayerGradients> = Vec::new();
        for (layer_idx, layer_cache) in cache.layers.iter().enumerate().rev() {
            Profiler::set_layer(Some(layer_idx));
            let layer = &self.model.layers[layer_idx];
            let grads = self.backward_transformer_layer(&grad_hidden, layer_cache, layer, batch_size, seq_len);
            grad_hidden = grads.grad_input.clone();
            layer_grads.push(grads);
        }
        Profiler::set_layer(None);
        layer_grads.reverse();

        (loss_val, layer_grads, grad_embed_out, grad_final_norm, grad_hidden)
    }

    /// Apply gradient accumulation logic.
    fn apply_gradient_accumulation(
        &mut self,
        loss: f32,
        layer_grads: Vec<LayerGradients>,
        grad_embed: Tensor,
        grad_final_norm: Tensor,
        accumulation_steps: usize,
    ) -> (f32, f32) {
        use crate::ops::{axpy_inplace, zero_tensor};

        // Scale factor for averaging gradients over accumulation steps
        let scale = 1.0 / accumulation_steps as f32;

        // Accumulate gradients into storage
        if let Some(ref acc) = self.accumulated_grads {
            axpy_inplace(&acc.grad_embed, &grad_embed, scale).unwrap();
            axpy_inplace(&acc.grad_final_norm, &grad_final_norm, scale).unwrap();

            for (layer_idx, grads) in layer_grads.iter().enumerate() {
                let acc_layer = &acc.layer_grads[layer_idx];
                axpy_inplace(&acc_layer.grad_attn_norm, &grads.grad_attn_norm, scale).unwrap();
                axpy_inplace(&acc_layer.grad_ffn_norm, &grads.grad_ffn_norm, scale).unwrap();
                axpy_inplace(&acc_layer.grad_wq, &grads.grad_wq, scale).unwrap();
                axpy_inplace(&acc_layer.grad_wk, &grads.grad_wk, scale).unwrap();
                axpy_inplace(&acc_layer.grad_wv, &grads.grad_wv, scale).unwrap();
                axpy_inplace(&acc_layer.grad_wo, &grads.grad_wo, scale).unwrap();
                axpy_inplace(&acc_layer.grad_w_gate, &grads.grad_w_gate, scale).unwrap();
                axpy_inplace(&acc_layer.grad_w_up, &grads.grad_w_up, scale).unwrap();
                axpy_inplace(&acc_layer.grad_w_down, &grads.grad_w_down, scale).unwrap();
            }
        }

        // Accumulate loss for averaging
        self.accumulated_loss += loss;
        self.micro_step += 1;

        // Check if we should apply optimizer
        if self.micro_step < accumulation_steps {
            // Not yet time to apply optimizer - just return the micro-batch loss
            if self.config.async_gpu {
                CommandBatch::commit_async();
            } else {
                CommandBatch::end();
            }
            Profiler::end_step();
            return (loss, 0.0);
        }

        // Final micro-batch of accumulation cycle - apply optimizer with accumulated gradients
        self.micro_step = 0;
        let avg_loss = self.accumulated_loss / accumulation_steps as f32;
        self.accumulated_loss = 0.0;

        // Use accumulated gradients for clipping and optimizer
        if let Some(ref acc) = self.accumulated_grads {
            // Compute gradient norm and clip using accumulated gradients
            let mut all_grads: Vec<&Tensor> = vec![&acc.grad_embed, &acc.grad_final_norm];
            for acc_layer in &acc.layer_grads {
                all_grads.push(&acc_layer.grad_attn_norm);
                all_grads.push(&acc_layer.grad_ffn_norm);
                all_grads.push(&acc_layer.grad_wq);
                all_grads.push(&acc_layer.grad_wk);
                all_grads.push(&acc_layer.grad_wv);
                all_grads.push(&acc_layer.grad_wo);
                all_grads.push(&acc_layer.grad_w_gate);
                all_grads.push(&acc_layer.grad_w_up);
                all_grads.push(&acc_layer.grad_w_down);
            }

            let total_grad_norm = total_l2_norm_gpu(&all_grads);
            let clip_scale = if total_grad_norm > self.config.max_grad_norm {
                self.config.max_grad_norm / total_grad_norm
            } else {
                1.0
            };

            if clip_scale != 1.0 {
                scale_gradients_inplace(&all_grads, clip_scale);
            }

            // Apply optimizer with accumulated gradients
            Profiler::set_phase(Phase::Optimizer);
            let lr = self.scheduler.get_lr(self.step);
            self.optimizer.set_lr(lr);
            let use_bf16 = self.config.use_bf16;

            macro_rules! opt_step {
                ($weights:expr, $grads:expr, $state:expr) => {
                    if use_bf16 {
                        self.optimizer.step_bf16($weights, $grads, $state);
                    } else {
                        self.optimizer.step($weights, $grads, $state);
                    }
                };
            }

            opt_step!(&self.model.embed_tokens, &acc.grad_embed, &mut self.model_state.embed_state);
            opt_step!(&self.model.final_norm, &acc.grad_final_norm, &mut self.model_state.final_norm_state);

            for (layer_idx, acc_layer) in acc.layer_grads.iter().enumerate() {
                let layer = &self.model.layers[layer_idx];
                let state = &mut self.model_state.layer_states[layer_idx];

                opt_step!(&layer.attn_norm, &acc_layer.grad_attn_norm, &mut state.attn_norm_state);
                opt_step!(&layer.ffn_norm, &acc_layer.grad_ffn_norm, &mut state.ffn_norm_state);
                opt_step!(&layer.attention.wq.weight, &acc_layer.grad_wq, &mut state.attention_state.wq_state);
                opt_step!(&layer.attention.wk.weight, &acc_layer.grad_wk, &mut state.attention_state.wk_state);
                opt_step!(&layer.attention.wv.weight, &acc_layer.grad_wv, &mut state.attention_state.wv_state);
                opt_step!(&layer.attention.wo.weight, &acc_layer.grad_wo, &mut state.attention_state.wo_state);
                opt_step!(&layer.ffn.w_gate.weight, &acc_layer.grad_w_gate, &mut state.ffn_state.w_gate_state);
                opt_step!(&layer.ffn.w_up.weight, &acc_layer.grad_w_up, &mut state.ffn_state.w_up_state);
                opt_step!(&layer.ffn.w_down.weight, &acc_layer.grad_w_down, &mut state.ffn_state.w_down_state);
            }

            // Zero accumulated gradients for next cycle
            zero_tensor(&acc.grad_embed).unwrap();
            zero_tensor(&acc.grad_final_norm).unwrap();
            for acc_layer in &acc.layer_grads {
                zero_tensor(&acc_layer.grad_attn_norm).unwrap();
                zero_tensor(&acc_layer.grad_ffn_norm).unwrap();
                zero_tensor(&acc_layer.grad_wq).unwrap();
                zero_tensor(&acc_layer.grad_wk).unwrap();
                zero_tensor(&acc_layer.grad_wv).unwrap();
                zero_tensor(&acc_layer.grad_wo).unwrap();
                zero_tensor(&acc_layer.grad_w_gate).unwrap();
                zero_tensor(&acc_layer.grad_w_up).unwrap();
                zero_tensor(&acc_layer.grad_w_down).unwrap();
            }

            self.step += 1;

            if self.config.async_gpu {
                CommandBatch::commit_async();
            } else {
                CommandBatch::end();
            }
            Profiler::end_step();
            return (avg_loss, total_grad_norm);
        }

        // Should not reach here if accumulated_grads is set correctly
        (loss, 0.0)
    }

    /// Apply optimizer directly without accumulation.
    fn apply_optimizer_direct(
        &mut self,
        loss: f32,
        layer_grads: Vec<LayerGradients>,
        grad_embed: Tensor,
        grad_final_norm: Tensor,
    ) -> (f32, f32) {
        // Compute gradient norm and clip
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

        if clip_scale != 1.0 {
            scale_gradients_inplace(&all_grads, clip_scale);
        }

        // ========== Apply optimizer to all parameters ==========
        Profiler::set_phase(Phase::Optimizer);
        let lr = self.scheduler.get_lr(self.step);
        self.optimizer.set_lr(lr);
        let use_bf16 = self.config.use_bf16;

        macro_rules! opt_step {
            ($weights:expr, $grads:expr, $state:expr) => {
                if use_bf16 {
                    self.optimizer.step_bf16($weights, $grads, $state);
                } else {
                    self.optimizer.step($weights, $grads, $state);
                }
            };
        }

        opt_step!(&self.model.embed_tokens, &grad_embed, &mut self.model_state.embed_state);
        opt_step!(&self.model.final_norm, &grad_final_norm, &mut self.model_state.final_norm_state);

        for (layer_idx, grads) in layer_grads.iter().enumerate() {
            let layer = &self.model.layers[layer_idx];
            let state = &mut self.model_state.layer_states[layer_idx];

            opt_step!(&layer.attn_norm, &grads.grad_attn_norm, &mut state.attn_norm_state);
            opt_step!(&layer.ffn_norm, &grads.grad_ffn_norm, &mut state.ffn_norm_state);
            opt_step!(&layer.attention.wq.weight, &grads.grad_wq, &mut state.attention_state.wq_state);
            opt_step!(&layer.attention.wk.weight, &grads.grad_wk, &mut state.attention_state.wk_state);
            opt_step!(&layer.attention.wv.weight, &grads.grad_wv, &mut state.attention_state.wv_state);
            opt_step!(&layer.attention.wo.weight, &grads.grad_wo, &mut state.attention_state.wo_state);
            opt_step!(&layer.ffn.w_gate.weight, &grads.grad_w_gate, &mut state.ffn_state.w_gate_state);
            opt_step!(&layer.ffn.w_up.weight, &grads.grad_w_up, &mut state.ffn_state.w_up_state);
            opt_step!(&layer.ffn.w_down.weight, &grads.grad_w_down, &mut state.ffn_state.w_down_state);
        }

        self.step += 1;

        if self.config.async_gpu {
            CommandBatch::commit_async();
        } else {
            CommandBatch::end();
        }

        Profiler::end_step();
        (loss, total_grad_norm)
    }
}
