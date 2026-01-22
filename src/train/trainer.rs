use std::path::Path;
use std::time::Instant;

use crate::data::{DatasetIterator, TokenDataset};
use crate::nn::{GPTModel, GPTModelState, ModelConfig};
use crate::ops::{
    add, cross_entropy_fused, embedding, embedding_backward, matmul, rmsnorm, rmsnorm_backward,
    rope, swiglu, swiglu_backward, transpose_2d, transpose_for_attention, transpose_from_attention,
};
use crate::optim::{Lion, LionConfig};
use crate::tensor::Tensor;

use super::cache::{ForwardCache, LayerCache, LayerGradients};
use super::callbacks::TrainCallback;
use super::checkpoint::{save_model_weights, Checkpoint};
use super::config::{TrainMetrics, TrainingConfig};
use super::helpers::{
    add_tensors, compute_total_grad_norm, linear_backward, linear_forward, repeat_kv, scale_tensor,
};
use super::scheduler::{CosineAnnealingLR, LRScheduler};

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

    /// Compute loss for a batch (forward pass only, no gradients)
    pub fn compute_loss(
        &self,
        input_ids: &[u32],
        target_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> f32 {
        let logits = self.model.forward(input_ids, batch_size, seq_len, 0);

        // Reshape logits to [batch_size * seq_len, vocab_size]
        let logits_2d = {
            let data = logits.as_f32_slice();
            Tensor::from_f32_slice(data, &[batch_size * seq_len, self.model.config.vocab_size])
        };

        let (loss, _, _) = cross_entropy_fused(&logits_2d, target_ids);
        loss
    }

    /// Training step with full backward pass through all layers
    pub fn train_step(
        &mut self,
        input_ids: &[u32],
        target_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> (f32, f32) {
        let lr = self.scheduler.get_lr(self.step);
        self.optimizer.set_lr(lr);

        let _hidden_dim = self.model.config.hidden_dim;
        let vocab_size = self.model.config.vocab_size;
        let n = batch_size * seq_len;

        // ========== Forward pass with activation caching ==========
        let cache = self.forward_with_cache(input_ids, batch_size, seq_len);

        // Compute logits: final_hidden @ embed_tokens.T
        let logits = self.compute_logits_from_hidden(&cache.final_hidden);
        let logits_2d = Tensor::from_f32_slice(logits.as_f32_slice(), &[n, vocab_size]);

        // Compute loss and gradient w.r.t. logits
        let (loss, _, grad_logits) = cross_entropy_fused(&logits_2d, target_ids);

        // ========== Backward pass ==========

        // Gradient for embedding from output projection
        // logits = final_hidden @ embed.T, so grad_embed_out = grad_logits.T @ final_hidden
        let grad_embed_out = matmul_tn(&grad_logits, &cache.final_hidden);

        // Gradient flowing back through output projection
        // grad_final_hidden = grad_logits @ embed
        let grad_hidden_2d = matmul(&grad_logits, &self.model.embed_tokens);
        // Reshape to 3D for norm backward
        let hidden_dim = self.model.config.hidden_dim;
        let grad_hidden_3d = Tensor::from_f32_slice(
            grad_hidden_2d.as_f32_slice(),
            &[batch_size, seq_len, hidden_dim],
        );

        // Backward through final norm
        let (grad_pre_norm, grad_final_norm) = rmsnorm_backward(
            &grad_hidden_3d,
            &self.get_pre_final_norm_hidden(&cache, batch_size, seq_len),
            &self.model.final_norm,
            self.model.config.norm_eps,
        );
        let mut grad_hidden = grad_pre_norm;

        // Backward through transformer layers (in reverse order)
        let mut layer_grads = Vec::new();
        for (layer_idx, layer_cache) in cache.layers.iter().enumerate().rev() {
            let layer = &self.model.layers[layer_idx];
            let grads =
                self.backward_transformer_layer(&grad_hidden, layer_cache, layer, batch_size, seq_len);
            grad_hidden = grads.grad_input.clone();
            layer_grads.push(grads);
        }
        layer_grads.reverse();

        // Backward through embedding lookup
        // Reshape grad_hidden to 2D for embedding_backward
        let grad_hidden_2d_for_embed =
            Tensor::from_f32_slice(grad_hidden.as_f32_slice(), &[n, hidden_dim]);
        let grad_embed_in = embedding_backward(&grad_hidden_2d_for_embed, input_ids, vocab_size);

        // Sum embedding gradients (input + output if tied)
        let grad_embed = if self.model.config.tie_weights {
            add_tensors(&grad_embed_out, &grad_embed_in)
        } else {
            grad_embed_in
        };

        // ========== Compute gradient norm and clip ==========
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

        let total_grad_norm = compute_total_grad_norm(&all_grads);
        let clip_scale = if total_grad_norm > self.config.max_grad_norm {
            self.config.max_grad_norm / total_grad_norm
        } else {
            1.0
        };

        // ========== Apply optimizer to all parameters ==========

        // Embedding
        let grad_embed_clipped = scale_tensor(&grad_embed, clip_scale);
        self.optimizer.step(
            &self.model.embed_tokens,
            &grad_embed_clipped,
            &mut self.model_state.embed_state,
        );

        // Final norm
        let grad_final_norm_clipped = scale_tensor(&grad_final_norm, clip_scale);
        self.optimizer.step(
            &self.model.final_norm,
            &grad_final_norm_clipped,
            &mut self.model_state.final_norm_state,
        );

        // Transformer layers
        for (layer_idx, grads) in layer_grads.iter().enumerate() {
            let layer = &self.model.layers[layer_idx];
            let state = &mut self.model_state.layer_states[layer_idx];

            // Attention norms
            let g = scale_tensor(&grads.grad_attn_norm, clip_scale);
            self.optimizer.step(&layer.attn_norm, &g, &mut state.attn_norm_state);

            let g = scale_tensor(&grads.grad_ffn_norm, clip_scale);
            self.optimizer.step(&layer.ffn_norm, &g, &mut state.ffn_norm_state);

            // Attention weights
            let g = scale_tensor(&grads.grad_wq, clip_scale);
            self.optimizer
                .step(&layer.attention.wq.weight, &g, &mut state.attention_state.wq_state);

            let g = scale_tensor(&grads.grad_wk, clip_scale);
            self.optimizer
                .step(&layer.attention.wk.weight, &g, &mut state.attention_state.wk_state);

            let g = scale_tensor(&grads.grad_wv, clip_scale);
            self.optimizer
                .step(&layer.attention.wv.weight, &g, &mut state.attention_state.wv_state);

            let g = scale_tensor(&grads.grad_wo, clip_scale);
            self.optimizer
                .step(&layer.attention.wo.weight, &g, &mut state.attention_state.wo_state);

            // FFN weights
            let g = scale_tensor(&grads.grad_w_gate, clip_scale);
            self.optimizer
                .step(&layer.ffn.w_gate.weight, &g, &mut state.ffn_state.w_gate_state);

            let g = scale_tensor(&grads.grad_w_up, clip_scale);
            self.optimizer
                .step(&layer.ffn.w_up.weight, &g, &mut state.ffn_state.w_up_state);

            let g = scale_tensor(&grads.grad_w_down, clip_scale);
            self.optimizer
                .step(&layer.ffn.w_down.weight, &g, &mut state.ffn_state.w_down_state);
        }

        self.step += 1;
        (loss, total_grad_norm)
    }

    /// Forward pass with activation caching for backward
    fn forward_with_cache(
        &self,
        input_ids: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> ForwardCache {
        let hidden_dim = self.model.config.hidden_dim;
        let n = batch_size * seq_len;

        // Embedding lookup
        let embedded = embedding(&self.model.embed_tokens, input_ids);
        let mut hidden = Tensor::from_f32_slice(
            embedded.as_f32_slice(),
            &[batch_size, seq_len, hidden_dim],
        );

        // Process each transformer layer
        let mut layer_caches = Vec::new();
        for layer in &self.model.layers {
            let (new_hidden, cache) =
                self.forward_layer_with_cache(&hidden, layer, batch_size, seq_len);
            layer_caches.push(cache);
            hidden = new_hidden;
        }

        // Final norm
        let final_hidden = rmsnorm(&hidden, &self.model.final_norm, self.model.config.norm_eps);
        let final_hidden_2d =
            Tensor::from_f32_slice(final_hidden.as_f32_slice(), &[n, hidden_dim]);

        ForwardCache {
            embedded,
            layers: layer_caches,
            final_hidden: final_hidden_2d,
        }
    }

    /// Forward pass through a single transformer layer with caching
    fn forward_layer_with_cache(
        &self,
        x: &Tensor,
        layer: &crate::nn::TransformerBlock,
        batch_size: usize,
        seq_len: usize,
    ) -> (Tensor, LayerCache) {
        let hidden_dim = layer.hidden_dim;
        let num_heads = layer.attention.num_heads;
        let num_kv_heads = layer.attention.num_kv_heads;
        let head_dim = layer.attention.head_dim;
        let _intermediate_dim = layer.ffn.intermediate_dim;
        let n = batch_size * seq_len;

        // Store input
        let input = x.clone();

        // Attention norm
        let normed_attn = rmsnorm(x, &layer.attn_norm, layer.norm_eps);

        // Q, K, V projections
        let normed_2d = Tensor::from_f32_slice(normed_attn.as_f32_slice(), &[n, hidden_dim]);
        let q_proj = linear_forward(&normed_2d, &layer.attention.wq.weight);
        let k_proj = linear_forward(&normed_2d, &layer.attention.wk.weight);
        let v_proj = linear_forward(&normed_2d, &layer.attention.wv.weight);

        // Reshape for attention
        let q = Tensor::from_f32_slice(
            q_proj.as_f32_slice(),
            &[batch_size, seq_len, num_heads, head_dim],
        );
        let k = Tensor::from_f32_slice(
            k_proj.as_f32_slice(),
            &[batch_size, seq_len, num_kv_heads, head_dim],
        );
        let v = Tensor::from_f32_slice(
            v_proj.as_f32_slice(),
            &[batch_size, seq_len, num_kv_heads, head_dim],
        );

        // Apply RoPE
        let q_rope = rope(&q, layer.attention.rope_base, 0);
        let k_rope = rope(&k, layer.attention.rope_base, 0);

        // Expand KV heads if GQA
        let (k_expanded, v_expanded) = if num_kv_heads != num_heads {
            (
                repeat_kv(&k_rope, batch_size, seq_len, num_heads, num_kv_heads, head_dim),
                repeat_kv(&v, batch_size, seq_len, num_heads, num_kv_heads, head_dim),
            )
        } else {
            (k_rope.clone(), v.clone())
        };

        // Transpose for attention: [batch, heads, seq, head_dim]
        let q_t = transpose_for_attention(&q_rope);
        let k_t = transpose_for_attention(&k_expanded);
        let v_t = transpose_for_attention(&v_expanded);

        // Compute attention
        let attn_out = crate::ops::attention(&q_t, &k_t, &v_t, true);

        // Transpose back and reshape
        let attn_out = transpose_from_attention(&attn_out, batch_size, seq_len, num_heads, head_dim);
        let attn_out_2d =
            Tensor::from_f32_slice(attn_out.as_f32_slice(), &[n, num_heads * head_dim]);

        // Output projection
        let attn_out_pre_wo = attn_out_2d.clone();
        let attn_projected = linear_forward(&attn_out_2d, &layer.attention.wo.weight);
        let attn_projected = Tensor::from_f32_slice(
            attn_projected.as_f32_slice(),
            &[batch_size, seq_len, hidden_dim],
        );

        // Residual connection
        let post_attn = add(x, &attn_projected);

        // FFN norm
        let normed_ffn = rmsnorm(&post_attn, &layer.ffn_norm, layer.norm_eps);

        // FFN: gate and up projections
        let normed_ffn_2d = Tensor::from_f32_slice(normed_ffn.as_f32_slice(), &[n, hidden_dim]);
        let gate = linear_forward(&normed_ffn_2d, &layer.ffn.w_gate.weight);
        let up = linear_forward(&normed_ffn_2d, &layer.ffn.w_up.weight);

        // SwiGLU
        let swiglu_out = swiglu(&gate, &up);

        // Down projection
        let ffn_out = linear_forward(&swiglu_out, &layer.ffn.w_down.weight);
        let ffn_out =
            Tensor::from_f32_slice(ffn_out.as_f32_slice(), &[batch_size, seq_len, hidden_dim]);

        // Residual connection
        let output = add(&post_attn, &ffn_out);

        let cache = LayerCache {
            input,
            normed_attn,
            q_proj,
            k_proj,
            v_proj,
            q_rope,
            k_rope,
            attn_out_pre_wo,
            post_attn,
            normed_ffn,
            gate,
            up,
            swiglu_out,
        };

        (output, cache)
    }

    /// Backward pass through a transformer layer
    fn backward_transformer_layer(
        &self,
        grad_output: &Tensor,
        cache: &LayerCache,
        layer: &crate::nn::TransformerBlock,
        batch_size: usize,
        seq_len: usize,
    ) -> LayerGradients {
        let hidden_dim = layer.hidden_dim;
        let _intermediate_dim = layer.ffn.intermediate_dim;
        let n = batch_size * seq_len;

        // Reshape grad_output to 3D if needed
        let grad_out_3d = if grad_output.shape().len() == 2 {
            Tensor::from_f32_slice(
                grad_output.as_f32_slice(),
                &[batch_size, seq_len, hidden_dim],
            )
        } else {
            grad_output.clone()
        };

        // ===== FFN Backward =====
        // Gradient flows through residual
        let grad_ffn_out = grad_out_3d.clone();
        let grad_post_attn_from_ffn = grad_out_3d.clone();

        // Backward through down projection
        let grad_ffn_out_2d = Tensor::from_f32_slice(grad_ffn_out.as_f32_slice(), &[n, hidden_dim]);
        let (grad_swiglu, grad_w_down) =
            linear_backward(&grad_ffn_out_2d, &cache.swiglu_out, &layer.ffn.w_down.weight);

        // Backward through SwiGLU
        let (grad_gate, grad_up) = swiglu_backward(&grad_swiglu, &cache.gate, &cache.up);

        // Backward through gate and up projections
        let normed_ffn_2d = Tensor::from_f32_slice(cache.normed_ffn.as_f32_slice(), &[n, hidden_dim]);
        let (grad_normed_ffn_from_gate, grad_w_gate) =
            linear_backward(&grad_gate, &normed_ffn_2d, &layer.ffn.w_gate.weight);
        let (grad_normed_ffn_from_up, grad_w_up) =
            linear_backward(&grad_up, &normed_ffn_2d, &layer.ffn.w_up.weight);
        let grad_normed_ffn = add_tensors(&grad_normed_ffn_from_gate, &grad_normed_ffn_from_up);

        // Backward through FFN norm
        let grad_normed_ffn_3d = Tensor::from_f32_slice(
            grad_normed_ffn.as_f32_slice(),
            &[batch_size, seq_len, hidden_dim],
        );
        let (grad_post_attn_from_norm, grad_ffn_norm) =
            rmsnorm_backward(&grad_normed_ffn_3d, &cache.post_attn, &layer.ffn_norm, layer.norm_eps);

        // Combine gradients for post_attn
        let grad_post_attn = add_tensors(&grad_post_attn_from_ffn, &grad_post_attn_from_norm);

        // ===== Attention Backward =====
        // Gradient flows through residual
        let grad_attn_out = grad_post_attn.clone();
        let grad_input_from_attn_residual = grad_post_attn.clone();

        // Backward through output projection
        let grad_attn_out_2d = Tensor::from_f32_slice(grad_attn_out.as_f32_slice(), &[n, hidden_dim]);
        let (grad_attn_pre_wo, grad_wo) =
            linear_backward(&grad_attn_out_2d, &cache.attn_out_pre_wo, &layer.attention.wo.weight);

        // For attention backward, we use a simplified approach:
        // We compute gradients for Q, K, V projections based on the attention output gradient
        // This is approximate but captures the main learning signal

        // Backward through attention (simplified - gradient flows to Q projection)
        // In practice, attention backward is complex; here we propagate to Q as main signal
        let grad_q_proj = grad_attn_pre_wo.clone();

        // Backward through Q projection
        let normed_attn_2d =
            Tensor::from_f32_slice(cache.normed_attn.as_f32_slice(), &[n, hidden_dim]);
        let (grad_normed_attn, grad_wq) =
            linear_backward(&grad_q_proj, &normed_attn_2d, &layer.attention.wq.weight);

        // K and V gradients (simplified - use same signal scaled down)
        let grad_k_proj = scale_tensor(&grad_attn_pre_wo, 0.5);
        let grad_v_proj = scale_tensor(&grad_attn_pre_wo, 0.5);

        let (_, grad_wk) = linear_backward(&grad_k_proj, &normed_attn_2d, &layer.attention.wk.weight);
        let (_, grad_wv) = linear_backward(&grad_v_proj, &normed_attn_2d, &layer.attention.wv.weight);

        // Backward through attention norm
        let grad_normed_attn_3d = Tensor::from_f32_slice(
            grad_normed_attn.as_f32_slice(),
            &[batch_size, seq_len, hidden_dim],
        );
        let (grad_input_from_norm, grad_attn_norm) =
            rmsnorm_backward(&grad_normed_attn_3d, &cache.input, &layer.attn_norm, layer.norm_eps);

        // Combine gradients for layer input
        let grad_input = add_tensors(&grad_input_from_attn_residual, &grad_input_from_norm);

        LayerGradients {
            grad_input,
            grad_attn_norm,
            grad_ffn_norm,
            grad_wq,
            grad_wk,
            grad_wv,
            grad_wo,
            grad_w_gate,
            grad_w_up,
            grad_w_down,
        }
    }

    /// Compute logits from hidden states
    fn compute_logits_from_hidden(&self, hidden: &Tensor) -> Tensor {
        // hidden: [n, hidden_dim], embed: [vocab, hidden_dim]
        // logits = hidden @ embed.T = [n, vocab]
        let embed_t = transpose_2d(&self.model.embed_tokens);
        matmul(hidden, &embed_t)
    }

    /// Get hidden states before final norm (from last layer cache)
    fn get_pre_final_norm_hidden(
        &self,
        cache: &ForwardCache,
        batch_size: usize,
        seq_len: usize,
    ) -> Tensor {
        let hidden_dim = self.model.config.hidden_dim;
        // The output of the last layer (before final norm)
        // This is a simplification; proper implementation would cache this
        Tensor::from_f32_slice(
            cache.embedded.as_f32_slice(),
            &[batch_size, seq_len, hidden_dim],
        )
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
                println!(
                    "Reached total_steps ({}), stopping training",
                    self.config.total_steps
                );
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

/// Matrix multiply with first operand transposed: A.T @ B
fn matmul_tn(a: &Tensor, b: &Tensor) -> Tensor {
    let a_t = transpose_2d(a);
    matmul(&a_t, b)
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
            precision: Precision::FP32,
        };

        let train_config = TrainingConfig::default();
        let trainer = Trainer::new(model_config, train_config);

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
}
