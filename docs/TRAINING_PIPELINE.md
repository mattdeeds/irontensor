# IronTensor Training Pipeline

This document provides a comprehensive overview of the IronTensor training pipeline, covering every component from data loading through optimization.

## Table of Contents

1. [Training Entry Point](#1-training-entry-point)
2. [Data Loading](#2-data-loading)
3. [Forward Pass](#3-forward-pass)
4. [Loss Computation](#4-loss-computation)
5. [Backward Pass](#5-backward-pass)
6. [Optimizer (Lion)](#6-optimizer-lion)
7. [Learning Rate Scheduling](#7-learning-rate-scheduling)
8. [Checkpointing](#8-checkpointing)
9. [Profiling and Logging](#9-profiling-and-logging)
10. [Command Buffer Batching](#10-command-buffer-batching)
11. [Complete Training Step Flow](#11-complete-training-step-flow)
12. [Key Data Structures](#12-key-data-structures)
13. [Performance Optimizations](#13-performance-optimizations)

---

## 1. Training Entry Point

**File:** `src/train/trainer.rs`

The training pipeline is initialized through the `Trainer` struct:

```rust
let mut trainer = Trainer::new(&model_config, &train_config);
```

### Initialization Process

1. Creates a fresh `GPTModel` (or loads from checkpoint via `from_checkpoint()`)
2. Initializes the Lion optimizer with training parameters
3. Creates a cosine annealing LR scheduler with warmup
4. Initializes model state for gradient accumulation (momentum tensors)

### Configuration

**TrainingConfig** (`src/train/config.rs`):
- `learning_rate` - Base learning rate
- `beta1`, `beta2` - Lion optimizer betas (default: 0.9, 0.99)
- `weight_decay` - Decoupled weight decay (default: 0.1)
- `max_grad_norm` - Gradient clipping threshold
- `warmup_steps`, `total_steps` - Scheduler parameters
- `use_bf16` - Enable BF16 training
- `async_gpu` - Enable CPU/GPU overlap
- `dropout_enabled` - Enable dropout layers
- `accumulation_steps` - Number of micro-batches to accumulate (default: 1)
- `early_stopping_patience` - Stop after N evals without improvement (None = disabled)
- `early_stopping_min_delta` - Minimum improvement threshold (default: 0.0)
- `checkpoint_config` - Activation checkpointing configuration (see [Activation Checkpointing](#activation-checkpointing))

**ModelConfig** (`src/nn/model.rs`):
- `vocab_size`, `hidden_dim`, `num_layers`, `num_heads`
- `num_kv_heads` - For grouped-query attention
- `intermediate_dim` - FFN hidden dimension
- `max_seq_len`, `rope_base`, `norm_eps`
- `tie_weights` - Share embedding and output weights
- `precision` - FP32 or BF16

### Model Presets

| Preset | Hidden Dim | Layers | Heads | Vocab |
|--------|------------|--------|-------|-------|
| `tiny()` | 256 | 4 | 4 | 32k |
| `small()` | 512 | 8 | 8 | 32k |
| `medium()` | 1024 | 16 | 16 | 32k |
| `shakespeare()` | 256 | 6 | 4 | 2k |

---

## 2. Data Loading

**File:** `src/data/dataset.rs`

### Memory-Mapped Format

IronTensor uses memory-mapped files for zero-copy data access:

```
Header: 8 bytes (u64 little-endian) = number of tokens
Data:   tokens as u32 (4 bytes each, little-endian)
```

### API

```rust
// Create dataset from token array
TokenDataset::create(path, tokens)?;

// Load existing dataset
let dataset = TokenDataset::open(path, seq_len)?;

// Get single batch
let (input_ids, target_ids) = dataset.get_batch(seq_idx);

// Batching iterator with shuffling
let mut iter = DatasetIterator::new(&dataset, batch_size, shuffle);
iter.shuffle(epoch as u64);  // Deterministic shuffle per epoch

for (input_ids, target_ids) in &mut iter {
    // input_ids:  [batch_size * seq_len]
    // target_ids: [batch_size * seq_len]
}
```

### Batching Strategy

Each sequence requires `seq_len + 1` tokens:
- **Input:** `tokens[i * seq_len : (i+1) * seq_len]`
- **Target:** `tokens[i * seq_len + 1 : (i+1) * seq_len + 1]`

Shuffling uses PHI_FRAC pseudo-random permutation for deterministic but well-distributed ordering.

---

## 3. Forward Pass

**File:** `src/train/forward.rs`

### Complete Flow

```
input_ids [batch_size * seq_len]
    │
    ▼
┌─────────────────────────────────────┐
│  Embedding Lookup                    │
│  → [batch_size, seq_len, hidden_dim] │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  [Embedding Dropout] (if enabled)    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Transformer Layer × num_layers      │
│  ┌─────────────────────────────────┐ │
│  │ Attention Block                  │ │
│  │  ├─ RMSNorm (attention)          │ │
│  │  ├─ Q, K, V projections          │ │
│  │  ├─ RoPE positional encoding     │ │
│  │  ├─ FlashAttention (causal)      │ │
│  │  ├─ Output projection (wo)       │ │
│  │  ├─ [Attention Dropout]          │ │
│  │  └─ Residual connection          │ │
│  ├─────────────────────────────────┤ │
│  │ FFN Block                        │ │
│  │  ├─ RMSNorm (FFN)                │ │
│  │  ├─ Gate & Up projections        │ │
│  │  ├─ SwiGLU activation            │ │
│  │  ├─ Down projection              │ │
│  │  ├─ [FFN Dropout]                │ │
│  │  └─ Residual connection          │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Final RMSNorm                       │
└─────────────────────────────────────┘
    │
    ▼
Output [batch_size * seq_len, hidden_dim]
```

### Activation Caching

All intermediate activations are cached in `ForwardCache` for the backward pass:

```rust
pub struct ForwardCache {
    pub layers: Vec<LayerCache>,      // Per-layer activations
    pub pre_final_norm: Tensor,       // Before final norm
    pub final_hidden: Tensor,         // After final norm
}

pub struct LayerCache {
    pub input: Tensor,                // Layer input
    pub normed_attn: Tensor,          // After attention norm
    pub q_for_attn: Tensor,           // Q after RoPE
    pub k_for_attn: Tensor,           // K after RoPE
    pub v_for_attn: Tensor,           // V projection
    pub attn_out_pre_wo: Tensor,      // Before output projection
    pub post_attn: Tensor,            // After attention + residual
    pub normed_ffn: Tensor,           // After FFN norm
    pub gate: Tensor,                 // Gate projection
    pub up: Tensor,                   // Up projection
    pub swiglu_out: Tensor,           // After SwiGLU
    // Dropout seeds for reproducible backward
}
```

### Precision Handling

- BF16 weights automatically converted to FP32 via `ensure_fp32()` during forward
- Gradients always computed in FP32 for numerical stability
- Weights updated in BF16 if `use_bf16` is enabled

---

## 4. Loss Computation

**Files:** `src/train/trainer.rs`, `src/ops/backward/loss.rs`, `src/ops/fused_linear_cross_entropy.rs`

### Standard Cross-Entropy

```rust
// Compute logits from hidden states
let logits = compute_logits_from_hidden(&cache.final_hidden);
// Shape: [batch_size, seq_len, vocab_size]

// Reshape for loss computation
let logits_2d = logits.view(&[batch_size * seq_len, vocab_size]);

// Fused cross-entropy (computes softmax + loss + gradients)
let (loss, log_softmax, grad_logits) = cross_entropy_fused(&logits_2d, &target_ids);
```

### Fused Linear Cross-Entropy (Memory Optimization)

For large vocabularies, materializing full logits is expensive (~200MB for 1024×50k).

`FusedLinearCrossEntropy` combines output projection + loss in one pass:

```rust
let (loss, grad_hidden, grad_weight) = fused_linear_cross_entropy(
    &hidden_states,    // [batch*seq, hidden_dim]
    &output_weights,   // [vocab_size, hidden_dim]
    &targets,          // [batch*seq]
);
```

**Memory savings:** 200MB → ~4MB by never materializing full logits tensor.

---

## 5. Backward Pass

**File:** `src/train/backward.rs`, `src/ops/backward/`

### Overview

Gradients flow backward through all layers in reverse order, computing gradients for every learnable parameter.

### Per-Layer Backward Flow

```
grad_output [batch, seq, hidden_dim]
    │
    ▼
┌─────────────────────────────────────┐
│  FFN Backward (reversed order)       │
│  ├─ Backward through FFN Dropout     │
│  ├─ Backward through down projection │
│  │   → grad_w_down, grad_swiglu      │
│  ├─ Backward through SwiGLU          │
│  │   → grad_gate, grad_up            │
│  ├─ Backward through gate/up         │
│  │   → grad_w_gate, grad_w_up        │
│  └─ Backward through FFN Norm        │
│      → grad_ffn_norm                 │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Attention Backward                  │
│  ├─ Backward through Attn Dropout    │
│  ├─ Backward through wo projection   │
│  │   → grad_wo, grad_attn            │
│  ├─ Backward through FlashAttention  │
│  │   → grad_q, grad_k, grad_v        │
│  ├─ Backward through Q, K, V         │
│  │   → grad_wq, grad_wk, grad_wv     │
│  └─ Backward through Attn Norm       │
│      → grad_attn_norm                │
└─────────────────────────────────────┘
    │
    ▼
Combine residual gradients → grad_input (for next layer)
```

### Complete Backward Flow

```
Loss scalar
    │ (CE backward)
    ▼
grad_logits [batch*seq, vocab]
    │ (output projection backward)
    ├──────────────────────────────────────┐
    ▼                                      ▼
grad_hidden [batch*seq, hidden]    grad_embed_out [vocab, hidden]
    │ (final norm backward)                │
    ▼                                      │
grad_pre_norm [batch, seq, hidden]         │
    │                                      │
    ▼                                      │
(through layers in reverse)                │
    │                                      │
    ▼                                      │
grad_hidden [batch*seq, hidden]            │
    │ (embedding backward)                 │
    ▼                                      │
grad_embed_in [vocab, hidden]              │
    │                                      │
    └──────────────────────────────────────┘
                    │
                    ▼ (combine if tie_weights)
            grad_embed [vocab, hidden]
```

### GPU-Based Gradient Norm

Instead of copying gradients to CPU for norm computation:

```rust
let total_grad_norm = total_l2_norm_gpu(&all_grads);
```

This dispatches reduction kernels to GPU, syncs once, and reads only partial sums—much faster than CPU-based accumulation.

---

## 6. Optimizer (Lion)

**File:** `src/optim/lion/`

### Update Rule

Lion (EvoLved Sign Momentum) uses sign-based updates:

```
update = sign(β₂ · m + (1 - β₂) · g)
m_new  = β₁ · m + (1 - β₁) · g
w_new  = w - lr · update - lr · weight_decay · w
```

### Key Properties

- **Sign-based:** Only gradient direction matters, not magnitude
- **Memory efficient:** Stores only momentum (not variance like Adam)
- **Defaults:** β₁=0.9, β₂=0.99, weight_decay=0.1

### API

```rust
// Standard FP32 step
optimizer.step(&mut weights, &gradients, &mut state);

// Mixed precision (BF16 weights, FP32 gradients)
optimizer.step_bf16(&mut weights, &gradients, &mut state);

// Per-parameter LR scaling
optimizer.step_scaled(&mut weights, &gradients, lr_scale, &mut state);

// Utilities
Lion::zero_gradients(&mut grad);              // GPU-accelerated zeroing
let norm = Lion::grad_norm(&grad);             // L2 norm
Lion::clip_grad_norm(&mut grad, max_norm);    // In-place clipping
```

### GPU Implementation

Metal shaders handle the update:
- `lion_step` kernel for FP32
- `lion_step_bf16` kernel for mixed precision

---

## 7. Learning Rate Scheduling

**File:** `src/train/scheduler.rs`

All schedulers implement:

```rust
trait LRScheduler: Send {
    fn get_lr(&self, step: usize) -> f32;
}
```

### Available Schedulers

| Scheduler | Formula |
|-----------|---------|
| `ConstantLR` | `lr = base_lr` |
| `WarmupConstantLR` | Linear warmup → constant |
| `CosineAnnealingLR` | Linear warmup → cosine decay (default) |
| `LinearDecayLR` | Linear warmup → linear decay |
| `InverseSqrtLR` | `lr = base_lr × min(1/√step, step × warmup⁻¹·⁵)` |

### Cosine Annealing (Default)

```
During warmup (step < warmup_steps):
    lr = max_lr × (step + 1) / warmup_steps

After warmup:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × progress))
```

Default `min_lr = 0.1 × max_lr`

### Integration

```rust
// Created automatically in Trainer::new()
let scheduler = CosineAnnealingLR::new(
    max_lr,
    warmup_steps,
    total_steps,
    min_lr,
);

// Applied every step
let lr = scheduler.get_lr(step);
optimizer.set_lr(lr);
```

---

## 8. Checkpointing

**File:** `src/train/checkpoint.rs`

### Binary Format

```
┌─────────────────────────────────────┐
│ Header                               │
│   Magic: 0x49524F4E ("IRON")        │
│   Version: 1                         │
├─────────────────────────────────────┤
│ Metadata                             │
│   step (u64), epoch (u64)           │
│   best_val_loss (f32)               │
│   learning_rate (f32)               │
├─────────────────────────────────────┤
│ Model Config                         │
│   vocab_size, hidden_dim, ...       │
│   tie_weights, precision            │
├─────────────────────────────────────┤
│ Weights                              │
│   embed_tokens [vocab, hidden]       │
│   Per layer:                         │
│     wq, wk, wv, wo                   │
│     w_gate, w_up, w_down             │
│     attn_norm, ffn_norm              │
│   final_norm                         │
└─────────────────────────────────────┘
```

### API

```rust
// Save checkpoint (weights only)
trainer.save_checkpoint("model.bin")?;

// Save checkpoint with optimizer state (for stable training resumption)
trainer.save_checkpoint_with_optimizer("model_full.bin")?;

// Resume training (automatically loads optimizer state if present)
let trainer = Trainer::from_checkpoint("model_full.bin", &train_config)?;
```

### Optimizer State Checkpointing

Optimizer state (Lion momentum tensors) can be saved alongside model weights for stable training resumption:

```rust
// Without optimizer state - momentum resets to zero on resume
trainer.save_checkpoint("weights_only.bin")?;

// With optimizer state - training continues smoothly
trainer.save_checkpoint_with_optimizer("full_checkpoint.bin")?;
```

**Why it matters:** When momentum is reset to zero, the optimizer needs several steps to "warm up" again, potentially causing:
- Temporary training instability
- Slightly different convergence path
- Wasted compute re-learning momentum statistics

### Notes

- BF16 weights automatically converted to FP32 on save for compatibility
- Optimizer state (momentum tensors) can optionally be saved/loaded
- Model config is embedded in checkpoint for validation
- Checkpoints without optimizer state are smaller but may cause brief instability on resume

### Evaluation Mode

The model tracks training mode to properly disable dropout during evaluation:

```rust
// Model starts in training mode by default
let mut model = GPTModel::new(&config);
assert!(model.is_training());

// Set evaluation mode (disables dropout)
model.set_training(false);

// Trainer.evaluate() handles this automatically
let val_loss = trainer.evaluate(&val_dataset, batch_size);  // dropout disabled
```

### Early Stopping

Early stopping prevents overfitting by stopping training when validation loss stops improving:

```rust
let train_config = TrainingConfig {
    early_stopping_patience: Some(5),  // Stop after 5 evals without improvement
    early_stopping_min_delta: 0.001,   // Minimum improvement threshold
    ..Default::default()
};
```

**Behavior:**
1. After each validation, check if `val_loss < best_val_loss - min_delta`
2. If improved: save best model, reset patience counter
3. If not improved: increment patience counter
4. When patience counter >= patience: stop training

---

## 9. Profiling and Logging

### Profiling

**File:** `src/profile/`

Hierarchical profiler with three levels: **Phase → Layer → Operation**

```
Training Step
├── Forward Pass
│   ├── Layer 0
│   │   ├── RMSNorm, Matmul/Q,K,V, Attention, Matmul/O, FFN
│   ├── Layer 1...
│   └── Final Norm
├── Backward Pass
│   └── [similar breakdown]
└── Optimizer
```

#### Enabling Profiling

```bash
IRONTENSOR_LOG=1 cargo run --release  # Enables profiling + logging
```

#### API

```rust
// Initialize
Profiler::init(ProfilerConfig {
    enabled: true,
    warmup_steps: 5,      // Skip initial steps
    report_interval: 0,   // 0 = report at end only
});

// In training loop
Profiler::begin_step();
Profiler::set_phase(Phase::Forward);
Profiler::set_layer(Some(layer_idx));
// ... operations ...
Profiler::end_step();

// Get report
let report = Profiler::report();
```

#### RAII Timing Guards

```rust
use crate::profile::{timed, OpCategory};

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::Matmul, a.shape()[0] * b.shape()[1]);
    // ... operation ...
}
```

### Logging

**File:** `src/logging/`

Single JSON file per run: `logs/run_{YYYYMMDD_HHMMSS}_{model_name}.json`

#### Environment Variables

```bash
IRONTENSOR_LOG=1           # Enable logging
IRONTENSOR_LOG_DIR=./logs  # Custom directory
IRONTENSOR_LOG_OPS=1       # Include op breakdown
```

#### Output Format

```json
{
  "run_id": "20240115_143022_shakespeare",
  "model_name": "shakespeare",
  "config": { ... },
  "training": {
    "total_time_sec": 63.2,
    "total_steps": 100,
    "final_loss": 6.95,
    "best_val_loss": 7.62,
    "avg_tokens_per_sec": 6800.0,
    "steps": [
      {"step": 10, "loss": 7.61, "perplexity": 2020.7, "grad_norm": 0.97, "tokens_per_sec": 6616, ...}
    ],
    "profiler_report": {
      "total_time_ms": 45320.5,
      "avg_step_ms": 453.2,
      "phase_breakdown": {"Forward": 182.4, "Backward": 241.8, "Optimizer": 29.0},
      "top_operations": [...]
    }
  },
  "inference": [...]
}
```

#### Metrics Tracked

**Training:**
- `total_time_sec`, `total_steps`, `epochs_completed`
- `final_loss`, `best_val_loss`, `avg_tokens_per_sec`
- Per-step: loss, perplexity, grad_norm, learning_rate, tokens_per_sec, step_time_ms
- Validation: val_loss, val_perplexity (when evaluated)

**Inference:**
- `time_to_first_token_ms` (TTFT)
- `inter_token_latency_ms`
- `tokens_per_sec`
- `total_time_ms`

---

## 10. Command Buffer Batching

**File:** `src/command_batch.rs`

### Purpose

Reduce GPU synchronization overhead by batching multiple Metal operations into a single command buffer.

### Without Batching

Each GPU operation:
1. Creates command buffer
2. Encodes operation
3. Commits and waits for completion

### With Batching

Multiple operations share one command buffer:
1. `begin()` - Start accumulating
2. Operations added to shared encoder
3. `sync()` - Commit and wait only when needed
4. `end()` - Finalize

### API

```rust
// Synchronous mode
CommandBatch::begin();
// ... GPU operations (matmul, norm, etc.) ...
CommandBatch::sync();       // Commit and wait
// Read GPU data here
CommandBatch::end();

// Asynchronous mode (CPU/GPU overlap)
CommandBatch::begin();
// ... GPU operations ...
CommandBatch::commit_async();           // Return immediately
// ... CPU work (prepare next batch) ...
CommandBatch::wait_for_completion();    // Wait before reading
```

### Important Notes

- Use `tensor.view(&[...])` for reshaping during batched operations
- Avoid `Tensor::from_f32_slice(tensor.as_f32_slice(), ...)` which requires sync
- Profiler integration tracks sync wait time and op count per batch

---

## 11. Complete Training Step Flow

**Method:** `Trainer::train_step()`

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Async completion check (if async_gpu mode)               │
│    CommandBatch::wait_for_completion()                      │
├─────────────────────────────────────────────────────────────┤
│ 2. Begin command batching                                   │
│    CommandBatch::begin()                                    │
├─────────────────────────────────────────────────────────────┤
│ 3. Get current learning rate                                │
│    lr = scheduler.get_lr(step)                              │
│    optimizer.set_lr(lr)                                     │
├─────────────────────────────────────────────────────────────┤
│ 4. FORWARD PASS                                             │
│    Profiler::set_phase(Forward)                             │
│    cache = forward_with_cache(input_ids, batch_size, seq)   │
├─────────────────────────────────────────────────────────────┤
│ 5. Compute logits                                           │
│    logits = compute_logits_from_hidden(cache.final_hidden)  │
├─────────────────────────────────────────────────────────────┤
│ 6. Sync before loss computation                             │
│    CommandBatch::sync()                                     │
├─────────────────────────────────────────────────────────────┤
│ 7. Compute loss and initial gradients                       │
│    logits_2d = logits.view([batch*seq, vocab])              │
│    (loss, _, grad_logits) = cross_entropy_fused(...)        │
├─────────────────────────────────────────────────────────────┤
│ 8. BACKWARD PASS                                            │
│    Profiler::set_phase(Backward)                            │
│    a. Gradient through output projection                    │
│    b. Backward through final norm                           │
│    c. Backward through each layer (reverse order)           │
│    d. Backward through embedding                            │
│    e. Combine embedding gradients (if tie_weights)          │
├─────────────────────────────────────────────────────────────┤
│ 9. Gradient clipping                                        │
│    total_grad_norm = total_l2_norm_gpu(all_grads)           │
│    clip_scale = max(1.0, max_grad_norm / norm)              │
│    if clip_scale != 1.0: scale_gradients_inplace(...)       │
├─────────────────────────────────────────────────────────────┤
│ 10. OPTIMIZER STEP                                          │
│     Profiler::set_phase(Optimizer)                          │
│     For each parameter:                                     │
│       optimizer.step[_bf16](weights, grads, state)          │
├─────────────────────────────────────────────────────────────┤
│ 11. End command batching                                    │
│     if async_gpu: CommandBatch::commit_async()              │
│     else: CommandBatch::end()                               │
├─────────────────────────────────────────────────────────────┤
│ 12. Record metrics and increment step                       │
│     Profiler::end_step()                                    │
│     return (loss, total_grad_norm)                          │
└─────────────────────────────────────────────────────────────┘
```

### Gradient Accumulation

When `accumulation_steps > 1`, the training step is modified to accumulate gradients over multiple micro-batches:

```
Micro-batch 1: forward → backward → scale(1/N) → accumulate → return (loss, 0.0)
Micro-batch 2: forward → backward → scale(1/N) → accumulate → return (loss, 0.0)
...
Micro-batch N: forward → backward → scale(1/N) → accumulate → clip → optimize → zero accumulators → return (avg_loss, grad_norm)
```

**Configuration:**

```rust
let train_config = TrainingConfig {
    accumulation_steps: 4,  // Effective batch = batch_size * 4
    ..Default::default()
};
```

**Return behavior:**
- During accumulation (micro_step < N): `(micro_batch_loss, 0.0)`
- On optimizer step (micro_step == N): `(average_loss, grad_norm)`

**Use case:** Train with larger effective batch sizes when GPU memory is limited.

### Evaluation and Early Stopping

During training, after each epoch validation is performed:

```
1. Set model to eval mode (disables dropout)
2. Compute average loss over validation dataset
3. Restore training mode
4. Check for improvement:
   - If val_loss < best_val_loss - min_delta:
       Save best checkpoint, reset patience counter
   - Else:
       Increment patience counter
       If patience_counter >= early_stopping_patience:
           Stop training early
```

---

## 12. Key Data Structures

### Trainer

```rust
pub struct Trainer {
    pub config: TrainingConfig,
    pub model: GPTModel,
    pub optimizer: Lion,
    pub model_state: GPTModelState,    // Optimizer momentum states
    pub scheduler: Box<dyn LRScheduler>,
    pub step: usize,
    pub epoch: usize,
    pub best_val_loss: f32,
    patience_counter: usize,           // For early stopping
}
```

### GPTModelState (Optimizer State)

```rust
pub struct GPTModelState {
    pub embed_state: ParamState,
    pub layer_states: Vec<TransformerBlockState>,
    pub final_norm_state: ParamState,
    pub output_weight_state: Option<ParamState>,
}

pub struct ParamState {
    pub momentum: Tensor,  // Lion momentum buffer
}
```

### ForwardCache

```rust
pub struct ForwardCache {
    pub layers: Vec<LayerCache>,
    pub pre_final_norm: Tensor,
    pub final_hidden: Tensor,
}
```

---

## 13. Performance Optimizations

| Optimization | Description | Impact |
|--------------|-------------|--------|
| **BF16 Training** | Half memory for weights | 50% memory reduction |
| **FlashAttention** | Online softmax, tiled computation | O(n) vs O(n²) memory |
| **Fused Linear CE** | Avoid materializing logits | ~200MB → ~4MB |
| **Command Batching** | Batch GPU operations | Reduced sync overhead |
| **Memory-Mapped Data** | Zero-copy dataset access | No loading latency |
| **GPU Gradient Norm** | Compute norm on GPU | Less memory bandwidth |
| **MPS GEMM** | Metal Performance Shaders | 2-4x faster matmul |
| **Lion Optimizer** | Sign-based, no variance | Less memory than Adam |
| **Async GPU Mode** | CPU/GPU overlap | Better utilization |
| **Activation Checkpointing** | Recompute activations in backward | ~90% activation memory reduction |

### Activation Checkpointing

Activation checkpointing (gradient checkpointing) trades compute for memory by storing only layer inputs during forward pass and recomputing activations during backward pass.

**Files:**
- `src/train/checkpoint_grad.rs` - Configuration and checkpoint storage
- `src/train/forward.rs` - Checkpointing forward pass implementation
- `src/train/trainer.rs` - Integration with train_step

**Configuration:**

```rust
use irontensor::train::{TrainingConfig, CheckpointConfig};

let train_config = TrainingConfig {
    // Checkpoint all layers (maximum memory savings)
    checkpoint_config: CheckpointConfig::enabled(),
    ..Default::default()
};

// Or checkpoint every N layers (balance compute vs memory)
let train_config = TrainingConfig {
    checkpoint_config: CheckpointConfig::with_interval(2), // Every other layer
    ..Default::default()
};

// Disabled by default
let train_config = TrainingConfig {
    checkpoint_config: CheckpointConfig::default(),  // enabled: false
    ..Default::default()
};
```

**Memory Savings:**

Without checkpointing, each layer stores 11 tensors (~112 MB per layer for batch=16, seq=256, hidden=512):
- input, normed_attn, q_for_attn, k_for_attn, v_for_attn
- attn_out_pre_wo, post_attn, normed_ffn
- gate, up, swiglu_out

With checkpointing, each checkpointed layer stores only:
- input tensor (~8 MB)
- dropout seeds (16 bytes)

**Memory reduction: ~14x per checkpointed layer**

**Trade-offs:**

| Aspect | Without Checkpointing | With checkpoint_every=1 |
|--------|----------------------|-------------------------|
| Activation memory | ~112 MB/layer | ~8 MB/layer |
| Backward compute | 1x | ~1.3-1.5x (recompute) |
| Training speed | Baseline | ~20-30% slower |

**How it works:**

1. **Forward pass:** For checkpointed layers, compute output normally but store only:
   - Layer input tensor
   - Dropout seeds (for deterministic replay)

2. **Backward pass:** Before computing gradients for a checkpointed layer:
   - Recompute full forward pass from stored input
   - Use stored dropout seeds to replicate exact dropout masks
   - Compute gradients using recomputed activations

3. **Dropout determinism:** Uses `dropout_with_seed()` to apply identical dropout masks during recomputation, ensuring gradients are mathematically identical to non-checkpointed training.

**When to use:**

- Training larger models that don't fit in memory
- Using longer sequence lengths
- When compute is cheap relative to memory (common on Apple Silicon)

**Verification:**

Activation checkpointing produces **identical gradients** to standard training. This is verified by the test `test_activation_checkpointing` which compares loss, gradient norms, and final weights.

### GPU Trace Capture

GPU trace capture allows you to capture `.gputrace` files that can be opened in Xcode for detailed shader analysis. This reveals information that CPU-side timing cannot provide:

- GPU kernel execution time (not the same as CPU wait time)
- Shader register usage and occupancy
- Memory bandwidth utilization
- Pipeline stalls and synchronization overhead
- GPU timeline visualization

**Files:**
- `src/gpu_trace.rs` - GPU trace capture API

**Environment Variables:**

```bash
# Enable programmatic GPU capture (required by Metal)
# Without this, GPU trace capture will silently fail
export METAL_CAPTURE_ENABLED=1

# Enable GPU trace capture
IRONTENSOR_GPU_TRACE=1 cargo run --release

# Specify output directory (default: current directory)
IRONTENSOR_GPU_TRACE_DIR=./traces cargo run --release

# Capture only a specific training step
IRONTENSOR_GPU_TRACE_STEP=50 cargo run --release

# Full example
METAL_CAPTURE_ENABLED=1 IRONTENSOR_GPU_TRACE=1 IRONTENSOR_GPU_TRACE_DIR=./traces IRONTENSOR_GPU_TRACE_STEP=10 cargo run --release
```

**Programmatic API:**

```rust
use irontensor::{GpuTrace, GpuTraceGuard};

// Check if GPU trace capture is supported
if GpuTrace::is_supported() {
    // Manual capture
    GpuTrace::start("/tmp/my_trace.gputrace")?;
    // ... GPU operations ...
    GpuTrace::stop()?;

    // Block capture (RAII style)
    let result = GpuTrace::capture("/tmp/forward.gputrace", || {
        trainer.compute_loss(&input_ids, &target_ids, batch_size, seq_len)
    })?;

    // Guard-based capture (auto-stops on drop)
    let _guard = GpuTraceGuard::start("/tmp/train_step.gputrace")?;
    trainer.train_step(&input_ids, &target_ids, batch_size, seq_len);
    // Capture stops automatically when _guard is dropped
}
```

**Training Integration:**

GPU trace capture is automatically integrated with the training loop. Set environment variables before running:

```bash
# Capture step 100 of training
IRONTENSOR_GPU_TRACE=1 IRONTENSOR_GPU_TRACE_STEP=100 cargo run --release
```

The trace will be saved to `{output_dir}/train_step_{step}.gputrace`.

**Opening Traces in Xcode:**

1. Double-click the `.gputrace` file to open in Xcode
2. Or: File → Open → select the .gputrace file
3. Navigate the GPU Timeline to see all command buffers
4. Click on individual compute passes to see shader statistics

**What to Look For:**

| Metric | What It Tells You |
|--------|-------------------|
| Shader Execution Time | Actual GPU time per kernel |
| Register Usage | Higher usage = fewer concurrent threads |
| Occupancy | % of GPU threads that can run simultaneously |
| Memory Bandwidth | How efficiently memory is accessed |
| ALU Utilization | How much compute capacity is used |
| Cache Hit Rate | Efficiency of memory access patterns |

**Limitations:**

- Requires `METAL_CAPTURE_ENABLED=1` environment variable for programmatic capture
- Capture files can be large (1-5GB+ for a single training step)
- Adds overhead during capture (first logged step after capture may show lower throughput)
- Output file must not exist before capture starts
- macOS only (requires Metal framework)
- Only one capture can be active at a time

---

## Example Training Loop

```rust
use irontensor::{
    nn::model::{GPTModel, ModelConfig},
    train::{Trainer, TrainingConfig},
    data::dataset::{TokenDataset, DatasetIterator},
    logging::{Logger, LogConfig},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration
    let model_config = ModelConfig::small();
    let train_config = TrainingConfig {
        learning_rate: 3e-4,
        warmup_steps: 100,
        total_steps: 10000,
        use_bf16: true,
        early_stopping_patience: Some(5),  // Stop if no improvement for 5 evals
        ..Default::default()
    };

    // Initialize
    let mut trainer = Trainer::new(&model_config, &train_config);
    let dataset = TokenDataset::open("data.bin", 256)?;
    let mut iter = DatasetIterator::new(&dataset, 16, true);  // batch=16, shuffle=true

    Logger::init(LogConfig::from_env());

    // Training loop
    for epoch in 0..10 {
        iter.shuffle(epoch as u64);

        for (input_ids, target_ids) in &mut iter {
            let (loss, grad_norm) = trainer.train_step(
                &input_ids,
                &target_ids,
                16,   // batch_size
                256,  // seq_len
            );

            // Checkpoint on improvement
            if trainer.step % 1000 == 0 {
                let val_loss = trainer.evaluate(&val_dataset)?;
                if val_loss < trainer.best_val_loss {
                    trainer.best_val_loss = val_loss;
                    trainer.save_checkpoint("best_model.bin")?;
                }
            }
        }
        trainer.epoch += 1;
    }

    Logger::shutdown();
    Ok(())
}
```
