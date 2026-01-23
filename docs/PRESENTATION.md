# IronTensor: A Minimal LLM Training Framework for Apple Silicon

## Overview

IronTensor is a tiny, performant tensor library designed specifically for training LLMs on Apple Silicon. It targets M-series chips (M1/M2/M3/M4) and their unified memory architecture, enabling training of small-to-medium language models on consumer hardware.

**Key Philosophy:**
- Minimal dependencies - just Rust + Metal
- No CUDA, no PyTorch, no Python
- Direct GPU programming via Metal shaders
- Optimized for Apple's unified memory (CPU and GPU share the same RAM)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Application Layer                            │
│  main.rs: Training script, data loading, tokenization               │
├─────────────────────────────────────────────────────────────────────┤
│                         Training Layer                               │
│  Trainer, LR Schedulers, Checkpointing, Profiler                    │
├─────────────────────────────────────────────────────────────────────┤
│                         Neural Network Layer                         │
│  GPTModel, TransformerBlock, Attention, FFN, Linear                 │
├─────────────────────────────────────────────────────────────────────┤
│                         Operations Layer                             │
│  matmul, rmsnorm, softmax, attention, embedding, rope               │
│  backward/: gradients for each operation                            │
├─────────────────────────────────────────────────────────────────────┤
│                         Core Layer                                   │
│  Tensor, CommandBatch, MetalContext, Precision (FP32/BF16)          │
├─────────────────────────────────────────────────────────────────────┤
│                         Metal Shaders                                │
│  .metal files: GPU compute kernels                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. MetalContext (src/device.rs)

Global singleton managing the Metal device and command queue:

```rust
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

static GLOBAL_CONTEXT: OnceLock<MetalContext> = OnceLock::new();

impl MetalContext {
    pub fn global() -> &'static MetalContext {
        GLOBAL_CONTEXT.get_or_init(MetalContext::new)
    }
}
```

**Why a singleton?** Metal pipelines are expensive to create. By having a global context, we compile shaders once and reuse them.

### 2. Tensor (src/tensor.rs)

Core data structure wrapping a Metal buffer:

```rust
pub struct Tensor {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    shape: Vec<usize>,
    precision: Precision,  // FP32 or BF16
}
```

**Key features:**
- **Unified Memory**: Uses `MTLResourceOptions::StorageModeShared` - CPU and GPU access the same memory without explicit copies
- **BF16 Support**: Half the memory of FP32, good enough precision for training
- **Zero-copy Views**: `tensor.view(&[new_shape])` creates a new tensor sharing the same buffer

```rust
// Creating tensors
let a = Tensor::zeros(&[1024, 512], Precision::FP32);
let b = Tensor::from_f32_slice(&data, &[batch, seq_len, hidden]);

// Precision conversion
let bf16_tensor = fp32_tensor.to_bf16();

// Zero-copy reshape
let reshaped = tensor.view(&[batch * seq_len, hidden_dim]);
```

### 3. Command Batching (src/command_batch.rs)

Reduces GPU synchronization overhead by batching multiple operations:

```
Without batching (slow):
┌────────┐   ┌────────┐   ┌────────┐
│ Op 1   │→  │ Op 2   │→  │ Op 3   │
│ commit │   │ commit │   │ commit │
│ wait   │   │ wait   │   │ wait   │
└────────┘   └────────┘   └────────┘

With batching (fast):
┌────────────────────────────────────┐
│ Op 1 → Op 2 → Op 3                 │
│                         commit     │
│                         wait       │
└────────────────────────────────────┘
```

**Usage:**
```rust
// Start accumulating operations
CommandBatch::begin();

// GPU operations are queued, not executed yet
let c = matmul(&a, &b);
let d = add(&c, &bias);
let e = silu(&d);

// Commit all operations and wait
CommandBatch::sync();

// Now safe to read results
let result = e.as_f32_slice();

// End batching mode
CommandBatch::end();
```

**Async Mode** (CPU/GPU overlap):
```rust
CommandBatch::begin();
// ... GPU operations ...
CommandBatch::commit_async();  // Returns immediately!
// ... CPU work (prepare next batch) ...
CommandBatch::wait_for_completion();  // Wait before reading
```

---

## GPU Programming Pattern

Each operation follows this pattern:

### Rust Side (e.g., src/ops/elementwise.rs)

```rust
pub fn silu(input: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::Elementwise, input.numel());

    let count = input.numel();
    let output = Tensor::zeros(input.shape(), Precision::FP32);

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();  // Cached, compiled once

    let grid_size = MTLSize { width: count, height: 1, depth: 1 };
    let threadgroup_size = MTLSize {
        width: pipelines.silu.threadExecutionWidth().min(count),
        height: 1, depth: 1
    };

    CommandBatch::dispatch(
        &pipelines.silu,
        |encoder| unsafe {
            encoder.setBuffer_offset_atIndex(Some(input.buffer()), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(output.buffer()), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&count_buffer), 0, 2);
        },
        grid_size,
        threadgroup_size,
    );

    output
}
```

### Metal Shader Side (e.g., src/shaders/elementwise.metal)

```metal
kernel void silu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= count) return;
    float x = input[idx];
    output[idx] = x / (1.0f + exp(-x));  // x * sigmoid(x)
}
```

**Key points:**
- Each thread processes one element
- `[[thread_position_in_grid]]` gives the global thread index
- Early exit for threads beyond data bounds
- Buffers are bound by index (0, 1, 2, ...)

---

## Model Architecture

### GPTModel (src/nn/model.rs)

Llama-style transformer architecture:

```
tokens → Embedding → [N × TransformerBlock] → RMSNorm → Output → logits
```

**TransformerBlock:**
```
                    ┌──────────────┐
         ┌──────────│   RMSNorm    │
         │          └──────────────┘
         │                 │
         │          ┌──────────────┐
         │          │   Attention   │
         │          │  (RoPE, GQA) │
         │          └──────────────┘
         │                 │
         └────────(+)──────┘
                   │
         ┌─────────│
         │         │
         │  ┌──────────────┐
         │  │   RMSNorm    │
         │  └──────────────┘
         │         │
         │  ┌──────────────┐
         │  │  FFN (SwiGLU) │
         │  └──────────────┘
         │         │
         └───(+)───┘
               │
            output
```

**Model Configurations:**
```rust
// Pre-defined sizes
ModelConfig::tiny()         // ~5M params, good for testing
ModelConfig::small()        // ~30M params
ModelConfig::medium()       // ~125M params
ModelConfig::shakespeare()  // Optimized for TinyShakespeare demo

// Custom config
ModelConfig {
    vocab_size: 32000,
    hidden_dim: 512,
    num_layers: 8,
    num_heads: 8,
    num_kv_heads: 8,        // For GQA (grouped-query attention)
    intermediate_dim: 1024,  // FFN hidden size
    max_seq_len: 2048,
    rope_base: 10000.0,
    norm_eps: 1e-5,
    tie_weights: true,       // Share embedding and output weights
    precision: Precision::FP32,
}
```

---

## Key Optimizations

### 1. FlashAttention (src/ops/flash_attention.rs)

Memory-efficient attention that avoids materializing the N×N attention matrix:

```
Standard Attention Memory: O(N²) - explodes with sequence length
FlashAttention Memory: O(N) - scales linearly
```

**How it works:**
- Process attention in tiles that fit in fast SRAM (threadgroup memory)
- Use "online softmax" to compute softmax incrementally
- Never materialize the full attention matrix

### 2. Fused Linear Cross-Entropy (src/ops/fused_linear_cross_entropy.rs)

Avoids materializing the full [batch×seq, vocab_size] logits tensor:

```
Standard:
hidden → matmul → logits → softmax → loss
           ↑
    [batch*seq, vocab]  ← HUGE for 32K vocab!

Fused:
hidden → fused_kernel → loss + grad_hidden
           (computes logits tile-by-tile, never fully materialized)
```

### 3. Lion Optimizer (src/optim/lion/)

Sign-based optimizer that's more memory-efficient than Adam:

```
Adam memory:  2 × model_params (momentum + variance)
Lion memory:  1 × model_params (momentum only)
```

**Update rule:**
```
update = sign(β₂ × m + (1 - β₂) × g)
m_new = β₁ × m + (1 - β₁) × g
w_new = w - lr × update - lr × wd × w
```

### 4. BF16 Mixed Precision

Store weights in BF16 (2 bytes vs 4 bytes), compute in FP32:

```rust
// Convert model to BF16
model.to_bf16();

// Training uses BF16 weights, FP32 gradients
trainer.train_step(&inputs, &targets, batch_size, seq_len);
```

**Memory savings:** 50% reduction in model weight storage.

### 5. MPS GEMM (src/ops/mps_gemm.rs)

Uses Metal Performance Shaders for matrix multiplication, which leverages Apple's AMX (Apple Matrix eXtensions) coprocessor:

```rust
// 2-4x faster than custom kernels for large matrices
let c = matmul(&a, &b);  // Automatically uses MPS when beneficial
```

---

## Training Pipeline

### 1. Data Loading (src/data/dataset.rs)

Memory-mapped binary dataset for zero-copy token loading:

```rust
// Create dataset from tokens
TokenDataset::create("train.bin", &token_ids)?;

// Open for training (uses mmap)
let dataset = TokenDataset::open("train.bin", seq_len)?;

// Get batch
let (input_ids, target_ids) = dataset.get_batch(batch_idx);
```

### 2. Training Loop (src/train/trainer.rs)

```rust
let mut trainer = Trainer::new(model_config, train_config);

for step in 0..total_steps {
    let (inputs, targets) = get_batch(&dataset, step);

    // Forward + backward + optimizer in one call
    let (loss, grad_norm) = trainer.train_step(
        &inputs, &targets, batch_size, seq_len
    );

    if step % log_interval == 0 {
        println!("Step {} | Loss: {:.4}", step, loss);
    }
}
```

### 3. Training Configuration

```rust
TrainingConfig {
    learning_rate: 3e-4,
    weight_decay: 0.1,
    beta1: 0.9,
    beta2: 0.99,
    max_grad_norm: 1.0,
    warmup_steps: 50,
    total_steps: 1000,
    log_interval: 10,
    save_interval: 100,
    eval_interval: 50,
    checkpoint_dir: "checkpoints".to_string(),
    use_bf16: false,      // Enable BF16 mixed precision
    async_gpu: true,      // Enable async GPU mode
}
```

### 4. LR Schedulers (src/train/scheduler.rs)

```rust
// Cosine annealing with warmup
let scheduler = CosineAnnealingLR::with_warmup(
    peak_lr,
    warmup_steps,
    total_steps,
);

// Get LR for current step
let lr = scheduler.get_lr(step);
```

Available schedulers: Constant, Linear, Cosine, InverseSqrt, Warmup variants.

---

## Profiling

Opt-in hierarchical profiler for identifying bottlenecks:

```bash
# Enable profiling via environment variable
IRONTENSOR_PROFILE=1 cargo run --release
```

**Output:**
```
================================================================================
                        IronTensor Profiling Report
================================================================================
Total Time: 45.32s | Steps: 100 | Avg Step: 453.2ms

Phase Breakdown:
  Forward Pass:    182.4ms (40.2%)
  Backward Pass:   241.8ms (53.4%)
  Optimizer:        29.0ms  (6.4%)

Top Operations by Time:
  Rank | Operation           | Total    | Count | Avg     | % Total
  -----+---------------------+----------+-------+---------+---------
    1  | Matmul              | 28.4s    | 4800  |  5.9ms  |  62.7%
    2  | RmsNormBackward     |  4.2s    |  800  |  5.3ms  |   9.3%
    3  | Softmax             |  3.1s    |  400  |  7.8ms  |   6.8%
================================================================================
```

**Adding profiling to operations:**
```rust
use crate::profile::{timed, OpCategory};

pub fn my_operation(input: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::Matmul, input.numel());
    // ... operation code ...
}
```

---

## How to Run

### Basic Training

```bash
# Build and run
cargo run --release

# This will:
# 1. Download TinyShakespeare
# 2. Train a BPE tokenizer
# 3. Create binary datasets
# 4. Train a small GPT model
# 5. Generate sample text
```

### With Options

```bash
# Enable BF16 mixed precision
IRONTENSOR_BF16=1 cargo run --release

# Enable profiling
IRONTENSOR_PROFILE=1 cargo run --release

# Disable async GPU (for debugging)
IRONTENSOR_SYNC_GPU=1 cargo run --release
```

### Running Tests

```bash
cargo test
# Runs 143+ tests covering all operations and layers
```

---

## Project Structure

```
src/
├── lib.rs              # Public API exports
├── main.rs             # Training example
├── command_batch.rs    # GPU command batching
├── device.rs           # MetalContext singleton
├── precision.rs        # FP32/BF16 types
├── tensor.rs           # Core Tensor type
├── profile/            # Hierarchical profiler
├── ops/                # GPU operations
│   ├── gemm.rs         # Matrix multiplication
│   ├── mps_gemm.rs     # MPS-accelerated matmul
│   ├── norm.rs         # RMSNorm
│   ├── softmax.rs      # Softmax
│   ├── attention.rs    # Standard attention
│   ├── flash_attention.rs
│   ├── embedding.rs    # Token embedding
│   ├── rope.rs         # Rotary position embeddings
│   ├── elementwise.rs  # add, mul, silu, etc.
│   ├── fused_linear_cross_entropy.rs
│   └── backward/       # Gradient operations
├── optim/
│   └── lion/           # Lion optimizer
├── nn/                 # Neural network modules
│   ├── model.rs        # GPTModel
│   ├── transformer.rs  # TransformerBlock
│   ├── attention.rs    # MultiHeadAttention
│   ├── ffn.rs          # SwiGLU FFN
│   └── linear.rs       # Linear layer
├── data/
│   └── dataset.rs      # Memory-mapped datasets
├── train/
│   ├── trainer.rs      # Main Trainer
│   ├── scheduler.rs    # LR schedulers
│   └── checkpoint.rs   # Model save/load
└── shaders/            # Metal compute kernels
    ├── gemm.metal
    ├── lion.metal
    ├── flash_attention.metal
    └── ... (one .metal per operation)
```

---

## Why IronTensor?

### vs PyTorch on Apple Silicon

| Aspect | IronTensor | PyTorch (MPS) |
|--------|------------|---------------|
| Startup | ~50ms | ~2-3s |
| Memory | Direct Metal buffers | Python overhead |
| Customization | Full shader control | Limited |
| Dependencies | Just Rust | Python ecosystem |

### vs MLX

| Aspect | IronTensor | MLX |
|--------|------------|-----|
| Language | Rust | Python/C++ |
| Training | Full autodiff | Primarily inference |
| Control | Raw Metal | Higher-level API |
| Target | LLM training | General ML |

### Ideal Use Cases

- Learning GPU programming fundamentals
- Training small models on M-series Macs
- Research requiring custom GPU kernels
- Embedded/low-dependency deployments

---

## Future Directions

- [ ] Single allocation memory pool
- [ ] 8-bit Adam via block-wise quantization
- [ ] Sophia optimizer
- [ ] Multi-GPU support (Mac Pro)
- [ ] Inference-optimized KV cache
- [ ] GGUF model loading

---

## Quick Reference: Key APIs

```rust
// === Tensor ===
let t = Tensor::zeros(&[m, n], Precision::FP32);
let t = Tensor::from_f32_slice(&data, &[m, n]);
let view = t.view(&[m * n]);
let bf16 = t.to_bf16();

// === Operations ===
let c = matmul(&a, &b);
let y = rmsnorm(&x, &gamma, eps);
let attn = flash_attention(&q, &k, &v, causal);
let y = silu(&x);

// === Model ===
let model = GPTModel::new(ModelConfig::small());
let logits = model.forward(&tokens, batch, seq_len, 0);

// === Training ===
let mut trainer = Trainer::new(model_config, train_config);
let (loss, grad_norm) = trainer.train_step(&inputs, &targets, batch, seq);
trainer.save_checkpoint("model.bin")?;

// === Batching ===
CommandBatch::begin();
// ... many GPU ops ...
CommandBatch::sync();
```

---

## Questions?

Repository: This codebase!

Key files to explore:
- `src/main.rs` - Complete training example
- `src/train/trainer.rs` - Training loop internals
- `src/shaders/` - Metal kernel implementations
