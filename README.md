# IronTensor

A experimental tensor library for training small LLMs on Apple Silicon. Built in Rust with Metal compute shaders, targeting the unified memory architecture of M-series chips.

Not intended for production use, just for fun.

## What it does

- Train small-to-medium GPT-style language models on a Mac
- BF16 training with Metal Performance Shaders (MPS) for fast matrix multiplication via the AMX coprocessor
- FlashAttention, fused kernels, activation checkpointing, and other memory optimizations
- Lion optimizer, learning rate schedulers, gradient accumulation
- Checkpoint save/load with optional optimizer state for stable resumption

## Requirements

- macOS with recent Apple Silicon
- Rust toolchain

## Quick start

```bash
cargo build --release
```

## License

[MIT](LICENSE)
