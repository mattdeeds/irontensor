# IronTensor

A experimental tensor library for training small LLMs on Apple Silicon. Built in Rust with Metal compute shaders, targeting the unified memory architecture of M-series chips.

Not intended for production use, just for fun.

## What it does

- Train small-to-medium GPT-style language models on a Mac
- BF16 training with Metal Performance Shaders (MPS) for fast matrix multiplication via the AMX coprocessor
- FlashAttention, fused kernels, activation checkpointing, and other memory optimizations
- Lion optimizer, learning rate schedulers, gradient accumulation
- Checkpoint save/load with optional optimizer state for stable resumption

## Performance

Benchmark on Apple M3 Pro comparing IronTensor to PyTorch (MPS backend), using a 10.8M parameter Llama-style model (TINY config: 4 layers, 256 hidden, 4 heads). FP32, batch size 16, sequence length 256, Lion optimizer. 5 warmup steps discarded, 50 timed steps.

### Training

|                        | IronTensor | PyTorch | Ratio         |
|------------------------|------------|---------|---------------|
| Avg step time (ms)     | 481        | 196     | 2.5x slower   |
| Avg tokens/sec         | 8,513      | 20,913  | 2.5x slower   |
| Peak GPU memory (MB)   | 87         | 3,840   | **44x less**  |

### Inference (100 tokens, greedy, no KV cache)

|                            | IronTensor | PyTorch | Ratio       |
|----------------------------|------------|---------|-------------|
| TTFT, prompt=5 (ms)       | 52         | 40      | 1.3x slower |
| Tokens/sec, prompt=5      | 18.5       | 49.7    | 2.7x slower |
| Tokens/sec, prompt=20     | 18.9       | 103.6   | 5.5x slower |

IronTensor is currently ~2.5x slower than PyTorch for training throughput but uses **44x less GPU memory**. The extreme memory efficiency comes from manual Metal buffer management versus PyTorch's autograd graph retention. Inference is slower due to the lack of KV caching and kernel-launch overhead on small batch sizes.

To reproduce: `bash benchmarks/run.sh`

## Requirements

- macOS with recent Apple Silicon
- Rust toolchain

## Quick start

```bash
cargo build --release
```

## License

[MIT](LICENSE)
