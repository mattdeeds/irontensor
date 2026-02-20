#!/usr/bin/env python3
"""PyTorch benchmark for comparison with IronTensor.

Implements a Llama-style model matching IronTensor's architecture exactly,
trained with Lion optimizer and cosine LR schedule on synthetic data.

Usage:
    python benchmarks/pytorch_benchmark.py --model tiny --steps 55 --batch-size 16 --seq-len 256
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def precompute_freqs_cis(dim: int, seq_len: int, base: float = 10000.0,
                         device: torch.device = torch.device("cpu")) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: [B, n_heads, S, head_dim]
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis[None, None, :x.shape[2], :]  # broadcast over batch, heads
    out = torch.view_as_real(xc * freqs).flatten(-2)
    return out.to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float = 10000.0, max_seq_len: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

        self.wq = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, max_seq_len * 2, rope_base),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.wq(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        freqs = self.freqs_cis[:S].to(x.device)
        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)

        # GQA: repeat KV heads if needed
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        attn = attn + mask[None, None, :, :]
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int,
                 intermediate_dim: int, rope_base: float, norm_eps: float,
                 max_seq_len: int):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim, norm_eps)
        self.attention = MultiHeadAttention(hidden_dim, num_heads, num_kv_heads,
                                            rope_base, max_seq_len)
        self.ffn_norm = RMSNorm(hidden_dim, norm_eps)
        self.ffn = SwiGLUFFN(hidden_dim, intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int,
                 num_heads: int, num_kv_heads: int, intermediate_dim: int,
                 max_seq_len: int, rope_base: float = 10000.0,
                 norm_eps: float = 1e-5, tie_weights: bool = True):
        super().__init__()
        self.config = dict(
            vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            intermediate_dim=intermediate_dim, max_seq_len=max_seq_len,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, num_kv_heads, intermediate_dim,
                             rope_base, norm_eps, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_dim, norm_eps)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        if tie_weights:
            self.output_proj.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.output_proj(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Lion optimizer (inline, matches IronTensor formula)
# ---------------------------------------------------------------------------

class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 3e-4, betas=(0.9, 0.99),
                 weight_decay: float = 0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                m = state["momentum"]
                update = (beta2 * m + (1.0 - beta2) * grad).sign()
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                p.sub_(lr * (update + wd * p))


# ---------------------------------------------------------------------------
# LR schedule: cosine annealing with linear warmup
# ---------------------------------------------------------------------------

def cosine_lr(step: int, max_lr: float, warmup_steps: int, total_steps: int) -> float:
    min_lr = max_lr * 0.1
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

CONFIGS = {
    "tiny": dict(
        vocab_size=32000, hidden_dim=256, num_layers=4, num_heads=4,
        num_kv_heads=4, intermediate_dim=512, max_seq_len=512,
    ),
    "small": dict(
        vocab_size=32000, hidden_dim=512, num_layers=8, num_heads=8,
        num_kv_heads=8, intermediate_dim=1024, max_seq_len=2048,
    ),
}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def benchmark_training(model: LlamaModel, device: torch.device,
                       batch_size: int, seq_len: int, total_steps: int,
                       warmup_steps: int, lr: float, max_grad_norm: float
                       ) -> dict:
    vocab_size = model.config["vocab_size"]
    optimizer = Lion(model.parameters(), lr=lr, betas=(0.9, 0.99),
                     weight_decay=0.1)

    step_times = []
    losses = []

    for step in range(total_steps):
        # Synthetic random tokens
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Set LR
        current_lr = cosine_lr(step, lr, warmup_steps, total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        torch.mps.synchronize()
        t0 = time.perf_counter()

        # Forward
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()

        torch.mps.synchronize()
        t1 = time.perf_counter()

        step_ms = (t1 - t0) * 1000.0
        loss_val = loss.item()

        # Discard warmup steps
        if step >= warmup_steps:
            step_times.append(step_ms)
            losses.append(loss_val)

        tokens_per_sec = batch_size * seq_len / (step_ms / 1000.0)
        if step % 10 == 0 or step == total_steps - 1:
            print(f"  Step {step:>4}/{total_steps} | loss={loss_val:.4f} | "
                  f"lr={current_lr:.2e} | {tokens_per_sec:.0f} tok/s | {step_ms:.1f}ms")

    # Memory measurement
    peak_mem_bytes = torch.mps.driver_allocated_memory()
    current_mem_bytes = torch.mps.current_allocated_memory()

    avg_step_ms = sum(step_times) / len(step_times) if step_times else 0
    avg_tokens_per_sec = (batch_size * seq_len) / (avg_step_ms / 1000.0) if avg_step_ms > 0 else 0

    return {
        "avg_step_time_ms": avg_step_ms,
        "median_step_time_ms": sorted(step_times)[len(step_times) // 2] if step_times else 0,
        "min_step_time_ms": min(step_times) if step_times else 0,
        "max_step_time_ms": max(step_times) if step_times else 0,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "final_loss": losses[-1] if losses else 0,
        "initial_loss": losses[0] if losses else 0,
        "peak_memory_bytes": peak_mem_bytes,
        "current_memory_bytes": current_mem_bytes,
        "timed_steps": len(step_times),
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "step_times_ms": step_times,
        "losses": losses,
    }


def benchmark_inference(model: LlamaModel, device: torch.device,
                        max_tokens: int, prompt_lengths: list[int]) -> list[dict]:
    vocab_size = model.config["vocab_size"]
    model.eval()
    results = []

    for prompt_len in prompt_lengths:
        prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)
        tokens = prompt.clone()

        torch.mps.synchronize()
        t_start = time.perf_counter()
        ttft = None
        token_times = []

        with torch.no_grad():
            for i in range(max_tokens):
                torch.mps.synchronize()
                t0 = time.perf_counter()

                logits = model(tokens)
                next_logit = logits[:, -1, :]
                next_token = next_logit.argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)

                torch.mps.synchronize()
                t1 = time.perf_counter()

                if i == 0:
                    ttft = (t1 - t_start) * 1000.0
                token_times.append((t1 - t0) * 1000.0)

        torch.mps.synchronize()
        total_ms = (time.perf_counter() - t_start) * 1000.0
        tokens_per_sec = max_tokens / (total_ms / 1000.0) if total_ms > 0 else 0
        inter_token_ms = (sum(token_times[1:]) / len(token_times[1:])) if len(token_times) > 1 else 0

        results.append({
            "prompt_length": prompt_len,
            "generated_tokens": max_tokens,
            "ttft_ms": ttft or 0,
            "total_time_ms": total_ms,
            "tokens_per_sec": tokens_per_sec,
            "inter_token_latency_ms": inter_token_ms,
        })

        print(f"  prompt_len={prompt_len}: TTFT={ttft:.1f}ms, "
              f"{tokens_per_sec:.1f} tok/s, total={total_ms:.0f}ms")

    model.train()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PyTorch LLM Benchmark")
    parser.add_argument("--model", choices=["tiny", "small"], default="tiny")
    parser.add_argument("--steps", type=int, default=55,
                        help="Total steps (warmup + timed)")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--inference-tokens", type=int, default=100)
    parser.add_argument("--output", type=str,
                        default="benchmarks/results/pytorch.json")
    parser.add_argument("--no-inference", action="store_true")
    args = parser.parse_args()

    # Device setup
    if not torch.backends.mps.is_available():
        print("WARNING: MPS not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
    print(f"Device: {device}")

    # Build model
    cfg = CONFIGS[args.model]
    model = LlamaModel(**cfg).to(device)
    print(f"Model: {args.model} ({model.num_params() / 1e6:.2f}M params)")
    print(f"Config: hidden={cfg['hidden_dim']} layers={cfg['num_layers']} "
          f"heads={cfg['num_heads']} intermediate={cfg['intermediate_dim']}")
    print()

    # Training benchmark
    print("=" * 60)
    print(f"Training benchmark: {args.steps} steps "
          f"(warmup={args.warmup_steps}, timed={args.steps - args.warmup_steps})")
    print(f"  batch_size={args.batch_size}, seq_len={args.seq_len}")
    print("=" * 60)

    train_results = benchmark_training(
        model, device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
    )

    print(f"\nTraining summary:")
    print(f"  Avg step time:  {train_results['avg_step_time_ms']:.1f}ms")
    print(f"  Avg tokens/sec: {train_results['avg_tokens_per_sec']:.0f}")
    print(f"  Peak memory:    {train_results['peak_memory_bytes'] / 1e6:.0f}MB")

    # Inference benchmark
    inference_results = []
    if not args.no_inference:
        print()
        print("=" * 60)
        print(f"Inference benchmark: {args.inference_tokens} tokens, "
              f"no KV cache")
        print("=" * 60)

        inference_results = benchmark_inference(
            model, device,
            max_tokens=args.inference_tokens,
            prompt_lengths=[5, 20],
        )

    # Save results
    output = {
        "framework": "pytorch",
        "model": args.model,
        "model_config": cfg,
        "model_params": model.num_params(),
        "device": str(device),
        "precision": "fp32",
        "training": train_results,
        "inference": inference_results,
    }

    # Remove per-step arrays to keep JSON small (keep summary stats)
    output["training"].pop("step_times_ms", None)
    output["training"].pop("losses", None)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
