#!/usr/bin/env python3
"""Compare IronTensor and PyTorch benchmark results.

Usage:
    python benchmarks/compare.py benchmarks/results/irontensor.json benchmarks/results/pytorch.json
"""

import json
import sys


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_ratio(iron_val: float, pytorch_val: float, lower_is_better: bool = True) -> str:
    """Format the ratio with a direction indicator."""
    if iron_val == 0 or pytorch_val == 0:
        return "N/A"
    if lower_is_better:
        if iron_val < pytorch_val:
            ratio = pytorch_val / iron_val
            return f"{ratio:.2f}x faster"
        else:
            ratio = iron_val / pytorch_val
            return f"{ratio:.2f}x slower"
    else:
        # Higher is better (e.g., tokens/sec)
        if iron_val > pytorch_val:
            ratio = iron_val / pytorch_val
            return f"{ratio:.2f}x faster"
        else:
            ratio = pytorch_val / iron_val
            return f"{ratio:.2f}x slower"


def fmt_mem_ratio(iron_val: float, pytorch_val: float) -> str:
    if iron_val == 0 or pytorch_val == 0:
        return "N/A"
    if iron_val < pytorch_val:
        ratio = pytorch_val / iron_val
        return f"{ratio:.2f}x less"
    else:
        ratio = iron_val / pytorch_val
        return f"{ratio:.2f}x more"


def print_comparison(iron: dict, pytorch: dict):
    iron_train = iron["training"]
    pt_train = pytorch["training"]
    model = iron.get("model", "unknown")
    precision = iron.get("precision", "fp32").upper()
    batch = iron_train.get("batch_size", "?")
    seq = iron_train.get("seq_len", "?")

    print()
    header = f"Training ({model}, {precision}, batch={batch}, seq={seq})"
    print(f"====== {header} ======")
    print()
    print(f"{'':30s} {'IronTensor':>14s} {'PyTorch':>14s} {'Ratio':>18s}")
    print("-" * 78)

    # Avg step time
    it_step = iron_train["avg_step_time_ms"]
    pt_step = pt_train["avg_step_time_ms"]
    print(f"{'Avg step time (ms)':30s} {it_step:>14.1f} {pt_step:>14.1f} {fmt_ratio(it_step, pt_step):>18s}")

    # Avg tokens/sec
    it_tok = iron_train["avg_tokens_per_sec"]
    pt_tok = pt_train["avg_tokens_per_sec"]
    print(f"{'Avg tokens/sec':30s} {it_tok:>14,.0f} {pt_tok:>14,.0f} {fmt_ratio(it_tok, pt_tok, lower_is_better=False):>18s}")

    # Peak memory
    it_mem = iron_train.get("peak_memory_bytes", 0) / 1e6
    pt_mem = pt_train.get("peak_memory_bytes", 0) / 1e6
    print(f"{'Peak memory (MB)':30s} {it_mem:>14,.0f} {pt_mem:>14,.0f} {fmt_mem_ratio(it_mem, pt_mem):>18s}")

    # Median step time
    it_med = iron_train.get("median_step_time_ms", 0)
    pt_med = pt_train.get("median_step_time_ms", 0)
    if it_med > 0 and pt_med > 0:
        print(f"{'Median step time (ms)':30s} {it_med:>14.1f} {pt_med:>14.1f} {fmt_ratio(it_med, pt_med):>18s}")

    # Final loss
    it_loss = iron_train.get("final_loss", 0)
    pt_loss = pt_train.get("final_loss", 0)
    print(f"{'Final loss':30s} {it_loss:>14.4f} {pt_loss:>14.4f} {'':>18s}")

    # Inference results
    iron_inf = iron.get("inference", [])
    pt_inf = pytorch.get("inference", [])

    if iron_inf and pt_inf:
        print()
        gen_tokens = iron_inf[0].get("generated_tokens", "?")
        print(f"====== Inference ({gen_tokens} tokens, no KV cache) ======")
        print()
        print(f"{'':30s} {'IronTensor':>14s} {'PyTorch':>14s} {'Ratio':>18s}")
        print("-" * 78)

        # Match by prompt length
        pt_by_plen = {r["prompt_length"]: r for r in pt_inf}

        for ir in iron_inf:
            plen = ir["prompt_length"]
            pr = pt_by_plen.get(plen)
            if pr is None:
                continue

            label = f"prompt_len={plen}"
            print(f"\n  {label}")

            # TTFT
            it_ttft = ir["ttft_ms"]
            pt_ttft = pr["ttft_ms"]
            print(f"{'    TTFT (ms)':30s} {it_ttft:>14.1f} {pt_ttft:>14.1f} {fmt_ratio(it_ttft, pt_ttft):>18s}")

            # Tokens/sec
            it_tps = ir["tokens_per_sec"]
            pt_tps = pr["tokens_per_sec"]
            print(f"{'    Tokens/sec':30s} {it_tps:>14.1f} {pt_tps:>14.1f} {fmt_ratio(it_tps, pt_tps, lower_is_better=False):>18s}")

            # Inter-token latency
            it_itl = ir["inter_token_latency_ms"]
            pt_itl = pr["inter_token_latency_ms"]
            print(f"{'    Inter-token latency (ms)':30s} {it_itl:>14.1f} {pt_itl:>14.1f} {fmt_ratio(it_itl, pt_itl):>18s}")

    print()


def main():
    if len(sys.argv) < 3:
        print("Usage: python benchmarks/compare.py <irontensor.json> <pytorch.json>")
        sys.exit(1)

    iron_path = sys.argv[1]
    pytorch_path = sys.argv[2]

    iron = load_results(iron_path)
    pytorch = load_results(pytorch_path)

    print_comparison(iron, pytorch)


if __name__ == "__main__":
    main()
