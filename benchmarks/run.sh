#!/bin/bash
# Run IronTensor vs PyTorch benchmark end-to-end.
#
# Usage:
#     bash benchmarks/run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

mkdir -p benchmarks/results

echo "============================================"
echo "  IronTensor vs PyTorch Benchmark"
echo "============================================"
echo

# Step 1: IronTensor benchmark
echo "[1/3] Running IronTensor benchmark..."
echo
cargo run --release --bin benchmark
echo

# Step 2: PyTorch benchmark
echo "[2/3] Running PyTorch benchmark..."
echo
python3 benchmarks/pytorch_benchmark.py --model tiny \
    --steps 55 --warmup-steps 5 \
    --batch-size 16 --seq-len 256 \
    --output benchmarks/results/pytorch.json
echo

# Step 3: Compare results
echo "[3/3] Comparison"
echo
python3 benchmarks/compare.py \
    benchmarks/results/irontensor.json \
    benchmarks/results/pytorch.json
