#!/bin/bash
# Time one short CPU training epoch (40 batches of 2 frames) to estimate
# full-sweep cost on a CPU node.  Run from the repo root.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=$REPO/env/venv
OUT=${1:-$REPO/results/timing_cpu}
THREADS=${OMP_NUM_THREADS:-$(nproc)}

mkdir -p "$OUT" && cd "$OUT"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
echo "threads=$THREADS host=$(hostname)"
/usr/bin/time -v "$VENV/bin/nequip-train" -cp "$REPO/training/configs" -cn timing_cpu \
    2>&1 | tee train.log
