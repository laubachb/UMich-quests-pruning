#!/bin/bash
# Train ONE NequIP model with the paper's config on a masked training set,
# then compile it to TorchScript for evaluation / LAMMPS.
#
#   training/train_one.sh MASKED.xyz TRAIN_SEED OUT_DIR [DEVICE]
#
# MASKED.xyz  pruned file from pruning/make_masks.py (full frames, per-atom
#             'weights' array = 0/1 selection mask; the force loss is masked by
#             ltau_nequip_modules.WeightedLoss, exactly as in the paper)
# TRAIN_SEED  integer; sets both data.seed and model seed (the paper's runs used
#             a different random seed per replicate, see env/README.md)
# OUT_DIR     Hydra run directory (created); best.ckpt and compiled model land here
# DEVICE      gpu (default) | cpu
# SPECIES     optional comma-separated chemical symbols for multi-element data,
#             e.g. "Mo,Nb,Ta,V,W" (default: C, as in config.yaml)
#
# As in the paper, the same file is used for train/val/test and early stopping
# monitors the training loss (config.yaml: monitored_metric).
set -euo pipefail
XYZ=$(readlink -f "$1"); SEED=$2; OUT=$(readlink -f "$3"); DEVICE=${4:-gpu}; SPECIES=${5:-}
SPECIES_OVERRIDE=()
[ -n "$SPECIES" ] && SPECIES_OVERRIDE=("chemical_symbols=[$SPECIES]")
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${NEQUIP_PYTHON:-$REPO/env/venv-paper/bin/python}
BIN=$(dirname "$PY")

mkdir -p "$OUT"
cd "$OUT"
"$BIN/nequip-train" -cp "$REPO/training" -cn config \
    hydra.run.dir="$OUT" \
    data.train_file_path="$XYZ" data.val_file_path="$XYZ" data.test_file_path="$XYZ" \
    data.seed="$SEED" training_module.model.seed="$SEED" \
    trainer.accelerator="$DEVICE" "${SPECIES_OVERRIDE[@]}" \
    2>&1 | tee train.log

CKPT="$OUT/train_dir/best.ckpt"
COMPILED="$OUT/compiled_best.nequip.pth"
CDEV=$([ "$DEVICE" = gpu ] && echo cuda || echo cpu)
"$BIN/nequip-compile" "$CKPT" "$COMPILED" --mode torchscript --device "$CDEV" 2>&1 | tee compile.log
echo "done: $COMPILED"
