#!/bin/bash
# Submit an arbitrary command to one dane CPU node.
#   scripts/sbatch_pdebug.sh <job-name> <command ...>
# Defaults: pdebug, 1 h (the pdebug limit). Override for longer work with
#   PARTITION=pbatch TIME=06:00:00 scripts/sbatch_pdebug.sh ...
# Runs from the repo root; log goes to results/slurm/<job-name>-<jobid>.out.
set -euo pipefail
NAME=$1; shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$REPO/results/slurm"
sbatch --parsable -J "$NAME" -p "${PARTITION:-pdebug}" -N 1 --exclusive -c 112 --mem=0 -t "${TIME:-01:00:00}" \
    -o "$REPO/results/slurm/$NAME-%j.out" \
    --wrap "cd $REPO && export OMP_NUM_THREADS=${OMP_THREADS:-4} MKL_NUM_THREADS=${OMP_THREADS:-4} NUMBA_NUM_THREADS=112 && $*"
