#!/bin/bash
# Submit an arbitrary command to one dane pdebug node (112 cores, <= 1 h).
#   scripts/sbatch_pdebug.sh <job-name> <command ...>
# Runs from the repo root; log goes to results/slurm/<job-name>-<jobid>.out.
set -euo pipefail
NAME=$1; shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$REPO/results/slurm"
sbatch --parsable -J "$NAME" -p pdebug -N 1 -t 01:00:00 \
    -o "$REPO/results/slurm/$NAME-%j.out" \
    --wrap "cd $REPO && export OMP_NUM_THREADS=112 NUMBA_NUM_THREADS=112 && $*"
