#!/bin/bash
#SBATCH -J quests-baselines
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH -t 04:00:00
#SBATCH -A TG-CHM250015
#SBATCH -o /scratch/09982/blaubach/UMich-quests-pruning-models/baselines-%j.out
# Published subsampling baselines on carbon (Reviewer 4 #3): DIRECT, QUESTS-MSC, CUR
# at 5% and 10% environment budgets.  CPU only.  Submit from the repo root.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
PY=$PWD/env/venv-select/bin/python
DATA=data/DS_2km6t3p8hxx2_0.xyz
echo "host=$(hostname) start=$(date)"
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 $PY pruning/baselines.py $DATA --method direct > results/slurm/baseline-direct-$SLURM_JOB_ID.out 2>&1 &
NUMBA_NUM_THREADS=24 OMP_NUM_THREADS=4 $PY pruning/baselines.py $DATA --method msc > results/slurm/baseline-msc-$SLURM_JOB_ID.out 2>&1 &
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 $PY pruning/baselines.py $DATA --method cur > results/slurm/baseline-cur-$SLURM_JOB_ID.out 2>&1 &
wait
echo "end=$(date)"; tail -n 3 results/slurm/baseline-*-$SLURM_JOB_ID.out
