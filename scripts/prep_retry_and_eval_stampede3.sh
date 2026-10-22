#!/bin/bash
#SBATCH -J quests-prep-eval
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH -t 08:00:00
#SBATCH -A TG-CHM250015
#SBATCH -o /scratch/09982/blaubach/UMich-quests-pruning-models/prep-eval-%j.out
# 1) all-ones-weight training files for E3 (full data), 2) E6 baseline matrix,
# 3) test-set evaluation of every model trained so far (paper protocol).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH=$PWD/env/venv-paper/bin:$PATH
OMP_NUM_THREADS=8 python pruning/make_masks.py data/DS_2km6t3p8hxx2_0.xyz results/fps/stratified_s0 results/pruned/full --fractions 1.000 --drop-descriptors
OMP_NUM_THREADS=8 python pruning/make_masks.py data/external/sio2/train.xyz results/ext/sio2/fps/stratified_s0 results/ext/sio2/pruned/full --fractions 1.000 --drop-descriptors
OMP_NUM_THREADS=8 python pruning/make_masks.py data/external/monbtavw/train.xyz results/ext/monbtavw/fps/stratified_s0 results/ext/monbtavw/pruned/full --fractions 1.000 --drop-descriptors
python training/make_baseline_matrix.py
EVAL_DIR=results/eval/paper_protocol OMP_NUM_THREADS=48 bash scripts/evaluate_all_stampede3.sh
