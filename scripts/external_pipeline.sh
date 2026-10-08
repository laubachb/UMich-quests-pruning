#!/bin/bash
# Selection-side pipeline for an external dataset (everything but training).
#   scripts/external_pipeline.sh <name> <stage> <species...>
# stages:
#   select  descriptors (train+test) -> FPS s0/s1 (stratified, global) -> random
#           baselines -> retention accounting -> covering radii      (~minutes)
#   sweep   f_uniq(h) per group with subsampling                      (hours for SiO2)
#   global  f_uniq(h) on the pooled dataset                            (hours)
# Inputs: data/external/<name>/{train,test}.xyz from data/prepare_external.py
# Outputs under results/ext/<name>/ mirroring the carbon layout.
set -euo pipefail
NAME=$1; STAGE=$2; shift 2; SPECIES="$*"
D=data/external/$NAME; R=results/ext/$NAME
TR=$D/train_desc.xyz; TE=$D/test_desc.xyz
mkdir -p "$R"

case $STAGE in
select)
  [ -f "$TR" ] || python3 pruning/compute_descriptors.py $D/train.xyz --out $TR --species $SPECIES
  [ -f "$TE" ] || python3 pruning/compute_descriptors.py $D/test.xyz  --out $TE --species $SPECIES
  for s in 0 1; do
    python3 pruning/fps.py $TR --out-dir $R/fps/stratified_s$s --mode stratified --fps-seed $s
    python3 pruning/fps.py $TR --out-dir $R/fps/global_s$s     --mode global     --fps-seed $s
  done
  python3 pruning/random_order.py $TR --out-dir $R/fps/random_stratified_s0 --mode stratified --seed 0
  python3 pruning/random_order.py $TR --out-dir $R/fps/random_global_s0     --mode global     --seed 0
  mkdir -p $R/stats
  for sel in stratified_s0 stratified_s1 global_s0 global_s1 random_stratified_s0 random_global_s0; do
    python3 pruning/retention_accounting.py $TR $R/fps/$sel $R/stats/retention_accounting_$sel.csv > /dev/null
  done
  python3 pruning/fps_radius.py $R/fps/stratified_s0 $R/stats/fps_radius_stratified_s0.csv
  ;;
sweep)
  python3 pruning/funiq_sweep.py $TR --out-dir $R/funiq/groups_s0 --scopes groups --batch-size 2000
  python3 pruning/elbow.py $R/funiq/groups_s0/funiq_curves.csv --out $R/funiq/groups_s0/elbows_all.csv
  ;;
global)
  python3 pruning/funiq_sweep.py $TR --out-dir $R/funiq/global_s0 --scopes global --subsample 0.25 0.5 --n-rep 2 --batch-size 2000
  ;;
*) echo "unknown stage $STAGE"; exit 1;;
esac
echo "DONE $NAME $STAGE"
