#!/bin/bash
#SBATCH -J quests-smoke
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH -t 01:30:00
#SBATCH -o results/slurm/smoke-%j.out
#SBATCH -A TG-CHM250015
# Stampede3 CPU smoke test: build all pruned training sets, then run a few
# training batches + compile + evaluate for one carbon and one SiO2 model with
# the paper-faithful stack (env/venv-paper).  Submit from the repo root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH=$PWD/env/venv-paper/bin:$PATH OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "host=$(hostname) python=$(which python)"; ls env/venv-paper/lib/python3.12/site-packages/nequip/ | tr "\n" " "; echo
python -c "import nequip,torch;print('nequip',nequip.__version__,'torch',torch.__version__)"

echo "== training sets"
if [ -z "${SKIP_SETS:-}" ]; then
DATA=data/DS_2km6t3p8hxx2_0.xyz bash training/make_training_sets.sh
bash training/make_training_sets_sio2.sh
bash training/make_training_sets_monbtavw.sh
fi
ls results/pruned/*/ results/ext/*/pruned/*/ | head -80

SMOKE="trainer.max_epochs=1 +trainer.limit_train_batches=3 +trainer.limit_val_batches=1 +trainer.limit_test_batches=1 +trainer.num_sanity_val_steps=0 data.train_dataloader.num_workers=0"
smoke() {  # xyz out species
  rm -rf "$2"; mkdir -p "$2"
  local XYZ=$(readlink -f $1) OUT=$(readlink -f $2) SP=$3; local EXTRA=()
  [ -n "$SP" ] && EXTRA=("chemical_symbols=[$SP]")
  cd "$OUT"
  nequip-train -cp "$SLURM_SUBMIT_DIR/training" -cn config hydra.run.dir="$OUT" \
    data.train_file_path="$XYZ" data.val_file_path="$XYZ" data.test_file_path="$XYZ" \
    data.seed=0 training_module.model.seed=0 trainer.accelerator=cpu $SMOKE "${EXTRA[@]}" 2>&1 | tail -25
  nequip-compile "$OUT/train_dir/best.ckpt" "$OUT/compiled_best.nequip.pth" --mode torchscript --device cpu 2>&1 | tail -3
  cd "$SLURM_SUBMIT_DIR"
}
echo "== smoke carbon";  smoke results/pruned/stratified_s0/DS_2km6t3p8hxx2_0_0.050_stratified_s0.xyz results/smoke/carbon ""
echo "== smoke sio2";    smoke results/ext/sio2/pruned/stratified_s0/train_0.050_stratified_s0.xyz results/smoke/sio2 "Si,O"
echo "== smoke monbtavw"; smoke results/ext/monbtavw/pruned/stratified_s0/train_0.050_stratified_s0.xyz results/smoke/monbtavw "Mo,Nb,Ta,V,W"
echo "== evaluate (CPU)"
python evaluation/evaluate_models.py --test data/random_C.xyz --models results/smoke/carbon/compiled_best.nequip.pth --out results/smoke/eval_carbon.csv
python evaluation/evaluate_models.py --test data/external/sio2/test.xyz --no-unit-conversion --models results/smoke/sio2/compiled_best.nequip.pth --out results/smoke/eval_sio2.csv
python evaluation/evaluate_models.py --test data/external/monbtavw/test.xyz --no-unit-conversion --models results/smoke/monbtavw/compiled_best.nequip.pth --out results/smoke/eval_monbtavw.csv
head -3 results/smoke/eval_*.csv
echo "SMOKE DONE"
