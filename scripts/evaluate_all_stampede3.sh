#!/bin/bash
#SBATCH -J quests-eval
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH -t 08:00:00
#SBATCH -A TG-CHM250015
#SBATCH -o results/slurm/eval-%j.out
# Per-group test force MAE for every trained model (CPU).  Rerunnable: each
# evaluate_models.py call appends only models not yet in its CSV.  Submit from
# the repo root once training/jobs*.txt are done (train_dir/best.ckpt.done).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PATH=$PWD/env/venv-paper/bin:$PATH OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${OMP_NUM_THREADS:-16}
EV=${EVAL_DIR:-results/eval}
mkdir -p $EV
skip_done() {  # matrix csv_out -> temp matrix with rows whose model is not yet evaluated
  python - "$1" "$2" <<'PY'
import csv, sys, pathlib
m, out = sys.argv[1], pathlib.Path(sys.argv[2])
done = {r["ckpt_file"] for r in csv.DictReader(open(out))} if out.exists() else set()
import os, re
models = os.environ.get("MODELS_DIRNAME", "models")   # e.g. models_relstop for the relative-stopping retrain
rows = []
for r in csv.DictReader(open(m)):
    r["out_dir"] = re.sub(r"/models/", f"/{models}/", "/" + r["out_dir"])[1:]
    if str(pathlib.Path(r["out_dir"]) / "train_dir/best.ckpt") not in done:
        rows.append(r)
w = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys() if rows else ["exp"]); w.writeheader(); w.writerows(rows)
PY
}
# carbon matrix = E1-E5 plus the E6 published baselines (training/experiments_baselines.csv)
python -c "import pandas as pd, sys; pd.concat([pd.read_csv(f) for f in sys.argv[1:] if __import__('os').path.exists(f)]).to_csv('/tmp/m_carbon_all.csv', index=False)" \
    training/experiments.csv training/experiments_baselines.csv
skip_done /tmp/m_carbon_all.csv             $EV/carbon_test_mae.csv   > /tmp/m_carbon.csv
skip_done training/experiments_sio2.csv     $EV/sio2_test_mae.csv     > /tmp/m_sio2.csv
skip_done training/experiments_monbtavw.csv $EV/monbtavw_test_mae.csv > /tmp/m_monbtavw.csv
python evaluation/evaluate_models.py --test data/random_C.xyz            --matrix /tmp/m_carbon.csv   --out $EV/carbon_test_mae.csv ${SAVE_FORCES:+--save-forces $SAVE_FORCES/carbon}
python evaluation/evaluate_models.py --test data/external/sio2/test.xyz     --matrix /tmp/m_sio2.csv     --out $EV/sio2_test_mae.csv ${SAVE_FORCES:+--save-forces $SAVE_FORCES/sio2}     --no-unit-conversion
python evaluation/evaluate_models.py --test data/external/monbtavw/test.xyz --matrix /tmp/m_monbtavw.csv --out $EV/monbtavw_test_mae.csv ${SAVE_FORCES:+--save-forces $SAVE_FORCES/monbtavw} --no-unit-conversion
echo "progress ($EV):"; for d in carbon sio2 monbtavw; do n=$(grep -c '^Full,' $EV/${d}_test_mae.csv 2>/dev/null || true); echo "  $d: $n models evaluated"; done
