#!/bin/bash
#SBATCH -J nequip-cpu-timing
#SBATCH -p pdebug
#SBATCH -N 1
#SBATCH -t 00:45:00
#SBATCH -o results/timing_cpu/slurm-%j.out
# Time 40 training batches on one full CPU node with the paper-faithful stack
# (nequip 0.9.1).  Submit from the repo root:  sbatch training/slurm_timing_cpu.sh
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO" && mkdir -p results/timing_cpu && cd results/timing_cpu
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-$(nproc)} MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-$(nproc)}
echo "host=$(hostname) threads=$OMP_NUM_THREADS"
"$REPO/env/venv-paper/bin/python" -c "import nequip,torch;print('nequip',nequip.__version__,'torch',torch.__version__)"
/usr/bin/time -v "$REPO/env/venv-paper/bin/nequip-train" -cp "$REPO/training/configs" -cn timing_cpu_paperenv 2>&1 \
  | tr '\r' '\n' | grep -E "Epoch 0:.*(10|20|30|40)/40|Elapsed|Maximum resident|Error|Traceback" 
