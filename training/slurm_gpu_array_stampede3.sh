#!/bin/bash
# Run a jobs file (training/jobs*.txt, one train_one.sh command per line) on
# Stampede3 GPU nodes: 4 GPUs per node, one training per GPU, N nodes.
#
#   PARTITION=<queue> NODES=4 JOBS=training/jobs.txt sbatch training/slurm_gpu_array_stampede3.sh
#
# Each worker (node, gpu) takes lines i with i % (4*NODES) == rank, in order.
# Finished models (train_dir/best.ckpt.done) are skipped and interrupted ones
# resume from last.ckpt (train_one.sh), so the same command can be resubmitted
# until everything is done.  Submit from the repo root.
#SBATCH -J quests-train
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 47:00:00
#SBATCH -A TG-CHM250015
#SBATCH -o results/slurm/train-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
JOBS=${JOBS:-training/jobs.txt}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NODES=${SLURM_JOB_NUM_NODES:-1}
export NEQUIP_PYTHON=$PWD/env/venv-paper/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${OMP_NUM_THREADS:-8}
mapfile -t LINES < "$JOBS"
echo "host=$(hostname) nodes=$NODES jobs=$JOBS lines=${#LINES[@]} gpus/node=$GPUS_PER_NODE start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null | head -5
"$NEQUIP_PYTHON" -c "import torch,nequip; assert torch.cuda.is_available(), 'no CUDA'; print('cuda ok', torch.cuda.device_count(), 'GPUs; nequip', nequip.__version__)" || { echo "CUDA check failed, aborting"; exit 1; }

worker() {  # node_index gpu
  local rank=$(( $1 * GPUS_PER_NODE + $2 )) nw=$(( NODES * GPUS_PER_NODE ))
  export CUDA_VISIBLE_DEVICES=$2
  for (( i=rank; i<${#LINES[@]}; i+=nw )); do
    local line=${LINES[$i]}; [ -z "$line" ] && continue
    local out=$(echo "$line" | awk '{print $4}')
    if [ -f "$out/train_dir/best.ckpt.done" ]; then echo "[w$rank] skip done: $out"; continue; fi
    echo "[w$rank] $(date +%F_%T) start line $i: $line"
    mkdir -p "$out"
    bash $line > "$out/worker.log" 2>&1 && echo "[w$rank] $(date +%F_%T) OK   $out" || echo "[w$rank] $(date +%F_%T) FAIL $out (see $out/worker.log)"
  done
}
export -f worker; export LINES GPUS_PER_NODE NODES
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
for n in $(seq 0 $((NODES-1))); do
  srun -N1 -n1 -w "${NODELIST[$n]}" --export=ALL bash -c "
    cd $PWD; mapfile -t LINES < '$JOBS'; export LINES
    for g in \$(seq 0 $((GPUS_PER_NODE-1))); do worker $n \$g & done; wait" &
done
wait
echo "all workers finished $(date)"
