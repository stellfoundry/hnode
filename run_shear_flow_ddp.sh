#!/bin/bash
# Data-parallel shear-flow training on Frontier: 1 node, 8 GPUs (GCDs), one rank per GPU.
#
#   sbatch run_shear_flow_ddp.sh                 # batch job
#   bash run_shear_flow_ddp.sh                   # inside an salloc allocation
#   MAX_STEPS=5 RUN_EVAL=0 sbatch run_shear_flow_ddp.sh   # short smoke test
#
#SBATCH -A fus184
#SBATCH -J hnode-sf-ddp
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o hnode-sf-ddp-%j.out

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source ./module-to-load-frontier.sh

# Call the env's python by path: when the submitting shell already had the env active, sbatch copies
# that environment, `module load python` puts the base python first on PATH, and `conda activate`
# is then a no-op, so a bare `python` would be the base interpreter (no torch).
PYTHON="$CONDA_PREFIX/bin/python"
"$PYTHON" -c "import torch" || { echo "torch not importable with $PYTHON" >&2; exit 1; }
echo "Using $PYTHON"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
# For multi-node runs, RCCL should use the Slingshot interfaces:
# export NCCL_SOCKET_IFNAME=hsn0

srun -N 1 -n 8 -c 7 --gpus-per-task=1 --gpu-bind=closest \
    "$PYTHON" -u 03_shear_flow_ddp.py
