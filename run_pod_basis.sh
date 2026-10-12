#!/bin/bash
# POD basis, coefficients and projection floor for shear flow (pod_basis.py): 1 node, 8 GPUs.
#
#   sbatch run_pod_basis.sh [OUT_DIR]
#
#SBATCH -A fus184
#SBATCH -J hnode-pod
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o hnode-pod-%j.out

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
[ -f pod_basis.py ] || { echo "submit from the hnode repo directory" >&2; exit 1; }
source ./module-to-load-frontier.sh
PYTHON="$CONDA_PREFIX/bin/python"  # see run_shear_flow_ddp.sh
"$PYTHON" -c "import torch" || { echo "torch not importable with $PYTHON" >&2; exit 1; }

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29503
OUT_DIR=${1:-../hnode_runs/pod/full}

srun -N 1 -n 8 -c 7 --gpus-per-task=1 --gpu-bind=closest \
    "$PYTHON" -u pod_basis.py "$OUT_DIR"
