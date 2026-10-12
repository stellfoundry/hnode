#!/bin/bash
# Reproduce the reference shear-flow checkpoint (shear_flow_redo_part_2.pt) from scratch with
# 03_shear_flow_ddp.py: 1 node, 8 GPUs x 1 trajectory = the notebook's batch_size 8, SOAP lr 1e-4.
# Each 24-hour job resumes from REPRO_DIR/checkpoint.pt (or starts from scratch if it is missing) and
# saves cleanly before the limit, so jobs can be chained or resubmitted at any time:
#
#   sbatch run_shear_flow_repro.sh                              # one 24-hour job
#   sbatch --dependency=afterany:<jobid> run_shear_flow_repro.sh   # the next one
#
# Other layouts: sbatch -N <nodes> with RANKS_PER_NODE (default 8, one per GCD) and BATCH_PER_RANK; the
# global batch is nodes x RANKS_PER_NODE x BATCH_PER_RANK. Across nodes RCCL uses the Slingshot
# interfaces, through the aws-ofi-rccl plugin in RCCL_PLUGIN_DIR when that is set.
#
#SBATCH -A fus184
#SBATCH -J hnode-sf-repro
#SBATCH -p extended
#SBATCH -N 1
#SBATCH -t 24:00:00

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
[ -f 03_shear_flow_ddp.py ] || { echo "submit from the hnode repo directory" >&2; exit 1; }
source ./module-to-load-frontier.sh
PYTHON="$CONDA_PREFIX/bin/python"  # see run_shear_flow_ddp.sh
"$PYTHON" -c "import torch" || { echo "torch not importable with $PYTHON" >&2; exit 1; }

REPRO_DIR=${REPRO_DIR:-$(realpath ../hnode_runs/repro)}
mkdir -p "$REPRO_DIR"
export CHECKPOINT_PATH="$REPRO_DIR/checkpoint.pt" SAVE_PATH="$REPRO_DIR/checkpoint.pt"
export TOTAL_STEPS=${TOTAL_STEPS:-144000}   # the reference's length
export TIME_LIMIT=${TIME_LIMIT:-84600}      # 23.5 h: leave time to save before the 24 h limit
export TEST_EVERY=${TEST_EVERY:-250}        # the rank-0 test loss stalls every rank; 25 costs ~30%
export NUM_EPOCHS=100000 RUN_EVAL=0 SOLVER_VERBOSE=0
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

NODES=${SLURM_NNODES:-1}
RANKS_PER_NODE=${RANKS_PER_NODE:-8}
if [ "$NODES" -gt 1 ]; then
    export NCCL_SOCKET_IFNAME=hsn0
    if [ -n "$RCCL_PLUGIN_DIR" ]; then
        export LD_LIBRARY_PATH=$RCCL_PLUGIN_DIR:$LD_LIBRARY_PATH
        export FI_CXI_ATS=0 FI_MR_CACHE_MONITOR=userfaultfd NCCL_NET_GDR_LEVEL=3
    fi
fi
echo "Layout: $NODES node(s) x $RANKS_PER_NODE ranks x ${BATCH_PER_RANK:-1} trajectories; plugin: ${RCCL_PLUGIN_DIR:-none}"

srun -N "$NODES" -n $((NODES * RANKS_PER_NODE)) --ntasks-per-node="$RANKS_PER_NODE" -c 7 --gpus-per-task=1 \
    --gpu-bind=closest "$PYTHON" -u 03_shear_flow_ddp.py
