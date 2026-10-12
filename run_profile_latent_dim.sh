#!/bin/bash
# HNODE cost vs latent dimension on shear-flow POD coefficients (profile_latent_dim.py), 1 node:
# every (F, config) case is its own process, 8 at a time (one per GCD), 40-minute limit each.
#
#   sbatch run_profile_latent_dim.sh [OUT_DIR]
#
#SBATCH -A fus184
#SBATCH -J hnode-latent-dim
#SBATCH -N 1
#SBATCH -t 02:00:00

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
[ -f profile_latent_dim.py ] || { echo "submit from the hnode repo directory" >&2; exit 1; }
source ./module-to-load-frontier.sh
PYTHON="$CONDA_PREFIX/bin/python"  # see run_shear_flow_ddp.sh
OUT=${1:-$(realpath ../hnode_runs/latent_dim_profile/sweep)}
mkdir -p "$OUT"

cases=()
for F in 1024 512 256 128 64; do  # largest first, so the long cases start early
    for c in dense block block+detach block+detach+chunk64; do
        [ "$c" = block+detach ] && { [ "$F" = 256 ] || [ "$F" = 1024 ]; } && c=block+detach+gradcheck
        cases+=("$F $c")
    done
done

run_gpu() {  # run_gpu GPU_INDEX: this GCD's share of the cases, one process each
    local g=$1 i=0
    for case in "${cases[@]}"; do
        if [ $((i % 8)) -eq "$g" ]; then
            set -- $case
            ROCR_VISIBLE_DEVICES=$g HIP_VISIBLE_DEVICES=0 timeout 2400 "$PYTHON" -u profile_latent_dim.py "$1" "$2" \
                "$OUT/F$1_$2.json" > "$OUT/F$1_$2.log" 2>&1 || echo "F=$1 $2: exit $?" >> "$OUT/failures.txt"
        fi
        i=$((i + 1))
    done
}
for g in 0 1 2 3 4 5 6 7; do run_gpu "$g" & done
wait
"$PYTHON" profile_latent_dim.py summary "$OUT"
