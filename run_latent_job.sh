#!/bin/bash
# One job of the encoder x time-stepper comparison (1 node, 8 GPUs). Submitted by submit_latent_row.sh
# and submit_ae.sh; JOB selects what runs:
#   hnode   latent_baselines.py hnode on 8 GPUs (data parallel; SOAP, lr 1e-4 as in the paper)
#   others  latent_baselines.py node1 node2 lstm koopman at Adam lr 1e-3, 3e-4, 1e-4 (OUT/MODEL-lrX) and dmd,
#           one GPU each, 8 at a time
#   eval    for each model the learning rate with the best valid rollout (best.json), then
#           eval_latent_rollout.py on the oracle and those models, then the comparison plot
#   ae      ae_train.py train $AE_KIND $AE_DIR
#   encode  ae_train.py encode $AE_DIR/ae.pt $LATENT_DIR
# Environment: LATENT_DIR, R, OUT (row directory), DECODER, AE_KIND, AE_DIR (+ the scripts' own knobs).
#
#SBATCH -A fus184
#SBATCH -N 1
#SBATCH -t 02:00:00
# (for a longer job: sbatch -p extended -t 24:00:00 ... with TIME_LIMIT=84600)

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
[ -f latent_baselines.py ] || { echo "submit from the hnode repo directory" >&2; exit 1; }
source ./module-to-load-frontier.sh
PYTHON="$CONDA_PREFIX/bin/python"  # see run_shear_flow_ddp.sh
"$PYTHON" -c "import torch" || { echo "torch not importable with $PYTHON" >&2; exit 1; }
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export TIME_LIMIT=${TIME_LIMIT:-6600}
SRUN8=${SRUN8:-"srun -N 1 -n 8 -c 7 --gpus-per-task=1 --gpu-bind=closest"}  # overridable, e.g. to run inside another allocation
echo "JOB=$JOB LATENT_DIR=$LATENT_DIR R=$R OUT=$OUT DECODER=$DECODER AE_KIND=$AE_KIND AE_DIR=$AE_DIR"

case "$JOB" in
hnode)
    MASTER_PORT=29504 $SRUN8 "$PYTHON" -u latent_baselines.py hnode "$OUT" ;;
others)
    port=29540
    runs=("dmd:")
    for m in node1 node2 lstm koopman; do for lr in 1e-3 3e-4 1e-4; do runs+=("$m:$lr"); done; done
    for run in "${runs[@]}"; do
        m=${run%%:*}; lr=${run#*:}
        tag=${lr:+-lr$lr}
        # srun queues the step until one of the 8 GPUs is free
        LR=${lr:-1e-3} TAG=$tag MASTER_PORT=$((port++)) srun -N 1 -n 1 -c 7 --gpus-per-task=1 --exact \
            "$PYTHON" -u latent_baselines.py "$m" "$OUT" > "$OUT/$m$tag.stdout" 2>&1 &
    done
    wait ;;
eval)
    # Best learning rate per model on the valid split (hnode and dmd have one run each).
    "$PYTHON" - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
best = {}
for h in glob.glob(os.path.join(out, "*", "history.json")):
    d = json.load(open(h))
    m = d["model"]
    if m not in best or d["best_valid"] < best[m]["best_valid"]:
        best[m] = {"dir": os.path.dirname(h), "best_valid": d["best_valid"], "lr": d.get("lr"), "optimizer": d.get("optimizer")}
json.dump(best, open(os.path.join(out, "best.json"), "w"), indent=1)
for m, b in sorted(best.items()):
    print(f"{m}: {os.path.basename(b['dir'])} ({b['optimizer']} lr {b['lr']}, valid {b['best_valid']:.4e})")
PY
    entries=()
    # Every HNODE variant (dt, optimizer, length) is scored separately, besides the best-per-model runs.
    for dir in "$OUT"/hnode*/; do
        dir=${dir%/}; name=$(basename "$dir")
        [ -f "$dir/pred.pt" ] && [ -f "$dir/history.json" ] || continue
        [ -f "$OUT/eval/$name/vrmse.npz" ] || MASTER_PORT=29502 $SRUN8 "$PYTHON" -u eval_latent_rollout.py \
            eval "$LATENT_DIR" "$dir/pred.pt" "$OUT/eval/$name"
    done
    for m in oracle hnode node1 node2 lstm koopman dmd; do
        if [ "$m" = oracle ]; then
            pred=oracle; name=oracle
        else
            dir=$("$PYTHON" -c "import json,sys; print(json.load(open('$OUT/best.json')).get('$m', {}).get('dir', ''))")
            [ -n "$dir" ] || { echo "skip $m: no finished run"; continue; }
            pred="$dir/pred.pt"; name=$(basename "$dir")
        fi
        [ -f "$OUT/eval/$name/vrmse.npz" ] || MASTER_PORT=29502 $SRUN8 "$PYTHON" -u eval_latent_rollout.py \
            eval "$LATENT_DIR" "$pred" "$OUT/eval/$name"
        entries+=("$name=$OUT/eval/$name")
    done
    for dir in "$OUT"/eval/hnode*/; do  # add HNODE variants not already listed
        [ -d "$dir" ] || continue  # no match: the pattern itself
        name=$(basename "${dir%/}")
        [[ " ${entries[*]} " == *" $name="* ]] || entries+=("$name=${dir%/}")
    done
    "$PYTHON" eval_latent_rollout.py compare "$OUT/eval/compare.png" "${entries[@]}" | tee "$OUT/eval/compare.txt" ;;
ae)
    MASTER_PORT=29505 $SRUN8 "$PYTHON" -u ae_train.py train "$AE_KIND" "$AE_DIR" ;;
encode)
    MASTER_PORT=29506 $SRUN8 "$PYTHON" -u ae_train.py encode "$AE_DIR/ae.pt" "$LATENT_DIR" ;;
*)
    echo "unknown JOB=$JOB" >&2; exit 1 ;;
esac
