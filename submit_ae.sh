#!/bin/bash
# Train a two-stage autoencoder (chained 2-hour jobs), encode every trajectory, then submit its row of
# time steppers. Run from the hnode repo.
#
#   ./submit_ae.sh KIND AE_DIR [CHAIN]      KIND: cnn | swin
#   e.g. ./submit_ae.sh swin ../hnode_runs/ae/swin64 5
#
# The latents go to AE_DIR/latents, the row to AE_DIR/row (see submit_latent_row.sh).
set -e
KIND=$1; AE_DIR=$(realpath -m "$2"); CHAIN=${3:-3}
mkdir -p "$AE_DIR"
common="--parsable -o $AE_DIR/%x-%j.out --export=ALL,AE_KIND=$KIND,AE_DIR=$AE_DIR,LATENT_DIR=$AE_DIR/latents"
dep=""
for i in $(seq "$CHAIN"); do
    id=$(sbatch $common,JOB=ae -J "ae-$KIND" $dep run_latent_job.sh)
    dep="--dependency=afterany:$id"
done
enc=$(sbatch $common,JOB=encode -J "ae-$KIND-encode" $dep run_latent_job.sh)
echo "autoencoder $KIND: training chain ends $id, encode $enc"
./submit_latent_row.sh "$AE_DIR/latents" "${LATENT:-64}" "$AE_DIR/row" "ae:$AE_DIR/ae.pt" "$enc"
