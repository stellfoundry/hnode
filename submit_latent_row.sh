#!/bin/bash
# Submit one row of the encoder x time-stepper comparison: all six time steppers on one latent space,
# as chained 2-hour jobs (each resumes the last), then the evaluation. Run from the hnode repo.
#
#   ./submit_latent_row.sh LATENT_DIR R OUT DECODER [AFTER_JOBID]
#   e.g. ./submit_latent_row.sh ../hnode_runs/pod/full 128 ../hnode_runs/rows/pod128 pod:128
#
# Environment: HNODE_CHAIN (jobs for HNODE, default 4), OTHERS_CHAIN (default 2).
set -e
LATENT_DIR=$(realpath "$1"); R=$2; OUT=$(realpath -m "$3"); DECODER=$4; AFTER=$5
mkdir -p "$OUT"
common="--parsable -o $OUT/%x-%j.out --export=ALL,LATENT_DIR=$LATENT_DIR,R=$R,OUT=$OUT,DECODER=$DECODER"
chain() {  # chain NAME JOB N -> last job id
    local dep=${AFTER:+--dependency=afterok:$AFTER} id
    for i in $(seq "$3"); do
        id=$(sbatch $common,JOB=$2 -J "$1" $dep run_latent_job.sh)
        dep="--dependency=afterany:$id"
    done
    echo "$id"
}
h=$(chain "row-hnode" hnode "${HNODE_CHAIN:-4}")
o=$(chain "row-others" others "${OTHERS_CHAIN:-2}")
e=$(sbatch $common,JOB=eval -J row-eval --dependency=afterany:$h:$o run_latent_job.sh)
echo "row $OUT: hnode chain ends $h, others chain ends $o, eval $e"
