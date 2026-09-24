#!/bin/bash
# ---------------------------------------------------------------------------
# Chain the round-1 pipeline. Run this from a LOGIN NODE, not with sbatch:
#
#     bash round1_code/round1_generateVASP_trainNEP.sbatch
#
#   1. VASP array over round-1 structures
#   2. audit + build round1_out/dataset from round0 + round1 (cumulative)
#   3. NEP training on that cumulative dataset
#
# It moves to the project root (the folder holding round1_code/) by itself,
# so every job below is submitted from there and all paths are relative to it.
# Step 1 (sbatch round1_code/sbatch_round1.sbatch) must have finished first,
# and round0_out/ must exist.
#
# It only calls sbatch, so it needs no allocation of its own.
# ---------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PATHFILE="round1_out/round1_1_vasp_job_paths.txt"
PREV_PATHFILE="round0_out/round0_1_vasp_job_paths.txt"
PREV_MANIFEST="round0_out/dataset/split_manifest.csv"
PREV_NEP="round0_out/round0_nep.txt"
VASP_SCRIPT="round1_code/round1_2_anvil_subVASP.sbatch"
CHECK_SCRIPT="round1_code/round1_34_anvil_check_and_build.sbatch"
NEP_SCRIPT="round1_code/round1_nep.sbatch"

# sbatch creates each -o file before the job script runs, so the log
# directory has to exist beforehand or the tasks fail instantly with no
# explanation. Every round-1 job logs to round1_out/logs.
mkdir -p round1_out/logs

for f in "$PATHFILE" "$VASP_SCRIPT" "$CHECK_SCRIPT" "$NEP_SCRIPT" \
         round1_code/round1_5_build_dataset.py round1_code/round1_nep-1.in; do
    [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done

# --- round-0 artefacts needed for accumulation ----------------------------
[ -f "$PREV_PATHFILE" ] || {
    echo "missing: $PREV_PATHFILE"
    echo "  round-1 trains on round0 + round1 data, so round-0's job list"
    echo "  is required. Cannot continue."
    exit 1
}
if [ ! -f "$PREV_MANIFEST" ]; then
    echo "WARNING: $PREV_MANIFEST not found."
    echo "         round-0's train/test split cannot be carried forward and"
    echo "         will be recomputed, so test.xyz will not match round 0's."
fi
if [ ! -f "$PREV_NEP" ]; then
    echo "NOTE: $PREV_NEP not found -- round 1 will train from scratch"
    echo "      on the cumulative dataset rather than continuing round 0."
fi

# --- keep the array range in sync with the path list ----------------------
N=$(grep -c . "$PATHFILE")
LAST=$((N - 1))
echo
echo "$PATHFILE has $N entries -> array 0-$LAST"
echo "$PREV_PATHFILE has $(grep -c . "$PREV_PATHFILE") entries (carried forward)"

# --- 1. VASP arrays, chunked to respect MaxArraySize ----------------------
# SLURM rejects any array whose highest index reaches MaxArraySize. Anvil's
# default is 1001, so 0-1559 fails with "Invalid job array specification".
# Split the list into chunks and pass each chunk's starting line as
# CHUNK_OFFSET; the VASP script adds it to SLURM_ARRAY_TASK_ID.
MAXARRAY=$(scontrol show config 2>/dev/null \
           | awk -F= '/^MaxArraySize/{gsub(/ /,"",$2); print $2}')
[ -n "${MAXARRAY:-}" ] || MAXARRAY=1001
CHUNK=$((MAXARRAY - 1))
THROTTLE=100

echo "MaxArraySize = $MAXARRAY  ->  chunks of $CHUNK"

ARRAY_IDS=""
OFFSET=0
CHUNK_NO=1
while [ "$OFFSET" -lt "$N" ]; do
    REMAIN=$((N - OFFSET))
    SIZE=$(( REMAIN < CHUNK ? REMAIN : CHUNK ))
    HI=$((SIZE - 1))

    JID=$(sbatch --parsable \
                 --array="0-${HI}%${THROTTLE}" \
                 --export=ALL,CHUNK_OFFSET="$OFFSET" \
                 "$VASP_SCRIPT")
    echo "  chunk $CHUNK_NO: array $JID  lines $((OFFSET + 1))-$((OFFSET + SIZE))  (${SIZE} tasks)"

    ARRAY_IDS="${ARRAY_IDS:+$ARRAY_IDS:}$JID"
    OFFSET=$((OFFSET + SIZE))
    CHUNK_NO=$((CHUNK_NO + 1))
done
echo "submitted $((CHUNK_NO - 1)) VASP array(s), ${N} tasks total"

# --- 2. audit + cumulative dataset build ----------------------------------
# Depends on EVERY chunk, so the build only starts once all of them end.
CHECK_ID=$(sbatch --parsable --dependency=afterany:"$ARRAY_IDS" "$CHECK_SCRIPT")
echo "submitted check+build: $CHECK_ID  (depends on $ARRAY_IDS)"

# --- 3. NEP training, only if the dataset build succeeded -----------------
NEP_ID=$(sbatch --parsable --dependency=afterok:"$CHECK_ID" "$NEP_SCRIPT")
echo "submitted NEP training: $NEP_ID  (depends on $CHECK_ID)"

echo
echo "watch with:  squeue -u \$USER"
echo "logs      :  round1_out/logs/"
echo "combined  :  round1_out/round1_1_combined_paths.txt"
echo "dataset   :  round1_out/dataset/train.xyz  round1_out/dataset/test.xyz"
echo "potential :  round1_out/round1_nep.txt"
echo "resubmit list (if any): round1_out/round1_3_resubmit.txt"
