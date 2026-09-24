#!/bin/bash
# ---------------------------------------------------------------------------
# Chain the round-0 pipeline. Run this from a LOGIN NODE, not with sbatch:
#
#     bash round0_code/round0_generateVASP+trainNEP.sbatch
#
# It moves to the project root (the folder holding round0_code/) by itself,
# so every job below is submitted from there and all paths are relative to it.
# Step 1 (sbatch round0_code/sbatch_round0.sbatch) must have finished first.
#
# It only calls sbatch three times, so it needs no allocation of its own.
# Submitting it as a job would burn walltime doing nothing but waiting.
# ---------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PATHFILE="round0_out/round0_1_vasp_job_paths.txt"
VASP_SCRIPT="round0_code/round0_2_anvil_subVASP.sbatch"
CHECK_SCRIPT="round0_code/round0_34_anvil_check_and_build.sbatch"
NEP_SCRIPT="round0_code/round0_8_nep.sbatch"

# sbatch creates the -o output file before the job script runs, so this
# directory has to exist beforehand or every task fails immediately.
mkdir -p round0_out/logs

for f in "$PATHFILE" "$VASP_SCRIPT" "$CHECK_SCRIPT" "$NEP_SCRIPT" \
         round0_code/round0_5_build_dataset.py round0_code/round0_nep-1.in; do
    [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done

# --- keep the array range in sync with the path list ----------------------
N=$(grep -c . "$PATHFILE")
LAST=$((N - 1))
echo "$PATHFILE has $N entries -> array 0-$LAST"

CUR=$(grep -oP '(?<=--array=)[^ ]+' "$VASP_SCRIPT" | head -1)
if [ "${CUR%\%*}" != "0-$LAST" ]; then
    echo "WARNING: $VASP_SCRIPT has --array=$CUR but the list needs 0-$LAST"
    echo "         overriding on the command line for this submission"
fi

# --- 1. VASP array --------------------------------------------------------
ARRAY_ID=$(sbatch --parsable --array="0-${LAST}%100" "$VASP_SCRIPT")
echo "submitted VASP array : $ARRAY_ID  (${N} tasks, 100 concurrent)"

# --- 2. audit + dataset build, after every array task ends ----------------
CHECK_ID=$(sbatch --parsable --dependency=afterany:"$ARRAY_ID" "$CHECK_SCRIPT")
echo "submitted check+build: $CHECK_ID  (depends on $ARRAY_ID)"

# --- 3. NEP training, only if the dataset build succeeded -----------------
# afterok, not afterany: if train.xyz was never written there is nothing to
# train on and the GPU allocation would be wasted.
NEP_ID=$(sbatch --parsable --dependency=afterok:"$CHECK_ID" "$NEP_SCRIPT")
echo "submitted NEP training: $NEP_ID  (depends on $CHECK_ID)"

echo
echo "watch with:  squeue -u \$USER"
echo "logs      :  round0_out/logs/"
echo "dataset   :  round0_out/dataset/train.xyz  round0_out/dataset/test.xyz"
echo "potential :  round0_out/round0_nep.txt"
echo "resubmit list (if any): round0_out/round0_3_resubmit.txt"
