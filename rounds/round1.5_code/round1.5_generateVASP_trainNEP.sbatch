#!/bin/bash
# ---------------------------------------------------------------------------
# Chain the round-1.5 pipeline. Run this from a LOGIN NODE, not with sbatch:
#
#     bash round1.5_code/round1.5_generateVASP_trainNEP.sbatch
#
#   1. rebuild the re-run list from round 0's and round 1's resubmit lists
#   2. re-run those VASP folders in place, keeping WAVECAR/CHGCAR so a job
#      that hits the walltime can be continued next time
#   3. audit them, then build the cumulative dataset (all round-0 + round-1
#      folders that now converge)
#   4. train the NEP on that dataset, continuing from round 1's restart
#
# Round 1.5 generates no new structures, so there is no round1.5_out/vasp/.
#
# Optional: cap how much restart data an unconverged folder may keep, e.g.
#     MAX_RESTART_MB=8000 bash round1.5_code/round1.5_generateVASP_trainNEP.sbatch
#
# It moves to the project root itself and only calls sbatch, so it needs no
# allocation of its own.
# ---------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PATHFILE="round1.5_out/round1.5_1_rerun_paths.txt"
LIST_SCRIPT="round1.5_code/round1.5_1_make_rerun_list.sh"
VASP_SCRIPT="round1.5_code/round1.5_2_anvil_rerun_vasp.sbatch"
CHECK_SCRIPT="round1.5_code/round1.5_34_anvil_check_and_build.sbatch"
NEP_SCRIPT="round1.5_code/round1.5_nep.sbatch"
PREV_MANIFEST="round1_out/dataset/split_manifest.csv"
PREV_NEP="round1_out/round1_nep.txt"

# sbatch creates each -o file before the job script runs, so the log
# directory has to exist beforehand or the tasks fail instantly with no
# explanation. Every round-1.5 job logs to round1.5_out/logs.
mkdir -p round1.5_out/logs

for f in "$LIST_SCRIPT" "$VASP_SCRIPT" "$CHECK_SCRIPT" "$NEP_SCRIPT" \
         round1.5_code/round1.5_5_build_dataset.py round1.5_code/round1.5_nep-1.in; do
    [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done

for f in round0_out/round0_1_vasp_job_paths.txt round1_out/round1_1_vasp_job_paths.txt; do
    [ -f "$f" ] || {
        echo "missing: $f"
        echo "  round 1.5 builds a cumulative dataset, so both rounds' job"
        echo "  lists are required. Cannot continue."
        exit 1
    }
done

if [ ! -f "$PREV_MANIFEST" ]; then
    echo "WARNING: $PREV_MANIFEST not found."
    echo "         The train/test split will fall back to round 0's manifest,"
    echo "         or be recomputed if that is missing too."
fi
if [ ! -f "$PREV_NEP" ]; then
    echo "NOTE: $PREV_NEP not found -- round 1.5 will train from scratch on"
    echo "      the cumulative dataset rather than continuing round 1."
fi

# --- 1. what still has to run --------------------------------------------
echo
bash "$LIST_SCRIPT"
echo

N=$(grep -c . "$PATHFILE" || true)

if [ "$N" -eq 0 ]; then
    echo "Nothing to re-run -- skipping the VASP array and going straight to"
    echo "the dataset build."
    CHECK_ID=$(sbatch --parsable "$CHECK_SCRIPT")
    echo "submitted check+build: $CHECK_ID"
else
    # --- 2. VASP arrays, chunked to respect MaxArraySize ------------------
    # SLURM rejects any array whose highest index reaches MaxArraySize. Anvil's
    # default is 1001. Each chunk passes its starting line as CHUNK_OFFSET;
    # the re-run script adds it to SLURM_ARRAY_TASK_ID.
    MAXARRAY=$(scontrol show config 2>/dev/null \
               | awk -F= '/^MaxArraySize/{gsub(/ /,"",$2); print $2}' || true)
    [ -n "${MAXARRAY:-}" ] || MAXARRAY=1001
    CHUNK=$((MAXARRAY - 1))
    THROTTLE=100

    echo "$PATHFILE has $N entries; MaxArraySize = $MAXARRAY -> chunks of $CHUNK"

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

    # --- 3. audit + cumulative dataset build ------------------------------
    # Depends on EVERY chunk, so the build only starts once all of them end.
    CHECK_ID=$(sbatch --parsable --dependency=afterany:"$ARRAY_IDS" "$CHECK_SCRIPT")
    echo "submitted check+build: $CHECK_ID  (depends on $ARRAY_IDS)"
fi

# --- 4. NEP training, only if the dataset build succeeded -----------------
NEP_ID=$(sbatch --parsable --dependency=afterok:"$CHECK_ID" "$NEP_SCRIPT")
echo "submitted NEP training: $NEP_ID  (depends on $CHECK_ID)"

echo
echo "watch with:  squeue -u \$USER"
echo "logs      :  round1.5_out/logs/"
echo "re-run list:  $PATHFILE"
echo "combined  :  round1.5_out/round1.5_1_combined_paths.txt"
echo "dataset   :  round1.5_out/dataset/train.xyz  round1.5_out/dataset/test.xyz"
echo "potential :  round1.5_out/round1.5_nep.txt"
echo "still unfinished afterwards: round1.5_out/round1.5_3_resubmit.txt"
echo "  (those stopped at the walltime are in round1.5_3_stopped_at_wall.txt"
echo "   and continue from their WAVECAR if you run this chain again)"
