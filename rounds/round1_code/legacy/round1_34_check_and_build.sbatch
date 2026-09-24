#!/bin/bash
#SBATCH -J r1_check
#SBATCH -A mat260051
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH -o ./out-round1/z-check-%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# 1. Audit every round-1 VASP folder and write resubmission lists (round1
#    jobs only -- round0's own audit already lives in round0_3_*.txt).
# 2. Build the CUMULATIVE dataset: round0's job paths + round1's job paths,
#    with round0's train/test split frozen via --carry-split-from, so
#    round0_dataset/test.xyz's membership doesn't get reshuffled just because
#    the pool grew. Output goes to round1_dataset/, round0_dataset/ is left
#    untouched.
#
# Prerequisite: round0's resubmit backlog should be resolved (or at least
# checked) before running this -- round0_3_resubmit.txt lists what's still
# missing. Anything in there is still absent from round1_dataset too, since
# the builder re-screens convergence itself; this script just warns rather
# than blocking, in case you're deliberately moving on without it.
# ---------------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR" || exit 1

PATHFILE="round1_1_vasp_job_paths.txt"
OUT_MAXED="round1_3_maxed_out_jobs.txt"
OUT_FAILED="round1_3_timed_out_or_cancelled.txt"
OUT_MISSING="round1_3_never_ran.txt"
OUT_RESUBMIT="round1_3_resubmit.txt"

: > "$OUT_MAXED"
: > "$OUT_FAILED"
: > "$OUT_MISSING"

n_ok=0; n_maxed=0; n_failed=0; n_missing=0

while IFS= read -r DIR; do
    [ -z "$DIR" ] && continue
    if [ ! -d "$DIR" ]; then
        echo "$DIR" >> "$OUT_MISSING"; n_missing=$((n_missing+1)); continue
    fi

    OUTCAR="$DIR/OUTCAR"
    LOG="$DIR/VASP_job.out"

    # never started, or killed before writing anything
    if [ ! -f "$OUTCAR" ]; then
        echo "$DIR" >> "$OUT_MISSING"; n_missing=$((n_missing+1)); continue
    fi

    # slurm-level failure, or VASP died mid-run
    if [ -f "$LOG" ] && grep -qE "DUE TO TIME LIMIT|CANCELLED|BAD TERMINATION|slurmstepd: error" "$LOG"; then
        echo "$DIR" >> "$OUT_FAILED"; n_failed=$((n_failed+1)); continue
    fi
    if ! grep -q "Voluntary context switches" "$OUTCAR"; then
        echo "$DIR" >> "$OUT_FAILED"; n_failed=$((n_failed+1)); continue
    fi

    if grep -q "aborting loop because EDIFF is reached" "$OUTCAR"; then
        n_ok=$((n_ok+1))
    else
        echo "$DIR" >> "$OUT_MAXED"; n_maxed=$((n_maxed+1))
    fi
done < "$PATHFILE"

cat "$OUT_MAXED" "$OUT_FAILED" "$OUT_MISSING" | sort -u > "$OUT_RESUBMIT"

echo "------------------------------------------------"
echo "round1 converged        : $n_ok"
echo "round1 hit NELM         : $n_maxed   -> $OUT_MAXED"
echo "round1 died / timed out : $n_failed  -> $OUT_FAILED"
echo "round1 never ran        : $n_missing -> $OUT_MISSING"
echo "round1 to resubmit      : $(wc -l < "$OUT_RESUBMIT") -> $OUT_RESUBMIT"
echo "------------------------------------------------"

if [ -s round0_3_resubmit.txt ]; then
    echo "WARNING: round0_3_resubmit.txt still has $(wc -l < round0_3_resubmit.txt)" \
         "entries -- those round0 folders are still missing/unconverged and" \
         "will be absent from round1_dataset too."
fi

# ---------------------------------------------------------------------------
# Build the cumulative dataset from round0 + round1 job paths, freezing
# round0's split via round0_dataset/split_manifest.csv (must already exist --
# i.e. round0's own dataset build must have run first).
# ---------------------------------------------------------------------------

cat round0_1_vasp_job_paths.txt "$PATHFILE" > round1_1_combined_paths.txt

module load conda
module use "$HOME/privatemodules"
module load conda-env/nepdata

if [ ! -f round0_dataset/split_manifest.csv ]; then
    echo "ERROR: round0_dataset/split_manifest.csv not found -- run round0's" \
         "dataset build first, or drop --carry-split-from below if you" \
         "intend round1_dataset to be a fresh, unfrozen split."
    exit 1
fi

python round0_5_build_dataset.py \
    --paths round1_1_combined_paths.txt \
    --outdir round1_dataset \
    --test-fraction 0.2 \
    --split-mode family \
    --family-eps 0.06 \
    --carry-split-from round0_dataset/split_manifest.csv
