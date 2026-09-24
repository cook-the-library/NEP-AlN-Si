#!/bin/bash
#SBATCH -J r0_check
#SBATCH -A mat260051
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH -o round0_out/logs/z-check-%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# 1. Audit every round-0 VASP folder and write resubmission lists.
# 2. Build train.xyz / test.xyz in round0_out/dataset/.
#
# Submit from the project root; every path below is relative to it.
# ---------------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -d round0_code ] || { echo "no round0_code/ in $PWD -- submit from the project root"; exit 1; }
mkdir -p round0_out

PATHFILE="round0_out/round0_1_vasp_job_paths.txt"
OUTDIR="round0_out/dataset"
OUT_MAXED="round0_out/round0_3_maxed_out_jobs.txt"
OUT_FAILED="round0_out/round0_3_timed_out_or_cancelled.txt"
OUT_MISSING="round0_out/round0_3_never_ran.txt"
OUT_RESUBMIT="round0_out/round0_3_resubmit.txt"

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

    # finished cleanly but the electronic loop hit NELM instead of EDIFF.
    # This is the correct test. Do NOT compare the iteration count against a
    # hardcoded number: the generated INCARs use NELM = 200, so a threshold
    # of 100 would discard jobs that converged legitimately at step 130.
    if grep -q "aborting loop because EDIFF is reached" "$OUTCAR"; then
        n_ok=$((n_ok+1))
    else
        echo "$DIR" >> "$OUT_MAXED"; n_maxed=$((n_maxed+1))
    fi
done < "$PATHFILE"

cat "$OUT_MAXED" "$OUT_FAILED" "$OUT_MISSING" | sort -u > "$OUT_RESUBMIT"

echo "------------------------------------------------"
echo "converged        : $n_ok"
echo "hit NELM         : $n_maxed   -> $OUT_MAXED"
echo "died / timed out : $n_failed  -> $OUT_FAILED"
echo "never ran        : $n_missing -> $OUT_MISSING"
echo "to resubmit      : $(wc -l < "$OUT_RESUBMIT") -> $OUT_RESUBMIT"
echo "------------------------------------------------"

# ---------------------------------------------------------------------------
# Build the dataset. The builder re-screens convergence itself, so it stays
# correct even if the audit above is out of date.
#
# ONE-TIME environment setup, run once from a login node, not here:
#     module load conda
#     conda-env-mod create -n nepdata -y
#     module use $HOME/privatemodules
#     module load conda-env/nepdata
#     pip install ase numpy
# ---------------------------------------------------------------------------

module load conda
module use "$HOME/privatemodules"
module load conda-env/nepdata

python round0_code/round0_5_build_dataset.py \
    --paths "$PATHFILE" \
    --outdir "$OUTDIR" \
    --test-fraction 0.2 \
    --split-mode family \
    --family-eps 0.06
