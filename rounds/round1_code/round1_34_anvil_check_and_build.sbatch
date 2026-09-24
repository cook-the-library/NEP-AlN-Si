#!/bin/bash
#SBATCH -J r1_check
#SBATCH -A mat260051
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH -o round1_out/logs/z-check-%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# 1. Audit the round-1 VASP folders and write resubmission lists.
# 2. Build round1_out/dataset/ from the CUMULATIVE round0 + round1 data.
#
# Submit from the project root; every path below is relative to it.
#
# The dataset accumulates: round N trains on rounds 0..N. Training round 1 on
# round-1 data alone would give a potential that is good at the round-1
# configurations and has forgotten bulk, surfaces and the repulsive wall --
# which is most of what keeps MD trajectories stable.
# ---------------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -d round1_code ] || { echo "no round1_code/ in $PWD -- submit from the project root"; exit 1; }
mkdir -p round1_out

# round-1 jobs, for the audit
PATHFILE="round1_out/round1_1_vasp_job_paths.txt"
# round-0 jobs, already done
PREV_PATHFILE="round0_out/round0_1_vasp_job_paths.txt"
# union of both, for the dataset build
COMBINED="round1_out/round1_1_combined_paths.txt"
PREV_MANIFEST="round0_out/dataset/split_manifest.csv"
OUTDIR="round1_out/dataset"

OUT_MAXED="round1_out/round1_3_maxed_out_jobs.txt"
OUT_FAILED="round1_out/round1_3_timed_out_or_cancelled.txt"
OUT_MISSING="round1_out/round1_3_never_ran.txt"
OUT_RESUBMIT="round1_out/round1_3_resubmit.txt"

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

    if [ ! -f "$OUTCAR" ]; then
        echo "$DIR" >> "$OUT_MISSING"; n_missing=$((n_missing+1)); continue
    fi

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

echo "--------------- round-1 VASP audit ---------------"
echo "converged        : $n_ok"
echo "hit NELM         : $n_maxed   -> $OUT_MAXED"
echo "died / timed out : $n_failed  -> $OUT_FAILED"
echo "never ran        : $n_missing -> $OUT_MISSING"
echo "to resubmit      : $(wc -l < "$OUT_RESUBMIT") -> $OUT_RESUBMIT"
echo "--------------------------------------------------"

# --- build the combined path list -----------------------------------------
# Order-preserving dedupe, so a folder listed in both rounds is read once.
if [ ! -f "$PREV_PATHFILE" ]; then
    echo "ERROR: $PREV_PATHFILE not found -- cannot accumulate round-0 data"
    exit 1
fi
cat "$PREV_PATHFILE" "$PATHFILE" | grep -v '^[[:space:]]*$' \
    | awk '!seen[$0]++' > "$COMBINED"

echo "round0 paths : $(grep -c . "$PREV_PATHFILE")"
echo "round1 paths : $(grep -c . "$PATHFILE")"
echo "combined     : $(grep -c . "$COMBINED")  -> $COMBINED"
echo

# ---------------------------------------------------------------------------
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

# --carry-split-from pins every round-0 folder to the train/test label it
# already had, and pins whole families with them, so a round-1 near-duplicate
# of a round-0 test structure cannot leak into train. Without it, adding
# round-1 data reshuffles round-0's split too and RMSE stops being comparable
# across rounds.
#
# Set --test-fraction 0 instead if you would rather freeze test.xyz at exactly
# round-0's contents and send all new data to train.
CARRY=""
if [ -f "$PREV_MANIFEST" ]; then
    CARRY="--carry-split-from $PREV_MANIFEST"
    echo "carrying round-0 train/test split from $PREV_MANIFEST"
else
    echo "WARNING: $PREV_MANIFEST not found -- the split will be recomputed"
    echo "         from scratch and round-0's test set will NOT be preserved"
fi

python round1_code/round1_5_build_dataset.py \
    --paths "$COMBINED" \
    --outdir "$OUTDIR" \
    --test-fraction 0.2 \
    --split-mode family \
    --family-eps 0.06 \
    $CARRY
