#!/bin/bash
#SBATCH -J r15_check
#SBATCH -A mat260051
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH -o round1.5_out/logs/z-check-%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# 1. Audit the folders round 1.5 re-ran and write new resubmission lists.
# 2. Build round1.5_out/dataset/ from EVERY round-0 and round-1 folder, not
#    just the re-run ones -- the dataset is cumulative, and the re-runs simply
#    add the structures that were missing from round1_out/dataset.
#
# One extra category compared to round 0/1: a folder that stopped on the
# walltime signal ends cleanly without reaching EDIFF, which the old audit
# would have filed under "hit NELM". Those are listed separately as
# round1.5_3_stopped_at_wall.txt -- they have WAVECAR/CHGCAR in place, so
# re-running them continues where they left off.
#
# Submit from the project root; every path below is relative to it.
# ---------------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -d round1.5_code ] || { echo "no round1.5_code/ in $PWD -- submit from the project root"; exit 1; }
mkdir -p round1.5_out

# folders re-run in this round, for the audit
PATHFILE="round1.5_out/round1.5_1_rerun_paths.txt"
# every folder of every round so far, for the dataset build
COMBINED="round1.5_out/round1.5_1_combined_paths.txt"
R0_PATHS="round0_out/round0_1_vasp_job_paths.txt"
R1_PATHS="round1_out/round1_1_vasp_job_paths.txt"
# round 1's split is carried forward (it already carries round 0's)
PREV_MANIFEST="round1_out/dataset/split_manifest.csv"
R0_MANIFEST="round0_out/dataset/split_manifest.csv"
OUTDIR="round1.5_out/dataset"

OUT_MAXED="round1.5_out/round1.5_3_maxed_out_jobs.txt"
OUT_WALL="round1.5_out/round1.5_3_stopped_at_wall.txt"
OUT_FAILED="round1.5_out/round1.5_3_timed_out_or_cancelled.txt"
OUT_MISSING="round1.5_out/round1.5_3_never_ran.txt"
OUT_RESUBMIT="round1.5_out/round1.5_3_resubmit.txt"

: > "$OUT_MAXED"
: > "$OUT_WALL"
: > "$OUT_FAILED"
: > "$OUT_MISSING"

n_ok=0; n_maxed=0; n_wall=0; n_failed=0; n_missing=0

if [ -f "$PATHFILE" ]; then
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

        # converged is converged, whatever else is lying around
        if grep -q "Voluntary context switches" "$OUTCAR" \
           && grep -q "aborting loop because EDIFF is reached" "$OUTCAR"; then
            n_ok=$((n_ok+1)); continue
        fi

        # stopped by the walltime signal: restartable, WAVECAR/CHGCAR kept
        if [ -f "$DIR/.wall_stop" ]; then
            echo "$DIR" >> "$OUT_WALL"; n_wall=$((n_wall+1)); continue
        fi

        if [ -f "$LOG" ] && grep -qE "DUE TO TIME LIMIT|CANCELLED|BAD TERMINATION|slurmstepd: error" "$LOG"; then
            echo "$DIR" >> "$OUT_FAILED"; n_failed=$((n_failed+1)); continue
        fi
        if ! grep -q "Voluntary context switches" "$OUTCAR"; then
            echo "$DIR" >> "$OUT_FAILED"; n_failed=$((n_failed+1)); continue
        fi

        # ended cleanly, no walltime signal, no EDIFF -> genuinely hit NELM
        echo "$DIR" >> "$OUT_MAXED"; n_maxed=$((n_maxed+1))
    done < "$PATHFILE"
else
    echo "NOTE: $PATHFILE not found -- nothing was re-run, going straight to the build"
fi

cat "$OUT_MAXED" "$OUT_WALL" "$OUT_FAILED" "$OUT_MISSING" | sort -u > "$OUT_RESUBMIT"

echo "-------------- round-1.5 re-run audit --------------"
echo "converged now        : $n_ok"
echo "stopped at walltime  : $n_wall     -> $OUT_WALL (restartable)"
echo "hit NELM             : $n_maxed   -> $OUT_MAXED"
echo "died / cancelled     : $n_failed  -> $OUT_FAILED"
echo "never ran            : $n_missing -> $OUT_MISSING"
echo "still to resubmit    : $(grep -c . "$OUT_RESUBMIT") -> $OUT_RESUBMIT"
echo "----------------------------------------------------"
echo
echo "Re-running round1.5 picks the walltime-stopped folders up from their"
echo "WAVECAR, so a second pass costs less than the first."
echo

# --- combined path list: every folder of every round ----------------------
if [ ! -f "$R0_PATHS" ] || [ ! -f "$R1_PATHS" ]; then
    echo "ERROR: need both $R0_PATHS and $R1_PATHS to build the cumulative dataset"
    exit 1
fi
cat "$R0_PATHS" "$R1_PATHS" | grep -v '^[[:space:]]*$' \
    | awk '!seen[$0]++' > "$COMBINED"

echo "round0 paths : $(grep -c . "$R0_PATHS")"
echo "round1 paths : $(grep -c . "$R1_PATHS")"
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

# Carry the newest split forward, so test.xyz keeps its membership and RMSE
# stays comparable with round 0 and round 1. Round 1's manifest already
# carries round 0's labels; fall back to round 0's if round 1 never built.
CARRY=""
if [ -f "$PREV_MANIFEST" ]; then
    CARRY="--carry-split-from $PREV_MANIFEST"
    echo "carrying the train/test split from $PREV_MANIFEST"
elif [ -f "$R0_MANIFEST" ]; then
    CARRY="--carry-split-from $R0_MANIFEST"
    echo "WARNING: $PREV_MANIFEST missing; carrying $R0_MANIFEST instead"
else
    echo "WARNING: no previous manifest found -- the split will be recomputed"
    echo "         from scratch and earlier test sets will NOT be preserved"
fi

python round1.5_code/round1.5_5_build_dataset.py \
    --paths "$COMBINED" \
    --outdir "$OUTDIR" \
    --test-fraction 0.2 \
    --split-mode family \
    --family-eps 0.06 \
    $CARRY
