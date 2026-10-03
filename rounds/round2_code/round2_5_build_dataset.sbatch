#!/bin/bash
#SBATCH -J r2_build
#SBATCH -A mat260051
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH -o round2_out/logs/z-build-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Round 2, step 5: from VASP folders to the NEP training set.
#
#   1. audit every round-2 VASP folder (MD-sampled + thermally rattled) and
#      write the resubmit list
#   2. round2_out/dataset_new   the NEW frames only, built with the
#      round-agnostic round1.5_5_build_dataset.py: SCF screening, virials,
#      family-based split stratified by bucket, so new structures go to test
#      too (20 %)
#   3. round2_out/dataset_new/nep_vs_dft.txt  error of the NEP that drove the
#      MD on the frames it produced (round2_6_nep_vs_dft.py)
#   4. round2_out/dataset_prev_filtered  the round-1wL dataset without the
#      frames whose geometry is broken (sheared interface film, overlapping
#      atoms; round2_5_contact_filter.py)
#   5. round2_out/dataset       filtered round-1wL + new frames, through
#      round1wL_5_build_dataset.py with round 1wL's rejection rules. Each
#      source keeps its own split.
#
# Settings pass through the environment:
#     PREV_DIR=round1wL_out/dataset MIN_CONTACT=0.65 EMIN=-8.5 \
#         sbatch round2_code/round2_5_build_dataset.sbatch
# The contact filter removes the sheared round 0/1 interface frames by their
# geometry and keeps interface frames that are fine. To drop a whole bucket of
# the old data anyway: DROP="r1wL:interface" (SOURCE:BUCKET[:REASON], spaces
# between several).
#
# Submit from the project root.
# ---------------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -d round2_code ] || { echo "no round2_code/ in $PWD -- submit from the project root"; exit 1; }

PATHFILE="round2_out/round2_4_vasp_job_paths.txt"
PREV_DIR="${PREV_DIR:-round1wL_out/dataset}"
NEW_DIR="round2_out/dataset_new"
FILT_DIR="round2_out/dataset_prev_filtered"
OUTDIR="round2_out/dataset"
NEP_IN="round2_code/round2_nep-1.in"
MIN_CONTACT="${MIN_CONTACT:-0.65}"
EMIN="${EMIN:--8.5}"
EMAX_COND="${EMAX_COND:-0.0}"
EMAX_GAS="${EMAX_GAS:-none}"
FMAX="${FMAX:-500}"
DROP="${DROP:-}"                # extra SOURCE:BUCKET[:REASON] drops, e.g. "r1wL:interface"

for f in "$PATHFILE" "$PREV_DIR/train.xyz" "$PREV_DIR/test.xyz" "$NEP_IN" \
         round1.5_code/round1.5_5_build_dataset.py \
         round1wL_code/round1wL_5_build_dataset.py; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
done

# --- 1. audit ---------------------------------------------------------------
OUT_RESUBMIT="round2_out/round2_5_resubmit.txt"
: > "$OUT_RESUBMIT"
n_ok=0; n_bad=0
while IFS= read -r DIR; do
    [ -z "$DIR" ] && continue
    if [ -f "$DIR/OUTCAR" ] && grep -q "Voluntary context switches" "$DIR/OUTCAR" \
       && grep -q "aborting loop because EDIFF is reached" "$DIR/OUTCAR"; then
        n_ok=$((n_ok + 1))
    else
        echo "$DIR" >> "$OUT_RESUBMIT"; n_bad=$((n_bad + 1))
    fi
done < "$PATHFILE"
echo "--------------- round-2 VASP audit ---------------"
echo "converged       : $n_ok"
echo "not converged   : $n_bad  -> $OUT_RESUBMIT"
echo "  (re-run them: bash round2_code/round2_generateVASP_trainNEP.sh;"
echo "   converged folders are skipped, cut-off ones continue from WAVECAR)"
echo "---------------------------------------------------"

module load conda
module use "$HOME/privatemodules"
module load conda-env/nepdata

# --- 2. new frames ------------------------------------------------------------
python round1.5_code/round1.5_5_build_dataset.py \
    --paths "$PATHFILE" \
    --outdir "$NEW_DIR" \
    --test-fraction 0.2 \
    --split-mode family \
    --family-eps 0.06 || exit 1

# --- 3. NEP vs DFT on its own MD frames -----------------------------------------
python round2_code/round2_6_nep_vs_dft.py \
    --nep-xyz round2_out/vasp_md/selected_nep.extxyz \
    --dft "$NEW_DIR/train.xyz" "$NEW_DIR/test.xyz" \
    --out "$NEW_DIR/nep_vs_dft.txt" \
    || echo "WARNING: NEP-vs-DFT report failed (the dataset is not affected)"

# --- 4. drop broken geometries from the round-1wL data ----------------------------
python round2_code/round2_5_contact_filter.py \
    --train "$PREV_DIR/train.xyz" --test "$PREV_DIR/test.xyz" \
    --outdir "$FILT_DIR" --min-contact "$MIN_CONTACT" || exit 1

# --- 5. merge ---------------------------------------------------------------------
ARGS=(--source r1wL "$FILT_DIR/train.xyz" "$FILT_DIR/test.xyz"
      --source r2   "$NEW_DIR/train.xyz"  "$NEW_DIR/test.xyz"
      --nep-in "$NEP_IN" --outdir "$OUTDIR"
      --emin="$EMIN" --emax-condensed="$EMAX_COND"
      --emax-gas="$EMAX_GAS" --fmax="$FMAX")
for d in $DROP; do ARGS+=(--drop "$d"); done
python round1wL_code/round1wL_5_build_dataset.py "${ARGS[@]}" || exit 1
