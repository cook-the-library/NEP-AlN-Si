#!/bin/bash
#SBATCH -J r1md_build
#SBATCH -A mat260051
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH -o round1md_out/logs/z-build-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Round 1md, step 5: from VASP folders to the NEP training set.
#
#   1. audit the round-1md VASP folders and write the resubmit list
#   2. round1md_out/dataset_new   the NEW frames only, built with the
#      round-agnostic round1.5_5_build_dataset.py: SCF screening, virials,
#      family-based train/test split stratified by bucket. Nothing to carry:
#      every folder is new, and none of them is a near-copy of an old one.
#   3. round1md_out/dataset_new/nep_vs_dft.txt  error of the NEP that drove
#      the MD on the frames it produced (round1md_6_nep_vs_dft.py)
#   4. round1md_out/dataset       round-1wL dataset + the new frames, through
#      round1wL_5_build_dataset.py with the same rejection rules as round 1wL.
#      Each source keeps its own split, so round 1wL's test.xyz is carried
#      unchanged and RMSEs stay comparable with round 1wL.
#
# Settings pass through the environment, as in round 1wL:
#     PREV_DIR=round1wL_out/dataset EMIN=-8.5 sbatch round1md_code/round1md_5_build_dataset.sbatch
#
# Submit from the project root.
# ---------------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -d round1md_code ] || { echo "no round1md_code/ in $PWD -- submit from the project root"; exit 1; }

PATHFILE="round1md_out/round1md_3_vasp_job_paths.txt"
PREV_DIR="${PREV_DIR:-round1wL_out/dataset}"
NEW_DIR="round1md_out/dataset_new"
OUTDIR="round1md_out/dataset"
NEP_IN="round1md_code/round1md_nep-1.in"
EMIN="${EMIN:--8.5}"
EMAX_COND="${EMAX_COND:-0.0}"
EMAX_GAS="${EMAX_GAS:-none}"
FMAX="${FMAX:-500}"

for f in "$PATHFILE" "$PREV_DIR/train.xyz" "$PREV_DIR/test.xyz" "$NEP_IN" \
         round1.5_code/round1.5_5_build_dataset.py \
         round1wL_code/round1wL_5_build_dataset.py; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
done

# --- 1. audit ---------------------------------------------------------------
OUT_RESUBMIT="round1md_out/round1md_5_resubmit.txt"
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
echo "--------------- round-1md VASP audit ---------------"
echo "converged       : $n_ok"
echo "not converged   : $n_bad  -> $OUT_RESUBMIT"
echo "  (re-run them: bash round1md_code/round1md_generateVASP_trainNEP.sh;"
echo "   converged folders are skipped, cut-off ones continue from WAVECAR)"
echo "-----------------------------------------------------"

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
python round1md_code/round1md_6_nep_vs_dft.py \
    --nep-xyz round1md_out/vasp/selected_nep.extxyz \
    --dft "$NEW_DIR/train.xyz" "$NEW_DIR/test.xyz" \
    --out "$NEW_DIR/nep_vs_dft.txt" \
    || echo "WARNING: NEP-vs-DFT report failed (the dataset is not affected)"

# --- 4. merge with round 1wL ------------------------------------------------------
python round1wL_code/round1wL_5_build_dataset.py \
    --source r1wL "$PREV_DIR/train.xyz" "$PREV_DIR/test.xyz" \
    --source r1md "$NEW_DIR/train.xyz"  "$NEW_DIR/test.xyz" \
    --nep-in "$NEP_IN" --outdir "$OUTDIR" \
    --emin="$EMIN" --emax-condensed="$EMAX_COND" \
    --emax-gas="$EMAX_GAS" --fmax="$FMAX" || exit 1
