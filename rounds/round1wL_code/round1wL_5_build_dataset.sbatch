#!/bin/bash
#SBATCH -J r1wL_build
#SBATCH -A mat260051
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH -o round1wL_out/logs/z-build-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Build round1wL_out/dataset/ = round-1.7 cumulative dataset + legacy B+C,
# with abnormal labels rejected (rules in round1wL_5_build_dataset.py).
# No VASP and no re-split: each source keeps its own family-based split.
#
# Round-1.7 buckets dropped whole are set in ROUND_DROP below. Everything
# else can be overridden from the submitting shell; DROP ADDS to ROUND_DROP:
#     DROP="r1.7:interface" EMAX_COND=none sbatch round1wL_code/round1wL_5_build_dataset.sbatch
#
# Submit from the project root; every path below is relative to it.
# ---------------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -d round1wL_code ] || { echo "no round1wL_code/ in $PWD -- submit from the project root"; exit 1; }

ROUND_DIR="${ROUND_DIR:-round1.7_out/dataset}"
LEGACY_DIR="${LEGACY_DIR:-roundL_out/dataset_legacy/BC}"
OUTDIR="round1wL_out/dataset"
NEP_IN="round1wL_code/round1wL_nep-1.in"

EMIN="${EMIN:--8.5}"            # eV/atom floor (default -8.5)
EMAX_COND="${EMAX_COND:-0.0}"   # eV/atom ceiling, condensed frames; none = off
EMAX_GAS="${EMAX_GAS:-none}"    # eV/atom ceiling, gas-phase frames; none = off
FMAX="${FMAX:-500}"             # eV/A; none = off
# whole round-1.7 buckets left out of round 1wL (the legacy frames of the
# same buckets are kept). SOURCE:BUCKET:REASON, reason without spaces.
ROUND_DROP="r1.7:disordered:too_low_E_per_atom
            r1.7:slab_AlN:too_low_E_per_atom
            r1.7:slab_Si:too_low_E_per_atom"
DROP="${DROP:-}"                # extra SOURCE:BUCKET[:REASON], space-separated

for f in "$ROUND_DIR/train.xyz" "$ROUND_DIR/test.xyz" \
         "$LEGACY_DIR/train.xyz" "$LEGACY_DIR/test.xyz" "$NEP_IN"; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
done

module load conda
module use "$HOME/privatemodules"
module load conda-env/nepdata
PY=$(command -v python || command -v python3)

ARGS=(--source r1.7     "$ROUND_DIR/train.xyz"  "$ROUND_DIR/test.xyz"
      --source legacyBC "$LEGACY_DIR/train.xyz" "$LEGACY_DIR/test.xyz"
      --nep-in "$NEP_IN" --outdir "$OUTDIR"
      --emin="$EMIN" --emax-condensed="$EMAX_COND"
      --emax-gas="$EMAX_GAS" --fmax="$FMAX")
for d in $ROUND_DROP $DROP; do ARGS+=(--drop "$d"); done

"$PY" round1wL_code/round1wL_5_build_dataset.py "${ARGS[@]}" || exit 1
