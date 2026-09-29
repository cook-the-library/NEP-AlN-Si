#!/bin/bash
#SBATCH -J r1wL_nep
#SBATCH -A mat260051-gpu
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -o round1wL_out/logs/z-nep-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Train the round-1wL N-Al-Si-Ar NEP on round1wL_out/dataset
# (round-1.7 cumulative data + legacy B+C, abnormal energies rejected).
#
# Always a FRESH start: no earlier round's nep.restart is used, and any
# nep.restart left in round1wL_out/dataset from an earlier attempt is deleted.
#
# The one exception is RESUME=1, which continues THIS round's own run after
# a walltime cut-off (same nep.in, same dataset):
#     sbatch --export=ALL,RESUME=1 round1wL_code/round1wL_nep.sbatch
#
# Submit from the project root.
# ---------------------------------------------------------------------------

NEP_BIN="${NEP_BIN:-/anvil/projects/x-mat260051/Kad/NEP_GPUMD/src/nep}"
ROOT="$SLURM_SUBMIT_DIR"
NEP_IN="$ROOT/round1wL_code/round1wL_nep-1.in"
DATADIR="$ROOT/round1wL_out/dataset"
ARCHIVE="$ROOT/round1wL_out"
RESUME="${RESUME:-0}"

module purge
module load modtree/gpu

[ -x "$NEP_BIN" ] || { echo "nep not executable: $NEP_BIN"; exit 1; }
[ -f "$NEP_IN"  ] || { echo "missing nep.in template: $NEP_IN"; exit 1; }
[ -d "$DATADIR" ] || { echo "missing $DATADIR -- did r1wL_build succeed?"; exit 1; }

cd "$DATADIR" || exit 1

if [ "$RESUME" = 1 ] && [ -f nep.restart ] && [ -f nep.in ] \
   && cmp -s nep.in "$NEP_IN"; then
    MODE="resume ($DATADIR/nep.restart)"
else
    [ "$RESUME" = 1 ] && echo "RESUME=1 requested but no matching nep.restart/nep.in here -- starting fresh"
    rm -f nep.txt nep.restart loss.out energy_*.out force_*.out \
          virial_*.out stress_*.out descriptor.out
    cp "$NEP_IN" nep.in
    MODE="fresh start"
fi
[ "$MODE" = "fresh start" ] && [ -f nep.restart ] \
    && { echo "ERROR: nep.restart still present for a fresh start"; exit 1; }

# the old archive goes only now, so a failed setup above keeps the last good one
rm -f "$ARCHIVE/round1wL_nep.txt" "$ARCHIVE/round1wL_nep.restart" \
      "$ARCHIVE/round1wL_loss-1.out"

for f in nep.in train.xyz test.xyz; do
    [ -f "$f" ] || { echo "missing $f in $DATADIR"; exit 1; }
done
grep -qE '^[[:space:]]*type[[:space:]]+4[[:space:]]+N[[:space:]]+Al[[:space:]]+Si[[:space:]]+Ar' nep.in \
    || { echo "nep.in is not 'type 4 N Al Si Ar'"; exit 1; }

echo
echo "=== NEP training in $DATADIR (mode: $MODE) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "train: $(grep -c 'Lattice=' train.xyz) structures  (round0 + 1 + 1.7 + legacy B+C)"
echo "test : $(grep -c 'Lattice=' test.xyz) structures"
[ -f report.txt ] && head -2 report.txt

touch .run_start
{ time "$NEP_BIN"; }
rc=$?

if ! { [ "$rc" -eq 0 ] && [ -s loss.out ] && [ -f nep.txt ] && [ nep.txt -nt .run_start ]; }; then
    echo
    echo "FAILED or cut off (rc=$rc). Nothing archived."
    [ -f loss.out ] && { echo "last loss line:"; tail -1 loss.out; }
    [ -f nep.restart ] && echo "nep.restart kept -- resubmit with RESUME=1 to continue."
    exit 1
fi

cp nep.txt  "$ARCHIVE/round1wL_nep.txt"
cp loss.out "$ARCHIVE/round1wL_loss-1.out"
cp nep.in   "$ARCHIVE/round1wL_nep-1.used.in"   # lets the next round verify architecture
[ -f nep.restart ] && cp nep.restart "$ARCHIVE/round1wL_nep.restart"

echo
echo "=== final loss line (gen | RMSE_E RMSE_F RMSE_V train | test) ==="
tail -1 loss.out
echo "wrote round1wL_out/round1wL_nep.txt (types: N Al Si Ar; started: $MODE)"
rm -f .run_start
exit 0
