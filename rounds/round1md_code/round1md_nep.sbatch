#!/bin/bash
#SBATCH -J r1md_nep
#SBATCH -A mat260051-gpu
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -o round1md_out/logs/z-nep-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Train the round-1md N-Al-Si-Ar NEP on round1md_out/dataset
# (round-1wL dataset + the MD-sampled amorphous and collision frames).
#
# Default: CONTINUE round 1wL's run from round1wL_out/round1wL_nep.restart,
# but only if round 1wL's nep.in has the same architecture (checked the way
# round1.5_nep.sbatch does it). A restart that still dies in the first
# minutes is treated as incompatible and the job retrains from scratch.
#     FRESH=1   train from scratch instead (set `generation` back to 200000)
#     RESUME=1  continue THIS round's own run after a walltime cut-off
#     sbatch --export=ALL,FRESH=1 round1md_code/round1md_nep.sbatch
#
# Submit from the project root.
# ---------------------------------------------------------------------------

NEP_BIN="${NEP_BIN:-/anvil/projects/x-mat260051/Kad/NEP_GPUMD/src/nep}"
ROOT="$SLURM_SUBMIT_DIR"
NEP_IN="$ROOT/round1md_code/round1md_nep-1.in"
DATADIR="$ROOT/round1md_out/dataset"
ARCHIVE="$ROOT/round1md_out"
PREV_RESTART="$ROOT/round1wL_out/round1wL_nep.restart"
PREV_IN="$ROOT/round1wL_out/round1wL_nep-1.used.in"
FRESH="${FRESH:-0}"
RESUME="${RESUME:-0}"
EARLY_FAIL_SEC=300

module purge
module load modtree/gpu

[ -x "$NEP_BIN" ] || { echo "nep not executable: $NEP_BIN"; exit 1; }
[ -f "$NEP_IN"  ] || { echo "missing nep.in template: $NEP_IN"; exit 1; }
[ -d "$DATADIR" ] || { echo "missing $DATADIR -- did r1md_build succeed?"; exit 1; }

# Keywords that fix the length/layout of the NEP parameter vector.
arch() {
    sed 's/#.*//' "$1" \
    | awk 'NF && $1 ~ /^(version|model_type|type|cutoff|n_max|basis_size|l_max|neuron|zbl|use_typewise_cutoff|use_typewise_cutoff_zbl)$/ {$1=$1; print}' \
    | sort
}

clean_outputs() {
    rm -f nep.txt nep.restart loss.out energy_*.out force_*.out \
          virial_*.out stress_*.out descriptor.out .run_start
}

cd "$DATADIR" || exit 1

if [ "$RESUME" = 1 ] && [ -f nep.restart ] && [ -f nep.in ] && cmp -s nep.in "$NEP_IN"; then
    MODE="resume"
else
    [ "$RESUME" = 1 ] && echo "RESUME=1 requested but no matching nep.restart/nep.in here"
    clean_outputs
    cp "$NEP_IN" nep.in
    MODE=scratch
    REASON=""
    if [ "$FRESH" = 1 ]; then
        REASON="FRESH=1"
    elif [ ! -f "$PREV_RESTART" ]; then
        REASON="$PREV_RESTART not found"
    elif [ ! -f "$PREV_IN" ]; then
        REASON="$PREV_IN not found, cannot verify the architecture"
    elif ! diff -q <(arch "$PREV_IN") <(arch nep.in) > /dev/null; then
        REASON="architecture differs from round 1wL"
        diff <(arch "$PREV_IN") <(arch nep.in)
    else
        MODE=restart
        cp "$PREV_RESTART" nep.restart
        echo "continuing from round 1wL: $PREV_RESTART"
    fi
    [ "$MODE" = scratch ] && echo "training FROM SCRATCH: $REASON"
fi

for f in nep.in train.xyz test.xyz; do
    [ -f "$f" ] || { echo "missing $f in $DATADIR"; exit 1; }
done
grep -qE '^[[:space:]]*type[[:space:]]+4[[:space:]]+N[[:space:]]+Al[[:space:]]+Si[[:space:]]+Ar' nep.in \
    || { echo "nep.in is not 'type 4 N Al Si Ar'"; exit 1; }

# the old archive goes only now, so a failed setup above keeps the last good one
rm -f "$ARCHIVE/round1md_nep.txt" "$ARCHIVE/round1md_nep.restart" \
      "$ARCHIVE/round1md_loss-1.out"

echo
echo "=== NEP training in $DATADIR ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "train: $(grep -c 'Lattice=' train.xyz) structures  (round 1wL + round 1md)"
echo "test : $(grep -c 'Lattice=' test.xyz) structures"
[ -f report.txt ] && head -2 report.txt

train() {
    echo
    echo ">>> mode: $MODE"
    touch .run_start
    local t0=$SECONDS
    { time "$NEP_BIN"; }
    rc=$?
    elapsed=$((SECONDS - t0))
    echo ">>> nep exit code $rc after ${elapsed}s"
}

succeeded() {
    [ "$rc" -eq 0 ] && [ -s loss.out ] && [ -f nep.txt ] && [ nep.txt -nt .run_start ]
}

train
if ! succeeded && [ "$MODE" = restart ] && [ "$elapsed" -lt "$EARLY_FAIL_SEC" ]; then
    echo "Restart-mode run failed early (rc=$rc): round 1wL's nep.restart is not"
    echo "compatible. Retrying FROM SCRATCH."
    clean_outputs
    MODE=scratch
    train
fi

if ! succeeded; then
    echo
    echo "FAILED or cut off (rc=$rc, mode=$MODE). Nothing archived."
    [ -f loss.out ] && { echo "last loss line:"; tail -1 loss.out; }
    [ -f nep.restart ] && echo "nep.restart kept -- resubmit with RESUME=1 to continue."
    exit 1
fi

cp nep.txt  "$ARCHIVE/round1md_nep.txt"
cp loss.out "$ARCHIVE/round1md_loss-1.out"
cp nep.in   "$ARCHIVE/round1md_nep-1.used.in"   # lets the next round verify architecture
[ -f nep.restart ] && cp nep.restart "$ARCHIVE/round1md_nep.restart"

echo
echo "=== final loss line (gen | RMSE_E RMSE_F RMSE_V train | test) ==="
tail -1 loss.out
echo "wrote round1md_out/round1md_nep.txt (types: N Al Si Ar; mode: $MODE)"
rm -f .run_start
exit 0
