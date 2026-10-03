#!/bin/bash
#SBATCH -J r2_nep
#SBATCH -A mat260051-gpu
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -o round2_out/logs/z-nep-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Train the round-2 N-Al-Si-Ar NEP on round2_out/dataset in two stages:
#
#   STAGE=1  round2_nep-1.in  lambda_e 0.1 / lambda_f 50, 200k generations:
#            forces first. Continues round 1wL's nep.restart if round 1wL's
#            nep.in has the same architecture (as round1.5_nep.sbatch checks);
#            a restart that dies in the first minutes is retried from scratch.
#   STAGE=2  round2_nep-2.in  lambda_e 10 / lambda_f 1, 200k generations:
#            then energy. Always continues stage 1's nep.restart.
#
# round2_generateVASP_trainNEP.sh submits both, stage 2 afterok stage 1:
#     sbatch --export=ALL,STAGE=1 round2_code/round2_nep.sbatch
#     sbatch --export=ALL,STAGE=2 --dependency=afterok:<id> round2_code/round2_nep.sbatch
# Options:
#     FRESH=1   stage 1 from scratch instead of round 1wL's restart
#     RESUME=1  continue THIS stage's own run after a walltime cut-off
#               (then submit stage 2 by hand once it finishes)
#
# Archive in round2_out/:
#   round2_nep-stage1.{txt,restart}, round2_loss-stage1.out   after stage 1
#   round2_nep.txt, round2_nep.restart, round2_loss-stage2.out,
#   round2_nep-2.used.in                                      the final model
#
# Submit from the project root.
# ---------------------------------------------------------------------------

NEP_BIN="${NEP_BIN:-/anvil/projects/x-mat260051/Kad/NEP_GPUMD/src/nep}"
ROOT="$SLURM_SUBMIT_DIR"
STAGE="${STAGE:-1}"
DATADIR="$ROOT/round2_out/dataset"
ARCHIVE="$ROOT/round2_out"
PREV_RESTART="$ROOT/round1wL_out/round1wL_nep.restart"
PREV_IN="$ROOT/round1wL_out/round1wL_nep-1.used.in"
STAGE1_RESTART="$ARCHIVE/round2_nep-stage1.restart"
FRESH="${FRESH:-0}"
RESUME="${RESUME:-0}"
EARLY_FAIL_SEC=300

case "$STAGE" in
    1) NEP_IN="$ROOT/round2_code/round2_nep-1.in" ;;
    2) NEP_IN="$ROOT/round2_code/round2_nep-2.in" ;;
    *) echo "STAGE must be 1 or 2, got '$STAGE'"; exit 1 ;;
esac

module purge
module load modtree/gpu

[ -x "$NEP_BIN" ] || { echo "nep not executable: $NEP_BIN"; exit 1; }
[ -f "$NEP_IN"  ] || { echo "missing nep.in template: $NEP_IN"; exit 1; }
[ -d "$DATADIR" ] || { echo "missing $DATADIR -- did r2_build succeed?"; exit 1; }

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
    if [ "$STAGE" = 2 ]; then
        [ -f "$STAGE1_RESTART" ] || { echo "stage 2 needs $STAGE1_RESTART -- run stage 1 first"; exit 1; }
        cp "$STAGE1_RESTART" nep.restart
        MODE=restart
        echo "stage 2 continues stage 1: $STAGE1_RESTART"
    elif [ "$FRESH" = 1 ]; then
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
        echo "stage 1 continues round 1wL: $PREV_RESTART"
    fi
    [ "$MODE" = scratch ] && echo "training FROM SCRATCH: $REASON"
fi

for f in nep.in train.xyz test.xyz; do
    [ -f "$f" ] || { echo "missing $f in $DATADIR"; exit 1; }
done
grep -qE '^[[:space:]]*type[[:space:]]+4[[:space:]]+N[[:space:]]+Al[[:space:]]+Si[[:space:]]+Ar' nep.in \
    || { echo "nep.in is not 'type 4 N Al Si Ar'"; exit 1; }

echo
echo "=== NEP training, stage $STAGE, in $DATADIR ==="
grep -E '^[[:space:]]*lambda_[ef]' nep.in
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "train: $(grep -c 'Lattice=' train.xyz) structures  (round 1wL filtered + round 2)"
echo "test : $(grep -c 'Lattice=' test.xyz) structures"

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
if ! succeeded && [ "$MODE" = restart ] && [ "$STAGE" = 1 ] \
   && [ "$elapsed" -lt "$EARLY_FAIL_SEC" ]; then
    echo "Restart-mode run failed early (rc=$rc): round 1wL's nep.restart is not"
    echo "compatible. Retrying FROM SCRATCH."
    clean_outputs
    MODE=scratch
    train
fi

if ! succeeded; then
    echo
    echo "FAILED or cut off (rc=$rc, stage $STAGE, mode=$MODE). Nothing archived."
    [ -f loss.out ] && { echo "last loss line:"; tail -1 loss.out; }
    [ -f nep.restart ] && echo "nep.restart kept -- resubmit with RESUME=1 STAGE=$STAGE."
    exit 1
fi

if [ "$STAGE" = 1 ]; then
    cp nep.txt     "$ARCHIVE/round2_nep-stage1.txt"
    cp nep.restart "$STAGE1_RESTART"
    cp loss.out    "$ARCHIVE/round2_loss-stage1.out"
    cp nep.in      "$ARCHIVE/round2_nep-1.used.in"
    echo "wrote round2_out/round2_nep-stage1.txt (mode: $MODE); stage 2 continues it"
else
    cp nep.txt  "$ARCHIVE/round2_nep.txt"
    cp loss.out "$ARCHIVE/round2_loss-stage2.out"
    cp nep.in   "$ARCHIVE/round2_nep-2.used.in"   # lets the next round verify architecture
    [ -f nep.restart ] && cp nep.restart "$ARCHIVE/round2_nep.restart"
    echo "wrote round2_out/round2_nep.txt (types: N Al Si Ar)"
fi
echo
echo "=== final loss line (gen | RMSE_E RMSE_F RMSE_V train | test) ==="
tail -1 loss.out
rm -f .run_start
exit 0
