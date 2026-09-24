#!/bin/bash
#SBATCH -J r15_nep
#SBATCH -A mat260051-gpu
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH -o round1.5_out/logs/z-nep-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Train the round-1.5 NEP on the CUMULATIVE dataset -- every round-0 and
# round-1 folder that converges, including the ones round 1.5 re-ran.
# Runs after round1.5_34_anvil_check_and_build.sbatch (job name r15_check).
# Submit from the project root; paths below hang off $SLURM_SUBMIT_DIR.
#
# Continues from round-1's nep.restart ONLY if it is verifiably compatible:
#   - round1_out/dataset/nep.txt is identical to round1_out/round1_nep.txt
#     (i.e. both came from the same round-1 run), and
#   - the model-shape keywords in round-1's nep.in match round-1.5's nep.in.
# A mismatched nep.restart makes nep segfault within ~1 s (it reads a
# parameter vector of the wrong length). If restart mode still fails early,
# the job cleans up and retries from scratch. Results are archived only if
# nep genuinely succeeded.
# ---------------------------------------------------------------------------

NEP_BIN=/anvil/projects/x-mat260051/Kad/NEP_GPUMD/src/nep
NEP_IN="$SLURM_SUBMIT_DIR/round1.5_code/round1.5_nep-1.in"
DATADIR="$SLURM_SUBMIT_DIR/round1.5_out/dataset"
PREV_DIR="$SLURM_SUBMIT_DIR/round1_out/dataset"
PREV_NEP="$SLURM_SUBMIT_DIR/round1_out/round1_nep.txt"
PREV_RESTART="$PREV_DIR/nep.restart"
EARLY_FAIL_SEC=300   # restart-mode crash within this window -> retry from scratch

module purge
module load modtree/gpu

[ -x "$NEP_BIN" ] || { echo "nep not executable: $NEP_BIN"; exit 1; }
[ -f "$NEP_IN"  ] || { echo "missing nep.in template: $NEP_IN"; exit 1; }
[ -d "$DATADIR" ] || { echo "missing $DATADIR -- did r15_check succeed?"; exit 1; }
mkdir -p "$SLURM_SUBMIT_DIR/round1.5_out"

# Remove round-1.5 archives from earlier failed attempts, so a crash can
# never leave a stale file that looks like a round-1.5 result.
rm -f "$SLURM_SUBMIT_DIR/round1.5_out/round1.5_nep.txt" \
      "$SLURM_SUBMIT_DIR/round1.5_out/round1.5_nep.restart" \
      "$SLURM_SUBMIT_DIR/round1.5_out/round1.5_loss-1.out"

# Keywords that determine the length/layout of the NEP parameter vector.
arch() {
    sed 's/#.*//' "$1" \
    | awk 'NF && $1 ~ /^(version|model_type|type|cutoff|n_max|basis_size|l_max|neuron|zbl|use_typewise_cutoff|use_typewise_cutoff_zbl)$/ {$1=$1; print}' \
    | sort
}

clean_outputs() {
    rm -f nep.txt nep.restart loss.out energy_*.out force_*.out \
          virial_*.out stress_*.out polarizability_*.out dipole_*.out \
          descriptor.out .run_start
}

cd "$DATADIR" || exit 1
clean_outputs
cp "$NEP_IN" nep.in
# Continuation of our own model uses nep.restart natively; fine_tune is for
# foundation models (e.g. NEP89) and is not used here.
sed -i 's/^[[:space:]]*fine_tune/#fine_tune/' nep.in

for f in nep.in train.xyz test.xyz; do
    [ -f "$f" ] || { echo "missing $f in $DATADIR"; exit 1; }
done

# --- decide whether round-0's restart is safe to use ------------------------
MODE=scratch
REASON=""
if [ ! -f "$PREV_NEP" ]; then
    REASON="$PREV_NEP is missing"
elif [ ! -f "$PREV_RESTART" ]; then
    REASON="no nep.restart in $PREV_DIR"
elif [ -f "$PREV_DIR/nep.txt" ] && ! cmp -s "$PREV_NEP" "$PREV_DIR/nep.txt"; then
    REASON="$PREV_DIR/nep.txt != $PREV_NEP, so that nep.restart is from a different run"
elif [ ! -f "$PREV_DIR/nep.in" ]; then
    MODE=restart
    echo "NOTE: $PREV_DIR/nep.in not found; cannot verify architecture, trying restart anyway."
elif ! diff -q <(arch "$PREV_DIR/nep.in") <(arch nep.in) >/dev/null; then
    REASON="model architecture differs between round-1 and round-1.5 nep.in"
    echo "--- architecture diff (round1 < > round1.5) ---"
    diff <(arch "$PREV_DIR/nep.in") <(arch nep.in)
    echo "---------------------------------------------"
else
    MODE=restart
fi

if [ "$MODE" = restart ]; then
    cp "$PREV_RESTART" nep.restart
    echo "continuing from round-1 state: $PREV_RESTART"
else
    echo "WARNING: not using round-1 restart: $REASON"
    echo "         Training FROM SCRATCH on the cumulative dataset."
fi

echo
echo "=== NEP training in $DATADIR ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "train: $(grep -c 'Lattice=' train.xyz) structures  (round0 + round1 + re-runs)"
echo "test : $(grep -c 'Lattice=' test.xyz) structures"

train() {
    echo
    echo ">>> mode: $MODE"
    touch .run_start
    local t0=$SECONDS
    { time "$NEP_BIN"; }
    rc=$?
    elapsed=$(( SECONDS - t0 ))
    echo ">>> nep exit code $rc after ${elapsed}s"
}

succeeded() {
    [ "$rc" -eq 0 ] && [ -s loss.out ] && [ -f nep.txt ] && [ nep.txt -nt .run_start ]
}

train
if ! succeeded && [ "$MODE" = restart ] && [ "$elapsed" -lt "$EARLY_FAIL_SEC" ]; then
    echo "Restart-mode run failed early (rc=$rc); round-1 restart is incompatible."
    echo "Cleaning up and retrying FROM SCRATCH."
    clean_outputs
    MODE=scratch
    train
fi

if ! succeeded; then
    echo
    echo "FAILED: nep did not complete (rc=$rc, mode=$MODE). Nothing archived."
    [ -f loss.out ] && { echo "last loss line:"; tail -1 loss.out; }
    exit 1
fi

# --- archive (only reached on genuine success) ------------------------------
cp nep.txt  "$SLURM_SUBMIT_DIR/round1.5_out/round1.5_nep.txt"
cp loss.out "$SLURM_SUBMIT_DIR/round1.5_out/round1.5_loss-1.out"
cp nep.in   "$SLURM_SUBMIT_DIR/round1.5_out/round1.5_nep-1.used.in"   # lets round2 verify architecture
if [ -f nep.restart ]; then
    cp nep.restart "$SLURM_SUBMIT_DIR/round1.5_out/round1.5_nep.restart"
    echo "wrote round1.5_out/round1.5_nep.restart"
else
    echo "NOTE: no nep.restart produced; round2 will train from scratch"
fi

echo
echo "=== final loss line (gen | RMSE_E RMSE_F RMSE_V train | test) ==="
tail -1 loss.out
echo
echo "wrote round1.5_out/round1.5_nep.txt and round1.5_out/round1.5_loss-1.out (mode: $MODE)"
rm -f .run_start
exit 0
