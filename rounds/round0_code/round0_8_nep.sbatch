#!/bin/bash
#SBATCH -J r0_nep
#SBATCH -A mat260051-gpu
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH -o round0_out/logs/z-nep-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Train NEP on the round-0 dataset.
# Runs after round0_34_anvil_check_and_build.sbatch (job name r0_check).
# Submit from the project root; paths below hang off $SLURM_SUBMIT_DIR.
# ---------------------------------------------------------------------------

NEP_BIN=/anvil/projects/x-mat260051/Kad/NEP_GPUMD/src/nep
NEP_IN="$SLURM_SUBMIT_DIR/round0_code/round0_nep-1.in"
DATADIR="$SLURM_SUBMIT_DIR/round0_out/dataset"

module purge
module load modtree/gpu

[ -x "$NEP_BIN" ] || { echo "nep not executable: $NEP_BIN"; exit 1; }
[ -f "$NEP_IN"  ] || { echo "missing nep.in template: $NEP_IN"; exit 1; }
[ -d "$DATADIR" ] || { echo "missing $DATADIR"; exit 1; }

# nep reads nep.in / train.xyz / test.xyz from the CURRENT directory and takes
# no path arguments, so the job must run inside the dataset directory.
cd "$DATADIR" || exit 1
cp "$NEP_IN" nep.in

for f in nep.in train.xyz test.xyz; do
    [ -f "$f" ] || { echo "missing $f in $DATADIR"; exit 1; }
done

echo "=== NEP training in $DATADIR ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "train: $(grep -c 'Lattice=' train.xyz) structures"
echo "test : $(grep -c 'Lattice=' test.xyz) structures"
echo

time "$NEP_BIN"
rc=$?

[ -f nep.txt ] || { echo "FAILED: no nep.txt produced"; exit 1; }

# nep.txt is the potential itself. Archive it under a round-tagged name or the
# next round's run overwrites it.
cp nep.txt  "$SLURM_SUBMIT_DIR/round0_out/round0_nep.txt"
cp loss.out "$SLURM_SUBMIT_DIR/round0_out/round0_loss-1.out"

echo
echo "=== final loss line (gen | RMSE_E RMSE_F RMSE_V train | test) ==="
tail -1 loss.out
echo
echo "wrote round0_out/round0_nep.txt and round0_out/round0_loss-1.out"
exit $rc
