#!/bin/bash
#SBATCH -J r1_vasp
#SBATCH -A mat260051
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time=12:00:00
# NOTE: in round0, 100% of the interface bucket and roughly a third of
# adsorption/slab_AlN needed resubmission, all timing/cancelling out under
# this same 12h/16-core allocation, with nothing hitting NELM. If round1
# regenerates similarly large/vacuum-heavy structures, consider raising
# --time (or --ntasks) before submitting rather than resubmitting into the
# same limit again. Left unchanged here since that's a compute-budget call.
#SBATCH --array=0-999%100
#SBATCH -o round1_out/logs/z-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Round-1 VASP single-points, one array task per structure folder.
# Adapted from round0_2_anvil_subVASP.sbatch -- same logic, round1 paths.
#
# Array range MUST match the number of lines in the path list.
#   wc -l < round1_out/round1_1_vasp_job_paths.txt   ->   N   =>   --array=0-(N-1)
#
# SLURM caps array indices at MaxArraySize-1 (1000 on Anvil), so a list longer
# than that must be submitted as several chunked arrays. Each chunk passes
# CHUNK_OFFSET via --export, and the line number is TASK_ID + OFFSET + 1.
# round1_generateVASP_trainNEP.sbatch does the chunking and overrides --array
# line at submit time, so the number above is only a fallback for manual
# submission -- keep it roughly right, but don't rely on it.
#
# Submit from the project root. round1_out/logs must exist BEFORE submission:
# sbatch opens the output file before this script runs.
# round1_code/round1_generateVASP_trainNEP.sbatch does both.
# ---------------------------------------------------------------------------

PATHFILE="round1_out/round1_1_vasp_job_paths.txt"

module --force purge
module load gcc/11.2.0 openmpi/4.1.6
module load vasp/5.4.4.pl2
module list

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -f "$PATHFILE" ] || { echo "no $PATHFILE in $PWD -- submit from the project root"; exit 1; }

# --- pick this task's directory ------------------------------------------
# CHUNK_OFFSET is exported by the chain script; 0 when submitted by hand.
OFFSET="${CHUNK_OFFSET:-0}"
LINE=$((SLURM_ARRAY_TASK_ID + OFFSET + 1))
DIR=$(sed -n "${LINE}p" "$PATHFILE")

if [ -z "$DIR" ]; then
    # Expected on the final chunk, whose array range is padded out to a
    # round size. Exit 0 so the chunk is not reported as failed.
    echo "No entry at line $LINE of $PATHFILE (offset $OFFSET) -- past the"
    echo "end of the list. Nothing to do."
    exit 0
fi
if [ ! -d "$DIR" ]; then
    echo "Directory not found: $DIR"
    exit 1
fi

cd "$DIR" || exit 1
echo "=== task $SLURM_ARRAY_TASK_ID + offset $OFFSET = line $LINE  ->  $DIR ==="

# --- skip jobs that already finished and converged ------------------------
if [ -f OUTCAR ] \
   && grep -q "Voluntary context switches" OUTCAR \
   && grep -q "aborting loop because EDIFF is reached" OUTCAR; then
    echo "Already converged. Skipping."
    exit 0
fi

if [ -f OUTCAR ]; then
    echo "Previous run incomplete or unconverged. Re-running."
    rm -f OUTCAR OSZICAR CONTCAR vasprun.xml WAVECAR CHGCAR CHG
fi

# --- inputs are already in place from generate_round1.py ------------------
# POSCAR KPOINTS POTCAR INCAR were written by the generator.
# Do NOT copy an INCAR template over them: the INCAR is per-structure
# (dipole correction and spin differ by bucket).
for f in POSCAR KPOINTS POTCAR INCAR; do
    if [ ! -f "$f" ]; then
        echo "Missing $f in $DIR"
        exit 1
    fi
done

# --- run ------------------------------------------------------------------
mpirun -n "$SLURM_NTASKS" vasp_std &> VASP_job.out
rc=$?

if grep -q "aborting loop because EDIFF is reached" OUTCAR 2>/dev/null; then
    echo "Converged."
else
    echo "WARNING: SCF did not reach EDIFF in $DIR"
fi

exit $rc
