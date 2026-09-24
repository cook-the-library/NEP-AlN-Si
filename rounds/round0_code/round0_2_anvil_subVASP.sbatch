#!/bin/bash
#SBATCH -J r0_vasp
#SBATCH -A mat260051
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --array=0-519%100 
#SBATCH -o round0_out/logs/z-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Round-0 VASP single-points, one array task per structure folder.
#
# Array range MUST match the number of lines in the path list.
#   wc -l < round0_out/round0_1_vasp_job_paths.txt   ->  520   =>   --array=0-519
# The %100 throttle caps concurrent tasks so 520 jobs do not hammer the
# shared filesystem at once.
#
# Submit from the project root. round0_out/logs must exist BEFORE submission:
# sbatch opens the output file before this script runs.
# round0_code/round0_generateVASP+trainNEP.sbatch does both.
# ---------------------------------------------------------------------------

PATHFILE="round0_out/round0_1_vasp_job_paths.txt"

module --force purge
module load gcc/11.2.0 openmpi/4.1.6
module load vasp/5.4.4.pl2
module list

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -f "$PATHFILE" ] || { echo "no $PATHFILE in $PWD -- submit from the project root"; exit 1; }

# --- pick this task's directory ------------------------------------------
DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PATHFILE")

if [ -z "$DIR" ]; then
    echo "No entry at line $((SLURM_ARRAY_TASK_ID + 1)) of $PATHFILE"
    echo "Array range is larger than the path list. Check: wc -l < $PATHFILE"
    exit 1
fi
if [ ! -d "$DIR" ]; then
    echo "Directory not found: $DIR"
    exit 1
fi

cd "$DIR" || exit 1
echo "=== task $SLURM_ARRAY_TASK_ID  ->  $DIR ==="

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

# --- inputs are already in place from generate_round0.py ------------------
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
