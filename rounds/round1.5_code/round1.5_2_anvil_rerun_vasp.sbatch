#!/bin/bash
#SBATCH -J r15_vasp
#SBATCH -A mat260051
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --array=0-99%100
#SBATCH -o round1.5_out/logs/z-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Round 1.5: re-run the VASP jobs round 0 and round 1 did not finish, IN
# PLACE, keeping WAVECAR and CHGCAR so a job that runs out of walltime can be
# continued later instead of starting over.
#
# Why this needs the STOPCAR trick
# --------------------------------
# VASP only writes WAVECAR/CHGCAR when the run ends normally. A job killed by
# SLURM at the walltime limit writes nothing, which is why re-running these
# folders under the old scripts repeated the same 12 hours of work. So:
#
#   --signal=B:USR1@1800 makes SLURM send SIGUSR1 to this script 30 minutes
#   before the limit. The trap writes STOPCAR with LABORT = .TRUE., VASP
#   checks STOPCAR at the next electronic step, stops the SCF loop there and
#   finishes the run properly -- which means WAVECAR and CHGCAR get written.
#
# The 30-minute grace has to cover ONE electronic step plus writing a WAVECAR
# that can be several GB. For the 196-atom interface folders one SCF step can
# take ~10 min, so do not cut it much below this.
#
# Restart ladder, per folder
#   WAVECAR present -> ISTART 1 (continue from the orbitals)
#   else CHGCAR      -> ISTART 0, ICHARG 1 (start from the saved density)
#   else                ISTART 0, ICHARG 2 (from scratch, as round 0/1 did)
# A restart that dies within EARLY_FAIL_SEC is treated as a corrupt WAVECAR:
# the restart files are deleted and the folder is retried once from scratch.
#
# On success WAVECAR/CHGCAR/CHG are deleted straight away. They are only kept
# for folders that did NOT converge. That is still a lot of disk: a WAVECAR
# for a 196-atom interface cell is roughly 4-5 GB. Check your quota, and set
# MAX_RESTART_MB in the environment to drop restart files above that size:
#
#     MAX_RESTART_MB=8000 bash round1.5_code/round1.5_generateVASP_trainNEP.sbatch
#
# INCAR is edited in place (LWAVE/LCHARG/ISTART/ICHARG only); the original is
# kept as INCAR.orig the first time a folder is touched.
#
# Submit from the project root. round1.5_out/logs must exist beforehand;
# round1.5_code/round1.5_generateVASP_trainNEP.sbatch does the mkdir and the
# chunking, and passes CHUNK_OFFSET.
# ---------------------------------------------------------------------------

PATHFILE="round1.5_out/round1.5_1_rerun_paths.txt"
EARLY_FAIL_SEC=180

module --force purge
module load gcc/11.2.0 openmpi/4.1.6
module load vasp/5.4.4.pl2
module list

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -f "$PATHFILE" ] || { echo "no $PATHFILE in $PWD -- submit from the project root"; exit 1; }

# --- pick this task's directory -------------------------------------------
OFFSET="${CHUNK_OFFSET:-0}"
LINE=$((SLURM_ARRAY_TASK_ID + OFFSET + 1))
DIR=$(sed -n "${LINE}p" "$PATHFILE")

if [ -z "$DIR" ]; then
    echo "No entry at line $LINE of $PATHFILE (offset $OFFSET) -- past the end"
    echo "of the list. Nothing to do."
    exit 0
fi
if [ ! -d "$DIR" ]; then
    echo "Directory not found: $DIR"
    exit 1
fi

cd "$DIR" || exit 1
echo "=== task $SLURM_ARRAY_TASK_ID + offset $OFFSET = line $LINE  ->  $DIR ==="

converged() {
    [ -f OUTCAR ] \
    && grep -q "Voluntary context switches" OUTCAR \
    && grep -q "aborting loop because EDIFF is reached" OUTCAR
}

if converged; then
    echo "Already converged. Skipping, and clearing any leftover restart files."
    rm -f WAVECAR CHGCAR CHG STOPCAR .wall_stop
    exit 0
fi

for f in POSCAR KPOINTS POTCAR INCAR; do
    if [ ! -f "$f" ]; then
        echo "Missing $f in $DIR"
        exit 1
    fi
done

# --- INCAR: write the restart files, and use them if they are there --------
[ -f INCAR.orig ] || cp INCAR INCAR.orig

set_incar() {                 # set_incar KEY VALUE
    local k=$1 v=$2
    if grep -qiE "^[[:space:]]*${k}[[:space:]]*=" INCAR; then
        sed -i -E "s#^[[:space:]]*${k}[[:space:]]*=.*#${k} = ${v}#I" INCAR
    else
        printf '%s = %s\n' "$k" "$v" >> INCAR
    fi
}

set_incar LWAVE  ".TRUE."
set_incar LCHARG ".TRUE."

if [ -s WAVECAR ]; then
    set_incar ISTART 1
    set_incar ICHARG 0
    RESTART=wavecar
    echo "continuing from WAVECAR ($(du -sh WAVECAR | cut -f1))"
elif [ -s CHGCAR ]; then
    set_incar ISTART 0
    set_incar ICHARG 1
    RESTART=chgcar
    echo "no WAVECAR; starting from CHGCAR ($(du -sh CHGCAR | cut -f1))"
else
    set_incar ISTART 0
    set_incar ICHARG 2
    RESTART=none
    echo "no restart files; starting from scratch"
fi

# --- run, with a graceful stop before the walltime limit ------------------
rm -f STOPCAR .wall_stop
[ -f VASP_job.out ] && mv -f VASP_job.out VASP_job.prev.out

on_wall_signal() {
    echo "$(date '+%F %T') walltime signal received: writing STOPCAR (LABORT)"
    echo "  VASP will stop at the next electronic step and write WAVECAR/CHGCAR."
    printf 'LABORT = .TRUE.\n' > STOPCAR
    touch .wall_stop
}
trap on_wall_signal USR1

run_vasp() {
    local t0=$SECONDS
    mpirun -n "$SLURM_NTASKS" vasp_std &> VASP_job.out &
    VPID=$!
    # wait returns early when the trap fires, so keep waiting for the real exit
    wait "$VPID"; rc=$?
    while [ "$rc" -gt 128 ]; do wait "$VPID"; rc=$?; done
    elapsed=$((SECONDS - t0))
    echo "vasp exit code $rc after ${elapsed}s (restart mode: $RESTART)"
}

run_vasp

if [ "$rc" -ne 0 ] && [ "$RESTART" = wavecar ] \
   && [ "$elapsed" -lt "$EARLY_FAIL_SEC" ] && [ ! -f .wall_stop ]; then
    echo "Died ${elapsed}s into a WAVECAR restart -- treating it as corrupt."
    echo "Deleting the restart files and retrying from scratch."
    rm -f WAVECAR CHGCAR CHG
    set_incar ISTART 0
    set_incar ICHARG 2
    RESTART=none
    rm -f STOPCAR .wall_stop
    mv -f VASP_job.out VASP_job.restartfail.out
    run_vasp
fi

rm -f STOPCAR CHG

# --- converged: drop the restart files. Not converged: keep them ----------
if converged; then
    echo "Converged. Deleting WAVECAR/CHGCAR to save space."
    rm -f WAVECAR CHGCAR .wall_stop
    exit 0
fi

if [ -f .wall_stop ]; then
    echo "Stopped at the walltime limit with the SCF still running."
else
    echo "WARNING: run ended without reaching EDIFF (rc=$rc), and not by the"
    echo "         walltime signal -- check VASP_job.out and OSZICAR."
fi

SZ=$(du -sm WAVECAR CHGCAR 2>/dev/null | awk '{s += $1} END {print s + 0}')
LIMIT=${MAX_RESTART_MB:-0}
if [ "$LIMIT" -gt 0 ] && [ "$SZ" -gt "$LIMIT" ]; then
    echo "restart files are ${SZ} MB > MAX_RESTART_MB=${LIMIT} -- deleting them"
    rm -f WAVECAR CHGCAR
else
    echo "kept ${SZ} MB of restart files; re-running this folder continues from them"
fi

exit $rc
