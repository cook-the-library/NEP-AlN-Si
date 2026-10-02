#!/bin/bash
#SBATCH -J r1md_harvest
#SBATCH -A mat260051
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=3:00:00
#SBATCH -o round1md_out/logs/z-harvest-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Round 1md, step 3: screen and select frames from every LAMMPS run, write
# them as VASP folders (round1md_3_harvest.py). One core does the work; the
# 4 tasks are there for memory (~150k candidate frames).
#
# Runs after the LAMMPS array with afterany, so it also harvests runs that
# crashed or hit the walltime -- whatever frames they dumped before.
#
# The round-1wL training set seeds the selection, so frames that look like
# data already in it are picked last. Override from the submitting shell:
#     REF=none sbatch round1md_code/round1md_3_harvest.sbatch
#     HARVEST_ARGS="--n-collision 600 --force" sbatch ...
#
# Submit from the project root.
# ---------------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -d round1md_code ] || { echo "no round1md_code/ in $PWD -- submit from the project root"; exit 1; }

RUNLIST="round1md_out/round1md_1_md_run_paths.txt"
REF="${REF:-round1wL_out/dataset/train.xyz}"
HARVEST_ARGS="${HARVEST_ARGS:-}"

[ -f "$RUNLIST" ] || { echo "missing $RUNLIST"; exit 1; }
echo "run folders : $(grep -c . "$RUNLIST")"
echo "  finished  : $(while read -r d; do [ -f "$d/DONE" ] && echo; done < "$RUNLIST" | wc -l)"
echo "  NEP failed: $(while read -r d; do [ -f "$d/HALTED" ] && echo; done < "$RUNLIST" | wc -l)"

module load conda
module use "$HOME/privatemodules"
module load conda-env/nepdata

ARGS=(--run-list "$RUNLIST" --outdir round1md_out/vasp
      --path-list round1md_out/round1md_3_vasp_job_paths.txt)
if [ "$REF" != none ] && [ -f "$REF" ]; then
    ARGS+=(--reference-xyz "$REF")
else
    echo "NOTE: no reference set ($REF) -- selection is not seeded"
fi

# shellcheck disable=SC2086
python round1md_code/round1md_3_harvest.py "${ARGS[@]}" $HARVEST_ARGS || exit 1

echo
echo "Check round1md_out/vasp/harvest_report.txt before submitting VASP:"
echo "    bash round1md_code/round1md_generateVASP_trainNEP.sh"
