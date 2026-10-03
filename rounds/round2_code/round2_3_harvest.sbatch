#!/bin/bash
#SBATCH -J r2_harvest
#SBATCH -A mat260051
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=3:00:00
#SBATCH -o round2_out/logs/z-harvest-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Round 2, step 3: screen and select frames from every LAMMPS run, write
# them as VASP folders (round2_3_harvest.py). Frames where the round-1wL and
# round-1.7 potentials disagree (committee rerun) are picked first. One core
# does the work; the 4 tasks are there for memory (~150k candidate frames).
#
# Runs after the LAMMPS array with afterany, so it also harvests runs that
# crashed or hit the walltime -- whatever frames they dumped before.
#
# The round-1wL training set seeds the selection, so frames that look like
# data already in it are picked last. Override from the submitting shell:
#     REF=none sbatch round2_code/round2_3_harvest.sbatch
#     HARVEST_ARGS="--n-collision 600 --force" sbatch ...
#
# Submit from the project root.
# ---------------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -d round2_code ] || { echo "no round2_code/ in $PWD -- submit from the project root"; exit 1; }

RUNLIST="round2_out/round2_1_md_run_paths.txt"
REF="${REF:-round1wL_out/dataset/train.xyz}"
HARVEST_ARGS="${HARVEST_ARGS:-}"

[ -f "$RUNLIST" ] || { echo "missing $RUNLIST"; exit 1; }
echo "run folders : $(grep -c . "$RUNLIST")"
echo "  finished  : $(while read -r d; do [ -f "$d/DONE" ] && echo; done < "$RUNLIST" | wc -l)"
echo "  NEP failed: $(while read -r d; do [ -f "$d/HALTED" ] && echo; done < "$RUNLIST" | wc -l)"
echo "  committee : $(while read -r d; do [ -f "$d/RERUN_DONE" ] && echo; done < "$RUNLIST" | wc -l) reruns done"

module load conda
module use "$HOME/privatemodules"
module load conda-env/nepdata

ARGS=(--run-list "$RUNLIST" --outdir round2_out/vasp_md
      --path-list round2_out/round2_3_md_vasp_paths.txt)
if [ "$REF" != none ] && [ -f "$REF" ]; then
    ARGS+=(--reference-xyz "$REF")
else
    echo "NOTE: no reference set ($REF) -- selection is not seeded"
fi

# shellcheck disable=SC2086
python round2_code/round2_3_harvest.py "${ARGS[@]}" $HARVEST_ARGS || exit 1

echo
echo "Check round2_out/vasp_md/harvest_report.txt before submitting VASP:"
echo "    bash round2_code/round2_generateVASP_trainNEP.sh"
