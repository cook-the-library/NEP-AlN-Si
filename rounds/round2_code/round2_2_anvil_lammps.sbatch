#!/bin/bash
#SBATCH -J r2_lmp
#SBATCH -A mat260051
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=8:00:00
#SBATCH --array=0-199%50
#SBATCH -o round2_out/logs/z-lmp-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Round 2, step 2: one LAMMPS run per array task, over the run folders
# listed by round2_1_build_md_inputs.py.
#
# Needs a LAMMPS built with NEP_CPU's USER-NEP package (pair_style nep):
#   https://github.com/brucefan1983/NEP_CPU  (interface/lammps/, cmake or make)
# Stock LAMMPS modules do not have it. Point LMP_BIN at your build and add its
# modules below if they differ from the ones VASP uses on Anvil:
#     LMP_BIN=$HOME/lammps/build/lmp sbatch --array=0-137%50 \
#         round2_code/round2_2_anvil_lammps.sbatch
# round2_code/round2_run_md.sh sets --array from the run list for you.
#
# After the MD, the committee potential (round-1.7 NEP, params committee_file)
# re-evaluates every dumped frame with in.rerun.lmp -> RERUN_DONE. A folder
# whose MD is finished but has no RERUN_DONE only gets the rerun.
#
# A run ends in one of three ways, each leaves a marker in its folder:
#   DONE     every stage finished
#   HALTED   the NEP failed (fix halt, or atoms flew out of range). This is a
#            RESULT, not a job failure: the frames before the failure are
#            harvested and are the most useful ones of the run. Exit 0.
#   (none)   LAMMPS crashed for another reason or hit the walltime. Exit 1;
#            resubmitting re-runs only these folders.
#
# 96 atoms melt-quenched at 10 K/ps is ~700k steps; at the ~10k atom-steps/s
# per core NEP_CPU gives, that is 1-2 h on 8 cores. Collision runs take minutes.
#
# Submit from the project root (the folder holding round2_code/).
# ---------------------------------------------------------------------------

PATHFILE="round2_out/round2_1_md_run_paths.txt"
LMP_BIN="${LMP_BIN:-lmp}"

module --force purge
module load gcc/11.2.0 openmpi/4.1.6
module list

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -f "$PATHFILE" ] || { echo "no $PATHFILE in $PWD -- submit from the project root"; exit 1; }

OFFSET="${CHUNK_OFFSET:-0}"
LINE=$((SLURM_ARRAY_TASK_ID + OFFSET + 1))
DIR=$(sed -n "${LINE}p" "$PATHFILE")
if [ -z "$DIR" ]; then
    echo "No entry at line $LINE of $PATHFILE -- past the end of the list."
    exit 0
fi
[ -d "$DIR" ] || { echo "Directory not found: $DIR"; exit 1; }
cd "$DIR" || exit 1
echo "=== task $SLURM_ARRAY_TASK_ID + offset $OFFSET = line $LINE  ->  $DIR ==="

run_committee() {
    # second NEP re-evaluates every dumped frame (in.rerun.lmp); RERUN_DONE marks it
    COM_FILE=$(sed -n 's/^variable committee_file *string "\(.*\)"/\1/p' params.lmp)
    if [ -z "$COM_FILE" ] || [ "$COM_FILE" = none ] || [ ! -f in.rerun.lmp ]; then
        return 0
    fi
    [ -f "$COM_FILE" ] || { echo "Missing committee potential $COM_FILE"; return 1; }
    rm -f dump.*.committee.lammpstrj log.rerun RERUN_DONE
    mpirun -n "$SLURM_NTASKS" "$LMP_BIN" -in in.rerun.lmp -log log.rerun > rerun.out 2>&1
    if grep -q "^MD_RERUN_DONE" log.rerun 2>/dev/null; then
        touch RERUN_DONE
        echo "Committee rerun finished."
        return 0
    fi
    echo "Committee rerun failed. Last lines:"
    tail -5 rerun.out
    return 1
}

command -v "$LMP_BIN" > /dev/null || { echo "LAMMPS binary not found: $LMP_BIN (set LMP_BIN)"; exit 1; }

if [ -f DONE ] || [ -f HALTED ]; then
    echo "MD already finished ($(ls DONE HALTED 2>/dev/null | tr '\n' ' '))."
    if [ -f RERUN_DONE ]; then
        echo "Committee rerun already done. Skipping."
        exit 0
    fi
    run_committee
    exit $?
fi

if ! "$LMP_BIN" -h 2>/dev/null | grep -qw nep; then
    echo "WARNING: 'nep' is not among the pair styles '$LMP_BIN -h' lists."
    echo "         Fine if it is loaded as a plugin (LAMMPS_PLUGIN_PATH);"
    echo "         otherwise this run will stop at 'pair_style nep'."
fi
for f in in.lmp params.lmp start.data run.json; do
    [ -f "$f" ] || { echo "Missing $f in $DIR"; exit 1; }
done
NEP_FILE=$(sed -n 's/^variable nep_file *string "\(.*\)"/\1/p' params.lmp)
[ -f "$NEP_FILE" ] || { echo "Missing potential $NEP_FILE (from $DIR)"; exit 1; }

# a re-run starts clean: dumps from a cut-off attempt would be read twice
rm -f dump.*.lammpstrj log.lammps lammps.out bulk_final.data slab_final.data \
      final.data RERUN_DONE log.rerun rerun.out

mpirun -n "$SLURM_NTASKS" "$LMP_BIN" -in in.lmp -log log.lammps > lammps.out 2>&1
rc=$?

if grep -q "^MD_DONE" log.lammps 2>/dev/null; then
    touch DONE
    echo "Finished (rc=$rc)."
    run_committee
    exit $?
fi
if grep -qE "Fix halt condition|Out of range atoms|Lost atoms|Non-numeric pressure" log.lammps lammps.out 2>/dev/null; then
    grep -hE "Fix halt condition|Out of range atoms|Lost atoms|Non-numeric pressure" \
        log.lammps lammps.out | head -1 > HALTED
    echo "NEP failure detected -- the frames before it will be harvested:"
    cat HALTED
    run_committee
    exit $?
fi
echo "LAMMPS ended without finishing (rc=$rc). Last lines:"
tail -5 lammps.out
exit 1
