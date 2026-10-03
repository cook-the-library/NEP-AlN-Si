#!/bin/bash
# ---------------------------------------------------------------------------
# Round 2, steps 2-3. Run on a LOGIN NODE from anywhere, after
# round2_1_build_md_inputs.py has written the run list. Two passes:
#
#   pass 1   python round2_code/round2_1_build_md_inputs.py
#            NO_HARVEST=1 LMP_BIN=/path/to/lmp bash round2_code/round2_run_md.sh
#   pass 2   (after pass 1 finished: quench slabs and precursor surfaces exist)
#            python round2_code/round2_1_build_md_inputs.py --pass2
#            LMP_BIN=/path/to/lmp bash round2_code/round2_run_md.sh
#
# Each call submits
#   1. the LAMMPS array, one task per run folder; folders already DONE or
#      HALTED only get the committee rerun if it is missing, so pass 2 redoes
#      nothing from pass 1
#   2. unless NO_HARVEST=1: the harvest, afterany:1 -> round2_out/vasp_md/
#
# LMP_BIN (LAMMPS with pair_style nep), THROTTLE, REF and HARVEST_ARGS pass
# through the environment to the jobs.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUNLIST="round2_out/round2_1_md_run_paths.txt"
LMP_SCRIPT="round2_code/round2_2_anvil_lammps.sbatch"
HARVEST_SCRIPT="round2_code/round2_3_harvest.sbatch"
THROTTLE="${THROTTLE:-50}"

for f in "$RUNLIST" "$LMP_SCRIPT" "$HARVEST_SCRIPT" round2_code/round2_3_harvest.py \
         round2_code/round2_common.py round1_code/generate_round1.py; do
    [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done
[ -n "${LMP_BIN:-}" ] || echo "NOTE: LMP_BIN not set -- the jobs will look for 'lmp' on PATH"

mkdir -p round2_out/logs     # sbatch opens every -o file before its job starts

N=$(grep -c . "$RUNLIST")
MAXARRAY=$(scontrol show config 2>/dev/null | awk -F= '/^MaxArraySize/{gsub(/ /,"",$2); print $2}')
[ -n "${MAXARRAY:-}" ] || MAXARRAY=1001
CHUNK=$((MAXARRAY - 1))
echo "$RUNLIST has $N run folders"

IDS=""
OFFSET=0
while [ "$OFFSET" -lt "$N" ]; do
    REMAIN=$((N - OFFSET))
    SIZE=$(( REMAIN < CHUNK ? REMAIN : CHUNK ))
    JID=$(sbatch --parsable --array="0-$((SIZE - 1))%${THROTTLE}" \
                 --export=ALL,CHUNK_OFFSET="$OFFSET" "$LMP_SCRIPT")
    echo "  LAMMPS array $JID  lines $((OFFSET + 1))-$((OFFSET + SIZE))"
    IDS="${IDS:+$IDS:}$JID"
    OFFSET=$((OFFSET + SIZE))
done

if [ "${NO_HARVEST:-0}" = 1 ]; then
    echo "  harvest not submitted (NO_HARVEST=1)"
    echo
    echo "when these finish:  python round2_code/round2_1_build_md_inputs.py --pass2"
    echo "                    LMP_BIN=${LMP_BIN:-lmp} bash round2_code/round2_run_md.sh"
    exit 0
fi
HID=$(sbatch --parsable --dependency=afterany:"$IDS" "$HARVEST_SCRIPT")
echo "  harvest      $HID  (after $IDS)"
echo
echo "watch   :  squeue -u \$USER"
echo "logs    :  round2_out/logs/"
echo "then    :  read round2_out/vasp_md/harvest_report.txt, and if the selection"
echo "           looks right: bash round2_code/round2_generateVASP_trainNEP.sh"
