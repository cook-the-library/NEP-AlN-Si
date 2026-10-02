#!/bin/bash
# ---------------------------------------------------------------------------
# Round 1md, steps 2-3 from ONE command. Run on a LOGIN NODE from anywhere,
# after step 1 (round1md_1_build_md_inputs.py) has written the run list:
#
#     LMP_BIN=/path/to/lmp bash round1md_code/round1md_run_md.sh
#
#   1. LAMMPS array, one task per run folder (folders with DONE or HALTED
#      are skipped, so re-running this only redoes unfinished runs)
#   2. harvest, afterany:1 -> round1md_out/vasp/ + harvest_report.txt
#
# LMP_BIN (LAMMPS with pair_style nep), THROTTLE, REF and HARVEST_ARGS pass
# through the environment to the jobs.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUNLIST="round1md_out/round1md_1_md_run_paths.txt"
LMP_SCRIPT="round1md_code/round1md_2_anvil_lammps.sbatch"
HARVEST_SCRIPT="round1md_code/round1md_3_harvest.sbatch"
THROTTLE="${THROTTLE:-50}"

for f in "$RUNLIST" "$LMP_SCRIPT" "$HARVEST_SCRIPT" round1md_code/round1md_3_harvest.py \
         round1_code/generate_round1.py; do
    [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done
[ -n "${LMP_BIN:-}" ] || echo "NOTE: LMP_BIN not set -- the jobs will look for 'lmp' on PATH"

mkdir -p round1md_out/logs     # sbatch opens every -o file before its job starts

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

HID=$(sbatch --parsable --dependency=afterany:"$IDS" "$HARVEST_SCRIPT")
echo "  harvest      $HID  (after $IDS)"
echo
echo "watch   :  squeue -u \$USER"
echo "logs    :  round1md_out/logs/"
echo "then    :  read round1md_out/vasp/harvest_report.txt, and if the selection"
echo "           looks right: bash round1md_code/round1md_generateVASP_trainNEP.sh"
