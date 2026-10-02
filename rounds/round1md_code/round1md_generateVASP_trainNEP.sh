#!/bin/bash
# ---------------------------------------------------------------------------
# Round 1md, steps 4-6 from ONE command. Run on a LOGIN NODE from anywhere,
# after the harvest (round1md_run_md.sh) and after reading its report:
#
#     bash round1md_code/round1md_generateVASP_trainNEP.sh
#
#   1. VASP arrays over round1md_out/round1md_3_vasp_job_paths.txt
#      (converged folders are skipped; cut-off ones continue from WAVECAR)
#   2. audit + dataset build + NEP-vs-DFT report + merge with round 1wL,
#      afterany:1
#   3. NEP training, afterok:2, continuing round 1wL's restart
#
# Settings pass through the environment (MAX_RESTART_MB, PREV_DIR, EMIN,
# FRESH, NEP_BIN ...). Set NO_NEP=1 to stop after the dataset build.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PATHFILE="round1md_out/round1md_3_vasp_job_paths.txt"
VASP_SCRIPT="round1md_code/round1md_4_anvil_subVASP.sbatch"
BUILD_SCRIPT="round1md_code/round1md_5_build_dataset.sbatch"
NEP_SCRIPT="round1md_code/round1md_nep.sbatch"
PREV_DIR="${PREV_DIR:-round1wL_out/dataset}"
THROTTLE="${THROTTLE:-100}"

for f in "$PATHFILE" "$VASP_SCRIPT" "$BUILD_SCRIPT" "$NEP_SCRIPT" \
         round1md_code/round1md_nep-1.in round1md_code/round1md_6_nep_vs_dft.py \
         round1.5_code/round1.5_5_build_dataset.py round1wL_code/round1wL_5_build_dataset.py \
         "$PREV_DIR/train.xyz" "$PREV_DIR/test.xyz"; do
    [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done
[ -f round1wL_out/round1wL_nep.restart ] \
    || echo "NOTE: round1wL_out/round1wL_nep.restart not found -- NEP will train from scratch"

mkdir -p round1md_out/logs

N=$(grep -c . "$PATHFILE")
MAXARRAY=$(scontrol show config 2>/dev/null | awk -F= '/^MaxArraySize/{gsub(/ /,"",$2); print $2}')
[ -n "${MAXARRAY:-}" ] || MAXARRAY=1001
CHUNK=$((MAXARRAY - 1))
echo "$PATHFILE has $N VASP folders"

IDS=""
OFFSET=0
while [ "$OFFSET" -lt "$N" ]; do
    REMAIN=$((N - OFFSET))
    SIZE=$(( REMAIN < CHUNK ? REMAIN : CHUNK ))
    JID=$(sbatch --parsable --array="0-$((SIZE - 1))%${THROTTLE}" \
                 --export=ALL,CHUNK_OFFSET="$OFFSET" "$VASP_SCRIPT")
    echo "  VASP array   $JID  lines $((OFFSET + 1))-$((OFFSET + SIZE))"
    IDS="${IDS:+$IDS:}$JID"
    OFFSET=$((OFFSET + SIZE))
done

BID=$(sbatch --parsable --dependency=afterany:"$IDS" "$BUILD_SCRIPT")
echo "  build        $BID  (after $IDS)"
if [ "${NO_NEP:-0}" = 1 ]; then
    echo "  NEP training not submitted (NO_NEP=1)"
else
    NID=$(sbatch --parsable --dependency=afterok:"$BID" --kill-on-invalid-dep=yes "$NEP_SCRIPT")
    echo "  NEP training $NID  (after $BID)"
fi
echo
echo "reports   :  round1md_out/dataset_new/nep_vs_dft.txt  round1md_out/dataset/report.txt"
echo "resubmit  :  round1md_out/round1md_5_resubmit.txt (unconverged VASP folders)"
echo "potential :  round1md_out/round1md_nep.txt"
