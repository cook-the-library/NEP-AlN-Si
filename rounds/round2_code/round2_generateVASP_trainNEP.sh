#!/bin/bash
# ---------------------------------------------------------------------------
# Round 2, steps 4-6 from ONE command. Run on a LOGIN NODE from anywhere,
# after the harvest (round2_run_md.sh) and after reading its report:
#
#     bash round2_code/round2_generateVASP_trainNEP.sh
#
#   0. round2_out/round2_4_vasp_job_paths.txt = thermal-rattle folders
#      (round2_1_thermal_rattle.py) + MD-sampled folders (harvest)
#   1. VASP arrays, 16 cores / 12 h (converged folders are skipped;
#      cut-off ones continue from WAVECAR)
#   2. audit + dataset build + NEP-vs-DFT report + contact filter of the
#      round-1wL data + merge, afterany:1
#   3. NEP stage 1 (lambda_e 0.1 / lambda_f 50, 200k), afterok:2,
#      continuing round 1wL's nep.restart
#   4. NEP stage 2 (lambda_e 10 / lambda_f 1, 200k), afterok:3
#
# Settings pass through the environment (MAX_RESTART_MB, PREV_DIR,
# MIN_CONTACT, EMIN, FRESH, NEP_BIN ...). NO_NEP=1 stops after the build.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

THERMAL="round2_out/round2_1_thermal_vasp_paths.txt"
MD="round2_out/round2_3_md_vasp_paths.txt"
PATHFILE="round2_out/round2_4_vasp_job_paths.txt"
VASP_SCRIPT="round2_code/round2_4_anvil_subVASP.sbatch"
BUILD_SCRIPT="round2_code/round2_5_build_dataset.sbatch"
NEP_SCRIPT="round2_code/round2_nep.sbatch"
PREV_DIR="${PREV_DIR:-round1wL_out/dataset}"
THROTTLE="${THROTTLE:-100}"

for f in "$VASP_SCRIPT" "$BUILD_SCRIPT" "$NEP_SCRIPT" \
         round2_code/round2_nep-1.in round2_code/round2_nep-2.in \
         round2_code/round2_6_nep_vs_dft.py round2_code/round2_5_contact_filter.py \
         round1.5_code/round1.5_5_build_dataset.py round1wL_code/round1wL_5_build_dataset.py \
         "$PREV_DIR/train.xyz" "$PREV_DIR/test.xyz"; do
    [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done
[ -f "$THERMAL" ] || echo "NOTE: $THERMAL not found -- no thermally rattled structures this round"
[ -f "$MD" ] || echo "NOTE: $MD not found -- no MD-sampled structures this round"
[ -f round1wL_out/round1wL_nep.restart ] \
    || echo "NOTE: round1wL_out/round1wL_nep.restart not found -- stage 1 will train from scratch"

mkdir -p round2_out/logs
cat $( [ -f "$THERMAL" ] && echo "$THERMAL" ) $( [ -f "$MD" ] && echo "$MD" ) \
    | grep -v '^[[:space:]]*$' | awk '!seen[$0]++' > "$PATHFILE"
N=$(grep -c . "$PATHFILE" || true)
[ "$N" -gt 0 ] || { echo "no VASP folders to run"; exit 1; }
echo "$PATHFILE has $N VASP folders"

MAXARRAY=$(scontrol show config 2>/dev/null | awk -F= '/^MaxArraySize/{gsub(/ /,"",$2); print $2}')
[ -n "${MAXARRAY:-}" ] || MAXARRAY=1001
CHUNK=$((MAXARRAY - 1))

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
    N1=$(sbatch --parsable --dependency=afterok:"$BID" --kill-on-invalid-dep=yes \
                --export=ALL,STAGE=1 "$NEP_SCRIPT")
    echo "  NEP stage 1  $N1  (after $BID; lambda_e 0.1 / lambda_f 50)"
    N2=$(sbatch --parsable --dependency=afterok:"$N1" --kill-on-invalid-dep=yes \
                -J r2_nep2 --export=ALL,STAGE=2 "$NEP_SCRIPT")
    echo "  NEP stage 2  $N2  (after $N1; lambda_e 10 / lambda_f 1)"
fi
echo
echo "reports   :  round2_out/dataset_new/nep_vs_dft.txt"
echo "             round2_out/dataset_prev_filtered/report.txt (old frames dropped)"
echo "             round2_out/dataset/report.txt"
echo "resubmit  :  round2_out/round2_5_resubmit.txt (unconverged VASP folders)"
echo "potential :  round2_out/round2_nep.txt (after stage 2)"
