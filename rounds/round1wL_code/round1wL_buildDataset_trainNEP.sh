#!/bin/bash
# ---------------------------------------------------------------------------
# Whole round 1wL from ONE command. Run on a LOGIN NODE from anywhere:
#
#     bash round1wL_code/round1wL_buildDataset_trainNEP.sh
#
# Submits two dependent jobs and returns:
#   1. dataset build  round-1.7 dataset + legacy B+C, abnormal labels rejected
#   2. NEP training   afterok:1, fresh start (no earlier nep.restart)
#
# Settings pass through the environment, e.g.
#     DROP="r1.7:interface" bash round1wL_code/round1wL_buildDataset_trainNEP.sh
# (EMIN, EMAX_COND, EMAX_GAS, FMAX, DROP, ROUND_DIR, LEGACY_DIR)
# If the build fails, the NEP job is cancelled automatically.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROUND_DIR="${ROUND_DIR:-round1.7_out/dataset}"
LEGACY_DIR="${LEGACY_DIR:-roundL_out/dataset_legacy/BC}"
BUILD_SCRIPT="round1wL_code/round1wL_5_build_dataset.sbatch"
NEP_SCRIPT="round1wL_code/round1wL_nep.sbatch"

for f in "$BUILD_SCRIPT" "$NEP_SCRIPT" \
         round1wL_code/round1wL_5_build_dataset.py round1wL_code/round1wL_nep-1.in \
         "$ROUND_DIR/train.xyz" "$ROUND_DIR/test.xyz" \
         "$LEGACY_DIR/train.xyz" "$LEGACY_DIR/test.xyz"; do
    [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done

echo "NEP will train from a fresh start (round 1.7's nep.restart is not used)"

# sbatch opens every -o file before its job starts
mkdir -p round1wL_out/logs

BUILD_ID=$(sbatch --parsable "$BUILD_SCRIPT")
echo "1. dataset build : $BUILD_ID"

NEP_ID=$(sbatch --parsable --dependency=afterok:"$BUILD_ID" \
                --kill-on-invalid-dep=yes "$NEP_SCRIPT")
echo "2. NEP training  : $NEP_ID"

echo
echo "watch     :  squeue -u \$USER"
echo "report    :  round1wL_out/dataset/report.txt  (+ rejected.csv)"
echo "logs      :  round1wL_out/logs/"
echo "potential :  round1wL_out/round1wL_nep.txt"
