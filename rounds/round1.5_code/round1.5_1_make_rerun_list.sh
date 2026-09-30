#!/bin/bash
# ---------------------------------------------------------------------------
# Round 1.5, step 1: work out what still has to run.
#
# Round 1.5 generates NO new structures. It re-runs the folders round 0 and
# round 1 still owe us -- the ones that timed out, were cancelled, or never
# started -- in place, inside round0_out/vasp/ and round1_out/vasp/. So there
# is no round1.5_out/vasp/.
#
# Input : round0_out/round0_3_resubmit.txt
#         round1_out/round1_3_resubmit.txt
# Output: round1.5_out/round1.5_1_rerun_paths.txt
#
# Folders that have since converged, paths that no longer exist, and
# structures with ATOM_LIMIT (150) or more atoms are dropped here rather than
# wasting an array task each. The dataset builder rejects >= 150-atom frames
# anyway, so every round keeps structures strictly below 150 atoms.
#
# Run from anywhere; it moves to the project root itself:
#     bash round1.5_code/round1.5_1_make_rerun_list.sh
# ---------------------------------------------------------------------------

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

OUT="round1.5_out"
LIST="$OUT/round1.5_1_rerun_paths.txt"
CAND="$OUT/.rerun_candidates"
ATOM_LIMIT=150

mkdir -p "$OUT/logs"

SRC=()
for f in round0_out/round0_3_resubmit.txt round1_out/round1_3_resubmit.txt; do
    if [ -f "$f" ]; then
        SRC+=("$f")
        echo "reading $f ($(grep -c . "$f") entries)"
    else
        echo "NOTE: $f not found -- skipping that round"
    fi
done

if [ ${#SRC[@]} -eq 0 ]; then
    echo "ERROR: no resubmit list found. Run each round's audit job first"
    echo "       (round0_code/round0_34_anvil_check_and_build.sbatch and"
    echo "        round1_code/round1_34_anvil_check_and_build.sbatch)."
    exit 1
fi

# order-preserving dedupe
cat "${SRC[@]}" | grep -v '^[[:space:]]*$' | awk '!seen[$0]++' > "$CAND"

natoms() {                    # natoms DIR -- atom count from a VASP5 POSCAR
    awk 'NR == 6 && $1 !~ /^[0-9]+$/ { next }       # species line
         NR >= 6 { n = 0; for (i = 1; i <= NF; i++) n += $i; print n; exit }' \
        "$1/POSCAR" 2>/dev/null
}

converged() {                 # converged DIR -- cheap test, footer first
    local o="$1/OUTCAR"
    [ -f "$o" ] || return 1
    tail -c 4000 "$o" | grep -q "Voluntary context switches" || return 1
    grep -q -m1 "aborting loop because EDIFF is reached" "$o"
}

: > "$LIST"
n_todo=0; n_done=0; n_missing=0; n_big=0
while IFS= read -r DIR; do
    if [ ! -d "$DIR" ]; then
        n_missing=$((n_missing + 1))
        echo "  missing directory, dropped: $DIR"
        continue
    fi
    n=$(natoms "$DIR")
    if [ -n "$n" ] && [ "$n" -ge "$ATOM_LIMIT" ]; then
        n_big=$((n_big + 1))
        echo "  $n atoms (limit: fewer than $ATOM_LIMIT), dropped: $DIR"
        continue
    fi
    if converged "$DIR"; then
        n_done=$((n_done + 1))
        continue
    fi
    printf '%s\n' "$DIR" >> "$LIST"
    n_todo=$((n_todo + 1))
done < "$CAND"
rm -f "$CAND"

echo "--------------------------------------------------"
echo "already converged since the audit : $n_done (dropped)"
echo "paths that no longer exist        : $n_missing (dropped)"
echo "structures with >= $ATOM_LIMIT atoms      : $n_big (dropped)"
echo "to re-run                         : $n_todo -> $LIST"
echo "--------------------------------------------------"

if [ "$n_todo" -eq 0 ]; then
    echo "Nothing left to re-run. round1.5 can go straight to the dataset build."
fi
