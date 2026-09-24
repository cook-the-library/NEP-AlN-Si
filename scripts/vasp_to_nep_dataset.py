#!/usr/bin/env python3
"""Stage 3a: Parse finished VASP jobs into a GPUMD NEP training dataset
(train.xyz / test.xyz in extended-XYZ format, which NEP >= 3.8 reads directly).

Only includes structures whose VASP job is COMPLETED (per job_ids.json / OUTCAR
sanity checks) -- silently including a failed/unconverged calculation would
poison the NEP training set with garbage energies/forces.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ConfigError, eprint, read_json, round_dir, write_json


def _require_ase():
    try:
        import ase  # noqa: F401
    except ImportError as e:
        raise ConfigError("ASE is required (`pip install ase`) to parse VASP OUTCAR/vasprun.xml output.") from e


def load_vasp_result(structure_dir: Path):
    """Returns an ase.Atoms with attached energy/forces/stress, or None if the
    calculation doesn't look finished/converged."""
    from ase.io import read

    outcar = structure_dir / "OUTCAR"
    vasprun = structure_dir / "vasprun.xml"
    vasp_out = structure_dir / "vasp.out"

    if vasp_out.exists() and "VASP_DONE" not in vasp_out.read_text()[-200:]:
        return None  # job didn't finish (or sentinel line wasn't appended -- treat as unfinished)

    try:
        if vasprun.exists():
            atoms = read(vasprun, format="vasp-xml")
        elif outcar.exists():
            atoms = read(outcar, format="vasp-out")
        else:
            return None
    except Exception as e:
        eprint(f"[vasp_to_nep_dataset] Failed to parse {structure_dir.name}: {e}")
        return None

    # Sanity: must actually have energy+forces attached to trust it for training.
    try:
        atoms.get_potential_energy()
        atoms.get_forces()
    except Exception:
        eprint(f"[vasp_to_nep_dataset] {structure_dir.name}: missing energy/forces after parse, skipping.")
        return None

    return atoms


def write_extxyz_dataset(atoms_list, path: Path):
    from ase.io import write

    write(path, atoms_list, format="extxyz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-previous-rounds", action="store_true",
                         help="Accumulate training data across all rounds 0..N (recommended for active learning) "
                              "instead of using only this round's structures.")
    args = parser.parse_args()

    _require_ase()

    rounds_to_include = list(range(args.round + 1)) if args.include_previous_rounds else [args.round]

    all_atoms = []
    per_round_counts = {}
    for rnum in rounds_to_include:
        r_dir = round_dir(rnum, create=False)
        vasp_dir = r_dir / "vasp"
        if not vasp_dir.exists():
            eprint(f"[vasp_to_nep_dataset] No vasp/ dir for round {rnum}, skipping.")
            continue
        count = 0
        for structure_dir in sorted(vasp_dir.iterdir()):
            if not structure_dir.is_dir():
                continue
            atoms = load_vasp_result(structure_dir)
            if atoms is not None:
                atoms.info["structure_id"] = structure_dir.name
                atoms.info["round"] = rnum
                all_atoms.append(atoms)
                count += 1
        per_round_counts[rnum] = count

    if not all_atoms:
        eprint("[vasp_to_nep_dataset] No completed VASP results found -- nothing to write. "
               "Check that submit_vasp.py's jobs have actually finished (see job_ids.json / squeue).")
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(all_atoms)
    n_test = max(1, int(len(all_atoms) * args.test_fraction))
    test_set = all_atoms[:n_test]
    train_set = all_atoms[n_test:]

    out_dir = round_dir(args.round) / "nep_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_extxyz_dataset(train_set, out_dir / "train.xyz")
    write_extxyz_dataset(test_set, out_dir / "test.xyz")

    write_json(out_dir / "dataset_summary.json", {
        "round": args.round,
        "rounds_included": rounds_to_include,
        "per_round_completed_counts": per_round_counts,
        "n_train": len(train_set),
        "n_test": len(test_set),
    })

    print(f"[vasp_to_nep_dataset] Round {args.round}: wrote {len(train_set)} train / {len(test_set)} test "
          f"structures to {out_dir} (from rounds {rounds_to_include})")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[vasp_to_nep_dataset] CONFIG ERROR: {e}")
        sys.exit(2)
