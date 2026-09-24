#!/usr/bin/env python3
"""Stage 1: Generate structures for deposition (or a follow-up active-learning round).

Round 0 generates a diverse initial set from the AlN-on-Si(111) seeds in
aln_si_structures.py: bulk cells at several strains, Si(111)/AlN(0001) slabs,
adatoms, N2, the AlN/Si interface, and rattled (randomly displaced) copies of each
to sample off-equilibrium configurations that AIMD training data needs to cover.

Round N>0 (called when decide_next_step.py reports insufficient) additionally biases
sampling toward the configurations where evaluate_potential.py found the largest
NEP/AIMD disagreement in the previous round, per config/criteria.yaml's
active_learning.bias_toward_high_error.

Requires ASE (`pip install ase`). Falls back to a clear error rather than emitting
fake structures if ASE isn't installed -- this stage produces real VASP inputs, not
placeholders.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ConfigError,
    RUNS_DIR,
    eprint,
    load_experiment_config,
    load_criteria_config,
    read_json,
    round_dir,
    write_json,
)


def _require_ase():
    try:
        import ase  # noqa: F401
    except ImportError as e:
        raise ConfigError(
            "ASE is required for structure generation (`pip install ase`). "
            "This stage writes real POSCAR files for VASP -- there's no "
            "meaningful fallback without an atomistic toolkit."
        ) from e


def build_seed_structures(experiment_cfg: dict):
    """AlN-on-Si(111) seeds from aln_si_structures.py: bulk AlN/Si/Al phases,
    Si(111) and AlN(0001) slabs, Al/N adatoms, N2, and the 5:4 coincidence
    AlN/Si interface with the polarity and termination set in
    config/experiment_correlations.yaml.
    """
    from aln_si_structures import build_all

    exp = experiment_cfg["experiment"]
    polarity = exp.get("growth_polarity") or "Al"
    termination = exp.get("interface_termination") or "Al"
    for key, value in (("growth_polarity", polarity), ("interface_termination", termination)):
        if value not in ("Al", "N"):
            raise ConfigError(
                f"config/experiment_correlations.yaml: experiment.{key} must be "
                f"'Al' or 'N', got {value!r}."
            )

    seeds, _ = build_all(polarity=polarity, termination=termination)
    return list(seeds.items())


def rattle_copies(atoms, n, *, stdev, rng):
    from ase import Atoms

    copies = []
    for i in range(n):
        a = atoms.copy()
        a.rattle(stdev=stdev, seed=rng.randint(0, 2**31 - 1))
        copies.append(a)
    return copies


def strained_copies(atoms, strains):
    copies = []
    for s in strains:
        a = atoms.copy()
        cell = a.get_cell() * (1.0 + s)
        a.set_cell(cell, scale_atoms=True)
        copies.append(a)
    return copies


def high_error_seed_structures(prev_round_dir: Path, top_k: int):
    """For round N>0: pull the structures evaluate_potential.py flagged as
    worst-agreement from the previous round, to rattle/strain again. Returns
    [] if no per-structure error breakdown is available (e.g. round 0, or an
    evaluation stage that didn't emit one) -- caller falls back to fresh seeds.
    """
    eval_path = prev_round_dir / "evaluation.json"
    if not eval_path.exists():
        return []
    evaluation = read_json(eval_path)
    per_structure = evaluation.get("per_structure_errors", [])
    if not per_structure:
        return []
    ranked = sorted(per_structure, key=lambda x: x.get("force_rmse", 0), reverse=True)
    return ranked[:top_k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--n-structures", type=int, default=None,
                         help="Overrides active_learning.structures_per_round from criteria.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _require_ase()

    criteria = load_criteria_config()
    al_cfg = criteria["active_learning"]
    n_structures = args.n_structures or al_cfg["structures_per_round"]

    experiment_cfg = load_experiment_config()
    rng = random.Random(args.seed + args.round)

    out_dir = round_dir(args.round) / "structures"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = build_seed_structures(experiment_cfg)

    all_structures = []
    for name, atoms in seeds:
        all_structures.append((name, atoms))
        # Isotropic strain only makes sense for fully periodic bulk cells;
        # slabs and the N2 box would just gain or lose vacuum.
        if not name.startswith("bulk_"):
            continue
        all_structures.extend(
            (f"{name}_strain{i}", a)
            for i, a in enumerate(strained_copies(atoms, [-0.03, -0.01, 0.01, 0.03]))
        )

    if args.round > 0 and al_cfg.get("bias_toward_high_error", True):
        prev_dir = round_dir(args.round - 1, create=False)
        worst = high_error_seed_structures(prev_dir, top_k=max(1, n_structures // 4))
        if worst:
            eprint(
                f"[generate_structures] Round {args.round}: biasing toward "
                f"{len(worst)} high-error structures from round {args.round - 1}."
            )
            # Note: worst[i] currently carries metadata (path/id/error), not an
            # Atoms object -- wire this up to re-load the actual structure from
            # round_{N-1}/structures/<id>.vasp once evaluate_potential.py is
            # emitting per_structure_errors with a resolvable structure id.
            eprint(
                "[generate_structures] NOTE: high-error re-sampling is stubbed -- "
                "extend this function once evaluate_potential.py's per-structure "
                "error output includes a structure_id you can re-load from "
                f"{prev_dir / 'structures'}."
            )

    # Rattle every base structure to fill out the round to n_structures total.
    while len(all_structures) < n_structures:
        base_name, base_atoms = rng.choice(seeds)
        idx = len(all_structures)
        rattled = rattle_copies(base_atoms, 1, stdev=rng.uniform(0.02, 0.15), rng=rng)[0]
        all_structures.append((f"{base_name}_rattle{idx}", rattled))

    all_structures = all_structures[:n_structures]

    manifest = []
    for name, atoms in all_structures:
        poscar_path = out_dir / f"{name}.vasp"
        atoms.write(poscar_path, format="vasp")
        manifest.append({"id": name, "path": str(poscar_path)})

    write_json(out_dir.parent / "structures_manifest.json", {"round": args.round, "structures": manifest})

    print(f"[generate_structures] Round {args.round}: wrote {len(manifest)} structures to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[generate_structures] CONFIG ERROR: {e}")
        sys.exit(2)
