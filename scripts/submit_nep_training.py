#!/usr/bin/env python3
"""Stage 3b: Train a GPUMD NEP (neuroevolution potential) on the dataset built
by vasp_to_nep_dataset.py, on either Anvil or ACES.

Writes nep.in (NEP training hyperparameters), stages train.xyz/test.xyz into
the job's working directory, renders the cluster's sbatch template, and
submits. Hyperparameter defaults below are standard NEP starting points (see
https://gpumd.org/nep/input_files/nep_in.html) -- NOT tuned for any specific
material system. If training diverges or overfits, that's the first place to
look, not something this script can auto-correct.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ConfigError,
    cluster_config,
    eprint,
    read_json,
    render_template,
    require_filled,
    round_dir,
    sbatch_context,
    submit_job,
    write_json,
)

DEFAULT_NEP_HYPERPARAMS = {
    "version": 4,
    "cutoff": "6 4",          # radial, angular cutoffs in Angstrom
    "n_max": "4 4",
    "basis_size": "8 8",
    "l_max": "4 2 1",
    "neuron": 30,
    "lambda_1": 0.05,
    "lambda_2": 0.05,
    "batch": 1000,
    "population": 50,
    "generation": 100000,
}


def detect_species(train_xyz: Path) -> list[str]:
    from ase.io import read

    frames = read(train_xyz, index=":")
    species = set()
    for atoms in frames:
        species.update(atoms.get_chemical_symbols())
    return sorted(species)


def write_nep_in(path: Path, species: list[str], overrides: dict | None = None):
    params = dict(DEFAULT_NEP_HYPERPARAMS)
    if overrides:
        params.update(overrides)
    lines = [f"type {len(species)} {' '.join(species)}"]
    for k, v in params.items():
        lines.append(f"{k} {v}")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--cluster", choices=["anvil", "aces"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = cluster_config(args.cluster)
    require_filled(cfg, context=f"submit_nep_training.py --cluster {args.cluster}")

    r_dir = round_dir(args.round)
    dataset_dir = r_dir / "nep_dataset"
    train_xyz, test_xyz = dataset_dir / "train.xyz", dataset_dir / "test.xyz"
    if not train_xyz.exists() or not test_xyz.exists():
        raise ConfigError(
            f"{train_xyz} / {test_xyz} not found -- run vasp_to_nep_dataset.py "
            f"for round {args.round} first."
        )

    model_dir = r_dir / "nep_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "train.xyz").write_text(train_xyz.read_text())
    (model_dir / "test.xyz").write_text(test_xyz.read_text())

    species = detect_species(train_xyz)
    write_nep_in(model_dir / "nep.in", species)

    ctx = sbatch_context(cfg, job_name=f"nep_train_{args.round}", workdir=model_dir, kind="nep")
    sbatch_text = render_template("nep.sbatch.template", ctx)
    sbatch_path = model_dir / "submit.sbatch"
    sbatch_path.write_text(sbatch_text)

    if args.dry_run:
        print(f"[submit_nep_training] (dry-run) rendered {sbatch_path}, not submitting")
        return

    job_id = submit_job(sbatch_path)
    write_json(model_dir / "job_id.json", {"round": args.round, "cluster": args.cluster, "job_id": job_id})
    print(f"[submit_nep_training] Round {args.round}: submitted NEP training as job {job_id} on {args.cluster} "
          f"(species: {species})")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[submit_nep_training] CONFIG ERROR: {e}")
        sys.exit(2)
