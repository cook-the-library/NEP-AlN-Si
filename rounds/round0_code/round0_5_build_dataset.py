#!/usr/bin/env python3
"""
Build NEP training data from finished VASP jobs. Round-agnostic: pass any
--paths file and --outdir, so the same script builds round0's dataset,
round1's, round2's, etc.

Replaces round0_4_selecting_outcars_for_training_data.py.

What it does differently:

  1. Writes virial, which NEP needs and which a plain extxyz dump omits.
  2. Splits by STRUCTURE FAMILY, not by frame. An EOS scan, a registry scan
     and a separation scan are each many near-duplicates of one parent; a
     random 80/20 split puts copies on both sides and the resulting test RMSE
     is meaningless. Families are found by descriptor-space clustering and
     assigned whole to train or test.
  3. Stratifies by bucket, so every structure type appears in both sets.
  4. Carries provenance (bucket, kind, folder) into the xyz, so later rounds
     can compute per-bucket RMSE and enforce explorer quotas.
  5. Screens unconverged SCF and unfinished jobs itself rather than trusting
     an upstream text file.

Usage
-----
    # round 0 (first time, nothing to carry forward)
    python round0_code/round0_5_build_dataset.py \
        --paths round0_out/round0_1_vasp_job_paths.txt \
        --outdir round0_out/dataset

    # round 1: --paths points at the UNION of every round's job-path file so
    # far (the round-1 check+build script writes that union to
    # round1_out/round1_1_combined_paths.txt), and --carry-split-from freezes
    # every previously-split folder's train/test label instead of re-shuffling the
    # whole pool. Without --carry-split-from, adding round1's data reshuffles
    # round0's split too, which breaks "test.xyz FIXED for the campaign" and
    # makes RMSE across rounds not comparable.
    python round1_code/round1_5_build_dataset.py \
        --paths round1_out/round1_1_combined_paths.txt \
        --outdir round1_out/dataset \
        --carry-split-from round0_out/dataset/split_manifest.csv

Output
------
    <outdir>/train.xyz          NEP training set
    <outdir>/test.xyz           held-out set, FIXED for the campaign
    <outdir>/split_manifest.csv folder -> split, family, energy, forces
    <outdir>/rejected.csv       every folder dropped, with the reason
    <outdir>/report.txt         summary and per-bucket statistics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from ase.io import read, write

# --------------------------------------------------------------------------
# reading and screening VASP output
# --------------------------------------------------------------------------


def parse_nelm(folder: str, default: int = 200) -> int:
    for name in ("INCAR", "OUTCAR"):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, errors="ignore") as fh:
                for line in fh:
                    if "NELM" in line and "NELMIN" not in line \
                            and "NELMDL" not in line:
                        token = line.split("NELM")[1].split("=")[1]
                        return int(float(token.split(";")[0].split()[0]))
        except (IndexError, ValueError, OSError):
            continue
    return default


def scf_converged(folder: str, nelm: int) -> tuple[bool, str]:
    """
    A static VASP run is trustworthy only if the electronic loop exited on
    EDIFF rather than on NELM. Check the strong signal first, then fall back
    to counting iterations in OSZICAR.
    """
    outcar = os.path.join(folder, "OUTCAR")
    if not os.path.isfile(outcar):
        return False, "no OUTCAR"

    with open(outcar, errors="ignore") as fh:
        text = fh.read()

    if "Voluntary context switches" not in text:
        return False, "job did not finish (no clean exit in OUTCAR)"

    if "aborting loop because EDIFF is reached" in text:
        return True, ""

    oszicar = os.path.join(folder, "OSZICAR")
    if os.path.isfile(oszicar):
        steps = []
        with open(oszicar, errors="ignore") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) > 1 and parts[0].rstrip(":") in (
                        "DAV", "RMM", "CG", "EDDAV", "DMP"):
                    try:
                        steps.append(int(parts[1]))
                    except ValueError:
                        pass
        if steps:
            last = steps[-1]
            if last >= nelm:
                return False, f"SCF hit NELM ({last}/{nelm})"
            return True, ""

    return False, "cannot confirm SCF convergence"


def load_provenance(folder: str) -> dict:
    path = os.path.join(folder, "info.json")
    if not os.path.isfile(path):
        base = os.path.basename(folder.rstrip("/"))
        bucket = base.rsplit("_", 1)[0] if "_" in base else base
        return {"bucket": bucket, "kind": "unknown"}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"bucket": "unknown", "kind": "unknown"}


def attach_nep_fields(atoms, info: dict, folder: str, write_stress: bool):
    """
    Put energy, forces and virial where NEP expects them.

    NEP reads the virial tensor in eV. ASE gives stress in eV/A^3 with the
    convention that positive is tension, so virial = -stress * volume.
    """
    energy = atoms.get_potential_energy(force_consistent=False)
    forces = atoms.get_forces()

    virial = None
    try:
        stress = atoms.get_stress(voigt=False)  # eV/A^3, 3x3
        virial = -np.asarray(stress) * atoms.get_volume()
    except Exception:
        pass

    new = atoms.copy()
    new.info = {}
    new.calc = None
    new.info["energy"] = float(energy)
    new.arrays["forces"] = np.asarray(forces)
    if virial is not None:
        new.info["virial"] = virial.reshape(9)
        if write_stress:
            new.info["stress"] = np.asarray(stress).reshape(9)

    new.info["config_type"] = str(info.get("bucket", "unknown"))
    new.info["kind"] = str(info.get("kind", "unknown"))
    new.info["folder"] = os.path.basename(folder.rstrip("/"))
    new.info["round"] = 0
    return new, energy, forces, virial


# --------------------------------------------------------------------------
# structure families
# --------------------------------------------------------------------------


def descriptor(atoms, r_cut=6.0, n_bins=32) -> np.ndarray:
    """
    Cheap, rotation- and permutation-invariant fingerprint: a smoothed
    histogram of pair distances per element pair, plus composition.
    Good enough to recognise that two frames are the same parent structure.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    species = sorted(set(symbols))
    pairs = [(a, b) for i, a in enumerate(species) for b in species[i:]]

    vec = []
    if len(atoms) > 1:
        d = atoms.get_all_distances(mic=any(atoms.pbc))
        np.fill_diagonal(d, np.inf)
        for a, b in pairs:
            ia = np.where(symbols == a)[0]
            ib = np.where(symbols == b)[0]
            sub = d[np.ix_(ia, ib)].ravel()
            sub = sub[np.isfinite(sub) & (sub < r_cut)]
            hist, _ = np.histogram(sub, bins=n_bins, range=(0.0, r_cut))
            hist = hist.astype(float)
            if hist.sum() > 0:
                hist /= hist.sum()
            vec.append(hist)
    if not vec:
        vec = [np.zeros(n_bins)]

    comp = np.array([np.sum(symbols == s) / len(symbols) for s in species])
    dens = np.array([len(atoms) / max(atoms.get_volume(), 1e-6)])
    return np.concatenate(vec + [comp, dens])


def cluster_families(records, eps=0.06):
    """
    Greedy clustering within each bucket. Two frames closer than eps in
    descriptor space belong to the same family and must not be split across
    train and test.
    """
    by_bucket = defaultdict(list)
    for i, r in enumerate(records):
        by_bucket[r["bucket"]].append(i)

    families = [None] * len(records)
    next_id = 0

    for bucket, idxs in by_bucket.items():
        # pad descriptors to a common length within the bucket
        dims = max(len(records[i]["descriptor"]) for i in idxs)
        vecs = {}
        for i in idxs:
            v = records[i]["descriptor"]
            vecs[i] = np.pad(v, (0, dims - len(v)))

        centers = []
        for i in idxs:
            v = vecs[i]
            placed = False
            for cid, cvec in centers:
                if np.linalg.norm(v - cvec) < eps:
                    families[i] = cid
                    placed = True
                    break
            if not placed:
                centers.append((next_id, v))
                families[i] = next_id
                next_id += 1

    return families


def grouped_stratified_split(records, test_fraction, rng):
    """
    Assign whole families to train or test, targeting test_fraction within
    each bucket so every structure type is represented on both sides.
    """
    by_bucket = defaultdict(list)
    for i, r in enumerate(records):
        by_bucket[r["bucket"]].append(i)

    split = ["train"] * len(records)
    for bucket, idxs in by_bucket.items():
        fam_members = defaultdict(list)
        for i in idxs:
            fam_members[records[i]["family"]].append(i)

        fams = list(fam_members)
        rng.shuffle(fams)
        target = test_fraction * len(idxs)

        n_test = 0
        for fam in fams:
            if n_test >= target:
                break
            # never send the only family in a bucket to test
            if len(fams) == 1:
                break
            for i in fam_members[fam]:
                split[i] = "test"
            n_test += len(fam_members[fam])
    return split


def load_split_manifest(path: str) -> dict:
    """folder -> 'train'/'test' from a previous round's split_manifest.csv."""
    carry = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            carry[row["folder"]] = row["split"]
    return carry


def carry_and_split(records, test_fraction, rng, carry_map):
    """
    Like grouped_stratified_split, but any folder present in carry_map keeps
    its previous-round label instead of being re-shuffled, and any family
    that contains a carried folder is pinned IN FULL to that folder's split
    -- so a new round's near-duplicate of an old test structure can't leak
    into train, or vice versa. Only families with no carried members (i.e.
    genuinely new this round) are split fresh, stratified per bucket among
    themselves, targeting test_fraction of the NEW data.

    This keeps every previous round's train/test membership frozen, which is
    what "test.xyz FIXED for the campaign" requires once you start
    accumulating rounds -- re-running grouped_stratified_split on the whole
    combined pool would reassign old folders too, since the shuffle order
    shifts as soon as the record list grows.
    """
    split = [None] * len(records)
    family_pin = {}

    for i, r in enumerate(records):
        prev = carry_map.get(r["folder"])
        if prev is not None:
            split[i] = prev
            family_pin.setdefault(r["family"], prev)

    for i, r in enumerate(records):
        if split[i] is None and r["family"] in family_pin:
            split[i] = family_pin[r["family"]]

    remaining = [i for i, s in enumerate(split) if s is None]
    if remaining:
        sub_records = [records[i] for i in remaining]
        sub_split = grouped_stratified_split(sub_records, test_fraction, rng)
        for i, s in zip(remaining, sub_split):
            split[i] = s

    n_carried = sum(1 for s in split if s is not None) - len(remaining)
    print(f"carried {n_carried} folders' split from previous round(s); "
          f"{len(remaining)} folders in newly-seen families were split "
          f"fresh")
    return split


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Build NEP train/test xyz from round-0 VASP jobs.")
    p.add_argument("--paths", default="round0_out/round0_1_vasp_job_paths.txt")
    p.add_argument("--outdir", default="round0_out/dataset")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--family-eps", type=float, default=0.06,
                   help="descriptor distance below which two frames are "
                        "treated as the same parent structure")
    p.add_argument("--split-mode", default="family",
                   choices=("family", "random"),
                   help="'family' avoids leakage; 'random' reproduces the "
                        "naive split for comparison only")
    p.add_argument("--carry-split-from", default=None,
                   help="split_manifest.csv from a previous round; folders "
                        "found in it keep their old train/test label "
                        "(and their whole family is pinned with them) so "
                        "the test set stays fixed as rounds accumulate. "
                        "Requires --split-mode family. Omit for round 0.")
    p.add_argument("--all-frames", action="store_true",
                   help="read every ionic step (use when NSW > 0)")
    p.add_argument("--write-stress", action="store_true",
                   help="also write a stress field alongside virial")
    p.add_argument("--max-force", type=float, default=500.0,
                   help="reject frames whose largest force exceeds this "
                        "(eV/A); short dimers legitimately reach ~100")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    if args.carry_split_from and args.split_mode != "family":
        print("! --carry-split-from has no effect with --split-mode random; "
              "ignoring it", file=sys.stderr)
        args.carry_split_from = None

    if not os.path.isfile(args.paths):
        print(f"path list not found: {args.paths}", file=sys.stderr)
        return 1
    with open(args.paths) as fh:
        folders = [ln.strip() for ln in fh if ln.strip()]
    print(f"{len(folders)} folders listed in {args.paths}")

    records, rejected = [], []

    for folder in folders:
        if not os.path.isdir(folder):
            rejected.append((folder, "directory missing"))
            continue

        nelm = parse_nelm(folder)
        ok, why = scf_converged(folder, nelm)
        if not ok:
            rejected.append((folder, why))
            continue

        try:
            index = ":" if args.all_frames else -1
            frames = read(os.path.join(folder, "OUTCAR"), index=index)
            if not isinstance(frames, list):
                frames = [frames]
        except Exception as exc:
            rejected.append((folder, f"OUTCAR unreadable: {exc}"))
            continue

        info = load_provenance(folder)
        for atoms in frames:
            try:
                new, energy, forces, virial = attach_nep_fields(
                    atoms, info, folder, args.write_stress)
            except Exception as exc:
                rejected.append((folder, f"missing energy/forces: {exc}"))
                continue

            fmax = float(np.abs(forces).max()) if len(forces) else 0.0
            if not np.isfinite(energy) or not np.isfinite(fmax):
                rejected.append((folder, "non-finite energy or force"))
                continue
            if fmax > args.max_force:
                rejected.append((folder, f"|F|max = {fmax:.1f} eV/A"))
                continue

            records.append({
                "folder": folder,
                "bucket": str(info.get("bucket", "unknown")),
                "kind": str(info.get("kind", "unknown")),
                "atoms": new,
                "natoms": len(new),
                "energy": float(energy),
                "epa": float(energy) / max(1, len(new)),
                "fmax": fmax,
                "has_virial": virial is not None,
                "descriptor": descriptor(new),
            })

    if not records:
        print("no usable structures; check rejected.csv", file=sys.stderr)
        with open(os.path.join(args.outdir, "rejected.csv"), "w",
                  newline="") as fh:
            csv.writer(fh).writerows([("folder", "reason")] + rejected)
        return 1

    print(f"{len(records)} usable structures, {len(rejected)} rejected")

    n_no_virial = sum(1 for r in records if not r["has_virial"])
    if n_no_virial:
        print(f"! {n_no_virial} structures have no stress tensor; NEP will "
              f"train on energy and forces only for those", file=sys.stderr)

    # ---- families and split ----
    if args.split_mode == "family":
        fams = cluster_families(records, eps=args.family_eps)
        for r, f in zip(records, fams):
            r["family"] = f
        n_fam = len(set(fams))
        print(f"grouped into {n_fam} structure families "
              f"(mean {len(records) / n_fam:.1f} frames each)")
        if args.carry_split_from:
            carry_map = load_split_manifest(args.carry_split_from)
            split = carry_and_split(records, args.test_fraction, rng,
                                     carry_map)
        else:
            split = grouped_stratified_split(records, args.test_fraction,
                                              rng)
    else:
        for i, r in enumerate(records):
            r["family"] = i
        idx = rng.permutation(len(records))
        n_test = int(round(args.test_fraction * len(records)))
        split = ["train"] * len(records)
        for i in idx[:n_test]:
            split[i] = "test"
        print("! using random frame-level split; test RMSE will be "
              "optimistic", file=sys.stderr)

    for r, s in zip(records, split):
        r["split"] = s

    train = [r["atoms"] for r in records if r["split"] == "train"]
    test = [r["atoms"] for r in records if r["split"] == "test"]

    write(os.path.join(args.outdir, "train.xyz"), train, format="extxyz")
    write(os.path.join(args.outdir, "test.xyz"), test, format="extxyz")

    # ---- manifests and report ----
    with open(os.path.join(args.outdir, "split_manifest.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["folder", "bucket", "kind", "family", "split",
                    "natoms", "energy_eV", "energy_per_atom", "fmax_eV_per_A"])
        for r in records:
            w.writerow([r["folder"], r["bucket"], r["kind"], r["family"],
                        r["split"], r["natoms"], f"{r['energy']:.6f}",
                        f"{r['epa']:.6f}", f"{r['fmax']:.3f}"])

    with open(os.path.join(args.outdir, "rejected.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["folder", "reason"])
        w.writerows(rejected)

    lines = []
    lines.append(f"round-0 dataset: {len(train)} train / {len(test)} test "
                 f"({len(records)} total, {len(rejected)} rejected)")
    lines.append(f"split mode: {args.split_mode}")
    lines.append("")
    lines.append(f"{'bucket':<20}{'train':>7}{'test':>7}{'families':>10}"
                 f"{'eV/atom min':>14}{'eV/atom max':>14}")
    by_bucket = defaultdict(list)
    for r in records:
        by_bucket[r["bucket"]].append(r)
    for bucket in sorted(by_bucket):
        g = by_bucket[bucket]
        ntr = sum(1 for r in g if r["split"] == "train")
        nte = len(g) - ntr
        nfam = len({r["family"] for r in g})
        epa = [r["epa"] for r in g]
        lines.append(f"{bucket:<20}{ntr:>7}{nte:>7}{nfam:>10}"
                     f"{min(epa):>14.3f}{max(epa):>14.3f}")

    lines.append("")
    if rejected:
        lines.append("rejection reasons:")
        for reason, count in Counter(
                r[1].split("(")[0].strip() for r in rejected).most_common():
            lines.append(f"  {count:5d}  {reason}")

    report = "\n".join(lines)
    with open(os.path.join(args.outdir, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    print("\n" + report)
    print(f"\nwrote {args.outdir}/train.xyz and {args.outdir}/test.xyz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
