#!/usr/bin/env python3
"""
Round 2, step 3: turn the LAMMPS trajectories of step 2 into VASP folders.

The MD was driven by an incomplete NEP, so not every frame is worth (or even
safe for) a DFT single point. Per run, in time order:

  1. read      every frame of every dump (positions, velocities, NEP forces,
               NEP per-atom energies) and, if the committee rerun ran, the
               second NEP's forces and energies for the same frame
  2. screen    contact ratio = min over pairs of d / (r_cov,i + r_cov,j)
                 below --keep-ratio      frame not used (too close for a
                                         sensible DFT point)
                 below --collapse-ratio  the NEP has collapsed two atoms:
                                         this frame and EVERY LATER frame of
                                         the run are discarded ("poisoned")
               NEP max |F| above --fmax-nep  frame not used
               A collision run is not poisoned by a close contact during the
               impact itself; it is if the contact survives into relaxation.
  3. flag      the --pre-failure frames just before a poisoned frame or a
               `fix halt` stop. The NEP was still producing plausible
               structures there but was about to go wrong: always selected.
  4. committee disagreement = max over atoms |F_nep - F_committee| (eV/A).
               Frames at or above --dev-lo are "uncertain": the two potentials
               were trained on overlapping data and still disagree, so the
               data does not pin the answer down. They fill the budget first;
               "confident" frames only top it up to --confident-fill of it.
  5. clean     (slab and collision frames) atoms and molecules more than
               --detach A from the slab are removed: free atoms only cost
               VASP time, and with ISPIN = 1 a lone N or Al atom gets the
               wrong spin state. The cell gets --vacuum A along z.
  6. select    farthest-point sampling in a structural descriptor, per bucket
               and stage, with a per-run cap so no single run dominates.
                 amorphous_bulk/slab  pair-distance histograms per element
                                      pair + composition, whole cell
                 collision            the same, centred only on the atoms the
                                      impacts changed (projectiles and target
                                      atoms displaced > --active-disp A)
               --reference-xyz seeds the bulk/slab sampling with an existing
               training set, so frames like data already there come last.
  7. write     VASP folders with generate_round1.py's own writer (same ENCUT
               rule, k-point density, dipole correction and ISPIN), then
               NCORE/KPAR for --vasp-ntasks MPI ranks.

Outputs
    <outdir>/<bucket>_<index>/       POSCAR KPOINTS POTCAR INCAR info.json
    <outdir>/selected_nep.extxyz     selected frames with the NEP energy and
                                     forces of the MD (atom order = POSCAR)
    <outdir>/candidates.csv          every frame read, and why it was or was
                                     not selected
    <outdir>/harvest_report.txt      per-run status, where the NEP failed,
                                     committee disagreement per bucket
    --path-list                      folder list for the VASP array

Usage (from the project root):
    python round2_code/round2_3_harvest.py \\
        --reference-xyz round1wL_out/dataset/train.xyz

Requires numpy and ase.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict

import numpy as np
from ase import Atoms
from ase.data import atomic_masses
from ase.io import read, write

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROUND1_CODE = os.path.join(os.path.dirname(CODE_DIR), "round1_code")
sys.path.insert(0, CODE_DIR)
from round2_common import (contact_ratio, contiguous_along_z,  # noqa: E402
                           pair_distances, remove_detached, set_vasp_parallel,
                           with_vacuum)

KB_EV = 8.617333262e-5
MV2_TO_EV = 1.0 / 9648.533212          # amu (A/ps)^2 -> eV

QUENCH_STAGES = ("melt", "quench", "anneal", "slab")
BUCKET_OF_STAGE = {"melt": "amorphous_bulk", "quench": "amorphous_bulk",
                   "anneal": "amorphous_bulk", "slab": "amorphous_slab",
                   "impact": "collision", "relax": "collision"}


# --------------------------------------------------------------------------
# LAMMPS output
# --------------------------------------------------------------------------


def read_dump(path):
    """Yield (step, cell 3x3, origin, pbc, columns dict) for each complete
    frame of a `dump custom` file; a frame cut off by a crash ends it."""
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            try:
                step = int(fh.readline())
                fh.readline()
                n = int(fh.readline())
                hdr = fh.readline().split()
                bounds = [[float(x) for x in fh.readline().split()] for _ in range(3)]
                cols = fh.readline().split()[2:]
            except (ValueError, IndexError):
                return
            rows = [fh.readline() for _ in range(n)]
            if n and (not rows[-1].strip() or len(rows[-1].split()) != len(cols)):
                return
            data = np.array([r.split() for r in rows], dtype=float).reshape(n, len(cols))
            flags = hdr[-3:]
            pbc = [f == "pp" for f in flags]
            if "xy" in hdr:
                (xlb, xhb, xy), (ylb, yhb, xz), (zlo, zhi, yz) = bounds
                xlo = xlb - min(0.0, xy, xz, xy + xz)
                xhi = xhb - max(0.0, xy, xz, xy + xz)
                ylo = ylb - min(0.0, yz)
                yhi = yhb - max(0.0, yz)
            else:
                (xlo, xhi), (ylo, yhi), (zlo, zhi) = [b[:2] for b in bounds]
                xy = xz = yz = 0.0
            cell = np.array([[xhi - xlo, 0.0, 0.0],
                             [xy, yhi - ylo, 0.0],
                             [xz, yz, zhi - zlo]])
            origin = np.array([xlo, ylo, zlo])
            yield step, cell, origin, pbc, {c: data[:, i] for i, c in enumerate(cols)}


def read_committee(path):
    """{step: (ids, forces, per-atom energies)} from a committee rerun dump."""
    out = {}
    if not os.path.isfile(path):
        return out
    for step, _, _, _, cols in read_dump(path):
        out[step] = (cols["id"].astype(int),
                     np.stack([cols["fx"], cols["fy"], cols["fz"]], 1),
                     cols.get("c_pea"))
    return out


def parse_log(path):
    """Status, markers and the halt step from log.lammps."""
    out = {"status": "not run", "halt_step": None, "error": "",
           "impacts": {}, "relax": {}, "stages": {}}
    if not os.path.isfile(path):
        return out
    out["status"] = "incomplete"
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("MD_DONE"):
                out["status"] = "done"
            elif line.startswith("MD_IMPACT"):
                t = line.split()
                out["impacts"][int(t[1])] = int(t[3])
            elif line.startswith("MD_RELAX"):
                t = line.split()
                out["relax"][int(t[1])] = int(t[3])
            elif line.startswith("MD_STAGE"):
                t = line.split()
                out["stages"][t[1]] = int(t[3])
            elif "Fix halt condition" in line:
                m = re.search(r"on step (\d+)", line)
                out["status"] = "halted"
                out["halt_step"] = int(m.group(1)) if m else None
                out["error"] = line.strip()
            elif line.startswith("ERROR") and out["status"] != "halted":
                out["status"] = "error"
                out["error"] = line.strip()
    return out


# --------------------------------------------------------------------------
# descriptors
# --------------------------------------------------------------------------


def pair_index(elements):
    idx, k = {}, 0
    for a in range(len(elements)):
        for b in range(a, len(elements)):
            idx[(a, b)] = idx[(b, a)] = k
            k += 1
    return idx, k


def descriptor(atoms, elements, centres=None, rc=6.0, nbins=24, w_comp=4.0,
               pairs=None):
    """Pair-distance histograms per element pair, counted per centre atom,
    plus composition. With `centres`, only pairs that start on those atoms
    are counted (local descriptor of the active region)."""
    eidx = {e: k for k, e in enumerate(elements)}
    pidx, npair = pair_index(elements)
    t = np.array([eidx[s] for s in atoms.get_chemical_symbols()])
    i, j, d = pairs if pairs is not None else pair_distances(atoms, rc)
    keep = d < rc
    i, j, d = i[keep], j[keep], d[keep]
    if centres is not None:
        mask = np.isin(i, centres)
        i, j, d = i[mask], j[mask], d[mask]
        ncen = max(1, len(centres))
        comp_atoms = t[centres] if len(centres) else t
    else:
        ncen = len(atoms)
        comp_atoms = t
    h = np.zeros((npair, nbins))
    if len(d):
        lookup = np.zeros((len(elements), len(elements)), int)
        for (a, b), k in pidx.items():
            lookup[a, b] = k
        p = lookup[t[i], t[j]]
        b = np.minimum((d / rc * nbins).astype(int), nbins - 1)
        np.add.at(h, (p, b), 1.0)
    comp = np.bincount(comp_atoms, minlength=len(elements)) / max(1, len(comp_atoms))
    return np.concatenate([h.ravel() / ncen, w_comp * comp])


def kinetic_temperature(atoms, vel):
    m = np.array([atomic_masses[z] for z in atoms.numbers])
    moving = np.any(vel != 0.0, axis=1)
    if moving.sum() < 2:
        return 0.0
    ke = 0.5 * (m[moving, None] * vel[moving] ** 2).sum() * MV2_TO_EV
    return float(2.0 * ke / (3.0 * moving.sum() * KB_EV))


# --------------------------------------------------------------------------
# farthest-point sampling
# --------------------------------------------------------------------------


def min_dist_to(X, Y, chunk=2048):
    """For each row of X, the distance to its nearest row of Y."""
    if Y is None or len(Y) == 0:
        return np.full(len(X), np.inf)
    out = np.empty(len(X))
    y2 = (Y * Y).sum(1)
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        d2 = (x * x).sum(1)[:, None] + y2[None, :] - 2.0 * x @ Y.T
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def fps(X, k, runs, run_cap, forced=(), seed_ref=None):
    """Greedy farthest-point selection of k rows of X, starting from the
    `forced` rows and (optionally) a reference set; at most run_cap rows per run."""
    n = len(X)
    chosen = []
    if n == 0 or k <= 0:
        return chosen
    taken = Counter()
    dist = min_dist_to(X, seed_ref)
    blocked = np.zeros(n, bool)
    _, code = np.unique(np.asarray(runs), return_inverse=True)

    def take(i):
        chosen.append(i)
        taken[code[i]] += 1
        blocked[i] = True
        if taken[code[i]] >= run_cap:
            blocked[code == code[i]] = True
        d = np.sqrt(((X - X[i]) ** 2).sum(1))
        np.minimum(dist, d, out=dist)

    for i in forced:
        if len(chosen) < k and not blocked[i]:
            take(i)
    while len(chosen) < k:
        score = np.where(blocked, -1.0, dist)
        i = int(np.argmax(score))
        if score[i] < 0:
            break
        take(i)
    return chosen


# --------------------------------------------------------------------------
# per-run harvesting
# --------------------------------------------------------------------------


def run_dumps(run_dir, kind):
    if kind == "quench":
        return [(s, os.path.join(run_dir, f"dump.{s}.lammpstrj")) for s in QUENCH_STAGES
                if os.path.isfile(os.path.join(run_dir, f"dump.{s}.lammpstrj"))]
    p = os.path.join(run_dir, "dump.collision.lammpstrj")
    return [("collision", p)] if os.path.isfile(p) else []


def frame_atoms(cell, origin, pbc, cols, elements):
    types = cols["type"].astype(int)
    symbols = [elements[t - 1] for t in types]
    pos = np.stack([cols["x"], cols["y"], cols["z"]], 1) - origin
    atoms = Atoms(symbols, positions=pos, cell=cell, pbc=pbc)
    atoms.arrays["md_id"] = cols["id"].astype(int)
    vel = np.stack([cols["vx"], cols["vy"], cols["vz"]], 1) if "vx" in cols else \
        np.zeros_like(pos)
    # collision dumps store the plain NEP forces as f_fstore (before setforce
    # and langevin); melt-quench dumps have no force-changing fix and use fx
    if "f_fstore[1]" in cols:
        frc = np.stack([cols["f_fstore[1]"], cols["f_fstore[2]"], cols["f_fstore[3]"]], 1)
    elif "fx" in cols:
        frc = np.stack([cols["fx"], cols["fy"], cols["fz"]], 1)
    else:
        frc = None
    pea = cols.get("c_pea")
    return atoms, vel, frc, pea


def committee_deviation(ids, frc, pea, com):
    """max |dF| (eV/A) and |dE|/atom (eV) over the atoms both dumps hold.
    The rerun drops atoms the MD dumped just outside a non-periodic box top;
    those are > rc above the surface and do not affect the others."""
    cid, cf, cpe = com
    common, ia, ib = np.intersect1d(ids, cid, return_indices=True)
    if len(common) == 0 or frc is None:
        return None, None
    dev_f = float(np.sqrt(((frc[ia] - cf[ib]) ** 2).sum(1)).max())
    dev_e = None
    if pea is not None and cpe is not None:
        dev_e = float(abs(pea[ia].sum() - cpe[ib].sum()) / len(common))
    return dev_f, dev_e


def harvest_run(run_dir, args, cands, run_rows):
    with open(os.path.join(run_dir, "run.json")) as fh:
        meta = json.load(fh)
    elements = meta["elements"]
    kind = meta["kind"]
    log = parse_log(os.path.join(run_dir, "log.lammps"))
    keep_ratio = args.keep_ratio_quench if kind == "quench" else args.keep_ratio_collision
    fmax_cap = args.fmax_nep_quench if kind == "quench" else args.fmax_nep_collision

    impact_steps = sorted(log["impacts"].items())       # [(k, step)]
    relax_steps = log["relax"]
    n_target = meta.get("n_target_atoms", 0)
    # atom ids of each impact's projectile (create_atoms numbers them in order)
    proj_ids, last = {}, n_target
    for imp in meta.get("impacts", []):
        proj_ids[imp["impact"]] = np.arange(last + 1, last + imp["n_atoms"] + 1)
        last += imp["n_atoms"]

    row = {"run": meta["run"], "kind": kind, "status": log["status"],
           "frames": 0, "screened_out": 0, "poisoned": 0, "poison_at": "",
           "halt_step": log["halt_step"] or "", "error": log["error"],
           "committee": "no"}
    frames = []                 # time-ordered metadata of every frame
    ref_pos = None
    poisoned = False

    for stage, path in run_dumps(run_dir, kind):
        com = read_committee(path.replace(".lammpstrj", ".committee.lammpstrj"))
        if com:
            row["committee"] = "yes"
        for fidx, (step, cell, origin, pbc, cols) in enumerate(read_dump(path)):
            atoms, vel, frc, pea = frame_atoms(cell, origin, pbc, cols, elements)
            row["frames"] += 1
            sub = stage
            impact = None
            if kind == "collision":
                for k, s0 in impact_steps:
                    if step >= s0:
                        impact = k
                if impact is None:
                    sub = "relax"           # before the first projectile
                else:
                    sub = "impact" if step < relax_steps.get(impact, 1 << 62) else "relax"
                if ref_pos is None:
                    ids0 = atoms.arrays["md_id"]
                    ref_pos = np.full((ids0.max() + 1, 3), np.nan)
                    ref_pos[ids0] = atoms.positions

            rec = {"run": meta["run"], "run_dir": run_dir, "kind": kind,
                   "stage": sub, "dump": path, "frame": fidx, "step": step,
                   "impact": impact, "natoms": len(atoms), "reason": "",
                   "pre_failure": False, "dev_f": None, "dev_e": None}
            if poisoned:
                rec["reason"] = "after_failure"
                row["poisoned"] += 1
                frames.append(rec)
                continue

            pairs = pair_distances(atoms, max(args.desc_rc, 2.6))
            close = pairs[2] < 2.6
            ratio, pair = contact_ratio(atoms, tuple(x[close] for x in pairs))
            rec["contact_ratio"] = round(ratio, 4)
            rec["closest_pair"] = pair
            fmax = float(np.sqrt((frc ** 2).sum(1)).max()) if frc is not None else 0.0
            rec["nep_fmax"] = round(fmax, 3)
            rec["nep_energy"] = float(pea.sum()) if pea is not None else None
            rec["T_K"] = round(kinetic_temperature(atoms, vel), 1)
            if step in com:
                dev_f, dev_e = committee_deviation(atoms.arrays["md_id"], frc,
                                                   pea, com[step])
                rec["dev_f"] = None if dev_f is None else round(dev_f, 4)
                rec["dev_e"] = None if dev_e is None else round(dev_e, 5)

            collapse = ratio < args.collapse_ratio and not (kind == "collision"
                                                            and sub == "impact")
            if collapse:
                poisoned = True
                rec["reason"] = "collapse"
                row["poison_at"] = f"{stage} step {step} ({pair} A, ratio {ratio:.2f})"
                frames.append(rec)
                continue
            if ratio < keep_ratio:
                rec["reason"] = "too_close"
            elif fmax > fmax_cap:
                rec["reason"] = "nep_force"
            if rec["reason"]:
                row["screened_out"] += 1
                frames.append(rec)
                continue

            # descriptor of the usable frame
            if kind == "collision":
                ids = atoms.arrays["md_id"]
                known = ids < len(ref_pos)
                dv = np.zeros((len(atoms), 3))
                dv[known] = atoms.positions[known] - ref_pos[ids[known]]
                dv[:, :2] -= np.round(dv[:, :2] @ np.linalg.inv(cell[:2, :2])) @ cell[:2, :2]
                disp = np.nan_to_num(np.linalg.norm(dv, axis=1))
                # atoms with nobody within --detach A: the incoming projectile
                # before it arrives, or atoms reflected/sputtered away
                near = np.zeros(len(atoms), bool)
                near[pairs[0][pairs[2] < args.detach]] = True
                incoming = np.isin(ids, proj_ids.get(impact, []))
                if sub == "impact" and incoming.any() and not near[incoming].any():
                    rec["reason"] = "approach"
                    frames.append(rec)
                    continue
                active = np.where(((ids > n_target) | (disp > args.active_disp)) & near)[0]
                if len(active) == 0:
                    rec["reason"] = "nothing_happened"
                    frames.append(rec)
                    continue
                rec["n_active"] = int(len(active))
                rec["desc"] = descriptor(atoms, elements, centres=active,
                                         rc=args.desc_rc, nbins=args.desc_bins,
                                         pairs=pairs)
            else:
                rec["desc"] = descriptor(atoms, elements, rc=args.desc_rc,
                                         nbins=args.desc_bins, pairs=pairs)
            frames.append(rec)

    # frames just before the failure (poison or fix halt) are always wanted
    fail_idx = next((k for k, r in enumerate(frames) if r["reason"] == "collapse"), None)
    if fail_idx is None and log["status"] == "halted":
        fail_idx = len(frames)
    if fail_idx is not None:
        good = [k for k in range(fail_idx) if not frames[k]["reason"]]
        for k in good[-args.pre_failure:]:
            frames[k]["pre_failure"] = True

    for r in frames:
        r["bucket"] = BUCKET_OF_STAGE.get(r["stage"], "collision")
    cands.extend(frames)
    run_rows.append(row)
    return meta


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def select_stratum(cands, idx, budget, args, ref):
    """Indices (into cands) chosen for one bucket/stage."""
    X = np.array([cands[i]["desc"] for i in idx])
    runs = [cands[i]["run"] for i in idx]
    cap = max(1, int(math.ceil(args.run_cap_factor * budget / len(set(runs)))))
    dev = [cands[i]["dev_f"] for i in idx]
    # no committee number -> treated as uncertain (selection by structure only)
    unc = [k for k, d in enumerate(dev) if d is None or d >= args.dev_lo]
    con = [k for k, d in enumerate(dev) if d is not None and d < args.dev_lo]
    forced = [k for k in unc + con if cands[idx[k]]["pre_failure"]]
    forced = forced[:max(1, budget // 5)]
    forced_set = set(forced)

    picks = []
    pool = forced + [k for k in unc if k not in forced_set]
    sub = fps(X[pool], min(budget, len(pool)), [runs[k] for k in pool], cap,
              forced=list(range(len(forced))), seed_ref=ref)
    picks += [pool[k] for k in sub]
    room = min(budget - len(picks), int(round(args.confident_fill * budget)))
    con = [k for k in con if k not in forced_set]
    if room > 0 and con:
        seed = X[picks] if picks else None
        if ref is not None:
            seed = ref if seed is None else np.vstack([seed, ref])
        sub = fps(X[con], min(room, len(con)), [runs[k] for k in con], cap,
                  seed_ref=seed)
        picks += [con[k] for k in sub]
    n_unc = sum(1 for k in picks if dev[k] is None or dev[k] >= args.dev_lo)
    return [idx[k] for k in picks], cap, len(forced), n_unc


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def load_reference(path, elements, max_frames, rng, args):
    frames = read(path, index=":")
    if len(frames) > max_frames:
        frames = [frames[i] for i in sorted(rng.choice(len(frames), max_frames,
                                                        replace=False))]
    out = []
    for a in frames:
        if not set(a.get_chemical_symbols()) <= set(elements):
            continue
        out.append(descriptor(a, elements, rc=args.desc_rc, nbins=args.desc_bins))
    return np.array(out) if out else None


def load_frames(recs):
    """{(dump, frame): (cell, origin, pbc, cols)} for the selected records,
    reading each dump file once."""
    want = defaultdict(set)
    for r in recs:
        want[r["dump"]].add(r["frame"])
    out = {}
    for path, idx in want.items():
        last = max(idx)
        for k, (step, cell, origin, pbc, cols) in enumerate(read_dump(path)):
            if k in idx:
                out[(path, k)] = (cell, origin, pbc, cols)
            if k >= last:
                break
    return out


def materialise(rec, meta, args, frame):
    """Turn the selected frame into a tagged Atoms."""
    elements = meta["elements"]
    if frame is None:
        raise RuntimeError(f"frame {rec['frame']} of {rec['dump']} vanished")
    cell, origin, pbc, cols = frame
    atoms, vel, frc, pea = frame_atoms(cell, origin, pbc, cols, elements)
    atoms.arrays["nep_forces"] = frc if frc is not None else np.zeros((len(atoms), 3))
    n_removed = 0
    if rec["bucket"] in ("amorphous_slab", "collision"):
        if atoms.pbc[2]:
            atoms = contiguous_along_z(atoms)
        atoms.pbc = (True, True, False)
        atoms, n_removed = remove_detached(atoms, args.detach)
        atoms = with_vacuum(atoms, args.vacuum)
    else:
        atoms.pbc = (True, True, True)
        atoms.wrap()
    atoms.arrays.pop("md_id", None)

    info = {"bucket": rec["bucket"], "kind": rec["stage"],
            "md_run": rec["run"], "md_step": rec["step"],
            "md_T_K": rec.get("T_K"), "pre_failure": rec["pre_failure"],
            "contact_ratio": rec.get("contact_ratio"),
            "nep_fmax": rec.get("nep_fmax"),
            "committee_dev_f": rec.get("dev_f"),
            "committee_dev_e": rec.get("dev_e"),
            "n_detached_removed": n_removed,
            "nep_model_sha256": args.nep_sha}
    # NEP energy of exactly this cell is only known when nothing was removed
    if rec.get("nep_energy") is not None and n_removed == 0:
        info["nep_energy"] = rec["nep_energy"]
    if meta["kind"] == "quench":
        info.update({"md_system": meta["system"],
                     "md_density_g_cm3": meta["density_g_cm3"],
                     "md_quench_rate_K_ps": meta["quench_rate_K_per_ps"]})
    else:
        info.update({"md_target": meta["target"], "md_energy_eV": meta["energy_eV"],
                     "md_theta_deg": meta["theta_deg"], "md_T_sub_K": meta["T_sub_K"],
                     "md_precursor": bool(meta.get("precursor", False)),
                     "md_impact": rec["impact"] or 0,
                     "md_projectile": (meta["impacts"][rec["impact"] - 1]["projectile"]
                                       if rec["impact"] else "")})
    atoms.info = {k: v for k, v in info.items() if v is not None}
    atoms.info["natoms"] = len(atoms)
    return atoms


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Select DFT candidates from the round-2 LAMMPS runs and "
                    "write them as VASP folders.")
    p.add_argument("--run-list", default="round2_out/round2_1_md_run_paths.txt")
    p.add_argument("--outdir", default="round2_out/vasp_md")
    p.add_argument("--path-list", default="round2_out/round2_3_md_vasp_paths.txt")
    p.add_argument("--force", action="store_true",
                   help="replace an existing --outdir")
    p.add_argument("--dry-run", action="store_true",
                   help="screen, select and report, but write no VASP folders")
    p.add_argument("--seed", type=int, default=7)

    s = p.add_argument_group("screening")
    s.add_argument("--keep-ratio-quench", type=float, default=0.60)
    s.add_argument("--keep-ratio-collision", type=float, default=0.50)
    s.add_argument("--collapse-ratio", type=float, default=0.40)
    s.add_argument("--fmax-nep-quench", type=float, default=50.0, help="eV/A")
    s.add_argument("--fmax-nep-collision", type=float, default=300.0, help="eV/A")
    s.add_argument("--pre-failure", type=int, default=3)
    s.add_argument("--detach", type=float, default=4.0,
                   help="remove atoms/molecules farther than this from the slab (A)")
    s.add_argument("--vacuum", type=float, default=15.0,
                   help="total vacuum of the VASP slab cells (A)")
    s.add_argument("--max-atoms", type=int, default=200)

    sel = p.add_argument_group("selection")
    sel.add_argument("--n-amorphous-bulk", type=int, default=300)
    sel.add_argument("--n-amorphous-slab", type=int, default=150)
    sel.add_argument("--n-collision", type=int, default=450)
    sel.add_argument("--bulk-stage-quota", nargs=3, type=float,
                     default=[0.3, 0.5, 0.2], metavar=("MELT", "QUENCH", "ANNEAL"))
    sel.add_argument("--impact-quota", type=float, default=0.6,
                     help="share of the collision budget for impact frames "
                          "(the rest: relaxation frames)")
    sel.add_argument("--dev-lo", type=float, default=0.3,
                     help="committee: frames with max |dF| >= this (eV/A) are "
                          "uncertain and fill the budget first")
    sel.add_argument("--confident-fill", type=float, default=0.25,
                     help="committee: at most this share of a budget goes to "
                          "frames below --dev-lo")
    sel.add_argument("--run-cap-factor", type=float, default=3.0,
                     help="per-run cap = factor x budget / runs")
    sel.add_argument("--active-disp", type=float, default=1.0,
                     help="a target atom is 'active' once it has moved this far (A)")
    sel.add_argument("--desc-rc", type=float, default=6.0)
    sel.add_argument("--desc-bins", type=int, default=24)
    sel.add_argument("--reference-xyz", default=None,
                     help="existing training set (extxyz); seeds the bulk/slab "
                          "selection so new frames are picked away from it")
    sel.add_argument("--reference-max", type=int, default=4000)

    v = p.add_argument_group("VASP (same settings as rounds 0/1)")
    v.add_argument("--potcar-path", default="aaa-potential")
    v.add_argument("--encut", type=float, default=None)
    v.add_argument("--encut-factor", type=float, default=1.3)
    v.add_argument("--kspacing", type=float, default=0.25)
    v.add_argument("--vasp-ntasks", type=int, default=16,
                   help="MPI ranks per VASP job; sets NCORE/KPAR")
    v.add_argument("--ncore", type=int, default=4)
    v.add_argument("--no-dipole", action="store_true")
    args = p.parse_args(argv)

    sys.path.insert(0, ROUND1_CODE)
    import generate_round1 as g1       # noqa: E402  (VASP writer of round 1)

    # fail now, not after an hour of harvesting
    if not args.dry_run:
        if args.potcar_path and not os.path.isdir(args.potcar_path):
            sys.exit(f"POTCAR directory {args.potcar_path}/ not found (expected "
                     f"<dir>/<element>/POTCAR). Run from the project root, set "
                     f"--potcar-path, or use --dry-run to only screen and select.")
        if glob.glob(os.path.join(args.outdir, "*", "POSCAR")) and not args.force:
            sys.exit(f"{args.outdir} already holds VASP folders; --force replaces "
                     f"them (do not do that once VASP has run)")

    rng = np.random.default_rng(args.seed)
    with open(args.run_list) as fh:
        run_dirs = [ln.strip() for ln in fh if ln.strip()]
    nep_src = os.path.join(os.path.dirname(os.path.dirname(run_dirs[0].rstrip("/"))),
                           "nep_source.json")
    args.nep_sha = ""
    if os.path.isfile(nep_src):
        with open(nep_src) as fh:
            args.nep_sha = json.load(fh).get("sha256", "")

    cands, run_rows, metas = [], [], {}
    for k, rd in enumerate(run_dirs, 1):
        rd = rd.rstrip("/")
        if not os.path.isfile(os.path.join(rd, "run.json")):
            print(f"  ! {rd}: no run.json, skipped", file=sys.stderr)
            continue
        meta = harvest_run(rd, args, cands, run_rows)
        metas[meta["run"]] = meta
        r = run_rows[-1]
        print(f"  [{k:3d}/{len(run_dirs)}] {meta['run']:34s} {r['status']:10s} "
              f"{r['frames']:5d} frames  committee {r['committee']:3s}"
              + (f"  FAILED at {r['poison_at']}" if r["poison_at"] else "")
              + (f"  halted step {r['halt_step']}" if r["halt_step"] else ""))
    if not cands:
        sys.exit("no frames found -- have the LAMMPS runs finished?")

    elements = metas[next(iter(metas))]["elements"]
    ref = None
    if args.reference_xyz:
        ref = load_reference(args.reference_xyz, elements, args.reference_max, rng, args)
        print(f"reference: {0 if ref is None else len(ref)} frames from {args.reference_xyz}")

    # ---- selection, per bucket and stratum ---------------------------------
    usable = [i for i, r in enumerate(cands) if not r["reason"]]
    qm, qq, qa = args.bulk_stage_quota
    strata = [("amorphous_bulk", "melt", args.n_amorphous_bulk * qm),
              ("amorphous_bulk", "quench", args.n_amorphous_bulk * qq),
              ("amorphous_bulk", "anneal", args.n_amorphous_bulk * qa),
              ("amorphous_slab", "slab", args.n_amorphous_slab),
              ("collision", "impact", args.n_collision * args.impact_quota),
              ("collision", "relax", args.n_collision * (1 - args.impact_quota))]
    selected = []
    for bucket, stage, budget in strata:
        idx = [i for i in usable if cands[i]["bucket"] == bucket and cands[i]["stage"] == stage]
        budget = int(round(budget))
        if not idx or budget <= 0:
            continue
        seed_ref = ref if (ref is not None and bucket != "collision") else None
        pick, cap, n_forced, n_unc = select_stratum(cands, idx, budget, args, seed_ref)
        for i in pick:
            cands[i]["selected"] = True
        selected += pick
        n_unc_all = sum(1 for i in idx if cands[i]["dev_f"] is None
                        or cands[i]["dev_f"] >= args.dev_lo)
        print(f"  {bucket:15s} {stage:7s} {len(pick):4d} of {len(idx):6d} usable "
              f"({n_unc_all} uncertain) from {len(set(cands[i]['run'] for i in idx))} "
              f"runs; picked {n_unc} uncertain, {n_forced} pre-failure, cap {cap}/run")

    # ---- materialise -------------------------------------------------------
    frames = []
    raw = load_frames([cands[i] for i in selected])
    for i in selected:
        a = materialise(cands[i], metas[cands[i]["run"]], args,
                        raw.get((cands[i]["dump"], cands[i]["frame"])))
        if len(a) > args.max_atoms:
            cands[i]["reason"] = "over_max_atoms"
            cands[i]["selected"] = False
            continue
        frames.append(g1.sorted_for_vasp(a))
    order = {"amorphous_bulk": 0, "amorphous_slab": 1, "collision": 2}
    frames.sort(key=lambda a: order[a.info["bucket"]])

    # ---- report ------------------------------------------------------------
    report_dir = args.outdir if not args.dry_run else (os.path.dirname(args.path_list) or ".")
    os.makedirs(report_dir, exist_ok=True)
    write_report(os.path.join(report_dir, "harvest_report.txt"), run_rows, cands,
                 frames, args)
    with open(os.path.join(report_dir, "candidates.csv"), "w", newline="") as fh:
        cols = ["run", "kind", "stage", "bucket", "impact", "step", "natoms",
                "T_K", "contact_ratio", "closest_pair", "nep_fmax", "dev_f",
                "dev_e", "reason", "pre_failure", "selected", "dump", "frame"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in cands:
            w.writerow({**r, "selected": r.get("selected", False)})
    if args.dry_run:
        print(f"\ndry run: {len(frames)} frames would be written; "
              f"report in {report_dir}/harvest_report.txt")
        return 0

    # ---- VASP folders ------------------------------------------------------
    for f in glob.glob(os.path.join(args.outdir, "*", "POSCAR")):
        shutil.rmtree(os.path.dirname(f))           # only reached with --force
    potcar_root = args.potcar_path or None
    encut = args.encut
    present = sorted({s for a in frames for s in a.get_chemical_symbols()})
    if potcar_root:
        enmax, auto = g1.survey_potcars(potcar_root, present, args.encut_factor)
        if encut is None:
            encut = auto
        print(f"POTCARs from {potcar_root}/: " +
              ", ".join(f"{k} ENMAX {v:.1f}" for k, v in sorted(enmax.items())) +
              f"  ->  ENCUT {encut} eV")
    elif encut is None:
        print("! no --potcar-path and no --encut: POSCAR/KPOINTS only", file=sys.stderr)
    records = g1.write_vasp_tree(frames, args.outdir, kspacing=args.kspacing,
                                 potcar_root=potcar_root, encut=encut,
                                 ncore=args.ncore, kpar=1,
                                 dipole=not args.no_dipole)
    kpars = Counter()
    if encut:
        for rec in records:
            kpars[set_vasp_parallel(os.path.join(args.outdir, rec[0]),
                                    args.vasp_ntasks, args.ncore)[1]] += 1
    for a, rec in zip(frames, records):
        a.info["folder"] = rec[0]
    # NEP forces stay under "nep_forces": a "forces" column would be read back
    # by ASE as if it were a DFT label
    write(os.path.join(args.outdir, "selected_nep.extxyz"), frames, format="extxyz")
    with open(args.path_list, "w") as fh:
        for rec in records:
            fh.write(f"./{os.path.join(args.outdir, rec[0])}\n"
                     if not os.path.isabs(args.outdir) else
                     os.path.join(args.outdir, rec[0]) + "\n")
    by_bucket = Counter(a.info["bucket"] for a in frames)
    print(f"\nwrote {len(records)} VASP folders under {args.outdir}/  "
          + "  ".join(f"{b} {n}" for b, n in sorted(by_bucket.items())))
    if kpars:
        print(f"INCAR for {args.vasp_ntasks} ranks: NCORE {args.ncore}, KPAR "
              + ", ".join(f"{k} ({n} folders)" for k, n in sorted(kpars.items())))
    print(f"folder list : {args.path_list}")
    print(f"report      : {args.outdir}/harvest_report.txt")
    return 0


def write_report(path, run_rows, cands, frames, args):
    L = ["round-2 MD harvest", ""]
    L.append(f"{'run':40s} {'status':10s} {'frames':>6s} {'screen':>6s} "
             f"{'poison':>6s} {'picked':>6s} {'comm':>4s}  failure")
    picked = Counter(r["run"] for r in cands if r.get("selected"))
    for r in run_rows:
        fail = r["poison_at"] or (f"fix halt at step {r['halt_step']}"
                                  if r["halt_step"] else r["error"][:60])
        L.append(f"{r['run'][:40]:40s} {r['status']:10s} {r['frames']:6d} "
                 f"{r['screened_out']:6d} {r['poisoned']:6d} {picked[r['run']]:6d} "
                 f"{r['committee']:>4s}  {fail}")
    L.append("")
    reasons = Counter(r["reason"] or "usable" for r in cands)
    L.append("frames by screening result: " +
             "  ".join(f"{k}={v}" for k, v in reasons.most_common()))
    L.append("")
    L.append(f"committee disagreement max |dF| (eV/A) of usable frames; "
             f"uncertain >= {args.dev_lo}")
    L.append(f"  {'bucket':16s} {'stage':8s} {'n':>6s} {'p10':>7s} {'p50':>7s} "
             f"{'p90':>7s} {'max':>7s} {'uncert':>7s}")
    groups = defaultdict(list)
    for r in cands:
        if not r["reason"] and r["dev_f"] is not None:
            groups[(r["bucket"], r["stage"])].append(r["dev_f"])
    for key in sorted(groups):
        d = np.array(groups[key])
        L.append(f"  {key[0]:16s} {key[1]:8s} {len(d):6d} {np.percentile(d, 10):7.3f} "
                 f"{np.percentile(d, 50):7.3f} {np.percentile(d, 90):7.3f} "
                 f"{d.max():7.3f} {np.mean(d >= args.dev_lo):7.1%}")
    if not groups:
        L.append("  (no committee dumps found)")
    L.append("")
    L.append("selected per bucket / stage:")
    for (b, s), n in sorted(Counter((a.info["bucket"], a.info["kind"]) for a in frames).items()):
        L.append(f"  {b:16s} {s:8s} {n:5d}")
    n_pre = sum(1 for a in frames if a.info.get("pre_failure"))
    L.append(f"  pre-failure frames among them: {n_pre}")
    failed = [r for r in run_rows if r["poison_at"] or r["status"] in ("halted", "error")]
    L.append("")
    L.append(f"runs where the NEP failed: {len(failed)} of {len(run_rows)}")
    by_sys = defaultdict(list)
    for r in failed:
        by_sys[r["run"].split("_d")[0].split("_E")[0]].append(r["run"])
    for k, v in sorted(by_sys.items()):
        L.append(f"  {k:24s} {len(v)} run(s)")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L[-(len(by_sys) + 2):]))


if __name__ == "__main__":
    raise SystemExit(main())
