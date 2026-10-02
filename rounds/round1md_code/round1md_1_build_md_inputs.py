#!/usr/bin/env python3
"""
Round 1md, step 1: LAMMPS runs that let the current (incomplete) NEP generate
new training structures, instead of building them by hand.

Two kinds of run, one folder each:

  quench      bulk melt-quench of an amorphous composition, then (optional)
              cleave the quenched cell and anneal it as a slab with vacuum.
              Input: in.melt_quench.lmp
  collision   a sequence of single-projectile impacts (Al, N, N2, Ar, Si) on
              Si(111), AlN(0001), AlN(000-1) or a quenched amorphous slab.
              Input: in.collision.lmp

Every dumped frame is only a CANDIDATE. round1md_3_harvest.py throws out
frames where the NEP has clearly gone wrong, picks a diverse subset, and
writes VASP folders in the same layout as generate_round1.py.

The NEP decides where MD goes, so the frames land where this potential
actually takes deposition MD -- including the places where it is wrong.
That is the point of running it while the potential is still incomplete.

Run folder layout (one per MD run):
    <outdir>/nep.txt                   copy of the potential every run uses
    <outdir>/quench/<name>/            in.lmp params.lmp start.data run.json
    <outdir>/collision/<name>/         in.lmp params.lmp start.data run.json impacts/
    round1md_out/round1md_1_md_run_paths.txt    every run folder under <outdir>

Atom types follow the element order of nep.txt, so `pair_coeff * * nep.txt
<elements>` is the same line for every run.

Existing run folders are never overwritten (use --force), so the script can
be run again to ADD runs, e.g. collisions on the amorphous slabs once the
quench runs have finished:

    python round1md_code/round1md_1_build_md_inputs.py --only collision \\
        --targets none --amorphous-targets 'round1md_out/md/quench/*/slab_final.data'

Usage (from the project root, i.e. the folder holding round1md_code/):
    python round1md_code/round1md_1_build_md_inputs.py \\
        --nep round1wL_out/round1wL_nep.txt

Requires numpy and ase.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import itertools
import json
import math
import os
import shutil
import sys

import numpy as np
from ase import Atoms
from ase.build import bulk, diamond111, make_supercell
from ase.data import atomic_masses, atomic_numbers, covalent_radii
from ase.io import read, write

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
EV_PER_AMU_A2_PS2 = 9648.533212    # 1 eV/amu in (A/ps)^2: v = sqrt(2 E / m)

# Lattice parameters used for the crystalline collision targets (A).
# Same experimental values as generate_round1.py / aln_si_structures.py.
A_SI = 5.431
A_ALN, C_ALN, U_ALN = 3.111, 4.981, 0.382

# --------------------------------------------------------------------------
# amorphous compositions for the melt-quench
# --------------------------------------------------------------------------
# name: (formula unit, density g/cm^3 at --density-scale 1.0, T_melt K)
#
# The densities are rough amorphous/liquid values, NOT fitted numbers: a-AlN
# ~3.0 (wurtzite 3.26), a-Si ~2.3, a-Si3N4 ~3.0, liquid Al-Si ~2.4-2.5. The
# density scan (--density-scale) exists because the potential has to see a
# range of densities anyway, and because at fixed volume the MD cannot find
# the right one by itself. T_melt is chosen well above the melting point so
# the cell loses its memory of the random packing within the melt stage.
QUENCH_SYSTEMS = {
    "AlN":        ({"Al": 1, "N": 1},           3.00, 3500),
    "Al3N2":      ({"Al": 3, "N": 2},           2.90, 3000),   # Al-rich
    "Al2N3":      ({"Al": 2, "N": 3},           2.80, 3500),   # N-rich: N2 forms
    "Si":         ({"Si": 1},                   2.30, 2500),
    "Si3N4":      ({"Si": 3, "N": 4},           3.00, 3500),
    "SiN":        ({"Si": 1, "N": 1},           2.80, 3200),
    "Si2N":       ({"Si": 2, "N": 1},           2.60, 3000),
    "AlSi":       ({"Al": 1, "Si": 1},          2.45, 2200),
    "Al3Si":      ({"Al": 3, "Si": 1},          2.45, 2000),
    "AlSiN2":     ({"Al": 1, "Si": 1, "N": 2},  3.00, 3500),   # intermixed interface
    "AlN_Ar":     ({"Al": 16, "N": 16, "Ar": 1}, 2.95, 3500),  # Ar trapped in a-AlN
}

# --------------------------------------------------------------------------
# projectiles for the collisions
# --------------------------------------------------------------------------
# name: (atoms, default weight in the per-run species draw)
# N2 is a rigid, non-rotating molecule at its gas-phase bond length; its
# kinetic energy is the energy of the whole molecule.
PROJECTILES = {
    "Al": (["Al"], 0.30),
    "N":  (["N"], 0.25),
    "N2": (["N", "N"], 0.20),
    "Ar": (["Ar"], 0.25),
    "Si": (["Si"], 0.0),
}
N2_BOND = 1.098


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def read_nep_header(path: str) -> dict:
    """Element order, cutoffs and ZBL radii from the head of nep.txt."""
    with open(path) as fh:
        first = fh.readline().split()
        if len(first) < 3 or not first[0].startswith("nep"):
            raise ValueError(f"{path}: first line {' '.join(first)!r} is not "
                             f"a NEP header (nep4 / nep4_zbl ...)")
        n = int(first[1])
        info = {"model": first[0], "elements": first[2:2 + n],
                "rc_radial": None, "rc_angular": None, "zbl": None}
        for _ in range(8):
            tok = fh.readline().split()
            if not tok:
                continue
            if tok[0] == "cutoff":
                info["rc_radial"], info["rc_angular"] = float(tok[1]), float(tok[2])
            elif tok[0] == "zbl":
                info["zbl"] = [float(t) for t in tok[1:3]]
    if info["rc_radial"] is None:
        raise ValueError(f"{path}: no cutoff line in the header")
    return info


def run_seed(base: int, run: str) -> int:
    """Seed from the run name, so a run gets the same seed whichever other
    runs are built (or skipped) in the same call."""
    return int(hashlib.sha256(f"{base}:{run}".encode()).hexdigest()[:8], 16) + 1


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_data(path: str, atoms: Atoms, elements) -> Atoms:
    """LAMMPS data file with types in nep.txt order; returns what LAMMPS reads."""
    write(path, atoms, format="lammps-data", specorder=list(elements),
          masses=True, units="metal", atom_style="atomic")
    back = read(path, format="lammps-data", atom_style="atomic", units="metal",
                Z_of_type={i + 1: atomic_numbers[e] for i, e in enumerate(elements)})
    if len(back) != len(atoms):
        raise RuntimeError(f"{path}: wrote {len(atoms)} atoms, read back {len(back)}")
    return back


def lmp_var(name: str, value) -> str:
    if isinstance(value, str):
        return f'variable {name:<14s} string "{value}"'
    if isinstance(value, (int, np.integer)):
        return f"variable {name:<14s} equal {int(value)}"
    return f"variable {name:<14s} equal {float(value):.10g}"


def write_params(path: str, header: str, values: dict):
    with open(path, "w") as fh:
        fh.write(f"# {header}\n# written by round1md_1_build_md_inputs.py; "
                 f"read by in.lmp via `include params.lmp`\n")
        for k, v in values.items():
            fh.write(lmp_var(k, v) + "\n")


def steps(t_ps: float, dt_ps: float) -> int:
    return max(1, int(round(t_ps / dt_ps)))


def dump_every(n_steps: int, n_frames: int, minimum: int = 10) -> int:
    return max(minimum, n_steps // max(1, n_frames))


def layers_by_z(atoms: Atoms, tol: float = 0.2):
    """Atom indices grouped into z layers, bottom to top."""
    order = np.argsort(atoms.positions[:, 2])
    layers, cur = [], [order[0]]
    for i in order[1:]:
        if atoms.positions[i, 2] - atoms.positions[cur[-1], 2] > tol:
            layers.append(cur)
            cur = []
        cur.append(i)
    layers.append(cur)
    return layers


def coordination(atoms: Atoms, cutoff: float) -> np.ndarray:
    from ase.neighborlist import neighbor_list
    i = neighbor_list("i", atoms, cutoff)
    return np.bincount(i, minlength=len(atoms))


# --------------------------------------------------------------------------
# quench: random packing at a target density
# --------------------------------------------------------------------------


def random_packing(counts: dict, density: float, rng,
                   factors=(0.80, 0.70, 0.60, 0.50)) -> Atoms:
    """Cubic cell at `density` (g/cm^3) with atoms no closer than
    factor x (sum of covalent radii). The factor is lowered only if the
    packing cannot be completed; the relax stage of in.melt_quench.lmp
    removes whatever close contacts are left."""
    symbols = [s for s, n in counts.items() for _ in range(n)]
    mass = sum(atomic_masses[atomic_numbers[s]] for s in symbols)
    length = (mass * 1.66053906660 / density) ** (1.0 / 3.0)
    radii = np.array([covalent_radii[atomic_numbers[s]] for s in symbols])

    for factor in factors:
        order = rng.permutation(len(symbols))
        pos = np.zeros((len(symbols), 3))
        placed = []
        ok = True
        for i in order:
            for _ in range(4000):
                cand = rng.random(3) * length
                if placed:
                    d = pos[placed] - cand
                    d -= length * np.round(d / length)
                    r = np.sqrt((d * d).sum(axis=1))
                    if np.any(r < factor * (radii[placed] + radii[i])):
                        continue
                pos[i] = cand
                placed.append(i)
                break
            else:
                ok = False
                break
        if ok:
            atoms = Atoms(symbols, positions=pos, cell=[length] * 3, pbc=True)
            atoms.info["packing_factor"] = factor
            return atoms
    raise RuntimeError(f"could not pack {counts} at {density} g/cm^3")


def formula_units(unit: dict, target_atoms: int) -> dict:
    per = sum(unit.values())
    n = max(1, int(round(target_atoms / per)))
    return {s: k * n for s, k in unit.items()}


# --------------------------------------------------------------------------
# collision targets
# --------------------------------------------------------------------------


def trim_to_bilayers(slab: Atoms, n_bilayers: int) -> Atoms:
    """Keep n complete bilayers, so the top and bottom atoms each have ONE
    dangling bond (the bulk-terminated (111)/(0001) surface).

    The ASE builders cut through the bilayers instead: their outermost atom
    is held by a single bond and has three dangling bonds."""
    layers = layers_by_z(slab)
    z = [slab.positions[l[0], 2] for l in layers]
    start = next(i for i in range(len(layers) - 1) if z[i + 1] - z[i] < 1.0)
    keep = [i for l in layers[start:start + 2 * n_bilayers] for i in l]
    if len(layers) - start < 2 * n_bilayers:
        raise ValueError("not enough layers to cut the requested bilayers")
    out = slab[sorted(keep)]
    out.positions[:, 2] -= out.positions[:, 2].min()
    return out


def si111_target(repeat: int, n_bilayers: int) -> Atoms:
    slab = diamond111("Si", (repeat, repeat, 2 * n_bilayers + 2), a=A_SI,
                      vacuum=0.0)
    slab.pbc = (True, True, False)
    return trim_to_bilayers(slab, n_bilayers)


def aln_target(repeat: int, n_bilayers: int, top: str) -> Atoms:
    """AlN slab with `top` = 'Al' (Al-polar (0001), the usual growth face on
    Si(111)) or 'N' (N-polar (000-1))."""
    unit = bulk("AlN", "wurtzite", a=A_ALN, c=C_ALN, u=U_ALN)
    unit = make_supercell(unit, [[1, 0, 0], [1, 1, 0], [0, 0, 1]])  # 60 deg cell
    slab = unit.repeat((repeat, repeat, n_bilayers // 2 + 2))
    slab.pbc = (True, True, False)
    # ASE's wurtzite has the vertical Al->N bond pointing -z, i.e. N-polar
    # along +z. Mirror z for an Al-polar slab.
    sym = np.array(slab.get_chemical_symbols())
    al = np.where(sym == "Al")[0][0]
    d = slab.get_distances(al, np.where(sym == "N")[0], mic=True, vector=True)
    vertical = d[(np.linalg.norm(d[:, :2], axis=1) < 0.1)
                 & (np.linalg.norm(d, axis=1) < 2.1)]
    al_polar_up = bool(len(vertical)) and vertical[0][2] > 0
    if al_polar_up != (top == "Al"):
        slab.positions[:, 2] *= -1.0
    slab.positions[:, 2] -= slab.positions[:, 2].min()
    return trim_to_bilayers(slab, n_bilayers)


def check_surface(slab: Atoms, cutoff: float, want_top=None):
    s = slab.copy()
    s.pbc = (True, True, False)
    cn = coordination(s, cutoff)
    z = s.positions[:, 2]
    top = z > z.max() - 0.2
    bot = z < z.min() + 0.2
    sym = np.array(s.get_chemical_symbols())
    if set(cn[top]) != {3} or set(cn[bot]) != {3}:
        raise RuntimeError(f"surface coordination top {set(cn[top])} "
                           f"bottom {set(cn[bot])}, expected 3")
    if want_top and set(sym[top]) != {want_top}:
        raise RuntimeError(f"top layer is {set(sym[top])}, expected {want_top}")


def contiguous_along_z(atoms: Atoms) -> Atoms:
    """Shift atoms across the periodic z boundary so the slab is in one piece
    and the vacuum sits on top (the largest gap in z becomes the vacuum)."""
    out = atoms.copy()
    cz = float(out.cell[2, 2])
    z = np.mod(out.positions[:, 2], cz)
    order = np.sort(z)
    gaps = np.diff(np.concatenate([order, [order[0] + cz]]))
    k = int(np.argmax(gaps))
    z_cut = order[k] + 0.5 * gaps[k]          # middle of the vacuum
    z = np.where(z > z_cut, z - cz, z)
    out.positions[:, 2] = z - z.min()
    return out


def amorphous_target(path: str, elements) -> Atoms:
    atoms = read(path, format="lammps-data", atom_style="atomic", units="metal",
                 Z_of_type={i + 1: atomic_numbers[e] for i, e in enumerate(elements)})
    if abs(atoms.cell[2, 0]) > 1e-6 or abs(atoms.cell[2, 1]) > 1e-6:
        raise ValueError(f"{path}: tilted c axis, cannot use as a z slab")
    return contiguous_along_z(atoms)


def collision_cell(slab: Atoms, headroom: float) -> Atoms:
    """Slab sitting 1 A above z=0 with `headroom` of empty box above it.
    The box is non-periodic in z for the MD (boundary p p f)."""
    out = slab.copy()
    out.positions[:, 2] += 1.0 - out.positions[:, 2].min()
    cell = np.array(out.cell)
    cell[2] = [0.0, 0.0, out.positions[:, 2].max() + headroom]
    out.set_cell(cell, scale_atoms=False)
    out.pbc = (True, True, False)
    return out


# --------------------------------------------------------------------------
# collision impacts
# --------------------------------------------------------------------------


def impact_commands(k, n_total, proj, energy, theta_deg, phi_deg, aim_xy,
                    z_start, cell, elements, rng):
    """LAMMPS lines that create one projectile and set its velocity."""
    syms = PROJECTILES[proj][0]
    mass = sum(atomic_masses[atomic_numbers[s]] for s in syms)
    speed = math.sqrt(2.0 * energy * EV_PER_AMU_A2_PS2 / mass)
    th, ph = math.radians(theta_deg), math.radians(phi_deg)
    direction = np.array([math.sin(th) * math.cos(ph),
                          math.sin(th) * math.sin(ph), -math.cos(th)])
    v = speed * direction

    # start on the line that hits the aim point at the top of the slab
    path = z_start["height"] / math.cos(th)
    centre = np.array([aim_xy[0], aim_xy[1], z_start["surface"]]) - path * direction

    if len(syms) == 1:
        offsets = [np.zeros(3)]
    else:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        offsets = [0.5 * N2_BOND * axis, -0.5 * N2_BOND * axis]

    inv = np.linalg.inv(np.array(cell)[:2, :2])
    lines = [f"# impact {k} of {n_total}: {proj}  E = {energy:g} eV  "
             f"theta = {theta_deg:g} deg  phi = {phi_deg:.1f} deg  "
             f"aim = ({aim_xy[0]:.3f}, {aim_xy[1]:.3f})  "
             f"|v| = {speed:.3f} A/ps"]
    for sym, off in zip(syms, offsets):
        p = centre + off
        frac = np.mod(p[:2] @ inv, 1.0)                 # wrap into the cell
        xy = frac @ np.array(cell)[:2, :2]
        t = elements.index(sym) + 1
        lines += [
            f"create_atoms {t} single {xy[0]:.5f} {xy[1]:.5f} {p[2]:.5f} units box",
            f"region rnew sphere {xy[0]:.5f} {xy[1]:.5f} {p[2]:.5f} 0.3 side in units box",
            "group pnew region rnew",
            f"velocity pnew set {v[0]:.5f} {v[1]:.5f} {v[2]:.5f} sum no units box",
            "group pnew delete",
            "region rnew delete",
        ]
    return "\n".join(lines) + "\n", {
        "impact": k, "projectile": proj, "energy_eV": energy,
        "theta_deg": theta_deg, "phi_deg": round(phi_deg, 2),
        "aim_xy": [round(float(aim_xy[0]), 4), round(float(aim_xy[1]), 4)],
        "speed_A_per_ps": round(speed, 4), "n_atoms": len(syms)}


# --------------------------------------------------------------------------
# run builders
# --------------------------------------------------------------------------


def make_run_dir(path: str, force: bool) -> bool:
    if os.path.exists(path):
        if not force:
            return False
        shutil.rmtree(path)
    os.makedirs(path)
    return True


def build_quench_runs(args, nep, outdir, records):
    elements = nep["elements"]
    systems = args.quench_systems or list(QUENCH_SYSTEMS)
    for name in systems:
        if name not in QUENCH_SYSTEMS:
            sys.exit(f"unknown quench system {name!r}; known: {', '.join(QUENCH_SYSTEMS)}")
    usable = []
    for name in systems:
        missing = [e for e in QUENCH_SYSTEMS[name][0] if e not in elements]
        if missing:
            print(f"  skip {name}: {' '.join(missing)} not in this NEP "
                  f"({' '.join(elements)})")
        else:
            usable.append(name)
    made = skipped = 0
    for name, dscale, rate in itertools.product(usable, args.density_scale,
                                                args.quench_rate):
        unit, rho0, t_melt = QUENCH_SYSTEMS[name]
        counts = formula_units(unit, args.quench_atoms)
        rho = rho0 * dscale
        run = f"{name}_d{dscale:.2f}_q{rate:g}"
        path = os.path.join(outdir, "quench", run)
        if not make_run_dir(path, args.force):
            skipped += 1
            continue

        seed = run_seed(args.seed, run)
        r = np.random.default_rng(seed)
        atoms = random_packing(counts, rho, r)
        write_data(os.path.join(path, "start.data"), atoms, elements)

        dt = args.dt_n if "N" in unit else args.dt
        t_final = args.t_final
        n_quench = steps((t_melt - t_final) / rate, dt)
        p = {
            "nep_file": "../../nep.txt",
            "elements": " ".join(elements),
            "data_file": "start.data",
            "seed": seed % 900000 + 1,
            "dt": dt,
            "T_relax": 300.0,
            "T_melt": float(t_melt),
            "T_final": float(t_final),
            "T_slab": float(args.slab_temp),
            "n_relax": steps(args.t_relax, dt),
            "n_melt": steps(args.t_melt, dt),
            "n_quench": n_quench,
            "n_anneal": steps(args.t_anneal, dt),
            "n_slab_heat": steps(args.t_slab_heat, dt),
            "n_slab_hold": steps(args.t_slab_hold, dt),
            "n_slab_cool": steps(args.t_slab_cool, dt),
            "every_melt": dump_every(steps(args.t_melt, dt), args.frames_melt),
            "every_quench": dump_every(n_quench, args.frames_quench),
            "every_anneal": dump_every(steps(args.t_anneal, dt), args.frames_anneal),
            "every_slab": dump_every(steps(args.t_slab_heat + args.t_slab_hold
                                           + args.t_slab_cool, dt), args.frames_slab),
            "vacuum": float(args.vacuum),
            "zshift": float(r.random() * atoms.cell[2, 2]),
            "epa_floor": float(args.epa_floor),
            "T_cap": float(2.0 * t_melt),
            "d_halt": float(args.d_halt_quench),
        }
        write_params(os.path.join(path, "params.lmp"),
                     f"melt-quench {name}, {rho:.3f} g/cm^3, {rate:g} K/ps", p)
        shutil.copy(os.path.join(CODE_DIR, "in.melt_quench.lmp"),
                    os.path.join(path, "in.lmp"))
        info = {"kind": "quench", "run": run, "system": name,
                "composition": counts, "natoms": len(atoms),
                "density_g_cm3": round(rho, 4), "density_scale": dscale,
                "quench_rate_K_per_ps": rate, "T_melt_K": t_melt,
                "T_final_K": t_final, "dt_ps": dt, "vacuum_A": args.vacuum,
                "packing_factor": atoms.info["packing_factor"],
                "elements": elements, "params": p}
        with open(os.path.join(path, "run.json"), "w") as fh:
            json.dump(info, fh, indent=2)
        records.append([run, "quench", name, len(atoms), f"{rho:.3f}",
                        f"{rate:g}", "", "", "", path])
        made += 1
    print(f"  quench   : {made} new run folders" +
          (f", {skipped} already there (kept)" if skipped else ""))


def collision_targets(args, nep):
    elements = nep["elements"]
    targets = {}
    names = [] if args.targets == ["none"] else args.targets
    for name in names:
        if name == "Si111":
            if "Si" not in elements:
                continue
            slab = si111_target(args.si_repeat, args.si_bilayers)
            check_surface(slab, 2.6)
            targets[name] = (slab, 1.5)          # fix the bottom bilayer
        elif name in ("AlN0001", "AlN000-1"):
            if not {"Al", "N"} <= set(elements):
                continue
            top = "Al" if name == "AlN0001" else "N"
            slab = aln_target(args.aln_repeat, args.aln_bilayers, top)
            check_surface(slab, 2.1, want_top=top)
            targets[name] = (slab, 1.2)
        else:
            sys.exit(f"unknown target {name!r} (Si111, AlN0001, AlN000-1, none)")

    paths = []
    for pattern in args.amorphous_targets or []:
        paths += sorted(glob.glob(pattern))
    if args.amorphous_targets and not paths:
        print(f"  ! no files match {args.amorphous_targets}", file=sys.stderr)
    rng = np.random.default_rng(args.seed + 7)
    if len(paths) > args.max_amorphous_targets:
        paths = sorted(rng.choice(paths, args.max_amorphous_targets, replace=False))
    for p in paths:
        run = os.path.basename(os.path.dirname(os.path.abspath(p)))
        try:
            slab = amorphous_target(p, elements)
        except Exception as exc:
            print(f"  ! skip amorphous target {p}: {exc}", file=sys.stderr)
            continue
        targets[f"a-{run}"] = (slab, 2.0)
    return targets


def build_collision_runs(args, nep, outdir, records):
    elements = nep["elements"]
    rc = nep["rc_radial"]
    pool = []
    for name in args.projectiles:
        if name not in PROJECTILES:
            sys.exit(f"unknown projectile {name!r}; known: {', '.join(PROJECTILES)}")
        if all(s in elements for s in PROJECTILES[name][0]):
            pool.append(name)
        else:
            print(f"  skip projectile {name}: not in this NEP")
    weights = np.array([args.projectile_weights.get(p, PROJECTILES[p][1])
                        for p in pool], float)
    if not pool or weights.sum() <= 0:
        sys.exit("no usable projectile")
    weights /= weights.sum()

    targets = collision_targets(args, nep)
    made = skipped = 0
    for (tname, (slab0, fix_thick)), energy, angle, t_sub in itertools.product(
            targets.items(), args.energies, args.angles, args.substrate_temps):
        run = f"{tname}_E{energy:g}_A{angle:g}_T{t_sub:g}"
        path = os.path.join(outdir, "collision", run)
        if not make_run_dir(path, args.force):
            skipped += 1
            continue
        seed = run_seed(args.seed, run)
        r = np.random.default_rng(seed)

        z_top = slab0.positions[:, 2].max() - slab0.positions[:, 2].min() + 1.0
        # projectiles start beyond the cutoff, with room for the surface to grow
        z_start = z_top + rc + args.start_gap
        cell_atoms = collision_cell(slab0, z_start - z_top + 3.0)
        back = write_data(os.path.join(path, "start.data"), cell_atoms, elements)
        cell = np.array(back.cell)
        z_top = back.positions[:, 2].max()
        z_bot = back.positions[:, 2].min()

        # bath at most 30% of the slab, so thin (amorphous) slabs keep a free top
        bath = min(args.bath_thickness, 0.3 * (z_top - z_bot))

        has_n = "N" in back.get_chemical_symbols() or any(
            "N" in PROJECTILES[p][0] for p in pool)
        dtmax = args.dt_n if has_n else args.dt
        species = r.choice(pool, size=args.impacts, p=weights)
        os.makedirs(os.path.join(path, "impacts"))
        impacts = []
        for k, proj in enumerate(species, start=1):
            phi = float(r.uniform(0.0, 360.0))
            aim = r.random(2) @ cell[:2, :2]
            text, meta = impact_commands(
                k, args.impacts, str(proj), float(energy), float(angle), phi, aim,
                {"z": z_start, "surface": z_top, "height": z_start - z_top},
                cell, elements, r)
            with open(os.path.join(path, "impacts", f"impact_{k}.lmp"), "w") as fh:
                fh.write(text)
            impacts.append(meta)

        p = {
            "nep_file": "../../nep.txt",
            "elements": " ".join(elements),
            "data_file": "start.data",
            "seed": seed % 900000 + 1,
            "T_sub": float(t_sub),
            "z_fix": float(z_bot + fix_thick),
            "z_bath": float(z_bot + fix_thick + bath),
            "dtmax": dtmax,
            "xmax": float(args.xmax),
            "n_equil": steps(args.t_equil, dtmax),
            "n_impacts": args.impacts,
            "n_fast": args.n_fast,
            "every_fast": args.every_fast,
            "n_slow": steps(args.t_slow, dtmax),
            "every_slow": dump_every(steps(args.t_slow, dtmax), args.frames_slow),
            "epa_floor": float(args.epa_floor),
            "d_halt": float(args.d_halt_collision),
        }
        write_params(os.path.join(path, "params.lmp"),
                     f"collisions on {tname}: E = {energy:g} eV, "
                     f"theta = {angle:g} deg, T_sub = {t_sub:g} K", p)
        shutil.copy(os.path.join(CODE_DIR, "in.collision.lmp"),
                    os.path.join(path, "in.lmp"))
        info = {"kind": "collision", "run": run, "target": tname,
                "n_target_atoms": len(back), "energy_eV": energy,
                "theta_deg": angle, "T_sub_K": t_sub, "dtmax_ps": dtmax,
                "z_top_A": round(float(z_top), 4),
                "z_start_A": round(float(z_start), 4),
                "projectile_pool": {p_: round(float(w), 4) for p_, w in zip(pool, weights)},
                "impacts": impacts, "elements": elements, "params": p}
        with open(os.path.join(path, "run.json"), "w") as fh:
            json.dump(info, fh, indent=2)
        records.append([run, "collision", tname, len(back), "", "", energy,
                        angle, t_sub, path])
        made += 1
    print(f"  collision: {made} new run folders on {len(targets)} target(s)" +
          (f", {skipped} already there (kept)" if skipped else ""))
    for tname, (slab, _) in targets.items():
        print(f"             {tname:28s} {len(slab):4d} atoms  "
              f"{slab.get_chemical_formula()}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def parse_weights(items):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k] = float(v)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Build LAMMPS melt-quench and collision runs that use the "
                    "current NEP to generate new training structures.")
    p.add_argument("--nep", default="round1wL_out/round1wL_nep.txt",
                   help="the (incomplete) potential that drives the MD")
    p.add_argument("--outdir", default="round1md_out/md")
    p.add_argument("--elements", nargs="+", default=None,
                   help="LAMMPS atom types: a subset of the nep.txt elements, "
                        "kept in nep.txt order (default: all of them). Only "
                        "needed for a multi-element foundation model")
    p.add_argument("--run-list", default="round1md_out/round1md_1_md_run_paths.txt")
    p.add_argument("--only", choices=("all", "quench", "collision"), default="all")
    p.add_argument("--seed", type=int, default=101,
                   help="not 0: round 0 and round 1 both used seed 0")
    p.add_argument("--force", action="store_true",
                   help="rebuild run folders that already exist (deletes them)")
    p.add_argument("--dt", type=float, default=0.001,
                   help="time step (ps) for N-free systems")
    p.add_argument("--dt-n", type=float, default=0.0005,
                   help="time step (ps) when N is present (N2 vibrates every 14 fs)")
    p.add_argument("--epa-floor", type=float, default=-10.0,
                   help="stop a run when PE/atom drops below this (eV); the "
                        "lowest real value in N-Al-Si-Ar is N2 at about -8.3")

    q = p.add_argument_group("melt-quench")
    q.add_argument("--quench-systems", nargs="+", default=None,
                   help=f"subset of: {' '.join(QUENCH_SYSTEMS)}")
    q.add_argument("--quench-atoms", type=int, default=96)
    q.add_argument("--density-scale", type=float, nargs="+",
                   default=[0.92, 1.0, 1.08])
    q.add_argument("--quench-rate", type=float, nargs="+", default=[50.0, 10.0],
                   help="cooling rates, K/ps")
    q.add_argument("--t-final", type=float, default=300.0)
    q.add_argument("--t-relax", type=float, default=1.0, help="ps")
    q.add_argument("--t-melt", type=float, default=20.0, help="ps")
    q.add_argument("--t-anneal", type=float, default=10.0, help="ps")
    q.add_argument("--vacuum", type=float, default=12.0,
                   help="vacuum added after the quench (A); 0 = bulk only")
    q.add_argument("--slab-temp", type=float, default=900.0,
                   help="slab anneal temperature after cleaving (K)")
    q.add_argument("--t-slab-heat", type=float, default=5.0, help="ps")
    q.add_argument("--t-slab-hold", type=float, default=10.0, help="ps")
    q.add_argument("--t-slab-cool", type=float, default=6.0, help="ps")
    q.add_argument("--frames-melt", type=int, default=100)
    q.add_argument("--frames-quench", type=int, default=200)
    q.add_argument("--frames-anneal", type=int, default=50)
    q.add_argument("--frames-slab", type=int, default=150)
    q.add_argument("--d-halt-quench", type=float, default=0.6,
                   help="stop the run when any pair gets closer than this (A)")

    c = p.add_argument_group("collision")
    c.add_argument("--targets", nargs="+", default=["Si111", "AlN0001", "AlN000-1"],
                   help="crystalline targets (Si111 AlN0001 AlN000-1), or 'none'")
    c.add_argument("--amorphous-targets", nargs="+", default=None,
                   help="glob(s) of slab_final.data from finished quench runs")
    c.add_argument("--max-amorphous-targets", type=int, default=6)
    c.add_argument("--si-repeat", type=int, default=4)
    c.add_argument("--si-bilayers", type=int, default=4)
    c.add_argument("--aln-repeat", type=int, default=4)
    c.add_argument("--aln-bilayers", type=int, default=4)
    c.add_argument("--projectiles", nargs="+", default=["Al", "N", "N2", "Ar"])
    c.add_argument("--projectile-weights", nargs="+", default=None,
                   help="e.g. Al=0.4 N=0.2 N2=0.2 Ar=0.2 (default weights in "
                        "PROJECTILES)")
    c.add_argument("--energies", type=float, nargs="+",
                   default=[1, 5, 10, 20, 50, 100], help="eV")
    c.add_argument("--angles", type=float, nargs="+", default=[0, 45],
                   help="polar angle from the surface normal, deg")
    c.add_argument("--substrate-temps", type=float, nargs="+", default=[300, 900])
    c.add_argument("--impacts", type=int, default=8,
                   help="sequential impacts per run (the surface accumulates them)")
    c.add_argument("--start-gap", type=float, default=2.0,
                   help="projectiles start rc + this above the initial surface (A)")
    c.add_argument("--bath-thickness", type=float, default=4.0,
                   help="Langevin layer above the fixed bottom (A), capped at "
                        "30%% of the slab thickness")
    c.add_argument("--xmax", type=float, default=0.04,
                   help="fix dt/reset: max displacement per step (A)")
    c.add_argument("--t-equil", type=float, default=1.0, help="ps")
    c.add_argument("--n-fast", type=int, default=1500,
                   help="steps per impact with fine dumps (approach + impact)")
    c.add_argument("--every-fast", type=int, default=10)
    c.add_argument("--t-slow", type=float, default=2.0,
                   help="ps of relaxation after each impact (coarse dumps)")
    c.add_argument("--frames-slow", type=int, default=40)
    c.add_argument("--d-halt-collision", type=float, default=0.45,
                   help="stop the run when any pair gets closer than this (A); "
                        "lower than for the quench because 100 eV impacts "
                        "reach ~0.8 A")
    args = p.parse_args(argv)
    args.projectile_weights = parse_weights(args.projectile_weights)

    if not os.path.isfile(args.nep):
        sys.exit(f"NEP file not found: {args.nep}  (set --nep)")
    nep = read_nep_header(args.nep)
    if args.elements:
        unknown = [e for e in args.elements if e not in nep["elements"]]
        if unknown:
            sys.exit(f"--elements {' '.join(unknown)}: not in {args.nep}")
        # keep nep.txt's order, so types match the element line of the model
        nep["elements"] = [e for e in nep["elements"] if e in args.elements]
    print(f"NEP       : {args.nep}  ({nep['model']}, "
          f"{' '.join(nep['elements'])}, rc {nep['rc_radial']:g}/{nep['rc_angular']:g} A, "
          f"zbl {nep['zbl']})")

    os.makedirs(args.outdir, exist_ok=True)
    nep_copy = os.path.join(args.outdir, "nep.txt")
    if os.path.isfile(nep_copy) and sha256(nep_copy) != sha256(args.nep):
        if not args.force:
            sys.exit(f"{nep_copy} is a different potential from {args.nep}.\n"
                     f"Every run of a round must use one potential: pick a new "
                     f"--outdir, or --force to rebuild everything.")
    shutil.copy(args.nep, nep_copy)
    with open(os.path.join(args.outdir, "nep_source.json"), "w") as fh:
        json.dump({"source": os.path.abspath(args.nep), "sha256": sha256(args.nep),
                   **nep}, fh, indent=2)

    records = []
    if args.only in ("all", "quench"):
        build_quench_runs(args, nep, args.outdir, records)
    if args.only in ("all", "collision"):
        build_collision_runs(args, nep, args.outdir, records)

    if records:
        man = os.path.join(args.outdir, "manifest.csv")
        new = not os.path.isfile(man)
        with open(man, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["run", "kind", "system_or_target", "natoms",
                            "density_g_cm3", "quench_rate_K_ps", "energy_eV",
                            "theta_deg", "T_sub_K", "path"])
            w.writerows(records)

    # the run list covers EVERY run folder under outdir, finished or not; the
    # LAMMPS array skips folders that already have a DONE marker
    runs = sorted(os.path.dirname(f) for f in
                  glob.glob(os.path.join(args.outdir, "*", "*", "in.lmp")))
    runs = sorted(runs, key=lambda r: (os.path.basename(os.path.dirname(r)) != "quench", r))
    os.makedirs(os.path.dirname(args.run_list) or ".", exist_ok=True)
    with open(args.run_list, "w") as fh:
        for r in runs:
            fh.write(("./" + r if not os.path.isabs(r) and not r.startswith(".") else r) + "\n")
    print(f"\n{len(runs)} run folders listed in {args.run_list}")
    print(f"next: sbatch --array=0-{max(0, len(runs) - 1)}%50 "
          f"round1md_code/round1md_2_anvil_lammps.sbatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
