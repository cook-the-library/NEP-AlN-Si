#!/usr/bin/env python3
"""
Round 2, step 1 (MD part): LAMMPS runs that let the current, incomplete NEP
make new training structures. The thermally rattled crystals are made
separately, in Python, by round2_1_thermal_rattle.py.

Kinds of run, one folder each:

  quench      bulk melt-quench of an amorphous composition, then cleave the
              quenched cell and anneal it as a slab with vacuum.
              Input: in.melt_quench.lmp
  collision   sequential single-projectile impacts (Al, N, N2, Ar) at normal
              incidence, 1-50 eV, on Si(111), AlN(0001), AlN(000-1) and the
              AlN/Si(111) interface. Input: in.collision.lmp
  precursor   collision runs that BUILD a surface: Al pre-deposition on
              Si(111) and nitridation of Si(111) by atomic N. Their frames
              are harvested like any collision, and their final.data becomes
              a target in pass 2.

Pass 2 (--pass2, after the pass-1 runs have finished) adds collisions on the
surfaces pass 1 produced: the cleaved amorphous slabs (slab_final.data of the
quench runs) and the Al-covered and nitrided Si(111) (final.data of the
precursor runs).

With a committee potential (--committee, default the round-1.7 NEP), every
run folder also gets in.rerun.lmp: after the MD, the second NEP re-evaluates
every dumped frame, and the harvester prefers frames where the two disagree.

Run folder layout:
    <outdir>/nep.txt, nep_committee.txt   copies every run uses
    <outdir>/{quench,collision}/<run>/    in.lmp in.rerun.lmp params.lmp
                                          start.data run.json [impacts/]
    round2_out/round2_1_md_run_paths.txt  every run folder under <outdir>

Atom types follow the element order of nep.txt. Existing run folders are
never overwritten (--force rebuilds them).

Usage (from the project root, the folder holding round2_code/):
    python round2_code/round2_1_build_md_inputs.py            # pass 1
    python round2_code/round2_1_build_md_inputs.py --pass2    # after pass 1

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
from ase.data import atomic_masses, atomic_numbers, covalent_radii
from ase.io import read, write

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
import round2_common as rc  # noqa: E402

EV_PER_AMU_A2_PS2 = 9648.533212    # 1 eV/amu in (A/ps)^2: v = sqrt(2 E / m)

# --------------------------------------------------------------------------
# amorphous compositions for the melt-quench
# --------------------------------------------------------------------------
# name: (formula unit, density g/cm^3 at --density-scale 1.0, T_melt K)
#
# The densities are rough amorphous/liquid values, NOT fitted numbers: a-AlN
# ~3.0 (wurtzite 3.26), a-Si ~2.3, a-Si3N4 ~3.0, liquid Al-Si ~2.4-2.5. The
# density scan (--density-scale) exists because the potential has to see a
# range of densities anyway, and because at fixed volume the MD cannot find
# the right one by itself. T_melt is well above the melting point so the cell
# loses its memory of the random packing within the melt stage.
QUENCH_SYSTEMS = {
    "AlN":        ({"Al": 1, "N": 1},            3.00, 3500),
    "Al3N2":      ({"Al": 3, "N": 2},            2.90, 3000),   # Al-rich
    "Al2N3":      ({"Al": 2, "N": 3},            2.80, 3500),   # N-rich: N2 forms
    "Si":         ({"Si": 1},                    2.30, 2500),
    "Si3N4":      ({"Si": 3, "N": 4},            3.00, 3500),
    "SiN":        ({"Si": 1, "N": 1},            2.80, 3200),
    "Si2N":       ({"Si": 2, "N": 1},            2.60, 3000),
    "AlSi":       ({"Al": 1, "Si": 1},           2.45, 2200),
    "Al3Si":      ({"Al": 3, "Si": 1},           2.45, 2000),
    "AlSiN2":     ({"Al": 1, "Si": 1, "N": 2},   3.00, 3500),   # intermixed interface
    "AlN_Ar":     ({"Al": 16, "N": 16, "Ar": 1}, 2.95, 3500),   # Ar trapped in a-AlN
    "Si_Ar":      ({"Si": 32, "Ar": 1},          2.30, 2500),   # Ar-damaged Si surface
}

# --------------------------------------------------------------------------
# projectiles
# --------------------------------------------------------------------------
# name: (atoms, default weight in the per-run species draw)
# N2 is a rigid, non-rotating molecule at its gas-phase bond length; its
# kinetic energy is that of the whole molecule. Ar stands for the sputter gas
# (reflected neutrals and ions, treated as neutral atoms by the NEP).
PROJECTILES = {
    "Al": (["Al"], 0.30),
    "N":  (["N"], 0.25),
    "N2": (["N", "N"], 0.20),
    "Ar": (["Ar"], 0.25),
    "Si": (["Si"], 0.0),
}
N2_BOND = 1.098

# Pass-1 collision runs that build the pass-2 targets.
# name: (base target, projectile, number of impacts, energy eV, T_sub K)
# Al: ~1.5 ML on the 16 top sites of 4x4 Si(111), at sputtered-atom energy.
# N:  24 atomic N at plasma-radical energy, at the hot substrate temperature.
PRECURSORS = {
    "AlSi111": ("Si111", "Al", 24, 2.0, 300.0),
    "NSi111":  ("Si111", "N", 24, 5.0, 900.0),
}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


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
    back = read_data(path, elements)
    if len(back) != len(atoms):
        raise RuntimeError(f"{path}: wrote {len(atoms)} atoms, read back {len(back)}")
    return back


def read_data(path: str, elements) -> Atoms:
    return read(path, format="lammps-data", atom_style="atomic", units="metal",
                Z_of_type={i + 1: atomic_numbers[e] for i, e in enumerate(elements)})


def lmp_var(name: str, value) -> str:
    if isinstance(value, str):
        return f'variable {name:<14s} string "{value}"'
    if isinstance(value, (int, np.integer)):
        return f"variable {name:<14s} equal {int(value)}"
    return f"variable {name:<14s} equal {float(value):.10g}"


def write_params(path: str, header: str, values: dict):
    with open(path, "w") as fh:
        fh.write(f"# {header}\n# written by round2_1_build_md_inputs.py; "
                 f"read by in.lmp and in.rerun.lmp via `include params.lmp`\n")
        for k, v in values.items():
            fh.write(lmp_var(k, v) + "\n")


def steps(t_ps: float, dt_ps: float) -> int:
    return max(1, int(round(t_ps / dt_ps)))


def dump_every(n_steps: int, n_frames: int, minimum: int = 10) -> int:
    return max(minimum, n_steps // max(1, n_frames))


def make_run_dir(path: str, force: bool) -> bool:
    if os.path.exists(path):
        if not force:
            return False
        shutil.rmtree(path)
    os.makedirs(path)
    return True


def copy_inputs(path: str, template: str, committee: bool):
    shutil.copy(os.path.join(CODE_DIR, template), os.path.join(path, "in.lmp"))
    if committee:
        shutil.copy(os.path.join(CODE_DIR, "in.rerun.lmp"),
                    os.path.join(path, "in.rerun.lmp"))


def common_params(elements, committee: bool, boundary: str, stages: str) -> dict:
    return {"nep_file": "../../nep.txt",
            "committee_file": "../../nep_committee.txt" if committee else "none",
            "elements": " ".join(elements),
            "data_file": "start.data",
            "bnd": boundary,
            "rerun_stages": stages}


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
        stages = "melt quench anneal slab" if args.vacuum > 0 else "melt quench anneal"
        p = common_params(elements, args.committee_on, "p p p", stages)
        p.update({
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
        })
        write_params(os.path.join(path, "params.lmp"),
                     f"melt-quench {name}, {rho:.3f} g/cm^3, {rate:g} K/ps", p)
        copy_inputs(path, "in.melt_quench.lmp", args.committee_on)
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


# --------------------------------------------------------------------------
# collision targets
# --------------------------------------------------------------------------


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


def crystal_targets(args, elements):
    """{name: (slab, fixed thickness A)} built in Python."""
    targets = {}
    names = [] if args.targets == ["none"] else args.targets
    for name in names:
        if name == "Si111":
            if "Si" not in elements:
                continue
            slab = rc.si111_slab(args.si_repeat, args.si_bilayers)
            rc.check_surface(slab, 2.6)
            targets[name] = (slab, 1.5)          # fix the bottom bilayer
        elif name in ("AlN0001", "AlN000-1"):
            if not {"Al", "N"} <= set(elements):
                continue
            top = "Al" if name == "AlN0001" else "N"
            slab = rc.aln_slab(args.aln_repeat, args.aln_bilayers, top)
            rc.check_surface(slab, 2.1, want_top=top)
            targets[name] = (slab, 1.2)
        elif name == "AlN_Si111":
            if not {"Al", "N", "Si"} <= set(elements):
                continue
            slab = rc.aln_si_interface(si_bilayers=args.interface_si_bilayers,
                                       aln_bilayers=args.interface_aln_bilayers,
                                       termination=args.interface_termination,
                                       polarity="Al")
            targets[f"AlN_Si111_{args.interface_termination}term"] = (slab, 1.5)
        else:
            sys.exit(f"unknown target {name!r} (Si111, AlN0001, AlN000-1, "
                     f"AlN_Si111, none)")
    return targets


def md_target(path: str, elements, detach: float = 4.0) -> Atoms:
    """A surface left by an earlier run (slab_final.data of a quench,
    final.data of a precursor), in one piece, free atoms removed."""
    atoms = read_data(path, elements)
    if abs(atoms.cell[2, 0]) > 1e-6 or abs(atoms.cell[2, 1]) > 1e-6:
        raise ValueError(f"{path}: tilted c axis, cannot use as a z slab")
    atoms.pbc = (True, True, True)
    atoms = rc.contiguous_along_z(atoms)
    atoms.pbc = (True, True, False)
    atoms, _ = rc.remove_detached(atoms, detach)
    return atoms


def pass2_targets(args, elements):
    targets = {}
    rng = np.random.default_rng(args.seed + 7)
    slabs = sorted(glob.glob(os.path.join(args.outdir, "quench", "*", "slab_final.data")))
    if len(slabs) > args.max_amorphous_targets:
        slabs = sorted(rng.choice(slabs, args.max_amorphous_targets, replace=False))
    for p in slabs:
        run = os.path.basename(os.path.dirname(p))
        targets[f"a-{run}"] = (md_target(p, elements), 2.0)
    for name in PRECURSORS:
        finals = sorted(glob.glob(os.path.join(args.outdir, "collision",
                                               f"pre-{name}_*", "final.data")))
        for p in finals:
            targets[name] = (md_target(p, elements), 1.5)
    if not targets:
        sys.exit("--pass2: no slab_final.data or precursor final.data yet -- "
                 "have the pass-1 runs finished?")
    return targets


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


def build_one_collision(args, nep, outdir, run, tname, slab0, fix_thick,
                        energy, angle, t_sub, pool, weights, n_impacts,
                        records, precursor=False):
    elements = nep["elements"]
    path = os.path.join(outdir, "collision", run)
    if not make_run_dir(path, args.force):
        return False
    seed = run_seed(args.seed, run)
    r = np.random.default_rng(seed)

    height = slab0.positions[:, 2].max() - slab0.positions[:, 2].min()
    z_top = height + 1.0
    # projectiles start beyond the cutoff, with room for the surface to grow
    z_start = z_top + nep["rc_radial"] + args.start_gap
    back = write_data(os.path.join(path, "start.data"),
                      collision_cell(slab0, z_start - z_top + 3.0), elements)
    cell = np.array(back.cell)
    z_top = back.positions[:, 2].max()
    z_bot = back.positions[:, 2].min()
    # bath at most 30% of the slab, so thin slabs keep a free top
    bath = min(args.bath_thickness, 0.3 * (z_top - z_bot))

    has_n = "N" in back.get_chemical_symbols() or any(
        "N" in PROJECTILES[p][0] for p in pool)
    dtmax = args.dt_n if has_n else args.dt
    species = r.choice(pool, size=n_impacts, p=weights)
    os.makedirs(os.path.join(path, "impacts"))
    impacts = []
    for k, proj in enumerate(species, start=1):
        phi = float(r.uniform(0.0, 360.0))
        aim = r.random(2) @ cell[:2, :2]
        text, meta = impact_commands(
            k, n_impacts, str(proj), float(energy), float(angle), phi, aim,
            {"z": z_start, "surface": z_top, "height": z_start - z_top},
            cell, elements, r)
        with open(os.path.join(path, "impacts", f"impact_{k}.lmp"), "w") as fh:
            fh.write(text)
        impacts.append(meta)

    p = common_params(elements, args.committee_on, "p p f", "collision")
    p.update({
        "seed": seed % 900000 + 1,
        "T_sub": float(t_sub),
        "z_fix": float(z_bot + fix_thick),
        "z_bath": float(z_bot + fix_thick + bath),
        "dtmax": dtmax,
        "xmax": float(args.xmax),
        "n_equil": steps(args.t_equil, dtmax),
        "n_impacts": n_impacts,
        "n_fast": args.n_fast,
        "every_fast": args.every_fast,
        "n_slow": steps(args.t_slow, dtmax),
        "every_slow": dump_every(steps(args.t_slow, dtmax), args.frames_slow),
        "epa_floor": float(args.epa_floor),
        "d_halt": float(args.d_halt_collision),
    })
    write_params(os.path.join(path, "params.lmp"),
                 f"{'precursor' if precursor else 'collisions'} on {tname}: "
                 f"E = {energy:g} eV, theta = {angle:g} deg, T_sub = {t_sub:g} K", p)
    copy_inputs(path, "in.collision.lmp", args.committee_on)
    info = {"kind": "collision", "run": run, "target": tname,
            "precursor": precursor,
            "n_target_atoms": len(back), "energy_eV": energy,
            "theta_deg": angle, "T_sub_K": t_sub, "dtmax_ps": dtmax,
            "z_top_A": round(float(z_top), 4),
            "z_start_A": round(float(z_start), 4),
            "projectile_pool": {p_: round(float(w), 4) for p_, w in zip(pool, weights)},
            "impacts": impacts, "elements": elements, "params": p}
    with open(os.path.join(path, "run.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    records.append([run, "precursor" if precursor else "collision", tname,
                    len(back), "", "", energy, angle, t_sub, path])
    return True


def build_collision_runs(args, nep, outdir, records, targets):
    elements = nep["elements"]
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
    max_proj = max(len(PROJECTILES[p][0]) for p in pool)

    made = skipped = 0
    for (tname, (slab0, fix_thick)), energy, angle, t_sub in itertools.product(
            targets.items(), args.energies, args.angles, args.substrate_temps):
        n_imp = min(args.impacts, (args.max_atoms - len(slab0)) // max_proj)
        if n_imp < 1:
            print(f"  ! {tname}: {len(slab0)} atoms leave no room under "
                  f"--max-atoms {args.max_atoms}", file=sys.stderr)
            continue
        run = f"{tname}_E{energy:g}_A{angle:g}_T{t_sub:g}"
        if build_one_collision(args, nep, outdir, run, tname, slab0, fix_thick,
                               energy, angle, t_sub, pool, weights, n_imp, records):
            made += 1
        else:
            skipped += 1
    print(f"  collision: {made} new run folders on {len(targets)} target(s)" +
          (f", {skipped} already there (kept)" if skipped else ""))
    for tname, (slab, _) in targets.items():
        print(f"             {tname:30s} {len(slab):4d} atoms  "
              f"{slab.get_chemical_formula()}")


def build_precursor_runs(args, nep, outdir, records, crystal):
    made = skipped = 0
    for name, (base, proj, n, energy, t_sub) in PRECURSORS.items():
        if base not in crystal or not all(s in nep["elements"]
                                          for s in PROJECTILES[proj][0]):
            continue
        slab0, fix_thick = crystal[base]
        run = f"pre-{name}_E{energy:g}_A0_T{t_sub:g}"
        if build_one_collision(args, nep, outdir, run, f"{base} (builds {name})",
                               slab0, fix_thick, energy, 0.0, t_sub, [proj],
                               np.array([1.0]), n, records, precursor=True):
            made += 1
        else:
            skipped += 1
    print(f"  precursor: {made} new run folders (Al pre-deposition, nitridation)"
          + (f", {skipped} already there (kept)" if skipped else ""))


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
        description="Build LAMMPS melt-quench, collision and precursor runs "
                    "that use the current NEP to generate training structures.")
    p.add_argument("--nep", default="round1wL_out/round1wL_nep.txt",
                   help="the (incomplete) potential that drives the MD")
    p.add_argument("--committee", default="round1.7_out/round1.7_nep.txt",
                   help="second NEP with the same elements; re-evaluates every "
                        "dumped frame so the harvester can pick frames the two "
                        "disagree on. 'none' to switch off")
    p.add_argument("--elements", nargs="+", default=None,
                   help="LAMMPS atom types: a subset of the nep.txt elements, "
                        "kept in nep.txt order (default: all of them). Only "
                        "needed for a multi-element foundation model")
    p.add_argument("--outdir", default="round2_out/md")
    p.add_argument("--run-list", default="round2_out/round2_1_md_run_paths.txt")
    p.add_argument("--pass2", action="store_true",
                   help="add collisions on the surfaces pass 1 made (quench "
                        "slab_final.data, precursor final.data)")
    p.add_argument("--only", choices=("all", "quench", "collision"), default="all")
    p.add_argument("--seed", type=int, default=202)
    p.add_argument("--force", action="store_true",
                   help="rebuild run folders that already exist (deletes them)")
    p.add_argument("--max-atoms", type=int, default=200,
                   help="VASP size cap; limits the impacts per collision run")
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
                   help="vacuum opened after the quench (A); 0 = bulk only")
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
    c.add_argument("--targets", nargs="+",
                   default=["Si111", "AlN0001", "AlN000-1", "AlN_Si111"],
                   help="Python-built targets (Si111 AlN0001 AlN000-1 "
                        "AlN_Si111), or 'none'")
    c.add_argument("--no-precursors", action="store_true",
                   help="skip the Al pre-deposition and nitridation runs")
    c.add_argument("--max-amorphous-targets", type=int, default=6,
                   help="pass 2: quenched slabs used as targets")
    c.add_argument("--si-repeat", type=int, default=4)
    c.add_argument("--si-bilayers", type=int, default=4)
    c.add_argument("--aln-repeat", type=int, default=4)
    c.add_argument("--aln-bilayers", type=int, default=4)
    c.add_argument("--interface-termination", choices=("Al", "N"), default="Al",
                   help="AlN species bonded to Si (config: interface_termination)")
    c.add_argument("--interface-si-bilayers", type=int, default=2)
    c.add_argument("--interface-aln-bilayers", type=int, default=2)
    c.add_argument("--projectiles", nargs="+", default=["Al", "N", "N2", "Ar"])
    c.add_argument("--projectile-weights", nargs="+", default=None,
                   help="e.g. Al=0.4 N=0.2 N2=0.2 Ar=0.2 (default weights in "
                        "PROJECTILES)")
    c.add_argument("--energies", type=float, nargs="+", default=[1, 5, 10, 20, 50],
                   help="eV (PVD: up to 50 eV)")
    c.add_argument("--angles", type=float, nargs="+", default=[0],
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
                   help="stop the run when any pair gets closer than this (A)")
    args = p.parse_args(argv)
    args.projectile_weights = parse_weights(args.projectile_weights)

    if not os.path.isfile(args.nep):
        sys.exit(f"NEP file not found: {args.nep}  (set --nep)")
    nep = rc.read_nep_header(args.nep)
    if args.elements:
        unknown = [e for e in args.elements if e not in nep["elements"]]
        if unknown:
            sys.exit(f"--elements {' '.join(unknown)}: not in {args.nep}")
        # keep nep.txt's order, so types match the element line of the model
        nep["elements"] = [e for e in nep["elements"] if e in args.elements]
    print(f"NEP       : {args.nep}  ({nep['model']}, "
          f"{' '.join(nep['elements'])}, rc {nep['rc_radial']:g}/{nep['rc_angular']:g} A, "
          f"zbl {nep['zbl']})")

    args.committee_on = args.committee != "none"
    if args.committee_on and not os.path.isfile(args.committee):
        sys.exit(f"committee NEP not found: {args.committee}\n"
                 f"Set --committee to the round-1.7 nep.txt, or --committee "
                 f"none to select by structure alone.")
    os.makedirs(args.outdir, exist_ok=True)
    for src, dst in ((args.nep, "nep.txt"), (args.committee, "nep_committee.txt")):
        if src == "none":
            continue
        copy = os.path.join(args.outdir, dst)
        if os.path.isfile(copy) and sha256(copy) != sha256(src) and not args.force:
            sys.exit(f"{copy} is a different potential from {src}.\n"
                     f"Every run of a round must use the same potentials: pick a "
                     f"new --outdir, or --force to rebuild everything.")

    if args.committee_on:
        com = rc.read_nep_header(args.committee)
        missing = [e for e in nep["elements"] if e not in com["elements"]]
        if missing:
            sys.exit(f"committee NEP lacks {' '.join(missing)}")
        if sha256(args.committee) == sha256(args.nep):
            sys.exit("--committee is the same potential as --nep")
        print(f"committee : {args.committee}  ({' '.join(com['elements'])})")
        shutil.copy(args.committee, os.path.join(args.outdir, "nep_committee.txt"))
    shutil.copy(args.nep, os.path.join(args.outdir, "nep.txt"))
    with open(os.path.join(args.outdir, "nep_source.json"), "w") as fh:
        json.dump({"source": os.path.abspath(args.nep), "sha256": sha256(args.nep),
                   "committee": (os.path.abspath(args.committee)
                                 if args.committee_on else None),
                   "committee_sha256": (sha256(args.committee)
                                        if args.committee_on else None),
                   **nep}, fh, indent=2)

    records = []
    if args.pass2:
        build_collision_runs(args, nep, args.outdir, records,
                             pass2_targets(args, nep["elements"]))
    else:
        if args.only in ("all", "quench"):
            build_quench_runs(args, nep, args.outdir, records)
        if args.only in ("all", "collision"):
            crystal = crystal_targets(args, nep["elements"])
            build_collision_runs(args, nep, args.outdir, records, crystal)
            if not args.no_precursors:
                if "Si111" not in crystal and "Si" in nep["elements"]:
                    slab = rc.si111_slab(args.si_repeat, args.si_bilayers)
                    crystal["Si111"] = (slab, 1.5)
                build_precursor_runs(args, nep, args.outdir, records, crystal)

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
    # LAMMPS array skips folders that already have a DONE/HALTED marker
    runs = sorted(os.path.dirname(f) for f in
                  glob.glob(os.path.join(args.outdir, "*", "*", "in.lmp")))
    runs = sorted(runs, key=lambda r: (os.path.basename(os.path.dirname(r)) != "quench",
                                       "pre-" not in r, r))
    os.makedirs(os.path.dirname(args.run_list) or ".", exist_ok=True)
    with open(args.run_list, "w") as fh:
        for r in runs:
            fh.write(("./" + r if not os.path.isabs(r) and not r.startswith(".") else r) + "\n")
    print(f"\n{len(runs)} run folders listed in {args.run_list}")
    if not args.pass2:
        print("after these runs finish: python round2_code/round2_1_build_md_inputs.py --pass2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
