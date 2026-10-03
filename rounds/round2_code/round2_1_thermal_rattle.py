#!/usr/bin/env python3
"""
Round 2, step 1 (Python part): thermally rattled crystals, slabs and
AlN/Si(111) interfaces for VASP. No MD and no potential: the amorphous and
collision structures come from NEP MD (round2_1_build_md_inputs.py); the
near-equilibrium crystal structures come from here.

What replaces the round 0/1 rattles:

  * Amplitudes follow temperature and mass. Each atom is displaced by a
    Gaussian whose mean-square amplitude per axis is the Debye-Waller value
        <u_x^2> = 3 hbar^2 T / (m k_B Theta_D^2) * [phi(x) + x/4],  x = Theta_D/T
    (phi: the Debye function; x/4 is the zero-point motion). For Si at 300 K
    this gives B = 8 pi^2 <u_x^2> = 0.45 A^2 (measured 0.46). Round 0/1 drew
    sigma from a fixed list up to 0.5 A, regardless of temperature or mass.
  * Atoms in the outermost 1 A of a slab get --surface-msd-factor x the bulk
    mean-square amplitude: surface atoms vibrate more.
  * A small random strain (bulk: all components; slabs: in plane) samples the
    lattice around its equilibrium, as thermal expansion and stress would.
  * Every frame is checked pair by pair: min d_ij / (r_cov,i + r_cov,j) must
    stay >= --min-contact, else it is redrawn. Round 0/1 used one threshold
    (the N-N one) for every pair and kept Al-Al pairs at 0.9 A.
  * The slabs are cut between bilayers (one dangling bond per surface atom)
    and the interface film is not sheared (see round2_common.py).

Buckets
    thermal_bulk       wurtzite, zincblende, rocksalt AlN; diamond Si; fcc Al
    thermal_slab       Si(111), AlN(0001) Al-polar, AlN(000-1) N-polar
    thermal_interface  AlN(0001)/Si(111), Al- and N-terminated, random
                       registry and +-0.15 A gap

Outputs
    <outdir>/<bucket>_<index>/     POSCAR KPOINTS POTCAR INCAR info.json
    <outdir>/thermal_all.extxyz    every structure, tagged
    <outdir>/amplitudes.txt        sigma per element, material and temperature
    --path-list                    folder list for the VASP array

Usage (from the project root):
    python round2_code/round2_1_thermal_rattle.py

Requires numpy and ase.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter

import numpy as np
from ase.build import bulk
from ase.data import atomic_masses, atomic_numbers
from ase.io import write

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROUND1_CODE = os.path.join(os.path.dirname(CODE_DIR), "round1_code")
sys.path.insert(0, CODE_DIR)
import round2_common as rc  # noqa: E402

HBAR = 1.054571817e-34
AMU = 1.66053906660e-27
KB = 1.380649e-23
# 3 hbar^2 / (amu k_B) in A^2 K
MSD_CONST = 3.0 * HBAR ** 2 / (AMU * KB) * 1e20

# Debye temperatures (K) for the amplitudes. Approximate literature values:
# Si 543 K is the Debye-Waller value (X-ray B factors); AlN ~950 K and Al
# ~390 K. All AlN polymorphs use the wurtzite value. --theta-scale scales them.
DEBYE_K = {"AlN": 950.0, "Si": 543.0, "Al": 390.0}

A_ZB_ALN, A_RS_ALN, A_AL = 4.38, 4.05, 4.05      # same as aln_si_structures.py

BULK_PHASES = {
    # name: (builder, material, temperatures K)
    "AlN_wurtzite": (lambda: bulk("AlN", "wurtzite", a=rc.A_ALN, c=rc.C_ALN,
                                  u=rc.U_ALN).repeat((3, 3, 2)),
                     "AlN", [300, 600, 900, 1200, 1500]),
    "AlN_zincblende": (lambda: bulk("AlN", "zincblende", a=A_ZB_ALN,
                                    cubic=True).repeat(2),
                       "AlN", [300, 600, 900, 1200, 1500]),
    "AlN_rocksalt": (lambda: bulk("AlN", "rocksalt", a=A_RS_ALN,
                                  cubic=True).repeat(2),
                     "AlN", [300, 600, 900, 1200, 1500]),
    "Si_diamond": (lambda: bulk("Si", "diamond", a=rc.A_SI, cubic=True).repeat(2),
                   "Si", [300, 600, 900, 1200, 1500]),
    "Al_fcc": (lambda: bulk("Al", "fcc", a=A_AL, cubic=True).repeat(3),
               "Al", [300, 600, 900]),               # Al melts at 933 K
}

SLABS = {
    "Si111": lambda: rc.si111_slab(4, 3),
    "AlN0001": lambda: rc.aln_slab(4, 3, "Al"),
    "AlN000-1": lambda: rc.aln_slab(4, 3, "N"),
}


def debye_phi(x: float) -> float:
    """phi(x) = (1/x) * integral_0^x t / (e^t - 1) dt"""
    if x < 1e-8:
        return 1.0
    t = np.linspace(1e-9, x, 2001)
    f = t / np.expm1(t)
    return float(((f[1:] + f[:-1]) * 0.5 * np.diff(t)).sum() / x)


def msd_per_axis(T: float, mass: float, theta: float) -> float:
    """Debye-Waller <u_x^2> (A^2) incl. zero-point motion."""
    x = theta / T
    return MSD_CONST * T / (mass * theta ** 2) * (debye_phi(x) + x / 4.0)


def material_of(atoms, default):
    """Debye material per atom: Si atoms of an interface are 'Si', the Al and
    N of a nitride are 'AlN'; otherwise the structure's own material."""
    out = []
    for s in atoms.get_chemical_symbols():
        if default == "interface":
            out.append("Si" if s == "Si" else "AlN")
        else:
            out.append(default)
    return out


def rattle(atoms, T, materials, rng, theta_scale, surface_factor, slab):
    a = atoms.copy()
    m = np.array([atomic_masses[z] for z in a.numbers])
    theta = np.array([DEBYE_K[k] * theta_scale for k in materials])
    msd = np.array([msd_per_axis(T, mi, th) for mi, th in zip(m, theta)])
    if slab:
        z = a.positions[:, 2]
        outer = (z > z.max() - 1.0) | (z < z.min() + 1.0)
        msd[outer] *= surface_factor
    sigma = np.sqrt(msd)
    a.positions += rng.normal(0.0, 1.0, a.positions.shape) * sigma[:, None]
    return a


def strained(atoms, rng, magnitude, in_plane_only):
    e = rng.uniform(-magnitude, magnitude, (3, 3))
    e = 0.5 * (e + e.T)
    if in_plane_only:
        e[2, :] = 0.0
        e[:, 2] = 0.0
    out = atoms.copy()
    out.set_cell(np.array(out.cell) @ (np.eye(3) + e).T, scale_atoms=True)
    return out, float(np.abs(e).max())


def draw(base, T, materials, rng, args, slab, in_plane, tries=50):
    for _ in range(tries):
        a, smax = strained(base, rng, args.strain, in_plane)
        a = rattle(a, T, materials, rng, args.theta_scale, args.surface_msd_factor, slab)
        ratio, pair = rc.contact_ratio(a)
        if ratio >= args.min_contact:
            return a, smax, ratio, pair
    raise RuntimeError(f"no frame above contact ratio {args.min_contact} in {tries} draws")


def tag(a, bucket, kind, T, smax, ratio, pair, **extra):
    a.info = {"bucket": bucket, "kind": kind, "thermal_T_K": float(T),
              "strain_max": round(smax, 5), "contact_ratio": round(ratio, 4),
              "closest_pair": pair, **extra}
    a.info["natoms"] = len(a)
    return a


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Thermally rattled crystals, slabs and interfaces for VASP.")
    p.add_argument("--outdir", default="round2_out/vasp_thermal")
    p.add_argument("--path-list", default="round2_out/round2_1_thermal_vasp_paths.txt")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--force", action="store_true",
                   help="replace an existing --outdir (not once VASP has run)")
    p.add_argument("--n-bulk", type=int, default=6, help="per phase and temperature")
    p.add_argument("--n-slab", type=int, default=4, help="per slab and temperature")
    p.add_argument("--n-interface", type=int, default=4,
                   help="per termination and temperature")
    p.add_argument("--slab-temps", type=float, nargs="+", default=[300, 600, 900])
    p.add_argument("--interface-temps", type=float, nargs="+", default=[300, 600])
    p.add_argument("--interface-terminations", nargs="+", default=["Al", "N"])
    p.add_argument("--strain", type=float, default=0.015,
                   help="max |strain component| of the random strain")
    p.add_argument("--theta-scale", type=float, default=1.0,
                   help="scales every Debye temperature (amplitude ~ 1/theta)")
    p.add_argument("--surface-msd-factor", type=float, default=1.5)
    p.add_argument("--min-contact", type=float, default=0.70,
                   help="redraw frames with min d/(r_cov,i+r_cov,j) below this")
    p.add_argument("--vacuum", type=float, default=15.0,
                   help="total vacuum of slab and interface cells (A)")
    p.add_argument("--max-atoms", type=int, default=200)
    p.add_argument("--potcar-path", default="aaa-potential")
    p.add_argument("--encut", type=float, default=None)
    p.add_argument("--encut-factor", type=float, default=1.3)
    p.add_argument("--kspacing", type=float, default=0.25)
    p.add_argument("--vasp-ntasks", type=int, default=16)
    p.add_argument("--ncore", type=int, default=4)
    args = p.parse_args(argv)

    sys.path.insert(0, ROUND1_CODE)
    import generate_round1 as g1       # noqa: E402  (VASP writer of round 1)

    if os.path.isdir(args.outdir) and any(
            os.path.isfile(os.path.join(args.outdir, d, "POSCAR"))
            for d in os.listdir(args.outdir)) and not args.force:
        sys.exit(f"{args.outdir} already holds VASP folders (--force replaces them)")
    if args.potcar_path and not os.path.isdir(args.potcar_path):
        sys.exit(f"POTCAR directory {args.potcar_path}/ not found; run from the "
                 f"project root or set --potcar-path ('' to skip POTCAR/INCAR)")
    rng = np.random.default_rng(args.seed)
    frames = []

    for name, (build, material, temps) in BULK_PHASES.items():
        base = build()
        mats = material_of(base, material)
        for T in temps:
            for k in range(args.n_bulk):
                a, smax, ratio, pair = draw(base, T, mats, rng, args, False, False)
                frames.append(tag(a, "thermal_bulk", f"{name}_T{T:g}", T, smax,
                                  ratio, pair, phase=name))

    for name, build in SLABS.items():
        base = build()
        mats = material_of(base, "Si" if name == "Si111" else "AlN")
        for T in args.slab_temps:
            for k in range(args.n_slab):
                a, smax, ratio, pair = draw(base, T, mats, rng, args, True, True)
                frames.append(tag(rc.with_vacuum(a, args.vacuum), "thermal_slab",
                                  f"{name}_T{T:g}", T, smax, ratio, pair, slab=name))

    for term in args.interface_terminations:
        for T in args.interface_temps:
            for k in range(args.n_interface):
                cell2 = rc.aln_si_interface(termination=term).cell[:2, :2]
                shift = rng.random(2) @ np.array(cell2)
                gap = (2.3 if term == "Al" else 1.9) + rng.uniform(-0.15, 0.15)
                base = rc.aln_si_interface(termination=term, gap=gap, shift=shift)
                mats = material_of(base, "interface")
                a, _, ratio, pair = draw(base, T, mats, rng, args, True, True)
                frames.append(tag(rc.with_vacuum(a, args.vacuum), "thermal_interface",
                                  f"AlN_Si111_{term}term_T{T:g}", T, 0.0, ratio, pair,
                                  termination=term, gap=round(float(gap), 3),
                                  registry_shift=[round(float(x), 3) for x in shift]))

    over = [a for a in frames if len(a) > args.max_atoms]
    if over:
        sys.exit(f"{len(over)} structures above --max-atoms {args.max_atoms}")
    frames = [g1.sorted_for_vasp(a) for a in frames]

    os.makedirs(args.outdir, exist_ok=True)
    write(os.path.join(args.outdir, "thermal_all.extxyz"), frames, format="extxyz")
    with open(os.path.join(args.outdir, "amplitudes.txt"), "w") as fh:
        fh.write("Debye-Waller sigma per axis (A), bulk atoms; surface atoms x "
                 f"sqrt({args.surface_msd_factor})\n")
        fh.write(f"{'material':9s} {'element':7s} " +
                 " ".join(f"{T:>7.0f}K" for T in (300, 600, 900, 1200, 1500)) + "\n")
        for mat, els in (("AlN", ("Al", "N")), ("Si", ("Si",)), ("Al", ("Al",))):
            for el in els:
                m = atomic_masses[atomic_numbers[el]]
                fh.write(f"{mat:9s} {el:7s} " + " ".join(
                    f"{math.sqrt(msd_per_axis(T, m, DEBYE_K[mat] * args.theta_scale)):8.3f}"
                    for T in (300, 600, 900, 1200, 1500)) + "\n")

    potcar_root = args.potcar_path or None
    encut = args.encut
    if potcar_root:
        present = sorted({s for a in frames for s in a.get_chemical_symbols()})
        enmax, auto = g1.survey_potcars(potcar_root, present, args.encut_factor)
        encut = encut or auto
        print(f"ENCUT {encut} eV (1.3 x max ENMAX of {', '.join(present)})")
    records = g1.write_vasp_tree(frames, args.outdir, kspacing=args.kspacing,
                                 potcar_root=potcar_root, encut=encut,
                                 ncore=args.ncore, kpar=1, dipole=True)
    kpars = Counter()
    if encut:
        for rec in records:
            kpars[rc.set_vasp_parallel(os.path.join(args.outdir, rec[0]),
                                       args.vasp_ntasks, args.ncore)[1]] += 1
    os.makedirs(os.path.dirname(args.path_list) or ".", exist_ok=True)
    with open(args.path_list, "w") as fh:
        for rec in records:
            fh.write(f"./{os.path.join(args.outdir, rec[0])}\n"
                     if not os.path.isabs(args.outdir) else
                     os.path.join(args.outdir, rec[0]) + "\n")

    print(f"wrote {len(records)} VASP folders under {args.outdir}/")
    for b, n in sorted(Counter(a.info["bucket"] for a in frames).items()):
        sizes = [len(a) for a in frames if a.info["bucket"] == b]
        ratios = [a.info["contact_ratio"] for a in frames if a.info["bucket"] == b]
        print(f"  {b:18s} {n:4d}  atoms {min(sizes)}-{max(sizes)}  "
              f"contact ratio min {min(ratios):.2f} median {np.median(ratios):.2f}")
    if kpars:
        print(f"INCAR for {args.vasp_ntasks} ranks: NCORE {args.ncore}, KPAR "
              + ", ".join(f"{k} ({n} folders)" for k, n in sorted(kpars.items())))
    print(f"amplitudes : {args.outdir}/amplitudes.txt")
    print(f"folder list: {args.path_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
