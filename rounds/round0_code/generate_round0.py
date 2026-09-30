#!/usr/bin/env python3
"""
Round-0 seed structure generation for NEP training on film deposition.

Generates the full seed set described in the active-learning workflow:

    bulk_substrate      EOS scan, random strain + rattle, point defects
    bulk_film           same
    dimer_trimer        pair and triple scans, weighted to short separations
    slab_substrate      multiple terminations, thicknesses, rattle, defects
    slab_film           same
    adsorption          single adatoms, clusters, coverages, diffusion paths
    interface           lattice-matched, registry scan, separation scan,
                        intermixed and amorphous interlayers
    isolated_cluster    single atoms and small gas-phase clusters
    disordered          random packings, disordered film on slab, compressed rattle

Defaults are diamond Si(111) substrate and wurtzite AlN(0001) film.

Every structure has strictly fewer than ATOM_LIMIT (150) atoms. --max-atoms
and --max-atoms-interface can lower the cap but never raise it past 149.

Requires only numpy and ase.

Usage
-----
    Run from the project root (aaa-potential/ is looked up from there):
    python round0_code/generate_round0.py
    python round0_code/generate_round0.py --max-atoms 120 --outdir round0_out/vasp
    python round0_code/generate_round0.py --substrate Si --substrate-miller 111 \
        --film AlN --film-structure wurtzite --film-a 3.111 --film-c 4.981 \
        --film-miller 001

Output  (default <outdir> = round0_out/vasp)
------
    <outdir>/round0_all.extxyz        every structure, tagged
    <outdir>/<bucket>.extxyz          one file per bucket
    <outdir>/manifest.csv             provenance table
    <outdir>/lattice_match.txt        candidate interface matches
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.build import bulk as ase_bulk
from ase.build import make_supercell, surface
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import get_distances
from ase.io import write

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

DEFAULT_COUNTS = {
    "bulk_substrate": 50,
    "bulk_film": 50,
    "dimer_trimer": 50,
    "slab_substrate": 50,
    "slab_film": 50,
    "adsorption": 100,
    "interface": 100,
    "isolated_cluster": 20,
    "disordered": 50,
}

# Hard ceiling for the whole campaign: every structure has strictly fewer
# atoms than this. Caps from the command line are clamped to ATOM_LIMIT - 1.
ATOM_LIMIT = 150


@dataclass
class MaterialSpec:
    """A crystalline material and the surface orientation of interest."""

    name: str
    structure: str
    a: float
    c: float | None = None
    miller: tuple = (0, 0, 1)
    elements: list = field(default_factory=list)

    def __post_init__(self):
        if not self.elements:
            self.elements = sorted(set(self.build_bulk().get_chemical_symbols()))

    def build_bulk(self) -> Atoms:
        kwargs = {"a": self.a}
        if self.c is not None:
            kwargs["c"] = self.c
        atoms = ase_bulk(self.name, self.structure, cubic=False, **kwargs)
        atoms.info["material"] = self.name
        return atoms

    def build_slab(self, layers: int, vacuum: float | None = 8.0,
                   periodic_z: bool = False) -> Atoms:
        slab = surface(self.build_bulk(), self.miller, layers, vacuum=vacuum)
        slab.pbc = (True, True, True)
        if periodic_z:
            slab.center(axis=2)
        slab.info["material"] = self.name
        slab.info["miller"] = "".join(str(i) for i in self.miller)
        return slab


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def parse_miller(text: str) -> tuple:
    """Accept '111', '0001' (4-index hexagonal), or '1,-1,0'."""
    text = text.strip()
    if "," in text:
        idx = tuple(int(t) for t in text.split(","))
    else:
        neg = text.startswith("-")
        if neg or any(ch == "-" for ch in text):
            raise ValueError("use comma form for negative indices, e.g. '1,-1,0'")
        idx = tuple(int(ch) for ch in text)
    if len(idx) == 4:  # hexagonal 4-index -> 3-index
        h, k, _, l = idx
        idx = (h, k, l)
    if len(idx) != 3:
        raise ValueError(f"cannot parse Miller indices from {text!r}")
    return idx


def min_pair_distance(atoms: Atoms) -> float:
    if len(atoms) < 2:
        return np.inf
    d = atoms.get_all_distances(mic=any(atoms.pbc))
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def covalent_min_dist(symbols, scale=0.75):
    out = {}
    for a, b in itertools.product(sorted(set(symbols)), repeat=2):
        ra = covalent_radii[atomic_numbers[a]]
        rb = covalent_radii[atomic_numbers[b]]
        out[(a, b)] = scale * (ra + rb)
    return out


def tag(atoms: Atoms, bucket: str, kind: str, **extra) -> Atoms:
    atoms.info["bucket"] = bucket
    atoms.info["kind"] = kind
    atoms.info["natoms"] = len(atoms)
    for k, v in extra.items():
        atoms.info[k] = v
    return atoms


def random_strain(rng, magnitude=0.05, shear=True):
    e = np.zeros((3, 3))
    e[0, 0], e[1, 1], e[2, 2] = rng.uniform(-magnitude, magnitude, 3)
    if shear:
        s = rng.uniform(-magnitude, magnitude, 3)
        e[0, 1] = e[1, 0] = s[0] / 2
        e[0, 2] = e[2, 0] = s[1] / 2
        e[1, 2] = e[2, 1] = s[2] / 2
    return np.eye(3) + e


def apply_strain(atoms: Atoms, F) -> Atoms:
    out = atoms.copy()
    out.set_cell(out.get_cell() @ F.T, scale_atoms=True)
    return out


def rattle(atoms: Atoms, amplitude, rng) -> Atoms:
    out = atoms.copy()
    out.positions += rng.normal(0.0, amplitude, out.positions.shape)
    return out


def repeat_to_cap(atoms: Atoms, cap: int, prefer=(2, 2, 2)) -> Atoms:
    """Largest uniform-ish repetition of atoms that stays under cap."""
    best = atoms.copy()
    for nx, ny, nz in itertools.product(range(1, prefer[0] + 1),
                                        range(1, prefer[1] + 1),
                                        range(1, prefer[2] + 1)):
        n = len(atoms) * nx * ny * nz
        if n <= cap and n > len(best):
            best = atoms.repeat((nx, ny, nz))
    return best


def layers_under_cap(spec: MaterialSpec, cap: int, vacuum: float,
                     nx: int = 1, ny: int = 1, max_layers: int = 12,
                     min_layers: int = 2) -> int:
    """Most slab layers whose nx x ny supercell fits under the atom cap."""
    best = min_layers
    for n in range(min_layers, max_layers + 1):
        slab = spec.build_slab(n, vacuum=vacuum)
        if len(slab) * nx * ny <= cap:
            best = n
        else:
            break
    return best


# --------------------------------------------------------------------------
# surface site detection
# --------------------------------------------------------------------------


def top_layer_indices(slab: Atoms, tol=0.6):
    z = slab.positions[:, 2]
    return np.where(z > z.max() - tol)[0]


def surface_sites(slab: Atoms, tol=0.6, max_per_kind=8):
    """
    Adsorption sites derived from the top-layer geometry.
    Returns dict of kind -> list of (x, y) in-plane coordinates.
    """
    idx = top_layer_indices(slab, tol)
    pos = slab.positions[idx][:, :2]
    cell = slab.get_cell()[:2, :2]

    tops = [tuple(p) for p in pos]

    # neighbours through periodic images
    images = []
    for i, j in itertools.product((-1, 0, 1), repeat=2):
        shift = i * cell[0] + j * cell[1]
        for p in pos:
            images.append(p + shift)
    images = np.array(images)

    def nearest(p, k):
        d = np.linalg.norm(images - p, axis=1)
        order = np.argsort(d)
        return images[order[1:k + 1]], d[order[1:k + 1]]

    bridges, hollows = [], []
    for p in pos:
        nb, d = nearest(p, 6)
        cut = d.min() * 1.25
        near = nb[d <= cut]
        for q in near:
            bridges.append(tuple((p + q) / 2))
        for q1, q2 in itertools.combinations(near, 2):
            if np.linalg.norm(q1 - q2) <= cut:
                hollows.append(tuple((p + q1 + q2) / 3))

    def dedupe(pts, eps=0.35):
        keep = []
        for p in pts:
            p = np.array(p)
            if all(np.linalg.norm(p - np.array(q)) > eps for q in keep):
                keep.append(tuple(p))
        return keep

    return {
        "top": dedupe(tops)[:max_per_kind],
        "bridge": dedupe(bridges)[:max_per_kind],
        "hollow": dedupe(hollows)[:max_per_kind],
    }


def place_adatom(slab: Atoms, symbol: str, xy, height: float) -> Atoms:
    out = slab.copy()
    z = out.positions[:, 2].max() + height
    out += Atoms(symbol, positions=[[xy[0], xy[1], z]])
    return out


# --------------------------------------------------------------------------
# 2D lattice matching (Zur-McGill style)
# --------------------------------------------------------------------------


def hnf_matrices(det: int):
    """Hermite normal forms of a given determinant: unique sublattices."""
    out = []
    for a in range(1, det + 1):
        if det % a:
            continue
        d = det // a
        for b in range(a):
            out.append(np.array([[a, b], [0, d]], dtype=int))
    return out


def cell_invariants(vecs):
    u, v = vecs
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    cosang = np.dot(u, v) / (lu * lv)
    return lu, lv, math.acos(np.clip(cosang, -1.0, 1.0))


def reduce_2d(vecs, n_iter=8):
    """Lagrange-Gauss reduction so equivalent cells compare correctly."""
    u, v = np.array(vecs[0], float), np.array(vecs[1], float)
    for _ in range(n_iter):
        if np.linalg.norm(v) < np.linalg.norm(u):
            u, v = v, u
        m = round(np.dot(u, v) / np.dot(u, u))
        if m == 0:
            break
        v = v - m * u
    if np.linalg.norm(v) < np.linalg.norm(u):
        u, v = v, u
    return np.array([u, v])


@dataclass
class LatticeMatch:
    m_sub: np.ndarray
    m_film: np.ndarray
    n_sub_cells: int
    n_film_cells: int
    strain_u: float
    strain_v: float
    strain_angle: float

    @property
    def max_strain(self):
        return max(abs(self.strain_u), abs(self.strain_v), abs(self.strain_angle))


def match_lattices(sub_2d, film_2d, max_cells=30, max_strain=0.06,
                   area_tol=0.12):
    """
    Find integer supercells of two 2D lattices that coincide within max_strain.
    sub_2d, film_2d are 2x3 arrays of in-plane cell vectors.
    """
    area_s = np.linalg.norm(np.cross(sub_2d[0], sub_2d[1]))
    area_f = np.linalg.norm(np.cross(film_2d[0], film_2d[1]))

    results = []
    for i in range(1, max_cells + 1):
        for j in range(1, max_cells + 1):
            if abs(i * area_s - j * area_f) / (i * area_s) > area_tol:
                continue
            for ms in hnf_matrices(i):
                sv = reduce_2d(ms @ sub_2d)
                ls_u, ls_v, ls_a = cell_invariants(sv)
                for mf in hnf_matrices(j):
                    fv = reduce_2d(mf @ film_2d)
                    lf_u, lf_v, lf_a = cell_invariants(fv)
                    for swap in (False, True):
                        au, av = (lf_v, lf_u) if swap else (lf_u, lf_v)
                        aa = lf_a
                        for flip in (1.0, -1.0):
                            ang = aa if flip > 0 else math.pi - aa
                            eu = (au - ls_u) / ls_u
                            ev = (av - ls_v) / ls_v
                            ea = (ang - ls_a) / ls_a
                            if max(abs(eu), abs(ev), abs(ea)) <= max_strain:
                                results.append(LatticeMatch(
                                    ms, mf, i, j, eu, ev, ea))
    results.sort(key=lambda r: (r.n_sub_cells + r.n_film_cells, r.max_strain))
    # drop duplicates with identical cell counts and near-identical strain
    unique, seen = [], set()
    for r in results:
        key = (r.n_sub_cells, r.n_film_cells, round(r.max_strain, 4))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def supercell_2d(slab: Atoms, m2: np.ndarray) -> Atoms:
    m3 = np.eye(3, dtype=int)
    m3[:2, :2] = m2
    return make_supercell(slab, m3)


# --------------------------------------------------------------------------
# bucket: bulk
# --------------------------------------------------------------------------


def gen_bulk(spec: MaterialSpec, n: int, cap: int, rng) -> list:
    base = repeat_to_cap(spec.build_bulk(), cap, prefer=(3, 3, 3))
    out = []

    n_eos = max(4, int(0.24 * n))
    n_strain = max(4, int(0.46 * n))
    n_defect = max(2, int(0.18 * n))
    n_extreme = max(2, n - n_eos - n_strain - n_defect)

    for s in np.linspace(-0.08, 0.08, n_eos):
        a = base.copy()
        a.set_cell(a.get_cell() * (1 + s), scale_atoms=True)
        out.append(tag(a, "bulk_" + spec.name, "eos", volume_strain=float(s)))

    for _ in range(n_strain):
        amp = rng.choice([0.02, 0.05, 0.10, 0.16, 0.22])
        a = apply_strain(base, random_strain(rng, magnitude=0.05))
        a = rattle(a, amp, rng)
        out.append(tag(a, "bulk_" + spec.name, "strain_rattle",
                       rattle_amplitude=float(amp)))

    defect_cell = repeat_to_cap(spec.build_bulk(), cap, prefer=(3, 3, 3))
    mind = covalent_min_dist(defect_cell.get_chemical_symbols())
    for k in range(n_defect):
        a = defect_cell.copy()
        mode = k % 4
        if mode == 0 and len(a) > 2:  # monovacancy
            del a[int(rng.integers(len(a)))]
            kind = "vacancy"
        elif mode == 1 and len(a) > 4:  # divacancy
            victims = sorted(rng.choice(len(a), 2, replace=False), reverse=True)
            for idx in victims:
                del a[int(idx)]
            kind = "divacancy"
        elif mode == 2 and len(a) < cap:  # interstitial
            sym = str(rng.choice(spec.elements))
            for _ in range(500):
                cand = rng.random(3) @ a.get_cell()
                trial = a.copy()
                trial += Atoms(sym, positions=[cand])
                if min_pair_distance(trial) > 0.7 * min(mind.values()):
                    a = trial
                    break
            kind = "interstitial"
        else:  # antisite, or vacancy fallback for single-element
            kind = "vacancy"
            if len(spec.elements) > 1:
                syms = a.get_chemical_symbols()
                i = int(rng.integers(len(a)))
                others = [e for e in spec.elements if e != syms[i]]
                syms[i] = str(rng.choice(others))
                a.set_chemical_symbols(syms)
                kind = "antisite"
            elif len(a) > 2:
                del a[int(rng.integers(len(a)))]
        a = rattle(a, 0.08, rng)
        out.append(tag(a, "bulk_" + spec.name, kind))

    for s in np.linspace(-0.15, 0.15, n_extreme):
        a = base.copy()
        a.set_cell(a.get_cell() * (1 + s), scale_atoms=True)
        a = rattle(a, 0.10, rng)
        out.append(tag(a, "bulk_" + spec.name, "extreme_volume",
                       volume_strain=float(s)))

    return out[:n]


# --------------------------------------------------------------------------
# bucket: dimers and trimers
# --------------------------------------------------------------------------


def gen_dimer_trimer(elements, n: int, box=16.0, rng=None) -> list:
    rng = rng or np.random.default_rng()
    pairs = list(itertools.combinations_with_replacement(sorted(elements), 2))
    out = []

    n_dimer = int(round(0.72 * n))
    per_pair = max(6, n_dimer // max(1, len(pairs)))

    for a, b in pairs:
        r0 = covalent_radii[atomic_numbers[a]] + covalent_radii[atomic_numbers[b]]
        # log spacing, weighted to short separations
        seps = np.geomspace(0.55, max(6.0, 2.6 * r0), per_pair)
        for r in seps:
            at = Atoms([a, b], positions=[[0, 0, 0], [r, 0, 0]],
                       cell=[box, box, box], pbc=False)
            at.center()
            out.append(tag(at, "dimer_trimer", "dimer",
                           pair=f"{a}-{b}", separation=float(r)))

    n_trimer = max(4, n - len(out))
    triples = list(itertools.combinations_with_replacement(sorted(elements), 3))
    for k in range(n_trimer):
        t = triples[k % len(triples)]
        r0 = np.mean([covalent_radii[atomic_numbers[s]] for s in t]) * 2
        r1 = float(rng.uniform(0.65, 1.6) * r0)
        r2 = float(rng.uniform(0.65, 1.6) * r0)
        theta = float(rng.uniform(35.0, 180.0))
        th = math.radians(theta)
        at = Atoms(list(t), positions=[
            [0.0, 0.0, 0.0],
            [r1, 0.0, 0.0],
            [r2 * math.cos(th), r2 * math.sin(th), 0.0],
        ], cell=[box, box, box], pbc=False)
        at.center()
        out.append(tag(at, "dimer_trimer", "trimer",
                       triple="-".join(t), angle_deg=theta))

    rng.shuffle(out)
    return out[:n]


# --------------------------------------------------------------------------
# bucket: slabs
# --------------------------------------------------------------------------


def gen_slabs(spec: MaterialSpec, n: int, cap: int, rng,
              vacuum=10.0) -> list:
    out = []
    bucket = "slab_" + spec.name

    base_layers = layers_under_cap(spec, cap, vacuum, nx=2, ny=2)
    thin = max(2, base_layers - 2)

    variants = []
    for nx, ny, layers in [(1, 1, base_layers), (2, 2, thin),
                           (2, 1, base_layers), (1, 1, thin)]:
        slab = spec.build_slab(layers, vacuum=vacuum).repeat((nx, ny, 1))
        if len(slab) <= cap:
            variants.append((slab, nx, ny, layers))
    if not variants:
        variants = [(spec.build_slab(2, vacuum=vacuum), 1, 1, 2)]

    per = max(1, n // max(1, len(variants)))
    for slab, nx, ny, layers in variants:
        out.append(tag(slab.copy(), bucket, "clean",
                       layers=layers, supercell=f"{nx}x{ny}"))
        for _ in range(per // 2):
            amp = float(rng.choice([0.03, 0.06, 0.12, 0.20]))
            out.append(tag(rattle(slab, amp, rng), bucket, "rattled",
                           layers=layers, rattle_amplitude=amp))
        # surface vacancy
        if len(slab) > 4:
            a = slab.copy()
            top = top_layer_indices(a)
            del a[int(rng.choice(top))]
            out.append(tag(rattle(a, 0.06, rng), bucket, "surface_vacancy",
                           layers=layers))
        # self adatom
        if len(slab) < cap:
            sites = surface_sites(slab)
            kind = "hollow" if sites["hollow"] else "top"
            if sites[kind]:
                xy = sites[kind][int(rng.integers(len(sites[kind])))]
                sym = str(rng.choice(spec.elements))
                a = place_adatom(slab, sym, xy, float(rng.uniform(1.4, 2.4)))
                out.append(tag(a, bucket, "self_adatom",
                               layers=layers, site=kind))
        # in-plane strained slab
        a = slab.copy()
        F = random_strain(rng, magnitude=0.03)
        F[2, :] = [0, 0, 1]
        F[:, 2] = [0, 0, 1]
        out.append(tag(apply_strain(a, F), bucket, "strained_slab",
                       layers=layers))

    while len(out) < n:
        slab, nx, ny, layers = variants[int(rng.integers(len(variants)))]
        amp = float(rng.uniform(0.03, 0.22))
        out.append(tag(rattle(slab, amp, rng), bucket, "rattled",
                       layers=layers, rattle_amplitude=amp))

    return out[:n]


# --------------------------------------------------------------------------
# bucket: adsorption
# --------------------------------------------------------------------------


def gen_adsorption(sub: MaterialSpec, film: MaterialSpec, n: int, cap: int,
                   rng, vacuum=12.0) -> list:
    out = []
    layers = layers_under_cap(sub, cap - 8, vacuum, nx=2, ny=2)
    slab_small = sub.build_slab(layers, vacuum=vacuum)
    slab_big = slab_small.repeat((2, 2, 1))
    if len(slab_big) > cap - 8:
        slab_big = slab_small
    sites = surface_sites(slab_big)
    film_elems = film.elements

    heights = [1.0, 1.5, 2.0, 2.6, 3.4, 4.5]

    # single adatoms over site x height
    for sym in film_elems:
        for kind in ("top", "bridge", "hollow"):
            pts = sites[kind]
            if not pts:
                continue
            for xy in pts[:2]:
                for h in heights:
                    if len(out) >= int(0.48 * n):
                        break
                    a = place_adatom(slab_big, sym, xy, h)
                    if len(a) <= cap:
                        out.append(tag(a, "adsorption", "single_adatom",
                                       species=sym, site=kind, height=float(h)))

    # small clusters on the surface
    mind = covalent_min_dist(list(film_elems) + sub.elements)
    n_cluster = int(0.25 * n)
    for _ in range(n_cluster):
        k = int(rng.integers(2, 6))
        a = slab_big.copy()
        z0 = a.positions[:, 2].max()
        placed = 0
        for _ in range(400):
            if placed >= k or len(a) >= cap:
                break
            sym = str(rng.choice(film_elems))
            base_xy = sites["hollow"] or sites["top"]
            xy = np.array(base_xy[int(rng.integers(len(base_xy)))])
            xy = xy + rng.normal(0, 1.2, 2)
            z = z0 + rng.uniform(1.3, 3.6)
            trial = a.copy()
            trial += Atoms(sym, positions=[[xy[0], xy[1], z]])
            if min_pair_distance(trial) > 0.8 * min(mind.values()):
                a = trial
                placed += 1
        if placed >= 2:
            out.append(tag(a, "adsorption", "cluster", n_adatoms=placed))

    # sub-monolayer coverages
    n_cov = int(0.18 * n)
    hollow = sites["hollow"] or sites["top"]
    for _ in range(n_cov):
        cov = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
        ordered = bool(rng.integers(2))
        a = slab_big.copy()
        z0 = a.positions[:, 2].max()
        k = max(1, int(round(cov * len(hollow))))
        order = np.arange(len(hollow)) if ordered else rng.permutation(len(hollow))
        for i in order[:k]:
            if len(a) >= cap:
                break
            sym = str(rng.choice(film_elems))
            xy = hollow[int(i)]
            h = rng.uniform(1.4, 2.2) if ordered else rng.uniform(1.2, 2.8)
            a += Atoms(sym, positions=[[xy[0], xy[1], z0 + h]])
        out.append(tag(a, "adsorption", "coverage",
                       coverage=cov, ordered=ordered))

    # diffusion path saddles: midpoints between adjacent sites
    while len(out) < n:
        sym = str(rng.choice(film_elems))
        pool = sites["hollow"] + sites["bridge"] + sites["top"]
        if len(pool) < 2:
            break
        p, q = [np.array(pool[int(i)]) for i in
                rng.choice(len(pool), 2, replace=False)]
        if np.linalg.norm(p - q) > 6.0:
            continue
        frac = float(rng.uniform(0.2, 0.8))
        xy = p + frac * (q - p)
        a = place_adatom(slab_big, sym, xy, float(rng.uniform(1.3, 2.3)))
        if len(a) <= cap:
            out.append(tag(a, "adsorption", "diffusion_path",
                           species=sym, path_fraction=frac))

    return out[:n]


# --------------------------------------------------------------------------
# bucket: interface
# --------------------------------------------------------------------------


def build_interface(sub: MaterialSpec, film: MaterialSpec,
                    match: LatticeMatch, sub_layers: int, film_layers: int,
                    gap: float, shift=(0.0, 0.0), vacuum=12.0) -> Atoms | None:
    """Stack a strained film supercell on a substrate supercell."""
    s_slab = supercell_2d(sub.build_slab(sub_layers, vacuum=6.0), match.m_sub)
    f_slab = supercell_2d(film.build_slab(film_layers, vacuum=6.0), match.m_film)

    s_cell = s_slab.get_cell()
    f_cell = f_slab.get_cell()

    # map the film in-plane lattice onto the substrate's: rotation + strain
    new_f_cell = np.array([s_cell[0], s_cell[1], f_cell[2]])
    if abs(np.linalg.det(new_f_cell)) < 1e-6:
        return None
    f_slab = f_slab.copy()
    f_slab.set_cell(new_f_cell, scale_atoms=True)

    s_slab.positions[:, 2] -= s_slab.positions[:, 2].min()
    f_slab.positions[:, 2] -= f_slab.positions[:, 2].min()

    s_top = s_slab.positions[:, 2].max()
    f_slab.positions[:, 2] += s_top + gap
    f_slab.positions[:, 0] += shift[0]
    f_slab.positions[:, 1] += shift[1]

    combined = s_slab + f_slab
    total_z = combined.positions[:, 2].max() + vacuum
    cell = np.array([s_cell[0], s_cell[1], [0.0, 0.0, total_z]])
    combined.set_cell(cell, scale_atoms=False)
    combined.pbc = (True, True, True)
    combined.info["n_substrate"] = len(s_slab)
    combined.info["n_film"] = len(f_slab)
    return combined


def choose_interface_geometry(sub, film, matches, cap, vacuum):
    """Pick the match plus layer counts giving the thickest cell under cap."""
    best = None
    for m in matches[:12]:
        for sl in range(6, 1, -1):
            for fl in range(6, 0, -1):
                try:
                    trial = build_interface(sub, film, m, sl, fl,
                                            gap=2.2, vacuum=vacuum)
                except Exception:
                    continue
                if trial is None or len(trial) > cap:
                    continue
                score = (min(sl, fl),
                         -(m.n_sub_cells + m.n_film_cells),
                         -m.max_strain)
                if best is None or score > best[0]:
                    best = (score, m, sl, fl, len(trial))
    return best


def gen_interface(sub: MaterialSpec, film: MaterialSpec, n: int, cap: int,
                  rng, vacuum=12.0, max_strain=0.06, report_path=None) -> list:
    s_slab = sub.build_slab(2, vacuum=6.0)
    f_slab = film.build_slab(2, vacuum=6.0)
    matches = match_lattices(s_slab.get_cell()[:2], f_slab.get_cell()[:2],
                             max_cells=30, max_strain=max_strain)

    if report_path:
        with open(report_path, "w") as fh:
            fh.write(f"substrate {sub.name}{sub.miller}  "
                     f"in-plane a={np.linalg.norm(s_slab.get_cell()[0]):.4f} "
                     f"b={np.linalg.norm(s_slab.get_cell()[1]):.4f}\n")
            fh.write(f"film      {film.name}{film.miller}  "
                     f"in-plane a={np.linalg.norm(f_slab.get_cell()[0]):.4f} "
                     f"b={np.linalg.norm(f_slab.get_cell()[1]):.4f}\n\n")
            fh.write(f"{'n_sub':>6} {'n_film':>7} {'strain_u':>10} "
                     f"{'strain_v':>10} {'strain_ang':>11} {'max':>8}\n")
            for m in matches[:40]:
                fh.write(f"{m.n_sub_cells:6d} {m.n_film_cells:7d} "
                         f"{m.strain_u:10.4f} {m.strain_v:10.4f} "
                         f"{m.strain_angle:11.4f} {m.max_strain:8.4f}\n")

    if not matches:
        print(f"  ! no lattice match within {max_strain:.1%}; "
              f"interface bucket skipped", file=sys.stderr)
        return []

    chosen = choose_interface_geometry(sub, film, matches, cap, vacuum)
    if chosen is None:
        print(f"  ! no interface fits in {cap} atoms (hard limit: fewer "
              f"than {ATOM_LIMIT}); see lattice_match.txt", file=sys.stderr)
        return []

    _, match, sl, fl, natoms = chosen

    def _slab_stats(spec, layers, m2):
        sl_ = supercell_2d(spec.build_slab(layers, vacuum=6.0), m2)
        z = sl_.positions[:, 2]
        return len(sl_), float(z.max() - z.min())

    n_s, t_s = _slab_stats(sub, sl, match.m_sub)
    n_f, t_f = _slab_stats(film, fl, match.m_film)
    per_s, per_f = n_s / max(1, sl), n_f / max(1, fl)

    print(f"  interface: {match.n_sub_cells} substrate cells / "
          f"{match.n_film_cells} film cells, mismatch {match.max_strain:.2%}")
    print(f"             {sl} substrate layers ({n_s} atoms, {t_s:.1f} A) + "
          f"{fl} film layers ({n_f} atoms, {t_f:.1f} A) = {natoms} atoms")

    if t_s < 6.0 or t_f < 6.0:
        want = int(math.ceil(3 * per_s + 2 * per_f))
        print(f"  ! this interface is very thin. Under a {cap}-atom cap the "
              f"only fit is {sl}+{fl} layers.\n"
              f"    Each substrate layer costs ~{per_s:.0f} atoms and each "
              f"film layer ~{per_f:.0f}.\n"
              f"    3 substrate + 2 film layers would need about {want} "
              f"atoms, but structures are held below {ATOM_LIMIT} atoms.",
              file=sys.stderr)

    out = []
    s_cell = supercell_2d(sub.build_slab(sl, vacuum=6.0),
                          match.m_sub).get_cell()

    # registry scan on a 3x3 in-plane grid
    n_reg = int(0.34 * n)
    grid = []
    for i, j in itertools.product(np.linspace(0, 1, 3, endpoint=False),
                                  repeat=2):
        grid.append(i * s_cell[0][:2] + j * s_cell[1][:2])
    for k in range(n_reg):
        shift = grid[k % len(grid)]
        gap = float(rng.uniform(1.9, 2.6))
        a = build_interface(sub, film, match, sl, fl, gap, tuple(shift), vacuum)
        if a is not None and len(a) <= cap:
            out.append(tag(a, "interface", "registry",
                           strain=float(match.max_strain), gap=gap))

    # separation scan at fixed registry
    n_sep = int(0.30 * n)
    for gap in np.linspace(1.4, 5.5, max(2, n_sep)):
        a = build_interface(sub, film, match, sl, fl, float(gap),
                            (0.0, 0.0), vacuum)
        if a is not None and len(a) <= cap:
            out.append(tag(a, "interface", "separation",
                           strain=float(match.max_strain), gap=float(gap)))

    # rattled at the nominal geometry
    n_rat = int(0.12 * n)
    base = build_interface(sub, film, match, sl, fl, 2.2, (0.0, 0.0), vacuum)
    if base is not None:
        for _ in range(n_rat):
            amp = float(rng.choice([0.04, 0.08, 0.15]))
            out.append(tag(rattle(base, amp, rng), "interface", "rattled",
                           rattle_amplitude=amp))

    # intermixed: swap atoms across the interfacial plane
    n_mix = int(0.14 * n)
    if base is not None:
        n_sub_atoms = base.info["n_substrate"]
        for _ in range(n_mix):
            a = base.copy()
            syms = a.get_chemical_symbols()
            z = a.positions[:, 2]
            zc = 0.5 * (z[:n_sub_atoms].max() + z[n_sub_atoms:].min())
            near_sub = [i for i in range(n_sub_atoms) if z[i] > zc - 3.0]
            near_film = [i for i in range(n_sub_atoms, len(a))
                         if z[i] < zc + 3.0]
            k = int(rng.integers(1, 4))
            if not near_sub or not near_film:
                break
            for _ in range(k):
                i = int(rng.choice(near_sub))
                j = int(rng.choice(near_film))
                syms[i], syms[j] = syms[j], syms[i]
            a.set_chemical_symbols(syms)
            out.append(tag(rattle(a, 0.08, rng), "interface", "intermixed",
                           n_swaps=k))

    # amorphous interlayer: heavily displaced atoms near the interface
    if base is not None:
        guard = 0
        while len(out) < n and guard < 40 * n:
            guard += 1
            a = base.copy()
            n_sub_atoms = a.info["n_substrate"]
            z = a.positions[:, 2]
            zc = 0.5 * (z[:n_sub_atoms].max() + z[n_sub_atoms:].min())
            width = float(rng.uniform(2.0, 4.0))
            mask = np.abs(z - zc) < width
            if not mask.any():
                break
            sigma = float(rng.uniform(0.25, 0.45))
            a.positions[mask] += rng.normal(0.0, sigma, (int(mask.sum()), 3))
            if min_pair_distance(a) < 0.8:
                continue
            out.append(tag(a, "interface", "amorphous_interlayer",
                           interlayer_width=width))

    return out[:n]


# --------------------------------------------------------------------------
# bucket: isolated atoms and gas-phase clusters
# --------------------------------------------------------------------------


def gen_isolated(elements, n: int, box=16.0, rng=None) -> list:
    rng = rng or np.random.default_rng()
    out = []
    for sym in sorted(elements):
        at = Atoms(sym, positions=[[0, 0, 0]], cell=[box, box, box], pbc=False)
        at.center()
        out.append(tag(at, "isolated_cluster", "isolated_atom", species=sym))

    mind = covalent_min_dist(elements)
    while len(out) < n:
        k = int(rng.integers(2, 7))
        syms = [str(rng.choice(sorted(elements))) for _ in range(k)]
        pos, tries = [], 0
        r0 = 2.0 * np.mean([covalent_radii[atomic_numbers[s]] for s in syms])
        while len(pos) < k and tries < 3000:
            tries += 1
            cand = rng.normal(0.0, 0.75 * r0 * k ** (1 / 3), 3)
            if all(np.linalg.norm(cand - p) > 0.85 * min(mind.values())
                   for p in pos):
                pos.append(cand)
        if len(pos) < k:
            continue
        at = Atoms(syms, positions=pos, cell=[box, box, box], pbc=False)
        at.center()
        out.append(tag(at, "isolated_cluster", "gas_cluster", n_atoms=k))
    return out[:n]


# --------------------------------------------------------------------------
# bucket: disordered
# --------------------------------------------------------------------------


def random_packed_cell(composition, density, min_dist, cap, rng,
                       cell_jitter=0.15, max_tries=40000):
    from ase.data import atomic_masses
    symbols = []
    for sym, k in composition.items():
        symbols += [sym] * k
    if len(symbols) > cap:
        return None
    rng.shuffle(symbols)

    mass = sum(atomic_masses[atomic_numbers[s]] for s in symbols)
    volume = mass * 1.66053906660 / density
    a = volume ** (1 / 3)
    cell = np.eye(3) * a
    e = rng.uniform(-cell_jitter, cell_jitter, (3, 3))
    e = 0.5 * (e + e.T)
    cell = cell @ (np.eye(3) + e)
    cell *= (volume / abs(np.linalg.det(cell))) ** (1 / 3)

    pos, syms, tries = [], [], 0
    for sym in symbols:
        while True:
            tries += 1
            if tries > max_tries:
                return None
            cand = rng.random(3) @ cell
            ok = True
            for p, s in zip(pos, syms):
                _, d = get_distances(np.array([p]), np.array([cand]),
                                     cell=cell, pbc=True)
                if float(np.asarray(d).ravel()[0]) < min_dist[(s, sym)]:
                    ok = False
                    break
            if ok:
                pos.append(cand)
                syms.append(sym)
                break
    return Atoms(symbols=syms, positions=pos, cell=cell, pbc=True)


def gen_disordered(sub: MaterialSpec, film: MaterialSpec, n: int, cap: int,
                   rng, vacuum=12.0) -> list:
    out = []
    film_bulk = film.build_bulk()
    from ase.data import atomic_masses
    rho0 = (sum(atomic_masses[atomic_numbers[s]]
                for s in film_bulk.get_chemical_symbols())
            * 1.66053906660 / film_bulk.get_volume())

    # random packings of the film at several densities
    n_pack = int(0.55 * n)
    per_fu = film_bulk.get_chemical_symbols()
    n_fu = max(2, min(12, cap // max(1, len(per_fu))))
    comp = {}
    for s in per_fu:
        comp[s] = comp.get(s, 0) + 1
    comp = {k: v * n_fu for k, v in comp.items()}
    mind = covalent_min_dist(list(comp.keys()), scale=0.72)

    densities = [0.75 * rho0, 0.88 * rho0, 1.0 * rho0, 1.12 * rho0]
    made = 0
    guard = 0
    while made < n_pack and guard < 8 * n_pack:
        guard += 1
        rho = float(densities[made % len(densities)])
        a = random_packed_cell(comp, rho, mind, cap, rng)
        if a is None:
            continue
        out.append(tag(a, "disordered", "random_packing",
                       density=rho, density_ratio=float(rho / rho0)))
        made += 1

    # disordered film on the substrate slab
    n_film = int(0.25 * n)
    layers = layers_under_cap(sub, cap // 2, vacuum, nx=2, ny=2)
    slab = sub.build_slab(layers, vacuum=vacuum).repeat((2, 2, 1))
    if len(slab) > cap - 12:
        slab = sub.build_slab(layers, vacuum=vacuum)
    mind_all = covalent_min_dist(
        list(comp.keys()) + slab.get_chemical_symbols(), scale=0.72)
    for _ in range(n_film):
        a = slab.copy()
        z0 = a.positions[:, 2].max()
        budget = cap - len(a)
        k = int(rng.integers(max(2, budget // 3), max(3, budget) + 1))
        thickness = float(rng.uniform(3.0, 7.0))
        placed, tries = 0, 0
        cell = a.get_cell()
        while placed < k and tries < 6000:
            tries += 1
            sym = str(rng.choice(sorted(comp.keys())))
            f = rng.random(2)
            xy = f[0] * cell[0][:2] + f[1] * cell[1][:2]
            z = z0 + rng.uniform(1.4, 1.4 + thickness)
            trial = a.copy()
            trial += Atoms(sym, positions=[[xy[0], xy[1], z]])
            if min_pair_distance(trial) > 0.85 * min(mind_all.values()):
                a = trial
                placed += 1
        if placed >= 2:
            out.append(tag(a, "disordered", "disordered_film_on_slab",
                           n_film_atoms=placed, film_thickness=thickness))

    # compressed and heavily rattled crystals
    guard = 0
    while len(out) < n and guard < 40 * n:
        guard += 1
        spec = sub if rng.integers(2) else film
        base = repeat_to_cap(spec.build_bulk(), cap, prefer=(3, 3, 3))
        comp_frac = float(rng.uniform(0.08, 0.30))
        amp = float(rng.uniform(0.20, 0.50))
        a = base.copy()
        a.set_cell(a.get_cell() * (1 - comp_frac) ** (1 / 3), scale_atoms=True)
        a = rattle(a, amp, rng)
        if min_pair_distance(a) < 0.75:
            continue
        out.append(tag(a, "disordered", "compressed_rattle",
                       compression=comp_frac, rattle_amplitude=amp))

    return out[:n]


# --------------------------------------------------------------------------
# validation and output
# --------------------------------------------------------------------------


def validate(frames, cap, hard_min=0.5, interface_cap=None):
    interface_cap = interface_cap or cap
    kept, dropped = [], []
    for a in frames:
        limit = interface_cap if a.info.get("bucket") == "interface" else cap
        if len(a) > limit or len(a) >= ATOM_LIMIT:
            dropped.append((a.info.get("kind", "?"), "over atom cap", len(a)))
            continue
        d = min_pair_distance(a)
        if a.info.get("kind") not in ("dimer", "trimer") and d < hard_min:
            dropped.append((a.info.get("kind", "?"), "atoms too close",
                            round(d, 3)))
            continue
        a.info["min_distance"] = round(float(d), 4) if np.isfinite(d) else -1.0
        kept.append(a)
    return kept, dropped


def sorted_for_vasp(atoms: Atoms) -> Atoms:
    """Group atoms by species so POSCAR order matches a concatenated POTCAR."""
    order = np.argsort(atoms.numbers, kind="stable")
    out = atoms[order]
    out.info = dict(atoms.info)
    return out


def species_blocks(atoms: Atoms):
    """[(symbol, count), ...] in POSCAR order."""
    blocks = []
    for sym in atoms.get_chemical_symbols():
        if blocks and blocks[-1][0] == sym:
            blocks[-1][1] += 1
        else:
            blocks.append([sym, 1])
    return [(s, n) for s, n in blocks]


def kpoint_mesh(atoms: Atoms, kspacing: float, gamma_only: bool = False):
    """Gamma-centred mesh at a fixed reciprocal-space density (VASP KSPACING)."""
    if gamma_only or not any(atoms.pbc):
        return (1, 1, 1)
    b = np.linalg.norm(np.asarray(atoms.cell.reciprocal()), axis=1) * 2 * np.pi
    mesh = []
    for i in range(3):
        if not atoms.pbc[i] or b[i] <= 0:
            mesh.append(1)
        else:
            mesh.append(max(1, int(math.ceil(b[i] / kspacing))))
    return tuple(mesh)


def write_kpoints(path, mesh):
    with open(path, "w") as fh:
        fh.write("Auto mesh from fixed k-spacing\n0\nGamma\n")
        fh.write(f"{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n")


def write_vasp_tree(frames, root, kspacing=0.25, write_kpts=True,
                    potcar_root=None, encut=None, ncore=16, kpar=1,
                    dipole=True, ispin_molecular=2):
    """
    One directory per structure:  root/<bucket>_<index>/POSCAR

    Also writes KPOINTS, an info.json with provenance, and a top-level
    dirlist.txt plus species_order.csv for POTCAR assembly.
    """
    import json

    os.makedirs(root, exist_ok=True)
    counters, records = {}, []

    molecular = {"dimer_trimer", "isolated_cluster"}

    for atoms in frames:
        bucket = str(atoms.info.get("bucket", "misc"))
        idx = counters.get(bucket, 0)
        counters[bucket] = idx + 1

        dirname = f"{bucket}_{idx:04d}"
        folder = os.path.join(root, dirname)
        os.makedirs(folder, exist_ok=True)

        a = sorted_for_vasp(atoms)
        if not any(a.pbc):
            a.pbc = (True, True, True)

        poscar = os.path.join(folder, "POSCAR")
        write(poscar, a, format="vasp", direct=True, sort=False, vasp5=True)

        blocks = species_blocks(a)
        order = " ".join(sym for sym, _ in blocks)

        mesh = kpoint_mesh(a, kspacing, gamma_only=bucket in molecular)
        if write_kpts:
            write_kpoints(os.path.join(folder, "KPOINTS"), mesh)

        is_mol = bucket in molecular
        used_dipole = None
        if potcar_root:
            assemble_potcar(folder, order, potcar_root)
        if encut:
            used_dipole = write_incar(
                os.path.join(folder, "INCAR"), a, encut, ncore, kpar,
                molecular=is_mol, dipole=dipole,
                ispin_molecular=ispin_molecular,
                system=f"{dirname} {a.get_chemical_formula()}")

        info = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in a.info.items()}
        info.update({
            "directory": dirname,
            "formula": a.get_chemical_formula(),
            "natoms": len(a),
            "species_order": order,
            "species_counts": [[sym, n] for sym, n in blocks],
            "kpoint_mesh": list(mesh),
            "is_molecular": is_mol,
            "encut": encut,
            "dipole_correction": used_dipole,
        })
        with open(os.path.join(folder, "info.json"), "w") as fh:
            json.dump(info, fh, indent=2, default=str)

        records.append((dirname, bucket, str(a.info.get("kind", "")),
                        a.get_chemical_formula(), len(a), order,
                        "x".join(str(m) for m in mesh),
                        "yes" if used_dipole else "no"))

    with open(os.path.join(root, "dirlist.txt"), "w") as fh:
        for r in records:
            fh.write(r[0] + "\n")

    with open(os.path.join(root, "species_order.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["directory", "bucket", "kind", "formula", "natoms",
                    "potcar_order", "kpoints", "dipole"])
        w.writerows(records)

    return records


# --------------------------------------------------------------------------
# VASP inputs: POTCAR assembly and INCAR generation
# --------------------------------------------------------------------------


def read_enmax(potcar_path: str) -> float:
    """Largest ENMAX in a POTCAR file, in eV."""
    best = 0.0
    with open(potcar_path, errors="ignore") as fh:
        for line in fh:
            if "ENMAX" in line:
                try:
                    val = line.split("ENMAX")[1].split("=")[1]
                    val = val.split(";")[0].split()[0]
                    best = max(best, float(val))
                except (IndexError, ValueError):
                    continue
    if best <= 0:
        raise ValueError(f"no ENMAX found in {potcar_path}")
    return best


def potcar_file(root: str, symbol: str) -> str:
    path = os.path.join(root, symbol, "POTCAR")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no POTCAR for {symbol} at {path}; expected "
            f"<potcar-path>/<element>/POTCAR")
    return path


def survey_potcars(root: str, elements, encut_factor=1.3, round_to=10):
    """One ENCUT for the whole campaign: factor x the largest ENMAX."""
    enmax = {}
    for sym in sorted(elements):
        enmax[sym] = read_enmax(potcar_file(root, sym))
    raw = encut_factor * max(enmax.values())
    encut = int(round_to * math.ceil(raw / round_to))
    return enmax, encut


def assemble_potcar(folder: str, order, root: str):
    """Concatenate element POTCARs in POSCAR order and verify the result."""
    out = os.path.join(folder, "POTCAR")
    syms = order.split()
    with open(out, "wb") as dst:
        for sym in syms:
            with open(potcar_file(root, sym), "rb") as src:
                dst.write(src.read())
    n_titel = sum(1 for line in open(out, errors="ignore") if "TITEL" in line)
    if n_titel != len(syms):
        raise RuntimeError(
            f"{out}: {n_titel} TITEL entries but POSCAR has {len(syms)} "
            f"species blocks ({order})")
    return out


def has_vacuum(atoms: Atoms, min_gap=5.0) -> bool:
    """True if there is a vacuum gap along z (slab, adsorption, interface)."""
    if not atoms.pbc[2] or len(atoms) < 2:
        return True
    cz = float(atoms.get_cell()[2, 2])
    if cz <= 0:
        return False
    z = np.sort(np.mod(atoms.positions[:, 2], cz))
    gaps = np.diff(z)
    wrap = cz - z[-1] + z[0]
    return float(max(gaps.max() if len(gaps) else 0.0, wrap)) >= min_gap


INCAR_BASE = """SYSTEM = {system}

# --- static single-point: energy, forces, stress for MLIP training ---
IBRION = -1
NSW    = 0
ISIF   = 2

# --- accuracy: held FIXED for the entire campaign ---
PREC   = Accurate
ENCUT  = {encut}
EDIFF  = 1E-06
LREAL  = .FALSE.
LASPH  = .TRUE.
ISYM   = 0
ALGO   = Normal
NELM   = 200
NELMIN = 5

# --- occupations ---
ISMEAR = 0
SIGMA  = {sigma}
ISPIN  = {ispin}

# --- output: keep the tree small across many folders ---
LWAVE  = .FALSE.
LCHARG = .FALSE.

# --- parallelisation ---
NCORE  = {ncore}
KPAR   = {kpar}
"""

INCAR_DIPOLE = """
# --- dipole correction: cell has vacuum along z ---
LDIPOL = .TRUE.
IDIPOL = 3
DIPOL  = {dx:.4f} {dy:.4f} {dz:.4f}
"""


def write_incar(path, atoms, encut, ncore, kpar, molecular=False,
                dipole=True, ispin_molecular=2, system="round0"):
    sigma = 0.03 if molecular else 0.05
    ispin = ispin_molecular if molecular else 1
    text = INCAR_BASE.format(system=system, encut=encut, sigma=sigma,
                             ispin=ispin, ncore=ncore, kpar=kpar)
    used_dipole = False
    if dipole and not molecular and has_vacuum(atoms):
        com = atoms.get_center_of_mass()
        frac = np.linalg.solve(np.asarray(atoms.get_cell()).T, com)
        frac = np.mod(frac, 1.0)
        text += INCAR_DIPOLE.format(dx=frac[0], dy=frac[1], dz=frac[2])
        used_dipole = True
    with open(path, "w") as fh:
        fh.write(text)
    return used_dipole


def write_manifest(frames, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "bucket", "kind", "formula", "natoms",
                    "min_distance", "notes"])
        for i, a in enumerate(frames):
            notes = {k: v for k, v in a.info.items()
                     if k not in ("bucket", "kind", "natoms", "min_distance")}
            w.writerow([i, a.info.get("bucket"), a.info.get("kind"),
                        a.get_chemical_formula(), len(a),
                        a.info.get("min_distance"), notes])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Generate round-0 seed structures for NEP training.")
    p.add_argument("--substrate", default="Si")
    p.add_argument("--substrate-structure", default="diamond")
    p.add_argument("--substrate-a", type=float, default=5.431)
    p.add_argument("--substrate-c", type=float, default=None)
    p.add_argument("--substrate-miller", default="111")
    p.add_argument("--film", default="AlN")
    p.add_argument("--film-structure", default="wurtzite")
    p.add_argument("--film-a", type=float, default=3.111)
    p.add_argument("--film-c", type=float, default=4.981)
    p.add_argument("--film-miller", default="001")
    p.add_argument("--max-atoms", type=int, default=ATOM_LIMIT - 1,
                   help=f"largest structure allowed, in atoms; clamped to "
                        f"{ATOM_LIMIT - 1} (structures are always < {ATOM_LIMIT})")
    p.add_argument("--max-atoms-interface", type=int, default=None,
                   help="separate cap for the interface bucket "
                        "(defaults to --max-atoms); also clamped to "
                        f"{ATOM_LIMIT - 1}")
    p.add_argument("--max-strain", type=float, default=0.09,
                   help="maximum interface lattice mismatch to accept. The "
                        "1.3%% 16:25 Si(111)/AlN(0001) match needs >= 164 "
                        "atoms, so under the 150-atom limit the default "
                        "0.09 lets the 6:9 match (8.1%%, 144 atoms) in")
    p.add_argument("--vacuum", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="round0_out/vasp")
    p.add_argument("--poscar-dir", default=None,
                   help="root for per-structure VASP folders "
                        "(default <outdir>); each becomes "
                        "<root>/<bucket>_<index>/POSCAR")
    p.add_argument("--no-poscar", action="store_true",
                   help="skip writing the per-structure VASP tree")
    p.add_argument("--kspacing", type=float, default=0.25,
                   help="reciprocal-space k-point density in 1/A used to "
                        "generate KPOINTS (fixed density, not fixed mesh)")
    p.add_argument("--no-kpoints", action="store_true")
    p.add_argument("--potcar-path", default="aaa-potential",
                   help="directory holding <element>/POTCAR; set to '' to "
                        "skip POTCAR assembly")
    p.add_argument("--encut", type=float, default=None,
                   help="fixed plane-wave cutoff in eV; default is "
                        "--encut-factor x the largest ENMAX in the POTCARs")
    p.add_argument("--encut-factor", type=float, default=1.3)
    p.add_argument("--no-incar", action="store_true")
    p.add_argument("--ncore", type=int, default=4)
    p.add_argument("--kpar", type=int, default=4)
    p.add_argument("--no-dipole", action="store_true",
                   help="skip LDIPOL/IDIPOL on cells containing vacuum")
    p.add_argument("--ispin-molecular", type=int, default=2, choices=(1, 2),
                   help="spin polarisation for isolated atoms and small "
                        "clusters (2 gives the correct atomic ground state)")
    for k, v in DEFAULT_COUNTS.items():
        p.add_argument(f"--n-{k.replace('_', '-')}", type=int, default=v,
                       dest=f"n_{k}")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    cap = min(args.max_atoms, ATOM_LIMIT - 1)
    if args.max_atoms > cap:
        print(f"! --max-atoms {args.max_atoms} clamped to {cap}: every "
              f"structure must have fewer than {ATOM_LIMIT} atoms",
              file=sys.stderr)

    sub = MaterialSpec(args.substrate, args.substrate_structure,
                       args.substrate_a, args.substrate_c,
                       parse_miller(args.substrate_miller))
    film = MaterialSpec(args.film, args.film_structure,
                        args.film_a, args.film_c,
                        parse_miller(args.film_miller))

    print(f"substrate : {sub.name} {sub.structure} {sub.miller} "
          f"elements={sub.elements}")
    print(f"film      : {film.name} {film.structure} {film.miller} "
          f"elements={film.elements}")
    print(f"atom cap  : {cap}\n")

    all_elements = sorted(set(sub.elements) | set(film.elements))
    frames = []

    print("generating buckets")
    frames += gen_bulk(sub, args.n_bulk_substrate, cap, rng)
    print(f"  bulk_substrate      {args.n_bulk_substrate}")
    frames += gen_bulk(film, args.n_bulk_film, cap, rng)
    print(f"  bulk_film           {args.n_bulk_film}")
    frames += gen_dimer_trimer(all_elements, args.n_dimer_trimer, rng=rng)
    print(f"  dimer_trimer        {args.n_dimer_trimer}")
    frames += gen_slabs(sub, args.n_slab_substrate, cap, rng, args.vacuum)
    print(f"  slab_substrate      {args.n_slab_substrate}")
    frames += gen_slabs(film, args.n_slab_film, cap, rng, args.vacuum)
    print(f"  slab_film           {args.n_slab_film}")
    frames += gen_adsorption(sub, film, args.n_adsorption, cap, rng,
                             args.vacuum)
    print(f"  adsorption          {args.n_adsorption}")
    icap = min(args.max_atoms_interface or cap, ATOM_LIMIT - 1)
    frames += gen_interface(sub, film, args.n_interface, icap, rng,
                            args.vacuum, args.max_strain,
                            os.path.join(args.outdir, "lattice_match.txt"))
    print(f"  interface           {args.n_interface}")
    frames += gen_isolated(all_elements, args.n_isolated_cluster, rng=rng)
    print(f"  isolated_cluster    {args.n_isolated_cluster}")
    frames += gen_disordered(sub, film, args.n_disordered, cap, rng,
                             args.vacuum)
    print(f"  disordered          {args.n_disordered}")

    kept, dropped = validate(frames, cap, interface_cap=icap)
    if dropped:
        print(f"\ndropped {len(dropped)} structures during validation:")
        for kind, why, val in dropped[:12]:
            print(f"  {kind:24s} {why} ({val})")

    write(os.path.join(args.outdir, "round0_all.extxyz"), kept)
    buckets = {}
    for a in kept:
        buckets.setdefault(a.info["bucket"], []).append(a)
    for name, group in buckets.items():
        write(os.path.join(args.outdir, f"{name}.extxyz"), group)
    write_manifest(kept, os.path.join(args.outdir, "manifest.csv"))

    if not args.no_poscar:
        root = args.poscar_dir or args.outdir
        potcar_root = args.potcar_path or None
        encut = args.encut

        if potcar_root:
            enmax, auto_encut = survey_potcars(
                potcar_root, all_elements, args.encut_factor)
            print(f"\nPOTCARs from {potcar_root}/")
            for sym in sorted(enmax):
                print(f"  {sym:3s} ENMAX {enmax[sym]:7.1f} eV")
            if encut is None:
                encut = auto_encut
                print(f"  ENCUT = {args.encut_factor} x "
                      f"{max(enmax.values()):.1f} -> {encut} eV "
                      f"(fixed for the whole campaign)")
            else:
                print(f"  ENCUT = {encut} eV (from --encut)")
        elif encut is None and not args.no_incar:
            print("\n! no --potcar-path and no --encut: skipping INCAR",
                  file=sys.stderr)

        records = write_vasp_tree(
            kept, root, kspacing=args.kspacing,
            write_kpts=not args.no_kpoints,
            potcar_root=potcar_root,
            encut=None if args.no_incar else encut,
            ncore=args.ncore, kpar=args.kpar,
            dipole=not args.no_dipole,
            ispin_molecular=args.ispin_molecular)
        meshes = sorted({r[6] for r in records})
        print(f"\nwrote {len(records)} VASP folders under {root}/")
        print(f"  {root}/<bucket>_<index>/POSCAR"
              + ("  + KPOINTS" if not args.no_kpoints else ""))
        print(f"  k-meshes used at {args.kspacing} 1/A: "
              f"{', '.join(meshes[:8])}"
              + (" ..." if len(meshes) > 8 else ""))
        n_dip = sum(1 for r in records if r[7] == "yes")
        contents = ["POSCAR"]
        if not args.no_kpoints:
            contents.append("KPOINTS")
        if potcar_root:
            contents.append("POTCAR")
        if encut and not args.no_incar:
            contents.append("INCAR")
        print(f"  each folder: {' '.join(contents)} info.json")
        if encut and not args.no_incar:
            print(f"  dipole correction applied to {n_dip} cells with vacuum")
        print(f"  folder list: {root}/dirlist.txt")
        print(f"  index: {root}/species_order.csv")

    sizes = [len(a) for a in kept]
    assert max(sizes) < ATOM_LIMIT, max(sizes)
    print(f"\nwrote {len(kept)} structures to {args.outdir}/")
    print(f"atoms per structure: min {min(sizes)}, "
          f"mean {np.mean(sizes):.1f}, max {max(sizes)}")
    print("\nper bucket:")
    for name in sorted(buckets):
        g = buckets[name]
        print(f"  {name:22s} {len(g):4d}  "
              f"atoms {min(len(a) for a in g)}-{max(len(a) for a in g)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
