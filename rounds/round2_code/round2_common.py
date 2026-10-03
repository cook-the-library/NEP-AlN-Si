"""
Round 2: structure builders and helpers shared by the MD builder
(round2_1_build_md_inputs.py), the thermal-rattle generator
(round2_1_thermal_rattle.py) and the harvester (round2_3_harvest.py).

Two things here fix the round 0/1 generator (see rounds/README.md):

  * slabs are cut BETWEEN bilayers, so every surface atom has one dangling
    bond. ase.build.surface / diamond111 cut through the bilayers and leave
    the outermost atoms singly bonded (three dangling bonds each).
  * the AlN/Si(111) interface puts a 60-degree AlN cell on the 60-degree
    Si(111) cell, so mapping the film onto the substrate is a uniform
    in-plane scale (-1.25 %), not a shear. generate_round1.py mapped a
    120-degree film cell onto the 60-degree substrate cell, which put
    50 Al-N pairs at 1.19 A in every interface structure.

Requires numpy and ase.
"""

from __future__ import annotations

import itertools
import math
import os
import re

import numpy as np
from ase import Atoms
from ase.build import bulk, diamond111, make_supercell
from ase.data import covalent_radii

# Lattice parameters (A); the same experimental values as generate_round1.py
A_SI = 5.431
A_ALN, C_ALN, U_ALN = 3.111, 4.981, 0.382


# --------------------------------------------------------------------------
# NEP header
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


# --------------------------------------------------------------------------
# distances
# --------------------------------------------------------------------------


def pair_distances(atoms: Atoms, rc: float):
    """(i, j, d) for every ordered pair i != j (and periodic self-images)
    closer than rc. Brute force over the periodic images that can matter:
    for the <= 200-atom cells here this is far faster than a cell list."""
    pos = atoms.positions
    cell = np.array(atoms.cell)
    ranges = []
    vol = abs(np.linalg.det(cell))
    for ax in range(3):
        if not atoms.pbc[ax] or vol <= 0:
            ranges.append([0])
            continue
        b, c = cell[(ax + 1) % 3], cell[(ax + 2) % 3]
        height = vol / np.linalg.norm(np.cross(b, c))
        m = int(math.ceil(rc / height))
        ranges.append(range(-m, m + 1))
    I, J, D = [], [], []
    diff = pos[None, :, :] - pos[:, None, :]          # r_j - r_i
    for shift in itertools.product(*ranges):
        sv = np.asarray(shift, float) @ cell
        d = np.sqrt(((diff + sv) ** 2).sum(-1))
        if not any(shift):
            np.fill_diagonal(d, np.inf)
        ii, jj = np.nonzero(d < rc)
        I.append(ii)
        J.append(jj)
        D.append(d[ii, jj])
    if not I:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)


def contact_ratio(atoms: Atoms, pairs=None, cutoff: float = 2.6):
    """min over pairs of d_ij / (r_cov,i + r_cov,j), and a label for that pair.
    A pair-specific measure: 0.9 A is a normal N-N contact ratio of 0.63 but
    an Al-Al ratio of 0.37."""
    i, j, d = pairs if pairs is not None else pair_distances(atoms, cutoff)
    if len(d) == 0:
        return 9.99, ""
    radii = covalent_radii[atoms.numbers]
    r = d / (radii[i] + radii[j])
    k = int(np.argmin(r))
    s = atoms.get_chemical_symbols()
    return float(r[k]), f"{s[i[k]]}-{s[j[k]]} {d[k]:.3f}"


def components(atoms: Atoms, bond: float = 3.0):
    """Connected components (list of index arrays), largest first."""
    i, j, _ = pair_distances(atoms, bond)
    parent = np.arange(len(atoms))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(i, j):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(a) for a in range(len(atoms))])
    groups = [np.where(roots == r)[0] for r in np.unique(roots)]
    return sorted(groups, key=len, reverse=True)


# --------------------------------------------------------------------------
# slabs along z
# --------------------------------------------------------------------------


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


def contiguous_along_z(atoms: Atoms) -> Atoms:
    """Shift atoms across the periodic z boundary so the slab is in one piece
    (the largest gap in z becomes the vacuum); the slab starts at z = 0."""
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


def remove_detached(atoms: Atoms, detach: float = 4.0, bond: float = 3.0):
    """Drop atoms/molecules farther than `detach` A from the largest piece.
    Returns (atoms, number removed)."""
    groups = components(atoms, bond)
    main = groups[0]
    drop = []
    for g in groups[1:]:
        probe = atoms[np.concatenate([main, g])]
        ii, jj, _ = pair_distances(probe, detach)
        nm = len(main)
        if not np.any((ii < nm) & (jj >= nm)):
            drop.extend(g.tolist())
    out = atoms.copy()
    if drop:
        del out[sorted(drop)]
    return out, len(drop)


def with_vacuum(slab: Atoms, vacuum: float) -> Atoms:
    """Periodic cell for VASP: the slab in one piece, `vacuum` A of empty
    space along z in total, centred."""
    a = slab.copy()
    if a.pbc[2]:
        a = contiguous_along_z(a)
    a.positions[:, 2] -= a.positions[:, 2].min()
    cell = np.array(a.cell)
    cell[2] = [0.0, 0.0, a.positions[:, 2].max() + vacuum]
    a.set_cell(cell, scale_atoms=False)
    a.positions[:, 2] += 0.5 * vacuum
    a.pbc = (True, True, True)
    return a


def coordination(atoms: Atoms, cutoff: float) -> np.ndarray:
    i, _, _ = pair_distances(atoms, cutoff)
    return np.bincount(i, minlength=len(atoms))


def trim_to_bilayers(slab: Atoms, n_bilayers: int) -> Atoms:
    """Keep n complete bilayers (closely spaced layer pairs), so the top and
    bottom atoms each have ONE dangling bond."""
    layers = layers_by_z(slab)
    z = [slab.positions[l[0], 2] for l in layers]
    start = next(i for i in range(len(layers) - 1) if z[i + 1] - z[i] < 1.0)
    if len(layers) - start < 2 * n_bilayers:
        raise ValueError("not enough layers to cut the requested bilayers")
    keep = [i for l in layers[start:start + 2 * n_bilayers] for i in l]
    out = slab[sorted(keep)]
    out.positions[:, 2] -= out.positions[:, 2].min()
    out.cell[2] = [0.0, 0.0, out.positions[:, 2].max()]
    return out


def si111_slab(repeat: int, n_bilayers: int) -> Atoms:
    """Bulk-terminated Si(111): n bilayers, 60-degree cell, z from 0."""
    slab = diamond111("Si", (repeat, repeat, 2 * n_bilayers + 2), a=A_SI,
                      vacuum=0.0)
    slab.pbc = (True, True, False)
    return trim_to_bilayers(slab, n_bilayers)


def aln_slab(repeat: int, n_bilayers: int, top: str = "Al") -> Atoms:
    """AlN slab with `top` = 'Al' (Al-polar (0001), the usual growth face on
    Si(111)) or 'N' (N-polar (000-1)); 60-degree cell, z from 0."""
    unit = bulk("AlN", "wurtzite", a=A_ALN, c=C_ALN, u=U_ALN)
    unit = make_supercell(unit, [[1, 0, 0], [1, 1, 0], [0, 0, 1]])  # 60 deg
    slab = unit.repeat((repeat, repeat, n_bilayers // 2 + 2))
    slab.pbc = (True, True, False)
    # ASE's wurtzite has the vertical Al->N bond pointing -z (N-polar along
    # +z). Mirror z for an Al-polar slab.
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


def check_surface(slab: Atoms, cutoff: float, want_top=None, bottom=True):
    """Every top (and bottom) atom must have coordination 3."""
    s = slab.copy()
    s.pbc = (True, True, False)
    cn = coordination(s, cutoff)
    z = s.positions[:, 2]
    top = z > z.max() - 0.2
    bot = z < z.min() + 0.2
    sym = np.array(s.get_chemical_symbols())
    if set(cn[top]) != {3} or (bottom and set(cn[bot]) != {3}):
        raise RuntimeError(f"surface coordination top {set(cn[top])} "
                           f"bottom {set(cn[bot])}, expected 3")
    if want_top and set(sym[top]) != {want_top}:
        raise RuntimeError(f"top layer is {set(sym[top])}, expected {want_top}")


# Si(111) 4x4 (15.36 A) carries AlN(0001) 5x5 (15.56 A): 5:4 coincidence,
# AlN compressed in-plane by 1.25 %. Si keeps its lattice (thick substrate).
def aln_si_interface(si_repeat: int = 4, aln_repeat: int = 5,
                     si_bilayers: int = 2, aln_bilayers: int = 2,
                     termination: str = "Al", polarity: str = "Al",
                     gap: float | None = None, shift=(0.0, 0.0)) -> Atoms:
    """AlN(0001) film on Si(111), no vacuum yet (pbc T T F, z from 0).

    termination: the AlN species bonded to Si.
      Al-polar film: 'N' gives the bottom N of a complete bilayer (one bond
      to Si each); 'Al' adds the Al layer below it, with three dangling bonds
      (the Al wetting layer of Al pre-deposition).
    gap: vertical Si-to-film spacing; default 2.3 A for Al, 1.9 A for N.
    shift: in-plane offset of the film (registry), A."""
    if gap is None:
        gap = 2.3 if termination == "Al" else 1.9
    si = si111_slab(si_repeat, si_bilayers)
    aln = aln_slab(aln_repeat, aln_bilayers + 1, top=polarity)
    layers = layers_by_z(aln)
    bottom = {aln[i].symbol for i in layers[0]}
    if bottom == {termination}:
        del aln[sorted(layers[0] + layers[1])]       # one bond per atom to Si
    else:
        del aln[sorted(layers[0])]                    # three per atom
    aln.positions[:, 2] -= aln.positions[:, 2].min()

    new_cell = np.array(aln.cell)
    new_cell[:2] = np.array(si.cell)[:2]
    if abs(np.dot(new_cell[0], new_cell[1]) / np.linalg.norm(new_cell[0]) ** 2
           - np.dot(np.array(aln.cell)[0], np.array(aln.cell)[1])
           / np.linalg.norm(np.array(aln.cell)[0]) ** 2) > 1e-6:
        raise RuntimeError("film and substrate cells differ in angle; refusing "
                           "to shear the film")
    strain = np.linalg.norm(new_cell[0]) / np.linalg.norm(np.array(aln.cell)[0]) - 1
    aln.set_cell(new_cell, scale_atoms=True)
    aln.positions[:, 0] += shift[0]
    aln.positions[:, 1] += shift[1]
    aln.positions[:, 2] += si.positions[:, 2].max() + gap
    out = si + aln
    out.set_cell([si.cell[0], si.cell[1], [0.0, 0.0, out.positions[:, 2].max()]])
    out.pbc = (True, True, False)
    out.wrap()
    out.info.update({"n_substrate": len(si), "n_film": len(aln),
                     "film_strain": round(float(strain), 5),
                     "termination": termination, "polarity": polarity,
                     "gap": float(gap)})
    return out


# --------------------------------------------------------------------------
# VASP parallelisation
# --------------------------------------------------------------------------


def irreducible_kpoints(mesh) -> int:
    """Gamma-centred mesh reduced by time reversal only (ISYM = 0)."""
    m = [int(x) for x in mesh]
    seen = set()
    for idx in itertools.product(*(range(n) for n in m)):
        neg = tuple((-i) % n for i, n in zip(idx, m))
        seen.add(min(idx, neg))
    return len(seen)


def set_vasp_parallel(folder: str, ntasks: int = 16, ncore: int = 4,
                      kpar_max: int = 4):
    """NCORE/KPAR for `ntasks` MPI ranks: KPAR is the largest divisor of
    ntasks/ncore that is <= kpar_max and <= the irreducible k-points, so no
    k-point group sits idle. Rewrites the INCAR in place; returns (ncore, kpar)."""
    with open(os.path.join(folder, "KPOINTS")) as fh:
        mesh = fh.read().split("\n")[3].split()[:3]
    nk = irreducible_kpoints(mesh)
    per = ntasks // ncore
    kpar = max(d for d in range(1, per + 1)
               if per % d == 0 and d <= kpar_max and d <= nk)
    path = os.path.join(folder, "INCAR")
    with open(path) as fh:
        text = fh.read()
    text = re.sub(r"(?m)^NCORE\s*=.*$", f"NCORE  = {ncore}", text)
    text = re.sub(r"(?m)^KPAR\s*=.*$", f"KPAR   = {kpar}", text)
    with open(path, "w") as fh:
        fh.write(text)
    return ncore, kpar
