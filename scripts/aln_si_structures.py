#!/usr/bin/env python3
"""Seed structures for an Al-N-Si NEP aimed at AlN deposition on Si(111).

generate_structures.py imports build_all() from here for round-0 seeds, and
generate_deposition_simulation.py uses si111_slab() for the substrate. Run it
directly to inspect the seeds: it writes one POSCAR per seed into --out
(default seeds/) plus a manifest.json with atom counts, minimum interatomic
distances and the in-plane strain applied to AlN in the interface cell.

Seeds cover the chemistry the potential has to see during growth:
  - bulk phases: wurtzite / zincblende / rocksalt AlN, diamond Si, fcc Al
  - surfaces:    Si(111) slab, AlN(0001) slab
  - interface:   AlN(0001)/Si(111) in the 5:4 coincidence cell (~1.3% AlN strain)
  - deposition:  Al and N adatoms above AlN(0001) and Si(111), an N2 molecule

Positions are unrelaxed starting points. They get rattled/strained by
scripts/generate_structures.py and relaxed or run with AIMD in VASP, so they
only need to be chemically sensible, not at the DFT minimum.

Requires ASE (`pip install ase`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Experimental room-temperature lattice parameters (Angstrom)
A_SI = 5.431
A_ALN, C_ALN, U_ALN = 3.112, 4.982, 0.382
A_AL = 4.05

# 5 AlN(0001) cells on 4 Si(111) surface cells: 5*3.112 = 15.56 vs 4*3.840 = 15.36
ALN_REPEAT, SI_REPEAT = 5, 4


def _require_ase():
    try:
        import ase  # noqa: F401
    except ImportError:
        sys.exit("ASE is required (`pip install ase`).")


def _layers(atoms, tol=0.2):
    """Group atom indices into z layers, bottom to top."""
    order = np.argsort(atoms.positions[:, 2])
    layers, current, z0 = [], [order[0]], atoms.positions[order[0], 2]
    for i in order[1:]:
        if atoms.positions[i, 2] - z0 > tol:
            layers.append(current)
            current, z0 = [], atoms.positions[i, 2]
        current.append(i)
    layers.append(current)
    return layers


def _upward_neighbors(atoms, i, cutoff=2.1):
    """Number of Al-N neighbours of atom i that sit above it (non-periodic in z)."""
    d = atoms.get_distances(i, range(len(atoms)), mic=True, vector=True)
    r = np.linalg.norm(d, axis=1)
    return int(np.sum((r > 0.1) & (r < cutoff) & (d[:, 2] > -0.1)))


def aln_slab(repeat, n_cells, polarity="Al", bottom="Al"):
    """AlN(0001) slab with a 60-degree in-plane cell (matches ASE's fcc/diamond111).

    polarity: "Al" puts the vertical Al->N bond pointing +z (Al-polar growth).
    bottom:   species of the bottom layer, i.e. the one that bonds to the substrate.
              For an Al-polar film a bottom N layer has one dangling bond per atom
              (Si-N interface); a bottom Al layer has three, which is the Al
              wetting layer left by Al pre-deposition.
    """
    from ase.build import bulk, make_supercell

    unit = bulk("AlN", "wurtzite", a=A_ALN, c=C_ALN, u=U_ALN)
    # ASE's wurtzite cell has a 120-degree gamma; a2' = a1 + a2 gives 60 degrees
    unit = make_supercell(unit, [[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    unit.wrap()

    # Polarity check on the bulk cell: find the Al-N bond along c
    al = [i for i, s in enumerate(unit.get_chemical_symbols()) if s == "Al"][0]
    d = unit.get_distances(al, range(len(unit)), mic=True, vector=True)
    vertical = [v for v, s in zip(d, unit.get_chemical_symbols())
                if s == "N" and abs(np.linalg.norm(v[:2])) < 0.1 and np.linalg.norm(v) < 2.1]
    is_al_polar = vertical[0][2] > 0
    if is_al_polar != (polarity == "Al"):
        pos = unit.get_scaled_positions()
        pos[:, 2] = (-pos[:, 2]) % 1.0
        unit.set_scaled_positions(pos)

    slab = unit.repeat((repeat, repeat, n_cells + 1))
    slab.pbc = (True, True, False)

    # Trim from the bottom until the bottom layer is the requested species and
    # still bonded into the slab (no isolated half-bilayer left hanging).
    while True:
        bottom_layer = _layers(slab)[0]
        species = {slab[i].symbol for i in bottom_layer}
        if species == {bottom} and all(_upward_neighbors(slab, i) > 0 for i in bottom_layer):
            break
        del slab[bottom_layer]
    # Trim the top back down to n_cells formula-unit layers
    target = repeat * repeat * 2 * n_cells
    while len(slab) > target:
        del slab[_layers(slab)[-1]]
    slab.positions[:, 2] -= slab.positions[:, 2].min()
    return slab


def si111_slab(repeat, n_bilayers, vacuum=0.0):
    from ase.build import diamond111

    # diamond111 layers = atomic layers; 2 per bilayer
    slab = diamond111("Si", (repeat, repeat, 2 * n_bilayers), a=A_SI, vacuum=vacuum)
    slab.positions[:, 2] -= slab.positions[:, 2].min()
    return slab


def interface(n_si_bilayers, n_aln_cells, polarity, termination, gap, vacuum):
    """AlN(0001)/Si(111) 5:4 coincidence cell. Si keeps its lattice; AlN is
    strained in-plane to match (Si is the thick substrate in the experiment)."""
    si = si111_slab(SI_REPEAT, n_si_bilayers)
    aln = aln_slab(ALN_REPEAT, n_aln_cells, polarity=polarity, bottom=termination)

    strain = si.cell[0, 0] / aln.cell[0, 0] - 1.0
    new_cell = aln.cell.array.copy()
    new_cell[:2] = si.cell.array[:2]
    aln.set_cell(new_cell, scale_atoms=True)

    aln.positions[:, 2] += si.positions[:, 2].max() + gap
    combined = si + aln
    combined.set_cell([si.cell[0], si.cell[1], [0, 0, combined.positions[:, 2].max()]])
    combined.center(vacuum=vacuum, axis=2)
    combined.pbc = (True, True, True)
    return combined, strain


def with_adatom(slab, symbol, height, vacuum):
    from ase import Atom

    s = slab.copy()
    top = s.positions[:, 2].max()
    xy = 0.5 * (s.cell[0] + s.cell[1])[:2]  # cell centre, generic (non-top) site
    s.append(Atom(symbol, (xy[0], xy[1], top + height)))
    s.center(vacuum=vacuum, axis=2)
    s.pbc = (True, True, True)
    return s


def build_all(polarity="Al", termination="Al", si_bilayers=4, aln_cells=2,
              gap=2.3, vacuum=7.5):
    """Return ({name: Atoms}, aln_interface_strain) for every seed structure."""
    from ase import Atoms
    from ase.build import bulk

    seeds = {}
    seeds["bulk_AlN_wurtzite"] = bulk("AlN", "wurtzite", a=A_ALN, c=C_ALN, u=U_ALN).repeat((3, 3, 2))
    seeds["bulk_AlN_zincblende"] = bulk("AlN", "zincblende", a=4.38, cubic=True).repeat(2)
    seeds["bulk_AlN_rocksalt"] = bulk("AlN", "rocksalt", a=4.05, cubic=True).repeat(2)
    seeds["bulk_Si_diamond"] = bulk("Si", "diamond", a=A_SI, cubic=True).repeat(2)
    seeds["bulk_Al_fcc"] = bulk("Al", "fcc", a=A_AL, cubic=True).repeat(2)

    n2 = Atoms("N2", positions=[(0, 0, 0), (0, 0, 1.098)], cell=[12, 12, 12], pbc=True)
    n2.center()
    seeds["molecule_N2"] = n2

    si_surf = si111_slab(2, 4)
    si_surf.center(vacuum=vacuum, axis=2)
    si_surf.pbc = True
    seeds["slab_Si111"] = si_surf

    aln_surf = aln_slab(3, 3, polarity=polarity, bottom=termination)
    aln_surf.center(vacuum=vacuum, axis=2)
    aln_surf.pbc = True
    seeds["slab_AlN0001"] = aln_surf

    for sym in ("Al", "N"):
        seeds[f"adatom_{sym}_on_Si111"] = with_adatom(si111_slab(2, 4), sym, 2.0, vacuum)
        seeds[f"adatom_{sym}_on_AlN0001"] = with_adatom(
            aln_slab(3, 3, polarity=polarity, bottom=termination), sym, 2.0, vacuum)

    iface, strain = interface(si_bilayers, aln_cells, polarity, termination, gap, vacuum)
    seeds[f"interface_AlN0001_Si111_{termination}-Si"] = iface
    return seeds, strain


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "seeds")
    parser.add_argument("--polarity", choices=["Al", "N"], default="Al",
                        help="AlN growth polarity (Al-polar is typical on Si(111))")
    parser.add_argument("--termination", choices=["Al", "N"], default="Al",
                        help="AlN species bonded to Si at the interface "
                             "(Al: Al pre-deposition to avoid SiNx; N: nitrided Si)")
    parser.add_argument("--si-bilayers", type=int, default=4)
    parser.add_argument("--aln-cells", type=int, default=2)
    parser.add_argument("--gap", type=float, default=2.3,
                        help="Si-to-AlN starting separation in Angstrom (Al-Si bond ~2.5, N-Si ~1.75)")
    parser.add_argument("--vacuum", type=float, default=7.5,
                        help="vacuum on each side of slabs, Angstrom")
    args = parser.parse_args()

    _require_ase()
    from ase.io import write

    seeds, strain = build_all(args.polarity, args.termination, args.si_bilayers,
                              args.aln_cells, args.gap, args.vacuum)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"aln_inplane_strain_in_interface": round(strain, 5), "structures": {}}
    for name, atoms in seeds.items():
        path = args.out / name / "POSCAR"
        path.parent.mkdir(exist_ok=True)
        write(path, atoms, format="vasp", direct=True, sort=True)
        manifest["structures"][name] = {
            "n_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "min_distance_A": round(float(np.min(
                atoms.get_all_distances(mic=True)[np.triu_indices(len(atoms), 1)])), 3),
        }
        print(f"{name:40s} {len(atoms):5d} atoms  {atoms.get_chemical_formula()}")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"AlN in-plane strain in interface cell: {strain * 100:+.2f}%")
    print(f"Wrote {len(seeds)} seeds to {args.out}")


if __name__ == "__main__":
    main()
