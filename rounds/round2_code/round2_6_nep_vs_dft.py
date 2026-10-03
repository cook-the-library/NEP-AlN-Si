#!/usr/bin/env python3
"""
Round 2, step 6: how wrong was the NEP on the structures its own MD made?

round2_3_harvest.py saved the NEP energy and forces of every selected frame
(selected_nep.extxyz). Once VASP has labelled those frames and the dataset
builder has turned them into train/test xyz, this script pairs the two by
folder and reports the NEP error per bucket and stage.

This is the most direct measure of "incomplete" there is: the error of the
potential on the configurations it actually visits in MD, not on a test set
someone built by hand. It is also the baseline that the NEP retrained on
these frames has to beat.

Frames the harvester modified (atoms far from the slab removed) have no NEP
energy for the cell that went to VASP and are skipped.

Usage (from the project root):
    python round2_code/round2_6_nep_vs_dft.py \\
        --nep-xyz round2_out/vasp_md/selected_nep.extxyz \\
        --dft round2_out/dataset_new/train.xyz round2_out/dataset_new/test.xyz \\
        --out round2_out/dataset_new/nep_vs_dft.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from ase.io import read


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--nep-xyz", default="round2_out/vasp_md/selected_nep.extxyz")
    p.add_argument("--dft", nargs="+", default=["round2_out/dataset_new/train.xyz",
                                                "round2_out/dataset_new/test.xyz"])
    p.add_argument("--out", default="round2_out/dataset_new/nep_vs_dft.txt")
    a = p.parse_args(argv)

    nep = {}
    for at in read(a.nep_xyz, index=":"):
        if "nep_energy" in at.info and int(at.info.get("n_detached_removed", 0)) == 0:
            nep[str(at.info["folder"])] = at
    dft = {}
    for path in a.dft:
        if os.path.isfile(path):
            for at in read(path, index=":"):
                dft[str(at.info.get("folder", ""))] = at

    rows = defaultdict(lambda: {"de": [], "df": [], "fmax_err": []})
    skipped = 0
    for folder, n in nep.items():
        d = dft.get(folder)
        if d is None or len(d) != len(n) or \
                d.get_chemical_symbols() != n.get_chemical_symbols():
            skipped += 1
            continue
        e_dft = d.info["energy"] if "energy" in d.info else d.get_potential_energy()
        f_dft = d.arrays["forces"] if "forces" in d.arrays else d.get_forces()
        f_nep = n.arrays["nep_forces"]
        de = (float(n.info["nep_energy"]) - float(e_dft)) / len(n)
        df = (f_nep - f_dft).ravel()
        tag = "pre_failure" if n.info.get("pre_failure") in (True, "True") else ""
        for key in [(n.info["bucket"], n.info["kind"], ""), (n.info["bucket"], "all", ""),
                    ("all", "all", "")] + ([("pre_failure", "all", "")] if tag else []):
            rows[key]["de"].append(de)
            rows[key]["df"].extend(df.tolist())
            rows[key]["fmax_err"].append(float(np.abs(f_nep - f_dft).max()))

    L = [f"NEP (the potential that drove the MD) vs DFT on {sum(len(v['de']) for k, v in rows.items() if k[0] == 'all')} "
         f"MD-sampled frames ({skipped} without a matching DFT frame)", "",
         f"{'bucket':16s} {'stage':8s} {'n':>5s} {'E RMSE':>10s} {'E mean':>10s} "
         f"{'F RMSE':>10s} {'worst |dF|':>11s}",
         f"{'':16s} {'':8s} {'':>5s} {'meV/atom':>10s} {'meV/atom':>10s} "
         f"{'meV/A':>10s} {'eV/A':>11s}"]
    for key in sorted(rows, key=lambda k: (k[0] == "all", k[0] == "pre_failure", k)):
        v = rows[key]
        de = np.array(v["de"]) * 1000
        df = np.array(v["df"]) * 1000
        L.append(f"{key[0]:16s} {key[1]:8s} {len(de):5d} {np.sqrt((de ** 2).mean()):10.1f} "
                 f"{de.mean():10.1f} {np.sqrt((df ** 2).mean()):10.1f} "
                 f"{max(v['fmax_err']):11.2f}")
    L.append("")
    L.append("Compare with the RMSEs in loss.out: an error much larger here means")
    L.append("the training set did not cover what the MD visits.")
    text = "\n".join(L) + "\n"
    with open(a.out, "w") as fh:
        fh.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
