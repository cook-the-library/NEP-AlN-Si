#!/usr/bin/env python3
"""
Round 2: drop old training frames whose geometry is unphysical, before the
round-1wL dataset is merged with the round-2 frames.

The round 0/1 generator (used up to round 1.7) produced two kinds of broken
structures, both of which a pair-specific contact ratio
    min over pairs of d_ij / (r_cov,i + r_cov,j)
finds without knowing where a frame came from:

  * every interface structure has its AlN film sheared: 50 Al-N pairs at
    1.19 A (ratio 0.62), unrattled registry/separation scans included
  * random placements were checked against the N-N distance for every pair:
    disordered_film_on_slab has Al-Al at ~0.9 A (ratio 0.37), and some
    interstitial, cluster and large-rattle frames are as bad

Condensed frames below --min-contact are dropped. Gas-phase frames (bucket
matching --gas-pattern, or <= 3 atoms) are kept: the dimer and trimer scans go
to short distances on purpose. Frames are copied verbatim, so the train/test
split and every tag are untouched.

Outputs in --outdir: train.xyz, test.xyz, dropped.csv, report.txt

Usage:
    python round2_code/round2_5_contact_filter.py \\
        --train round1wL_out/dataset/train.xyz --test round1wL_out/dataset/test.xyz \\
        --outdir round2_out/dataset_prev_filtered

Requires numpy and ase.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from ase import Atoms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round2_common import contact_ratio, pair_distances  # noqa: E402

TAG_RE = re.compile(r'([A-Za-z_][\w.\-]*)=("[^"]*"|\'[^\']*\'|\{[^}]*\}|\S+)')


def parse_tags(comment):
    tags = {}
    for m in TAG_RE.finditer(comment):
        v = m.group(2)
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        tags[m.group(1).lower()] = v
    return tags


def read_frames(path):
    """Yield (count_line, comment_line, atom_lines) verbatim."""
    with open(path) as fh:
        while True:
            head = fh.readline()
            if not head:
                return
            if not head.strip():
                continue
            n = int(head.split()[0])
            comment = fh.readline()
            atoms = [fh.readline() for _ in range(n)]
            if not comment or (n and not atoms[-1]):
                sys.exit(f"ERROR: {path}: file ends inside a frame")
            yield head, comment, atoms


def to_atoms(comment, lines):
    tags = parse_tags(comment)
    lat = [float(x) for x in tags["lattice"].split()]
    props = tags["properties"].split(":")
    off, s0, p0 = 0, None, None
    for i in range(0, len(props), 3):
        name, n = props[i].lower(), int(props[i + 2])
        if name == "species":
            s0 = off
        elif name == "pos":
            p0 = off
        off += n
    syms, pos = [], []
    for ln in lines:
        w = ln.split()
        syms.append(w[s0])
        pos.append([float(w[p0]), float(w[p0 + 1]), float(w[p0 + 2])])
    pbc = [c == "T" for c in tags.get("pbc", "T T T").split()]
    return Atoms(syms, positions=pos, cell=np.reshape(lat, (3, 3)), pbc=pbc), tags


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--min-contact", type=float, default=0.65,
                   help="drop condensed frames with min d/(r_cov,i+r_cov,j) "
                        "below this (sheared interfaces sit at 0.62)")
    p.add_argument("--gas-pattern", default=r"dimer|trimer|cluster|isolated")
    a = p.parse_args(argv)

    gas_re = re.compile(a.gas_pattern, re.I)
    os.makedirs(a.outdir, exist_ok=True)
    stats = defaultdict(Counter)
    ratios = defaultdict(list)
    dropped = []
    for split, path in (("train", a.train), ("test", a.test)):
        out_path = os.path.join(a.outdir, f"{split}.xyz")
        with open(out_path + ".tmp", "w") as out:
            for k, (head, comment, lines) in enumerate(read_frames(path)):
                atoms, tags = to_atoms(comment, lines)
                bucket = tags.get("config_type", tags.get("bucket", "unknown"))
                src = tags.get("data_source", "?")
                key = (src, bucket)
                stats[key]["in"] += 1
                gas = bool(gas_re.search(bucket)) or len(atoms) <= 3
                if gas:
                    out.write(head + comment)
                    out.writelines(lines)
                    stats[key]["gas_kept"] += 1
                    continue
                ratio, pair = contact_ratio(atoms, pair_distances(atoms, 2.6))
                ratios[key].append(ratio)
                if ratio < a.min_contact:
                    stats[key]["dropped"] += 1
                    dropped.append({"split": split, "frame": k, "data_source": src,
                                    "bucket": bucket, "kind": tags.get("kind", ""),
                                    "folder": tags.get("folder", ""),
                                    "natoms": len(atoms), "contact_ratio": f"{ratio:.4f}",
                                    "closest_pair": pair})
                    continue
                out.write(head + comment)
                out.writelines(lines)
        os.replace(out_path + ".tmp", out_path)

    with open(os.path.join(a.outdir, "dropped.csv"), "w", newline="") as fh:
        cols = ["split", "frame", "data_source", "bucket", "kind", "folder",
                "natoms", "contact_ratio", "closest_pair"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(dropped)

    L = [f"contact filter: condensed frames with min d/(r_cov,i+r_cov,j) < "
         f"{a.min_contact} dropped; gas-phase frames kept", "",
         f"{'source':10s} {'bucket':22s} {'in':>6s} {'dropped':>8s} "
         f"{'ratio min':>10s} {'median':>8s}"]
    for key in sorted(stats):
        r = ratios.get(key, [])
        L.append(f"{key[0][:10]:10s} {key[1][:22]:22s} {stats[key]['in']:6d} "
                 f"{stats[key]['dropped']:8d} "
                 + (f"{min(r):10.3f} {np.median(r):8.3f}" if r else f"{'gas':>10s}"))
    n_in = sum(s["in"] for s in stats.values())
    L.append("")
    L.append(f"total: {n_in} in, {len(dropped)} dropped "
             f"({Counter(d['split'] for d in dropped)['test']} of them from test)")
    by_kind = Counter((d["bucket"], d["kind"]) for d in dropped)
    if by_kind:
        L.append("dropped by bucket / kind:")
        for (b, k), n in by_kind.most_common():
            L.append(f"  {n:6d}  {b} / {k or '?'}")
    text = "\n".join(L) + "\n"
    with open(os.path.join(a.outdir, "report.txt"), "w") as fh:
        fh.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
