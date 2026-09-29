#!/usr/bin/env python3
"""
Round-1wL dataset: the round-1.7 cumulative dataset (rounds 0 + 1 + 1.5 + 1.7)
plus the legacy class-B+C frames, with abnormal labels rejected.

Nothing is re-labelled or re-split. Each source already carries its own
family-based train/test split (round builder: carried split manifest;
legacy: scan_legacy.py), so a source's train frames go to train.xyz and its
test frames to test.xyz. Kept frames are copied VERBATIM -- atom lines and
comment line untouched -- except for one appended tag, data_source=<name>,
which GPUMD ignores and which lets the error tools split errors by source.

Rejection rules, applied to every frame of every source, first match wins
(so each rejected frame has exactly one reason):

  malformed      energy, Lattice, Properties, species or force column missing
  nonfinite      energy or a force component is nan/inf
  species        an element that is not on the nep.in `type` line
  dropped        whole (source, bucket) named with --drop SOURCE:BUCKET[:REASON];
                 REASON (no spaces) goes to the detail column of rejected.csv
                 and to the report. Round 1wL drops round-1.7 disordered,
                 slab_AlN and slab_Si for too-low E/atom (set in the sbatch);
                 the legacy frames of those buckets are kept.
  e_below_floor  E/atom < --emin (default -8.5 eV/atom)
                 N2 (-8.33 eV/atom in these PBE data) has the lowest energy
                 per atom of anything in N-Al-Si-Ar; the nitrides (AlN -7.45,
                 Si3N4 about -8.1) and the elements all sit above it. A frame
                 below the floor is an SCF blow-up, e.g. the round-1.7
                 slab_AlN frame at -612 eV/atom and disordered at -28.
  e_above_ceil   condensed frame with E/atom > --emax-condensed (default 0)
                 VASP's zero is the POTCAR reference atoms, i.e. roughly free
                 atoms. A bulk/slab/interface cell whose AVERAGE energy is
                 above free atoms has overlapping cores all over the cell,
                 which no deposition event produces.
                 Gas-phase frames (bucket matches --gas-pattern, or <= 3
                 atoms) are exempt: the dimer/trimer scans go down to 1.0 A
                 and reach +80 eV/atom by design. --emax-gas caps them too.
  f_above_cap    max |F| > --fmax (default 500 eV/A). The round builder
                 already dropped frames at this level (its smallest rejection
                 was 557 eV/A); the legacy scan had no cap (max_force=None).

Outputs in --outdir: train.xyz, test.xyz, report.txt, rejected.csv.
train/test are written to *.tmp and renamed only after every check passes,
so a failed build never leaves a half dataset for the NEP job to pick up.

Standard library only (runs on any python >= 3.6, no conda env needed).

Usage (from the project root; the sbatch wrapper does this):
    python round1wL_code/round1wL_5_build_dataset.py \
        --source r1.7     round1.7_out/dataset/train.xyz  round1.7_out/dataset/test.xyz \
        --source legacyBC roundL_out/dataset_legacy/BC/train.xyz roundL_out/dataset_legacy/BC/test.xyz \
        --nep-in round1wL_code/round1wL_nep-1.in --outdir round1wL_out/dataset
"""

import argparse
import collections
import csv
import math
import os
import re
import statistics
import sys

TAG_RE = re.compile(r'([A-Za-z_][\w.\-]*)=("[^"]*"|\'[^\']*\'|\{[^}]*\}|\S+)')
BUCKET_KEYS = ("bucket", "config_type")
FOLDER_KEYS = ("folder", "directory", "path", "dir", "source_dir", "outcar",
               "structure", "name")
SOURCE_TAG = "data_source"
REASONS = ("malformed", "nonfinite", "species", "dropped",
           "e_below_floor", "e_above_ceil", "f_above_cap")


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_tags(comment):
    tags = {}
    for m in TAG_RE.finditer(comment):
        v = m.group(2)
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        tags[m.group(1).lower()] = v
    return tags


def parse_properties(spec):
    """'species:S:1:pos:R:3:force:R:3' -> {'species': (0, 1), ...}"""
    f = spec.split(":")
    if len(f) % 3:
        raise ValueError("Properties has %d fields" % len(f))
    cols, off = {}, 0
    for i in range(0, len(f), 3):
        n = int(f[i + 2])
        cols[f[i].lower()] = (off, n)
        off += n
    return cols


def read_frames(path):
    """Yield (index, count_line, comment_line, atom_lines). Fatal on a broken file."""
    with open(path) as fh:
        idx = 0
        while True:
            head = fh.readline()
            if not head:
                return
            if not head.strip():
                continue
            try:
                n = int(head.split()[0])
            except ValueError:
                sys.exit("ERROR: %s frame %d: atom-count line is %r" % (path, idx, head))
            comment = fh.readline()
            atoms = [fh.readline() for _ in range(n)]
            if not comment or (n and not atoms[-1]):
                sys.exit("ERROR: %s frame %d: file ends inside the frame" % (path, idx))
            yield idx, head, comment, atoms
            idx += 1


def nep_types(nep_in):
    with open(nep_in) as fh:
        for line in fh:
            tok = line.split("#", 1)[0].split()
            if tok and tok[0] == "type":
                k = int(tok[1])
                return tok[2:2 + k]
    sys.exit("ERROR: no `type` line in %s" % nep_in)


def opt_float(s):
    return None if str(s).lower() in ("none", "off", "") else float(s)


# --------------------------------------------------------------------------
# one frame
# --------------------------------------------------------------------------
def inspect(comment, atoms):
    """Return a dict of what the filters need; 'bad' holds a malformed reason."""
    info = {"n": len(atoms), "bucket": "unknown", "folder": "", "e": None,
            "fmax": None, "species": set(), "bad": None, "tags": {}}
    tags = parse_tags(comment)
    info["tags"] = tags
    for k in BUCKET_KEYS:
        if tags.get(k):
            info["bucket"] = tags[k]
            break
    for k in FOLDER_KEYS:
        if tags.get(k):
            info["folder"] = tags[k]
            break

    if len(atoms) == 0:
        info["bad"] = "no atoms"
        return info
    if "energy" not in tags:
        info["bad"] = "no energy="
        return info
    if "lattice" not in tags or len(tags["lattice"].split()) != 9:
        info["bad"] = "no 9-number Lattice="
        return info
    if "properties" not in tags:
        info["bad"] = "no Properties="
        return info
    try:
        cols = parse_properties(tags["properties"])
        e_tot = float(tags["energy"])
    except ValueError as exc:
        info["bad"] = "unreadable header (%s)" % exc
        return info
    if "species" not in cols:
        info["bad"] = "no species column"
        return info
    fkey = "force" if "force" in cols else ("forces" if "forces" in cols else None)
    if fkey is None:
        info["bad"] = "no force column"
        return info

    s0 = cols["species"][0]
    f0, nf = cols[fkey]
    if nf != 3:
        info["bad"] = "force column is not 3-wide"
        return info
    fmax2 = 0.0
    finite = math.isfinite(e_tot)
    try:
        for line in atoms:
            w = line.split()
            info["species"].add(w[s0])
            fx, fy, fz = float(w[f0]), float(w[f0 + 1]), float(w[f0 + 2])
            if not (math.isfinite(fx) and math.isfinite(fy) and math.isfinite(fz)):
                finite = False
                continue
            f2 = fx * fx + fy * fy + fz * fz
            if f2 > fmax2:
                fmax2 = f2
    except (IndexError, ValueError):
        info["bad"] = "unreadable atom line"
        return info
    info["e"] = e_tot / len(atoms)
    info["fmax"] = math.sqrt(fmax2)
    info["finite"] = finite
    return info


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", nargs=3, action="append", required=True,
                   metavar=("NAME", "TRAIN_XYZ", "TEST_XYZ"))
    p.add_argument("--nep-in", required=True,
                   help="nep.in whose `type` line defines the allowed species")
    p.add_argument("--outdir", required=True)
    p.add_argument("--emin", type=float, default=-8.5,
                   help="reject E/atom below this, eV/atom (default -8.5)")
    p.add_argument("--emax-condensed", type=opt_float, default=0.0,
                   help="reject condensed frames with E/atom above this "
                        "(default 0.0; 'none' disables)")
    p.add_argument("--emax-gas", type=opt_float, default=None,
                   help="reject gas-phase frames with E/atom above this "
                        "(default none: dimer scans kept)")
    p.add_argument("--fmax", type=opt_float, default=500.0,
                   help="reject max |F| above this, eV/A (default 500; 'none' disables)")
    p.add_argument("--gas-pattern", default=r"dimer|trimer|cluster|isolated",
                   help="regex on the bucket name marking gas-phase frames")
    p.add_argument("--drop", action="append", default=[],
                   metavar="SOURCE:BUCKET[:REASON]",
                   help="drop a whole bucket of one source, e.g. "
                        "r1.7:slab_AlN:too_low_E_per_atom")
    a = p.parse_args()

    names = [s[0] for s in a.source]
    if len(set(names)) != len(names):
        sys.exit("ERROR: --source names must be unique")
    for name, tr, te in a.source:
        for f in (tr, te):
            if not os.path.isfile(f):
                sys.exit("ERROR: %s: %s not found" % (name, f))
    drops = {}                             # (source, bucket lower) -> (bucket, reason)
    for d in a.drop:
        f = d.split(":")
        if len(f) not in (2, 3) or not all(f):
            sys.exit("ERROR: --drop wants SOURCE:BUCKET[:REASON], got %r" % d)
        if f[0] not in names:
            sys.exit("ERROR: --drop %s: no --source named %r" % (d, f[0]))
        drops[(f[0], f[1].lower())] = (f[1], f[2] if len(f) == 3 else "")
    gas_re = re.compile(a.gas_pattern, re.I)
    types = nep_types(a.nep_in)
    allowed = set(types)

    os.makedirs(a.outdir, exist_ok=True)
    out_path = {sp: os.path.join(a.outdir, sp + ".xyz") for sp in ("train", "test")}
    tmp_path = {sp: out_path[sp] + ".tmp" for sp in out_path}
    out = {sp: open(tmp_path[sp], "w") for sp in tmp_path}

    # stats
    n_in = collections.Counter()           # (source, split)
    n_keep = collections.Counter()         # (source, split)
    b_in = collections.Counter()           # (source, bucket)
    b_keep = collections.Counter()         # (source, bucket, split)
    b_rej = collections.defaultdict(collections.Counter)  # (source, bucket) -> reason
    b_e = collections.defaultdict(list)    # (source, bucket) -> kept E/atom
    b_gas = {}                             # (source, bucket) -> gas?
    f_hi = collections.Counter()           # (source, threshold) kept frames above
    e_hi_gas = collections.Counter()       # source -> kept gas frames > 20 eV/atom
    species = {"train": collections.Counter(), "test": collections.Counter()}
    drop_hits = collections.Counter()
    rejected = []

    for name, tr, te in a.source:
        for split, path in (("train", tr), ("test", te)):
            for idx, head, comment, atoms in read_frames(path):
                info = inspect(comment, atoms)
                bucket = info["bucket"]
                key = (name, bucket)
                is_gas = bool(gas_re.search(bucket)) or info["n"] <= 3
                b_gas.setdefault(key, is_gas)
                n_in[(name, split)] += 1
                b_in[key] += 1
                e, fm = info["e"], info["fmax"]

                reason, detail = None, ""
                if info["bad"]:
                    reason, detail = "malformed", info["bad"]
                elif not info["finite"]:
                    reason = "nonfinite"
                elif not info["species"] <= allowed:
                    reason = "species"
                    detail = " ".join(sorted(info["species"] - allowed))
                elif (name, bucket.lower()) in drops:
                    reason = "dropped"
                    detail = drops[(name, bucket.lower())][1]
                    drop_hits[(name, bucket.lower())] += 1
                elif e < a.emin:
                    reason = "e_below_floor"
                elif not is_gas and a.emax_condensed is not None and e > a.emax_condensed:
                    reason = "e_above_ceil"
                elif is_gas and a.emax_gas is not None and e > a.emax_gas:
                    reason, detail = "e_above_ceil", "gas"
                elif a.fmax is not None and fm > a.fmax:
                    reason = "f_above_cap"

                if reason:
                    b_rej[key][reason] += 1
                    rejected.append({
                        "source": name, "split": split, "frame": idx,
                        "bucket": bucket, "phase": "gas" if is_gas else "condensed",
                        "natoms": info["n"],
                        "e_per_atom": "" if e is None else "%.6f" % e,
                        "fmax": "" if fm is None else "%.3f" % fm,
                        "reason": reason, "detail": detail,
                        "folder": info["folder"]})
                    continue

                if SOURCE_TAG in info["tags"]:
                    line2 = comment
                else:
                    line2 = comment.rstrip() + " %s=%s\n" % (SOURCE_TAG, name)
                out[split].write(head)
                out[split].write(line2)
                out[split].writelines(atoms)

                n_keep[(name, split)] += 1
                b_keep[(name, bucket, split)] += 1
                b_e[key].append(e)
                for thr in (20, 50, 100):
                    if fm > thr:
                        f_hi[(name, thr)] += 1
                if is_gas and e > 20:
                    e_hi_gas[name] += 1
                for sp in info["species"]:
                    species[split][sp] += 1

    for f in out.values():
        f.close()

    # ------------------------------------------------------------------ report
    L = []
    tot_tr = sum(n_keep[(s, "train")] for s in names)
    tot_te = sum(n_keep[(s, "test")] for s in names)
    L.append("round-1wL dataset: %d train / %d test (%d total, %d rejected)"
             % (tot_tr, tot_te, tot_tr + tot_te, len(rejected)))
    L.append("test fraction %.3f" % (tot_te / max(1, tot_tr + tot_te)))
    L.append("filters: emin=%g  emax_condensed=%s  emax_gas=%s  fmax=%s  gas_pattern=%s"
             % (a.emin, "none" if a.emax_condensed is None else "%g" % a.emax_condensed,
                "none" if a.emax_gas is None else "%g" % a.emax_gas,
                "none" if a.fmax is None else "%g" % a.fmax, a.gas_pattern))
    L.append("drop   : %s" % ("  ".join(
        "%s:%s%s" % (k[0], v[0], " (%s)" % v[1] if v[1] else "")
        for k, v in sorted(drops.items())) or "none"))
    L.append("types  : %s  (from %s)" % (" ".join(types), a.nep_in))
    L.append("")
    L.append("sources")
    for name, tr, te in a.source:
        L.append("  %-10s %6d / %-5d in  ->  %6d / %-5d kept   (train / test)"
                 % (name, n_in[(name, "train")], n_in[(name, "test")],
                    n_keep[(name, "train")], n_keep[(name, "test")]))
        L.append("  %-10s %s" % ("", tr))
        L.append("  %-10s %s" % ("", te))
    L.append("")

    hdr = "%-9s %-18s %-5s %6s %6s %6s %5s %9s %9s %9s  %s" % (
        "source", "bucket", "phase", "in", "train", "test", "rej",
        "eV/at min", "median", "max", "rejected by")
    L.append(hdr)
    L.append("-" * len(hdr))
    notes = []
    for key in sorted(b_in, key=lambda k: (names.index(k[0]), k[1])):
        name, bucket = key
        es = b_e.get(key, [])
        rej = b_rej.get(key, collections.Counter())
        rej_txt = " ".join("%s=%d" % (r, rej[r]) for r in REASONS if rej[r])
        if es:
            stats = "%9.3f %9.3f %9.3f" % (min(es), statistics.median(es), max(es))
        else:
            stats = "%9s %9s %9s" % ("-", "-", "-")
        L.append("%-9s %-18s %-5s %6d %6d %6d %5d %s  %s" % (
            name, bucket[:18], "gas" if b_gas[key] else "cond", b_in[key],
            b_keep[(name, bucket, "train")], b_keep[(name, bucket, "test")],
            sum(rej.values()), stats, rej_txt))
        if es and not b_gas[key] and statistics.median(es) > -3.0:
            notes.append(
                "%s %s: median %.2f eV/atom is above elemental Al (about -3.7); "
                "a condensed Al-N-Si cell near equilibrium should sit well below "
                "that. Check a few of these folders (POSCAR vs POTCAR order, SCF "
                "convergence) before trusting them; --drop %s:%s removes them."
                % (name, bucket, statistics.median(es), name, bucket))
    L.append("")

    L.append("kept frames with max |F| above 20 / 50 / 100 eV/A")
    for name in names:
        L.append("  %-10s %6d / %d / %d" % (name, f_hi[(name, 20)], f_hi[(name, 50)],
                                           f_hi[(name, 100)]))
    L.append("kept gas-phase frames above 20 eV/atom (dimer scans; --emax-gas caps them)")
    for name in names:
        L.append("  %-10s %6d" % (name, e_hi_gas[name]))
    L.append("")

    by_reason = collections.Counter(
        (r["source"], r["reason"],
         r["bucket"] if r["reason"] == "dropped" else "", r["detail"])
        for r in rejected)
    L.append("rejection reasons")
    for (name, reason, bucket, detail), n in sorted(by_reason.items()):
        L.append("  %6d  %-10s %s%s%s" % (n, name, reason,
                                         " " + bucket if bucket else "",
                                         "  (%s)" % detail if detail else ""))
    worst = sorted((r for r in rejected if r["reason"] in ("e_below_floor", "e_above_ceil")),
                   key=lambda r: -abs(float(r["e_per_atom"])))[:15]
    if worst:
        L.append("")
        L.append("most extreme rejected energies")
        for r in worst:
            L.append("  %12s eV/atom  %-9s %-5s %-16s %-13s %s" % (
                r["e_per_atom"], r["source"], r["split"], r["bucket"][:16],
                r["reason"], r["folder"] or "frame %d" % r["frame"]))
    for d in sorted(drops):
        if not drop_hits[d]:
            notes.append("--drop %s:%s matched no frame (check the spelling)"
                         % (d[0], drops[d][0]))
    L.append("")
    L.append("species (frames containing it)   train: %s   test: %s" % (
        " ".join("%s=%d" % (s, species["train"][s]) for s in types),
        " ".join("%s=%d" % (s, species["test"][s]) for s in types)))
    if notes:
        L.append("")
        L.append("NOTES")
        for n in notes:
            L.append("  - " + n)

    # ------------------------------------------------------------------ checks
    fatal = []
    if tot_tr == 0 or tot_te == 0:
        fatal.append("empty train or test set")
    missing = [s for s in types if species["train"][s] == 0]
    if missing:
        fatal.append("type(s) %s on the nep.in type line never occur in train.xyz; "
                     "NEP refuses that" % " ".join(missing))

    report = "\n".join(L) + "\n"
    with open(os.path.join(a.outdir, "report.txt"), "w") as fh:
        fh.write(report)
    with open(os.path.join(a.outdir, "rejected.csv"), "w", newline="") as fh:
        cols = ["source", "split", "frame", "bucket", "phase", "natoms",
                "e_per_atom", "fmax", "reason", "detail", "folder"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rejected)
    sys.stdout.write(report)

    if fatal:
        for f in fatal:
            print("ERROR: " + f)
        for f in tmp_path.values():
            os.remove(f)
        sys.exit(1)
    for sp in tmp_path:
        os.replace(tmp_path[sp], out_path[sp])
    print("\nwrote %s and %s" % (out_path["train"], out_path["test"]))


if __name__ == "__main__":
    main()
