#!/usr/bin/env python3
"""Stage 9: Aggregate completed deposition runs and rank conditions.

For every completed condition (job_id.json state == COMPLETED), parses:
  - final potential energy per atom and final density from the LAMMPS run
    (structural/energetic proxies for deposition quality -- always computed)
  - the configured target_property (config/criteria.yaml's
    deposition_sweep.target_property, "tbc" or "kappa"), IF that property was
    also computed for this exact final structure. As shipped, this skill's
    stage 8 does not automatically run a follow-up NEMD/HNEMD calculation on
    each deposited film (that's a second production MD run per condition, on
    top of the deposition run itself) -- so by default this stage ranks the
    top candidates by the structural/energetic proxies and reports which
    conditions should be fed through a TBC/kappa follow-up (reusing
    evaluate_potential.py's maybe_run_kappa_or_tbc logic against
    final_state.data) to get a true target_property-based ranking, rather
    than silently substituting the proxy for the real target metric.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ConfigError, DEPOSITION_DIR, eprint, job_state, load_criteria_config, read_json, write_json


def parse_final_energy(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text()
    # LAMMPS thermo table: find the last numeric line under a "Step ... PotEng ..." header.
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Step") and "PotEng" in line:
            header_idx = i
    if header_idx is None:
        return None
    header = lines[header_idx].split()
    try:
        pe_col = header.index("PotEng")
    except ValueError:
        return None
    last_val = None
    for line in lines[header_idx + 1:]:
        parts = line.split()
        if len(parts) != len(header) or not re.match(r"^-?\d", parts[0] if parts else ""):
            break
        try:
            last_val = float(parts[pe_col])
        except (ValueError, IndexError):
            break
    return last_val


def parse_atom_count(data_path: Path) -> int | None:
    if not data_path.exists():
        return None
    for line in data_path.read_text().splitlines()[:20]:
        if "atoms" in line:
            try:
                return int(line.split()[0])
            except (ValueError, IndexError):
                continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    criteria = load_criteria_config()
    target_property = criteria["deposition_sweep"].get("target_property", "tbc")

    manifest_path = DEPOSITION_DIR / "sim_inputs" / "manifest.json"
    if not manifest_path.exists():
        raise ConfigError(f"{manifest_path} not found -- run generate_deposition_simulation.py "
                           f"and submit_deposition_optimization.py first.")
    manifest = read_json(manifest_path)

    results = []
    for cond in manifest["conditions"]:
        cond_dir = Path(cond["dir"])
        job_id_file = cond_dir / "job_id.json"
        if not job_id_file.exists():
            continue
        info = read_json(job_id_file)
        state = job_state(info["job_id"])
        if state != "COMPLETED":
            results.append({**cond, "job_state": state, "status": "not_complete"})
            continue

        final_pe = parse_final_energy(cond_dir / "lammps.log")
        n_atoms = parse_atom_count(cond_dir / "final_state.data")
        pe_per_atom = final_pe / n_atoms if (final_pe is not None and n_atoms) else None

        results.append({
            **cond,
            "job_state": state,
            "status": "complete",
            "final_potential_energy_eV": final_pe,
            "n_atoms": n_atoms,
            "potential_energy_per_atom_eV": pe_per_atom,
            f"{target_property}_status": "requires_followup_md",
        })

    completed = [r for r in results if r["status"] == "complete"]
    ranked = sorted(
        [r for r in completed if r.get("potential_energy_per_atom_eV") is not None],
        key=lambda r: r["potential_energy_per_atom_eV"],
    )

    output = {
        "target_property": target_property,
        "n_conditions_total": len(manifest["conditions"]),
        "n_completed": len(completed),
        "ranked_by_proxy_energy_per_atom": ranked,
        "all_results": results,
        "note": (
            f"Ranking above is by potential-energy-per-atom as a structural proxy, "
            f"NOT by '{target_property}' directly. To get a true '{target_property}' "
            f"ranking, run a kappa/TBC follow-up MD calculation (see "
            f"evaluate_potential.py's maybe_run_kappa_or_tbc, pointed at each "
            f"top-ranked condition's final_state.data instead of the training-stage "
            f"reference structure) on the top few candidates below, then re-rank by "
            f"that value."
        ),
    }
    write_json(DEPOSITION_DIR / "optimal_conditions.json", output)

    if ranked:
        best = ranked[0]
        print(f"[find_optimal_conditions] Best proxy candidate: {best['condition_id']} "
              f"(PE/atom={best['potential_energy_per_atom_eV']:.4f} eV). "
              f"{len(completed)}/{len(manifest['conditions'])} conditions complete. "
              f"See deposition/optimal_conditions.json for the full ranking and the "
              f"'{target_property}' follow-up note.")
    else:
        print(f"[find_optimal_conditions] No completed conditions with parseable energy yet "
              f"({len(completed)}/{len(manifest['conditions'])} complete).")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[find_optimal_conditions] CONFIG ERROR: {e}")
        sys.exit(2)
