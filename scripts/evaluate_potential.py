#!/usr/bin/env python3
"""Stage 4: Evaluate the trained NEP against the criteria in config/criteria.yaml
-- energy/force/virial RMSE vs the AIMD (VASP) test set, plus kappa and TBC if
enabled and reference structures are configured.

Energy/force/virial RMSE come directly from GPUMD's own test-set output files
(energy_test.out, force_test.out, virial_test.out), which NEP writes after
training completes -- no separate calculation needed, just parsing.

kappa (thermal conductivity, via HNEMD or Green-Kubo) and TBC (thermal boundary
conductance, via NEMD) require their own MD production runs with the trained
potential on a specific reference structure (a bulk cell for kappa, an
interface for TBC). This skill does not know your interface geometry, so:
  - If config/criteria.yaml's kappa/tbc sections are enabled but no reference
    structure is configured, this stage reports them as "not evaluated" (which
    decide_next_step.py treats as blocking, not as a silent pass -- see that
    script) rather than guessing a structure.
  - If a reference structure IS configured (kappa_reference_structure /
    tbc_reference_structure under evaluation: in criteria.yaml), this stage
    renders a standard HNEMD/NEMD run.in, submits it, and (on a later
    invocation, once the job is done) parses the result. The run.in templates
    follow standard GPUMD recipes (https://gpumd.org) but WILL need review for
    your specific system size / run length / temperature.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ConfigError,
    cluster_config,
    eprint,
    job_state,
    load_criteria_config,
    read_json,
    render_template,
    require_filled,
    round_dir,
    sbatch_context,
    submit_job,
    write_json,
)

import math


def _rmse(pred, target):
    n = len(pred)
    if n == 0:
        return None
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, target)) / n)


def parse_gpumd_out_columns(path: Path):
    """GPUMD NEP *_test.out files are whitespace-columns of [predicted..., target...]
    (equal split). Returns (predicted_flat, target_flat) as flat lists of floats,
    one entry per scalar component (so force files contribute 3 entries/atom/frame)."""
    if not path.exists():
        return None, None
    predicted, target = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        vals = [float(x) for x in parts]
        half = len(vals) // 2
        predicted.extend(vals[:half])
        target.extend(vals[half:])
    return predicted, target


def compute_accuracy_metrics(model_dir: Path) -> dict:
    metrics = {}
    for kind, fname in [("energy", "energy_test.out"), ("force", "force_test.out"), ("virial", "virial_test.out")]:
        pred, tgt = parse_gpumd_out_columns(model_dir / fname)
        rmse = _rmse(pred, tgt) if pred else None
        metrics[f"{kind}_rmse"] = rmse
        metrics[f"{kind}_n_points"] = len(pred) if pred else 0
    return metrics


def maybe_run_kappa_or_tbc(kind: str, cfg: dict, cluster_cfg: dict, cluster: str, model_dir: Path,
                            r_dir: Path, dry_run: bool) -> dict:
    """kind: 'kappa' or 'tbc'. Returns a result dict with at least 'status'."""
    section = cfg["evaluation"][kind]
    if not section.get("enabled"):
        return {"status": "disabled"}

    ref_key = f"{kind}_reference_structure"
    ref_path_str = section.get(ref_key) or cfg["evaluation"].get(ref_key)
    if not ref_path_str:
        return {
            "status": "not_evaluated",
            "reason": (
                f"criteria.yaml evaluation.{kind}.enabled is true but no "
                f"'{ref_key}' path is configured. Add one (an extxyz/POSCAR "
                f"reference structure for the {kind} calculation) to "
                f"config/criteria.yaml before this can run."
            ),
        }

    ref_path = Path(ref_path_str)
    if not ref_path.exists():
        return {"status": "not_evaluated", "reason": f"Reference structure {ref_path} does not exist."}

    run_dir = r_dir / f"{kind}_eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    job_id_file = run_dir / "job_id.json"
    if job_id_file.exists():
        # Already submitted -- check status instead of resubmitting.
        info = read_json(job_id_file)
        state = job_state(info["job_id"])
        if state != "COMPLETED":
            return {"status": "pending", "job_id": info["job_id"], "job_state": state}
        return parse_kappa_or_tbc_result(kind, run_dir)

    # Not yet submitted: stage inputs and submit.
    (run_dir / "model.xyz").write_text(ref_path.read_text())
    run_in = build_run_in(kind, section)
    (run_dir / "run.in").write_text(run_in)
    # nep.txt from training
    nep_txt = model_dir / "nep.txt"
    if not nep_txt.exists():
        return {"status": "blocked", "reason": f"{nep_txt} not found -- has NEP training finished?"}
    (run_dir / "nep.txt").write_text(nep_txt.read_text())

    ctx = sbatch_context(cluster_cfg, job_name=f"{kind}_eval", workdir=run_dir, kind="deposition")
    sbatch_text = render_template("deposition.sbatch.template", ctx)
    sbatch_path = run_dir / "submit.sbatch"
    sbatch_path.write_text(sbatch_text)

    if dry_run:
        return {"status": "dry_run", "sbatch": str(sbatch_path)}

    job_id = submit_job(sbatch_path)
    write_json(job_id_file, {"job_id": job_id})
    return {"status": "pending", "job_id": job_id, "job_state": "SUBMITTED"}


def build_run_in(kind: str, section: dict) -> str:
    """Standard GPUMD run.in recipes. These are reasonable starting points
    (see https://gpumd.org/tutorials) but review timestep/run-length/thermostat
    choices against your actual system before trusting the result."""
    if kind == "kappa":
        method = section.get("method", "hnemd")
        if method == "hnemd":
            return (
                "potential nep.txt\n"
                "velocity 300\n"
                "ensemble npt_scr 300 300 200 0 200 2000\n"
                "time_step 1\n"
                "run 100000\n"
                "\n"
                "ensemble nvt_bdp 300 300 200\n"
                "compute_hnemd 100000 0.000025 0 0\n"
                "compute_shc 5 250\n"
                "run 1000000\n"
            )
        return (  # green_kubo
            "potential nep.txt\n"
            "velocity 300\n"
            "ensemble npt_scr 300 300 200 0 200 2000\n"
            "time_step 1\n"
            "run 100000\n"
            "\n"
            "ensemble nve\n"
            "compute_hac 5 250 1000\n"
            "run 2000000\n"
        )
    # tbc via NEMD
    return (
        "potential nep.txt\n"
        "velocity 300\n"
        "ensemble npt_scr 300 300 200 0 200 2000\n"
        "time_step 1\n"
        "run 100000\n"
        "\n"
        "ensemble heat_nhc 300 200 20 1 2\n"
        "compute_temperature 5 10 group 0\n"
        "run 2000000\n"
    )


def parse_kappa_or_tbc_result(kind: str, run_dir: Path) -> dict:
    """Parses GPUMD's kappa.out (HNEMD/HAC) or a temperature-gradient output for
    NEMD-based TBC. Output file formats depend on the exact run.in used above --
    adjust this parser if you change the method. Returns status='not_evaluated'
    with a clear reason rather than a fabricated number if parsing fails."""
    candidates = ["kappa.out", "shc.out", "thermo.out"]
    for fname in candidates:
        fpath = run_dir / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            return {
                "status": "needs_manual_review",
                "reason": (
                    f"{fpath} exists but this skeleton doesn't parse it into a "
                    f"single {kind} number yet -- GPUMD's raw kappa/SHC/thermal "
                    f"output format depends on your run.in choices above. "
                    f"Inspect {fpath} and extend parse_kappa_or_tbc_result() to "
                    f"extract the steady-state value for your setup."
                ),
                "raw_output_file": str(fpath),
            }
    return {"status": "not_evaluated", "reason": f"No recognized output file found in {run_dir} yet."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--cluster", choices=["anvil", "aces"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    criteria = load_criteria_config()
    cluster_cfg = cluster_config(args.cluster)
    require_filled(cluster_cfg, context=f"evaluate_potential.py --cluster {args.cluster}")

    r_dir = round_dir(args.round)
    model_dir = r_dir / "nep_model"

    job_info_path = model_dir / "job_id.json"
    if job_info_path.exists():
        info = read_json(job_info_path)
        state = job_state(info["job_id"])
        if state != "COMPLETED":
            eprint(f"[evaluate_potential] NEP training job {info['job_id']} is {state}, not COMPLETED yet. "
                   f"Nothing to evaluate.")
            write_json(r_dir / "evaluation.json", {"round": args.round, "status": "training_not_complete", "job_state": state})
            sys.exit(1)

    accuracy = compute_accuracy_metrics(model_dir)
    kappa_result = maybe_run_kappa_or_tbc("kappa", criteria, cluster_cfg, args.cluster, model_dir, r_dir, args.dry_run)
    tbc_result = maybe_run_kappa_or_tbc("tbc", criteria, cluster_cfg, args.cluster, model_dir, r_dir, args.dry_run)

    evaluation = {
        "round": args.round,
        "cluster": args.cluster,
        "accuracy": accuracy,
        "kappa": kappa_result,
        "tbc": tbc_result,
        # decide_next_step.py reads this back; per_structure_errors is left [] here
        # since GPUMD's *_test.out files don't carry structure ids by default --
        # wire this up (e.g. via NEP's per-structure dump options) if you want
        # generate_structures.py's error-biased resampling to be non-stubbed.
        "per_structure_errors": [],
    }
    write_json(r_dir / "evaluation.json", evaluation)

    print(f"[evaluate_potential] Round {args.round}: energy_rmse={accuracy.get('energy_rmse')}, "
          f"force_rmse={accuracy.get('force_rmse')}, kappa={kappa_result['status']}, tbc={tbc_result['status']}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[evaluate_potential] CONFIG ERROR: {e}")
        sys.exit(2)
