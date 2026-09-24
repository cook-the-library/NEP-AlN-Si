#!/usr/bin/env python3
"""Stage 8: Submit the deposition simulations built by
generate_deposition_simulation.py to Anvil or ACES, one LAMMPS job per
condition in the sweep grid.

Default mode ("grid") submits every condition. If
config/criteria.yaml's deposition_sweep.optimization_method is "bayesian" and
scikit-optimize is installed, this instead submits an initial batch, and
subsequent invocations of this script (re-run after find_optimal_conditions.py
inspects completed results) suggest and submit the next condition(s) to
evaluate -- true closed-loop Bayesian optimization needs this script called
repeatedly by agentic_orchestrator.py rather than once. If scikit-optimize
isn't installed, this stage falls back to grid mode with a warning rather than
failing, since grid is always a valid (if less sample-efficient) choice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ConfigError,
    DEPOSITION_DIR,
    cluster_config,
    eprint,
    job_state,
    load_criteria_config,
    read_json,
    render_template,
    require_filled,
    sbatch_context,
    submit_job,
    write_json,
)


def submit_condition(cond: dict, cluster_cfg: dict) -> str | None:
    cond_dir = Path(cond["dir"])
    job_id_file = cond_dir / "job_id.json"
    if job_id_file.exists():
        info = read_json(job_id_file)
        state = job_state(info["job_id"])
        if state in ("PENDING", "RUNNING", "COMPLETED"):
            return info["job_id"]  # already submitted, don't resubmit

    if not (cond_dir / "structure.data").exists():
        eprint(f"[submit_deposition_optimization] {cond['condition_id']}: no structure.data "
               f"present (see generate_deposition_simulation.py note about missing substrate) "
               f"-- skipping.")
        return None

    ctx = sbatch_context(cluster_cfg, job_name=f"dep_{cond['condition_id']}", workdir=cond_dir, kind="lammps")
    sbatch_text = render_template("lammps.sbatch.template", ctx)
    sbatch_path = cond_dir / "submit.sbatch"
    sbatch_path.write_text(sbatch_text)

    job_id = submit_job(sbatch_path)
    write_json(job_id_file, {"job_id": job_id})
    return job_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", choices=["anvil", "aces"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cluster_cfg = cluster_config(args.cluster)
    require_filled(cluster_cfg, context=f"submit_deposition_optimization.py --cluster {args.cluster}")

    criteria = load_criteria_config()
    method = criteria["deposition_sweep"].get("optimization_method", "grid")

    manifest_path = DEPOSITION_DIR / "sim_inputs" / "manifest.json"
    if not manifest_path.exists():
        raise ConfigError(f"{manifest_path} not found -- run generate_deposition_simulation.py first.")
    manifest = read_json(manifest_path)

    if method == "bayesian":
        try:
            import skopt  # noqa: F401
        except ImportError:
            eprint("[submit_deposition_optimization] optimization_method='bayesian' but "
                   "scikit-optimize isn't installed (`pip install scikit-optimize`) -- "
                   "falling back to submitting the full grid instead.")
            method = "grid"
        else:
            eprint("[submit_deposition_optimization] NOTE: bayesian mode submits the full "
                   "grid on this pass too -- the suggest-next-point loop over "
                   "find_optimal_conditions.py's partial results is not wired up yet in "
                   "this skeleton. Extend this branch to call skopt.Optimizer.tell()/ask() "
                   "using completed conditions' parsed results before treating this as "
                   "truly sample-efficient.")

    submitted = {}
    for cond in manifest["conditions"]:
        if args.dry_run:
            eprint(f"[submit_deposition_optimization] (dry-run) would submit {cond['condition_id']}")
            continue
        job_id = submit_condition(cond, cluster_cfg)
        if job_id:
            submitted[cond["condition_id"]] = job_id

    write_json(DEPOSITION_DIR / "sim_inputs" / "job_ids.json", {"cluster": args.cluster, "jobs": submitted})
    print(f"[submit_deposition_optimization] Submitted/confirmed {len(submitted)}/{len(manifest['conditions'])} "
          f"deposition jobs on {args.cluster} (method={method})")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[submit_deposition_optimization] CONFIG ERROR: {e}")
        sys.exit(2)
