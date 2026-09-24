#!/usr/bin/env python3
"""Top-level driver for the full agentic loop (stages 1-9). Runs the diagram
end to end: generate structures -> submit VASP -> train NEP -> evaluate ->
decide (loop back or proceed) -> identify deposition setup -> generate
deposition sims -> optimize deposition -> find optimal conditions.

This calls the individual stage scripts as subprocesses (rather than
importing their internals) so each stage stays independently runnable and
inspectable -- recommended for the first run of this pipeline on a new
material system, since VASP/NEP/LAMMPS settings are project-specific and you
will want to check intermediate output before trusting the next stage.

Stops (does not loop forever, does not silently paper over problems) when:
  - decide_next_step.py reports "sufficient"  -> proceeds to deposition stages
  - decide_next_step.py reports "blocked"      -> exits, prints why
  - --max-rounds is reached without "sufficient" -> exits, prints why
  - any stage's subprocess exits non-zero        -> exits, prints the failing
                                                     stage and its stderr tail
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    RUNS_DIR,
    cluster_config,
    eprint,
    job_state,
    load_clusters_config,
    read_json,
    round_dir,
)

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_stage(script: str, args: list[str]) -> int:
    cmd = [sys.executable, str(SCRIPTS_DIR / script), *args]
    print(f"\n[orchestrator] === Running: {' '.join(cmd)} ===")
    result = subprocess.run(cmd)
    return result.returncode


def poll_until(job_id_getter, poll_interval: int, max_wait_hours: float, label: str) -> str:
    """job_id_getter: callable returning a SLURM job id string, called once
    (jobs already submitted by the stage script before this is invoked)."""
    job_id = job_id_getter()
    if job_id is None:
        return "NO_JOB"
    print(f"[orchestrator] Waiting on {label} (job {job_id})...")
    deadline = time.time() + max_wait_hours * 3600
    while time.time() < deadline:
        state = job_state(job_id)
        if state in ("COMPLETED", "FAILED"):
            print(f"[orchestrator] {label} (job {job_id}): {state}")
            return state
        time.sleep(poll_interval)
    return "TIMED_OUT"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", choices=["anvil", "aces"], required=True)
    parser.add_argument("--max-rounds", type=int, default=None,
                         help="Overrides active_learning.max_rounds from criteria.yaml")
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--skip-training-loop", action="store_true",
                         help="Skip stages 1-5 and go straight to deposition stages 6-9, "
                              "using the latest round already marked sufficient.")
    args = parser.parse_args()

    all_clusters_cfg = load_clusters_config()
    common_cfg = all_clusters_cfg.get("common", {})
    poll_interval = common_cfg.get("poll_interval_seconds", 60)
    max_poll_hours = common_cfg.get("max_poll_hours", 48)

    round_num = args.start_round

    if not args.skip_training_loop:
        while True:
            r_dir = round_dir(round_num)

            rc = run_stage("generate_structures.py", ["--round", str(round_num)])
            if rc != 0:
                eprint(f"[orchestrator] generate_structures.py failed for round {round_num} (exit {rc}). Stopping.")
                sys.exit(rc)

            rc = run_stage("submit_vasp.py", ["--round", str(round_num), "--cluster", args.cluster])
            if rc != 0:
                eprint(f"[orchestrator] submit_vasp.py failed for round {round_num} (exit {rc}). Stopping.")
                sys.exit(rc)

            job_ids_path = r_dir / "vasp" / "job_ids.json"
            job_ids = read_json(job_ids_path).get("jobs", {}) if job_ids_path.exists() else {}
            print(f"[orchestrator] Waiting on {len(job_ids)} VASP job(s) for round {round_num}...")
            deadline = time.time() + max_poll_hours * 3600
            pending = set(job_ids.values())
            failed = set()
            while pending and time.time() < deadline:
                for jid in list(pending):
                    state = job_state(jid)
                    if state == "COMPLETED":
                        pending.discard(jid)
                    elif state == "FAILED":
                        pending.discard(jid)
                        failed.add(jid)
                if pending:
                    time.sleep(poll_interval)
            if pending:
                eprint(f"[orchestrator] Timed out waiting on VASP jobs {pending} for round {round_num}. Stopping.")
                sys.exit(1)
            if failed:
                eprint(f"[orchestrator] VASP job(s) {failed} FAILED for round {round_num}. "
                       f"Inspect round_{round_num:03d}/vasp/*/slurm-*.err before continuing. Stopping.")
                sys.exit(1)

            rc = run_stage("vasp_to_nep_dataset.py", ["--round", str(round_num), "--include-previous-rounds"])
            if rc != 0:
                eprint(f"[orchestrator] vasp_to_nep_dataset.py failed for round {round_num} (exit {rc}). Stopping.")
                sys.exit(rc)

            rc = run_stage("submit_nep_training.py", ["--round", str(round_num), "--cluster", args.cluster])
            if rc != 0:
                eprint(f"[orchestrator] submit_nep_training.py failed for round {round_num} (exit {rc}). Stopping.")
                sys.exit(rc)

            nep_job_path = r_dir / "nep_model" / "job_id.json"
            nep_job_id = read_json(nep_job_path)["job_id"] if nep_job_path.exists() else None
            state = poll_until(lambda: nep_job_id, poll_interval, max_poll_hours, f"NEP training round {round_num}")
            if state != "COMPLETED":
                eprint(f"[orchestrator] NEP training for round {round_num} ended in state {state}. Stopping.")
                sys.exit(1)

            rc = run_stage("evaluate_potential.py", ["--round", str(round_num), "--cluster", args.cluster])
            if rc not in (0, 1):
                eprint(f"[orchestrator] evaluate_potential.py failed unexpectedly for round {round_num} "
                       f"(exit {rc}). Stopping.")
                sys.exit(rc)

            rc = run_stage("decide_next_step.py", ["--round", str(round_num)])
            decision = read_json(r_dir / "decision.json")
            print(f"[orchestrator] Round {round_num} decision: {decision['outcome']} -- {decision['reason']}")

            if decision["outcome"] == "sufficient":
                break
            if decision["outcome"] == "blocked":
                eprint(f"[orchestrator] Blocked at round {round_num}: {decision['reason']}")
                sys.exit(1)

            max_rounds = args.max_rounds or 5
            round_num += 1
            if round_num >= max_rounds:
                eprint(f"[orchestrator] Reached max_rounds={max_rounds} without a sufficient potential. Stopping.")
                sys.exit(1)

    # Stages 6-9: deposition setup, simulation, optimization, and final ranking.
    rc = run_stage("identify_deposition_setup.py", [])
    if rc != 0:
        sys.exit(rc)

    rc = run_stage("generate_deposition_simulation.py", [])
    if rc != 0:
        sys.exit(rc)

    rc = run_stage("submit_deposition_optimization.py", ["--cluster", args.cluster])
    if rc != 0:
        sys.exit(rc)

    from _common import DEPOSITION_DIR

    job_ids_path = DEPOSITION_DIR / "sim_inputs" / "job_ids.json"
    job_ids = read_json(job_ids_path).get("jobs", {}) if job_ids_path.exists() else {}
    print(f"[orchestrator] Waiting on {len(job_ids)} deposition simulation job(s)...")
    deadline = time.time() + max_poll_hours * 3600
    pending = set(job_ids.values())
    while pending and time.time() < deadline:
        for jid in list(pending):
            if job_state(jid) in ("COMPLETED", "FAILED"):
                pending.discard(jid)
        if pending:
            time.sleep(poll_interval)
    if pending:
        eprint(f"[orchestrator] Timed out waiting on deposition jobs {pending}. "
               f"Run find_optimal_conditions.py manually once they finish.")

    rc = run_stage("find_optimal_conditions.py", [])
    sys.exit(rc)


if __name__ == "__main__":
    main()
