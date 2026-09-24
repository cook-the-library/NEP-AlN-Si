#!/usr/bin/env python3
"""Stage 5: The loop's decision node. Reads round_XXX/evaluation.json and
config/criteria.yaml's thresholds, and decides one of three outcomes:

  sufficient        -> proceed to stage 6 (identify_deposition_setup.py)
  insufficient       -> loop back to stage 1 for round+1 (generate_structures.py)
  blocked             -> stop; a human needs to do something evaluate_potential.py
                          couldn't (most commonly: finish wiring up kappa/TBC output
                          parsing for this specific run.in, or fill in a missing
                          reference structure). Looping back to generate more
                          training structures would NOT fix a blocked evaluation,
                          so this is deliberately a separate outcome from
                          "insufficient" -- agentic_orchestrator.py stops on it
                          rather than burning another round of VASP+NEP compute.

Writes round_XXX/decision.json and prints a one-line summary + exits 0 always
(the decision is the output, not a pass/fail exit code) -- callers should read
the JSON's "outcome" field.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ConfigError, eprint, load_criteria_config, read_json, round_dir, write_json


def check_accuracy(evaluation: dict, criteria: dict) -> list[str]:
    """Returns a list of human-readable failure reasons (empty if all pass)."""
    acc = evaluation.get("accuracy", {})
    eval_cfg = criteria["evaluation"]
    failures = []

    checks = [
        ("energy_rmse", "energy_rmse_meV_per_atom_max", 1000, "meV/atom"),
        ("force_rmse", "force_rmse_meV_per_A_max", 1000, "meV/A"),
        ("virial_rmse", "virial_rmse_meV_per_atom_max", 1000, "meV/atom"),
    ]
    for metric_key, threshold_key, scale, unit in checks:
        value = acc.get(metric_key)
        threshold = eval_cfg.get(threshold_key)
        if threshold is None:
            continue
        if value is None:
            failures.append(f"{metric_key} unavailable (n_points={acc.get(metric_key.replace('_rmse', '_n_points'), 0)})")
            continue
        scaled = value * scale
        if scaled > threshold:
            failures.append(f"{metric_key}={scaled:.3f} {unit} exceeds max {threshold} {unit}")
    return failures


def check_physical_criterion(name: str, result: dict, section_cfg: dict) -> tuple[str, str | None]:
    """Returns (status, reason) where status is 'pass', 'fail', 'skip', or 'blocked'."""
    if not section_cfg.get("enabled"):
        return "skip", None
    status = result.get("status")
    if status == "disabled":
        return "skip", None
    if status in ("not_evaluated", "blocked", "needs_manual_review", "pending", "dry_run"):
        return "blocked", result.get("reason") or f"{name} status is '{status}'"
    value = result.get("value")
    target = section_cfg.get("target_W_per_mK") or section_cfg.get("target_MW_per_m2K")
    if value is None:
        return "blocked", f"{name} result has no parsed 'value' field yet."
    if target is None:
        # No experimental target to compare against -- accept whatever was computed,
        # since there's nothing to fail it against. Deposition optimization stage
        # will still use the value.
        return "pass", None
    tol = section_cfg.get("relative_tolerance", 0.2)
    if abs(value - target) / abs(target) > tol:
        return "fail", f"{name}={value} outside +/-{tol * 100:.0f}% of target {target}"
    return "pass", None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()

    criteria = load_criteria_config()
    r_dir = round_dir(args.round)
    eval_path = r_dir / "evaluation.json"
    if not eval_path.exists():
        raise ConfigError(f"{eval_path} not found -- run evaluate_potential.py for round {args.round} first.")
    evaluation = read_json(eval_path)

    if evaluation.get("status") == "training_not_complete":
        decision = {"round": args.round, "outcome": "blocked",
                    "reason": "NEP training job not complete yet.", "failures": []}
        write_json(r_dir / "decision.json", decision)
        print(f"[decide_next_step] Round {args.round}: BLOCKED (training not complete)")
        return

    accuracy_failures = check_accuracy(evaluation, criteria)

    kappa_status, kappa_reason = check_physical_criterion(
        "kappa", evaluation.get("kappa", {}), criteria["evaluation"]["kappa"])
    tbc_status, tbc_reason = check_physical_criterion(
        "tbc", evaluation.get("tbc", {}), criteria["evaluation"]["tbc"])

    blocked_reasons = [r for r in (kappa_reason, tbc_reason) if r and
                        (kappa_status == "blocked" or tbc_status == "blocked")]
    physical_failures = []
    if kappa_status == "fail":
        physical_failures.append(kappa_reason)
    if tbc_status == "fail":
        physical_failures.append(tbc_reason)

    require_all = criteria["evaluation"].get("require_all_criteria", True)

    if "blocked" in (kappa_status, tbc_status):
        outcome = "blocked"
        reason = "; ".join(blocked_reasons)
    elif accuracy_failures or (physical_failures and require_all):
        outcome = "insufficient"
        reason = "; ".join(accuracy_failures + physical_failures)
    elif physical_failures and not require_all:
        # Partial pass allowed by config -- still flag it clearly rather than
        # silently treating as fully sufficient.
        outcome = "insufficient"
        reason = "; ".join(physical_failures) + " (require_all_criteria=false, but at least one physical criterion still failed)"
    else:
        outcome = "sufficient"
        reason = "All enabled criteria met."

    max_rounds = criteria["active_learning"].get("max_rounds", 5)
    if outcome == "insufficient" and args.round + 1 >= max_rounds:
        outcome = "blocked"
        reason = (f"Would loop back for round {args.round + 1}, but that reaches/exceeds "
                  f"active_learning.max_rounds={max_rounds}. Raise max_rounds in "
                  f"config/criteria.yaml if more rounds are warranted, or accept the "
                  f"current potential and proceed manually. Original reason: {reason}")

    decision = {
        "round": args.round,
        "outcome": outcome,   # "sufficient" | "insufficient" | "blocked"
        "reason": reason,
        "accuracy_failures": accuracy_failures,
        "physical_failures": physical_failures,
        "next_round": args.round + 1 if outcome == "insufficient" else None,
    }
    write_json(r_dir / "decision.json", decision)
    print(f"[decide_next_step] Round {args.round}: {outcome.upper()} -- {reason}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[decide_next_step] CONFIG ERROR: {e}")
        sys.exit(2)
