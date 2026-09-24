#!/usr/bin/env python3
"""Stage 6: Identify the deposition setup correlating with the real experiment.

Finds the most recent round whose decide_next_step.py outcome was "sufficient",
pulls its trained NEP potential (nep.txt), and combines it with
config/experiment_correlations.yaml (the real deposition setup this should be
validated against, if known) and config/criteria.yaml's deposition_sweep
parameters to produce deposition/setup.json -- the single source of truth the
remaining stages read from.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ConfigError,
    DEPOSITION_DIR,
    RUNS_DIR,
    eprint,
    load_criteria_config,
    load_experiment_config,
    read_json,
    write_json,
)


def find_sufficient_round(explicit_round: int | None) -> int:
    if explicit_round is not None:
        decision_path = RUNS_DIR / f"round_{explicit_round:03d}" / "decision.json"
        if not decision_path.exists():
            raise ConfigError(f"{decision_path} not found.")
        decision = read_json(decision_path)
        if decision["outcome"] != "sufficient":
            raise ConfigError(
                f"Round {explicit_round}'s decision outcome was "
                f"'{decision['outcome']}', not 'sufficient'. Pass a round that "
                f"passed evaluation, or fix why this one didn't."
            )
        return explicit_round

    candidates = []
    if RUNS_DIR.exists():
        for round_dir in sorted(RUNS_DIR.glob("round_*")):
            decision_path = round_dir / "decision.json"
            if decision_path.exists():
                decision = read_json(decision_path)
                if decision.get("outcome") == "sufficient":
                    candidates.append(decision["round"])
    if not candidates:
        raise ConfigError(
            "No round has a 'sufficient' decision yet. Run the training loop "
            "(stages 1-5) until decide_next_step.py reports sufficient, or "
            "pass --round explicitly if you're intentionally proceeding with a "
            "potential you've accepted by other means."
        )
    return max(candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, default=None,
                         help="Which training round's NEP to use. Default: the latest round marked 'sufficient'.")
    args = parser.parse_args()

    chosen_round = find_sufficient_round(args.round)
    nep_txt = RUNS_DIR / f"round_{chosen_round:03d}" / "nep_model" / "nep.txt"
    if not nep_txt.exists():
        raise ConfigError(f"{nep_txt} not found even though round {chosen_round} was marked sufficient.")

    experiment = load_experiment_config()["experiment"]
    criteria = load_criteria_config()

    unfilled = [k for k in ("deposition_technique", "substrate_material", "deposit_material")
                if experiment.get(k) in (None, "FILL_ME_IN")]
    if unfilled:
        eprint(
            f"[identify_deposition_setup] NOTE: config/experiment_correlations.yaml "
            f"is missing {unfilled} -- proceeding, but the deposition simulation "
            f"setup below can't be checked against a real experimental "
            f"configuration for these fields. Fill them in if you want stage 9's "
            f"optimum to be validated against actual experimental correlation "
            f"rather than simulated criteria alone."
        )

    setup = {
        "source_round": chosen_round,
        "nep_potential_path": str(nep_txt),
        "experiment": experiment,
        "sweep": criteria["deposition_sweep"],
    }

    DEPOSITION_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DEPOSITION_DIR / "setup.json", setup)
    print(f"[identify_deposition_setup] Using NEP from round {chosen_round}. "
          f"Wrote {DEPOSITION_DIR / 'setup.json'}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        eprint(f"[identify_deposition_setup] CONFIG ERROR: {e}")
        sys.exit(2)
