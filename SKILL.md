---
name: nep-deposition-workflow
description: Run the closed-loop "agentic AI" materials workflow for NEP-potential-driven deposition optimization — generating candidate structures, submitting VASP jobs, training a GPUMD NEP (neuroevolution potential), evaluating it against AIMD/TBC/kappa criteria, looping back to generate more training structures if the potential is insufficient, then using the trained NEP to simulate and optimize deposition conditions on an HPC cluster (Purdue Anvil or TAMU ACES). Use when the user mentions this workflow, NEP training, VASP+NEP active learning, thermal boundary conductance (TBC), thermal conductivity (kappa) evaluation, deposition simulation/optimization, or Anvil/ACES job submission for this pipeline.
---

# NEP-Driven Deposition Optimization — Agentic Workflow

> In this repository the workflow is set up for **AlN on Si(111)**: stage 1 seeds come
> from `scripts/aln_si_structures.py`, stage 7 co-deposits Al and N onto Si(111), and
> the configs carry AlN/Si values. See `docs/AlN_Si_development_plan.md`.

This skill packages the closed-loop workflow below into runnable stages. Each stage is a
script under `scripts/`. Claude's job when this skill is invoked is to act as the
orchestrator: figure out which stage the user is at, run/help run that stage, interpret
its output, and decide (per the rules below) whether to advance or loop back — the same
way the diagram does.

## The loop

```
┌─────────────────────┐   submit VASP jobs   ┌──────────────┐   train NEP   ┌────────────────────────┐
│ 1. Generate          │ ───(comp. resource)──▶│ 2. VASP AIMD │──(comp.     ─▶│ 3. Choose evaluation     │
│    structures for    │                       │    jobs      │  resource)   │    criteria: TBC, kappa, │
│    deposition        │                       └──────────────┘              │    energy vs AIMD        │
└───────────▲───────────┘                                                    └────────────┬─────────────┘
            │                                                                              │
            │            generate more structures if insufficient                          │ sufficient
            └──────────────────────────────────────────────────────────────────────────────┘potential
                                                                                             │
                                                                                             ▼
┌──────────────────┐   run NEP    ┌───────────────────┐   generate dep.  ┌─────────────────────────────┐
│ 6. Find optimal   │◀─(comp.    ─│ 5. Optimize        │◀── sim. setup ───│ 4. Identify deposition setup │
│    conditions     │  resource)  │    deposition based │                  │    correlating experiment    │
└────────────────────┘             │    on criteria      │                  └─────────────────────────────┘
                                    └────────────────────┘
```

Stages, in order, and the script that implements each:

1. **Generate structures for deposition** — `scripts/generate_structures.py`
2. **Submit VASP jobs** (on Anvil or ACES) — `scripts/submit_vasp.py`
3. **Train NEP** (on Anvil or ACES) — `scripts/vasp_to_nep_dataset.py` then `scripts/submit_nep_training.py`
4. **Choose evaluation criteria / evaluate** (TBC, kappa, energy vs AIMD) — `scripts/evaluate_potential.py`
5. **Decision: sufficient potential?** — `scripts/decide_next_step.py`
   - `insufficient` → go back to stage 1, generating a new round of structures (active-learning style, biased toward the configurations the current NEP got most wrong)
   - `sufficient` → continue to stage 6
6. **Identify deposition setup correlating experiment** — `scripts/identify_deposition_setup.py`
7. **Generate deposition simulation setup** — `scripts/generate_deposition_simulation.py`
8. **Optimize deposition based on criteria** (run NEP on Anvil/ACES) — `scripts/submit_deposition_optimization.py`
9. **Find optimal conditions** — `scripts/find_optimal_conditions.py`

`scripts/agentic_orchestrator.py` drives stages 1–9 end to end, polling SLURM between
compute-resource stages and calling the decision function at stage 5. Claude can run it
directly, or run stages one at a time when the user wants to inspect intermediate output
(recommended the first time through, since VASP/NEP settings are project-specific).

## Before running anything

This skill ships with placeholders, not your credentials or allocation details. Before
the first real run, fill in:

1. **`config/clusters.yaml`** — SLURM account, partition/queue, module names, and
   VASP/GPUMD executable paths for Anvil and ACES. See `references/hpc_notes.md`.
2. **`config/criteria.yaml`** — thresholds for "sufficient potential" (energy/force RMSE
   vs AIMD, kappa and TBC tolerance vs reference/experiment) and the deposition
   parameter ranges to explore (temperature, incident energy, angle, flux, substrate).
3. **`config/experiment_correlations.yaml`** — the real experimental deposition setup
   (technique, substrate, measured target TBC/kappa) that stage 6 correlates against.
   Leave a field blank/`null` if there's no experimental value yet — the workflow will
   optimize toward the simulated criteria alone in that case.

None of the scripts fabricate materials-science numbers; unfilled config is treated as
"ask the user" rather than guessed.

## Running a stage

Every script takes `--round N` (which training-data generation round it's operating on,
starting at 0) and `--cluster {anvil,aces}` where relevant, and reads/writes under a
single `runs/` working directory so state is inspectable and resumable:

```
runs/
  round_000/
    structures/          # stage 1 output (POSCARs)
    vasp/                # stage 2 output (per-structure VASP dirs + job ids)
    nep_dataset/          # stage 3a output (train.xyz / test.xyz)
    nep_model/            # stage 3b output (nep.txt, loss.out, job id)
    evaluation.json        # stage 4 output (RMSEs, kappa, TBC, verdict inputs)
    decision.json           # stage 5 output ({"sufficient": bool, "reason": ...})
  round_001/               # only created if round_000 was insufficient
    ...
deposition/
  setup.json                # stage 6 output
  sim_inputs/                # stage 7 output
  sweep/                      # stage 8 output (per-condition sim dirs + job ids)
  optimal_conditions.json      # stage 9 output — the end goal
```

Run an individual stage, e.g.:

```bash
python scripts/generate_structures.py --round 0 --n-structures 40
python scripts/submit_vasp.py --round 0 --cluster anvil
python scripts/vasp_to_nep_dataset.py --round 0
python scripts/submit_nep_training.py --round 0 --cluster anvil
python scripts/evaluate_potential.py --round 0 --cluster anvil
python scripts/decide_next_step.py --round 0
```

Or run the whole loop unattended once the config is trustworthy:

```bash
python scripts/agentic_orchestrator.py --cluster anvil --max-rounds 5
```

`agentic_orchestrator.py` exits (rather than looping forever) once
`decide_next_step.py` reports `sufficient: true`, once `--max-rounds` is hit, or if a
SLURM job fails — in the last case it stops and reports the failing job so the human
stays in the loop for anything compute-cluster related that this skill cannot fix
itself (queue limits, module errors, VASP convergence failures, license issues, etc.).

## What Claude should do vs. what the human must do

Claude (running this skill) should: generate structures, render and submit job
scripts, parse VASP/NEP output, compute evaluation metrics, make the sufficient/loop-back
call using the configured thresholds, build deposition simulation inputs, and summarize
results.

Claude should NOT: invent SLURM account numbers, partition names, VASP pseudopotential
choices, or experimental TBC/kappa target values. If `config/*.yaml` still has a
placeholder (`FILL_ME_IN`) where a stage needs a real value, stop and ask the user for
it rather than guessing — a wrong SLURM account fails fast and loudly, but a wrong
INCAR/experimental-target choice can silently waste a large compute allocation.

## Compute resources: Anvil vs ACES

Both are SLURM clusters, so the stage scripts are cluster-agnostic Python that renders
a small Jinja-style `.sbatch` template per cluster (`templates/vasp_<cluster>.sbatch`,
`templates/nep_<cluster>.sbatch`, `templates/deposition_<cluster>.sbatch`) and submits
with `sbatch`. The differences that matter are account/partition names, module load
lines, and executable paths — all isolated in `config/clusters.yaml`. See
`references/hpc_notes.md` for what to fill in and how to verify it (`sinfo`, `module
avail`, your ACCESS/TAMU HPRC allocation page) before the first submission.
