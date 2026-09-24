# NEP-AlN-Si

> **This repository is a worked example of
> [Agentic-AI-devleloping-NEP-for-Deposition](https://github.com/cook-the-library/Agentic-AI-devleloping-NEP-for-Deposition)**:
> the general agentic NEP-for-deposition workflow applied to one real material
> system, AlN on Si(111). Use that repository for the generic workflow, and this
> one to see how it is adapted to a specific system and what running it produces.

Agentic, closed-loop development of a GPUMD NEP (neuroevolution potential) for
**AlN deposited on Si(111)**, and use of that potential to optimize deposition
conditions on Purdue Anvil or TAMU ACES.

The loop generates Al–N–Si structures, runs VASP AIMD, trains the NEP, checks it
against energy/force RMSE, thermal conductivity (κ) and AlN/Si thermal boundary
conductance (TBC), and goes back for more training data until those pass. The trained
NEP then drives LAMMPS co-deposition of Al and N onto Si(111) across a sweep of
substrate temperature, incident energy, angle and V/III ratio.

The reasoning behind the training set and the validation steps is in
[`docs/AlN_Si_development_plan.md`](docs/AlN_Si_development_plan.md). Stage-by-stage
docs are in [`SKILL.md`](SKILL.md).

## Quick start

```bash
pip install -r requirements.txt
python scripts/aln_si_structures.py          # optional: inspect seed structures in seeds/
# Fill in every FILL_ME_IN in config/clusters.yaml and config/experiment_correlations.yaml,
# and review thresholds and the deposition sweep in config/criteria.yaml.
python scripts/agentic_orchestrator.py --cluster anvil --max-rounds 6
```

## What this example changes from the general workflow

- **Seeds** (`scripts/aln_si_structures.py`):
  - bulk wurtzite, zincblende and rocksalt AlN; bulk Si and Al
  - an N₂ molecule
  - Si(111) and AlN(0001) slabs, and Al and N atoms placed above each
  - a 228-atom AlN(0001)/Si(111) interface in a 5:4 coincidence cell (AlN strained −1.28%)
- **Polarity and interface termination** are set in `config/experiment_correlations.yaml`.
  Use Al-terminated for Al pre-deposition, N-terminated for nitrided Si.
- **Deposition** (`scripts/generate_deposition_simulation.py`):
  - separate Al and N streams on a 768-atom Si(111) slab
  - each species gets the speed matching the requested kinetic energy for its own mass
  - the incident angle is applied, and the V/III ratio is one of the sweep variables
- **Criteria** (`config/criteria.yaml`) are loosened for a 3-element system with
  surfaces and interfaces. Active learning is sized at 60 structures per round.

## Layout

| Path | Contents |
| --- | --- |
| `scripts/` | One script per stage, `agentic_orchestrator.py`, shared helpers in `_common.py`, AlN/Si structure builders in `aln_si_structures.py` |
| `config/` | Cluster, evaluation-criteria and experiment configs |
| `templates/` | SLURM `.sbatch` templates for VASP, NEP training and deposition runs |
| `docs/` | AlN/Si potential-development plan |
| `rounds/` | Scripts actually run on Anvil for training rounds 0, 1 and 1.5 (see `rounds/README.md`) |
| `references/` | HPC notes for Anvil vs ACES |

Outputs go to `runs/`, `deposition/`, `seeds/` and `rounds/round*_out/`, which are git-ignored.
