# Production rounds on Anvil (round 0, 1, 1.5)

The scripts that were actually run on Purdue Anvil to build the Al–N–Si training
set and train the NEP. They are kept as they were run. `../scripts/` is the
general agentic workflow; this folder is the hand-driven campaign it grew out of.

**`rounds/` is the project root.** Every script finds its files relative to the
folder that holds `round0_code/`, so submit from here. Put the POTCARs in
`rounds/aaa-potential/<element>/POTCAR` (git-ignored, VASP licence). Outputs land in
`rounds/round*_out/` (also git-ignored).

## Pipeline per round

| Step | Round 0 | Round 1 | Round 1.5 |
| --- | --- | --- | --- |
| 1. structures | `sbatch round0_code/sbatch_round0.sbatch` → `generate_round0.py` | `sbatch round1_code/sbatch_round1.sbatch` → `generate_round1.py` | none: re-runs unfinished round 0/1 folders (`round1.5_1_make_rerun_list.sh`) |
| 2. VASP array | `round0_2_anvil_subVASP.sbatch` (16 cores, 12 h) | `round1_2_anvil_subVASP.sbatch` (24 cores, 12 h, chunked arrays) | `round1.5_2_anvil_rerun_vasp.sbatch` (32 cores, 24 h, WAVECAR/CHGCAR restart + STOPCAR 30 min before walltime) |
| 3–4. audit + dataset | `round0_34_anvil_check_and_build.sbatch` | `round1_34_…` (round 0 + 1 combined, round-0 split carried) | `round1.5_34_…` (all round 0 + 1 folders, round-1 split carried) |
| 5. dataset builder | `round0_5_build_dataset.py` | `round1_5_build_dataset.py` | `round1.5_5_build_dataset.py` |
| 6. NEP | `round0_8_nep.sbatch` + `round0_nep-1.in` | `round1_nep.sbatch` (continues round-0 `nep.restart` if the architecture matches) | `round1.5_nep.sbatch` (continues round 1) |
| chain | `bash round0_code/round0_generateVASP+trainNEP.sbatch` | `bash round1_code/round1_generateVASP_trainNEP.sbatch` | `bash round1.5_code/round1.5_generateVASP_trainNEP.sbatch` |

Run each chain script with `bash` from a login node, after step 1 has finished. It
submits the VASP array, then the audit/build, then NEP training, with SLURM dependencies.

**What stays the same across rounds**
- The structure generator covers 9 buckets: bulk substrate and film, dimers and
  trimers, slabs, adsorption, interface, isolated clusters and disordered.
- Round 0 generates 520 structures. Round 1 uses the same generator with 3× the counts.
- One ENCUT is used for the whole campaign: 1.3 × the largest ENMAX. The k-point
  density is fixed.
- The dipole correction is applied to every cell with vacuum.
- The dataset builder:
  - writes virials
  - splits train/test by structure family, so near-duplicates stay on one side
  - stratifies the split by bucket
  - carries earlier rounds' train/test labels forward, so `test.xyz` stays fixed
    for the whole campaign

**`legacy/`** holds the earlier versions these replaced: a frame-level random split
with no virials, and a check script that assumed a flat directory layout.

## Known issues (not fixed; the code is kept as it ran)

- **Every dataset is tagged round 0.** `attach_nep_fields()` in all three
  `*_5_build_dataset.py` sets `info["round"] = 0`. The round 1 and 1.5 datasets
  therefore label every frame as round 0, so any per-round statistics built on
  that field are wrong.
- **Round 1 repeats part of round 0.** `generate_round1.py` uses the same default
  `--seed 0` as round 0, and several scans are deterministic. Regenerating both
  rounds locally without POTCARs gave 100 of round 1's 1,560 structures that are
  exact copies of round-0 ones. By bucket: adsorption 48/300, bulk Si 25/150,
  dimers/trimers 12/150, and a few in each other bucket. These cost VASP time and
  add no new information.
- **Round 1 is still random sampling.** Nothing selects structures where the
  round-0 NEP is uncertain. For round 2, use a new `--seed` and pick new
  structures by NEP error or uncertainty.
- **The interface is one AlN layer thick.** Under the default `--max-atoms 200`, the
  lattice match is 16 Si(111) cells to 25 AlN cells (1.26% strain). Only 3 Si
  layers and 1 AlN layer (about 3 Å) fit, so every interface structure is
  196 atoms. The generator itself warns about this, but `sbatch_round*.sbatch`
  doesn't pass `--max-atoms-interface`. The 3 Si + 2 AlN layers it recommends
  would need about 296 atoms per cell.
- **Loss weights.** `lambda_e 0.1` against `lambda_f 50` weights forces about
  500× more than energies. The round 1.5 notes say round 0's energy RMSE never
  came down. `round0_nep-2.in` (`lambda_e 10`, `lambda_f 3`) is the alternative
  that was prepared.
- **Round-1 `fine_tune` line.** `round1_nep-1.in` starts with an uncommented
  `fine_tune` line. `round1_nep.sbatch` comments it out before training, so it
  is harmless there, but running `nep` directly with that file would try to
  fine-tune.
- **Hard-coded Anvil settings.** Allocation names (`mat260051`, `mat260051-gpu`),
  the mail address, the module versions and the `nep` binary path are fixed in
  the `.sbatch` headers. Edit them before running under another account or on ACES.
