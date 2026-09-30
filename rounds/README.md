# Production rounds on Anvil (round 0, 1, 1.5)

The scripts that were actually run on Purdue Anvil to build the Al–N–Si training
set and train the NEP. They are kept as they were run, except for the
150-atom limit and the geometry fixes described below. `../scripts/` is the
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
- Every structure has fewer than 150 atoms (`ATOM_LIMIT = 150`). The rules are
  applied at every stage, in every round:
  - the generators clamp `--max-atoms` and `--max-atoms-interface` to 149 and
    drop anything larger;
  - `round1.5_1_make_rerun_list.sh` skips folders whose POSCAR has 150 or more
    atoms;
  - every `*_5_build_dataset.py` rejects frames with 150 or more atoms
    (`too_many_atoms` in round 1wL).
- The structure generator covers 9 buckets: bulk substrate and film, dimers and
  trimers, slabs, adsorption, interface, isolated clusters and disordered.
- Each round generates 1,000 structures, split across the buckets in the same
  proportions (96 per bulk/slab/dimer/disordered bucket, 192 adsorption,
  192 interface, 40 isolated). Round 1 defaults to `--seed 1` so it doesn't
  regenerate round 0. The folders already run on Anvil came from the older
  counts: 520 in round 0 and 1,560 in round 1.
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

## Known issues (only the geometry bugs are fixed in the code)

- **Every dataset is tagged round 0.** `attach_nep_fields()` in all three
  `*_5_build_dataset.py` sets `info["round"] = 0`. The round 1 and 1.5 datasets
  therefore label every frame as round 0, so any per-round statistics built on
  that field are wrong.
- **Round 1 repeated part of round 0** (fixed: round 1 now defaults to `--seed 1`).
  The round 1 that ran used the same `--seed 0` as round 0, and several scans are deterministic. Regenerating both
  rounds locally without POTCARs gave 100 of round 1's 1,560 structures that are
  exact copies of round-0 ones. By bucket: adsorption 48/300, bulk Si 25/150,
  dimers/trimers 12/150, and a few in each other bucket. These cost VASP time and
  add no new information.
- **Round 1 is still random sampling.** Nothing selects structures where the
  round-0 NEP is uncertain. For round 2, use a new `--seed` and pick new
  structures by NEP error or uncertainty.
- **Round 0 and 1 data were built with two geometry bugs, since fixed in the
  generators.** The VASP folders that already ran on Anvil still carry them:
  - *Broken interface films.* The matcher compared reduced 2D cells, allowing
    a 60°/120° flip, but `build_interface` mapped the unreduced film supercell
    onto the substrate's. The Si(111) cell has a 60° angle and the AlN cell a
    120° one, so the "1.26% strain" film was really stretched about +71% / −43%.
    Al–N contacts went down to 1.19 Å and coordination to 2–3. This affects all
    400 interface folders (196 atoms each). The 150-atom limit already keeps
    them out of every dataset.
  - *Shuffle-cut Si(111).* ase's diamond(111) cut runs through the long vertical
    bond, so every Si slab, adsorption substrate and interface substrate had
    surface atoms with one bond and three dangling bonds. That affects all
    `slab_Si`, `adsorption` and `interface` folders. The slab and adsorption
    folders are still in the datasets; drop or recompute them.

  The generators now trim singly-bonded planes, so Si(111) surfaces keep three
  bonds per atom. The matcher returns the exact vector pairs it compared,
  `build_interface` counts thickness in atomic planes, and `lattice_match.txt`
  reports the real film strain. Under 150 atoms the Si(111)/AlN(0001)
  interface is the 5:4 coincidence cell (16 Si : 25 AlN cells, 1.25% strain)
  with 3 Si bilayers and 1 Al–N bilayer, 146 atoms. The same Si fix is in
  `scripts/aln_si_structures.py`, which builds the deposition substrate.
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
