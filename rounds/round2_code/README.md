# Round 2: MD-sampled and thermally rattled structures

Rounds 0 and 1 built every training structure by hand. Round 2 splits the job:

- **Amorphous and collision structures come from MD with the incomplete NEP**
  (`round1wL_out/round1wL_nep.txt`, N Al Si Ar). The MD goes wherever this
  potential takes deposition MD, including the places where it is wrong, and
  those are the structures the next training set needs.
- **Thermally rattled crystals, slabs and interfaces come from Python**, with
  temperature- and mass-dependent amplitudes and no potential at all.

The decisions that set this round (from your answers):

- **Process:** PVD with Ar sputter gas.
- **Collisions:** normal incidence, up to 50 eV, substrate at 300 and 900 K.
  Targets include Al-covered Si(111), nitrided Si(111) and the AlN/Si
  interface.
- **VASP:** 16 cores and 12 h per job, ISPIN = 1, and the campaign's k-point
  density kept for accuracy.
- **Committee:** the round-1.7 NEP checks every MD frame.
- **Old data:** the broken round 0/1 geometries are removed from it.
- **Training:** forces first, then energy, starting from round 1wL's
  `nep.restart`. New structures go into test as well as train.

## Pipeline

| Step | Script | What it does |
| --- | --- | --- |
| 1a | `round2_1_build_md_inputs.py` | MD run folders under `round2_out/md/`: melt-quench, collisions, precursors (pass 1), then collisions on the surfaces pass 1 made (`--pass2`). |
| 1b | `round2_1_thermal_rattle.py` | 190 thermally rattled structures as VASP folders in `round2_out/vasp_thermal/`. |
| 2 | `round2_2_anvil_lammps.sbatch` | LAMMPS array: the MD (`in.melt_quench.lmp`, `in.collision.lmp`), then the committee rerun (`in.rerun.lmp`). |
| 3 | `round2_3_harvest.py` (+ `.sbatch`) | Screens the frames, picks about 900 (uncertain ones first), writes `round2_out/vasp_md/`. |
| 4 | `round2_4_anvil_subVASP.sbatch` | VASP array over both folder sets, 16 cores, 12 h, WAVECAR/CHGCAR restart. |
| 5 | `round2_5_build_dataset.sbatch` | New-frame dataset, NEP-vs-DFT report (`round2_6_nep_vs_dft.py`), filter of the old data (`round2_5_contact_filter.py`), merge. |
| 6 | `round2_nep.sbatch` + `round2_nep-{1,2}.in` | Stage 1: λ_e 0.1, λ_f 50, 200k. Stage 2: λ_e 10, λ_f 1, 200k. |

`round2_common.py` holds the structure builders and helpers that steps 1a,
1b and 3 share.

Run everything from `rounds/` (the project root), with the conda env loaded
(`module load conda; module use $HOME/privatemodules; module load conda-env/nepdata`):

```bash
python round2_code/round2_1_thermal_rattle.py                        # step 1b
python round2_code/round2_1_build_md_inputs.py                       # step 1a, pass 1
NO_HARVEST=1 LMP_BIN=/path/to/lmp bash round2_code/round2_run_md.sh  # pass-1 MD
# when pass 1 has finished:
python round2_code/round2_1_build_md_inputs.py --pass2
LMP_BIN=/path/to/lmp bash round2_code/round2_run_md.sh               # pass-2 MD + harvest
# read round2_out/vasp_md/harvest_report.txt, then
bash round2_code/round2_generateVASP_trainNEP.sh                     # steps 4-6
```

The builder expects the committee potential at `round1.7_out/round1.7_nep.txt`.
Use `--committee <path>` if it lives elsewhere, or `--committee none` to switch
it off. LAMMPS needs NEP_CPU's `pair_style nep`, with the current syntax:
`pair_style nep` / `pair_coeff * * nep.txt N Al Si Ar`. Atom types follow the
element order of `nep.txt`.

## MD part

### Melt-quench (`in.melt_quench.lmp`)

The bulk cell is quenched first, then cleaved and given vacuum. The volume is
fixed throughout (NVT); a density scan over run folders takes the place of
NPT, which an incomplete potential can collapse or blow apart.

| Stage | Ensemble | Dump |
| --- | --- | --- |
| relax | `nve/limit` + Langevin, 300 K, 1 ps. Removes the close contacts of the random packing. | none |
| melt | NVT at T_melt, 20 ps | `dump.melt.lammpstrj` |
| quench | NVT ramp from T_melt to 300 K at 50 or 10 K/ps | `dump.quench.lammpstrj` |
| anneal | NVT at 300 K, 10 ps | `dump.anneal.lammpstrj`, `bulk_final.data` |
| slab | Shift by a random `zshift`, add 12 Å of vacuum, heat to 900 K, hold, cool to 300 K | `dump.slab.lammpstrj`, `slab_final.data` |

Each system is run at 0.92, 1.00 and 1.08 × the listed density, with about 96
atoms. The densities are rough starting values, not fitted numbers.

| System | Why | Density g/cm³ | T_melt K |
| --- | --- | --- | --- |
| AlN, Al₃N₂, Al₂N₃ | a-AlN, Al-rich and N-rich films (N₂ forms in Al₂N₃) | 3.0 / 2.9 / 2.8 | 3500 / 3000 / 3500 |
| Si | a-Si | 2.3 | 2500 |
| Si₃N₄, SiN, Si₂N | SiNₓ from N reaching bare Si | 3.0 / 2.8 / 2.6 | 3500 / 3200 / 3000 |
| AlSi, Al₃Si | Al–Si phases of Al pre-deposition | 2.45 | 2200 / 2000 |
| AlSiN₂ | intermixed interface | 3.0 | 3500 |
| AlN + 3% Ar, Si + 3% Ar | sputter gas trapped in the film and in the Ar-damaged Si surface | 2.95 / 2.3 | 3500 / 2500 |

### Collisions (`in.collision.lmp`)

- **Pass-1 targets.** All are built in `round2_common.py` and cut between
  bilayers; the builder checks that every surface atom has coordination 3.
  - Si(111), Al-polar AlN(0001) and N-polar AlN(000-1): 4 × 4 × 4 bilayers,
    128 atoms each.
  - AlN(0001)/Si(111) interface: a 5:4 coincidence cell, Al-terminated as in
    `config/experiment_correlations.yaml`, 189 atoms. The film is strained
    −1.25 % in plane and not sheared.
- **Precursors (pass 1).** These runs build the remaining targets:
  - `pre-AlSi111`: 24 Al atoms at 2 eV onto Si(111) at 300 K (Al
    pre-deposition).
  - `pre-NSi111`: 24 N atoms at 5 eV at 900 K (nitridation).

  Their frames are harvested like any other collision.
- **Pass-2 targets.** Collisions on what pass 1 produced: Al-covered Si(111),
  nitrided Si(111), and up to 6 cleaved amorphous slabs.
- **Impacts.** There are up to 8 per run, fewer if the cell would exceed 200
  atoms (the interface gets 5).
  - Species are drawn per impact: Al 30%, N 25%, N₂ 20%, Ar 25%.
  - Energies are 1, 5, 10, 20 or 50 eV, at normal incidence, on a random
    aim point.
  - Substrate temperature is 300 or 900 K.
  - The surface keeps what earlier impacts did to it.
- **Layers.** The bottom layer is fixed, above it is a Langevin bath, and the
  top and the projectiles run NVE. `fix dt/reset` limits every atom to 0.04 Å
  per step.
- **Dumped forces.** The dump holds the plain NEP forces, stored with
  `fix store/force` before `setforce` and the Langevin bath change them. The
  first version dumped the thermostatted forces, which made bath and fixed
  atoms look like a 2–8 eV/Å disagreement.

### Committee rerun (`in.rerun.lmp`)

After the MD, the round-1.7 NEP re-evaluates every dumped frame. Snapshots are
read whole (`purge yes add keep`), because collision frames gain projectiles
and lose sputtered atoms. With the same potential on both sides the rerun
reproduces the MD forces to the dump precision (6 × 10⁻³ eV/Å).

### Screening and selection (`round2_3_harvest.py`)

The contact ratio is the smallest d_ij / (r_cov,i + r_cov,j) over all pairs
in a frame.

| Check | Default | Consequence |
| --- | --- | --- |
| contact ratio < keep ratio | 0.60 quench, 0.50 collision | frame not used |
| contact ratio < 0.40 | collapse | this and every later frame of the run discarded (not during the impact itself) |
| NEP max \|F\| | > 50 / 300 eV/Å | frame not used |
| projectile still > 4 Å from the surface | | not used (it is just the slab) |
| last 3 good frames before a collapse or `fix halt` | | always selected |
| committee max \|F_NEP − F_1.7\| ≥ 0.3 eV/Å | uncertain | fills the budget first; frames below 0.3 eV/Å top it up to at most 25% |

Within each stratum, frames are picked by farthest-point sampling in a
pair-distance-histogram descriptor, with a per-run cap. Collision frames are
described only by the atoms the impacts changed.

| Bucket | Frames | Split |
| --- | --- | --- |
| amorphous_bulk | 300 | melt 30%, quench 50%, anneal 20% |
| amorphous_slab | 150 | |
| collision | 450 | impact 60%, relaxation 40% |

The round-1wL training set seeds the bulk/slab sampling, so frames that look
like data already there are picked last.

Slab and collision frames lose any atom more than 4 Å from the slab. With
ISPIN = 1 a lone N or Al atom would get the wrong spin state. They then get
15 Å of vacuum in total.

The report gives the committee disagreement per bucket and stage (p10, p50,
p90). Use it to tune `--dev-lo`.

## Thermally rattled structures (`round2_1_thermal_rattle.py`)

Each atom is displaced by a Gaussian. Its mean-square amplitude per axis is
the Debye–Waller value, including zero-point motion:

  ⟨u_x²⟩ = 3ħ²T / (m k_B Θ_D²) · [φ(x) + x/4],  with x = Θ_D/T

- **Debye temperatures:** Si 543 K (the Debye–Waller value), AlN 950 K,
  Al 390 K. These are approximate literature values; `--theta-scale` scales
  them.
- **Check:** for Si at 300 K this gives B = 8π²⟨u_x²⟩ = 0.453 Å², against
  0.46 Å² measured.
- **Surfaces:** atoms within 1 Å of a surface get 1.5 × the mean-square
  amplitude.
- **Strain:** a random strain of up to ±1.5% is applied, in plane only for
  slabs.
- **Screening:** a frame is redrawn if any pair falls below 0.70 of its
  covalent-radius sum.

σ per axis in Å (`amplitudes.txt`):

| | 300 K | 600 K | 900 K | 1200 K | 1500 K |
| --- | --- | --- | --- | --- | --- |
| Al in AlN | 0.047 | 0.062 | 0.074 | 0.085 | 0.095 |
| N in AlN | 0.066 | 0.086 | 0.103 | 0.119 | 0.132 |
| Si | 0.076 | 0.104 | 0.126 | 0.146 | 0.163 |
| Al (fcc) | 0.105 | 0.147 | 0.179 | | |

| Bucket | Structures | Temperatures | Frames |
| --- | --- | --- | --- |
| thermal_bulk | wurtzite, zincblende and rocksalt AlN; diamond Si; fcc Al (64–108 atoms) | 300–1500 K (Al ≤ 900 K) | 138 |
| thermal_slab | Si(111), AlN(0001), AlN(000-1) (96 atoms) | 300, 600, 900 K | 36 |
| thermal_interface | AlN/Si(111), Al- and N-terminated, random registry, gap ± 0.15 Å (164/189 atoms) | 300, 600 K | 16 |

## VASP

All folders use round 1's own writer:
- ENCUT = 1.3 × the largest ENMAX (520 eV)
- k-spacing 0.25 Å⁻¹, which keeps 2 k-points along the vacuum direction of
  slabs (accuracy first)
- dipole correction on cells with vacuum
- ISPIN = 1

The INCARs are written for 16 MPI ranks: NCORE = 4, and KPAR = 4 unless a
cell has fewer irreducible k-points (`round2_common.set_vasp_parallel`).

The array is round 1.5's restart-capable script with 16 cores and 12 h. A job
that reaches the walltime stops cleanly 30 min early and keeps WAVECAR, so the
164–189-atom interface folders may need a second pass.

## Dataset and training

- **New frames** (`round2_out/dataset_new`): the round-agnostic round-1.5
  builder, with a family-based split stratified by bucket. 20% of the new
  frames go to test.
- **Old frames** (`round2_out/dataset_prev_filtered`): the round-1wL set
  without condensed frames whose closest pair is below 0.65 of its
  covalent-radius sum (`--min-contact`, env `MIN_CONTACT`). That catches the
  sheared interface film (ratio 0.62) and the overlapping placements; dimers
  and trimers are kept. On structures regenerated with `generate_round1.py` it
  drops:
  - all 300 interface frames
  - all 37 `disordered_film_on_slab` frames
  - 29 of 31 `compressed_rattle` frames
  - the large-σ bulk rattles, the 1.0 Å adatoms and the overlapping clusters

  `report.txt` and `dropped.csv` list what is dropped from the real data.
- **Merge** (`round2_out/dataset`): round 1wL's builder, with its rejection
  rules; each source keeps its own split.
- **Training:**
  - Stage 1 continues round 1wL's `nep.restart`, with λ_e 0.1 / λ_f 50 for
    200k generations.
  - Stage 2 continues stage 1's restart, with λ_e 10 / λ_f 1 for 200k
    generations.
  - The architecture is the same as round 1wL.
  - The final model is `round2_out/round2_nep.txt`.
- **NEP vs DFT:** `nep_vs_dft.txt` reports the round-1wL potential's error on
  the frames its own MD produced.

## Compared with the round 0/1 adsorption and displaced structures

The generators of round 0 and round 1 use the same algorithms; round 1 only
triples the counts. I ran `generate_round1.py` locally without POTCARs and
measured what it produces.

### Adsorption (`gen_adsorption`)

What it does:
- Al or N adatoms at the top, bridge and hollow sites of a 2 × 2 Si(111)
  slab, at 6 fixed heights from 1.0 to 4.5 Å.
- Random 2–5-atom clusters.
- Coverages from 0.25 to 1 ML.
- Points along straight lines between sites, as stand-ins for diffusion paths.

All of it is static, and the substrate stays frozen at its ideal positions.

1. **The substrate never responds.** A collision produces the approach, the
   impact, the substrate's response and the relaxed bound state that the
   static scan only guesses at.
2. **The heights are not physical.** 1.0 Å above a top site is Al–Si at
   0.43 of the covalent-radius sum, deep in the ZBL range. In a collision the
   closest approach is set by the kinetic energy (≤ 50 eV here).
3. **The Si(111) slab has the wrong termination.** Every top and bottom Si
   atom has coordination 1 (three dangling bonds) instead of 3. Round 2 cuts
   between bilayers, here and in `scripts/aln_si_structures.py`.
4. **Only Si is ever a substrate.** Round 2 adds both AlN polarities, the
   interface, Al-covered and nitrided Si(111), and amorphous surfaces.
5. **There is no Ar**, the main energetic species in PVD. Round 2 uses Ar as a
   projectile and in two amorphous systems.
6. **Same seed, deterministic scan.** Round 1 repeats 48 of round 0's 300
   adsorption structures exactly.

What collisions do not replace is a static height scan for adsorption-energy
curves, and NEB for diffusion barriers, which short MD rarely crosses. Both
would be a small separate set, from 1.8 Å up, on a correctly terminated slab.

### Displaced ("dislocated") structures

What it does:
- Independent Gaussian displacements of every atom, at fixed amplitudes:
  - σ up to 0.22 Å for bulk
  - 0.20–0.50 Å for `compressed_rattle`
  - 0.25–0.45 Å for `amorphous_interlayer`
- Random packings at 0.75–1.12 × the crystal density.
- A random film placed on a slab.
- Atoms swapped across the interface.

1. **The amplitudes ignore temperature and mass.** `compressed_rattle` has a
   median contact ratio of 0.53 (Si–Si at 1.15 Å). Round 2's Python rattle
   uses Debye–Waller amplitudes; its thermal_bulk frames have a contact ratio
   of 0.72 or more (median 0.85).
2. **A random packing is not an amorphous solid.** 53% of the first-neighbour
   Al/N bonds in round-1 `random_packing` AlN are Al–Al or N–N. In NEP
   melt-quenched a-AlN it is 14%. Round 2 makes amorphous structures only by
   MD.
3. **One distance threshold for every pair.** Every `disordered_film_on_slab`
   frame has an Al–Al pair at about 0.90 Å, because placements were checked
   against the N–N distance. Round 2 checks each pair against its own sum of
   covalent radii.
4. **The interface film is sheared.** Every round 0/1 interface structure has
   50 Al–N pairs at 1.19 Å: a 120° film cell was scaled onto the 60° Si(111)
   cell. Round 2 builds a 60° film cell, so the mapping is a −1.25% scale and
   Al–N stays at 1.869 Å. The old frames are filtered out of the training data.
5. **No crystallographic dislocations.** Real misfit dislocations need cells
   far beyond VASP size. A later round could run NEP MD on a large interface
   and cut the dislocation cores out as DFT-sized clusters.

## Testing

I tested round 2 locally with LAMMPS 22 Jul 2025 and NEP_CPU built as a
plugin. NEP89 (`nep89_20250409`, which covers N, Al, Si and Ar) stood in for
`round1wL_nep.txt`. A copy of NEP89 with every weight perturbed by 2% stood in
for the round-1.7 committee. Neither real potential is in the repository.

- **Pass 1 and pass 2.** Short runs of every kind finished through the sbatch
  script, with committee reruns: quench (AlN, Si+Ar), collisions on Si(111)
  and the 189-atom interface, both precursors. Pass 2 then ran on the
  amorphous slabs and the precursor surfaces.
- **Precursors at production settings.** With the default 2 ps relaxation
  per impact, the mobile layer cools back to about the substrate temperature
  before each impact: 310–400 K in the Al run (T_sub 300 K), 790–980 K in the
  N run (T_sub 900 K). With 0.2 ps in a short test the slab overheated and
  lost its bilayer layering, so do not shorten `--t-slow` for real runs.
  The full-length runs with NEP89 ended as follows:
  - Al pre-deposition: the lower Si bilayers stay intact and the Al mixes
    into the top bilayer.
  - Nitridation: the Si above the fixed bilayer turns into amorphous SiNₓ.
    20 of the 24 N are bonded to three Si (as in Si₃N₄), and one N₂ forms.
  Your NEP will give its own surfaces; check them before pass 2.
- **Harvest.** It wrote 900 folders with committee-first selection. The
  committee comparison uses the plain NEP forces (`fix store/force`); with the
  same potential on both sides every atom agrees to 6 × 10⁻³ eV/Å. Against
  the 2%-perturbed copy, the median disagreement is 0.2 eV/Å for collision
  frames and 0.3–0.5 eV/Å for amorphous frames. 12% of collision frames and
  50–64% of amorphous frames are above `--dev-lo` 0.3.
- **Dataset steps.** On the harvested frames with synthetic labels (noise
  15 meV/atom, 150 meV/Å), `round2_6_nep_vs_dft.py` reports those values
  exactly. The contact filter and the merge ran on them.
- **Thermal rattle.** 190 folders, contact ratio ≥ 0.72, NCORE 4 / KPAR 4.
- **Contact filter.** Run on the regenerated round-1 structures (numbers
  above).
- **NEP job.** Tested with a stub `nep`: stage 1 continues round 1wL's
  restart, stage 2 continues stage 1's, and FRESH, RESUME and a wrong STAGE
  behave as documented.
- **Chains.** Tested with a stub `sbatch`: array chunking at MaxArraySize, and
  the dependencies VASP → build → stage 1 → stage 2.
- **Not run on Anvil.**
