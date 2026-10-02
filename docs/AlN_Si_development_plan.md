# Potential development plan: AlN on Si

How this repository trains an Al–N–Si NEP that can simulate c-plane AlN
growing on Si(111), and then uses it in the deposition-optimization stages
(see [`../SKILL.md`](../SKILL.md) for the stage-by-stage workflow).

## Why this system is hard for a potential

- **Three elements, very different bonding.** The potential has to handle ionic/covalent
  Al–N, covalent Si–Si, metallic Al–Al, and the Si–N and Al–Si bonds that only
  show up at the interface.
- **Large lattice mismatch.** AlN (a = 3.112 Å) on Si(111) (surface a = 3.840 Å)
  is about 19% off. Real films relax through a 5:4 coincidence lattice with misfit
  dislocations, so the training set needs the coincidence cell, not a coherently
  strained one.
- **Interface chemistry depends on the process.** Nitrogen reaching bare Si forms
  amorphous SiNₓ. Growers usually lay down an Al layer first to prevent this. The
  potential must cover both outcomes, because the deposition sweep can produce either.
- **Polarity.** Films grown on Si(111) are usually Al-polar. The polarity sets which
  AlN species bonds to Si, and this changes the interface phonons and therefore the TBC.
- **Thermal targets.** κ of bulk AlN (~285–340 W/mK) and Si (~148 W/mK) at 300 K
  are good checks on the bulk parts of the potential. TBC is the property the
  deposition sweep actually ranks by.

## Training-data plan

| Block | Structures | Why | How |
| --- | --- | --- | --- |
| Bulk | wurtzite, zincblende, rocksalt AlN; diamond Si; fcc Al | lattice, elastic and phonon behaviour; rocksalt covers high-energy impacts | `aln_si_structures.py` → strain + rattle; AIMD at 300–1500 K |
| Molecules | N₂ in a box | N₂ release and recombination at the surface | seed + AIMD |
| Surfaces | Si(111), AlN(0001), both polarities | the growth front | seeds; add 7×7-like / 2×2 reconstructions later |
| Adatoms | Al and N on Si(111) and AlN(0001) | the individual deposition events | seeds; add NEB paths for diffusion barriers |
| Interface | AlN(0001)/Si(111), 5:4 coincidence, Al–Si and N–Si terminated | TBC and interface stability | 228-atom seed; AIMD at 300 and ~1000 K |
| Disorder | amorphous SiNₓ, Al–Si liquid, amorphous AlN | outcomes of bad growth conditions the MD will visit | NEP melt-quench (bulk, then cleaved with vacuum) + DFT single points: `rounds/round1md_code/` |
| Collisions | Al, N, N₂, Ar impacts at 1–100 eV on Si(111), AlN(0001)/(000-1), amorphous slabs | the deposition events themselves, with the substrate responding | NEP MD + DFT single points: `rounds/round1md_code/` |
| Active learning | frames from NEP deposition MD with the highest uncertainty | cover what the MD actually visits | main loop, `bias_toward_high_error` |

The interface cell is 228 atoms (4 Si bilayers + 2 AlN cells). That's expensive but
workable for short AIMD with Γ-only k-points. Use `--si-bilayers` / `--aln-cells` to
make it thinner for the first rounds.

**VASP settings to fix before round 0.** Use one functional for the whole dataset
(PBE or optB86b-vdW, not mixed), PAW `Al`, `N`, `Si`, ENCUT ≥ 520 eV, and
dipole correction (`LDIPOL`/`IDIPOL = 3`) for polar AlN slabs. Keep ENCUT, PREC
and the k-spacing identical across every structure so energies are consistent.

## Validation ladder

Go on to the next check only when the current one passes:

1. Energy, force and virial RMSE on the held-out split (thresholds in `config/criteria.yaml`)
2. Lattice constants, elastic constants and phonon dispersions of AlN and Si, compared with DFT
3. Surface energies of Si(111) and AlN(0001), and Al/N adatom adsorption energies
4. Interface adhesion energy for both terminations; MD at 1200 K must keep the interface intact
5. κ(AlN) and κ(Si) from HNEMD at 300 K, compared with experiment
6. AlN/Si TBC from NEMD, compared with your own TDTR measurement (`config/experiment_correlations.yaml`)

## Where this lives in the code

| Path | Role |
| --- | --- |
| `scripts/aln_si_structures.py` | builds every seed above; run it directly to inspect them in `seeds/` |
| `scripts/generate_structures.py` | stage 1: strains bulk seeds, rattles everything to fill a round |
| `scripts/generate_deposition_simulation.py` | stage 7: Si(111) substrate with separate Al and N deposition streams |
| `config/experiment_correlations.yaml` | growth polarity, interface termination, measured targets |
| `config/criteria.yaml` | RMSE thresholds, κ/TBC checks, deposition sweep (substrate T, energy, angle, V/III ratio) |

```bash
python scripts/aln_si_structures.py                    # Al-polar, Al-Si interface -> seeds/
python scripts/aln_si_structures.py --termination N    # N-Si interface
```

## Open items

- [ ] Optional Al pre-deposition step in stage 7 (deposit an Al wetting layer before switching on N)
      to match `interface_termination: "Al"` growth.
- [x] Add amorphous SiNₓ / a-AlN melt-quench inputs (round 1md runs them with the
      NEP instead of AIMD, and labels selected frames with VASP).
- [ ] Fix the Si(111) termination and the interface-film shear in the round
      generators (see "Known issues" in `rounds/README.md`).
- [ ] Decide on the deposition technique (sputtering vs MOCVD vs MBE). It sets the
      incident-energy range and whether NH₃/H needs to be a fourth element.
- [ ] Fill in measured κ / TBC targets.
