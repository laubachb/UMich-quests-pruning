# Data

Raw datasets are **not committed** (see `.gitignore`). Checksums are in
`SHA256SUMS`; paths below are on LLNL lustre.

| File | Role | Location |
|---|---|---|
| `ChIMES_C-2.0-Small_model_training_set.xyz` | Original training set: 685 frames, 134,784 atoms, DFT energies/forces/stress. `config_tag` is a file path, **not** a state-point label. | `/p/lustre1/laubach2/` |
| `ChIMES_C-2.0-Small_model_training_set.json` | ASE-db JSON dump of the same frames (positions/cell only, no labels or forces). | `/p/lustre1/laubach2/` |
| `DS_2km6t3p8hxx2_0.xyz` | Training set as used in the paper: 684 frames with QUESTS descriptors (`quests_descriptor_descriptors`, 63-dim), per-atom `weights`, and state-point `config_tag` labels. | `nequip_models/quests_descriptors-Mar2026/` |
| `random_C.xyz` | Held-out test set: 100 frames × 12 state points, sampled from the same DFT-MD trajectories. **Forces are in Hartree/Bohr** (×51.422 → eV/Å). | `nequip_models/becky_files/from_kyl-apr2026/` |

## Group labels

`build_group_labels.py` reconstructs the 14 state-point labels for the original
file by geometric matching against `DS_2km6t3p8hxx2_0.xyz`, producing
`group_labels.csv` (`orig_index, labeled_index, natoms, density_gcc, group, dup_of_orig`).

Findings from that matching (2026-10-08):

* **Duplicate frames.** The original contains 10 exact duplicate pairs
  (consecutive frames); the labeled file contains 9 of them. The 685th frame in
  the original is the second copy of a 0.5 g/cc, 1000 K liquid frame
  (orig 613/614). `dup_of_orig >= 0` marks the second copy; drop these to get
  675 unique frames.
* **Boundary mislabels in the paper's file.** For a duplicate pair in the
  labeled file, the second copy carries the *next* group's label (e.g. labeled
  frame 7 is a 3.6 g/cc liquid tagged `HD liquid_2.0gcc_6000`; frame 598 is a
  2.0 g/cc liquid tagged `LD liquid_1.0gcc_2000`). ~9 frames in the paper's
  stratification are therefore in the wrong group. `group_labels.csv` assigns
  the density-consistent label.
* **Nominal-density mismatches.** `diamond_3.68gcc_300` frames are actually
  3.56 g/cc; `diamond_3.67gcc_3000` frames are 3.68 g/cc. The `_0.0gcc_CC`
  (cold-curve) groups span a range of densities by construction. Test tag
  `test_LD_liquid_0.5gcc_2000` is really 1.0 g/cc.

## Test set independence

`random_C.xyz` is drawn from the same trajectories as training (Reviewer 1 #2).
A trajectory-disjoint split still needs to be built; the `CASE-*_INDEP_*`
naming in the MD inputs suggests independent trajectories exist upstream.
