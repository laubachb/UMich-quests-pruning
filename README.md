# Entropy-Guided Dataset Reduction — reproduction

Reproduces and extends Laubach et al., *Entropy-Guided Dataset Reduction for Machine-Learned Interatomic Potentials* (ChemRxiv 10.26434/chemrxiv.15008542).

Layout:
- `data/`       dataset pointers, checksums, group-label mapping (raw xyz not committed)
- `pruning/`    QUESTS descriptors, environment-level FPS, elbow bandwidth selection
- `training/`   NequIP config + custom masked-loss module + submission scripts
- `evaluation/` force-MAE evaluation, bootstrap CIs
- `md/`         LAMMPS/TRAVIS inputs for RDF validation
- `figures/`    one script per figure, reads only from `results/`
- `results/`    small CSV outputs (committed)
- `env/`        pinned environment
