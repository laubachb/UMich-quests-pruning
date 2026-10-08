# Entropy-Guided Dataset Reduction — reproduction

Reproduces and extends Laubach et al., *Entropy-Guided Dataset Reduction for Machine-Learned Interatomic Potentials* (ChemRxiv 10.26434/chemrxiv.15008542).

Layout:
- `data/`       dataset pointers, checksums, group-label mapping (raw xyz not committed)
- `pruning/`    QUESTS descriptors, environment-level FPS, elbow bandwidth selection
- `training/`   NequIP config + custom masked-loss module + submission scripts
- `evaluation/` force-MAE evaluation, bootstrap CIs
- `md/`         LAMMPS/TRAVIS inputs for RDF validation
- `figures/`    one script per figure, reads only from `results/`
- `results/`    small CSV outputs and FPS orderings (committed); pruned xyz / models are not
- `env/`        pinned environment

## Workflow split

**Selection machine (CPU, e.g. dane)** — everything except training:

```bash
# 1. group labels, FPS orderings (seeded), random baselines   [already committed for seeds 0,1]
python pruning/fps.py DATA.xyz --out-dir results/fps/stratified_s0 --mode stratified --fps-seed 0
# 2. f_uniq(h) sweep + elbows (R2 #1, R1 #3)
python pruning/funiq_sweep.py DATA.xyz --out-dir results/funiq/groups_s0 --scopes groups
python pruning/elbow.py results/funiq/groups_s0/funiq_curves.csv --out results/funiq/groups_s0/elbows_all.csv
# 3. experiment matrix -> training/experiments.csv, jobs.txt, make_training_sets.sh
python training/make_experiment_matrix.py --data DATA.xyz --seeds 0 1 2
```

**Training machine (GPU)** — clone the repo, copy `DATA.xyz`
(`nequip_models/quests_descriptors-Mar2026/DS_2km6t3p8hxx2_0.xyz`, sha256 in
`data/SHA256SUMS`) and the test set `random_C.xyz`, build `env/venv-paper`
(see `env/README.md`), then:

```bash
bash training/make_training_sets.sh            # pruned xyz from committed orderings (CPU, minutes)
# one model:
training/train_one.sh results/pruned/stratified_s0/DS_..._0.050_stratified_s0.xyz 0 results/models/test gpu
# all of them: each line of training/jobs.txt is one job -> Slurm/Flux array
python evaluation/evaluate_models.py --test random_C.xyz --matrix training/experiments.csv \
       --out results/eval/new_models_test_mae.csv
```

`results/eval/*.csv` is small; commit it and pull it back on the selection
machine for figures.
