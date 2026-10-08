# Environments

Two virtualenvs live here (both git-ignored); lock files are committed.

| venv | Python | Stack | Purpose |
|---|---|---|---|
| `venv-paper/` | 3.12.4 | nequip 0.9.1, e3nn 0.5.6, torch 2.5.0, allegro 0.5.1 | **Faithful reproduction.** These are the versions recorded in the paper's Hydra `hparams.yaml` (`info_dict.versions`, runs of 2025-08-29 → 2025-10-21). `training/config.yaml` is written for this API. |
| `venv/` | 3.13.2 | nequip 0.19.1, current torch | Modern stack for new experiments. Needs the config patch in `training/configs/timing_cpu.yaml` (`ChemicalSpeciesToAtomTypeMapper` now takes `model_type_names` + `chemical_species_to_atom_type_map`). |

Rebuild:

```bash
# paper-faithful
uv venv --python /usr/tce/packages/python/python-3.12.4/bin/python3 env/venv-paper
uv pip install --python env/venv-paper/bin/python -r env/requirements-lock-paper.txt
env/venv-paper/bin/pip install -e training/        # ltau_nequip_modules (masked loss)

# modern
python3 -m venv env/venv && env/venv/bin/pip install -r env/requirements-lock.txt
env/venv/bin/pip install -e training/
```

Other tools used by the paper: QUESTS (`quests 2026.2.22` in user site-packages
on dane), LAMMPS with the NequIP pair style (built in
`nequip_models/becky_files/for_ben-pace_input/try_pace/lammps`), TRAVIS.
The original training jobs were submitted through LLNL Orchestrator
(`workflow_lsf` → an LSF/GPU machine), which is not yet public.

## Compute

`dane` is CPU-only (112 cores/node). Measured per batch of 2 frames on dane5:

| stack | threads | s/batch | h/epoch (342 batches) |
|---|---|---|---|
| nequip 0.19.1 (`venv`) | 32 | 25–29 | ~2.5 |
| nequip 0.9.1 (`venv-paper`, `training/configs/paper_smoke_cpu.yaml`) | 16 | ~5 | ~0.5 |

The paper config loads and trains unmodified in `venv-paper`. The paper's models train for up to 1000 epochs with
early stopping (patience 20); the full sweep is 28 retention levels × 2 methods
× ~10 seeds ≈ 560 models. **Training must run on a GPU machine.** CPU nodes are
fine for everything else: QUESTS descriptors/entropies, FPS, elbow selection,
force-MAE evaluation of compiled models, and LAMMPS/TRAVIS RDF runs.
