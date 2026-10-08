"""Convert an external multi-element dataset to the repo's conventions.

Writes <out_dir>/train.xyz and <out_dir>/test.xyz (extxyz; energy in eV,
forces in eV/A, per-frame ``config_tag`` = state-point / structure-type label
supplied by the dataset authors) plus <out_dir>/groups.csv.

Held-out split (Reviewer 1 #2): within each group, frames are kept in file
order and the LAST ``--test-frac`` fraction is held out as a contiguous block.
Where the source orders frames along MD/quench trajectories this gives a
trajectory-disjoint test set rather than a random interleaved one.  Groups with
fewer than ``--min-group`` frames are dropped (too small to stratify) and
non-periodic frames (dimers, isolated atoms, clusters) are dropped because the
QUESTS descriptor needs k=32 neighbours.

Datasets:
  sio2      Erhard et al. 2022 (Zenodo 6353684): dataset.scan.2.xyz.
            Label = sub_simulation_type for crystals (14 polymorphs), else
            config_type (liquid / quench / bulk_amo).
  monbtavw  Byggmastar et al. PRB 2021 via ColabFit parquet (co/, cs/, cs_co_map/).
            Label = ColabFit configuration-set name (BCC_alloys, liquid, ...).

Usage:
    python data/prepare_external.py sio2 /p/lustre1/laubach2/external_data/sio2_erhard2022/dataset.scan.2.xyz data/external/sio2
    python data/prepare_external.py monbtavw /p/lustre1/laubach2/external_data/monbtavw_prb2021 data/external/monbtavw
"""
import argparse
import collections
import csv
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


def load_sio2(path):
    frames = read(path, ":")
    out = []
    for a in frames:
        ct = a.info.get("config_type", "")
        if ct == "cluster" or not all(a.pbc):
            continue
        tag = a.info.get("sub_simulation_type", "") if ct == "bulk_cryst" else ct
        if not tag or tag == "-":
            continue
        b = Atoms(a.symbols, positions=a.get_positions(), cell=a.cell, pbc=True)
        b.info["energy"] = float(a.get_potential_energy())
        b.arrays["forces"] = np.asarray(a.get_forces(), dtype=float)
        b.info["config_tag"] = tag
        b.info["source_config_type"] = ct
        out.append(b)
    return out


def load_monbtavw(root):
    import pandas as pd
    root = Path(root)
    co = pd.read_parquet(root / "co/co_0.parquet")
    cs = pd.read_parquet(root / "cs/cs_0.parquet").set_index("id")
    m = pd.read_parquet(root / "cs_co_map/cs_co_map_0.parquet")
    name_of = {cid: cs.loc[sid, "name"] for sid, cid in zip(m["configuration_set_id"], m["configuration_id"])}
    stack = lambda v: np.stack([np.asarray(x, dtype=float) for x in v])  # nested object arrays
    out = []
    for _, r in co.iterrows():
        tag = name_of.get(r["configuration_id"])
        pbc = np.asarray(r["pbc"]).astype(bool)
        if tag is None or not pbc.all():
            continue
        pos = stack(r["positions"]).reshape(-1, 3)
        cell = stack(r["cell"]).reshape(3, 3)
        b = Atoms(numbers=np.asarray(r["atomic_numbers"]).astype(int), positions=pos, cell=cell, pbc=True)
        b.info["energy"] = float(r["energy"])
        b.arrays["forces"] = stack(r["atomic_forces"]).reshape(-1, 3)
        b.info["config_tag"] = tag
        out.append(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["sio2", "monbtavw"])
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--min-group", type=int, default=20)
    ap.add_argument("--drop", nargs="*", default=[], help="group labels to exclude (e.g. dimer)")
    args = ap.parse_args()

    frames = load_sio2(args.src) if args.dataset == "sio2" else load_monbtavw(args.src)
    by = collections.defaultdict(list)
    for a in frames:
        by[a.info["config_tag"]].append(a)
    train, test, rows = [], [], []
    for tag in sorted(by):
        fr = by[tag]
        if tag in args.drop:
            print(f"  drop {tag:32s} (--drop)")
            continue
        if len(fr) < args.min_group:
            print(f"  drop {tag:32s} ({len(fr)} frames < {args.min_group})")
            continue
        n_test = max(1, int(round(args.test_frac * len(fr))))
        tr, te = fr[:-n_test], fr[-n_test:]
        for a in te:
            a.info["config_tag"] = "test_" + tag
        train += tr
        test += te
        rows.append({"group": tag, "frames_total": len(fr), "frames_train": len(tr), "frames_test": len(te),
                     "atoms_train": sum(len(a) for a in tr), "atoms_test": sum(len(a) for a in te),
                     "elements": "".join(sorted(set(sum((a.get_chemical_symbols() for a in fr), []))))})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "train.xyz", train)
    write(out / "test.xyz", test)
    with open(out / "groups.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{args.dataset}: {len(train)} train frames ({sum(len(a) for a in train)} atoms), "
          f"{len(test)} test frames ({sum(len(a) for a in test)} atoms), {len(rows)} groups")
    for r in rows:
        print(f"  {r['group']:32s} train {r['frames_train']:4d} fr / {r['atoms_train']:6d} at   "
              f"test {r['frames_test']:3d} fr / {r['atoms_test']:5d} at")


if __name__ == "__main__":
    main()
