"""Experiment rows for the published-baseline selections (E6, Reviewer 4 #3).

Reads results/baselines/<method>/meta.json from pruning/baselines.py and writes
    training/experiments_baselines.csv   same columns as training/experiments.csv
    training/jobs_baselines.txt          one train_one.sh command per line

E6_baseline: DIRECT / QUESTS-MSC / CUR at each fraction x train seeds, carbon only.

Usage:
    python training/make_baseline_matrix.py [--seeds 0 1 2] [--methods direct msc cur]
"""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=["direct", "msc", "cur"])
    args = ap.parse_args()

    rows = []
    for m in args.methods:
        meta_path = REPO / f"results/baselines/{m}/meta.json"
        if not meta_path.exists():
            print(f"WARNING: {meta_path} missing, skipping {m}")
            continue
        meta = json.load(open(meta_path))
        stem = Path(meta["xyz"]).stem
        for tag in sorted(meta["fractions"]):
            xyz = f"results/pruned/baseline_{m}/{stem}_{tag}_{m}.xyz"
            assert (REPO / xyz).exists(), f"missing {xyz}"
            for s in args.seeds:
                rows.append(("E6_baseline", m, float(tag), s, xyz, f"results/models/E6_baseline/{m}/f{tag}/seed{s}"))

    with open(REPO / "training/experiments_baselines.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["exp", "selection", "fraction", "train_seed", "xyz", "out_dir", "dataset", "species"])
        w.writerows([r + ("carbon", "") for r in rows])
    with open(REPO / "training/jobs_baselines.txt", "w") as fh:
        for exp, sel, f, s, xyz, out in rows:
            fh.write(f"training/train_one.sh {xyz} {s} {out} gpu\n")
    print(f"{len(rows)} models -> training/experiments_baselines.csv, training/jobs_baselines.txt")


if __name__ == "__main__":
    main()
