"""Error vs force magnitude from the saved per-atom predictions (Reviewer 4 #6).

RDFs barely discriminate between pruning strategies; force MAE averaged over
all atoms is dominated by the bulk of near-equilibrium environments.  Here the
saved predictions for every paper model (Full_<rate>_<model>.npy, one row per
test atom, eV/A) are binned by reference force magnitude |F| so that the
high-force tail -- the environments that matter for reactive / far-from-
equilibrium behaviour -- gets its own error curve.

Output: results/stats_test/force_tail.csv with one row per
(method, rate, model, group, bin) and the MAE in that bin; a console summary of
the top-10% |F| bin at a few retention levels.

Usage:
    python evaluation/force_tail_analysis.py --test random_C.xyz \
        --yes-dir .../yes_forces --no-dir .../no_forces --out results/stats_test/force_tail.csv
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

HARTREE_BOHR_TO_EV_A = 27.211386245988 * 1.8897261246257702
BINS = [("all", 0, 1.0), ("bottom50", 0, 0.5), ("mid40", 0.5, 0.9), ("top10", 0.9, 0.99), ("top1", 0.99, 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--yes-dir", required=True, help="stratified-adaptive predictions")
    ap.add_argument("--no-dir", required=True, help="global-fixed predictions")
    ap.add_argument("--out", default="results/stats_test/force_tail.csv")
    args = ap.parse_args()

    frames = read(args.test, ":")
    g = np.concatenate([np.full(len(a), a.info["config_tag"]) for a in frames])
    F = np.concatenate([a.get_forces() for a in frames]) * HARTREE_BOHR_TO_EV_A
    Fmag = np.linalg.norm(F, axis=1)
    groups = sorted(set(g))
    # quantile edges of |F| computed PER GROUP so "top10" means the group's own tail
    edges = {grp: np.quantile(Fmag[g == grp], [0, 0.5, 0.9, 0.99, 1.0]) for grp in groups}
    edges["Full"] = np.quantile(Fmag, [0, 0.5, 0.9, 0.99, 1.0])
    qmap = {0: 0, 0.5: 1, 0.9: 2, 0.99: 3, 1.0: 4}

    rows = []
    for method, d in (("stratified_adaptive", args.yes_dir), ("global_fixed", args.no_dir)):
        files = sorted(Path(d).glob("Full_*.npy"))
        print(f"{method}: {len(files)} models")
        for f in files:
            m = re.match(r"Full_([\d.]+)_(.*)\.npy", f.name)
            rate, model = float(m.group(1)), m.group(2)
            P = np.load(f)
            if P.shape != F.shape:
                print("  shape mismatch, skipping", f.name)
                continue
            err = np.abs(P - F).mean(1)  # per-atom MAE over xyz
            for grp in groups + ["Full"]:
                gm = np.ones(len(g), bool) if grp == "Full" else (g == grp)
                e = edges[grp]
                for name, lo, hi in BINS:
                    bm = gm & (Fmag >= e[qmap[lo]]) & (Fmag <= e[qmap[hi]])
                    rows.append((method, rate, model, grp, name, int(bm.sum()), float(err[bm].mean())))
    df = pd.DataFrame(rows, columns=["method", "rate", "model", "group", "bin", "n_atoms", "mae"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n== median-over-models MAE (eV/A) in the top-10% |F| bin vs all atoms, Full test set")
    s = (df[df.group == "Full"].groupby(["method", "rate", "bin"]).mae.median().unstack("bin"))
    for r in (0.05, 0.1, 0.2, 1.0):
        for meth in ("global_fixed", "stratified_adaptive"):
            try:
                x = s.loc[(meth, r)]
                print(f"  rate={r:<5} {meth:20s} all={x['all']:.3f}  bottom50={x['bottom50']:.3f}  "
                      f"top10={x['top10']:.3f}  top1={x['top1']:.3f}")
            except KeyError:
                pass
    print("\n== relative improvement (stratified vs global), per group, top-10% bin, rate 0.05")
    t = df[(df.rate == 0.05) & (df.bin == "top10")].groupby(["group", "method"]).mae.median().unstack("method")
    t["improvement"] = 1 - t["stratified_adaptive"] / t["global_fixed"]
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
