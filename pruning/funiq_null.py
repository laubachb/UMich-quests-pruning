"""Null test for the elbow (Reviewer 2 #1): does f_uniq(h) show the same elbow
for structureless data of the same size, dimension and scale?

For every group, two surrogates of the real descriptor matrix X (n x d):
  gauss     multivariate Gaussian with X's mean and covariance (keeps 2nd moments,
            destroys clusters / manifolds)
  shuffle   each column permuted independently (keeps marginals, destroys correlations)
f_uniq(h) is computed on the same h grid as the real sweep and written in the
same format, so elbow_logh.py applies unchanged.  Compare elbow h* and f_uniq
at the elbow with the real group: if the real data elbows at a much smaller h
(or a much lower f_uniq) than its surrogate, the elbow reflects structure.

Usage: python pruning/funiq_null.py DATA.xyz --out-dir results/funiq/null_s0 [--batch-size 2000]
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from ase.io import read
from quests.entropy import entropy

ap = argparse.ArgumentParser()
ap.add_argument("xyz"); ap.add_argument("--out-dir", required=True)
ap.add_argument("--desc-key", default="quests_descriptor_descriptors")
ap.add_argument("--h-min", type=float, default=0.005); ap.add_argument("--h-max", type=float, default=0.5)
ap.add_argument("--n-h", type=int, default=40); ap.add_argument("--batch-size", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-n", type=int, default=15000,
                help="subsample larger groups to this size (nulls are O(n^2); elbow is subsample-stable)")
args = ap.parse_args()
out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
h_grid = np.logspace(np.log10(args.h_min), np.log10(args.h_max), args.n_h)
frames = read(args.xyz, ":")
X = np.concatenate([a.arrays[args.desc_key] for a in frames]).astype(np.float64)
grp = np.concatenate([np.full(len(a), a.info["config_tag"]) for a in frames])
rng = np.random.default_rng(args.seed)
with open(out / "funiq_curves.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["scope", "subsample", "rep", "n", "h", "H", "f_uniq"])
    for g in sorted(set(grp)):
        idx = np.flatnonzero(grp == g)
        if len(idx) > args.max_n:
            idx = rng.choice(idx, args.max_n, replace=False)
        Xg = X[idx]
        sur = {"real": Xg,
               "gauss": rng.multivariate_normal(Xg.mean(0), np.cov(Xg, rowvar=False), size=len(Xg)),
               "shuffle": np.stack([rng.permutation(Xg[:, j]) for j in range(Xg.shape[1])], 1)}
        for kind, Z in sur.items():
            t = time.time(); H = np.asarray(entropy(Z, h=h_grid, batch_size=args.batch_size)); f = H / np.log(len(Z))
            w.writerows([(f"{g}|{kind}", 1.0, 0, len(Z), h, Hh, fh_) for h, Hh, fh_ in zip(h_grid, H, f)]); fh.flush()
            print(f"  {g:28s} {kind:8s} n={len(Z):6d} f(0.015)={np.interp(np.log10(0.015), np.log10(h_grid), f):.3f} ({time.time()-t:.0f}s)", flush=True)
json.dump({"xyz": args.xyz, "h_grid": h_grid.tolist(), "max_n": args.max_n, "seed": args.seed}, open(out / "meta.json", "w"), indent=1)
print("done")
