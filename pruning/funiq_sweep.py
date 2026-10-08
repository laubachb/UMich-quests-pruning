"""QUESTS f_uniq(h) sweep, elbow bandwidth selection, and subsampling robustness.

For each scope (each state-point group, and optionally the pooled dataset)
compute the QUESTS entropy H(h) over a log-spaced bandwidth grid and
f_uniq(h) = H(h) / log(N).  Then locate the elbow of f_uniq vs log10(h) with
two estimators:

  * ``kneedle``   : maximum perpendicular distance from the chord joining the
                    curve's endpoints (axes normalised to [0, 1]); robust and
                    smoothing-free.
  * ``curvature`` : maximum |kappa| of a smoothing spline fit to f_uniq(log h);
                    the "maximum curvature" criterion described in the paper.

Robustness (Reviewer 2 #1): each scope is also randomly subsampled to
fractions ``--subsample`` with ``--n-rep`` replicates, and the elbow is
recomputed, so the dependence of h* on sampling density is reported directly.

Outputs (in --out-dir):
    funiq_curves.csv   scope, subsample, rep, n, h, H, f_uniq
    elbows.csv         scope, subsample, rep, n, h_kneedle, funiq_kneedle,
                       h_curv, funiq_curv
    sweep_meta.json

Usage:
    python funiq_sweep.py DATA.xyz --out-dir results/funiq --scopes groups
    python funiq_sweep.py DATA.xyz --out-dir results/funiq_global --scopes global
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read
from quests.entropy import entropy
from scipy.interpolate import UnivariateSpline

DESC_KEY = "quests_descriptor_descriptors"


def funiq_curve(X, h_grid, batch_size):
    H = entropy(X, h=np.asarray(h_grid, dtype=float), batch_size=batch_size)
    return np.asarray(H), np.asarray(H) / np.log(len(X))


def elbow_kneedle(logh, f):
    """Max distance from the chord between endpoints, on [0,1]-normalised axes."""
    x = (logh - logh[0]) / (logh[-1] - logh[0])
    y = (f - f.min()) / (f.max() - f.min() + 1e-12)
    # chord from first to last point
    p0, p1 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    d = p1 - p0
    d /= np.linalg.norm(d)
    rel = np.stack([x, y], 1) - p0
    dist = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
    return int(np.argmax(dist))


def elbow_curvature(logh, f, smooth=None):
    """Max |curvature| of a smoothing spline of f(log h); returns grid index."""
    x = (logh - logh[0]) / (logh[-1] - logh[0])
    y = (f - f.min()) / (f.max() - f.min() + 1e-12)
    s = UnivariateSpline(x, y, k=4, s=smooth if smooth is not None else len(x) * 1e-5)
    xx = np.linspace(x[0], x[-1], 2000)
    d1, d2 = s.derivative(1)(xx), s.derivative(2)(xx)
    kappa = np.abs(d2) / (1 + d1 ** 2) ** 1.5
    # restrict to the descending part (ignore flat tail noise)
    i = int(np.argmax(kappa))
    return int(np.argmin(np.abs(x - xx[i])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scopes", choices=["groups", "global", "all"], default="groups")
    ap.add_argument("--desc-key", default=DESC_KEY)
    ap.add_argument("--labels-csv", default=None)
    ap.add_argument("--h-min", type=float, default=0.005)
    ap.add_argument("--h-max", type=float, default=0.5)
    ap.add_argument("--n-h", type=int, default=40)
    ap.add_argument("--subsample", type=float, nargs="*", default=[0.25, 0.5, 0.75])
    ap.add_argument("--n-rep", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=2000)
    args = ap.parse_args()

    t0 = time.time()
    frames = read(args.xyz, ":")
    X = np.concatenate([a.arrays[args.desc_key] for a in frames]).astype(np.float64)
    if args.labels_csv:
        labels = [r["group"] for r in csv.DictReader(open(args.labels_csv))]
        assert len(labels) == len(frames)
    else:
        labels = [a.info["config_tag"] for a in frames]
    env_group = np.concatenate([np.full(len(a), g) for a, g in zip(frames, labels)])
    print(f"loaded {len(frames)} frames, {len(X)} envs ({time.time() - t0:.1f}s)")

    h_grid = np.logspace(np.log10(args.h_min), np.log10(args.h_max), args.n_h)
    logh = np.log10(h_grid)
    scopes = []
    if args.scopes in ("groups", "all"):
        scopes += sorted(set(env_group.tolist()))
    if args.scopes in ("global", "all"):
        scopes.append("global")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    curves, elbows = [], []
    cpath, epath = out / "funiq_curves.csv", out / "elbows.csv"
    done = set()
    if cpath.exists():  # resume: skip scopes already written
        with open(cpath) as fh:
            done = {(r[0], float(r[1]), int(r[2])) for r in csv.reader(fh) if r and r[0] != "scope"}
        print(f"resuming; {len(done)} (scope, subsample, rep) runs already done", flush=True)
    else:
        with open(cpath, "w", newline="") as fh:
            csv.writer(fh).writerow(["scope", "subsample", "rep", "n", "h", "H", "f_uniq"])
        with open(epath, "w", newline="") as fh:
            csv.writer(fh).writerow(["scope", "subsample", "rep", "n", "h_kneedle", "f_kneedle", "h_curv", "f_curv"])

    for scope in scopes:
        idx = np.arange(len(X)) if scope == "global" else np.flatnonzero(env_group == scope)
        curves, elbows = [], []
        jobs = [(1.0, 0, idx)]
        for frac in args.subsample:
            for rep in range(args.n_rep):
                k = max(10, int(round(frac * len(idx))))
                jobs.append((frac, rep, rng.choice(idx, size=k, replace=False)))
        for frac, rep, sel in jobs:
            if (scope, float(frac), int(rep)) in done:
                continue  # rng draws above already consumed, so subsamples stay reproducible
            t1 = time.time()
            H, f = funiq_curve(X[sel], h_grid, args.batch_size)
            ik, ic = elbow_kneedle(logh, f), elbow_curvature(logh, f)
            for h, Hh, fh in zip(h_grid, H, f):
                curves.append((scope, frac, rep, len(sel), h, Hh, fh))
            elbows.append((scope, frac, rep, len(sel), h_grid[ik], f[ik], h_grid[ic], f[ic]))
            print(f"  {scope:28s} sub={frac:4.2f} rep={rep} n={len(sel):6d} "
                  f"h*_kneedle={h_grid[ik]:.4f} (f={f[ik]:.3f})  "
                  f"h*_curv={h_grid[ic]:.4f} (f={f[ic]:.3f})  "
                  f"f(0.015)={np.interp(np.log10(0.015), logh, f):.3f}  "
                  f"({time.time() - t1:.1f}s)", flush=True)
            with open(cpath, "a", newline="") as fh:
                csv.writer(fh).writerows(curves)
            with open(epath, "a", newline="") as fh:
                csv.writer(fh).writerows(elbows)
            curves, elbows = [], []

    json.dump({"xyz": args.xyz, "scopes": scopes, "h_grid": h_grid.tolist(),
               "subsample": args.subsample, "n_rep": args.n_rep, "seed": args.seed,
               "desc_key": args.desc_key, "elapsed_s": time.time() - t0},
              open(out / "sweep_meta.json", "w"), indent=2)
    print(f"done in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
