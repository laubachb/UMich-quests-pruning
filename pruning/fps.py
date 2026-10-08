"""Environment-level farthest-point sampling on QUESTS descriptors.

Replaces Orchestrator's ``simple_prune_dataset(..., 'percentage_fps')`` with a
self-contained, seeded implementation.  Bandwidth never enters the selection:
FPS uses Euclidean distance in descriptor space only (this matches the
original workflow; see docs/reviews.txt, Reviewer 2 #2).

Rather than re-running FPS for every retention fraction, we compute one full
greedy FPS *ordering* per scope (one per group for stratified pruning, or one
for the whole dataset for global pruning).  Retaining a fraction ``f`` then
means taking the first ``round(f * n)`` environments of the ordering, so a
single run serves the whole retention sweep and nested subsets are guaranteed.

Outputs (in --out-dir):
    fps_order_<scope>.npy   int64 array of global environment indices, in
                            selection order (scope = 'global' or group name)
    fps_meta.json           seeds, start indices, group sizes, descriptor key

Usage:
    python fps.py DATA.xyz --out-dir results/fps_s0 --mode stratified --fps-seed 0
    python fps.py DATA.xyz --out-dir results/fps_global_s0 --mode global --fps-seed 0
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from ase.io import read

DESC_KEY = "quests_descriptor_descriptors"


def safe_name(tag):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def fps_order(X, start, dtype=np.float32):
    """Greedy FPS ordering of all rows of X starting from row ``start``.

    Incremental min-distance update: O(n * d) per step, O(n^2 d) total.
    Returns an int64 array of row indices in selection order.
    """
    X = np.ascontiguousarray(X, dtype=dtype)
    n = len(X)
    sq = np.einsum("ij,ij->i", X, X)
    order = np.empty(n, dtype=np.int64)
    # covering radius after k+1 selections: max over unselected points of the
    # distance to the nearest selected point (the FPS-native resolution scale,
    # cf. docs/reviews.txt Reviewer 2 #3)
    radius = np.zeros(n, dtype=np.float64)
    min_d2 = np.full(n, np.inf, dtype=dtype)
    cur = int(start)
    for k in range(n):
        order[k] = cur
        # squared distance from the newly selected point to all points
        d2 = sq + sq[cur] - 2.0 * (X @ X[cur])
        np.minimum(min_d2, d2, out=min_d2)
        min_d2[cur] = -np.inf  # never reselect
        cur = int(np.argmax(min_d2))
        radius[k] = np.sqrt(max(float(min_d2[cur]), 0.0)) if k < n - 1 else 0.0
    return order, radius


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["stratified", "global"], default="stratified")
    ap.add_argument("--fps-seed", type=int, default=0,
                    help="seed for the random starting environment")
    ap.add_argument("--desc-key", default=DESC_KEY)
    ap.add_argument("--labels-csv", default=None,
                    help="optional data/group_labels.csv to override config_tag "
                         "(expects one row per frame, column 'group')")
    args = ap.parse_args()

    t0 = time.time()
    frames = read(args.xyz, ":")
    X = np.concatenate([a.arrays[args.desc_key] for a in frames])
    if args.labels_csv:
        import csv
        labels = [r["group"] for r in csv.DictReader(open(args.labels_csv))]
        assert len(labels) == len(frames), "labels_csv rows != frames"
    else:
        labels = [a.info["config_tag"] for a in frames]
    env_group = np.concatenate(
        [np.full(len(a), g) for a, g in zip(frames, labels)])
    n = len(X)
    print(f"loaded {len(frames)} frames, {n} environments, d={X.shape[1]} "
          f"({time.time() - t0:.1f}s)")

    rng = np.random.default_rng(args.fps_seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"xyz": args.xyz, "mode": args.mode, "fps_seed": args.fps_seed,
            "desc_key": args.desc_key, "n_env": int(n), "scopes": {}}

    scopes = (["global"] if args.mode == "global"
              else sorted(set(env_group.tolist())))
    for scope in scopes:
        idx = (np.arange(n) if scope == "global"
               else np.flatnonzero(env_group == scope))
        start_local = int(rng.integers(len(idx)))
        t1 = time.time()
        order_local, radius = fps_order(X[idx], start_local)
        order = idx[order_local]
        name = safe_name(scope)
        np.save(out / f"fps_order_{name}.npy", order)
        np.save(out / f"fps_radius_{name}.npy", radius)
        meta["scopes"][scope] = {"n": int(len(idx)), "start": int(idx[start_local]),
                                 "file": f"fps_order_{name}.npy",
                                 "radius_file": f"fps_radius_{name}.npy"}
        print(f"  {scope:28s} n={len(idx):7d} start={idx[start_local]:7d} "
              f"({time.time() - t1:.1f}s)")

    json.dump(meta, open(out / "fps_meta.json", "w"), indent=2)
    print(f"done in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
