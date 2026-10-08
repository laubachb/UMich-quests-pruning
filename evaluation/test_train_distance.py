"""How close is the test set to the training set in descriptor space? (Reviewer 1 #2)

The held-out test set was sampled from the same DFT-MD trajectories as the
training data.  This script quantifies the resulting proximity: for every test
environment, the Euclidean distance (QUESTS descriptor space, same k/cutoff as
training) to its nearest training environment, compared with the leave-one-out
nearest-neighbour distance among training environments themselves, per group.
If test->train distances are much smaller than train->train distances the test
set is "inside" the training cloud; if comparable, it is no closer than a
fresh sample of the same trajectory would be.

Also reports, per test group, the fraction of test environments whose nearest
training environment lies in the *same* group (label consistency) and the
distances to the 5 %-retention stratified subset (what a pruned model saw).

Outputs (in --out-dir): test_descriptors.npy (cached), nn_distances.csv (per
test env: group, d_train, nn_group, d_train5pct), summary.csv.

Usage:
    python evaluation/test_train_distance.py --train DATA.xyz --test random_C.xyz \
        --fps-dir results/fps/stratified_s0 --out-dir results/test_distance
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read
from quests.descriptor import get_descriptors
from quests.matrix import cdist


def nn_dist(X, Y, batch=4000, exclude_self=False):
    """min_j ||x_i - y_j|| and argmin for each row of X (Y may equal X)."""
    d = np.empty(len(X))
    j = np.empty(len(X), dtype=np.int64)
    for i in range(0, len(X), batch):
        D = cdist(X[i:i + batch], Y)
        if exclude_self:
            rows = np.arange(D.shape[0])
            D[rows, i + rows] = np.inf
        j[i:i + batch] = D.argmin(1)
        d[i:i + batch] = D[np.arange(D.shape[0]), j[i:i + batch]]
    return d, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--fps-dir", default=None, help="stratified FPS dir for the 5%% subset comparison")
    ap.add_argument("--out-dir", default="results/test_distance")
    ap.add_argument("--desc-key", default="quests_descriptor_descriptors")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--cutoff", type=float, default=5.0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train = read(args.train, ":")
    Xtr = np.concatenate([a.arrays[args.desc_key] for a in train]).astype(np.float64)
    gtr = np.concatenate([np.full(len(a), a.info["config_tag"]) for a in train])
    print(f"train: {len(Xtr)} envs ({time.time() - t0:.0f}s)")

    cache = out / "test_descriptors.npy"
    test = read(args.test, ":")
    gte = np.concatenate([np.full(len(a), a.info["config_tag"]) for a in test])
    if cache.exists():
        Xte = np.load(cache)
    else:
        t1 = time.time()
        Xte = get_descriptors(test, k=args.k, cutoff=args.cutoff).astype(np.float64)
        np.save(cache, Xte)
        print(f"test descriptors: {Xte.shape} ({time.time() - t1:.0f}s)")
    assert Xte.shape[1] == Xtr.shape[1], "descriptor dimension mismatch (k/cutoff?)"

    t1 = time.time()
    d_te, j_te = nn_dist(Xte, Xtr)
    print(f"test->train NN ({time.time() - t1:.0f}s)")
    t1 = time.time()
    d_tr, _ = nn_dist(Xtr, Xtr, exclude_self=True)
    print(f"train->train LOO NN ({time.time() - t1:.0f}s)")

    d_te5 = None
    if args.fps_dir:
        meta = json.load(open(Path(args.fps_dir) / "fps_meta.json"))
        sel = np.concatenate([np.load(Path(args.fps_dir) / v["file"])[:int(round(0.05 * v["n"]))]
                              for v in meta["scopes"].values()])
        d_te5, _ = nn_dist(Xte, Xtr[sel])
        print(f"test->train(5% stratified) NN")

    with open(out / "nn_distances.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test_group", "d_train", "nn_train_group", "d_train_5pct"])
        for i in range(len(Xte)):
            w.writerow([gte[i], f"{d_te[i]:.5f}", gtr[j_te[i]], f"{d_te5[i]:.5f}" if d_te5 is not None else ""])

    def canon(t):
        t = t.replace("test_", "").replace("HD_liquid", "HD liquid").replace("LD_liquid", "LD liquid")
        return "LD liquid_1.0gcc_2000" if t == "LD liquid_0.5gcc_2000" else t

    rows = []
    print(f"\n{'test group':28s} {'n':>6} {'test->train':>22} {'train->train LOO':>22} {'ratio':>6} {'same-grp':>8} {'->5% subset':>12}")
    for g in sorted(set(gte)):
        m = gte == g
        mt = gtr == canon(g)
        q = lambda x: np.percentile(x, [50, 90])
        a, b = q(d_te[m]), q(d_tr[mt]) if mt.any() else (np.nan, np.nan)
        same = float(np.mean(np.array([canon(x) for x in gtr[j_te[m]]]) == canon(g)))
        r = {"test_group": g, "n": int(m.sum()), "d_test_train_p50": a[0], "d_test_train_p90": a[1],
             "d_train_train_p50": b[0], "d_train_train_p90": b[1], "ratio_p50": a[0] / b[0],
             "frac_nn_same_group": same,
             "d_test_train5pct_p50": float(np.median(d_te5[m])) if d_te5 is not None else np.nan}
        rows.append(r)
        print(f"{g:28s} {r['n']:6d} {a[0]:10.4f} {a[1]:10.4f} {b[0]:10.4f} {b[1]:10.4f} {r['ratio_p50']:6.2f} "
              f"{same:8.2f} {r['d_test_train5pct_p50']:12.4f}")
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"done ({time.time() - t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
