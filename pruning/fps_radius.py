"""FPS covering-radius decay per scope (Reviewer 2 #3).

The covering radius r(k) after k FPS selections is the largest distance from
any unselected environment to its nearest selected one, i.e. the descriptor-
space resolution bought by the k-th retained environment.  It is produced by
FPS itself, with no kernel, bandwidth, or entropy.  This script tabulates
r(k) at the paper's retention fractions and locates a knee of log r vs
log(retained fraction) by the chord (kneedle) rule, so it can be compared
directly with the QUESTS elbow-derived retention level.

Usage:
    python fps_radius.py FPS_DIR OUT_CSV
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

FRACTIONS = np.concatenate([np.linspace(0.01, 0.1, 19), np.linspace(0.2, 1.0, 9)])


def kneedle(x, y):
    x = (x - x[0]) / (x[-1] - x[0])
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    p0, p1 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    rel = np.stack([x, y], 1) - p0
    return int(np.argmax(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fps_dir")
    ap.add_argument("out_csv")
    args = ap.parse_args()
    meta = json.load(open(Path(args.fps_dir) / "fps_meta.json"))

    rows, knees = [], []
    for scope, v in meta["scopes"].items():
        r = np.load(Path(args.fps_dir) / v["radius_file"])
        n = len(r)
        for f in FRACTIONS:
            k = max(1, int(round(f * n)))
            rows.append({"scope": scope, "fraction": round(float(f), 4), "k": k,
                         "radius": float(r[k - 1])})
        # knee on a dense log-spaced grid of k (exclude the last few points, r -> 0)
        ks = np.unique(np.geomspace(1, max(2, n - 2), 200).astype(int))
        rr = r[ks - 1]
        ok = rr > 0
        i = kneedle(np.log(ks[ok] / n), np.log(rr[ok]))
        knees.append({"scope": scope, "n": n, "knee_fraction": float(ks[ok][i] / n),
                      "knee_radius": float(rr[ok][i]),
                      "r_at_1pct": float(r[max(1, int(round(0.01 * n))) - 1]),
                      "r_at_5pct": float(r[max(1, int(round(0.05 * n))) - 1]),
                      "r_at_20pct": float(r[max(1, int(round(0.20 * n))) - 1])})

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    knee_csv = Path(args.out_csv).with_name(Path(args.out_csv).stem + "_knees.csv")
    with open(knee_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(knees[0]))
        w.writeheader()
        w.writerows(knees)

    print(f"{'scope':28s} {'n':>7} {'r@1%':>8} {'r@5%':>8} {'r@20%':>8} {'knee_frac':>9} {'knee_r':>8}")
    for k in knees:
        print(f"{k['scope']:28s} {k['n']:7d} {k['r_at_1pct']:8.4f} {k['r_at_5pct']:8.4f} "
              f"{k['r_at_20pct']:8.4f} {k['knee_fraction']:9.3f} {k['knee_radius']:8.4f}")


if __name__ == "__main__":
    main()
