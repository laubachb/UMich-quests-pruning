"""Retention accounting (Reviewer 1 #1): what "x% retained" actually means.

For each retention fraction and each group, report
    n_env        environments (atoms) whose forces enter the loss
    n_frames     distinct simulation frames containing >= 1 selected env
    n_atoms      atoms in those frames (what the model actually sees)
    frac_frames  n_frames / frames in group
    frac_atoms   n_env / atoms in group
computed directly from FPS orderings (nested prefixes), without re-reading
the pruned xyz files.

Usage:
    python retention_accounting.py DATA.xyz FPS_DIR OUT.csv [--fractions ...]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ase.io import read

DEFAULT_FRACTIONS = np.concatenate([np.linspace(0.01, 0.1, 19), np.linspace(0.2, 1.0, 9)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("fps_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--fractions", type=float, nargs="*", default=DEFAULT_FRACTIONS.tolist())
    args = ap.parse_args()

    frames = read(args.xyz, ":")
    natoms = np.array([len(a) for a in frames])
    offsets = np.concatenate([[0], np.cumsum(natoms)])
    env_frame = np.repeat(np.arange(len(frames)), natoms)
    labels = [a.info["config_tag"] for a in frames]
    env_group = np.repeat(np.array(labels), natoms)
    frame_group = np.array(labels)

    meta = json.load(open(Path(args.fps_dir) / "fps_meta.json"))
    orders = {s: np.load(Path(args.fps_dir) / v["file"]) for s, v in meta["scopes"].items()}
    groups = sorted(set(labels))

    rows = []
    for f in args.fractions:
        sel = np.zeros(int(offsets[-1]), dtype=bool)
        for s, order in orders.items():
            k = int(round(f * len(order)))
            sel[order[:k]] = True
        for g in groups + ["Full"]:
            gmask = np.ones_like(sel) if g == "Full" else (env_group == g)
            fmask = np.ones(len(frames), bool) if g == "Full" else (frame_group == g)
            n_env = int((sel & gmask).sum())
            frames_hit = np.unique(env_frame[sel & gmask])
            n_frames = len(frames_hit)
            n_atoms = int(natoms[frames_hit].sum())
            rows.append({
                "mode": meta["mode"], "fps_seed": meta["fps_seed"], "fraction": round(f, 4),
                "group": g, "n_env": n_env, "n_frames": n_frames, "n_atoms": n_atoms,
                "group_frames": int(fmask.sum()), "group_atoms": int(gmask.sum()),
                "frac_frames": n_frames / fmask.sum(), "frac_atoms": n_env / gmask.sum(),
            })

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    full = [r for r in rows if r["group"] == "Full"]
    print(f"{'frac':>6} {'n_env':>7} {'frames':>7} {'atoms_seen':>10} {'frac_frames':>11}")
    for r in full:
        print(f"{r['fraction']:6.3f} {r['n_env']:7d} {r['n_frames']:7d} {r['n_atoms']:10d} {r['frac_frames']:11.3f}")


if __name__ == "__main__":
    main()
