"""Random-selection baseline in the same format as fps.py (Reviewer 4 #3).

Writes a uniformly random permutation per scope as ``fps_order_<scope>.npy``
plus ``fps_meta.json`` so that make_masks.py and retention_accounting.py can
consume it unchanged.  Prefixes of a permutation are nested random subsets,
so one file serves every retention fraction.

Usage:
    python random_order.py DATA.xyz --out-dir results/fps/random_stratified_s0 --mode stratified --seed 0
    python random_order.py DATA.xyz --out-dir results/fps/random_global_s0 --mode global --seed 0
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from ase.io import read


def safe_name(tag):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["stratified", "global"], default="stratified")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = read(args.xyz, ":")
    labels = [a.info["config_tag"] for a in frames]
    env_group = np.concatenate([np.full(len(a), g) for a, g in zip(frames, labels)])
    n = len(env_group)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"xyz": args.xyz, "mode": f"random_{args.mode}", "fps_seed": args.seed,
            "n_env": int(n), "scopes": {}}
    scopes = ["global"] if args.mode == "global" else sorted(set(env_group.tolist()))
    for scope in scopes:
        idx = np.arange(n) if scope == "global" else np.flatnonzero(env_group == scope)
        order = rng.permutation(idx)
        name = safe_name(scope)
        np.save(out / f"fps_order_{name}.npy", order)
        meta["scopes"][scope] = {"n": int(len(idx)), "start": int(order[0]),
                                 "file": f"fps_order_{name}.npy"}
    json.dump(meta, open(out / "fps_meta.json", "w"), indent=2)
    print(f"wrote {len(scopes)} random orderings -> {out}")


if __name__ == "__main__":
    main()
