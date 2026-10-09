"""Turn FPS orderings into per-atom selection masks and write pruned xyz files.

For each retention fraction f, environments in the first round(f * n_scope)
positions of each scope's FPS ordering get weight 1, all others 0.  The full
frames are written (matching the paper: training sees every frame, and the
``weights`` array masks the force loss via ltau_nequip_modules.WeightedLoss).

Usage:
    python make_masks.py DATA.xyz FPS_DIR OUT_DIR --fractions 0.01 0.05 0.1 ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("fps_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--fractions", type=float, nargs="*", default=[])
    ap.add_argument("--budget", default=None,
                    help="JSON {scope: fraction} giving a DIFFERENT retention per scope "
                         "(e.g. elbow-derived, from pruning/elbow.py); writes one file "
                         "tagged 'adaptive' instead of a sweep")
    ap.add_argument("--drop-descriptors", action="store_true",
                    help="omit the descriptor array from the output (smaller files)")
    args = ap.parse_args()
    if not args.fractions and not args.budget:
        ap.error("give --fractions and/or --budget")

    meta = json.load(open(Path(args.fps_dir) / "fps_meta.json"))
    frames = read(args.xyz, ":")
    natoms = np.array([len(a) for a in frames])
    offsets = np.concatenate([[0], np.cumsum(natoms)])
    n = int(offsets[-1])
    assert n == meta["n_env"], "xyz does not match FPS run"

    orders = {s: np.load(Path(args.fps_dir) / v["file"]) for s, v in meta["scopes"].items()}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    jobs = [(f, {s: f for s in orders}, f"{f:.3f}") for f in args.fractions]
    if args.budget:
        budget = {k: v for k, v in json.load(open(args.budget)).items() if not k.startswith("_")}
        missing = set(orders) - set(budget)
        assert not missing, f"budget.json lacks scopes: {sorted(missing)}"
        tot = sum(budget[s] * len(orders[s]) for s in orders) / n
        jobs.append((tot, budget, "adaptive"))
    for f, per_scope, tag in jobs:
        mask = np.zeros(n, dtype=np.int8)
        for s, order in orders.items():
            k = int(round(per_scope[s] * len(order)))
            mask[order[:k]] = 1
        for i, a in enumerate(frames):
            a.arrays["weights"] = mask[offsets[i]:offsets[i + 1]].astype(float)
            if args.drop_descriptors:
                a.arrays.pop(meta.get("desc_key", "quests_descriptor_descriptors"), None)  # random_order.py metas have no desc_key
        path = out / f"{Path(args.xyz).stem}_{tag}_{meta['mode']}_s{meta['fps_seed']}.xyz"
        write(path, frames)
        frames_touched = int(sum(mask[offsets[i]:offsets[i + 1]].any() for i in range(len(frames))))
        summary.append({"fraction": f, "n_selected": int(mask.sum()),
                        "frames_with_selection": frames_touched, "file": path.name})
        print(f"f={tag}: {mask.sum():7d}/{n} envs, {frames_touched}/{len(frames)} frames -> {path.name}")
    json.dump(summary, open(out / "mask_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
