"""Published subsampling baselines at matched force-label budgets (Reviewer 4 #3).

Each method selects atomic environments whose forces enter the (masked) loss,
and writes pruned xyz files in exactly the format of make_masks.py: every frame
is kept, the per-atom ``weights`` array is the 0/1 selection mask.  Budgets are
matched to the environment-level retention used elsewhere: a fraction f means
round(f * n_env) force labels.

Methods
  direct  DIRECT (Qi et al., npj Comput. Mater. 10, 43, 2024; manuscript Ref. 35),
          applied to environments, label-free and global: StandardScaler -> PCA,
          keep PCs with eigenvalue > 1 (Kaiser), weight by explained-variance ratio
          -> BIRCH into k clusters (threshold tuned down until k clusters exist)
          -> the environment nearest each cluster centre.  Re-implemented line by
          line from maml.sampling (commit MAML_COMMIT) without maml's structure
          encoders; the cluster "centre" follows maml (dict over subcluster labels,
          i.e. the last subcluster carrying that label).  Not nested: one run per f.
  msc     QUESTS dataset compression, "msc" (Yu, Lordi, Schwalbe-Koda, J. Chem.
          Phys. 164, 2026; Ref. 38), upstream code at QUESTS_COMMIT, default
          bandwidth 0.015.  Selects whole frames; frames are taken in MSC order
          until the cumulative environment count is closest to the budget.
  cur     Greedy CUR sample selection (Imbalzano et al. 2018, Ref. 40; skmatter,
          Cersonsky et al. 2021, Ref. 42) on per-frame fingerprints: mean and
          standard deviation of the frame's QUESTS environment descriptors
          (126 dims), standardised.  Selects whole frames as in the references;
          per-environment CUR on the 63-dim descriptor is degenerate (the greedy
          deflation is exhausted after <= 63 picks and skmatter fills at random).
          Frames are taken in CUR order until the budget is met, as for msc.

Outputs (per method, in --out-dir):
    selected_f<f>.npy     sorted global environment indices
    accounting.csv        per fraction x group: n_env, n_frames, share of budget
    meta.json             parameters, upstream commits, counts, timings
and pruned xyz in --pruned-dir:  <stem>_<f>_<method>.xyz

Usage:
    python pruning/baselines.py data/DS_2km6t3p8hxx2_0.xyz --method direct --fractions 0.05 0.10
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

DESC_KEY = "quests_descriptor_descriptors"
QUESTS_COMMIT = "29e4a405e9266a4be071d9b1c0790844286fbe8f"
MAML_COMMIT = "8916c6ea05ff19a1b351bd2b8f11e210750ba032"


# ---------------------------------------------------------------- DIRECT
def direct_select(X, k, threshold_init=0.5, log=print):
    from sklearn.cluster import Birch
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(X)
    pca = PCA().fit(Z)
    m = int((pca.explained_variance_ > 1).sum())
    ratio = pca.explained_variance_ratio_
    F = pca.transform(Z)[:, :m] * ratio[:m]
    log(f"  DIRECT: {m} PCs (Kaiser), {100 * ratio[:m].sum():.1f}% variance")
    thr = threshold_init
    model = Birch(n_clusters=k, threshold=thr).fit(F)
    while len(set(model.subcluster_labels_)) < k:  # maml BirchClustering.transform
        thr = thr / k * len(set(model.subcluster_labels_))
        log(f"  DIRECT: {len(set(model.subcluster_labels_))} clusters < {k}; threshold -> {thr:.3g}")
        model = Birch(n_clusters=k, threshold=thr).fit(F)
    labels = model.predict(F)
    centers = dict(zip(model.subcluster_labels_, model.subcluster_centers_))
    sel = []
    for lab in np.unique(labels):  # maml SelectKFromClusters, k=1, "center"
        idx = np.flatnonzero(labels == lab)
        d = np.linalg.norm(F[idx] - centers[lab], axis=1)
        sel.append(idx[np.argmin(d)])
    info = {"n_pcs": m, "pc_variance": float(ratio[:m].sum()), "birch_threshold": thr,
            "n_subclusters": int(len(model.subcluster_centers_)), "n_clusters_nonempty": int(len(np.unique(labels)))}
    return np.array(sorted(set(sel)), dtype=np.int64), info


# ---------------------------------------------------------------- frame-level helpers
def prefix_to_budget(order, natoms, budget):
    """Smallest-error prefix of a frame ordering w.r.t. an environment budget."""
    cum = np.cumsum(natoms[order])
    L = int(np.argmin(np.abs(cum - budget))) + 1
    return order[:L], int(cum[L - 1])


def frames_to_envs(frames_sel, offsets):
    return np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in sorted(frames_sel)]).astype(np.int64)


def msc_order(descs, n_frames_needed, h, batch_size, log=print):
    from quests.compression.fps import msc
    from quests.entropy import entropy
    t = time.time()
    ent = np.array([entropy(x, h=h, batch_size=batch_size) for x in descs])
    log(f"  MSC: per-frame entropies ({time.time() - t:.0f}s)")
    t = time.time()
    order = np.array(msc(descs, ent, n_frames_needed, h=h, batch_size=batch_size), dtype=np.int64)
    log(f"  MSC: ordered {len(order)} frames ({time.time() - t:.0f}s)")
    return order


def cur_order(descs, n_frames_needed, log=print):
    from skmatter.sample_selection import CUR
    from sklearn.preprocessing import StandardScaler
    fp = np.array([np.concatenate([x.mean(0), x.std(0)]) for x in descs])
    fp = StandardScaler().fit_transform(fp)
    rank = int(np.linalg.matrix_rank(fp))
    sel = CUR(n_to_select=n_frames_needed, full=True, random_state=0).fit(fp)
    order = np.asarray(sel.selected_idx_, dtype=np.int64)
    log(f"  CUR: fingerprint {fp.shape}, rank {rank}; ordered {len(order)} frames "
        f"(first {min(rank, len(order))} greedy, any beyond the rank are skmatter's random fill)")
    return order, rank


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--method", required=True, choices=["direct", "msc", "cur"])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--out-dir", default=None, help="default results/baselines/<method>")
    ap.add_argument("--pruned-dir", default=None, help="default results/pruned/baseline_<method>")
    ap.add_argument("--h", type=float, default=0.015, help="QUESTS bandwidth for msc")
    ap.add_argument("--batch-size", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out_dir or f"results/baselines/{args.method}")
    pdir = Path(args.pruned_dir or f"results/pruned/baseline_{args.method}")
    out.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    frames = read(args.xyz, ":")
    natoms = np.array([len(a) for a in frames])
    offsets = np.concatenate([[0], np.cumsum(natoms)])
    n_env = int(offsets[-1])
    X = np.concatenate([a.arrays[DESC_KEY] for a in frames]).astype(np.float64)
    descs = [X[offsets[i]:offsets[i + 1]] for i in range(len(frames))]
    labels = np.array([a.info["config_tag"] for a in frames])
    env_group = np.repeat(labels, natoms)
    env_frame = np.repeat(np.arange(len(frames)), natoms)
    print(f"{len(frames)} frames, {n_env} environments, d={X.shape[1]} ({time.time() - t0:.0f}s)", flush=True)

    budgets = {f: int(round(f * n_env)) for f in args.fractions}
    meta = {"method": args.method, "xyz": args.xyz, "n_env": n_env, "n_frames": len(frames),
            "quests_commit": QUESTS_COMMIT, "maml_commit": MAML_COMMIT, "fractions": {}}
    selections = {}
    if args.method == "direct":
        for f, k in budgets.items():
            t = time.time()
            sel, info = direct_select(X, k, log=lambda s: print(s, flush=True))
            selections[f] = sel
            meta["fractions"][f"{f:.3f}"] = {"budget": k, **info, "seconds": round(time.time() - t)}
    else:
        need = min(len(frames), int(np.ceil(max(budgets.values()) / natoms.min())) + 1)
        t = time.time()
        if args.method == "msc":
            order = msc_order(descs, need, args.h, args.batch_size, log=lambda s: print(s, flush=True))
            meta["h"] = args.h
        else:
            order, rank = cur_order(descs, need, log=lambda s: print(s, flush=True))
            meta["fingerprint"] = "mean+std of frame QUESTS descriptors, standardised"
            meta["fingerprint_rank"] = rank
        meta["seconds_ordering"] = round(time.time() - t)
        np.save(out / "frame_order.npy", order)
        for f, k in budgets.items():
            fr, n = prefix_to_budget(order, natoms, k)
            if np.cumsum(natoms[order])[-1] < k:
                raise RuntimeError(f"ordering too short for budget {k}")
            selections[f] = frames_to_envs(fr, offsets)
            meta["fractions"][f"{f:.3f}"] = {"budget": k, "frames_selected": int(len(fr))}
            if args.method == "cur":
                meta["fractions"][f"{f:.3f}"]["frames_beyond_rank"] = int(max(0, len(fr) - meta["fingerprint_rank"]))

    rows = []
    groups = sorted(set(labels))
    for f, sel in selections.items():
        tag = f"{f:.3f}"
        np.save(out / f"selected_f{tag}.npy", sel)
        mask = np.zeros(n_env, dtype=np.int8)
        mask[sel] = 1
        meta["fractions"][tag].update({"n_selected": int(mask.sum()), "actual_fraction": float(mask.mean()),
                                       "frames_with_selection": int(len(np.unique(env_frame[sel])))})
        for g in groups:
            gm = env_group == g
            rows.append({"method": args.method, "fraction": tag, "group": g, "n_env": int(mask[gm].sum()),
                         "group_env": int(gm.sum()), "frac_of_group": float(mask[gm].mean()),
                         "share_of_budget": float(mask[gm].sum() / mask.sum()),
                         "n_frames": int(len(np.unique(env_frame[sel][env_group[sel] == g])))})
        for i, a in enumerate(frames):
            a.arrays["weights"] = mask[offsets[i]:offsets[i + 1]].astype(float)
            a.arrays.pop(DESC_KEY, None)
        path = pdir / f"{Path(args.xyz).stem}_{tag}_{args.method}.xyz"
        write(path, frames)
        m = meta["fractions"][tag]
        print(f"f={tag}: {m['n_selected']}/{n_env} envs (budget {m['budget']}), "
              f"{m['frames_with_selection']} frames -> {path}", flush=True)

    with open(out / "accounting.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta["seconds_total"] = round(time.time() - t0)
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(f"done ({meta['seconds_total']}s) -> {out}")


if __name__ == "__main__":
    main()
