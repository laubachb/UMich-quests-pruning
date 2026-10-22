"""Error vs. force magnitude for the retrained models (Reviewer 4 #6), all datasets.

Uses the per-model predicted test forces saved by evaluate_models.py --save-forces
(one .npy per model, file name = model directory path with "/" -> "__") and the
experiment matrices to label them.  For every model and test group, bins the
reference force magnitude |F| by the group's own quantiles and reports the MAE
per bin (all, bottom50, mid40, top10, top1).  Then compares selections at equal
retention in the top-10% bin: stratified vs global FPS, stratified FPS vs random
stratified, E4 adaptive vs E5 flat, and the E6 published baselines vs stratified
FPS, with a 95% bootstrap CI over seeds.

Outputs results/eval/<dir>/force_tail.csv (per model x group x bin) and
results/eval/<dir>/force_tail_contrasts.csv.

Usage:
    python evaluation/force_tail_retrained.py --pred-dir $SCRATCH/UMich-quests-pruning-models/relstop/pred_forces \
        [--eval-dir results/eval/relstop] [--models-dirname models_relstop]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

from compare_experiments import boot_ci

REPO = Path(__file__).resolve().parent.parent
HARTREE_BOHR_TO_EV_A = 27.211386245988 * 1.8897261246257702
DATA = {
    "carbon": ("data/random_C.xyz", True, ["training/experiments.csv", "training/experiments_baselines.csv"]),
    "sio2": ("data/external/sio2/test.xyz", False, ["training/experiments_sio2.csv"]),
    "monbtavw": ("data/external/monbtavw/test.xyz", False, ["training/experiments_monbtavw.csv"]),
}
BINS = [("all", 0, 1.0), ("bottom50", 0, 0.5), ("mid40", 0.5, 0.9), ("top10", 0.9, 1.0), ("top1", 0.99, 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--eval-dir", default="results/eval/relstop")
    ap.add_argument("--models-dirname", default="models_relstop")
    ap.add_argument("--exclude-sio2", type=int, nargs="*", default=[122, 488, 265],
                    help="SiO2 test frames to drop (out-of-distribution Si-Si contacts; see recompute_excluding_frames.py)")
    args = ap.parse_args()
    rows = []
    for ds, (test, convert, mats) in DATA.items():
        pdir = Path(args.pred_dir) / ds
        if not pdir.exists():
            print(f"{ds}: no predictions in {pdir}")
            continue
        frames = read(REPO / test, ":")
        drop = set(args.exclude_sio2) if ds == "sio2" else set()
        keep_atoms = np.repeat(np.array([i not in drop for i in range(len(frames))]), [len(a) for a in frames])
        g = np.concatenate([np.full(len(a), a.info["config_tag"]) for a in frames])
        F = np.concatenate([a.get_forces() for a in frames]) * (HARTREE_BOHR_TO_EV_A if convert else 1.0)
        Fmag = np.linalg.norm(F, axis=1)
        groups = sorted(set(g))
        masks = {}
        for grp in groups + ["Full"]:
            gm = (np.ones(len(g), bool) if grp == "Full" else (g == grp)) & keep_atoms
            q = np.quantile(Fmag[gm], [0, 0.5, 0.9, 0.99, 1.0])
            qm = {0: q[0], 0.5: q[1], 0.9: q[2], 0.99: q[3], 1.0: q[4]}
            for name, lo, hi in BINS:
                masks[(grp, name)] = gm & (Fmag >= qm[lo]) & (Fmag <= qm[hi])
        mat = pd.concat([pd.read_csv(REPO / m) for m in mats if (REPO / m).exists()])
        for r in mat.itertuples():
            od = re.sub(r"/models/", f"/{args.models_dirname}/", "/" + r.out_dir)[1:]
            # predictions are named by the absolute model directory with "/" -> "__"
            hits = list(pdir.glob("*" + od.replace("/", "__") + ".npy"))
            if not hits:
                continue
            P = np.load(hits[0])
            err = np.abs(P - F).mean(1)
            rate = 1.0 if r.exp == "E3_full" else round(float(r.fraction), 3)
            for (grp, name), m in masks.items():
                rows.append((ds, r.exp, r.selection, rate, int(r.train_seed), grp, name, int(m.sum()), float(err[m].mean())))
        print(f"{ds}: {len(set((x[1], x[2], x[3], x[4]) for x in rows if x[0] == ds))} models")
    df = pd.DataFrame(rows, columns=["dataset", "exp", "selection", "rate", "train_seed", "group", "bin", "n_atoms", "mae"])
    out = REPO / args.eval_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "force_tail.csv", index=False)

    # contrasts in the top-10% bin
    def S(d, exp, sel, rate=None):
        m = (d.exp == exp) & (d.selection == sel)
        if rate is not None:
            m &= d.rate == rate
        return d[m].sort_values("train_seed").mae.values
    con = []
    for (ds, grp), d in df[df.bin == "top10"].groupby(["dataset", "group"]):
        e5 = d[d.exp == "E5_matched"].rate.unique()
        tests = []
        for r in (0.05, 0.1, 0.2):
            tests += [("strat_vs_global", r, S(d, "E1_method", "stratified_s0", r), S(d, "E1_method", "global_s0", r)),
                      ("strat_vs_randstrat", r, S(d, "E1_method", "stratified_s0", r), S(d, "E1_method", "random_stratified_s0", r))]
            for m in ("direct", "msc", "cur"):
                tests.append((f"{m}_vs_strat", r, S(d, "E6_baseline", m, r), S(d, "E1_method", "stratified_s0", r)))
        if len(e5):
            for e4 in sorted(x for x in d.exp.unique() if x.startswith("E4")):
                tests.append((f"{e4.split('_')[-1]}_vs_matched", float(e5[0]), S(d, e4, "stratified_s0"), S(d, "E5_matched", "stratified_s0")))
        for name, r, a, b in tests:
            if len(a) and len(b):
                st, lo, hi = boot_ci(a, b)
                con.append(dict(dataset=ds, group=grp, rate=r, contrast=name, mae_a=float(np.median(a)), mae_b=float(np.median(b)),
                                rel_improvement=st, ci_lo=lo, ci_hi=hi))
    C = pd.DataFrame(con, columns=["dataset", "group", "rate", "contrast", "mae_a", "mae_b", "rel_improvement", "ci_lo", "ci_hi"])
    C.to_csv(out / "force_tail_contrasts.csv", index=False)
    for ds in C.dataset.unique():
        print(f"\n== {ds}: top-10% |F| bin, all test atoms: improvement of A over B, 95% CI")
        for _, x in C[(C.dataset == ds) & (C.group == "Full")].iterrows():
            flag = "" if x.ci_lo > 0 else ("  WORSE" if x.ci_hi < 0 else "  (CI includes 0)")
            print(f"  {x.contrast:22s} rate={x.rate:<6} {x.mae_a:.3f} vs {x.mae_b:.3f}  {x.rel_improvement:+6.1%} [{x.ci_lo:+6.1%}, {x.ci_hi:+6.1%}]{flag}")
    print(f"\nwrote {out}/force_tail.csv and force_tail_contrasts.csv")


if __name__ == "__main__":
    main()
