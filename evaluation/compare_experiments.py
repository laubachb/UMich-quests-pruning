"""Head-to-head comparisons of the reviewer-response experiments (E1-E6).

Reads the per-model, per-group test MAE CSVs written by evaluate_models.py for one
evaluation directory (results/eval/<dir>/{carbon,sio2,monbtavw}_test_mae.csv) and,
for every dataset, group and retention level, reports the median MAE per
selection and the relative improvement of selected contrasts with a 95%
bootstrap CI over training seeds (paired by seed where both sides have the same
seeds).

Contrasts (A vs B; improvement = 1 - median(A)/median(B)):
  strat_vs_global     stratified FPS vs global FPS            (R2 #2: stratification)
  strat_vs_randstrat  stratified FPS vs random stratified     (R4 #3: does FPS matter?)
  global_vs_randglob  global FPS vs random global
  adaptive_vs_matched E4 per-group elbow budgets vs E5 stratified flat fraction (R2 #2: the adaptive claim)
  matched_strat_vs_global  E5 stratified vs E5 global at the same overall fraction
  <method>_vs_strat   E6 DIRECT / MSC / CUR vs stratified FPS at the same label budget (carbon)
  seed1_vs_seed0      E2 second FPS start vs first (same selection mode)

Usage:
    python evaluation/compare_experiments.py results/eval/relstop [--out results/eval/relstop/comparisons.csv]
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["carbon", "sio2", "monbtavw"]
RNG = np.random.default_rng(0)


def boot_ci(a, b, n=5000):
    """95% bootstrap CI of 1 - median(a)/median(b); paired resampling when lengths match."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    stat = 1 - np.median(a) / np.median(b)
    if len(a) == len(b):
        idx = RNG.integers(0, len(a), size=(n, len(a)))
        boots = 1 - np.median(a[idx], 1) / np.median(b[idx], 1)
    else:
        ia = RNG.integers(0, len(a), size=(n, len(a)))
        ib = RNG.integers(0, len(b), size=(n, len(b)))
        boots = 1 - np.median(a[ia], 1) / np.median(b[ib], 1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return stat, lo, hi


def load(evdir, ds):
    p = Path(evdir) / f"{ds}_test_mae.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["rate"] = pd.to_numeric(d["rate"], errors="coerce").round(3)
    d["train_seed"] = pd.to_numeric(d["train_seed"], errors="coerce").astype("Int64")
    d["dataset"] = ds
    return d


def sel(d, exp, selection, rate=None):
    m = (d.exp == exp) & (d.selection == selection)
    if rate is not None:
        m &= d.rate == rate
    return d[m].sort_values("train_seed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evdir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or Path(args.evdir) / "comparisons.csv")

    rows, summary = [], []
    for ds in DATASETS:
        d = load(args.evdir, ds)
        if d is None:
            continue
        groups = sorted(set(d.group) - {"Full"}) + ["Full"]
        rates = sorted(set(d[d.exp == "E1_method"].rate.dropna()))
        e4 = sorted(set(d[d.exp.str.startswith("E4")].exp))
        e5rate = d[d.exp == "E5_matched"].rate.dropna().unique()
        e5rate = float(e5rate[0]) if len(e5rate) else None
        for g in groups:
            dg = d[d.group == g]
            contrasts = []
            for r in rates:
                contrasts += [
                    ("strat_vs_global", r, sel(dg, "E1_method", "stratified_s0", r), sel(dg, "E1_method", "global_s0", r)),
                    ("strat_vs_randstrat", r, sel(dg, "E1_method", "stratified_s0", r), sel(dg, "E1_method", "random_stratified_s0", r)),
                    ("global_vs_randglob", r, sel(dg, "E1_method", "global_s0", r), sel(dg, "E1_method", "random_global_s0", r)),
                    ("seed1_vs_seed0_strat", r, sel(dg, "E2_fpsseed", "stratified_s1", r), sel(dg, "E1_method", "stratified_s0", r)),
                    ("seed1_vs_seed0_global", r, sel(dg, "E2_fpsseed", "global_s1", r), sel(dg, "E1_method", "global_s0", r)),
                ]
                for m in ("direct", "msc", "cur"):
                    b = sel(dg, "E6_baseline", m, r)
                    if len(b):
                        contrasts.append((f"{m}_vs_strat", r, b, sel(dg, "E1_method", "stratified_s0", r)))
            if e5rate is not None:
                contrasts.append(("matched_strat_vs_global", e5rate, sel(dg, "E5_matched", "stratified_s0"), sel(dg, "E5_matched", "global_s0")))
                for e in e4:
                    contrasts.append((f"adaptive_{e.split('_')[-1]}_vs_matched", e5rate, sel(dg, e, "stratified_s0"), sel(dg, "E5_matched", "stratified_s0")))
            for name, r, a, b in contrasts:
                if len(a) == 0 or len(b) == 0:
                    continue
                stat, lo, hi = boot_ci(a.forces_mae.values, b.forces_mae.values)
                rows.append({"dataset": ds, "group": g, "rate": r, "contrast": name, "n_a": len(a), "n_b": len(b),
                             "mae_a": float(a.forces_mae.median()), "mae_b": float(b.forces_mae.median()),
                             "rel_improvement": stat, "ci_lo": lo, "ci_hi": hi})
        # per-selection medians (Full group) for the printed overview
        full = d[d.group == "Full"].groupby(["exp", "selection", "rate"]).forces_mae.agg(["median", "count"]).reset_index()
        full.insert(0, "dataset", ds)
        summary.append(full)

    res = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    pd.concat(summary).to_csv(out.with_name(out.stem + "_medians.csv"), index=False)

    pd.set_option("display.width", 200)
    for ds in DATASETS:
        r = res[(res.dataset == ds) & (res.group == "Full")]
        if r.empty:
            continue
        print(f"\n== {ds}: all test atoms, relative MAE improvement of A over B (median over seeds, 95% bootstrap CI)")
        for _, x in r.iterrows():
            flag = "" if x.ci_lo > 0 else ("  WORSE" if x.ci_hi < 0 else "  (CI includes 0)")
            print(f"  {x.contrast:28s} rate={x.rate:<6} {x.mae_a:.4f} vs {x.mae_b:.4f}  {x.rel_improvement:+6.1%} [{x.ci_lo:+6.1%}, {x.ci_hi:+6.1%}]{flag}")
    print(f"\nwrote {out} and {out.with_name(out.stem + '_medians.csv')}")


if __name__ == "__main__":
    main()
