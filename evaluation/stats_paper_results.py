"""Statistics for the paper's existing force-MAE results (Reviewer 1 #4, Reviewer 4 #5).

Reads the two test-MAE CSVs (one row per model x group) and reports, per group
and retention level: n independent runs, median, IQR; plus bootstrap CIs on the
relative improvement of Stratified-Adaptive over Global-Fixed at chosen
retention levels, and a scan for outlier runs (the ~40% spike in Fig. 2).

Usage:
    python stats_paper_results.py results/paper/stratified_adaptive_test_mae.csv \
        results/paper/global_fixed_test_mae.csv --out results/stats
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(path, label):
    df = pd.read_csv(path)
    df = df[["group", "rate", "forces_mae", "compiled_model"]].copy()
    df["method"] = label
    return df


def bootstrap_rel_improvement(a, b, n_boot=20000, seed=0):
    """Relative improvement of a over b in the median: 1 - med(a)/med(b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    stat = 1 - np.median(a) / np.median(b)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = 1 - np.median(rng.choice(a, len(a))) / np.median(rng.choice(b, len(b)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return stat, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stratified_csv")
    ap.add_argument("global_csv")
    ap.add_argument("--out", default="results/stats")
    ap.add_argument("--rates", type=float, nargs="*", default=[0.05, 0.1, 0.2])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.concat([load(args.stratified_csv, "stratified_adaptive"),
                    load(args.global_csv, "global_fixed")])
    df["rate"] = df["rate"].round(4)

    # per (method, group, rate): n, median, IQR
    summ = (df.groupby(["method", "group", "rate"])["forces_mae"]
              .agg(n="count", median="median",
                   q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
                   mean="mean", std="std")
              .reset_index())
    summ.to_csv(out / "per_point_summary.csv", index=False)

    nruns = summ.pivot_table(index="rate", columns="method", values="n", aggfunc="min")
    print("== independent runs per retention level (min over groups)")
    print(nruns.to_string())

    # bootstrap CIs for headline improvements
    print("\n== relative improvement (median), Stratified-Adaptive vs Global-Fixed, 95% bootstrap CI")
    rows = []
    for g in sorted(df["group"].unique()):
        for r in args.rates:
            a = df[(df.method == "stratified_adaptive") & (df.group == g) & (df.rate == r)]["forces_mae"]
            b = df[(df.method == "global_fixed") & (df.group == g) & (df.rate == r)]["forces_mae"]
            if len(a) < 3 or len(b) < 3:
                continue
            stat, lo, hi = bootstrap_rel_improvement(a, b)
            rows.append({"group": g, "rate": r, "n_strat": len(a), "n_global": len(b),
                         "rel_improvement": stat, "ci_lo": lo, "ci_hi": hi})
    ci = pd.DataFrame(rows)
    ci.to_csv(out / "bootstrap_improvement.csv", index=False)
    for r in args.rates:
        print(f"-- retention {r:.0%}")
        for _, x in ci[ci.rate == r].iterrows():
            flag = ("" if x.ci_lo > 0 else
                    "  (significantly WORSE)" if x.ci_hi < 0 else "  (CI includes 0)")
            print(f"  {x.group:28s} {x.rel_improvement:+6.1%}  [{x.ci_lo:+6.1%}, {x.ci_hi:+6.1%}]"
                  f"  n={x.n_strat}/{x.n_global}{flag}")

    # outlier scan: runs whose 'Full' MAE is > 3 IQR above the median at that rate
    print("\n== outlier runs (Full-group MAE > median + 3*IQR at that rate)")
    full = df[df.group == "Full"]
    for (m, r), grp in full.groupby(["method", "rate"]):
        med, iqr = grp.forces_mae.median(), grp.forces_mae.quantile(0.75) - grp.forces_mae.quantile(0.25)
        bad = grp[grp.forces_mae > med + 3 * max(iqr, 1e-6)]
        for _, x in bad.iterrows():
            print(f"  {m:20s} rate={r:<6} mae={x.forces_mae:.3f} (median {med:.3f})  {x.compiled_model}")


if __name__ == "__main__":
    main()
