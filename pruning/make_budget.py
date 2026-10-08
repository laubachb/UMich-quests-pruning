"""Turn per-group elbows into a per-group retention budget (Stratified-Adaptive).

The paper defines the adaptive retention of group c as f_uniq(h*_c): the
fraction of environments counted as unique at the group's elbow bandwidth
(Fig. 2 caption, Sec. 2.2).  This script reads elbows_all.csv from
pruning/elbow.py (full-group rows, subsample == 1.0) and writes
budget_<rule>.json = {scope: fraction} for make_masks.py --budget.

Rules (columns of elbows_all.csv):
  lin_knee   chord knee on linear h      (f_lin_knee)
  lin_curv   max curvature, linear h     (f_lin_curv)
  log_knee   chord knee on log10 h       (f_log_knee)
  ftarget    fixed f_uniq target (R4 #1)  -> constant budget, --f-target
  fixed_h    f_uniq at a fixed h (R4 #1)  -> --h-fixed (default 0.015)

Usage:
    python pruning/make_budget.py results/funiq/groups_s0/elbows_all.csv --rule lin_knee \
        --curves results/funiq/groups_s0/funiq_curves.csv --out results/funiq/groups_s0/budget_lin_knee.json
"""
import argparse
import json

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elbows_csv")
    ap.add_argument("--rule", default="lin_knee",
                    choices=["lin_knee", "lin_curv", "log_knee", "ftarget", "fixed_h"])
    ap.add_argument("--curves", help="funiq_curves.csv (needed for fixed_h)")
    ap.add_argument("--f-target", type=float, default=0.3)
    ap.add_argument("--h-fixed", type=float, default=0.015)
    ap.add_argument("--min", type=float, default=0.01, help="floor on any group's fraction")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    e = pd.read_csv(args.elbows_csv)
    e = e[e.subsample == 1.0].set_index("scope")
    if args.rule == "ftarget":
        budget = {s: args.f_target for s in e.index}
    elif args.rule == "fixed_h":
        c = pd.read_csv(args.curves)
        c = c[c.subsample == 1.0]
        budget = {}
        for s, g in c.groupby("scope"):
            g = g.sort_values("h")
            budget[s] = float(np.interp(args.h_fixed, g.h, g.f_uniq))
    else:
        col = {"lin_knee": "f_lin_knee", "lin_curv": "f_lin_curv", "log_knee": "f_log_knee"}[args.rule]
        budget = {s: float(e.loc[s, col]) for s in e.index}
    budget = {s: max(args.min, min(1.0, v)) for s, v in budget.items()}
    json.dump(budget, open(args.out, "w"), indent=2)
    n = e["n"]
    tot = sum(budget[s] * n[s] for s in budget) / n.sum()
    print(f"rule={args.rule}: overall retention {tot:.3f} ({int(tot * n.sum())} of {n.sum()} envs) -> {args.out}")
    for s in sorted(budget):
        print(f"  {s:28s} {budget[s]:.3f}")


if __name__ == "__main__":
    main()
