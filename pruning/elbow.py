"""Elbow selection on saved f_uniq(h) curves, with several explicit definitions.

Reads ``funiq_curves.csv`` from funiq_sweep.py and, for every (scope,
subsample, rep) curve, reports the bandwidth h* under each rule:

  lin_knee   chord/kneedle knee of f_uniq vs h on a LINEAR h axis (Fig. 1b of
             the paper plots f_uniq against linear h); axes normalised to [0,1].
  log_knee   same rule on log10(h).
  lin_curv   maximum curvature of a cubic smoothing spline of f_uniq(h),
             linear h, evaluated on a dense grid; the "max curvature" rule.
  f_target   smallest h at which f_uniq <= FTARGET (default 0.3); the
             fixed-f_uniq alternative raised by Reviewer 4 #1.

Also prints a calibration table against the h* values quoted in the paper so
that the rule matching the published analysis can be identified, and a
robustness table (h* vs subsample fraction) for Reviewer 2 #1.

Usage:
    python elbow.py results/funiq/groups_s0/funiq_curves.csv --out results/funiq/groups_s0/elbows_all.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

# h* values quoted in the manuscript text (Sec. 3.2); f_uniq at elbow where given
PAPER = {
    "diamond_3.68gcc_300": (0.017, None),
    "graphite_2.39gcc_300": (0.018, None),
    "graphite_2.56gcc_1500": (0.0347, None),
    "graphite_2.67gcc_3000": (0.0448, None),
    "HD liquid_3.0gcc_8000": (None, 0.21),
}


def knee(x, y):
    x = (x - x[0]) / (x[-1] - x[0])
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    p0, p1 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    rel = np.stack([x, y], 1) - p0
    return int(np.argmax(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])))


def max_curvature(h, f, n_dense=4000):
    x = (h - h[0]) / (h[-1] - h[0])
    y = (f - f.min()) / (f.max() - f.min() + 1e-12)
    s = UnivariateSpline(x, y, k=4, s=1e-4 * len(x))
    xx = np.linspace(0, 1, n_dense)
    d1, d2 = s.derivative(1)(xx), s.derivative(2)(xx)
    kappa = np.abs(d2) / (1 + d1 ** 2) ** 1.5
    # the convex "elbow" (decline flattening) has d2 > 0; ignore the concave onset
    kappa[d2 < 0] = 0
    return float(h[0] + xx[np.argmax(kappa)] * (h[-1] - h[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("curves_csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--f-target", type=float, default=0.3)
    ap.add_argument("--h-max", type=float, default=None,
                    help="restrict the knee/curvature search to h <= h_max (the chord knee is range-dependent; "
                         "h_max=0.05 reproduces the paper's crystalline h* values, see elbow_calibrate.py)")
    args = ap.parse_args()

    df = pd.read_csv(args.curves_csv)
    rows = []
    for (scope, sub, rep), g in df.groupby(["scope", "subsample", "rep"]):
        g = g.sort_values("h")
        h_all, f_all = g["h"].to_numpy(), g["f_uniq"].to_numpy()
        m = np.ones(len(h_all), bool) if args.h_max is None else (h_all <= args.h_max)
        h, f = h_all[m], f_all[m]
        i_lin, i_log = knee(h, f), knee(np.log10(h), f)
        h_curv = max_curvature(h, f)
        below = np.flatnonzero(f <= args.f_target)
        h_t = float(h[below[0]]) if len(below) else np.nan
        rows.append({"scope": scope, "subsample": sub, "rep": rep, "n": int(g["n"].iloc[0]),
                     "h_lin_knee": h[i_lin], "f_lin_knee": f[i_lin],
                     "h_log_knee": h[i_log], "f_log_knee": f[i_log],
                     "h_lin_curv": h_curv, "f_lin_curv": float(np.interp(h_curv, h, f)),
                     "h_ftarget": h_t, "f_at_0.015": float(np.interp(0.015, h_all, f_all)),
                     "f_at_hmax": f_all[-1], "h_max_search": args.h_max or h_all[-1]})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    full = out[out.subsample == 1.0].set_index("scope")
    print("== elbows on full groups")
    print(full[["n", "f_at_0.015", "h_lin_knee", "f_lin_knee", "h_lin_curv", "f_lin_curv",
                "h_log_knee", "f_log_knee", "h_ftarget", "f_at_hmax"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n== calibration vs values quoted in the paper")
    for s, (hp, fp) in PAPER.items():
        if s in full.index:
            r = full.loc[s]
            print(f"  {s:24s} paper h*={hp} f*={fp} | lin_knee {r.h_lin_knee:.4f} (f={r.f_lin_knee:.2f})"
                  f"  lin_curv {r.h_lin_curv:.4f} (f={r.f_lin_curv:.2f})  log_knee {r.h_log_knee:.4f}")

    print("\n== robustness: h_lin_knee (mean over reps) vs subsample fraction")
    rob = out.pivot_table(index="scope", columns="subsample", values="h_lin_knee", aggfunc="mean")
    print(rob.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
