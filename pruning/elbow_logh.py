"""Elbow rules on log10(h) tested against the paper's full elbow table.

Hypothesis from figures/out/funiq_curves.png: f_uniq vs log h is a sigmoid and
the paper's elbow lies on its steep limb.  Candidate rules (all on a dense
interpolant of f vs x = log10 h):
  inflect   x of max |df/dx|                                 (sigmoid midpoint)
  kappa_lo  x of max curvature on the lower (convex) bend    (where the fall flattens)
  kappa_hi  x of max curvature on the upper (concave) bend   (where the fall begins)
  mid_lo    midpoint between inflect and kappa_lo
Compared with paper h* (text, crystals) and paper elbow f_uniq (Fig. 3, all).
"""
import json, sys
import numpy as np, pandas as pd
from scipy.interpolate import UnivariateSpline

PAPER_H = {"diamond_3.68gcc_300": 0.017, "graphite_2.39gcc_300": 0.018,
           "graphite_2.56gcc_1500": 0.0347, "graphite_2.67gcc_3000": 0.0448}
pf = {k: v for k, v in json.load(open("results/funiq/paper/budget_paper_fig3.json")).items() if not k.startswith("_")}
c = pd.read_csv(sys.argv[1]); c = c[c.subsample == 1.0]

def rules(h, f):
    x = np.log10(h); s = UnivariateSpline(x, f, k=4, s=0); xx = np.linspace(x[0], x[-1], 3000)
    d1, d2 = s.derivative(1)(xx), s.derivative(2)(xx); kap = np.abs(d2) / (1 + d1**2)**1.5
    i_inf = np.argmax(np.abs(d1))
    lo = np.where((d2 > 0) & (xx > xx[i_inf]))[0]; hi = np.where((d2 < 0) & (xx < xx[i_inf]))[0]
    i_lo = lo[np.argmax(kap[lo])] if len(lo) else i_inf; i_hi = hi[np.argmax(kap[hi])] if len(hi) else i_inf
    out = {"inflect": xx[i_inf], "kappa_lo": xx[i_lo], "kappa_hi": xx[i_hi], "mid_lo": 0.5 * (xx[i_inf] + xx[i_lo])}
    return {k: (10**v, float(s(v))) for k, v in out.items()}

rows = []
for sc, g in c.groupby("scope"):
    g = g.sort_values("h"); r = rules(g.h.to_numpy(), g.f_uniq.to_numpy())
    h_pf = float(np.exp(np.interp(-pf[sc], -g.f_uniq.to_numpy(), np.log(g.h.to_numpy()))))
    rows.append({"scope": sc, "paper_h": PAPER_H.get(sc, np.nan), "paper_f": pf[sc], "h_at_paper_f": h_pf,
                 **{f"{k}_h": v[0] for k, v in r.items()}, **{f"{k}_f": v[1] for k, v in r.items()}})
t = pd.DataFrame(rows).set_index("scope")
print(t[["paper_h", "paper_f", "h_at_paper_f", "inflect_h", "inflect_f", "mid_lo_h", "mid_lo_f", "kappa_lo_h", "kappa_lo_f"]]
      .to_string(float_format=lambda v: f"{v:.3f}"))
print("\nRMS error vs paper (f_uniq, all 14 groups) and vs paper h* (log ratio, 4 crystals):")
for k in ["inflect", "mid_lo", "kappa_lo", "kappa_hi"]:
    ef = np.sqrt(np.mean((t[f"{k}_f"] - t.paper_f) ** 2))
    m = t.paper_h.notna(); eh = np.sqrt(np.mean(np.log(t.loc[m, f"{k}_h"] / t.loc[m, "paper_h"]) ** 2))
    print(f"  {k:9s} rms_f={ef:.3f}  rms_log_h={eh:.3f}")
t.to_csv(sys.argv[2])
