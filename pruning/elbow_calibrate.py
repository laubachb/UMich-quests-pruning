"""Which elbow rule / h-range reproduces the h* values quoted in the paper?

For each candidate (rule, h_max) the elbow is located on the full-group
f_uniq(h) curve restricted to h <= h_max, and compared with the paper's
quoted values.  Reports log-ratio RMS error per candidate.
"""
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

PAPER_H = {"diamond_3.68gcc_300": 0.017, "graphite_2.39gcc_300": 0.018,
           "graphite_2.56gcc_1500": 0.0347, "graphite_2.67gcc_3000": 0.0448}
PAPER_F = {"HD liquid_3.0gcc_8000": 0.21, "diamond_3.68gcc_300": 0.29, "graphite_2.39gcc_300": 0.36,
           "HD liquid_2.0gcc_6000": 0.26, "LD liquid_1.0gcc_2000": 0.55}  # Fig. 3 red bars (read off)


def knee(x, y):
    x = (x - x[0]) / (x[-1] - x[0]); y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    p0, p1 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    d = (p1 - p0) / np.linalg.norm(p1 - p0); rel = np.stack([x, y], 1) - p0
    return int(np.argmax(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])))


def curv(h, f):
    x = (h - h[0]) / (h[-1] - h[0]); y = (f - f.min()) / (f.max() - f.min() + 1e-12)
    s = UnivariateSpline(x, y, k=4, s=1e-4 * len(x)); xx = np.linspace(0, 1, 2000)
    d1, d2 = s.derivative(1)(xx), s.derivative(2)(xx); k = np.abs(d2) / (1 + d1 ** 2) ** 1.5; k[d2 < 0] = 0
    return h[0] + xx[np.argmax(k)] * (h[-1] - h[0])


df = pd.read_csv(sys.argv[1]); df = df[df.subsample == 1.0]
curves = {s: g.sort_values("h") for s, g in df.groupby("scope")}
rows = []
for hmax in [0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]:
    for rule in ["lin_knee", "log_knee", "lin_curv", "lin_knee_dense"]:
        est = {}
        for s, g in curves.items():
            m = g.h <= hmax; h, f = g.h[m].to_numpy(), g.f_uniq[m].to_numpy()
            if len(h) < 6: continue
            if rule == "lin_knee": est[s] = h[knee(h, f)]
            elif rule == "log_knee": est[s] = h[knee(np.log10(h), f)]
            elif rule == "lin_curv": est[s] = curv(h, f)
            else:  # knee on a LINEARLY spaced h grid (interpolated), as a linear sweep would give
                hh = np.linspace(h[0], hmax, 200); ff = np.interp(hh, h, f); est[s] = hh[knee(hh, ff)]
        if not any(s in est for s in PAPER_H): continue
        err_h = np.sqrt(np.mean([np.log(est[s] / PAPER_H[s]) ** 2 for s in PAPER_H if s in est]))
        fest = {s: float(np.interp(est[s], curves[s].h, curves[s].f_uniq)) for s in PAPER_F if s in est}
        err_f = np.sqrt(np.mean([(fest[s] - PAPER_F[s]) ** 2 for s in fest]))
        rows.append((rule, hmax, err_h, err_f, *[est.get(s, np.nan) for s in PAPER_H], *[fest.get(s, np.nan) for s in PAPER_F]))
out = pd.DataFrame(rows, columns=["rule", "hmax", "rms_log_err_h", "rms_err_f", *[f"h*{k[:10]}" for k in PAPER_H], *[f"f*{k[:10]}" for k in PAPER_F]])
print("paper h*:", PAPER_H); print("paper f*:", PAPER_F)
print(out.sort_values("rms_log_err_h").head(12).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
