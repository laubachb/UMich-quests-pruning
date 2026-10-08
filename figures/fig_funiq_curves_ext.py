"""f_uniq(h) small multiples for an external dataset with the log-h elbow marked."""
import argparse, math
import matplotlib, matplotlib.pyplot as plt, numpy as np, pandas as pd
matplotlib.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                            "axes.edgecolor": "#8a8984", "grid.color": "#e6e5e1", "grid.linewidth": 0.5})
ap = argparse.ArgumentParser(); ap.add_argument("name"); args = ap.parse_args()
D = f"results/ext/{args.name}/funiq/groups_s0"
c = pd.read_csv(f"{D}/funiq_curves.csv"); c = c[c.subsample == 1.0]
e = pd.read_csv(f"{D}/elbows_logh.csv").set_index("scope")
scopes = list(e.sort_values("mid_lo_h").index); n = len(scopes); cols = 5; rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows), sharey=True)
for ax, s in zip(axes.flat, scopes):
    g = c[c.scope == s].sort_values("h"); ax.plot(g.h, g.f_uniq, "-", color="#0b0b0b", lw=1.3)
    ax.plot(e.loc[s, "mid_lo_h"], e.loc[s, "mid_lo_f"], "o", color="#eb6834", ms=5, mec="white", mew=0.6)
    ax.axvline(0.015, color="#2a78d6", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(0.005, 0.5); ax.set_ylim(0, 1.02); ax.grid(axis="y")
    ax.set_title(f"{s.replace('_', ' ')}  (h*={e.loc[s, 'mid_lo_h']:.3f}, f={e.loc[s, 'mid_lo_f']:.2f})", loc="left")
    ax.set_xticks([0.01, 0.03, 0.1, 0.3]); ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3"])
for ax in axes.flat[n:]: ax.axis("off")
for ax in axes[-1]: ax.set_xlabel("bandwidth $h$")
for ax in axes[:, 0]: ax.set_ylabel("$f_{uniq}$")
fig.suptitle(f"{args.name}: $f_{{uniq}}(h)$ per group; orange = log-h elbow, dotted = h = 0.015", x=0.01, ha="left", fontsize=8)
fig.tight_layout(); fig.savefig(f"figures/out/funiq_curves_{args.name}.png", dpi=170); print("wrote", f"figures/out/funiq_curves_{args.name}.png")
