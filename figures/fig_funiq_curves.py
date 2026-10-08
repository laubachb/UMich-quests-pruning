"""Small multiples of f_uniq(h) per group with paper elbow values overlaid.

Markers: paper h* (text, crystals) as a vertical dashed line; paper elbow
f_uniq (Fig. 3 red bars) as a horizontal dotted line, with the h where the
re-run curve reaches that f_uniq as a filled circle; chord knee on full range
(open square) and on h <= 0.05 (open triangle).
"""
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.spines.top": False,
                            "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                            "grid.color": "#e6e5e1", "grid.linewidth": 0.5})
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
PAPER_H = {"diamond_3.68gcc_300": 0.017, "graphite_2.39gcc_300": 0.018,
           "graphite_2.56gcc_1500": 0.0347, "graphite_2.67gcc_3000": 0.0448}
ORDER = ["diamond_3.68gcc_300", "diamond_3.67gcc_3000", "diamond_0.0gcc_CC",
         "graphite_2.39gcc_300", "graphite_2.56gcc_1500", "graphite_2.67gcc_3000", "graphite_0.0gcc_CC",
         "LD liquid_0.5gcc_1000", "LD liquid_1.0gcc_2000", "HD liquid_2.0gcc_6000", "HD liquid_2.0gcc_7000",
         "HD liquid_2.5gcc_6000", "HD liquid_3.6gcc_6000", "HD liquid_3.0gcc_8000"]

c = pd.read_csv("results/funiq/groups_s0/funiq_curves.csv"); c = c[c.subsample == 1.0]
e_full = pd.read_csv("results/funiq/groups_s0/elbows_all.csv").query("subsample==1").set_index("scope")
e_05 = pd.read_csv("results/funiq/groups_s0/elbows_hmax05.csv").query("subsample==1").set_index("scope")
pf = {k: v for k, v in json.load(open("results/funiq/paper/budget_paper_fig3.json")).items() if not k.startswith("_")}

fig, axes = plt.subplots(3, 5, figsize=(10, 6), sharey=True)
for ax, s in zip(axes.flat, ORDER):
    g = c[c.scope == s].sort_values("h"); h, f = g.h.to_numpy(), g.f_uniq.to_numpy()
    ax.plot(h, f, "-", color=INK, lw=1.3)
    ax.axhline(pf[s], color=ORANGE, ls=":", lw=1)
    h_at_pf = np.exp(np.interp(-pf[s], -f, np.log(h)))  # f decreasing in h
    ax.plot(h_at_pf, pf[s], "o", color=ORANGE, ms=5, mec="white", mew=0.6, label="paper elbow $f_{uniq}$ (Fig. 3)")
    if s in PAPER_H:
        ax.axvline(PAPER_H[s], color=BLUE, ls="--", lw=1, label="paper $h^*$ (text)")
    ax.plot(e_full.loc[s, "h_lin_knee"], e_full.loc[s, "f_lin_knee"], "s", mfc="none", color=AQUA, ms=5, label="chord knee, full range")
    ax.plot(e_05.loc[s, "h_lin_knee"], e_05.loc[s, "f_lin_knee"], "^", mfc="none", color=AQUA, ms=5, label="chord knee, $h\\leq0.05$")
    ax.set_xscale("log"); ax.set_xlim(0.005, 0.5); ax.set_ylim(0, 1.02); ax.grid(axis="y")
    ax.set_title(s.replace("_", " ").replace("gcc", " g/cc"), loc="left")
    ax.set_xticks([0.01, 0.03, 0.1, 0.3]); ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3"])
axes.flat[-1].axis("off")
h_, l_ = axes.flat[0].get_legend_handles_labels()
axes.flat[-1].legend(h_, l_, loc="center", frameon=False)
for ax in axes[-1]: ax.set_xlabel("bandwidth $h$")
for ax in axes[:, 0]: ax.set_ylabel("$f_{uniq}$")
fig.tight_layout()
fig.savefig("figures/out/funiq_curves.png", dpi=200); fig.savefig("figures/out/funiq_curves.pdf")
print("wrote figures/out/funiq_curves.png")
