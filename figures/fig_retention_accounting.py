"""Figure: what a retention fraction means (Reviewer 1 #1 / Reviewer 4 #2).

(a) Share of the retained environment budget taken by each state point at a
    given retention (default 5%), Global-FPS vs Stratified-FPS vs Random.
(b) Fraction of simulation frames that contain >= 1 retained environment
    (and therefore enter training with a masked force loss) vs retention.

Reads results/stats_train/retention_accounting_{global,stratified,random_global}_s0.csv
Writes figures/out/retention_accounting.{png,pdf}
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
    "lines.linewidth": 1.5, "lines.markersize": 4,
})
SERIES = {  # validated 3-slot categorical palette, fixed order
    "global": ("Global FPS", "#2a78d6"),
    "stratified": ("Stratified FPS", "#eb6834"),
    "random_global": ("Random", "#1baf7a"),
}
ORDER = [
    "diamond_3.68gcc_300", "diamond_3.67gcc_3000", "diamond_0.0gcc_CC",
    "graphite_2.39gcc_300", "graphite_2.56gcc_1500", "graphite_2.67gcc_3000", "graphite_0.0gcc_CC",
    "LD liquid_0.5gcc_1000", "LD liquid_1.0gcc_2000",
    "HD liquid_2.0gcc_6000", "HD liquid_2.0gcc_7000", "HD liquid_2.5gcc_6000",
    "HD liquid_3.6gcc_6000", "HD liquid_3.0gcc_8000",
]


def pretty(g):
    g = g.replace("_0.0gcc_CC", " cold curve").replace("gcc_", " g/cc, ").replace("_", " ")
    return g + (" K" if g[-1].isdigit() else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-dir", default="results/stats_train")
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--out", default="figures/out/retention_accounting")
    args = ap.parse_args()

    data = {k: pd.read_csv(f"{args.stats_dir}/retention_accounting_{k}_s0.csv") for k in SERIES}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) budget share per group at the chosen retention
    y = range(len(ORDER))
    for k, (label, color) in SERIES.items():
        d = data[k]
        d = d[(d.fraction.round(4) == round(args.fraction, 4)) & (d.group != "Full")].set_index("group")
        share = d.n_env / d.n_env.sum()
        ax1.plot(share.reindex(ORDER).values, y, "o", color=color, label=label, zorder=3,
                 markeredgecolor="white", markeredgewidth=0.8)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels([pretty(g) for g in ORDER])
    ax1.invert_yaxis()
    ax1.grid(axis="x", zorder=0)
    ax1.set_xlabel(f"share of retained environments at {args.fraction:.0%} retention")
    ax1.set_xlim(0, None)
    ax1.set_title("(a) where the budget goes", loc="left")
    ax1.legend(frameon=False, loc="lower right")
    # direct annotation of the striking value
    d = data["global"]
    d = d[(d.fraction.round(4) == round(args.fraction, 4)) & (d.group != "Full")].set_index("group")
    # place the note to the right of the largest marker in that row
    rowmax = {}
    for k in SERIES:
        dk = data[k]
        dk = dk[(dk.fraction.round(4) == round(args.fraction, 4)) & (dk.group != "Full")].set_index("group")
        sh = dk.n_env / dk.n_env.sum()
        for g in ORDER:
            rowmax[g] = max(rowmax.get(g, 0), float(sh.get(g, 0)))
    for g in ("diamond_3.68gcc_300", "graphite_2.39gcc_300"):
        ax1.annotate(f"global keeps {int(d.loc[g, 'n_env'])} env.", (rowmax[g], ORDER.index(g)),
                     xytext=(6, 0), textcoords="offset points", va="center", fontsize=6.5,
                     color="#52514e")

    # (b) fraction of frames entering training vs retention
    for k, (label, color) in SERIES.items():
        d = data[k]
        d = d[d.group == "Full"].sort_values("fraction")
        ax2.plot(d.fraction * 100, d.frac_frames, "-", color=color, label=label)
    ax2.set_xscale("log")
    ax2.set_xlabel("retained environments (% of dataset)")
    ax2.set_ylabel("fraction of frames entering training")
    ax2.set_ylim(0, 1.02)
    ax2.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax2.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax2.grid(axis="y", zorder=0)
    ax2.set_title("(b) frames vs environments", loc="left")
    ax2.legend(frameon=False, loc="lower right")

    fig.tight_layout(w_pad=2.0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out + ".png", dpi=300)
    fig.savefig(args.out + ".pdf")
    print("wrote", args.out + ".{png,pdf}")


if __name__ == "__main__":
    main()
