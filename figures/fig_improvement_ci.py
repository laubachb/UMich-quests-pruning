"""Figure: relative test-MAE improvement of Stratified over Global FPS with
bootstrap 95% CIs, per state point, at several retention levels (Reviewer 1 #4).

Reads results/stats_test/bootstrap_improvement.csv (from evaluation/stats_paper_results.py)
Writes figures/out/improvement_ci.{png,pdf}
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#8a8984", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
})
# ordinal: one hue (blue), three validated steps light -> dark for 5 / 10 / 20 %
STEPS = {0.05: "#86b6ef", 0.10: "#2a78d6", 0.20: "#104281"}
ORDER = [
    "test_diamond_3.68gcc_300", "test_diamond_3.67gcc_3000",
    "test_graphite_2.39gcc_300", "test_graphite_2.56gcc_1500", "test_graphite_2.67gcc_3000",
    "test_LD_liquid_0.5gcc_1000", "test_LD_liquid_0.5gcc_2000",
    "test_HD_liquid_2.0gcc_6000", "test_HD_liquid_2.0gcc_7000", "test_HD_liquid_2.5gcc_6000",
    "test_HD_liquid_3.6gcc_6000", "test_HD_liquid_3.0gcc_8000", "Full",
]
LABEL = {"test_LD_liquid_0.5gcc_2000": "LD liquid 1.0 g/cc, 2000 K"}  # mislabeled tag (see data/README.md)


def pretty(g):
    if g in LABEL:
        return LABEL[g]
    if g == "Full":
        return "all test atoms"
    g = g.replace("test_", "").replace("_liquid", " liquid").replace("gcc_", " g/cc, ").replace("_", " ")
    return g + " K"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/stats_test/bootstrap_improvement.csv")
    ap.add_argument("--out", default="figures/out/improvement_ci")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    n = len(STEPS)
    for i, (rate, color) in enumerate(STEPS.items()):
        d = df[df.rate.round(3) == rate].set_index("group").reindex(ORDER)
        y = [j + (i - (n - 1) / 2) * 0.22 for j in range(len(ORDER))]
        ax.errorbar(d.rel_improvement * 100, y,
                    xerr=[(d.rel_improvement - d.ci_lo) * 100, (d.ci_hi - d.rel_improvement) * 100],
                    fmt="o", color=color, ecolor=color, elinewidth=1.2, capsize=0, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.6, label=f"{rate:.0%} retained", zorder=3)
    ax.axvline(0, color="#8a8984", linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([pretty(g) for g in ORDER])
    ax.invert_yaxis()
    ax.grid(axis="x", zorder=0)
    ax.set_xlabel("test force-MAE reduction, Stratified vs Global FPS (%)\nmedian over runs, 95% bootstrap CI")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out + ".png", dpi=300)
    fig.savefig(args.out + ".pdf")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
