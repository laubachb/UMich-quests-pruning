"""Calibration of QUESTS efficiency scores against measured prunable fractions.

The paper's claim is that a per-group (adaptive) bandwidth gives a more
informative efficiency score f_uniq than the fixed default.  This script tests
that directly on carbon:

  measured  For each test state point, the paper's stratified retention sweep
            (results/paper/stratified_adaptive_test_mae.csv: 28 retention levels,
            7-10 runs each; per-group retention == sweep rate) gives the smallest
            retention r*(tau) whose median test MAE is within a tolerance tau of
            the 100%-data model.  Medians are made monotone in retention (pool-
            adjacent-violators) before thresholding.  r* is the model-measured
            "unique fraction" of the group.
  scores    f_uniq of the same group under several bandwidth rules, all read from
            the committed f_uniq(h) curves (results/funiq/groups_s0/funiq_curves.csv):
              fixed      h = 0.015 (QUESTS default) and every h on the 40-point grid
              elbow      h* from pruning/elbow_logh.py (mid_lo = paper's rule; also
                         inflect, kappa_lo) and the paper's Fig. 3 read-off values
              scale      h = c * d_nn, where d_nn is the group's median nearest-
                         neighbour descriptor distance (results/test_distance/
                         summary.csv, train->train leave-one-out); c swept.  This is
                         the "bandwidth set by the group's own length scale" rule
                         the manuscript motivates in Sec. 2.2.
              radius     f at the kernel-free knee of the FPS covering radius curve
                         (results/stats_train/radius_vs_elbow.csv)
  test      Across the 12 state points: Spearman and Pearson correlation of each
            score with r*(tau), and the calibration error RMS(score - r*).

Outputs: results/calibration/carbon_{measured,scores,summary,sweeps}.csv and
figures/out/calibration_carbon.{png,pdf}.

Usage:  python evaluation/calibration.py [--tols 0.02 0.05 0.10]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parent.parent
ORDER = ["diamond_3.68gcc_300", "diamond_3.67gcc_3000", "graphite_2.39gcc_300", "graphite_2.56gcc_1500",
         "graphite_2.67gcc_3000", "LD liquid_0.5gcc_1000", "LD liquid_1.0gcc_2000", "HD liquid_2.0gcc_6000",
         "HD liquid_2.0gcc_7000", "HD liquid_2.5gcc_6000", "HD liquid_3.6gcc_6000", "HD liquid_3.0gcc_8000"]
SHORT = {g: g.replace("_", " ").replace("gcc", " g/cc").replace("HD liquid", "HD").replace("LD liquid", "LD") for g in ORDER}


def canon(t):
    t = t.replace("test_", "").replace("HD_liquid", "HD liquid").replace("LD_liquid", "LD liquid")
    return "LD liquid_1.0gcc_2000" if t == "LD liquid_0.5gcc_2000" else t


def pav_decreasing(y):
    """Pool-adjacent-violators: largest non-increasing fit (in index order) to y."""
    y = list(map(float, y))
    blocks = [[v, 1] for v in y]  # [mean, count]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:  # violation of non-increasing
            m = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [m, blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for m, n in blocks:
        out += [m] * n
    return np.array(out)


MODES = {  # how "within tolerance of the full-data model" is defined
    "rel": "MAE <= (1 + tau) * MAE_full                      (relative to the full-data error)",
    "abs": "MAE <= MAE_full + tau  [eV/A]                    (absolute excess error)",
    "rng": "MAE <= MAE_full + tau * (MAE_1% - MAE_full)       (fraction of the group's own error range)",
}
TOLS = {"rel": [0.02, 0.05, 0.10], "abs": [0.005, 0.01, 0.02], "rng": [0.10, 0.20, 0.50]}


def key(mode, tau):
    return f"rstar_{mode}{tau:.3g}"


def measured_rstar(df, rng=None, n_boot=300):
    """r*(mode, tau) per group from a stratified sweep; bootstrap over runs gives 16-84% bands."""
    rows = []
    for g, d in df.groupby("group"):
        if g == "Full":
            continue
        rates = np.array(sorted(d.rate.unique()))
        runs = {r: d[d.rate == r].forces_mae.values for r in rates}

        def rstar_from(sample):
            med = np.array([np.median(sample[r]) for r in rates])
            fit = pav_decreasing(med)
            full, top = fit[-1], fit[0]
            out = {}
            for mode, tols in TOLS.items():
                for tau in tols:
                    thr = {"rel": (1 + tau) * full, "abs": full + tau, "rng": full + tau * (top - full)}[mode]
                    ok = np.flatnonzero(fit <= thr)
                    out[key(mode, tau)] = float(rates[ok[0]]) if len(ok) else 1.0
            return out, full, med

        pt, full, med = rstar_from(runs)
        row = {"group": canon(g), "mae_full": full, "mae_1pct": float(med[0]), "n_runs_min": int(min(len(v) for v in runs.values())), **pt}
        if rng is not None:
            boots = {k: [] for k in pt}
            for _ in range(n_boot):
                b, _, _ = rstar_from({r: rng.choice(v, len(v)) for r, v in runs.items()})
                for k in pt:
                    boots[k].append(b[k])
            for k in pt:
                lo, hi = np.percentile(boots[k], [16, 84])
                row[k + "_lo"], row[k + "_hi"] = min(lo, pt[k]), max(hi, pt[k])
        for r in (0.05, 0.1, 0.3):
            k = int(np.argmin(np.abs(rates - r)))
            row[f"excess_{r:.2f}"] = float(med[k] / full - 1)
            row[f"dmae_{r:.2f}"] = float(med[k] - full)
        rows.append(row)
    return pd.DataFrame(rows).set_index("group").loc[ORDER]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/calibration")
    args = ap.parse_args()
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pairs = [(m, t) for m, ts in TOLS.items() for t in ts]

    # ---- measured
    sw = pd.read_csv(REPO / "results/paper/stratified_adaptive_test_mae.csv")
    sw["rate"] = sw.rate.astype(float).round(3)
    meas = measured_rstar(sw, rng=np.random.default_rng(0))
    meas.to_csv(out / "carbon_measured.csv")

    # ---- scores
    curves = pd.read_csv(REPO / "results/funiq/groups_s0/funiq_curves.csv")
    curves = curves[curves.subsample == 1.0]
    h_grid = np.array(sorted(curves.h.unique()))
    F = {}  # group -> f_uniq on grid
    for g, d in curves.groupby("scope"):
        d = d.sort_values("h")
        F[g] = np.interp(np.log(h_grid), np.log(d.h.values), d.f_uniq.values)

    def f_at(g, h):
        return float(np.interp(np.log(h), np.log(h_grid), F[g]))

    elb = pd.read_csv(REPO / "results/funiq/groups_s0/elbows_logh.csv").set_index("scope")
    paper = {k: v for k, v in json.load(open(REPO / "results/funiq/paper/budget_paper_fig3.json")).items() if not k.startswith("_")}
    rad = pd.read_csv(REPO / "results/stats_train/radius_vs_elbow.csv").set_index("scope")
    td = pd.read_csv(REPO / "results/test_distance/summary.csv")
    dnn = {canon(r.test_group): r.d_train_train_p50 for r in td.itertuples()}

    scores = pd.DataFrame(index=ORDER)
    scores["fixed_h0.015"] = [f_at(g, 0.015) for g in ORDER]
    scores["elbow_midlo"] = [elb.loc[g, "mid_lo_f"] for g in ORDER]
    scores["elbow_inflect"] = [elb.loc[g, "inflect_f"] for g in ORDER]
    scores["elbow_kappa_lo"] = [elb.loc[g, "kappa_lo_f"] for g in ORDER]
    scores["paper_fig3"] = [paper[g] for g in ORDER]
    scores["radius_knee"] = [rad.loc[g, "radius_knee_f"] for g in ORDER]
    scores["h_elbow_midlo"] = [elb.loc[g, "mid_lo_h"] for g in ORDER]
    scores["d_nn_p50"] = [dnn[g] for g in ORDER]
    c_grid = np.exp(np.linspace(np.log(0.25), np.log(16), 25))
    for c in c_grid:
        scores[f"scale_c{c:.3g}"] = [f_at(g, c * dnn[g]) for g in ORDER]
    for h in h_grid:
        scores[f"fixed_h{h:.4g}"] = [f_at(g, h) for g in ORDER]
    scores.to_csv(out / "carbon_scores.csv")

    # ---- calibration test
    def evaluate(col, mt):
        mode, tau = mt
        x = scores[col].values
        y = meas[key(mode, tau)].values
        rho, p = spearmanr(x, y)
        r, _ = pearsonr(x, y)
        return {"score": col, "mode": mode, "tau": tau, "spearman": rho, "spearman_p": p, "pearson": r,
                "rms_error": float(np.sqrt(np.mean((x - y) ** 2))), "mean_score": float(np.mean(x)), "mean_rstar": float(np.mean(y)),
                "score_spread": float(np.std(x))}

    named = ["fixed_h0.015", "elbow_midlo", "elbow_inflect", "elbow_kappa_lo", "paper_fig3", "radius_knee"]
    summ = pd.DataFrame([evaluate(c, mt) for c in named for mt in pairs])
    sweeps = pd.DataFrame([{**evaluate(f"fixed_h{h:.4g}", mt), "rule": "fixed", "param": h} for h in h_grid for mt in pairs]
                          + [{**evaluate(f"scale_c{c:.3g}", mt), "rule": "scale", "param": c} for c in c_grid for mt in pairs])
    best = []
    for rule in ("fixed", "scale"):
        for m, t in pairs:
            s = sweeps[(sweeps.rule == rule) & (sweeps["mode"] == m) & (sweeps.tau == t)]
            b = s.loc[s.spearman.idxmax()]
            best.append({**b.to_dict(), "score": f"best_{rule} (param={b.param:.3g})"})
    summ = pd.concat([summ, pd.DataFrame(best)[summ.columns]], ignore_index=True)
    summ.to_csv(out / "carbon_summary.csv", index=False)
    sweeps.to_csv(out / "carbon_sweeps.csv", index=False)

    # ---- print
    pd.set_option("display.width", 200)
    print("== measured prunable fraction r*(tau) per carbon state point (stratified sweep, paper runs)")
    for m in TOLS:
        print(f"-- {m}: {MODES[m]}")
        print(meas[[key(m, t) for t in TOLS[m]]].round(3).to_string())
    print(meas[["mae_full", "mae_1pct", "dmae_0.05", "excess_0.05"]].round(3).to_string())
    print("\n== scores per group")
    print(scores[named].round(3).to_string())
    print("\n== across 12 groups: Spearman rho(score, r*), RMS(score - r*)")
    for m, t in pairs:
        print(f"-- {m} tau = {t:g}")
        for _, x in summ[(summ["mode"] == m) & (summ.tau == t)].iterrows():
            print(f"  {x.score:32s} rho={x.spearman:+.2f} (p={x.spearman_p:.2f})  pearson={x.pearson:+.2f}  rms={x.rms_error:.3f}  "
                  f"mean score {x.mean_score:.2f} vs mean r* {x.mean_rstar:.2f}  spread {x.score_spread:.3f}")

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                                "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                                "axes.edgecolor": "#8a8984", "grid.color": "#e6e5e1", "grid.linewidth": 0.5})
    COL = {"fixed_h0.015": "#2a78d6", "elbow_midlo": "#e8643a", "scale": "#1fa37a", "fixed": "#2a78d6"}
    fig, AX = plt.subplots(3, 3, figsize=(10.5, 9.6))
    for row, (mode, tau) in enumerate([("rel", 0.05), ("abs", 0.01), ("rng", 0.2)]):
        axes = AX[row]
        k = key(mode, tau)
        y = meas[k].values
        ylo, yhi = meas[k + "_lo"].values, meas[k + "_hi"].values
        bf = summ[(summ["mode"] == mode) & (summ.tau == tau) & summ.score.str.startswith("best_scale")].iloc[0]
        cbest = float(bf.score.split("param=")[1].rstrip(")"))
        ax = axes[0]
        for col, lab, c, mk in [("fixed_h0.015", "fixed h = 0.015", COL["fixed_h0.015"], "o"), ("elbow_midlo", "elbow h*", COL["elbow_midlo"], "s"),
                                (f"scale_c{cbest:.3g}", f"scale rule, c = {cbest:.2g}", COL["scale"], "^")]:
            ax.errorbar(scores[col].values, y, yerr=[np.clip(y - ylo, 0, None), np.clip(yhi - y, 0, None)], fmt=mk, color=c, ecolor=c,
                        elinewidth=0.8, capsize=0, ms=5, markeredgecolor="white", markeredgewidth=0.5, label=lab, zorder=3)
        ax.plot([0, 1], [0, 1], color="#8a8984", lw=0.8, ls="--", zorder=1)
        ax.set_xlabel("efficiency score f_uniq (predicted unique fraction)")
        unit = {"rel": f"MAE within {tau:.0%} of full", "abs": f"MAE within {tau} eV/A of full", "rng": f"MAE within {tau:.0%} of its range"}[mode]
        ax.set_ylabel(f"measured prunable fraction r*\n({unit})")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.grid(True, zorder=0); ax.legend(frameon=False, loc="upper left")
        ax.set_title(f"({'abc'[row]}1) {mode}: score vs. measurement, 12 carbon state points")
        for rule, ax, xl in (("fixed", axes[1], "fixed bandwidth h"), ("scale", axes[2], "c in h = c x median NN distance")):
            for t, ls in zip(TOLS[mode], (":", "-", "--")):
                sv = sweeps[(sweeps.rule == rule) & (sweeps["mode"] == mode) & (sweeps.tau == t)]
                ax.plot(sv.param, sv.spearman, color=COL[rule], lw=1.5, ls=ls, label=f"tau = {t:g}")
            ax.set_xscale("log"); ax.axhline(0, color="#8a8984", lw=0.8); ax.set_ylim(-1, 1); ax.grid(True)
            ax.set_xlabel(xl); ax.set_ylabel("Spearman rho(score, r*)")
            if rule == "fixed":
                ax.axvline(0.015, color="#52514e", lw=0.8, ls=":"); ax.text(0.0155, -0.95, "default", fontsize=7, color="#52514e")
            ax.legend(frameon=False, loc="lower right")
        axes[1].set_title(f"({'abc'[row]}2) one fixed bandwidth for every group")
        axes[2].set_title(f"({'abc'[row]}3) bandwidth scaled to each group")
    fig.tight_layout()
    (REPO / "figures/out").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPO / "figures/out/calibration_carbon.png", dpi=300)
    fig.savefig(REPO / "figures/out/calibration_carbon.pdf")
    print("\nwrote", out, "and figures/out/calibration_carbon.png")


if __name__ == "__main__":
    main()
