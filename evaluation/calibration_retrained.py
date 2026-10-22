"""Calibration of QUESTS efficiency scores on the retrained models, all three datasets.

Same test as evaluation/calibration.py, but the measured prunable fraction comes
from the retrained stratified models (relative early stopping) instead of the
paper's dense carbon sweep, so it also covers SiO2 and MoNbTaVW.  Retention
levels available per group: 5, 10, 20% (E1 stratified_s0 + E2 stratified_s1,
up to 6 runs), the matched fraction (E5 stratified_s0) and 100% (E3).  The
grid is coarse, so r* is quantised to those levels; ties are expected.

measured  r*(mode, tau): smallest retention whose median test MAE is within the
          tolerance of the 100% model (modes as in calibration.py: rel, abs, rng;
          "rng" uses the 5% level as the top of the range).
scores    f_uniq of each group at h = 0.015, at the elbow (mid_lo and inflect),
          at every fixed h on the grid, and at h = c * median NN distance.

Outputs results/calibration/retrained_{dataset}_{measured,scores}.csv and
results/calibration/retrained_summary.csv; prints Spearman rho and RMS error.

Usage:  python evaluation/calibration_retrained.py [--eval-dir results/eval/relstop]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from calibration import pav_decreasing

REPO = Path(__file__).resolve().parent.parent
TOLS = {"rel": [0.02, 0.05, 0.10], "abs": [0.005, 0.01, 0.02], "rng": [0.10, 0.20, 0.50]}
PATHS = {
    "carbon": ("results/funiq/groups_s0", "results/test_distance/summary.csv"),
    "sio2": ("results/ext/sio2/funiq/groups_s0", "results/ext/sio2/test_distance/summary.csv"),
    "monbtavw": ("results/ext/monbtavw/funiq/groups_s0", "results/ext/monbtavw/test_distance/summary.csv"),
}


def canon(t):
    t = t.replace("test_", "").replace("HD_liquid", "HD liquid").replace("LD_liquid", "LD liquid")
    return "LD liquid_1.0gcc_2000" if t == "LD liquid_0.5gcc_2000" else t


def measured(ev):
    """Per group: median MAE per retention level of the stratified series, then r*."""
    s = ev[((ev.exp == "E1_method") & (ev.selection == "stratified_s0")) | ((ev.exp == "E2_fpsseed") & (ev.selection == "stratified_s1"))
           | ((ev.exp == "E5_matched") & (ev.selection == "stratified_s0")) | (ev.exp == "E3_full")].copy()
    s.loc[s.exp == "E3_full", "rate"] = 1.0
    rows = []
    for g, d in s.groupby("group"):
        if g == "Full":
            continue
        rates = np.array(sorted(d.rate.unique()))
        med = np.array([d[d.rate == r].forces_mae.median() for r in rates])
        n = [int((d.rate == r).sum()) for r in rates]
        fit = pav_decreasing(med)
        full, top = fit[-1], fit[0]
        row = {"group": canon(g), "mae_full": full, "mae_top": top, "levels": " ".join(f"{r:g}" for r in rates), "n_runs": " ".join(map(str, n))}
        for mode, tols in TOLS.items():
            for tau in tols:
                thr = {"rel": (1 + tau) * full, "abs": full + tau, "rng": full + tau * (top - full)}[mode]
                ok = np.flatnonzero(fit <= thr)
                row[f"rstar_{mode}{tau:g}"] = float(rates[ok[0]]) if len(ok) else 1.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def scores_for(ds, groups):
    fdir, tdist = PATHS[ds]
    curves = pd.read_csv(REPO / fdir / "funiq_curves.csv")
    curves = curves[curves.subsample == 1.0]
    h_grid = np.array(sorted(curves.h.unique()))
    F = {g: np.interp(np.log(h_grid), np.log(d.sort_values("h").h.values), d.sort_values("h").f_uniq.values) for g, d in curves.groupby("scope")}
    f_at = lambda g, h: float(np.interp(np.log(h), np.log(h_grid), F[g]))
    elb = pd.read_csv(REPO / fdir / "elbows_logh.csv").set_index("scope")
    td = pd.read_csv(REPO / tdist)
    dnn = {canon(r.test_group): r.d_train_train_p50 for r in td.itertuples()}
    sc = pd.DataFrame(index=groups)
    sc["fixed_h0.015"] = [f_at(g, 0.015) for g in groups]
    sc["elbow_midlo"] = [elb.loc[g, "mid_lo_f"] for g in groups]
    sc["elbow_inflect"] = [elb.loc[g, "inflect_f"] for g in groups]
    sc["d_nn_p50"] = [dnn.get(g, np.nan) for g in groups]
    c_grid = np.exp(np.linspace(np.log(0.25), np.log(16), 25))
    for c in c_grid:
        sc[f"scale_c{c:.3g}"] = [f_at(g, c * dnn[g]) if np.isfinite(dnn.get(g, np.nan)) and dnn[g] > 0 else np.nan for g in groups]
    for h in h_grid:
        sc[f"fixed_h{h:.4g}"] = [f_at(g, h) for g in groups]
    return sc, h_grid, c_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="results/eval/relstop")
    ap.add_argument("--out-dir", default="results/calibration")
    args = ap.parse_args()
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pairs = [(m, t) for m, ts in TOLS.items() for t in ts]
    summ = []
    for ds in PATHS:
        p = REPO / args.eval_dir / f"{ds}_test_mae.csv"
        if not p.exists():
            print(f"{ds}: no evaluation at {p}, skipping")
            continue
        ev = pd.read_csv(p)
        ev["rate"] = pd.to_numeric(ev.rate, errors="coerce").round(3)
        meas = measured(ev)
        sc, h_grid, c_grid = scores_for(ds, list(meas.index))
        meas.to_csv(out / f"retrained_{ds}_measured.csv")
        sc.to_csv(out / f"retrained_{ds}_scores.csv")

        def ev_one(col, mode, tau):
            x, y = sc[col].values, meas[f"rstar_{mode}{tau:g}"].values
            ok = np.isfinite(x)
            rho = spearmanr(x[ok], y[ok])[0] if ok.sum() > 3 else np.nan
            return rho, float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2))), float(np.mean(x[ok])), float(np.mean(y[ok])), float(np.std(x[ok]))

        for mode, tau in pairs:
            for col in ("fixed_h0.015", "elbow_midlo", "elbow_inflect"):
                rho, rms, ms, mr, spr = ev_one(col, mode, tau)
                summ.append(dict(dataset=ds, mode=mode, tau=tau, score=col, spearman=rho, rms=rms, mean_score=ms, mean_rstar=mr, spread=spr))
            for rule, params in (("fixed", h_grid), ("scale", c_grid)):
                best = max(((ev_one(f"{rule}_h{v:.4g}" if rule == "fixed" else f"scale_c{v:.3g}", mode, tau), v) for v in params),
                           key=lambda z: -np.inf if np.isnan(z[0][0]) else z[0][0])
                (rho, rms, ms, mr, spr), v = best
                summ.append(dict(dataset=ds, mode=mode, tau=tau, score=f"best_{rule} ({v:.3g})", spearman=rho, rms=rms, mean_score=ms, mean_rstar=mr, spread=spr))
        print(f"\n== {ds}: {len(meas)} groups; retention levels per group: {meas.levels.iloc[0]} (runs {meas.n_runs.iloc[0]})")
        print(meas[[f"rstar_abs{t:g}" for t in TOLS["abs"]] + [f"rstar_rng{t:g}" for t in TOLS["rng"]] + ["mae_full"]].round(3).to_string())
    S = pd.DataFrame(summ)
    S.to_csv(out / "retrained_summary.csv", index=False)
    pd.set_option("display.width", 200)
    for ds in S.dataset.unique():
        print(f"\n== {ds}: Spearman rho(score, r*) / RMS(score - r*)")
        t = S[S.dataset == ds].pivot_table(index="score", columns=["mode", "tau"], values="spearman")
        print(t.round(2).to_string())
        print("mean score vs mean r*:")
        print(S[(S.dataset == ds) & (S["mode"] == "abs")].pivot_table(index="score", columns="tau", values=["mean_score", "mean_rstar", "rms"]).round(2).to_string())
    print(f"\nwrote {out}/retrained_*.csv")


if __name__ == "__main__":
    main()
