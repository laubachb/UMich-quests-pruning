"""Per-group force MAE for compiled NequIP models on the held-out test set.

Replaces the original compute_errors_*.py + unit-correction step.  Reference
forces in random_C.xyz are in Hartree/Bohr and are converted to eV/A here
(the paper's *_with_full_test.csv files were produced WITHOUT this conversion;
only the *_corrected_results.csv files are in eV/A).

Output CSV has the same columns as results/paper/*_test_mae.csv
(group, rate, forces_mae, train_file, ckpt_file, compiled_model) plus
exp, selection, train_seed, natoms_test, so old and new results concatenate.

Usage (GPU or CPU; CPU is fine, ~1 min per model):
    python evaluation/evaluate_models.py --test random_C.xyz \
        --matrix training/experiments.csv --out results/eval/new_models_test_mae.csv
    python evaluation/evaluate_models.py --test random_C.xyz \
        --models path/a.nequip.pth path/b.nequip.pth --out results/eval/adhoc.csv
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
from ase.io import read
from nequip.integrations.ase import NequIPCalculator

HARTREE_BOHR_TO_EV_A = 27.211386245988 * 1.8897261246257702  # 51.42208...


def load_test(path, convert):
    frames = read(path, ":")
    groups = np.array([a.info["config_tag"] for a in frames])
    natoms = np.array([len(a) for a in frames])
    f_true = np.concatenate([a.get_forces() for a in frames])
    if convert:
        f_true = f_true * HARTREE_BOHR_TO_EV_A
    env_group = np.repeat(groups, natoms)
    return frames, env_group, f_true


def predict(frames, compiled, device):
    calc = NequIPCalculator.from_compiled_model(compile_path=str(compiled), device=device,
                                                species_to_type_name={})
    out = []
    for a in frames:
        a.calc = calc
        out.append(a.get_forces())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--matrix", help="training/experiments.csv (evaluates each out_dir/compiled_best.nequip.pth)")
    ap.add_argument("--models", nargs="*", help="explicit compiled model paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-unit-conversion", action="store_true",
                    help="set if the test file's forces are already in eV/A")
    args = ap.parse_args()

    frames, env_group, f_true = load_test(args.test, not args.no_unit_conversion)
    groups = sorted(set(env_group))
    print(f"test set: {len(frames)} frames, {len(f_true)} atoms, {len(groups)} groups")

    jobs = []  # (compiled_path, meta dict)
    if args.matrix:
        for r in csv.DictReader(open(args.matrix)):
            p = Path(r["out_dir"]) / "compiled_best.nequip.pth"
            if p.exists():
                jobs.append((p, {"exp": r["exp"], "selection": r["selection"], "rate": r["fraction"],
                                 "train_seed": r["train_seed"], "train_file": r["xyz"],
                                 "ckpt_file": str(Path(r["out_dir"]) / "train_dir/best.ckpt")}))
            else:
                print(f"  missing (not trained yet?): {p}")
    for m in args.models or []:
        rate = re.search(r"_(\d\.\d{3})_", m)
        jobs.append((Path(m), {"exp": "adhoc", "selection": "", "rate": rate.group(1) if rate else "",
                               "train_seed": "", "train_file": "", "ckpt_file": ""}))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True) if out.suffix == "" else out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["group", "rate", "forces_mae", "train_file", "ckpt_file", "compiled_model",
              "exp", "selection", "train_seed", "natoms_test"]
    write_header = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            w.writeheader()
        for p, meta in jobs:
            f_pred = predict(frames, p, args.device)
            err = np.abs(f_pred - f_true)
            rows = []
            for g in groups + ["Full"]:
                m = np.ones(len(err), bool) if g == "Full" else (env_group == g)
                rows.append({"group": g, "forces_mae": float(err[m].mean()), "compiled_model": str(p),
                             "natoms_test": int(m.sum()), **meta})
            w.writerows(rows)
            fh.flush()
            full = rows[-1]["forces_mae"]
            print(f"  {p.parent.name:30s} rate={meta['rate']:<6} seed={meta['train_seed']:<3} Full MAE={full:.4f} eV/A")


if __name__ == "__main__":
    main()
