"""Recompute per-group test MAE from saved predictions with out-of-distribution frames excluded.

SiO2: three held-out frames (bulk_amo 122, quench 488, liquid 265 in data/external/sio2/test.xyz,
0-based) contain Si-Si contacts of 1.79, 2.05 and 2.12 A; the shortest Si-Si contact in the
training liquid/quench/amorphous frames is 2.08 A, so two lie outside the training range and the
third at its edge.  Every model extrapolates there (median-model frame MAE 2.7-4.6 eV/A against ~0.2 for the
other disordered frames, predicted |F| up to 180 eV/A against a reference maximum of 12-20), and
the size of the excursion depends on the training seed, so these three frames decide the group
MAE of liquid, quench and amorphous.  This script reports the shortest same-species contacts
that justify the exclusion and writes the MAE table without them, in the evaluate_models.py
format, so compare_experiments.py and calibration_retrained.py can read it unchanged.

Usage:
    python evaluation/recompute_excluding_frames.py --dataset sio2 --exclude 122 488 265 \
        --pred-dir $SCRATCH/UMich-quests-pruning-models/relstop/pred_forces \
        --matrix training/experiments_sio2.csv --out results/eval/relstop_sio2_excl/sio2_test_mae.csv
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

REPO = Path(__file__).resolve().parent.parent
TEST = {"sio2": ("data/external/sio2/test.xyz", "data/external/sio2/train.xyz", 1.0)}


def min_contacts(frames, pairs, cut=3.0):
    out = {p: np.inf for p in pairs}
    for a in frames:
        i, j, d = neighbor_list("ijd", a, cut)
        s = np.array(a.get_chemical_symbols())
        for p in pairs:
            m = (s[i] == p[0]) & (s[j] == p[1])
            if m.any():
                out[p] = min(out[p], float(d[m].min()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sio2")
    ap.add_argument("--exclude", type=int, nargs="+", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--models-dirname", default="models_relstop")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    test_p, train_p, unit = TEST[args.dataset]
    frames = read(REPO / test_p, ":")
    train = read(REPO / train_p, ":")
    pairs = [("Si", "Si"), ("O", "O"), ("Si", "O")]
    tr_dis = [a for a in train if a.info["config_tag"] in ("bulk_amo", "quench", "liquid")]
    print("shortest contacts, training disordered frames:", {"-".join(k): round(v, 2) for k, v in min_contacts(tr_dis, pairs).items()})
    for k in args.exclude:
        print(f"  excluded frame {k} ({frames[k].info['config_tag']}):", {"-".join(p): round(v, 2) for p, v in min_contacts([frames[k]], pairs).items()})

    natoms = np.array([len(a) for a in frames])
    keep_frame = np.ones(len(frames), bool)
    keep_frame[args.exclude] = False
    keep = np.repeat(keep_frame, natoms)
    g = np.concatenate([np.full(len(a), a.info["config_tag"]) for a in frames])
    F = np.concatenate([a.get_forces() for a in frames]) * unit
    groups = sorted(set(g))
    pdir = Path(args.pred_dir) / args.dataset
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["group", "rate", "forces_mae", "train_file", "ckpt_file", "compiled_model", "exp", "selection", "train_seed", "natoms_test"]
    n = 0
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in csv.DictReader(open(REPO / args.matrix)):
            od = re.sub(r"/models/", f"/{args.models_dirname}/", "/" + r["out_dir"])[1:]
            f = pdir / (od.replace("/", "__") + ".npy")
            if not f.exists():
                continue
            err = np.abs(np.load(f) - F)
            meta = {"exp": r["exp"], "selection": r["selection"], "rate": r["fraction"], "train_seed": r["train_seed"],
                    "train_file": r["xyz"], "ckpt_file": str(Path(od) / "train_dir/best.ckpt"), "compiled_model": str(Path(od) / "compiled_best.nequip.pth")}
            for grp in groups + ["Full"]:
                m = keep & (np.ones(len(g), bool) if grp == "Full" else (g == grp))
                w.writerow({"group": grp, "forces_mae": float(err[m].mean()), "natoms_test": int(m.sum()), **meta})
            n += 1
    print(f"{n} models, {(~keep_frame).sum()} frames ({(~keep).sum()} atoms) excluded -> {out}")


if __name__ == "__main__":
    main()
