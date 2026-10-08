"""Write the reviewer-response experiment matrix and the commands that realise it.

Dataset-aware: ``--dataset carbon`` (default) uses the ChIMES layout
(results/fps, results/pruned, results/models); any other name uses
results/ext/<name>/{fps,pruned,models} as produced by scripts/external_pipeline.sh
and passes ``--species`` through to train_one.sh (NequIP chemical_symbols).

Produces (suffix "" for carbon, "_<name>" otherwise)
    training/experiments<sfx>.csv         one row per model
    training/make_training_sets<sfx>.sh   make_masks.py calls generating every pruned xyz needed
    training/jobs<sfx>.txt                one `train_one.sh` command per line (job array)

Experiments (default settings; override with flags):
    E1 method   global_s0 / stratified_s0 / random_global_s0 / random_stratified_s0
                x fractions {0.05, 0.10, 0.20} x train seeds   -> R2 #2, R4 #3
    E2 fpsseed  global_s1 / stratified_s1 x same fractions x seeds -> R1 #4
    E3 full     100% x train seeds                                -> reference
    E4 adaptive per-group elbow-derived fractions (requires --budget JSON from
                pruning/elbow.py; skipped with a note if absent)   -> R2 #2
    E5 matched  global_s0 / stratified_s0 at ONE fraction equal to E4's overall
                retention (--matched-fraction)                     -> R2 crux

Usage:
    python training/make_experiment_matrix.py --data DATA.xyz [--seeds 0 1 2]
    python training/make_experiment_matrix.py --dataset monbtavw --data data/external/monbtavw/train_desc.xyz --species Mo Nb Ta V W
"""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="labeled training xyz (with descriptors)")
    ap.add_argument("--dataset", default="carbon")
    ap.add_argument("--species", nargs="*", default=None, help="chemical symbols for NequIP (external datasets)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--budget", default=None, help="budget.json with per-group fractions for E4")
    ap.add_argument("--matched-fraction", type=float, default=None,
                    help="E5: global_s0 and stratified_s0 at this single fraction (the overall retention of the "
                         "E4 budget), isolating per-group allocation from retention level")
    args = ap.parse_args()

    carbon = args.dataset == "carbon"
    sfx = "" if carbon else f"_{args.dataset}"
    root = "results" if carbon else f"results/ext/{args.dataset}"
    fps_root, pruned_root, model_root = f"{root}/fps", f"{root}/pruned", f"{root}/models"
    stem = Path(args.data).stem
    species_arg = (" " + ",".join(args.species)) if args.species else ""

    def pruned_name(sel, tag):
        meta = json.load(open(REPO / fps_root / sel / "fps_meta.json"))
        return f"{stem}_{tag}_{meta['mode']}_s{meta['fps_seed']}.xyz"

    rows, mask_cmds = [], {}
    for exp, sels in (("E1_method", ["global_s0", "stratified_s0", "random_global_s0", "random_stratified_s0"]),
                      ("E2_fpsseed", ["global_s1", "stratified_s1"])):
        for sel in sels:
            if not (REPO / fps_root / sel / "fps_meta.json").exists():
                print(f"WARNING: {fps_root}/{sel} missing, skipping")
                continue
            mask_cmds.setdefault(sel, set()).update(args.fractions)
            for f in args.fractions:
                for s in args.seeds:
                    rows.append((exp, sel, f, s, f"{pruned_root}/{sel}/{pruned_name(sel, f'{f:.3f}')}",
                                 f"{model_root}/{exp}/{sel}/f{f:.3f}/seed{s}"))
    for s in args.seeds:
        rows.append(("E3_full", "none", 1.0, s, args.data, f"{model_root}/E3_full/seed{s}"))
    if args.matched_fraction:
        for sel in ("global_s0", "stratified_s0"):
            mask_cmds.setdefault(sel, set()).add(args.matched_fraction)
            for s in args.seeds:
                rows.append(("E5_matched", sel, args.matched_fraction, s,
                             f"{pruned_root}/{sel}/{pruned_name(sel, f'{args.matched_fraction:.3f}')}",
                             f"{model_root}/E5_matched/{sel}/f{args.matched_fraction:.3f}/seed{s}"))
    budget = args.budget and Path(args.budget).exists()
    if budget:
        for s in args.seeds:
            rows.append(("E4_adaptive", "stratified_s0", -1, s,
                         f"{pruned_root}/adaptive_s0/{pruned_name('stratified_s0', 'adaptive')}",
                         f"{model_root}/E4_adaptive/seed{s}"))
    else:
        print("NOTE: E4_adaptive skipped (no --budget file yet)")

    with open(REPO / f"training/experiments{sfx}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["exp", "selection", "fraction", "train_seed", "xyz", "out_dir", "dataset", "species"])
        w.writerows([r + (args.dataset, ",".join(args.species or [])) for r in rows])

    with open(REPO / f"training/make_training_sets{sfx}.sh", "w") as fh:
        fh.write("#!/bin/bash\n# generated by make_experiment_matrix.py -- run from repo root\nset -e\n"
                 f"DATA=${{DATA:-{args.data}}}   # override: DATA=/path/on/this/machine.xyz bash $0\n")
        for sel, fracs in sorted(mask_cmds.items()):
            fl = " ".join(f"{f:.3f}" for f in sorted(fracs))
            fh.write(f"python pruning/make_masks.py $DATA {fps_root}/{sel} {pruned_root}/{sel} "
                     f"--fractions {fl} --drop-descriptors\n")
        if budget:
            fh.write(f"python pruning/make_masks.py $DATA {fps_root}/stratified_s0 {pruned_root}/adaptive_s0 "
                     f"--budget {args.budget} --drop-descriptors\n")

    with open(REPO / f"training/jobs{sfx}.txt", "w") as fh:
        for exp, sel, f, s, xyz, out in rows:
            fh.write(f"training/train_one.sh {xyz} {s} {out} gpu{species_arg}\n")

    print(f"{len(rows)} models -> training/experiments{sfx}.csv, jobs{sfx}.txt, make_training_sets{sfx}.sh")
    by = {}
    for r in rows:
        by[r[0]] = by.get(r[0], 0) + 1
    for k, v in by.items():
        print(f"  {k:12s} {v:4d} models")


if __name__ == "__main__":
    main()
