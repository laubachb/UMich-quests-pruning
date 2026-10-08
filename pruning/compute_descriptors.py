"""Attach QUESTS descriptors to an xyz so the pruning scripts can consume it.

Single-element data uses the standard QUESTS descriptor (sorted neighbour
distances, 2k-1 = 63 dims for k=32); multi-element data uses QUESTS's
multicomponent variant, which appends one species-resolved block per element
((n_species+1) * (2k-1) dims).  The result is written to ``--out`` with the
per-atom array key ``quests_descriptor_descriptors`` -- the same key the
ChIMES carbon file uses -- so fps.py, funiq_sweep.py, etc. work unchanged.

Usage:
    python pruning/compute_descriptors.py data/external/monbtavw/train.xyz \
        --out data/external/monbtavw/train_desc.xyz --species Mo Nb Ta V W
"""
import argparse
import time

import numpy as np
from ase.io import read, write
from quests.descriptor import get_descriptors, get_descriptors_multicomponent

KEY = "quests_descriptor_descriptors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--species", nargs="*", default=None,
                    help="element list for the multicomponent descriptor; omit for single-element")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    t0 = time.time()
    frames = read(args.xyz, ":")
    natoms = np.array([len(a) for a in frames])
    print(f"{len(frames)} frames, {natoms.sum()} atoms ({time.time() - t0:.0f}s)")
    t1 = time.time()
    if args.species:
        X = get_descriptors_multicomponent(frames, k=args.k, cutoff=args.cutoff,
                                           species=list(args.species), dtype=args.dtype)
    else:
        X = get_descriptors(frames, k=args.k, cutoff=args.cutoff, dtype=args.dtype)
    print(f"descriptors {X.shape} ({time.time() - t1:.0f}s)")
    assert len(X) == natoms.sum()
    off = np.concatenate([[0], np.cumsum(natoms)])
    for i, a in enumerate(frames):
        a.arrays[KEY] = X[off[i]:off[i + 1]]
        a.info["quests_k"] = args.k
        a.info["quests_cutoff"] = args.cutoff
        a.info["quests_species"] = ",".join(args.species) if args.species else "single"
    write(args.out, frames)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
