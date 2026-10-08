"""Reconstruct state-point group labels for the original ChIMES C-2.0 Small training set.

The original xyz carries no thermodynamic labels (its ``config_tag`` is a file
path).  The QUESTS-descriptor file used for the paper (DS_2km6t3p8hxx2_0.xyz)
carries ``config_tag`` labels such as ``HD liquid_2.0gcc_6000``.  Match frames
between the two files by geometry (natoms, cell, positions) and write
``group_labels.csv`` with one row per original frame.

Known data quirks handled here (see data/README.md):
  * Both files contain exact duplicate frames (consecutive pairs).  Each
    duplicate is marked with ``dup_of_orig`` so downstream code can drop it.
  * In the labeled file the second copy of a duplicate pair sometimes carries
    the *next* group's label.  When a frame matches several labeled frames,
    the label whose nominal density agrees with the computed density is
    chosen; ties fall back to the first match.

Usage:
    python build_group_labels.py ORIGINAL.xyz LABELED.xyz OUT.csv
"""
import hashlib
import re
import sys

import numpy as np
from ase.io import read

AMU_PER_GCC_A3 = 0.6022  # 1 g/cc = 0.6022 amu/Å^3


def frame_key(atoms, decimals=3):
    cell = np.round(atoms.cell.array, decimals)
    pos = np.round(atoms.get_positions(), decimals)
    h = hashlib.sha1()
    h.update(cell.tobytes())
    h.update(pos.tobytes())
    return len(atoms), h.hexdigest()


def density(atoms):
    return atoms.get_masses().sum() / AMU_PER_GCC_A3 / atoms.get_volume()


def nominal_density(tag):
    m = re.search(r"([\d.]+)gcc", tag)
    return float(m.group(1)) if m else None


def pick_label(hits, rho):
    """hits: list of (labeled_index, tag). Prefer density-consistent label."""
    consistent = [
        h for h in hits
        if nominal_density(h[1]) and abs(nominal_density(h[1]) - rho) < 0.05
    ]
    return consistent[0] if consistent else hits[0]


def main(orig_path, labeled_path, out_path):
    orig = read(orig_path, ":")
    labeled = read(labeled_path, ":")

    label_by_key = {}
    for i, a in enumerate(labeled):
        label_by_key.setdefault(frame_key(a), []).append((i, a.info["config_tag"]))

    seen = {}
    rows, unmatched, ambiguous = [], [], []
    for i, a in enumerate(orig):
        key = frame_key(a)
        rho = density(a)
        dup_of = seen.setdefault(key, i)
        hits = label_by_key.get(key, [])
        if not hits:
            unmatched.append(i)
            j, tag = -1, "UNMATCHED"
        else:
            if len(hits) > 1:
                ambiguous.append(i)
            j, tag = pick_label(hits, rho)
        rows.append((i, j, len(a), rho, tag, dup_of if dup_of != i else -1))

    with open(out_path, "w") as f:
        f.write("orig_index,labeled_index,natoms,density_gcc,group,dup_of_orig\n")
        for i, j, n, rho, tag, dup in rows:
            f.write(f"{i},{j},{n},{rho:.3f},{tag},{dup}\n")

    n_dup = sum(1 for r in rows if r[5] >= 0)
    print(f"original frames: {len(orig)}  labeled frames: {len(labeled)}")
    print(f"unmatched: {len(unmatched)}  ambiguous (resolved by density): {len(ambiguous)}")
    print(f"duplicate frames in original: {n_dup}  -> unique frames: {len(orig) - n_dup}")
    for i in unmatched:
        print(f"  unmatched orig frame {i}: natoms={len(orig[i])} rho={rows[i][3]:.3f}")
    print("duplicate pairs (orig_index -> dup_of_orig, label):")
    for i, j, n, rho, tag, dup in rows:
        if dup >= 0:
            print(f"  {i:4d} -> {dup:4d}  {tag}  rho={rho:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
