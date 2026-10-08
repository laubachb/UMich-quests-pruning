import time, numpy as np
from ase.io import read
from quests.descriptor import get_descriptors_multicomponent
fr = read("data/external/monbtavw/train.xyz", ":")
by = {}
for a in fr: by.setdefault(a.info["config_tag"], []).append(a)
sp = ["Mo","Nb","Ta","V","W"]
get_descriptors_multicomponent(by["liquid"][:2], k=32, cutoff=5.0, species=sp)
for g, frs in sorted(by.items(), key=lambda kv: len(kv[1])):
    t = time.time(); worst = 0.0; wi = 0
    for i, a in enumerate(frs):
        t1 = time.time(); get_descriptors_multicomponent([a], k=32, cutoff=5.0, species=sp); dt = time.time() - t1
        if dt > worst: worst, wi = dt, i
        if time.time() - t > 180:
            print(f"  {g:32s} ABORT after {i+1} frames, {time.time()-t:.0f}s; worst frame {wi} {worst:.1f}s natoms={len(frs[wi])} cell={frs[wi].cell.lengths().round(1)}", flush=True); break
    else:
        print(f"  {g:32s} {len(frs):4d} frames {time.time()-t:6.1f}s  worst {worst:.2f}s (frame {wi}, natoms={len(frs[wi])})", flush=True)
