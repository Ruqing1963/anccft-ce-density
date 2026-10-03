r"""Task D1 at m = 10 on a sample of blocks: all rho-orbit representatives with dim <= MAXD.
For each block: excess of sigma' (taskC1_m10.pkl), excess of tau = y2.2*y1.0 + y2.0*y1.1 + y2.1*y1.2 (F2, rho-invariant),
and the generic excess (best of random F16 elements, then F256 for blocks not reaching maximal rank).
usage: python taskD1_m10.py MAXD"""
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

from cengine import Split, Blocks
from taskD_common import DeltaSpace, block_rank_F2, block_rank_F2k, set_basis_words, orbit_rep

POLYS = {4: 0b10011, 8: 0b100011011}
m = 10
MAXD = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
t0 = time.time()
sp = Split(m)
B = Blocks(sp)
ds = DeltaSpace(sp, B)
set_basis_words(ds)
with open("taskC1_m10.pkl", "rb") as fh:
    C = pickle.load(fh)
tau = None
for v in range(1, 64):
    if ds.label(v) == "y2.2*y1.0 + y2.0*y1.1 + y2.1*y1.2":
        tau = v
assert tau is not None
rng = np.random.default_rng(10)
res = {}
tot = defaultdict(int)
for g in sorted(B.blocks, key=lambda g: B.dim(g)):
    if orbit_rep(g) != g or B.dim(g) > MAXD:
        continue
    h = (g[0] + 1, g[1] + 1, g[2] + 1)
    d, dh = B.dim(g), B.dim(h)
    if dh == 0:
        continue
    target = min(d, dh)
    ex_sig = target - (d - C["ker"][g])
    ex_tau = target - block_rank_F2(sp, B, g, ds.words_of(tau))
    best = -1
    mats = {}
    for k, ntr in ((4, 2), (8, 3)):
        for _ in range(ntr):
            coeffs = [int(c) for c in rng.integers(1, 1 << k, 6)]
            best = max(best, block_rank_F2k(sp, B, g, coeffs, k, POLYS[k], mats))
            if best == target:
                break
        if best == target:
            break
    ex_gen = target - best
    res[g] = (d, dh, ex_sig, ex_tau, ex_gen)
    tot["blocks"] += 3
    tot["dim"] += 3 * d
    tot["sig"] += 3 * ex_sig
    tot["tau"] += 3 * ex_tau
    tot["gen"] += 3 * ex_gen
    if ex_gen:
        print(f"   generic excess {ex_gen} on block {g} (n={sum(g)}, {d} -> {dh}); sigma' {ex_sig}, tau {ex_tau}", flush=True)
print(f"m=10, blocks with dim <= {MAXD} (all rotations counted): {tot['blocks']} blocks, total dim {tot['dim']} "
      f"({tot['dim'] / (1 << sp.nl):.3f} of |G|); excess: sigma' {tot['sig']}, tau {tot['tau']}, generic {tot['gen']}"
      f"  [{time.time() - t0:.0f}s]")
with open("taskD1_m10_sample.pkl", "wb") as fh:
    pickle.dump(res, fh)
