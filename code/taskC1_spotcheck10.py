"""Spot check of the rotation-representative shortcut used for m = 10 (--reps): compute some NON-representative
blocks directly and compare with the stored ranks (taskC1_m10.pkl)."""
import pickle
import time

import numpy as np

from cengine import Split, Blocks, build_block, pack_words
from gf2 import gf2_rank_inplace
from taskC1_blocks import orbit_rep

r = pickle.load(open("taskC1_m10.pkl", "rb"))
sp = Split(10)
B = Blocks(sp)
words, wlen = pack_words(sp.sigma_words())
cands = [g for g in r["rank"] if orbit_rep(g) != g and r["ker"][g] > 0 and 2000 < r["dims"][g] < 40000]
cands.sort(key=lambda g: r["dims"][g])
pick = cands[::max(1, len(cands) // 8)][:8]
for g in pick:
    h = tuple(a + 1 for a in g)
    t = time.time()
    P, bad = build_block(B.blocks[g], words, wlen, sp.br, sp.sq, B.local, B.key, np.int32(B.gkey(h)), B.dim(h), 64)
    assert bad == 0
    rk = int(gf2_rank_inplace(P, B.dim(h)))
    print(f"m=10 block {g}: dim {B.dim(g)}, direct rank {rk}, stored (from rep {orbit_rep(g)}) {r['rank'][g]}  "
          f"{'OK' if rk == r['rank'][g] else 'MISMATCH'}  ({time.time() - t:.1f}s)", flush=True)
