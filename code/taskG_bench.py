"""Task G micro-benchmarks: level-10 echelon of a large block, and straightening throughput at level 13."""
import os
import sys
import time

import numpy as np

from taskE_lift import Level, image_solver, products, DELTA, add
from gf2 import gf2_rank_inplace
from taskC2_low import build_block_sorted

lev = Level(10)
t = time.time()
B = lev.blocks(57)
print("enum deg 57", time.time() - t, flush=True)
gs = sorted(B, key=lambda g: -B[g].size)
which = int(sys.argv[1]) if len(sys.argv) > 1 else 3
g = gs[which]
tau = add(g, DELTA)
print("block", g, B[g].size, "target", lev.block(tau).size, flush=True)
t = time.time()
rows, cols, P = lev.matrix(g)
print("build", time.time() - t, flush=True)
t = time.time()
rk = gf2_rank_inplace(P.copy(), cols.size)
print("rank", rk, time.time() - t, flush=True)
t = time.time()
cols, src, Wc, Bm, pc = image_solver(lev, tau)
print("solver echelon", Bm.shape, time.time() - t, flush=True)

hi = Level(13)
t = time.time()
Bh = hi.blocks(60)
print("enum level13 deg60", sum(b.size for b in Bh.values()), time.time() - t, flush=True)
g13 = max(Bh, key=lambda g: Bh[g].size)
monos = Bh[g13][:200000].copy()
starts = np.arange(monos.size + 1, dtype=np.int64)
products(monos[:1000], starts[:1001], hi.words, hi.wlen, hi.sp.br, hi.sp.sq)
t = time.time()
outs = products(monos, starts, hi.words, hi.wlen, hi.sp.br, hi.sp.sq)
dt = time.time() - t
print("straighten", monos.size, "monos in", dt, "s; avg terms", np.mean([o.size for o in outs]), flush=True)
