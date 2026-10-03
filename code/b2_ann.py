"""Task B2 (linear part): approximate annihilators of a FIXED lift x~ in A_N = F2[[P]]/I^N.
For b >= b0, d_b(N) = dim { y_b in u_b : exists y = y_b + (weight > b) with x~ y in I^N }  (side R: right annihilator)
(resp. y x~ in I^N, side L).  d_b(N) is non-increasing in N; its limit is the dimension of the weight-b leading
terms of the true annihilator of x~ in F2[[P]] (compactness).  At N = a + b + 1 it equals the graded kernel dim.
Computed as d_b = n_b - (rank M_{>=b} - rank M_{>=b+1}),  M_{>=b} = images of all basis elements of weights b..N-1-a.
usage: python b2_ann.py N"""
import random
import sys
import time

import numpy as np

from jtable import JTable
from jalg import run_big_stack
from gf2 import gf2_rank
from b2common import get_V, get_x3, unit_block


def ann_profile(T, x, a, bmin, bmax, side):
    """d_b for b = bmin..bmax (x of leading weight a)"""
    top = T.N - 1 - a
    ranks = {}
    for b in range(bmin, top + 2):
        if b > top:
            ranks[b] = 0
            continue
        cols = list(range(T.wstart[b], T.wstart[top + 1]))
        Y = unit_block(T, cols)
        img = T.apply_block(x, Y, "L" if side == "R" else "R")   # side R: y -> x y  (x acts from the left)
        ranks[b] = gf2_rank(img, len(cols))
    out = {}
    for b in range(bmin, min(bmax, top) + 1):
        nb = T.wstart[b + 1] - T.wstart[b]
        out[b] = nb - (ranks[b] - ranks[b + 1])
    return out


if __name__ == "__main__":
    N = int(sys.argv[1])

    def go():
        t0 = time.time()
        T = JTable(N)
        T.build_right()
        V = get_V(T)
        x3 = get_x3(T)
        rng = random.Random(5)

        def corr(ws):
            e = set()
            for w in ws:
                e ^= {m for m in T.U.monos(w) if rng.random() < 0.5}
            return e
        cases = [("v1", V[0], 4, 4, 9), ("v2", V[1], 4, 4, 9), ("v3", V[2], 4, 4, 9),
                 ("v1+c(5,6)", V[0] ^ corr((5, 6)), 4, 4, 9),
                 ("v1+c(5..9)", V[0] ^ corr((5, 6, 7, 8, 9)), 4, 4, 9),
                 ("x3", x3, 3, 12, 16),
                 ("x3+c(4,5,6)", x3 ^ corr((4, 5, 6)), 3, 12, 16),
                 ("rand4", {m for m in T.U.monos(4) if rng.random() < 0.5}, 4, 4, 9)]
        for name, X, a, bmin, bmax in cases:
            if bmin > N - 1 - a:
                continue
            x = T.vec(X)
            for side in ("R", "L"):
                t1 = time.time()
                prof = ann_profile(T, x, a, bmin, bmax, side)
                s = "  ".join(f"b={b}:{d}" for b, d in prof.items())
                print(f"N={N} {name:12s} side {side} ({'x~y' if side == 'R' else 'yx~'} in I^N): {s}   [{time.time() - t1:.0f}s]",
                      flush=True)
        print(f"total {time.time() - t0:.0f}s")
    run_big_stack(go)
