"""Task 3b/3c: exhaustive homogeneous zero-divisor search in u(L).

For every nonzero x in u_a (all 2^dim(u_a) - 1 of them) and side in {L, R}: is y -> x y (resp. y x)
injective u_b -> u_{a+b}?  (A pair x y = 0 with deg x = a, deg y = b is detected both as a
non-injective L_x on u_b and as a non-injective R_y on u_a.)
usage: python task3b_exhaustive.py KIND "a:bmax,a:bmax,..."     KIND = F8 | split
"""
import sys
import time

import numpy as np

from ulie import ULie
from zdtools import basis_matrices, batch_search, elem_from_bits, elem_str
from gf2 import gf2_rank

if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "F8"
    plan = [tuple(map(int, p.split(":"))) for p in (sys.argv[2] if len(sys.argv) > 2 else "1:10,2:8,3:6").split(",")]
    D = max(a + bm for a, bm in plan)
    U = ULie(kind, Dl=D)
    t0 = time.time()
    print(f"model {kind}, D={D}, dims u_j: {[len(U.monos(j)) for j in range(D + 1)]}", flush=True)
    for a, bmax in plan:
        na = len(U.monos(a))
        for b in range(1, bmax + 1):
            for side in ("L", "R"):
                t = time.time()
                B, nc = basis_matrices(U, a, b, side)
                rows = B.shape[1]
                t1 = time.time()
                if (1 << na) <= 128:
                    bad, minr, ex = 0, rows, None
                    for g in range(1, 1 << na):
                        M = np.zeros_like(B[0])
                        for i in range(na):
                            if (g >> i) & 1:
                                M ^= B[i]
                        r = gf2_rank(M, nc)
                        minr = min(minr, r)
                        if r < rows:
                            bad += 1
                            ex = ex or g
                    hits = [ex] if ex else []
                else:
                    nchunks = 512 if (1 << na) >= 512 else (1 << na)
                    minr, bad, hits = batch_search(B, nchunks, 2)
                    minr, bad = int(minr), int(bad)
                    hits = [int(h) for h in hits.ravel() if h >= 0][:2]
                msg = (f"a={a} b={b:2d} side={side}: #x={(1 << na) - 1}, map {rows}->{nc}, "
                       f"min rank {minr}, #non-injective x = {bad}")
                if bad:
                    msg += "  ZERO DIVISOR e.g. x = " + elem_str(U, elem_from_bits(U, a, hits[0]))
                print(msg + f"  ({t1 - t:.1f}+{time.time() - t1:.1f}s) [{time.time() - t0:.0f}s]", flush=True)
