"""Independent check of single Q-blocks with ulie.ULie('split') (ascending PBW order, memoized left straightening):
rank of R_sigma': (u_m^split)_gamma -> (u_m^split)_{gamma+delta}.  usage: python taskC_check_block.py m g0 g1 g2 [...]"""
import sys

import numpy as np

from cengine import wt
from gf2 import gf2_rank
from taskA2_kernel import nullspace_rows
from ulie import ULie


def block_monos(U, g):
    lw = [wt(k, b.bit_length() - 1) for (k, b) in U.letters]
    n = sum(g)
    res = []
    for x in U.monos(n):
        w = [0, 0, 0]
        y = x
        while y:
            low = y & -y
            i = low.bit_length() - 1
            for s in range(3):
                w[s] += lw[i][s]
            y ^= low
        if tuple(w) == tuple(g):
            res.append(x)
    return res


if __name__ == "__main__":
    m = int(sys.argv[1])
    gs = [tuple(int(a) for a in sys.argv[i:i + 3]) for i in range(2, len(sys.argv), 3)]
    top = sum(k * (3 if k % 3 else 2) for k in range(1, m))
    U = ULie("split", Dl=m - 1, maxdeg=top)
    L = U.lidx
    sig = [{1 << L[(1, 1)] | 1 << L[(1, 2)] | 1 << L[(1, 4)]}, {1 << L[(1, 4)] | 1 << L[(2, 2)]},
           {1 << L[(1, 1)] | 1 << L[(2, 4)]}, {1 << L[(3, 2)]}]
    X = set()
    for s in sig:
        X ^= s
    for g in gs:
        rows = block_monos(U, g)
        h = tuple(a + 1 for a in g)
        cols = block_monos(U, h)
        cidx = {c: i for i, c in enumerate(cols)}
        M = np.zeros((len(rows), len(cols)), dtype=np.uint8)
        for i, y in enumerate(rows):
            acc = set()
            for x in X:
                acc ^= U.mul_mono(y, x)
            for t in acc:
                M[i, cidx[t]] ^= 1
        from gf2 import rank_bool
        r = rank_bool(M)
        print(f"m={m} block {g} (n={sum(g)}): dim {len(rows)} -> {len(cols)}, rank {r}, dim ker {len(rows) - r}", flush=True)
        if len(rows) - r and len(rows) <= 3000:
            K = nullspace_rows(M)
            for v in K:
                el = sorted(rows[i] for i in np.nonzero(v)[0])
                print("   kernel element (ascending-order PBW, y{k}.{bit}):",
                      " + ".join(U.mono_str(x).replace("x", "y") for x in el))
