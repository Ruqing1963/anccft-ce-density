"""Task G helper: Q-block dimensions of u^s_m (generating function prod_letters (1 + x^wt)), truncated in principal degree.
usage: python taskG_dims.py m nmax   -> prints per degree: h_n, #blocks, #orbit reps, largest block"""
import sys
import numpy as np
from cengine import wt


def letters(m):
    return [(k, p) for k in range(m - 1, 0, -1) for p in (range(3) if k % 3 else range(2))]


def block_dims(m, nmax):
    D = nmax + 1
    G = np.zeros((D, D, D), dtype=object)
    G[0, 0, 0] = 1
    for (k, p) in letters(m):
        w = wt(k, p)
        if k > nmax:
            continue
        H = G.copy()
        H[w[0]:, w[1]:, w[2]:] += G[:D - w[0], :D - w[1], :D - w[2]]
        G = H
    out = {}
    for a in range(D):
        for b in range(D - a):
            for c in range(D - a - b):
                if G[a, b, c]:
                    out[(a, b, c)] = int(G[a, b, c])
    return out


if __name__ == "__main__":
    m, nmax = int(sys.argv[1]), int(sys.argv[2])
    nmin = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    d = block_dims(m, nmax)
    for n in range(nmin, nmax + 1):
        bl = {g: v for g, v in d.items() if sum(g) == n}
        if not bl:
            continue
        reps = [g for g in bl if g == min(g, (g[2], g[0], g[1]), (g[1], g[2], g[0]))]
        print(f"m={m} n={n}: h={sum(bl.values())} blocks={len(bl)} reps={len(reps)} max={max(bl.values())} "
              f"sum_reps={sum(bl[g] for g in reps)}", flush=True)
