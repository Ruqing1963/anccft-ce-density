r"""Task F: does every kernel element of R_sigma' on u_m involve a letter of degree >= m - a ?
i.e. is ker R^{(m)} contained in J_a := ker(pi: u_m -> u_{m-a}) = span{PBW monomials with a letter of degree >= m-a}?
For each Q-block: dim ker R_gamma  and  dim pi(ker R_gamma) = dim ker - (#J-rows - rank(J-rows)).
usage: python taskF_top.py m nmax a1 [a2 ...]      (one block per rho-orbit; uses taskC1_m{m}.pkl for the kernel dims)
"""
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

from cengine import Split, pack_words
from gf2 import gf2_rank_inplace
from taskC2_low import blocks_of_degree, build_block_sorted


def rot(g):
    return (g[2], g[0], g[1])


def orbit_rep(g):
    return min(g, rot(g), rot(rot(g)))


def main(m, nmax, alist):
    t0 = time.time()
    sp = Split(m)
    words, wlen = pack_words(sp.sigma_words())
    with open(f"taskC1_m{m}.pkl", "rb") as fh:
        D = pickle.load(fh)
    ker = D["ker"]
    top = D["top"]
    tot = defaultdict(lambda: defaultdict(int))
    for n in range(0, min(nmax, top - 3) + 1):
        B = blocks_of_degree(sp, n)
        Bn = blocks_of_degree(sp, n + 3)
        for g, rows in B.items():
            if orbit_rep(g) != g or not ker.get(g, 0):
                continue
            h = tuple(x + 1 for x in g)
            cols = Bn.get(h)
            k = ker[g]
            res = {}
            for a in alist:
                s = sum(1 for (kk, p) in sp.letters if kk >= m - a)
                jm = (rows & ((1 << s) - 1)) != 0
                Jr = rows[jm]
                if cols is None or Jr.size == 0:
                    kJ = Jr.size
                else:
                    P, bad = build_block_sorted(Jr, words, wlen, sp.br, sp.sq, cols, 16)
                    assert bad == 0
                    kJ = Jr.size - int(gf2_rank_inplace(P, cols.size))
                res[a] = k - kJ          # dim pi_{m-a}(ker)
                tot[n][a] += len({g, rot(g), rot(rot(g))}) * (k - kJ)
            tot[n]["ker"] += len({g, rot(g), rot(rot(g))}) * k
            if any(res[a] for a in alist) and n <= 60:
                print(f"  n={n} block {g}: dim ker {k}, dim pi_(m-a)(ker) for a={alist}: {[res[a] for a in alist]}",
                      flush=True)
        if tot[n]["ker"]:
            print(f"m={m} n={n}: dim ker {tot[n]['ker']}, dim pi_(m-a)(ker): "
                  f"{ {a: tot[n][a] for a in alist} }  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    m, nmax = int(sys.argv[1]), int(sys.argv[2])
    main(m, nmax, [int(x) for x in sys.argv[3:]])
