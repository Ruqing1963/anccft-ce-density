r"""Task H: check of the 'forced lifting' theorem (THEORY 8.1) on small levels.

Theorem 8.1 (proved): let K_a = letters of degree m-a..m-1 (an ideal of L_m), gamma a Q-block of u_{m-a}.  If
coker R^{(m-a)} vanishes in every target block gamma + delta - wt(S), S a nonempty set of K_a-letters, then
pi_{m-a}(ker R^{(m)}_gamma) = ker R^{(m-a)}_gamma (every level-(m-a) kernel vector lifts to a level-m kernel vector).
In particular, if top(u(K_a)) <= k_{m-a} + 2 then omega_{K_a} is NOT in u_m sigma' (C_a fails at m).

For each block gamma (one per rho-orbit) with ker R^{(m-a)}_gamma != 0 this script computes
   kerA = dim ker R^{(m-a)}_gamma,  forced = theorem hypothesis,  piK = dim pi_{m-a}(ker R^{(m)}_gamma)  (exact)
and checks: forced => piK == kerA.  Also prints per degree the totals.
usage: python taskH_lift_check.py m a [nmin nmax]
"""
import itertools
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


def main(m, a, nmin, nmax):
    t0 = time.time()
    sp = Split(m)
    words, wlen = pack_words(sp.sigma_words())
    with open(f"taskC1_m{m}.pkl", "rb") as fh:
        DM = pickle.load(fh)
    with open(f"taskC1_m{m - a}.pkl", "rb") as fh:
        DA = pickle.load(fh)
    kerM, kerA, cokA = DM["ker"], DA["ker"], DA["cok"]
    Kl = [(k, p) for (k, p) in sp.letters if k >= m - a]
    from cengine import wt
    Kw = [wt(k, p) for (k, p) in Kl]
    Sw = set()
    for r in range(1, len(Kw) + 1):
        for c in itertools.combinations(Kw, r):
            Sw.add(tuple(sum(x[i] for x in c) for i in range(3)))
    topA = DA["top"]
    TK = sum(k for k, p in Kl)
    print(f"m={m} a={a}: |K_a|={len(Kl)}, top u(K_a)={TK}, top u_(m-a)={topA}, top u_m={sp.top}", flush=True)
    s = len(Kl)
    viol = 0
    nforced = 0
    tot = defaultdict(lambda: [0, 0, 0, 0])   # n -> [kerA, piK, kerM, forced kerA]
    for n in range(nmin, min(nmax, topA) + 1):
        B = blocks_of_degree(sp, n)
        Bn = blocks_of_degree(sp, n + 3)
        for g, rows in B.items():
            if orbit_rep(g) != g:
                continue
            kA = kerA.get(g, 0)
            if not kA:
                continue
            mult = len({g, rot(g), rot(rot(g))})
            forced = all(cokA.get(tuple(g[i] + 1 - w[i] for i in range(3)), 0) == 0 for w in Sw)
            k = kerM.get(g, 0)
            h = tuple(x + 1 for x in g)
            cols = Bn.get(h)
            jm = (rows & ((1 << s) - 1)) != 0
            Jr = rows[jm]
            if cols is None or Jr.size == 0:
                kJ = Jr.size
            else:
                P, bad = build_block_sorted(Jr, words, wlen, sp.br, sp.sq, cols, 16)
                assert bad == 0
                kJ = Jr.size - int(gf2_rank_inplace(P, cols.size))
            piK = k - kJ
            assert 0 <= piK <= kA
            if forced:
                nforced += 1
                if piK != kA:
                    viol += 1
                    print(f"  VIOLATION n={n} block {g}: kerA={kA} piK={piK}", flush=True)
            tot[n][0] += mult * kA
            tot[n][1] += mult * piK
            tot[n][2] += mult * k
            tot[n][3] += mult * kA * forced
        if tot[n][0]:
            print(f"  n={n}: dim ker^(m-a)={tot[n][0]}, dim pi(ker^(m))={tot[n][1]}, dim ker^(m)={tot[n][2]}, "
                  f"forced-lift part={tot[n][3]}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"m={m} a={a}: blocks with forced lifting: {nforced}, violations: {viol}", flush=True)


if __name__ == "__main__":
    m, a = int(sys.argv[1]), int(sys.argv[2])
    nmin = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    nmax = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    main(m, a, nmin, nmax)
