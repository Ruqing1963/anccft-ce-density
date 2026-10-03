"""Task C1: dim ker / coker of R_sigma' on u_m^split, block by block over the Q-grading (sigma' has Q-degree delta).
rank over F2 of R_sigma on u_m (F8 model) = rank over F8 of R_sigma (x) F8 = rank over F2 of R_sigma' on u_m^split.
Blocks related by the vertex rotation rho (which fixes sigma') have equal rank; with --reps only one block per
rho-orbit is computed (for m <= 8 all blocks are computed and the orbit equality is checked).
usage: python taskC1_blocks.py m [--reps]   -> out pickle taskC1_m{m}.pkl"""
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

from cengine import Split, Blocks, build_block, pack_words
from gf2 import gf2_rank_inplace

DELTA = (1, 1, 1)


def rot(g):
    return (g[2], g[0], g[1])


def orbit_rep(g):
    return min(g, rot(g), rot(rot(g)))


def run(m, reps, nthreads_chunks=64, verbose=True):
    t0 = time.time()
    sp = Split(m)
    B = Blocks(sp)
    words, wlen = pack_words(sp.sigma_words())
    t1 = time.time()
    if verbose:
        print(f"m={m}: {sp.nl} letters, |G_m| = {1 << sp.nl}, top {sp.top}, {len(B.blocks)} Q-blocks; setup {t1 - t0:.1f}s",
              flush=True)
    rank = {}
    tb = tr = 0.0
    order = sorted(B.blocks, key=lambda g: B.dim(g))
    for g in order:
        h = tuple(a + b for a, b in zip(g, DELTA))
        if B.dim(h) == 0:
            rank[g] = 0
            continue
        if reps and orbit_rep(g) != g:
            continue
        rows = B.blocks[g]
        ta = time.time()
        P, bad = build_block(rows, words, wlen, sp.br, sp.sq, B.local, B.key, np.int32(B.gkey(h)), B.dim(h),
                             nthreads_chunks)
        tb += time.time() - ta
        assert bad == 0, (g, bad)
        ta = time.time()
        rank[g] = int(gf2_rank_inplace(P, B.dim(h)))
        tr += time.time() - ta
        del P
        if verbose and B.dim(g) > 20000:
            print(f"   block {g} (n={sum(g)}): {B.dim(g)} -> {B.dim(h)}, rank {rank[g]}  [build {tb:.0f}s, rank {tr:.0f}s]",
                  flush=True)
    if reps:
        for g in B.blocks:
            if g not in rank:
                rank[g] = rank[orbit_rep(g)]
    else:
        for g in B.blocks:
            assert rank[g] == rank[orbit_rep(g)], ("rotation symmetry fails", g)
    dims = {g: B.dim(g) for g in B.blocks}
    ker = {g: dims[g] - rank[g] for g in dims}
    cok = {}
    for g in dims:
        f = tuple(a - b for a, b in zip(g, DELTA))
        cok[g] = dims[g] - rank.get(f, 0)
    res = {"m": m, "top": sp.top, "dims": dims, "rank": rank, "ker": ker, "cok": cok,
           "time_build": tb, "time_rank": tr, "time_total": time.time() - t0}
    with open(f"taskC1_m{m}.pkl", "wb") as fh:
        pickle.dump(res, fh)
    return res


def summary(res):
    m, top = res["m"], res["top"]
    kn = defaultdict(int)
    cn = defaultdict(int)
    hn = defaultdict(int)
    for g, k in res["ker"].items():
        kn[sum(g)] += k
        cn[sum(g)] += res["cok"][g]
        hn[sum(g)] += res["dims"][g]
    N = sum(hn.values())
    K = sum(kn.values())
    C = sum(cn.values())
    assert K == C
    kerlist = [kn[n] for n in range(top + 1)]
    coklist = [cn[n] for n in range(top + 1)]
    # kernel of R: u_n -> u_{n+3} excluding the 3 top degrees (where the target is 0)
    kmid = [kn[n] if n + 3 <= top else None for n in range(top + 1)]
    first = next(n for n in range(top + 1) if kn[n] and n + 3 <= top)
    print(f"m={m}: |G_m|={N}, total dim ker = total dim coker = {C},  coker/n_m = {C / (3 * N):.6f},  "
          f"coker/|G_m| = {C / N:.6f}   (build {res['time_build']:.1f}s, rank {res['time_rank']:.1f}s, total {res['time_total']:.1f}s)")
    print(f"m={m}: coker by degree = {coklist}")
    print(f"m={m}: ker R:u_n->u_(n+3) by degree (n <= top-3) = {[x for x in kmid if x is not None]}")
    print(f"m={m}: first non-injective degree {first}  (m(m-1)/2 = {m * (m - 1) // 2}), dim ker there {kn[first]}")
    blk = {g: k for g, k in res["ker"].items() if sum(g) == first and k}
    print(f"m={m}: Q-degrees of the kernel at n={first}: {blk}")
    return C, N


if __name__ == "__main__":
    m = int(sys.argv[1])
    reps = "--reps" in sys.argv
    res = run(m, reps)
    summary(res)
