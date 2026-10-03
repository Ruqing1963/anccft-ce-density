r"""Task C2/C4: kernel of R_sigma' on low principal degrees of u_m^split for large m (m = 10, 11, 12), block by block.

Blocks with dim <= CAP are computed exactly (rank over F2).  Larger blocks are certified injective, when possible, by the
central filtration: L_{m-1} is central in L/L_{>=m}, J = L_{m-1} u_m, gr_J u_m = Lambda(L_{m-1}) (x) u_{m-1} and
R_sigma' acts as 1 (x) R_sigma'^{(m-1)}; hence
    dim ker(R^{(m)} on block gamma) <= sum_{S subset of letters of degree m-1} dim ker(R^{(m-1)} on block gamma - wt(S))
(semicontinuity of the kernel for a filtered map).  The m-1 kernels are read from taskC1_m{m-1}.pkl / taskC2_m{m-1}.pkl.
usage: python taskC2_low.py m nmin nmax [CAP]
"""
import itertools
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
from numba import njit, prange

from cengine import Split, pack_words, rmul_word, MAXOUT, MAXST, MAXW
from gf2 import gf2_rank_inplace

DELTA = (1, 1, 1)


@njit
def enum_degree(ldeg, n):
    """all masks (letters 0..nl-1, degrees ldeg) of total degree n"""
    nl = ldeg.size
    out = []
    # iterative DFS: stack of (next letter, remaining, mask)
    stI = np.empty(nl + 2, dtype=np.int64)
    stR = np.empty(nl + 2, dtype=np.int64)
    stM = np.empty(nl + 2, dtype=np.int64)
    sp = 0
    stI[0] = 0
    stR[0] = n
    stM[0] = 0
    sp = 1
    while sp > 0:
        sp -= 1
        i = stI[sp]
        r = stR[sp]
        mk = stM[sp]
        if r == 0:
            out.append(mk)
            continue
        if i >= nl:
            continue
        # option: skip letter i ; option: take letter i
        stI[sp] = i + 1
        stR[sp] = r
        stM[sp] = mk
        sp += 1
        if ldeg[i] <= r:
            stI[sp] = i + 1
            stR[sp] = r - ldeg[i]
            stM[sp] = mk | (np.int64(1) << i)
            sp += 1
    return np.array(out, dtype=np.int64)


def qdeg_arr(sp, masks):
    q = np.zeros((masks.size, 3), dtype=np.int64)
    for i in range(sp.nl):
        b = (masks >> i) & 1
        q += b[:, None] * sp.lwt[i][None, :]
    return q


def blocks_of_degree(sp, n):
    masks = enum_degree(sp.ldeg, n)
    if masks.size == 0:
        return {}
    q = qdeg_arr(sp, masks)
    K = 4 * sp.top + 8
    key = (q[:, 0] * K + q[:, 1]) * K + q[:, 2]
    order = np.lexsort((masks, key))
    ks = key[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [ks.size]])
    out = {}
    for s, e in zip(starts, ends):
        g = tuple(int(x) for x in q[order[s]])
        out[g] = np.sort(masks[order[s:e]])
    return out


@njit(parallel=True)
def build_block_sorted(rows, words, wlen, br, sq, cols, nchunk):
    nr = rows.size
    ncols = cols.size
    Wd = (ncols + 63) // 64
    P = np.zeros((nr, Wd), dtype=np.uint64)
    bad = np.zeros(nchunk, dtype=np.int64)
    csz = (nr + nchunk - 1) // nchunk
    for c in prange(nchunk):
        out = np.empty(MAXOUT, dtype=np.int64)
        stS = np.empty(MAXST, dtype=np.int64)
        stW = np.empty((MAXST, MAXW), dtype=np.int64)
        stL = np.empty(MAXST, dtype=np.int64)
        Wb = np.empty(MAXW, dtype=np.int64)
        lo = c * csz
        hi = min(nr, lo + csz)
        for r in range(lo, hi):
            n = 0
            for w in range(wlen.size):
                n = rmul_word(rows[r], words[w, :wlen[w]], br, sq, out, n, stS, stW, stL, Wb)
                if n < 0:
                    bad[c] += 1
                    break
            if n < 0:
                continue
            for i in range(n):
                j = np.searchsorted(cols, out[i])
                if j >= ncols or cols[j] != out[i]:
                    bad[c] += 1000000
                    continue
                P[r, j >> 6] ^= np.uint64(1) << np.uint64(j & 63)
    return P, bad.sum()


def rot(g):
    return (g[2], g[0], g[1])


def orbit_rep(g):
    return min(g, rot(g), rot(rot(g)))


def load_prev(m):
    for fn in (f"taskC1_m{m}.pkl", f"taskC2_m{m}.pkl"):
        try:
            with open(fn, "rb") as fh:
                d = pickle.load(fh)
            return d
        except FileNotFoundError:
            continue
    return None


if __name__ == "__main__":
    m, nmin, nmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 40000
    sp = Split(m)
    words, wlen = pack_words(sp.sigma_words())
    prev = load_prev(m - 1)
    pker = prev["ker"] if prev else {}
    pknown = prev.get("known") if prev else None      # set of blocks whose kernel is exactly known (taskC2 files)
    # Q-weights of subsets of the letters of degree m-1
    top_letters = [i for i in range(sp.nl) if sp.ldeg[i] == m - 1]
    lam = defaultdict(int)
    for r in range(len(top_letters) + 1):
        for S in itertools.combinations(top_letters, r):
            w = tuple(int(sum(sp.lwt[i][s] for i in S)) for s in range(3))
            lam[w] += 1
    fname = f"taskC2_m{m}.pkl"
    try:
        with open(fname, "rb") as fh:
            res = pickle.load(fh)
    except FileNotFoundError:
        res = {"m": m, "ker": {}, "dims": {}, "known": set(), "bound": {}, "ndone": set()}
    t0 = time.time()
    for n in range(nmin, nmax + 1):
        B = blocks_of_degree(sp, n)
        Bn = blocks_of_degree(sp, n + 3)
        kn = 0
        unknown = []
        maxb = 0
        for g in sorted(B, key=orbit_rep):
            rows = B[g]
            h = tuple(a + 1 for a in g)
            res["dims"][g] = rows.size
            if g in res["known"]:
                kn += res["ker"][g]
                continue
            if h not in Bn:
                res["ker"][g] = rows.size
                res["known"].add(g)
                kn += rows.size
                continue
            # certification bound from m-1
            bnd = None
            if prev is not None:
                bnd = 0
                for w, mult in lam.items():
                    gg = tuple(a - b for a, b in zip(g, w))
                    if pknown is not None and (sum(gg) not in prev["ndone"] or (gg in prev["dims"] and gg not in pknown)):
                        bnd = None
                        break
                    bnd += mult * pker.get(gg, 0)
                res["bound"][g] = bnd
            if bnd == 0:
                res["ker"][g] = 0
                res["known"].add(g)
                continue
            rep = orbit_rep(g)
            if rep != g and rep in res["known"]:
                res["ker"][g] = res["ker"][rep]
                res["known"].add(g)
                kn += res["ker"][g]
                continue
            if rows.size > CAP:
                unknown.append((g, rows.size, bnd))
                continue
            P, bad = build_block_sorted(rows, words, wlen, sp.br, sp.sq, Bn[h], 64)
            assert bad == 0
            rk = int(gf2_rank_inplace(P, Bn[h].size))
            res["ker"][g] = rows.size - rk
            res["known"].add(g)
            kn += rows.size - rk
            maxb = max(maxb, rows.size)
        kblocks = {g: res["ker"][g] for g in B if g in res["known"] and res["ker"][g]}
        print(f"m={m} n={n}: h_n={sum(b.size for b in B.values())}, {len(B)} blocks, largest computed {maxb}; "
              f"kernel (known blocks) = {kn}; uncertified blocks: {[(g, s, b) for g, s, b in unknown]}; "
              f"kernel blocks: {dict(sorted(kblocks.items()))}  [{time.time() - t0:.0f}s]", flush=True)
        res["ndone"].add(n)        # all blocks of degree n are in res["dims"]; exact ones are in res["known"]
        with open(fname, "wb") as fh:
            pickle.dump(res, fh)
