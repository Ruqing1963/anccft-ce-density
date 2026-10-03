r"""Task E: the induced-module bound.

For a graded (Q-graded) restricted subalgebra K of L_m = L^s/L^s_{>=m}, put M_K := F2 (x)_{u(K)} u_m = u_m / K u_m (a right
u_m-module).  With a PBW order in which the letters of K come FIRST, M_K has basis the PBW monomials in the complementary
letters, and (1 (x) v) sigma' = (v sigma' with all monomials containing a K-letter dropped).
THEORY §6: dim ker(R_sigma' | (u_m)_n) <= sum_mu dim gr u(K)_mu * dim ker(sigma' | (M_K)_{n-mu}); in particular
k_m >= k(M_K) := least degree with ker(sigma' | M_K) != 0.

This module builds an engine with an arbitrary letter order (same rewriting system as cengine.py) and computes
ker(sigma' | M_K) per Q-block.
usage: python taskE_induced.py m nmax KIND   (KIND: up012, up021, ..., or none)
"""
import itertools
import sys
import time
from collections import defaultdict

import numpy as np
from numba import njit, prange

from cengine import Split, pack_words, rmul_word, MAXOUT, MAXST, MAXW, wt
from gf2 import gf2_rank_inplace


def rowcol(k, p):
    """y_{k,p} = e_p phi^k ~ E_{p, p-k} t^*  (row, column) mod 3"""
    return p % 3, (p - k) % 3


class SplitOrd(Split):
    """Split model with letters in a custom order: `first` = list of (k,p) placed first (lowest bit positions)."""

    def __init__(self, m, first=()):
        super().__init__(m)
        base = self.letters                       # degree-descending
        first = [l for l in base if l in set(first)]
        rest = [l for l in base if l not in set(first)]
        self.nK = len(first)
        self.letters = first + rest
        nl = self.nl
        self.pos = {l: i for i, l in enumerate(self.letters)}
        self.ldeg = np.array([k for k, p in self.letters], dtype=np.int64)
        self.lwt = np.array([wt(k, p) for k, p in self.letters], dtype=np.int64)
        self.br = np.full((max(nl, 1), max(nl, 1), 2), -1, dtype=np.int64)
        self.sq = np.full((max(nl, 1), 2), -1, dtype=np.int64)
        for a, (i, p) in enumerate(self.letters):
            for b, (j, q) in enumerate(self.letters):
                if a == b:
                    continue
                v = 0
                if p == (q + i) % 3:
                    v ^= 1 << p
                if q == (p + j) % 3:
                    v ^= 1 << q
                self._store(self.br[a, b], i + j, v)
            v = (1 << p) if i % 3 == 0 else 0
            self._store(self.sq[a], 2 * i, v)
        self.Kmask = (1 << self.nK) - 1


@njit
def enum_degree_from(ldeg, n, start):
    nl = ldeg.size
    out = []
    stI = np.empty(nl + 2, dtype=np.int64)
    stR = np.empty(nl + 2, dtype=np.int64)
    stM = np.empty(nl + 2, dtype=np.int64)
    stI[0] = start
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


def blocks_M(sp, n):
    masks = enum_degree_from(sp.ldeg, n, sp.nK)
    out = defaultdict(list)
    for mk in masks:
        out[sp.qdeg(int(mk))].append(int(mk))
    return {g: np.array(sorted(v), dtype=np.int64) for g, v in out.items()}


@njit(parallel=True)
def build_block_M(rows, words, wlen, br, sq, cols, Kmask, nchunk):
    nr = rows.size
    ncols = cols.size
    Wd = (ncols + 63) // 64
    P = np.zeros((nr, max(Wd, 1)), dtype=np.uint64)
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
                if out[i] & Kmask:
                    continue
                j = np.searchsorted(cols, out[i])
                if j >= ncols or cols[j] != out[i]:
                    bad[c] += 1000000
                    continue
                P[r, j >> 6] ^= np.uint64(1) << np.uint64(j & 63)
    return P, bad.sum()


def kernel_profile(sp, nmax, nmin=0, verbose=False):
    words, wlen = pack_words(sp.sigma_words())
    res = {}
    for n in range(nmin, nmax + 1):
        B = blocks_M(sp, n)
        Bn = blocks_M(sp, n + 3)
        kn = 0
        kb = {}
        for g, rows in B.items():
            h = tuple(a + 1 for a in g)
            cols = Bn.get(h)
            if cols is None:
                k = rows.size
            else:
                P, bad = build_block_M(rows, words, wlen, sp.br, sp.sq, cols, np.int64(sp.Kmask), 16)
                assert bad == 0, bad
                k = rows.size - int(gf2_rank_inplace(P, cols.size))
            if k:
                kb[g] = k
            kn += k
        res[n] = (kn, sum(r.size for r in B.values()), kb)
        if verbose:
            print(f"  n={n}: dim M_n={res[n][1]}, ker={kn}  {dict(sorted(kb.items())) if kn and len(kb) < 8 else ''}",
                  flush=True)
    return res


def K_upper(m, perm):
    """real letters (k,p) with perm[row] < perm[col]"""
    out = []
    for k in range(1, m):
        for p in range(3):
            if k % 3 == 0:
                continue
            r, c = rowcol(k, p)
            if perm[r] < perm[c]:
                out.append((k, p))
    return out


if __name__ == "__main__":
    m, nmax, kind = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    if kind == "none":
        K = []
    elif kind.startswith("up"):
        perm = [int(ch) for ch in kind[2:]]
        K = K_upper(m, perm)
    elif kind.startswith("tower"):           # tower{p}: e_p t^{2^r}, p in {0,1}
        p = int(kind[5:])
        K = [(3 * 2 ** r, p) for r in range(8) if 3 * 2 ** r < m]
    elif kind.startswith("colreal"):         # colreal{j}: reals c^{(j)}_k = e_{k-1+j} phi^k
        j = int(kind[7:])
        K = [(k, (k - 1 + j) % 3) for k in range(1, m) if k % 3]
    elif kind.startswith("rowreal"):         # rowreal{q}: reals y_{k,q}
        q = int(kind[7:])
        K = [(k, q) for k in range(1, m) if k % 3]
    elif kind.startswith("row"):             # row{q}: Theta'(H^{(j)}) with 1-j = q in {0,1}: y_{k,q} (3!|k) + e_q t^{2^r}
        q = int(kind[3:])
        K = [(k, q) for k in range(1, m) if k % 3] + [(3 * 2 ** r, q) for r in range(8) if 3 * 2 ** r < m]
    else:
        raise SystemExit("unknown kind")
    sp = SplitOrd(m, K)
    t0 = time.time()
    print(f"m={m} K={kind}: dim K = {sp.nK} of {sp.nl} letters; K = {[f'y{k}.{p}' for k, p in sp.letters[:sp.nK]]}",
          flush=True)
    res = kernel_profile(sp, nmax, verbose=True)
    first = next((n for n in sorted(res) if res[n][0]), None)
    print(f"m={m} K={kind}: first kernel degree of sigma' on M_K = {first}  [{time.time() - t0:.0f}s]", flush=True)
