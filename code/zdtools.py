"""Multiplication matrices in u(L) and batch zero-divisor search."""
import numpy as np
from numba import njit, prange

from gf2 import gf2_rank


def mult_matrix(U, X, a, b, side):
    """packed matrix of y -> X*y (side='L') or y -> y*X (side='R') from u_b to u_{a+b}.
    X: set of monomials of degree a."""
    rows = U.monos(b)
    cidx = U.index(a + b)
    nc = len(cidx)
    W = max(1, (nc + 63) // 64)
    P = np.zeros((len(rows), W), dtype=np.uint64)
    for r, y in enumerate(rows):
        acc = set()
        for x in X:
            acc ^= U.mul_mono(x, y) if side == "L" else U.mul_mono(y, x)
        for t in acc:
            c = cidx[t]
            P[r, c >> 6] ^= np.uint64(1) << np.uint64(c & 63)
    return P, nc


def basis_matrices(U, a, b, side):
    """stack B[i] = matrix of multiplication by the i-th PBW monomial of degree a"""
    mats = []
    for x in U.monos(a):
        P, nc = mult_matrix(U, {x}, a, b, side)
        mats.append(P)
    return np.ascontiguousarray(np.stack(mats)), nc


@njit
def _small_rank(M, nrows, W):
    r = 0
    one = np.uint64(1)
    for w in range(W):
        for bit in range(64):
            if r == nrows:
                return r
            p = -1
            for i in range(r, nrows):
                if (M[i, w] >> np.uint64(bit)) & one:
                    p = i
                    break
            if p < 0:
                continue
            if p != r:
                for k in range(w, W):
                    tmp = M[p, k]
                    M[p, k] = M[r, k]
                    M[r, k] = tmp
            for i in range(p + 1, nrows):
                if (M[i, w] >> np.uint64(bit)) & one:
                    for k in range(w, W):
                        M[i, k] ^= M[r, k]
            r += 1
    return r


@njit(parallel=True)
def batch_search(B, nchunks, maxhits):
    """B: (n, rows, W).  For every nonzero x in F2^n, rank of sum_i x_i B[i].
    Returns (min rank, number of x with rank < rows, first hits)."""
    n, rows, W = B.shape
    total = (1 << n)
    csize = (total + nchunks - 1) // nchunks
    minr = np.full(nchunks, rows, dtype=np.int64)
    cnt = np.zeros(nchunks, dtype=np.int64)
    hits = np.full((nchunks, maxhits), -1, dtype=np.int64)
    for c in prange(nchunks):
        lo = c * csize
        hi = min(total, lo + csize)
        if lo >= hi:
            continue
        M = np.zeros((rows, W), dtype=np.uint64)
        g = lo ^ (lo >> 1)
        for i in range(n):
            if (g >> i) & 1:
                for r in range(rows):
                    for k in range(W):
                        M[r, k] ^= B[i, r, k]
        T = np.empty((rows, W), dtype=np.uint64)
        nh = 0
        for s in range(lo, hi):
            if s > lo:
                # gray code step s-1 -> s flips bit = trailing zeros of s
                t = s
                i = 0
                while not (t & 1):
                    t >>= 1
                    i += 1
                g ^= (1 << i)
                for r in range(rows):
                    for k in range(W):
                        M[r, k] ^= B[i, r, k]
            if g == 0:
                continue
            for r in range(rows):
                for k in range(W):
                    T[r, k] = M[r, k]
            rk = _small_rank(T, rows, W)
            if rk < minr[c]:
                minr[c] = rk
            if rk < rows:
                cnt[c] += 1
                if nh < maxhits:
                    hits[c, nh] = g
                    nh += 1
    return minr.min(), cnt.sum(), hits


def search(U, a, b, side, nchunks=256, maxhits=4):
    B, nc = basis_matrices(U, a, b, side)
    mr, cnt, hits = batch_search(B, nchunks, maxhits)
    h = [int(x) for x in hits.ravel() if x >= 0][:maxhits]
    return int(mr), int(cnt), h, B.shape[1], nc


def elem_from_bits(U, a, g):
    ms = U.monos(a)
    return {ms[i] for i in range(len(ms)) if (g >> i) & 1}


def elem_str(U, X):
    return " + ".join(U.mono_str(x) for x in sorted(X)) or "0"
