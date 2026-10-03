"""Task A2: Jordan type of R_sigma : y -> y sigma(F) on u' = u(L/L_{>=m}) = gr F2[G_m].

sigma = sigma(F) is homogeneous of degree 3, so sigma^k maps u'_n -> u'_{n+3k} and
r_k = rank R_{sigma^k} = sum_n r_k(n),  r_k(n) = rank(u'_n -> u'_{n+3k}).
r_k(n) is computed by composing the sparse matrices M_d of R_sigma : u'_d -> u'_{d+3}:
T_0 = identity on u'_n (rows = basis of the current degree, columns = basis of u'_n, bit-packed),
T_{k} = M^T T_{k-1}   (row j of T_k = XOR of rows i of T_{k-1} with (M_d)_{ij} = 1), r_k(n) = rank T_k.
String (Jordan block) decomposition of the graded nilpotent operator: the number of blocks starting in
degree n of length exactly l is  s(n,l) = [r_{l-1}(n) - r_l(n-3)] - [r_l(n) - r_{l+1}(n-3)]  (r_0(n) = dim u'_n).
usage: python taskA2_graded_jordan.py 2 3 4 5 6 7 [8]
"""
import pickle
import sys
import time

import numpy as np
from numba import njit, prange

from gf2 import gf2_rank_inplace
from task3a_sigmaF import SIGMA_F
from ulie import ULie


def sparse_R(U, X, d):
    """columns of R_sigma: u_d -> u_{d+3} as transposed CSR: for each target j, list of sources i"""
    rows = U.monos(d)
    cidx = U.index(d + 3)
    nc = len(cidx)
    cols = [[] for _ in range(nc)]
    for i, y in enumerate(rows):
        acc = set()
        for x in X:
            acc ^= U.mul_mono(y, x)
        for t in acc:
            cols[cidx[t]].append(i)
    indptr = np.zeros(nc + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(c) for c in cols])
    ind = np.fromiter((i for c in cols for i in c), dtype=np.int64, count=int(indptr[-1]))
    return indptr, ind


@njit(parallel=True)
def _step(T, indptr, ind, W):
    nc = indptr.size - 1
    out = np.zeros((nc, W), dtype=np.uint64)
    for j in prange(nc):
        for p in range(indptr[j], indptr[j + 1]):
            i = ind[p]
            for k in range(W):
                out[j, k] ^= T[i, k]
    return out


def identity_packed(n):
    W = (n + 63) // 64
    T = np.zeros((n, W), dtype=np.uint64)
    i = np.arange(n)
    T[i, i >> 6] = np.left_shift(np.uint64(1), (i & 63).astype(np.uint64))
    return T


def build(m):
    Dl = m - 1
    top = sum(k * (3 if k % 3 else 2) for k in range(1, m))
    U = ULie("F8", Dl=Dl, maxdeg=top)
    X = set(x for x in SIGMA_F if x < (1 << U.nl))
    dims = [len(U.monos(n)) for n in range(top + 1)]
    assert sum(dims) == 2 ** (3 * (m - 1) - (m - 1) // 3)
    mats = {d: sparse_R(U, X, d) for d in range(0, top - 2)}
    return top, dims, mats


def ranks_table(top, dims, mats):
    """r[k][n] = rank(sigma^k : u_n -> u_{n+3k}), k >= 0"""
    r = {0: {n: dims[n] for n in range(top + 1)}}
    for n in range(top + 1):
        if dims[n] == 0:
            continue
        T = identity_packed(dims[n])
        W = T.shape[1]
        k = 0
        d = n
        while d + 3 <= top:
            indptr, ind = mats[d]
            T = _step(T, indptr, ind, W)
            k += 1
            d += 3
            A = T.copy()
            rk = int(gf2_rank_inplace(A, dims[n])) if A.shape[0] else 0
            r.setdefault(k, {})[n] = rk
            if rk == 0:
                break
    return r


def analyse(m, top, dims, r):
    R = lambda k, n: 0 if n < 0 or n > top else r.get(k, {}).get(n, 0)
    kmax = max(k for k in r if any(v for v in r[k].values()))
    Nsig = kmax + 1
    tot = [sum(R(k, n) for n in range(top + 1)) for k in range(Nsig + 1)]
    N = sum(dims)
    blocks = {}
    for l in range(1, Nsig + 1):
        c = tot[l - 1] - 2 * tot[l] + (tot[l + 1] if l + 1 < len(tot) else 0)
        if c:
            blocks[l] = c
    assert sum(l * c for l, c in blocks.items()) == N
    coker = [dims[n] - R(1, n - 3) for n in range(top + 1)]
    ker = [dims[n] - R(1, n) for n in range(top + 1)]
    pred = [dims[n] - (dims[n - 3] if n >= 3 else 0) for n in range(top + 1)]
    strings = {}
    for n in range(top + 1):
        for l in range(1, Nsig + 1):
            s = (R(l - 1, n) - R(l, n - 3)) - (R(l, n) - R(l + 1, n - 3))
            if s:
                strings[(n, l)] = s
    assert sum(l * s for (n, l), s in strings.items()) == N
    return Nsig, tot, blocks, coker, ker, pred, strings


if __name__ == "__main__":
    for m in [int(a) for a in sys.argv[1:]]:
        t0 = time.time()
        top, dims, mats = build(m)
        t1 = time.time()
        r = ranks_table(top, dims, mats)
        t2 = time.time()
        Nsig, tot, blocks, coker, ker, pred, strings = analyse(m, top, dims, r)
        N = sum(dims)
        with open(f"taskA2_ranks_m{m}.pkl", "wb") as fh:
            pickle.dump({"m": m, "top": top, "dims": dims, "r": r}, fh)
        print(f"m={m}: |G_m|={N}, top degree {top}, Loewy length {top + 1}; build {t1 - t0:.1f}s, ranks {t2 - t1:.1f}s")
        print(f"m={m}: dims u'_n = {dims}")
        print(f"m={m}: N_sigma = {Nsig}   (ceil((top+1)/3) = {-(-(top + 1) // 3)})")
        print(f"m={m}: ranks of sigma^k, k=0..N_sigma: {tot}")
        print(f"m={m}: Jordan blocks {{size: count}} = {blocks}")
        nb = sum(blocks.values())
        print(f"m={m}: #blocks = graded coker = {nb} (sum coker {sum(coker)});  |G|/N_sigma = {N / Nsig:.2f};"
              f"  ratio = {nb * Nsig / N:.4f};  mean block size {N / nb:.3f}")
        print(f"m={m}: coker by degree   = {coker}")
        print(f"m={m}: (1-z^3)H_m coeffs = {pred}")
        print(f"m={m}: ker R_sigma on u'_n = {ker}")
        bad = [n for n in range(top + 1) if ker[n]]
        print(f"m={m}: degrees n with R_sigma: u'_n -> u'_(n+3) non-injective: {bad}"
              f"  (first = {bad[0] if bad else None}; top-first = {top - bad[0] if bad else None}; first coker deviation at n = "
              f"{bad[0] + 3 if bad else None})")
        surj = [n for n in range(3, top + 1) if coker[n] == 0]
        print(f"m={m}: degrees n where R_sigma: u'_(n-3) -> u'_n is surjective: {surj}")
        print(f"m={m}: strings {{(start degree, length): count}} = {dict(sorted(strings.items()))}", flush=True)
