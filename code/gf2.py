"""Bit-packed GF(2) rank in numba: 64-column panels, Gauss-Jordan on the <= 64 panel pivots,
then 8 Four-Russians tables of 256 combinations each, applied to all remaining rows in parallel.
Pivot rows are discarded after use (only the rank is needed).  The input array is destroyed."""
import numpy as np
from numba import njit, prange


@njit
def _lowbit(v):
    b = 0
    while not (v >> np.uint64(b)) & np.uint64(1):
        b += 1
    return b


@njit(parallel=True)
def _apply(A, act, nact, w, W, lb, npiv, T):
    one = np.uint64(1)
    for ii in prange(nact):
        r = act[ii]
        v = A[r, w]
        if v == 0:
            continue
        comb = np.uint64(0)
        for j in range(npiv):
            if (v >> np.uint64(lb[j])) & one:
                comb |= one << np.uint64(j)
        for c in range(8):
            idx = (comb >> np.uint64(8 * c)) & np.uint64(255)
            if idx:
                for k in range(w, W):
                    A[r, k] ^= T[c, idx, k - w]


@njit(parallel=True)
def _tables(PIV, npiv, w, W, T):
    for c in prange(8):
        for idx in range(1, 256):
            lo = idx & (idx - 1)
            j = 8 * c
            t = idx ^ lo
            while t > 1:
                t >>= 1
                j += 1
            if j >= npiv:
                for k in range(W - w):
                    T[c, idx, k] = 0
                continue
            for k in range(W - w):
                T[c, idx, k] = T[c, lo, k] ^ PIV[j, w + k]


@njit
def gf2_rank_inplace(A, ncols):
    R, W = A.shape
    act = np.arange(R)
    nact = R
    rank = 0
    PIV = np.zeros((64, W), dtype=np.uint64)
    T = np.zeros((8, 256, W), dtype=np.uint64)
    lb = np.zeros(64, dtype=np.int64)
    pvec = np.zeros(64, dtype=np.uint64)
    one = np.uint64(1)
    nw = (ncols + 63) // 64
    for w in range(nw):
        if nact == 0:
            break
        npiv = 0
        keep = np.ones(nact, dtype=np.bool_)
        for ii in range(nact):
            if npiv == 64:
                break
            r = act[ii]
            v = A[r, w]
            if v == 0:
                continue
            comb = np.uint64(0)
            for j in range(npiv):
                if (v >> np.uint64(lb[j])) & one:
                    v ^= pvec[j]
                    comb |= one << np.uint64(j)
            if v == 0:
                continue
            # new pivot: full row = A[r] + sum of pivots in comb
            for k in range(w, W):
                PIV[npiv, k] = A[r, k]
            for j in range(npiv):
                if (comb >> np.uint64(j)) & one:
                    for k in range(w, W):
                        PIV[npiv, k] ^= PIV[j, k]
            pvec[npiv] = v
            lb[npiv] = _lowbit(v)
            npiv += 1
            keep[ii] = False
        if npiv == 0:
            continue
        # back substitution: pivot j is the only one with bit lb[j]
        for j in range(npiv - 1, -1, -1):
            for k2 in range(j):
                if (PIV[k2, w] >> np.uint64(lb[j])) & one:
                    for k in range(w, W):
                        PIV[k2, k] ^= PIV[j, k]
        rank += npiv
        newact = np.empty(nact - npiv, dtype=np.int64)
        t = 0
        for ii in range(nact):
            if keep[ii]:
                newact[t] = act[ii]
                t += 1
        act = newact
        nact = t
        _tables(PIV, npiv, w, W, T)
        _apply(A, act, nact, w, W, lb, npiv, T)
    return rank


def gf2_rank(A, ncols=None):
    A = np.ascontiguousarray(A, dtype=np.uint64).copy()
    if ncols is None:
        ncols = 64 * A.shape[1]
    if A.shape[0] == 0 or A.shape[1] == 0:
        return 0
    return int(gf2_rank_inplace(A, ncols))


def pack_bool(M):
    """bool matrix (r x c) -> packed uint64 (r x ceil(c/64)), little-endian bits."""
    M = np.asarray(M, dtype=bool)
    r, c = M.shape
    W = (c + 63) // 64
    pad = np.zeros((r, W * 64), dtype=bool)
    pad[:, :c] = M
    b = np.packbits(pad.reshape(r, W * 8, 8), axis=2, bitorder="little").reshape(r, W * 8)
    return b.view(np.uint64).copy()


def rank_bool(M):
    M = np.asarray(M, dtype=bool)
    if M.size == 0:
        return 0
    return gf2_rank(pack_bool(M), M.shape[1])


def rank_slow(M):
    """reference: python ints"""
    rows = [int("".join("1" if x else "0" for x in row[::-1]) or "0", 2) for row in np.asarray(M, bool)]
    piv = {}
    r = 0
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h in piv:
                v ^= piv[h]
            else:
                piv[h] = v
                r += 1
                break
    return r


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(1)
    for (r, c, k) in [(5, 7, 3), (100, 130, 60), (300, 200, 150), (700, 700, 650), (1000, 1500, 999), (64, 64, 64)]:
        X = rng.integers(0, 2, (r, k)).astype(np.uint8)
        Y = rng.integers(0, 2, (k, c)).astype(np.uint8)
        M = (X @ Y) % 2
        a, b = rank_bool(M), rank_slow(M)
        print(r, c, k, a, b)
        assert a == b
    N = 8192
    M = rng.integers(0, 2, (N, N)).astype(bool)
    t = time.time()
    print("random 8192 rank", rank_bool(M), time.time() - t)

