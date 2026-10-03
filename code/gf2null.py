"""GF(2) left null space and row-space basis (numba, Four-Russians panels as in gf2.py).
left_kernel(P, ncols): rows v with v*P = 0, P packed (r x W).  rowbasis(P, ncols): echelon basis of the row space."""
import numpy as np
from numba import njit

from gf2 import _lowbit, _tables, _apply


@njit
def _elim(A, ncols):
    """eliminate on the first ncols columns; all words of A are updated.  Returns (rank, pivot row ids, remaining row ids)."""
    R, W = A.shape
    act = np.arange(R)
    nact = R
    rank = 0
    pivrows = np.empty(R, dtype=np.int64)
    PIV = np.zeros((64, W), dtype=np.uint64)
    T = np.zeros((8, 256, W), dtype=np.uint64)
    lb = np.zeros(64, dtype=np.int64)
    pvec = np.zeros(64, dtype=np.uint64)
    pidx = np.zeros(64, dtype=np.int64)
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
            for k in range(w, W):
                PIV[npiv, k] = A[r, k]
            for j in range(npiv):
                if (comb >> np.uint64(j)) & one:
                    for k in range(w, W):
                        PIV[npiv, k] ^= PIV[j, k]
            pvec[npiv] = v
            lb[npiv] = _lowbit(v)
            pidx[npiv] = r
            npiv += 1
            keep[ii] = False
        if npiv == 0:
            continue
        for j in range(npiv - 1, -1, -1):
            for k2 in range(j):
                if (PIV[k2, w] >> np.uint64(lb[j])) & one:
                    for k in range(w, W):
                        PIV[k2, k] ^= PIV[j, k]
        # write the reduced pivot rows back (so that the caller can read a row basis)
        for j in range(npiv):
            for k in range(w, W):
                A[pidx[j], k] = PIV[j, k]
            pivrows[rank + j] = pidx[j]
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
    return rank, pivrows[:rank], act[:nact]


def left_kernel(P, ncols):
    """basis (packed, nrows-bit vectors) of {v : v P = 0} where P is packed r x W.  P is not modified."""
    r, W1 = P.shape
    W2 = (r + 63) // 64
    A = np.zeros((r, W1 + W2), dtype=np.uint64)
    A[:, :W1] = P
    idx = np.arange(r)
    A[idx, W1 + (idx >> 6)] = np.left_shift(np.uint64(1), (idx & 63).astype(np.uint64))
    rank, piv, rest = _elim(A, W1 * 64)
    K = A[rest][:, W1:].copy()
    assert not A[rest][:, :W1].any()
    return K, int(rank)


def rowbasis(P, ncols):
    """independent rows spanning the row space of P (packed)."""
    A = P.copy()
    rank, piv, rest = _elim(A, ncols)
    # pivot rows were reduced only on columns >= their panel; they remain independent and span the row space
    return A[piv].copy()
