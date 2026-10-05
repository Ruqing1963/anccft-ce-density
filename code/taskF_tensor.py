r"""Task F: sigma' on tensor products of dual evaluation modules V*(t_1) (x) ... (x) V*(t_r).

rho*: L^s -> gl_3(F2[t]),  e0 = y1.0 -> t E20,  e1 = y1.1 -> E01,  e2 = y1.2 -> E12  (transpose of the natural rep, trace
corrected; a restricted representation).  sigma' = e0e1e2 + e1e2e0 + e2e0e1 acts on V*(t) as t*I.
Psi_r(e_i) = sum_s 1 (x) .. (x) R_i(t_s) (x) .. (x) 1 ;  Psi_r(sigma') = cyclic sum.  We test det Psi_r(sigma') != 0 at
random points of GF(2^16)^r  (nonzero at one point => det is a nonzero polynomial).
usage: python taskF_tensor.py rmax [trials]
"""
import sys
import numpy as np
from numba import njit

K = 16
POLY = (1 << 16) | (1 << 12) | (1 << 3) | (1 << 1) | 1     # x^16 + x^12 + x^3 + x + 1 (primitive)


def tables():
    exp = np.zeros(2 * (1 << K), dtype=np.int64)
    log = np.zeros(1 << K, dtype=np.int64)
    x = 1
    for i in range((1 << K) - 1):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & (1 << K):
            x ^= POLY
    assert x == 1
    for i in range((1 << K) - 1, 2 * (1 << K)):
        exp[i] = exp[i - ((1 << K) - 1)]
    return exp, log


EXP, LOG = tables()
Q1 = (1 << K) - 1


@njit
def gmul(a, b, EXP, LOG):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]


@njit
def ginv(a, EXP, LOG):
    return EXP[(Q1 - LOG[a]) % Q1]


@njit
def matmul(A, B, EXP, LOG):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for k in range(n):
            a = A[i, k]
            if a == 0:
                continue
            la = LOG[a]
            for j in range(n):
                b = B[k, j]
                if b != 0:
                    C[i, j] ^= EXP[la + LOG[b]]
    return C


@njit
def rank(A0, EXP, LOG):
    A = A0.copy()
    n, m = A.shape
    r = 0
    for c in range(m):
        p = -1
        for i in range(r, n):
            if A[i, c] != 0:
                p = i
                break
        if p < 0:
            continue
        for j in range(m):
            t = A[r, j]
            A[r, j] = A[p, j]
            A[p, j] = t
        inv = ginv(A[r, c], EXP, LOG)
        for j in range(m):
            A[r, j] = gmul(A[r, j], inv, EXP, LOG)
        for i in range(n):
            if i != r and A[i, c] != 0:
                f = A[i, c]
                for j in range(m):
                    if A[r, j] != 0:
                        A[i, j] ^= gmul(f, A[r, j], EXP, LOG)
        r += 1
    return r


def site_ops(r, ts):
    """Psi_r(e_i), i = 0,1,2, as 3^r x 3^r matrices over GF(2^K)"""
    n = 3 ** r
    E = [np.zeros((n, n), dtype=np.int64) for _ in range(3)]
    for s in range(r):
        # single-site matrices: R0 = t E20, R1 = E01, R2 = E12  (row, col, value)
        R = [[(2, 0, ts[s])], [(0, 1, 1)], [(1, 2, 1)]]
        stride = 3 ** (r - 1 - s)
        for idx in range(n):
            d = (idx // stride) % 3
            for i in range(3):
                for (row, col, val) in R[i]:
                    if col == d:
                        jdx = idx + (row - d) * stride
                        E[i][jdx, idx] ^= val
    return E


def sigma_op(r, ts):
    E = site_ops(r, ts)
    M = np.zeros_like(E[0])
    for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        M ^= matmul(matmul(E[a], E[b], EXP, LOG), E[c], EXP, LOG)
    return M


if __name__ == "__main__":
    rmax = int(sys.argv[1])
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    rng = np.random.default_rng(7)
    # sanity: r = 1 gives t*I
    M = sigma_op(1, [5])
    assert np.array_equal(M, 5 * np.eye(3, dtype=np.int64)), M
    for r in range(1, rmax + 1):
        res = []
        for _ in range(trials):
            ts = [int(x) for x in rng.integers(1, 1 << K, r)]
            M = sigma_op(r, ts)
            res.append(int(rank(M, EXP, LOG)))
        # also equal points
        ts = [int(rng.integers(1, 1 << K))] * r
        req = int(rank(sigma_op(r, ts), EXP, LOG))
        print(f"r={r}: dim {3 ** r}, rank at random points {res}, rank at t_1=...=t_r: {req}", flush=True)
