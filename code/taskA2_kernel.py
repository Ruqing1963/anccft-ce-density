"""Task A2 (structure): explicit kernel of R_sigma : u'_n -> u'_{n+3} in the first non-injective degree,
u' = u(L/L_{>=m}).  usage: python taskA2_kernel.py m n"""
import sys

import numpy as np

from task3a_sigmaF import SIGMA_F
from ulie import ULie


def nullspace_rows(M):
    """left kernel {v : v M = 0} of a GF(2) 0/1 matrix M (rows x cols), via row reduction of [M | I]"""
    r, c = M.shape
    A = np.concatenate([M.astype(np.uint8), np.eye(r, dtype=np.uint8)], axis=1)
    piv = 0
    for col in range(c):
        p = next((i for i in range(piv, r) if A[i, col]), None)
        if p is None:
            continue
        A[[piv, p]] = A[[p, piv]]
        for i in range(r):
            if i != piv and A[i, col]:
                A[i] ^= A[piv]
        piv += 1
    K = A[piv:, c:]
    # reduce kernel basis
    rr = 0
    for col in range(r):
        p = next((i for i in range(rr, K.shape[0]) if K[i, col]), None)
        if p is None:
            continue
        K[[rr, p]] = K[[p, rr]]
        for i in range(K.shape[0]):
            if i != rr and K[i, col]:
                K[i] ^= K[rr]
        rr += 1
    return K


if __name__ == "__main__":
    m, n = int(sys.argv[1]), int(sys.argv[2])
    top = sum(k * (3 if k % 3 else 2) for k in range(1, m))
    U = ULie("F8", Dl=m - 1, maxdeg=top)
    X = set(x for x in SIGMA_F if x < (1 << U.nl))
    rows = U.monos(n)
    cidx = U.index(n + 3)
    M = np.zeros((len(rows), len(cidx)), dtype=np.uint8)
    for i, y in enumerate(rows):
        acc = set()
        for x in X:
            acc ^= U.mul_mono(y, x)
        for t in acc:
            M[i, cidx[t]] ^= 1
    K = nullspace_rows(M)
    print(f"m={m}, n={n}: dim u'_n={len(rows)}, dim ker R_sigma = {K.shape[0]}")
    for v in K:
        el = {rows[i] for i in np.nonzero(v)[0]}
        degs = {tuple(sorted(U.ldeg[i] for i in range(U.nl) if (x >> i) & 1)) for x in el}
        print(f"   {len(el)} PBW terms; letter-degree patterns of the terms: {sorted(degs)}")
        if len(sys.argv) > 3:
            print("  ", " + ".join(U.mono_str(x) for x in sorted(el)))
        # also left multiplication by sigma and squares, for structure
        lv = set()
        for x in X:
            for y in el:
                lv ^= U.mul_mono(x, y)
        print("     sigma*v == 0 ?", len(lv) == 0)
