r"""Task F: the criterion of THEORY §7.3.
For the restricted ideal K_a := L_{[m-a, m)} of L_m = L^s/L^s_{>=m} (abelian with zero 2-map when 2(m-a) >= m), with
integral omega_{K_a} = product of all letters of degree m-a..m-1:
      ker R^{(m)}  subset  ker(pi: u_m -> u_{m-a})    <==>    omega_{K_a}  in  u_m sigma'   (<==> in sigma' u_m).
Test: is the PBW monomial omega_{K_a} in the image of R_sigma' : (u_m)_{gamma - delta} -> (u_m)_gamma ?
Also prints a preimage w (number of PBW terms) for inspection.
usage: python taskF_omegaK.py mmin mmax a1 [a2 ...]
"""
import sys
import time
import numpy as np

from cengine import Split, pack_words
from gf2 import gf2_rank_inplace
from taskC2_low import blocks_of_degree, build_block_sorted


def test(m, a, verbose=False):
    sp = Split(m)
    words, wlen = pack_words(sp.sigma_words())
    s = sum(1 for (k, p) in sp.letters if k >= m - a)
    omega = (1 << s) - 1
    g = sp.qdeg(omega)
    n = sum(g)
    gm = tuple(x - 1 for x in g)
    rows = blocks_of_degree(sp, n - 3).get(gm, np.zeros(0, np.int64))
    cols = blocks_of_degree(sp, n)[g]
    if rows.size == 0:
        return False, 0, 0, None
    P, bad = build_block_sorted(rows, words, wlen, sp.br, sp.sq, cols, 16)
    assert bad == 0
    j = int(np.searchsorted(cols, omega))
    assert cols[j] == omega
    W = P.shape[1]
    e = np.zeros((1, W), dtype=np.uint64)
    e[0, j >> 6] = np.uint64(1) << np.uint64(j & 63)
    r1 = int(gf2_rank_inplace(P.copy(), cols.size))
    r2 = int(gf2_rank_inplace(np.concatenate([P, e]).copy(), cols.size))
    inimg = (r1 == r2)
    pre = None
    if inimg and verbose:
        from gf2null import left_kernel
        A = np.concatenate([P, e])
        K, rk = left_kernel(A, cols.size)
        for v in K:
            bits = np.unpackbits(v.view(np.uint8), bitorder="little")[:rows.size + 1]
            if bits[rows.size]:
                pre = rows[np.nonzero(bits[:rows.size])[0]]
                break
    return inimg, rows.size, cols.size, (g, n, pre)


if __name__ == "__main__":
    mmin, mmax = int(sys.argv[1]), int(sys.argv[2])
    alist = [int(x) for x in sys.argv[3:]]
    for m in range(mmin, mmax + 1):
        for a in alist:
            if 2 * (m - a) < m:
                continue
            t = time.time()
            ok, nr, nc, info = test(m, a, verbose=(m <= 9))
            g, n, pre = info if info else (None, None, None)
            extra = f", preimage with {pre.size} PBW terms" if pre is not None else ""
            print(f"m={m} a={a}: omega_K weight {g} degree {n}: block {nr} -> {nc}; omega_K in u_m sigma': {ok}{extra}"
                  f"  [{time.time() - t:.1f}s]", flush=True)
