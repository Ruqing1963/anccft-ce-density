r"""Task C2: explicit kernel of R_sigma' on single Q-blocks of u_m^split (engine basis: PBW monomials, letters
y{k}.{p} = e_p phi^k, product of letters in DESCENDING degree order).  Also tests whether the kernel elements are
products of root vectors (single monomials) and whether they lie in u_m * omega_j.
usage: python taskC2_kernel.py m g0 g1 g2 [g0 g1 g2 ...]"""
import sys

import numpy as np

from cengine import Split, pack_words
from taskC2_low import blocks_of_degree, build_block_sorted
from taskA2_kernel import nullspace_rows


def unpack(P, ncols):
    r = P.shape[0]
    b = np.unpackbits(P.view(np.uint8).reshape(r, -1), axis=1, bitorder="little")
    return b[:, :ncols]


def kernel_block(sp, g):
    words, wlen = pack_words(sp.sigma_words())
    n = sum(g)
    B = blocks_of_degree(sp, n)
    Bn = blocks_of_degree(sp, n + 3)
    rows = B[g]
    h = tuple(a + 1 for a in g)
    cols = Bn.get(h, np.zeros(0, dtype=np.int64))
    if cols.size == 0:
        return rows, np.eye(rows.size, dtype=np.uint8)
    P, bad = build_block_sorted(rows, words, wlen, sp.br, sp.sq, cols, 8)
    assert bad == 0
    M = unpack(P, cols.size)
    return rows, nullspace_rows(M)


if __name__ == "__main__":
    m = int(sys.argv[1])
    sp = Split(m)
    gs = [tuple(int(a) for a in sys.argv[i:i + 3]) for i in range(2, len(sys.argv), 3)]
    for g in gs:
        rows, K = kernel_block(sp, g)
        print(f"m={m} block {g} (n={sum(g)}): dim {rows.size}, dim ker {K.shape[0]}")
        for v in K:
            el = [int(rows[i]) for i in np.nonzero(v)[0]]
            print(f"   {len(el)} terms: " + " + ".join(sp.mstr(x) for x in el))
