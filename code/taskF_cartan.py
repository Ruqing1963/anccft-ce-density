r"""Task F probe: does the left ideal U*sigma' meet the Cartan polynomial ring u(h^) = F2[y_{3j,p}] (j odd)?
If z = s*sigma' with 0 != z in u(h^), then Theta'(z) = sigma'*Theta'(s) and R_sigma' is injective on U = u(n^)
(U is free over the domain u(h^)).  Weight k*delta, degree 3k, computed in u_{3k+1} (= U in degrees <= 3k).
usage: python taskF_cartan.py kmax
"""
import sys
import time
import numpy as np
from cengine import Split, pack_words
from gf2 import gf2_rank_inplace
from taskC2_low import blocks_of_degree, build_block_sorted


def run(k):
    m = 3 * k + 1
    sp = Split(m)
    words, wlen = pack_words(sp.sigma_words())
    rows = blocks_of_degree(sp, 3 * k - 3).get((k - 1,) * 3, np.zeros(0, np.int64))
    cols = blocks_of_degree(sp, 3 * k)[(k,) * 3]
    P, bad = build_block_sorted(rows, words, wlen, sp.br, sp.sq, cols, 16)
    assert bad == 0
    cart = np.array([i for i in range(sp.nl) if sp.ldeg[i] % 3 == 0], dtype=np.int64)
    cmask = 0
    for i in cart:
        cmask |= 1 << int(i)
    C = [j for j, mono in enumerate(cols) if (int(mono) & ~cmask) == 0]
    W = P.shape[1]
    CM = np.zeros((len(C), W), dtype=np.uint64)
    for r, j in enumerate(C):
        CM[r, j >> 6] |= np.uint64(1) << np.uint64(j & 63)
    rR = int(gf2_rank_inplace(P.copy(), cols.size))
    rRC = int(gf2_rank_inplace(np.concatenate([P, CM]).copy(), cols.size))
    inter = rR + len(C) - rRC
    print(f"k={k}: dim U_(k-1)d = {rows.size}, dim U_kd = {cols.size}, rank R = {rR}, dim u(h)_kd = {len(C)}, "
          f"dim (U sigma' cap u(h))_kd = {inter}", flush=True)


if __name__ == "__main__":
    for k in range(2, int(sys.argv[1]) + 1):
        t = time.time()
        run(k)
        print(f"   [{time.time() - t:.1f}s]", flush=True)
