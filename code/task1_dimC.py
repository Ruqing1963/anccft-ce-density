"""Task 1/4: dim C_m = dim F2[G_m]/F2[G_m]F_m by bit-packed GF(2) rank.
usage: python task1_dimC.py 2 3 4 5 6 [7]"""
import sys
import time

import numpy as np

from common import F_support, F_matrix_packed, n_of, nbits
from gf2 import gf2_rank

if __name__ == "__main__":
    ms = [int(a) for a in sys.argv[1:]] or [2, 3, 4, 5, 6]
    for m in ms:
        t0 = time.time()
        supp = F_support(m)
        P, ns = F_matrix_packed(m, supp)
        N = P.shape[0]
        assert N == 2 ** nbits(m) == n_of(m) // 3
        t1 = time.time()
        r = gf2_rank(P, N)
        t2 = time.time()
        d = N - r
        print(f"m={m}: |G_m|={N}, |supp F_m|={ns}, rank={r}, dim C_m={d}, dim C_m/n_m={d / n_of(m):.6f}, "
              f"build {t1 - t0:.1f}s, rank {t2 - t1:.1f}s", flush=True)
