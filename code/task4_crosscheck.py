"""Independent cross-checks of dim C_m (different matrices with provably equal rank):
 (a) transpose route: rank of x -> x F equals rank of x -> x F*, F* = sum_h h^{-1} (matrix transpose);
 (b) left route: rank of x -> F x (group algebras are Frobenius, so dim A F = dim F A);
 (c) row/column order randomly permuted for x -> x F."""
import sys
import time

import numpy as np

from common import F_support, F_matrix_packed, TowerRing, all_elements, encode_arr, lmul_arr
from gf2 import gf2_rank

m = int(sys.argv[1]) if len(sys.argv) > 1 else 6
R = TowerRing(m)
supp = F_support(m)
N = 2 ** (3 * (m - 1) - (m - 1) // 3)
t = time.time()
Pi, _ = F_matrix_packed(m, sorted(R.ninv(h) for h in supp))
ra = gf2_rank(Pi, N)
del Pi
print(f"m={m} (a) F*: dim = {N - ra}  ({time.time() - t:.1f}s)", flush=True)

t = time.time()
C = all_elements(m)
W = (N + 63) // 64
P = np.zeros((N, W), dtype=np.uint64)
rows = np.arange(N)
for h in supp:
    col = encode_arr(lmul_arr(h, C, m), m)
    assert np.unique(col).size == N
    P[rows, col >> 6] |= np.left_shift(np.uint64(1), (col & 63).astype(np.uint64))
rb = gf2_rank(P, N)
del P
print(f"m={m} (b) F x: dim = {N - rb}  ({time.time() - t:.1f}s)", flush=True)

t = time.time()
rng = np.random.default_rng(5)
pr, pc = rng.permutation(N), rng.permutation(N)
P = np.zeros((N, W), dtype=np.uint64)
from common import rmul_arr
for h in supp:
    col = pc[encode_arr(rmul_arr(C, h, m), m)]
    P[pr, col >> 6] |= np.left_shift(np.uint64(1), (col & 63).astype(np.uint64))
rc = gf2_rank(P, N)
print(f"m={m} (c) permuted x F: dim = {N - rc}  ({time.time() - t:.1f}s)", flush=True)
