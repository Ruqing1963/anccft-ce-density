"""Task B1: kernel densities dim ker / |G_m| of y -> y x~ (side R) and y -> x~ y (side L) on F2[G_m]
for Jennings lifts x~ of the graded zero divisors (V in u_4, x in u_3), lifts + random higher-weight
corrections, and controls (F, Jennings lift of sigma(F), random elements of u_3 / u_4 lifted).
Also the graded kernel density of L_v, R_v on gr F2[G_m] = u(L/L_{>=m}).
An element of F2[[P]] is a fixed combination of Jennings monomials J(S) (letters of any degree);
its image in F2[G_m] drops every J(S) containing a letter of degree >= m (g_s in P_m -> 1).
usage: python b1_rank.py m [names...]"""
import os
import random
import sys
import time

import numpy as np

from common import F_support, all_elements, encode_arr, rmul_arr, lmul_arr, nbits
from gf2 import gf2_rank
from ulie import ULie
from task2_leading import Jennings
from task3a_sigmaF import SIGMA_F
from zdtools import elem_from_bits, mult_matrix
from verify_zd import lift

UB = ULie("F8", Dl=12)
VB = [elem_from_bits(UB, 4, g) for g in (4005, 524823, 527794)]  # v (first), and two others; span = V
X3 = {mm for mm in UB.monos(3) if UB.mono_str(mm) in
      "x1.1*x2.1 + x1.2*x2.2 + x1.2*x2.4 + x1.4*x2.4".split(" + ")}
assert len(X3) == 4


def rand_elem(U, d, rng):
    return {mm for mm in U.monos(d) if rng.random() < 0.5}


def corr(seed, weights):
    rng = random.Random(seed)
    out = set()
    for w in weights:
        out ^= rand_elem(UB, w, rng)
    return out


ELEMS = {
    "v1": VB[0], "v2": VB[1], "v3": VB[2],
    "v1+c(5,6)": VB[0] ^ corr(11, (5, 6)),
    "v1+c(5,6,7,8)": VB[0] ^ corr(12, (5, 6, 7, 8)),
    "x3": X3,
    "x3+c(4,5,6)": X3 ^ corr(13, (4, 5, 6)),
    "rand4a": rand_elem(UB, 4, random.Random(21)),
    "rand4b": rand_elem(UB, 4, random.Random(22)) ^ corr(23, (5, 6)),
    "rand3": rand_elem(UB, 3, random.Random(31)),
    "J(sigmaF)": set(SIGMA_F),
}


def restrict(X, m):
    return {mm for mm in X if UB.mdeg(mm) >= 0 and all(UB.ldeg[i] < m for i in range(UB.nl) if (mm >> i) & 1)}


def kernel_dim(supp, m, side):
    C = all_elements(m)
    N = C.shape[0]
    W = (N + 63) // 64
    P = np.zeros((N, W), dtype=np.uint64)
    rows = np.arange(N)
    for h in supp:
        col = encode_arr(rmul_arr(C, h, m) if side == "R" else lmul_arr(h, C, m), m)
        P[rows, col >> 6] ^= np.left_shift(np.uint64(1), (col & 63).astype(np.uint64))
    del C
    return N - gf2_rank(P, N), N


def graded_kernel(X, a, m, side):
    """sum over degrees of dim ker of y -> Xy / yX on u(L/L_{>=m}); X homogeneous of degree a"""
    top = sum(k * (3 if k % 3 else 2) for k in range(1, m))
    U = ULie("F8", Dl=m - 1, maxdeg=top)
    Xr = {mm for mm in X if mm < (1 << U.nl)}
    tot = 0
    for n in range(top + 1):
        P, nc = mult_matrix(U, Xr, a, n, side)
        r = gf2_rank(P, nc) if (P.shape[0] and nc) else 0
        tot += P.shape[0] - r
    return tot


if __name__ == "__main__":
    m = int(sys.argv[1])
    names = sys.argv[2:] or list(ELEMS)
    print(f"NUMBA threads {os.environ.get('NUMBA_NUM_THREADS')}", flush=True)
    assert m <= 12
    Jn = Jennings(m, UB)
    N = 2 ** nbits(m)
    for name in names:
        t0 = time.time()
        if name == "F":
            supp = F_support(m)
        else:
            X = restrict(ELEMS[name], m)
            supp = sorted(lift(Jn, X))
        res = []
        for side in ("R", "L"):
            k, N = kernel_dim(supp, m, side)
            res.append(k)
        line = (f"m={m} {name:15s} |supp|={len(supp):6d}  dim ker R = {res[0]:6d} ({res[0] / N:.4f})  "
                f"dim ker L = {res[1]:6d} ({res[1] / N:.4f})  |G_m|={N}")
        if name in ("v1", "v2", "v3", "x3", "rand4a", "rand3") and m <= 7:
            a = 4 if name.startswith(("v", "rand4")) else 3
            gR = graded_kernel(ELEMS[name], a, m, "R")
            gL = graded_kernel(ELEMS[name], a, m, "L")
            line += f"  | graded: ker R = {gR} ({gR / N:.4f}), ker L = {gL} ({gL / N:.4f})"
        print(line + f"  [{time.time() - t0:.0f}s]", flush=True)
