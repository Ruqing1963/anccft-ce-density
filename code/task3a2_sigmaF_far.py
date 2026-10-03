"""Task 3a, pushed further: injectivity of left multiplication by sigma(F) and by S(sigma(F)),
S = antipode (anti-automorphism, S(x) = x on L in char 2).  Right multiplication by sigma(F) is
injective on u_b iff left multiplication by S(sigma(F)) is (since S is a bijective anti-automorphism
preserving degrees), and left multiplication is much cheaper with lmul straightening.
Cross-check: for b <= 15 compare ranks with the direct right multiplication."""
import sys
import time

from ulie import ULie
from zdtools import mult_matrix, elem_str
from gf2 import gf2_rank
from task3a_sigmaF import SIGMA_F

D = int(sys.argv[1]) if len(sys.argv) > 1 else 27
U = ULie("F8", Dl=D)


def antipode(X):
    out = set()
    for mono in X:
        letters = [i for i in range(U.nl) if (mono >> i) & 1]
        cur = {0}
        for i in reversed(letters):
            cur = U.mul(cur, {1 << i})
        out ^= cur
    return out


X = set(SIGMA_F)
SX = antipode(X)
print("sigma(F)    =", elem_str(U, X))
print("S(sigma(F)) =", elem_str(U, SX))
t0 = time.time()
for b in range(0, D - 2):
    line = f"b={b:2d}: {len(U.monos(b)):6d} -> {len(U.monos(b + 3)):6d}"
    for name, Z in (("L_sigmaF", X), ("L_S(sigmaF)=R_sigmaF", SX)):
        t = time.time()
        P, nc = mult_matrix(U, Z, 3, b, "L")
        r = gf2_rank(P, nc)
        line += f" | {name}: rank {r} {'INJ' if r == P.shape[0] else 'NOT INJ'} ({time.time() - t:.1f}s)"
    if b <= 15:
        P, nc = mult_matrix(U, X, 3, b, "R")
        r2 = gf2_rank(P, nc)
        line += f" | direct R_sigmaF rank {r2}"
    print(line + f"  [{time.time() - t0:.0f}s, memo {len(U.memo)}]", flush=True)
