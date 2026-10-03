"""Task 3a: injectivity of left/right multiplication by sigma(F) on u(L)_b -> u(L)_{b+3}."""
import sys
import time

from ulie import ULie
from zdtools import mult_matrix
from gf2 import gf2_rank

SIGMA_F = [7, 17, 20, 33, 34]       # masks from task2_leading.py (letters ordered (k,b))

if __name__ == "__main__":
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    U = ULie("F8", Dl=D)
    print("sigma(F) =", " + ".join(U.mono_str(x) for x in SIGMA_F))
    X = set(SIGMA_F)
    t0 = time.time()
    for b in range(0, D - 2):
        line = f"b={b:2d}: dim u_b={len(U.monos(b)):6d} -> dim u_(b+3)={len(U.monos(b + 3)):6d}"
        for side in ("R", "L"):
            t = time.time()
            P, nc = mult_matrix(U, X, 3, b, side)
            t1 = time.time()
            r = gf2_rank(P, nc)
            line += f" | {side}: rank {r} {'INJ' if r == P.shape[0] else 'NOT INJECTIVE'} ({t1 - t:.1f}+{time.time() - t1:.1f}s)"
        print(line + f"  memo={len(U.memo)}  [{time.time() - t0:.0f}s]", flush=True)
