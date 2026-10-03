"""Task 5: graded upper bound  dim coker( y -> y sigma(F) on u(L/L_{>=m}) ) >= dim C_m.
u' = u(L/L_{>=m}) = gr F2[G_m]: letters of degree < m, brackets / 2-maps landing in degree >= m are 0.
usage: python task5_graded.py 2 3 4 5 6 7"""
import sys
import time

from ulie import ULie
from zdtools import mult_matrix
from gf2 import gf2_rank
from task3a_sigmaF import SIGMA_F

KNOWN = {2: 7, 3: 37, 4: 100, 5: 547, 6: 3170, 7: 10226}

if __name__ == "__main__":
    for m in [int(a) for a in sys.argv[1:]]:
        t0 = time.time()
        Dl = m - 1
        top = sum(k * (3 if k % 3 else 2) for k in range(1, m))
        U = ULie("F8", Dl=Dl, maxdeg=top)
        X = set(x for x in SIGMA_F if x < (1 << U.nl))
        # sigma(F) uses letters of degree 1, 2 only; for m = 2 the degree-2 letters vanish
        dims = [len(U.monos(n)) for n in range(top + 1)]
        assert sum(dims) == 2 ** (3 * (m - 1) - (m - 1) // 3)
        cok = []
        for n in range(top + 1):
            if n < 3:
                cok.append(dims[n])
                continue
            P, nc = mult_matrix(U, X, 3, n - 3, "R")
            r = gf2_rank(P, nc) if P.shape[0] else 0
            cok.append(dims[n] - r)
        tot = sum(cok)
        print(f"m={m}: |G_m|={sum(dims)}, top degree {top}, graded coker dims by degree {cok}", flush=True)
        print(f"m={m}: dim coker(R_sigma(F) on gr F2[G_m]) = {tot}   vs dim C_m = {KNOWN.get(m)}"
              f"   ratio {tot / KNOWN[m] if m in KNOWN else float('nan'):.4f}   /n_m = {tot / (3 * sum(dims)):.6f}"
              f"   ({time.time() - t0:.1f}s, memo {len(U.memo)})", flush=True)
