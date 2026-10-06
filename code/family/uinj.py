"""Exact injectivity of R_sigma and L_sigma on the UNtruncated U in degrees n <= nmax:
U_n -> U_{n+d} agrees with u_m for m = n + d + 1.  usage: python uinj.py p d nmax [orient]
Prints, per degree, dim U_n, dim ker R, dim ker L; for the first kernel prints a basis vector."""
import sys
import time
import numpy as np
from fam import Fam, _acc, rank_mod_p


def nullspace_mod_p(M, p):
    """basis of {x : x M = 0} (left null space) over F_p, as rows"""
    A = np.concatenate([M % p, np.eye(M.shape[0], dtype=np.int64)], axis=1)
    nr, nc = M.shape
    r = 0
    for c in range(nc):
        piv = None
        for i in range(r, nr):
            if A[i, c] % p:
                piv = i
                break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p
        for i in range(nr):
            if i != r and A[i, c] % p:
                A[i] = (A[i] - A[i, c] * A[r]) % p
        r += 1
    return A[r:, nc:]


def mats(F, n, orient, side):
    d = F.d
    src = F.monomials_by_block([n])
    tgt = F.monomials_by_block([n + d])
    sig = F.elem(F.sigma(orient))
    out = []
    for g, rows in sorted(src.items()):
        gt = tuple(a + 1 for a in g)
        cols = tgt.get(gt, [])
        ci = {mo: j for j, mo in enumerate(cols)}
        M = np.zeros((len(rows), len(cols)), dtype=np.int64)
        for r, mo in enumerate(rows):
            e = F.mul({mo: 1}, sig) if side == "R" else F.mul(sig, {mo: 1})
            for mo2, v in e.items():
                M[r, ci[mo2]] = v
        out.append((g, rows, M))
    return out


def main():
    p, d, nmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    orient = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    shown = False
    print(f"(p,d)=({p},{d}) orient={orient}: injectivity of sigma' on U", flush=True)
    for n in range(0, nmax + 1):
        t0 = time.time()
        F = Fam(p, d, n + d + 1)
        res = {}
        for side in ("R", "L"):
            tot = 0
            kd = 0
            for g, rows, M in mats(F, n, orient, side):
                tot += len(rows)
                k = len(rows) - (rank_mod_p(M, p) if M.shape[1] else 0)
                kd += k
                if k and side == "R" and not shown:
                    N = nullspace_mod_p(M, p)
                    v = N[0]
                    terms = [f"{int(c)}*{F.mstr(rows[i])}" for i, c in enumerate(v) if c]
                    print(f"   first right kernel vector, block {g}: " + " + ".join(terms[:40])
                          + (" ..." if len(terms) > 40 else ""), flush=True)
                    shown = True
            res[side] = (tot, kd)
        print(f"n={n}: dim U_n={res['R'][0]}  ker R={res['R'][1]}  ker L={res['L'][1]}  ({time.time()-t0:.1f}s)",
              flush=True)


if __name__ == "__main__":
    main()
