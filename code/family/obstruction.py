"""Integer obstruction to  sigma' in c_1 U + ... + c_d U  (column 0), computed modulo a large prime P (> d, so no restricted
effects in degree d; the result is the characteristic-0 obstruction) and lifted to small integers.
We row-reduce the ideal generators, reduce sigma' against them, and print the residual coordinates on the non-pivot
monomials (symmetric lift to (-P/2, P/2)).  Since the ideal is spanned by integral PBW vectors whose pivot structure may
depend on P, we also report the gcd test for small primes directly with fam (colideal.py)."""
import sys
import numpy as np
from fam import Fam, _acc

P = 1000003


def residual(p, d):
    F = Fam(p, d, d + 1)
    delta = tuple([1] * d)
    basis = F.monomials_by_block([d])[delta]
    idx = {b: i for i, b in enumerate(basis)}
    allmono = F.monomials_by_block(list(range(0, d)))
    gens = []
    for k in range(1, d + 1):
        v = [0] * d
        v[(k - 1) % d] = 1
        c = {}
        for pos, cf in F._vec(k, v):
            l = [0] * F.nl
            l[pos] = 1
            _acc(c, {tuple(l): 1}, cf, F.p)
        wk = F._wt(k, (k - 1) % d)
        need = tuple(a - b for a, b in zip(delta, wk))
        for mo in allmono.get(need, []):
            gens.append(F.mul(c, {mo: 1}))
    n = len(basis)
    rows = []
    for g in gens:
        r = [0] * n
        for mo, v in g.items():
            r[idx[mo]] = v % p
        rows.append(r)
    # Gaussian elimination mod p (python ints)
    piv = []
    R = []
    for r in rows:
        r = r[:]
        for (c, pr) in zip(piv, R):
            if r[c]:
                f = r[c]
                r = [(a - f * b) % p for a, b in zip(r, pr)]
        nz = [i for i, a in enumerate(r) if a]
        if nz:
            c = nz[0]
            iv = pow(r[c], p - 2, p)
            r = [(a * iv) % p for a in r]
            # back-reduce existing
            R = [[(a - pr[c] * b) % p for a, b in zip(pr, r)] for pr in R]
            piv.append(c)
            R.append(r)
    out = {}
    for o in (1, -1):
        s = F.elem(F.sigma(o))
        r = [0] * n
        for mo, v in s.items():
            r[idx[mo]] = v % p
        for (c, pr) in zip(piv, R):
            if r[c]:
                f = r[c]
                r = [(a - f * b) % p for a, b in zip(r, pr)]
        lift = {F.mstr(basis[i]): (a if a <= p // 2 else a - p) for i, a in enumerate(r) if a}
        out[o] = lift
    return out, n, len(piv)


if __name__ == "__main__":
    for d in map(int, sys.argv[1:]):
        out, n, rk = residual(P, d)
        print(f"d={d}: dim U_delta={n}, ideal rank={rk}, quotient dim={n-rk}")
        for o in (1, -1):
            vals = list(out[o].values())
            g = int(np.gcd.reduce([abs(v) for v in vals])) if vals else 0
            print(f"  orient {o:+d}: residual = {out[o]}   gcd = {g}")
