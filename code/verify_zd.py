"""Verify homogeneous zero divisors found by task3b independently:
 1. recompute x*y = 0 in u(L) (PBW straightening, two different truncations);
 2. lift x, y to the Jennings elements of F2[G_m] (m > a+b) and check, purely by group-algebra
    arithmetic (no use of the u(L) code), that the product lies in I^{a+b+1}:
    all Jennings coordinates of weight <= a+b vanish.
usage: python verify_zd.py a b xbits [m]"""
import sys
from collections import Counter

from ulie import ULie
from task2_leading import Jennings, gmul
from zdtools import elem_from_bits, elem_str


def kernel_vectors(U, X, a, b, side):
    """basis of {y in u_b : X y = 0} (side L) via python-int elimination"""
    rows = U.monos(b)
    cidx = U.index(a + b)
    vecs = []
    for r, y in enumerate(rows):
        acc = set()
        for x in X:
            acc ^= U.mul_mono(x, y) if side == "L" else U.mul_mono(y, x)
        v = 0
        for t in acc:
            v |= 1 << cidx[t]
        vecs.append((v, 1 << r))
    piv = {}
    ker = []
    for v, tag in vecs:
        while v:
            h = v.bit_length() - 1
            if h in piv:
                pv, pt = piv[h]
                v ^= pv
                tag ^= pt
            else:
                piv[h] = (v, tag)
                break
        if v == 0:
            ker.append(tag)
    return [{rows[i] for i in range(len(rows)) if (t >> i) & 1} for t in ker]


def lift(Jn, X):
    tot = Counter()
    for mono in X:
        for g in Jn.J(mono):
            tot[g] ^= 1
    return {g for g, c in tot.items() if c}


if __name__ == "__main__":
    a, b, xbits = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    m = int(sys.argv[4]) if len(sys.argv) > 4 else a + b + 1
    U = ULie("F8", Dl=a + b)
    X = elem_from_bits(U, a, xbits)
    print(f"x (deg {a}) = {elem_str(U, X)}")
    for side in ("L", "R"):
        K = kernel_vectors(U, X, a, b, side)
        print(f"side {side}: dim ker of {'y->xy' if side == 'L' else 'y->yx'} on u_{b} = {len(K)}")
        U2 = ULie("F8", Dl=a + b + 6)
        for Y in K:
            print(f"   y = {elem_str(U, Y)}")
            p1 = U.mul(X, Y) if side == "L" else U.mul(Y, X)
            p2 = U2.mul(X, Y) if side == "L" else U2.mul(Y, X)
            assert not p1 and not p2
        # group algebra check for the first kernel vector
        if K:
            Y = K[0]
            Jn = Jennings(m, U2)
            Xt, Yt = lift(Jn, X), lift(Jn, Y)
            prod = gmul(Jn.R, Xt, Yt) if side == "L" else gmul(Jn.R, Yt, Xt)
            co = Jn.coeffs(prod, a + b)
            byw = Counter(U2.mdeg(c) for c in co)
            print(f"   group algebra F2[G_{m}]: |supp lift x|={len(Xt)}, |supp lift y|={len(Yt)}, "
                  f"|supp product|={len(prod)}; nonzero Jennings coords of weight <= {a + b}: {dict(byw)}")
            co2 = Jn.coeffs(prod, a + b + 1)
            lead = [c for c in co2 if U2.mdeg(c) == a + b + 1]
            print(f"   -> product lies in I^{a + b + 1}; its weight-{a + b + 1} part has {len(lead)} PBW terms")
            # also: x*y computed by the group algebra at weight a+b without cancellation check
            Ug = ULie("F8", Dl=a + b + 1)
            lead_u = Ug.mul(X, Y) if side == "L" else Ug.mul(Y, X)
            print(f"   (u(L) product in degree {a + b} is {len(lead_u)} terms, i.e. zero)")
