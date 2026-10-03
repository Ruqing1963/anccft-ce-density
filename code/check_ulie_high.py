"""Extra validation of the u(L) code:
 (1) F8 model: random PBW products x*y (single monomials, deg x + deg y up to 11) against the
     group algebra F2[G_12] (Jennings products mod higher weight);
 (2) associativity of the split model and of the F8 model on random triples of larger degree."""
import random
import time

from ulie import ULie
from task2_leading import Jennings, gmul

rng = random.Random(11)
t0 = time.time()
U = ULie("F8", Dl=16)
m = 12
Jn = Jennings(m, U)
n = 0
for tot in range(7, 12):
    for _ in range(12):
        a = rng.randint(1, tot - 1)
        b = tot - a
        x, y = rng.choice(U.monos(a)), rng.choice(U.monos(b))
        prod = gmul(Jn.R, Jn.J(x), Jn.J(y))
        co = Jn.coeffs(prod, tot)
        assert all(U.mdeg(c) == tot for c in co)
        assert co == U.mul({x}, {y}), (U.mono_str(x), U.mono_str(y))
        n += 1
print(f"(1) {n} random PBW products with total degree 7..11 agree with F2[G_12]  ({time.time() - t0:.0f}s)")

for kind in ("split", "F8"):
    V = ULie(kind, Dl=14)
    k = 0
    for _ in range(400):
        a, b, c = rng.randint(1, 6), rng.randint(1, 6), rng.randint(1, 6)
        if a + b + c > 14:
            continue
        X = {rng.choice(V.monos(a)), rng.choice(V.monos(a))}
        Y = {rng.choice(V.monos(b))}
        Z = {rng.choice(V.monos(c)), rng.choice(V.monos(c))}
        assert V.mul(V.mul(X, Y), Z) == V.mul(X, V.mul(Y, Z))
        k += 1
    print(f"(2) {kind}: associativity on {k} random triples, total degree <= 14  ({time.time() - t0:.0f}s)")
