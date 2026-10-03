"""Group-algebra verification (F2[G_18], Jennings coordinates up to weight 17) of a degree-(3,14)
zero-divisor pair x*y = 0 found in u(L)."""
import sys
import time
from collections import Counter

from ulie import ULie
from task2_leading import Jennings, gmul
from verify_zd import kernel_vectors, lift
from zdtools import elem_str

t0 = time.time()
U = ULie("F8", Dl=20)
want = "x1.1*x2.1 + x1.2*x2.2 + x1.2*x2.4 + x1.4*x2.4"
X = {mm for mm in U.monos(3) if U.mono_str(mm) in want.split(" + ")}
assert elem_str(U, X) == want, elem_str(U, X)
K = kernel_vectors(U, X, 3, 14, "L")
print(f"x = {want}; dim ker(y -> x y on u_14) = {len(K)}")
Y = min(K, key=len)
print(f"y has {len(Y)} PBW monomials; u(L): x*y = {len(U.mul(X, Y))} terms  ({time.time() - t0:.0f}s)", flush=True)
m = 18
Jn = Jennings(m, U)
Xt, Yt = lift(Jn, X), lift(Jn, Y)
print(f"lifts: |x~| = {len(Xt)}, |y~| = {len(Yt)}  ({time.time() - t0:.0f}s)", flush=True)
prod = gmul(Jn.R, Xt, Yt)
print(f"|x~ y~| = {len(prod)}  ({time.time() - t0:.0f}s)", flush=True)
co = Jn.coeffs(prod, 17)
byw = Counter(U.mdeg(c) for c in co)
print(f"nonzero Jennings coordinates of x~y~ of weight <= 17: {dict(byw)}  ({time.time() - t0:.0f}s)")
# control: a y' NOT in the kernel must give a nonzero weight-17 part
Yc = {min(U.monos(14))}
co2 = Jn.coeffs(gmul(Jn.R, Xt, lift(Jn, Yc)), 17)
print(f"control (y' = single monomial): weights of nonzero coords <= 17: {dict(Counter(U.mdeg(c) for c in co2))}; "
      f"matches u(L) product: {set(c for c in co2 if U.mdeg(c) == 17) == U.mul(X, Yc)}  ({time.time() - t0:.0f}s)")
