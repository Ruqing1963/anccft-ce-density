"""Structure of the zero divisors found in u(L) (F8 model)."""
import itertools
import time

from ulie import ULie
from zdtools import elem_from_bits, elem_str, basis_matrices, batch_search
from verify_zd import kernel_vectors

U = ULie("F8", Dl=20)
V = [elem_from_bits(U, 4, g) for g in (4005, 524823, 527794, 1310958, 1314635, 1835769, 1838428)]
print("V = 7 nonzero elements of a subspace of u_4; check it is closed under addition:",
      all(any(set(a) ^ set(b) == set(c) for c in V) for a, b in itertools.combinations(V, 2)))
print("all products v*w (v, w in V) zero:", all(not U.mul(a, b) for a in V for b in V))

# commutators with degree-1 letters
for v in V[:1]:
    for i in range(3):
        e = {1 << i}
        c = U.mul(e, v) ^ U.mul(v, e)
        print(f"[x1.{1 << i}, v] = {len(c)} terms")

# scaling automorphism lambda = alpha: x_{k,b} -> N_k(alpha) b, N_k = alpha^(2^k-1)
from common import F8, ALPHA


def auto_letter(k, b, lam_pow, frob):
    a = b
    for _ in range(frob):
        a = F8[a][a]
    n = ALPHA[(lam_pow * (2 ** k - 1)) % 7]
    return U.elem(k, F8[a][n])


def apply_auto(X, lam_pow, frob):
    out = set()
    for mono in X:
        cur = {0}
        letters = [i for i in range(U.nl) if (mono >> i) & 1]
        for i in letters:            # product in PBW order of images
            k, b = U.letters[i]
            img = {1 << j for j in auto_letter(k, b, lam_pow, frob)}
            cur = U.mul(cur, img)
        out ^= cur
    return out


Vs = [frozenset(v) for v in V]
for lp in range(1, 7):
    imgs = [frozenset(apply_auto(v, lp, 0)) for v in V]
    print(f"scaling by alpha^{lp}: V -> V ? {set(imgs) == set(Vs)}")
for fr in (1, 2):
    imgs = [frozenset(apply_auto(v, 0, fr)) for v in V]
    print(f"Galois a->a^(2^{fr}): V -> V ? {set(imgs) == set(Vs)}")
# sanity: the automorphism maps sigma(F) to?
SF = {7, 17, 20, 33, 34}
print("sigma(F) fixed by Galois:", apply_auto(SF, 0, 1) == SF, "; by alpha-scaling:", apply_auto(SF, 1, 0) == SF)

# is v a product of lower-degree elements?  rank of u_1*u_3 + u_2*u_2 + u_3*u_1 image in u_4 = all of u_4?
# (u(L) is generated in degree 1, so u_4 = u_1 u_3 automatically.)

# (3,14) zero divisors: kernels
t = time.time()
B, nc = basis_matrices(U, 3, 14, "L")
mr, cnt, h = batch_search(B, 512, 64)
hs = sorted(set(int(x) for x in h.ravel() if x >= 0))
print(f"(3,14): {cnt} non-injective x in u_3, min rank {mr} / {B.shape[1]} ({time.time() - t:.0f}s)")
for g in hs[:14]:
    X = elem_from_bits(U, 3, g)
    K = kernel_vectors(U, X, 3, 14, "L")
    print(f"  x = {elem_str(U, X)}   dim ker on u_14 = {len(K)}, |y| = {[len(k) for k in K]}")
