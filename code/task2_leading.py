"""Task 2: u(L) sanity checks and the leading form sigma(F) in u(L)_3.

Jennings basis of F2[G_m]: letters g_{k,b} = 1 + b phi^k (k < m, b in basis(k), ordered like the
PBW letters of ulie.ULie), basis elements J(S) = prod_{s in S, ordered} (g_s - 1), weight = sum deg.
Every g in G_m is uniquely g = prod_{s in S(g), ordered} g_s (peel layer by layer), hence
g = sum_{T subset S(g)} J(T): the Jennings coordinates of g are the subsets of S(g).
"""
import itertools
import random
import sys
import time
from collections import Counter

import numpy as np

from common import TowerRing, F_support
from ulie import ULie


def series_dims(D, kind="F8"):
    """coefficients of prod_k (1+z^k)^{d_k} and prod_{k odd} (1-z^k)^{-d_k}"""
    d = lambda k: 3 if k % 3 else 2
    p = [1] + [0] * D
    for k in range(1, D + 1):
        for _ in range(d(k)):
            for j in range(D, k - 1, -1):
                p[j] += p[j - k]
    q = [1] + [0] * D
    for k in range(1, D + 1, 2):
        for _ in range(d(k)):
            for j in range(k, D + 1):       # multiply by 1/(1-z^k)
                q[j] += q[j - k]
    return p, q


class Jennings:
    def __init__(self, m, U):
        self.m, self.R, self.U = m, TowerRing(m), U
        self.let = [(i, k, b) for i, (k, b) in enumerate(U.letters) if k < m]
        self.gel = {}
        for (i, k, b) in self.let:
            e = [0] * m
            e[0] = 1
            e[k] = b
            self.gel[i] = self.R.normalize(tuple(e))
        self.cache = {}

    def S(self, g):
        if g in self.cache:
            return self.cache[g]
        R, U = self.R, self.U
        h = g
        out = []
        for k in range(1, self.m):
            c = U.M.coords(k, h[k])
            idx = [U.first[k] + t for t, ci in enumerate(c) if ci]
            P = R.one()
            for i in idx:
                P = R.nmul(P, self.gel[i])
            h = R.nmul(R.ninv(P), h)
            assert h[k] == 0, (g, k, h)
            out += idx
        assert all(x == 0 for x in h[1:])
        self.cache[g] = tuple(out)
        return self.cache[g]

    def coeffs(self, elems, maxw):
        """Jennings coordinates (weight <= maxw) of sum of group elements (Counter or iterable)"""
        cnt = Counter()
        U = self.U
        for g in elems:
            S = self.S(g)
            # subsets of S with weight <= maxw
            def rec(pos, w, mask):
                cnt[mask] ^= 1
                for t in range(pos, len(S)):
                    dw = U.ldeg[S[t]]
                    if w + dw <= maxw:
                        rec(t + 1, w + dw, mask | (1 << S[t]))
            rec(0, 0, 0)
        return {k for k, v in cnt.items() if v}

    def J(self, mask):
        """group-algebra element prod (g_s - 1) as set of group elements (F2 coefficients)"""
        R = self.R
        cur = {R.one()}
        while mask:
            low = mask & -mask
            i = low.bit_length() - 1
            mask ^= low
            nxt = Counter()
            for x in cur:
                nxt[R.nmul(x, self.gel[i])] ^= 1
                nxt[x] ^= 1
            cur = {k for k, v in nxt.items() if v}
        return cur


def gmul(R, A, B):
    c = Counter()
    for a in A:
        for b in B:
            c[R.nmul(a, b)] ^= 1
    return {k for k, v in c.items() if v}


def sigmaF(U, m):
    Jn = Jennings(m, U)
    supp = F_support(m)
    co = Jn.coeffs(supp, 3)
    byw = Counter(U.mdeg(c) for c in co)
    lead = sorted(c for c in co if U.mdeg(c) == 3)
    return byw, lead, len(supp)


if __name__ == "__main__":
    t0 = time.time()
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    U = ULie("F8", Dl=D)
    p, q = series_dims(D)
    dims = [len(U.monos(d)) for d in range(D + 1)]
    print("dim u(L)_j, j=0..D:", dims)
    assert dims == p == q
    print("matches prod(1+z^k)^{d_k} = prod_{k odd}(1-z^k)^{-d_k}")

    # generation of L by L_1 as restricted Lie algebra (degreewise spans)
    def lie_vec(elem):
        return sum(1 << i for i in elem)
    span = {1: [1 << i for i in range(3)]}
    for n in range(2, D + 1):
        vecs = []
        for i in range(1, n):
            for a in span[i]:
                for b in span.get(n - i, []):
                    # bracket of combinations: bilinear
                    v = 0
                    for x in range(U.nl):
                        if (a >> x) & 1:
                            for y in range(U.nl):
                                if (b >> y) & 1 and x != y:
                                    v ^= lie_vec(U.br[(max(x, y), min(x, y))])
                    vecs.append(v)
        if n % 2 == 0:
            for a in span[n // 2]:
                # 2-map of a combination: sum x^[2] + sum_{x<y} [x,y]
                xs = [x for x in range(U.nl) if (a >> x) & 1]
                v = 0
                for x in xs:
                    v ^= lie_vec(U.sq[x])
                for x, y in itertools.combinations(xs, 2):
                    v ^= lie_vec(U.br[(y, x)])
                vecs.append(v)
        # row reduce
        basis = []
        for v in vecs:
            for bv in basis:
                v = min(v, v ^ bv)
            if v:
                basis.append(v)
                basis.sort(reverse=True)
        span[n] = basis
        full = 3 if n % 3 else 2
        assert len(basis) == full, (n, len(basis))
    print(f"L is generated by L_1 as a restricted Lie algebra up to degree {D} "
          "(so the weight filtration = I-adic filtration, Jennings series = P_k)")

    # associativity on random triples
    rng = random.Random(7)
    nt = 0
    for _ in range(300):
        a, b, c = rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 5)
        if a + b + c > D:
            continue
        X = {rng.choice(U.monos(a))}
        Y = {rng.choice(U.monos(b))}
        Z = {rng.choice(U.monos(c))}
        assert U.mul(U.mul(X, Y), Z) == U.mul(X, U.mul(Y, Z))
        nt += 1
    # also exhaustive associativity on all triples of letters/monomials of small degree
    for a, b, c in [(1, 1, 1), (1, 2, 1), (2, 1, 2), (1, 1, 3), (3, 1, 1), (2, 2, 2)]:
        for X in U.monos(a):
            for Y in U.monos(b):
                for Z in U.monos(c):
                    assert U.mul(U.mul({X}, {Y}), {Z}) == U.mul({X}, U.mul({Y}, {Z}))
                    nt += 1
    print(f"associativity verified on {nt} triples")

    # comparison with the group algebra: gr F2[G_m] = u(L/L_{>=m})
    m = 7
    Jn = Jennings(m, U)
    nc = 0
    for a in range(1, 6):
        for b in range(1, 7 - a):
            ms_a, ms_b = U.monos(a), U.monos(b)
            pairs = [(x, y) for x in ms_a for y in ms_b]
            rng.shuffle(pairs)
            for (x, y) in pairs[:40]:
                prod = gmul(Jn.R, Jn.J(x), Jn.J(y))
                co = Jn.coeffs(prod, a + b)
                assert all(U.mdeg(c) == a + b for c in co), "lower weight term"
                assert co == U.mul({x}, {y}), (U.mono_str(x), U.mono_str(y))
                nc += 1
    print(f"group algebra F2[G_7] check: Jennings product J(x)J(y) = PBW product x*y "
          f"mod higher weight, {nc} pairs, deg(x)+deg(y) <= 6")

    # leading form
    res = {}
    for m in (4, 5, 6, 7, 8):
        byw, lead, ns = sigmaF(U, m)
        res[m] = lead
        print(f"m={m}: |supp F|={ns}, # nonzero Jennings coeffs of weight w<=3: {dict(sorted(byw.items()))}")
        assert byw.get(0, 0) == byw.get(1, 0) == byw.get(2, 0) == 0
    assert all(res[m] == res[4] for m in res)
    lead = res[4]
    print("sigma(F) in u(L)_3 (independent of m=4..8), PBW monomials (x{k}.{b}: b=1,2,4 means 1,alpha,alpha^2):")
    print("  sigma(F) = " + " + ".join(U.mono_str(c) for c in lead))
    print("  as masks:", lead)
    print(f"done {time.time() - t0:.1f}s")
