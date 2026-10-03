"""Task C1: the isomorphism L (x) F8  ~=  L_split (x) F8  (= positive part of the loop algebra pgl_3 = sl_3 (x) F8[t^{+-1}],
principal grading = path algebra of the cyclic 3-quiver modulo cycles), and sigma(F) in the split root-vector basis.

psi : F8[phi;s] (x) F8 -> F8^3[phi; shift],  a phi^k (x) lam  |->  (lam*a, lam*s^{-1}(a), lam*s^{-2}(a)) phi^k
(componentwise product, shift s(e_p) = e_{p+1}; s^3 = 1, t = phi^3 central on both sides).
Split letters: (k, e_p) = e_p phi^k ; for 3 | k basis {e_0, e_1}, e_2 == e_0 + e_1 (mod the central t^{k/3}).
Q-weight of e_p phi^k :  sum_{s=0}^{k-1} beta_{(p-s) mod 3}  (vector (n0,n1,n2)); then
e_p phi^i * e_q phi^j = [q == p-i] e_p phi^{i+j} and weights add.
usage: python taskC1_iso.py [D]
"""
import random
import sys
from collections import defaultdict

from common import F8, FROB
from ulie import ULie

INV = [0] * 8
for a in range(1, 8):
    for b in range(1, 8):
        if F8[a][b] == 1:
            INV[a] = b


def sinv(a, i=1):
    for _ in range(i % 3):
        a = FROB[FROB[a]]
    return a


def wt(k, p):
    w = [0, 0, 0]
    for s in range(k):
        w[(p - s) % 3] += 1
    return tuple(w)


class Iso:
    def __init__(self, D, maxdeg=None):
        self.D = D
        self.UF = ULie("F8", Dl=D, maxdeg=maxdeg)
        self.US = ULie("split", Dl=D, maxdeg=maxdeg)
        # image of each F8 letter as {split letter index: F8 coeff}
        self.img = []
        for (k, b) in self.UF.letters:
            v = [b, sinv(b, 1), sinv(b, 2)]          # coefficient of e_0, e_1, e_2
            d = defaultdict(int)
            if k % 3:
                for p in range(3):
                    if v[p]:
                        d[self.US.lidx[(k, 1 << p)]] ^= v[p]
            else:   # e_2 = e_0 + e_1
                c0, c1 = v[0] ^ v[2], v[1] ^ v[2]
                if c0:
                    d[self.US.lidx[(k, 1)]] ^= c0
                if c1:
                    d[self.US.lidx[(k, 2)]] ^= c1
            self.img.append({a: c for a, c in d.items() if c})
        self.lwt = []
        for (k, b) in self.US.letters:
            self.lwt.append(wt(k, b.bit_length() - 1))

    # --------------------------------------------------- Lie level, F8-linear combinations of split letters
    def lie_br(self, X, Y):
        out = defaultdict(int)
        for a, ca in X.items():
            for b, cb in Y.items():
                if a == b:
                    continue
                i, j = (a, b) if a > b else (b, a)
                for w in self.US.br[(i, j)]:
                    out[w] ^= F8[ca][cb]
        return {a: c for a, c in out.items() if c}

    def lie_sq(self, X):
        out = defaultdict(int)
        for a, ca in X.items():
            for w in self.US.sq[a]:
                out[w] ^= F8[ca][ca]
        items = sorted(X.items())
        for s in range(len(items)):
            for t in range(s + 1, len(items)):
                for w, c in self.lie_br({items[s][0]: items[s][1]}, {items[t][0]: items[t][1]}).items():
                    out[w] ^= c
        return {a: c for a, c in out.items() if c}

    def img_elem(self, tup):
        out = defaultdict(int)
        for i in tup:
            for a, c in self.img[i].items():
                out[a] ^= c
        return {a: c for a, c in out.items() if c}

    def check_lie(self):
        UF = self.UF
        nb = ns = 0
        for i in range(UF.nl):
            for j in range(i):
                lhs = self.img_elem(UF.br[(i, j)])
                rhs = self.lie_br(self.img[i], self.img[j])
                assert lhs == rhs, (UF.letters[i], UF.letters[j], lhs, rhs)
                nb += 1
            assert self.img_elem(UF.sq[i]) == self.lie_sq(self.img[i]), UF.letters[i]
            ns += 1
        # bijectivity degree by degree (det over F8 of the coefficient matrix)
        for k in range(1, self.D + 1):
            src = [i for i in range(UF.nl) if UF.ldeg[i] == k]
            dst = [a for a in range(self.US.nl) if self.US.ldeg[a] == k]
            M = [[self.img[i].get(a, 0) for a in dst] for i in src]
            assert f8_rank(M) == len(dst) == len(src)
        return nb, ns

    # --------------------------------------------------- enveloping algebra level
    def img_mono(self, mono):
        """psi(PBW monomial of u(L)) as {split PBW monomial: F8 coeff}"""
        cur = {0: 1}
        lets = []
        while mono:
            low = mono & -mono
            lets.append(low.bit_length() - 1)
            mono ^= low
        for z in lets:            # cur * psi(z)
            nxt = defaultdict(int)
            for t, c in cur.items():
                for a, ca in self.img[z].items():
                    cc = F8[c][ca]
                    for r in self.US.mul_mono(t, 1 << a):
                        nxt[r] ^= cc
            cur = {a: c for a, c in nxt.items() if c}
        return cur

    def img_poly(self, X):
        out = defaultdict(int)
        for x in X:
            for t, c in self.img_mono(x).items():
                out[t] ^= c
        return {a: c for a, c in out.items() if c}

    def smul(self, X, Y):
        out = defaultdict(int)
        for a, ca in X.items():
            for b, cb in Y.items():
                cc = F8[ca][cb]
                for r in self.US.mul_mono(a, b):
                    out[r] ^= cc
        return {a: c for a, c in out.items() if c}

    def qdeg(self, mono):
        w = [0, 0, 0]
        while mono:
            low = mono & -mono
            i = low.bit_length() - 1
            for s in range(3):
                w[s] += self.lwt[i][s]
            mono ^= low
        return tuple(w)

    def sstr(self, mono):
        s = []
        while mono:
            low = mono & -mono
            i = low.bit_length() - 1
            k, b = self.US.letters[i]
            s.append(f"y{k}.{b.bit_length() - 1}")
            mono ^= low
        return "*".join(s) if s else "1"


def f8_rank(M):
    M = [row[:] for row in M]
    r = 0
    ncol = len(M[0]) if M else 0
    for c in range(ncol):
        p = next((i for i in range(r, len(M)) if M[i][c]), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        iv = INV[M[r][c]]
        M[r] = [F8[iv][x] for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [x ^ F8[f][y] for x, y in zip(M[i], M[r])]
        r += 1
    return r


F8NAME = {0: "0", 1: "1", 2: "a", 4: "a2", 3: "a3", 6: "a4", 7: "a5", 5: "a6"}   # a^3=a+1 -> 3, a^4=6, a^5=7, a^6=5

if __name__ == "__main__":
    from task3a_sigmaF import SIGMA_F
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    I = Iso(D)
    nb, ns = I.check_lie()
    print(f"[Lie level, letters of degree <= {D}] psi preserves all {nb} brackets and {ns} 2-maps of basis letters; "
          f"psi is bijective in every degree <= {D}  -> L (x) F8 ~= L_split (x) F8 (restricted, graded)")
    print("letter images (degrees 1,2):")
    for i, (k, b) in enumerate(I.UF.letters):
        if k <= 2:
            print(f"   x{k}.{b} -> " + " + ".join(f"{F8NAME[c]}*{I.sstr(1 << a)}" for a, c in sorted(I.img[i].items())))
    print("Q-weights of split letters:", {I.sstr(1 << a): I.lwt[a] for a in range(I.US.nl) if I.US.ldeg[a] <= 6})
    # sigma(F) in the split basis
    sF = I.img_poly(set(SIGMA_F))
    print(f"\nsigma(F)' = psi(sigma F) has {len(sF)} split PBW terms:")
    byq = defaultdict(list)
    for t, c in sorted(sF.items()):
        byq[I.qdeg(t)].append((t, c))
    for q, terms in sorted(byq.items()):
        print(f"  Q-degree {q}: " + " + ".join(f"{F8NAME[c]}*{I.sstr(t)}" for t, c in terms))
    print("Q-homogeneous of degree delta=(1,1,1):", list(byq) == [(1, 1, 1)])
    # algebra-level check psi(y sigmaF) = psi(y) sigmaF' and psi(y z) = psi(y) psi(z) on random elements
    rnd = random.Random(7)
    nck = 0
    for trial in range(60):
        a = rnd.randint(1, 7)
        b = rnd.randint(1, D - a if D - a >= 1 else 1)
        if a + b > D:
            continue
        Y = set(rnd.sample(I.UF.monos(a), min(3, len(I.UF.monos(a)))))
        Z = set(rnd.sample(I.UF.monos(b), min(3, len(I.UF.monos(b)))))
        lhs = I.img_poly(I.UF.mul(Y, Z))
        rhs = I.smul(I.img_poly(Y), I.img_poly(Z))
        assert lhs == rhs, (a, b)
        nck += 1
    for n in range(0, D - 2):
        ms = I.UF.monos(n)
        for y in rnd.sample(ms, min(4, len(ms))):
            lhs = I.img_poly(I.UF.mul({y}, set(SIGMA_F)))
            rhs = I.smul(I.img_mono(y), sF)
            assert lhs == rhs, n
            nck += 1
    print(f"[algebra level] psi(YZ) = psi(Y)psi(Z) and psi(y sigmaF) = psi(y) sigmaF' on {nck} random products (deg <= {D}): OK")
