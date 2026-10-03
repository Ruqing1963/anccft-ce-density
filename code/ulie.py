"""Graded restricted Lie algebra L = gr P and its restricted enveloping algebra u(L) in PBW normal form.

L_k = E (3 does not divide k) or E/F2*1 (3 | k), [a phi^i, b phi^j] = (a s^i(b) + b s^j(a)) phi^{i+j},
(a phi^k)^[2] = a s^k(a) phi^{2k}.  Two models of (E, s):
  'F8'    : E = F8 = F2(alpha), s = Frobenius a -> a^2 (the division-algebra case of the paper);
            basis {1, alpha, alpha^2} (ints 1, 2, 4) for 3 !| k, {alpha, alpha^2} for 3 | k.
  'split' : E = F2 x F2 x F2 (bits), componentwise product, s = cyclic shift of the bits
            (matrix-algebra / split case); basis {e0, e1, e2} for 3 !| k, {e0, e1} mod (1,1,1) for 3 | k.
Letters x_{k,i} are ordered by (k, i); monomials are int bitmasks over letter indices.
Truncation: letters of degree <= Dl (L/L_{>Dl}); products of total degree > maxdeg are dropped.
"""
import sys
from functools import lru_cache

from common import F8 as _F8, FROB as _FROB

sys.setrecursionlimit(100000)


class Model:
    def __init__(self, kind):
        self.kind = kind
        if kind == "F8":
            self.mul = lambda a, b: _F8[a][b]
            self.s = lambda a: _FROB[a]
            self.one = 1
        elif kind == "split":
            self.mul = lambda a, b: a & b
            self.s = lambda a: ((a << 1) | (a >> 2)) & 7
            self.one = 7
        else:
            raise ValueError(kind)

    def sig(self, a, i):
        for _ in range(i % 3):
            a = self.s(a)
        return a

    def basis(self, k):
        if k % 3:
            return [1, 2, 4]
        return [2, 4] if self.kind == "F8" else [1, 2]

    def coords(self, k, a):
        """coordinates of a in E (or E/F2) w.r.t. basis(k), as list of 0/1"""
        if k % 3:
            return [(a >> b) & 1 for b in range(3)]
        if self.kind == "F8":
            if a & 1:
                a ^= 1
            return [(a >> 1) & 1, (a >> 2) & 1]
        if a & 4:
            a ^= 7
        return [a & 1, (a >> 1) & 1]


class ULie:
    def __init__(self, kind="F8", Dl=12, maxdeg=None):
        self.M = Model(kind)
        self.kind = kind
        self.Dl = Dl
        self.maxdeg = Dl if maxdeg is None else maxdeg
        self.letters = [(k, b) for k in range(1, Dl + 1) for b in self.M.basis(k)]
        self.nl = len(self.letters)
        self.lidx = {l: i for i, l in enumerate(self.letters)}
        self.ldeg = [k for (k, b) in self.letters]
        self.first = {}
        for i, (k, b) in enumerate(self.letters):
            self.first.setdefault(k, i)
        M = self.M
        self.br = {}
        for i, (k, a) in enumerate(self.letters):
            for j, (l, b) in enumerate(self.letters):
                if i > j:
                    self.br[(i, j)] = self.elem(k + l, M.mul(a, M.sig(b, k)) ^ M.mul(b, M.sig(a, l)))
        self.sq = [self.elem(2 * k, M.mul(a, M.sig(a, k))) for (k, a) in self.letters]
        self.memo = {}

    def elem(self, k, a):
        """element a phi^k of L as tuple of letter indices (empty if k > Dl)"""
        if k > self.Dl:
            return ()
        c = self.M.coords(k, a)
        f = self.first[k]
        return tuple(f + t for t, ci in enumerate(c) if ci)

    def mdeg(self, mono):
        d = 0
        while mono:
            low = mono & -mono
            d += self.ldeg[low.bit_length() - 1]
            mono ^= low
        return d

    def lmul(self, z, mono, dmono=None):
        """x_z * mono  -> frozenset of monomials (F2 coefficients)"""
        key = (z, mono)
        r = self.memo.get(key)
        if r is not None:
            return r
        if dmono is None:
            dmono = self.mdeg(mono)
        if self.ldeg[z] + dmono > self.maxdeg:
            r = frozenset()
        elif mono == 0:
            r = frozenset((1 << z,))
        else:
            low = mono & -mono
            y = low.bit_length() - 1
            rest = mono ^ low
            drest = dmono - self.ldeg[y]
            if z < y:
                r = frozenset((mono | (1 << z),))
            elif z == y:
                acc = set()
                for w in self.sq[z]:
                    acc ^= self.lmul(w, rest, drest)
                r = frozenset(acc)
            else:
                acc = set(t | low for t in self.lmul(z, rest, drest))
                for w in self.br[(z, y)]:
                    acc ^= self.lmul(w, rest, drest)
                r = frozenset(acc)
        self.memo[key] = r
        return r

    def mul_mono(self, A, B):
        """monomial A times monomial B"""
        cur = {B}
        letters = []
        while A:
            low = A & -A
            letters.append(low.bit_length() - 1)
            A ^= low
        for z in reversed(letters):
            nxt = set()
            for t in cur:
                nxt ^= self.lmul(z, t)
            cur = nxt
        return cur

    def mul(self, X, Y):
        """X, Y: sets of monomials"""
        out = set()
        for a in X:
            for b in Y:
                out ^= self.mul_mono(a, b)
        return out

    # ---------------------------------------------------------------- degree pieces
    def monos(self, d, maxletter=None):
        """all PBW monomials of degree d (letters of degree <= Dl), sorted"""
        if not hasattr(self, "_mcache"):
            self._mcache = {}
        if d in self._mcache:
            return self._mcache[d]
        res = []

        def rec(start, rem, mask):
            if rem == 0:
                res.append(mask)
                return
            for i in range(start, self.nl):
                k = self.ldeg[i]
                if k > rem:
                    break
                rec(i + 1, rem - k, mask | (1 << i))
        rec(0, d, 0)
        res.sort()
        self._mcache[d] = res
        return res

    def index(self, d):
        if not hasattr(self, "_icache"):
            self._icache = {}
        if d not in self._icache:
            self._icache[d] = {mm: i for i, mm in enumerate(self.monos(d))}
        return self._icache[d]

    def mono_str(self, mono):
        s = []
        while mono:
            low = mono & -mono
            i = low.bit_length() - 1
            k, b = self.letters[i]
            s.append(f"x{k}.{b}")
            mono ^= low
        return "*".join(s) if s else "1"
