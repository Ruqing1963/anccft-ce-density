r"""Generic (p, d) engine for u_m = u(L / L_{>=m}) over F_p, L = positive part of F_p^d[phi; s] modulo T = t F_p[t],
t = (1,...,1) phi^d  (split model of gr of the pro-p group attached to the degree-d cyclic algebra over F_p((t))).

Letters (k, i): e_i phi^k, k = 1..m-1, i in Z/d; for d | k only i = 0..d-2 (e_{d-1} = -(e_0+...+e_{d-2}) mod T).
Bracket:   [e_a phi^i, e_b phi^j] = (delta_{a, b+i} e_a - delta_{b, a+j} e_b) phi^{i+j}.
p-map:     (e_r phi^k)^[p] = [d | k] e_r phi^{pk}.
Q-weight:  wt(e_i phi^k) = sum_{s<k} beta_{(i-s) mod d}.
PBW: letters ordered by DEGREE DESCENDING then i ascending (position 0 = highest degree); a monomial is the product of its
letters in increasing position, exponents 0..p-1.  Monomials are tuples of exponents.

sigma(p,d, orient=+1) = sum over cyclic rotations of the Coxeter word e_r e_{r+1} ... e_{r+d-1}  (e_i = y_{1,i}),
orient=-1 uses e_r e_{r-1} ... e_{r-d+1}.
"""
import sys
import numpy as np
from numba import njit

sys.setrecursionlimit(100000)


class Fam:
    def __init__(self, p, d, m):
        self.p, self.d, self.m = p, d, m
        self.letters = [(k, i) for k in range(m - 1, 0, -1) for i in (range(d) if k % d else range(d - 1))]
        self.nl = len(self.letters)
        self.pos = {l: n for n, l in enumerate(self.letters)}
        self.ldeg = [k for k, i in self.letters]
        self.lwt = [self._wt(k, i) for k, i in self.letters]
        self.br = {}
        for a, (i, x) in enumerate(self.letters):
            for b, (j, y) in enumerate(self.letters):
                if a <= b:
                    continue           # only needed for h=a > z=b (positions): [h, z]
                v = [0] * d
                if x == (y + i) % d:
                    v[x] += 1
                if y == (x + j) % d:
                    v[y] -= 1
                self.br[(a, b)] = self._vec(i + j, v)
        self.pm = {}
        for a, (k, r) in enumerate(self.letters):
            v = [0] * d
            if k % d == 0:
                v[r] = 1
            self.pm[a] = self._vec(p * k, v)
        self.top = (p - 1) * sum(self.ldeg)
        self.memo = {}
        self.memo_cap = 1_500_000

    def _wt(self, k, i):
        w = [0] * self.d
        for s in range(k):
            w[(i - s) % self.d] += 1
        return tuple(w)

    def _vec(self, k, v):
        """vector v over e_0..e_{d-1} in degree k -> list of (letter position, coeff) ; reduces mod T"""
        p, d = self.p, self.d
        if k >= self.m:
            return []
        v = [x % p for x in v]
        if k % d == 0 and v[d - 1]:
            c = v[d - 1]
            v = [(x - c) % p for x in v]
        out = []
        for i in range(d):
            if v[i]:
                assert not (k % d == 0 and i == d - 1)
                out.append((self.pos[(k, i)], v[i]))
        return out

    # ------------------------------------------------------------------ products
    def mul1(self, mono, z):
        """mono (tuple of exponents) * letter z  -> dict mono -> coeff"""
        key = (mono, z)
        r = self.memo.get(key)
        if r is not None:
            return r
        p = self.p
        h = -1
        for t in range(self.nl - 1, -1, -1):
            if mono[t]:
                h = t
                break
        if h < z:
            l = list(mono)
            l[z] = 1
            res = {tuple(l): 1}
        elif h == z:
            l = list(mono)
            if l[h] + 1 < p:
                l[h] += 1
                res = {tuple(l): 1}
            else:
                l[h] = 0
                base = tuple(l)
                res = {}
                for c, cf in self.pm[h]:
                    _acc(res, self.mul1(base, c), cf, p)
        else:
            l = list(mono)
            l[h] -= 1
            base = tuple(l)
            res = {}
            # base * h * z = base * z * h + base * [h, z]
            for mo, cf in self.mul1(base, z).items():
                _acc(res, self.mul1(mo, h), cf, p)
            for c, cf in self.br[(h, z)]:
                _acc(res, self.mul1(base, c), cf, p)
        if len(self.memo) > self.memo_cap:
            self.memo.clear()
        self.memo[key] = res
        return res

    def mulword(self, mono, word):
        cur = {mono: 1}
        for z in word:
            nxt = {}
            for mo, cf in cur.items():
                _acc(nxt, self.mul1(mo, z), cf, self.p)
            cur = nxt
            if not cur:
                break
        return cur

    def mul(self, X, Y):
        """X, Y: dicts mono->coeff"""
        out = {}
        for b, cb in Y.items():
            word = []
            for t in range(self.nl):
                word += [t] * b[t]
            for a, ca in X.items():
                _acc(out, self.mulword(a, word), ca * cb, self.p)
        return out

    # ------------------------------------------------------------------ elements
    def one(self):
        return tuple([0] * self.nl)

    def y(self, k, i):
        if k >= self.m:
            return None
        return self.pos[(k, i)]

    def sigma(self, orient=1):
        """list of (word, coeff); cyclic sum of the Coxeter word"""
        d = self.d
        out = []
        for r in range(d):
            w = [self.y(1, (r + orient * s) % d) if (1 % d or True) else None for s in range(d)]
            out.append((w, 1))
        return out

    def elem(self, terms):
        """terms: list of (word, coeff) -> dict"""
        res = {}
        for w, c in terms:
            if any(x is None for x in w):
                continue
            _acc(res, self.mulword(self.one(), w), c, self.p)
        return res

    def wt(self, mono):
        w = [0] * self.d
        for t, a in enumerate(mono):
            if a:
                for s in range(self.d):
                    w[s] += a * self.lwt[t][s]
        return tuple(w)

    def deg(self, mono):
        return sum(a * self.ldeg[t] for t, a in enumerate(mono))

    def mstr(self, mono):
        s = []
        for t, a in enumerate(mono):
            if a:
                k, i = self.letters[t]
                s.append(f"y{k}.{i}" + (f"^{a}" if a > 1 else ""))
        return "*".join(s) if s else "1"

    # ------------------------------------------------------------------ blocks
    def monomials_by_block(self, degs=None):
        """enumerate all monomials (optionally only principal degrees in degs) grouped by Q-weight"""
        p, nl = self.p, self.nl
        blocks = {}
        # DFS over letters with pruning on degree
        maxdeg = max(degs) if degs else self.top
        degset = set(degs) if degs else None
        suffix = [0] * (nl + 1)
        for t in range(nl - 1, -1, -1):
            suffix[t] = suffix[t + 1] + (p - 1) * self.ldeg[t]
        mindeg = min(degs) if degs else 0
        cur = [0] * nl
        wt = [0] * self.d

        def rec(t, dg):
            if dg > maxdeg or dg + suffix[t] < mindeg:
                return
            if t == nl:
                if degset is None or dg in degset:
                    blocks.setdefault(tuple(wt), []).append(tuple(cur))
                return
            amax = min(p - 1, (maxdeg - dg) // self.ldeg[t])
            for a in range(amax + 1):
                cur[t] = a
                if a:
                    for s in range(self.d):
                        wt[s] += self.lwt[t][s]
                rec(t + 1, dg + a * self.ldeg[t])
            for s in range(self.d):
                wt[s] -= amax * self.lwt[t][s]
            cur[t] = 0

        rec(0, 0)
        return blocks


def _acc(res, D, cf, p):
    for k, v in D.items():
        x = (res.get(k, 0) + v * cf) % p
        if x:
            res[k] = x
        else:
            res.pop(k, None)


@njit(cache=True)
def rank_mod_p(A, p):
    A = A.copy()
    nr, nc = A.shape
    r = 0
    inv = np.zeros(p, dtype=np.int64)
    for x in range(1, p):
        for y in range(1, p):
            if (x * y) % p == 1:
                inv[x] = y
    for c in range(nc):
        piv = -1
        for i in range(r, nr):
            if A[i, c] % p:
                piv = i
                break
        if piv < 0:
            continue
        if piv != r:
            for j in range(nc):
                t = A[r, j]
                A[r, j] = A[piv, j]
                A[piv, j] = t
        iv = inv[A[r, c] % p]
        for j in range(nc):
            A[r, j] = (A[r, j] * iv) % p
        for i in range(nr):
            if i != r and A[i, c] % p:
                f = A[i, c] % p
                for j in range(c, nc):
                    A[i, j] = (A[i, j] - f * A[r, j]) % p
        r += 1
        if r == nr:
            break
    return r


def block_matrix(F, rows, cols_index, sig_terms):
    """rows: list of monos; returns int64 matrix of rows*sigma in basis cols_index (dict mono->col)"""
    M = np.zeros((len(rows), len(cols_index)), dtype=np.int64)
    for r, mo in enumerate(rows):
        acc = {}
        for w, c in sig_terms:
            _acc(acc, F.mulword(mo, w), c, F.p)
        for mo2, v in acc.items():
            M[r, cols_index[mo2]] = v
    return M


def right_kernel_dims(F, sig_terms, degs=None, verbose=False):
    """dim ker(R_sigma) on each principal degree n in degs (all if None) -> dict n -> (dim u_n, dim ker)"""
    d = F.d
    delta = tuple([1] * d)
    sdeg = d
    if degs is not None:
        src = F.monomials_by_block(list(degs))
        tgt = F.monomials_by_block(sorted(set(n + sdeg for n in degs)))
    else:
        src = F.monomials_by_block()
        tgt = src
    out = {}
    for g, rows in sorted(src.items()):
        n = sum(g)
        gt = tuple(a + 1 for a in g)
        cols = tgt.get(gt, [])
        ci = {mo: j for j, mo in enumerate(cols)}
        if cols:
            M = block_matrix(F, rows, ci, sig_terms)
            rk = rank_mod_p(M, F.p)
        else:
            rk = 0
        dimk = len(rows) - rk
        a = out.setdefault(n, [0, 0])
        a[0] += len(rows)
        a[1] += dimk
    return out
