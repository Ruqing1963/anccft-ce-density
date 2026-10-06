r"""Numba engine for u_m = u(L_{p,d} / L_{>=m}) over F_p (same conventions as fam.py).

Monomial = int64 code  sum_t a_t p^t  (t = letter position, position 0 = highest degree; product in increasing t).
Right multiplication by a word is a rewriting system with an explicit stack (coefficients mod p):
  state (S, W[0:L] reversed, c):  z = next letter, h = highest position present in S
    S == 0 or z > h : S += p^z
    z == h          : exponent+1, or if it reaches p: S -= (p-1)p^h and replace z by h^[p] (linear combination)
    z <  h          : S' = S - p^h ;  S*z = S'*z*h + S'*[h,z]
"""
import numpy as np
from numba import njit, prange
import fam

MAXST = 8192
MAXW = 96
MAXT = 8          # max terms in a bracket / p-map


class FamNB:
    def __init__(self, p, d, m):
        F = fam.Fam(p, d, m)
        self.F = F
        self.p, self.d, self.m, self.nl = p, d, m, F.nl
        nl = max(F.nl, 1)
        assert p ** F.nl < 2 ** 62, "code overflow"
        self.P = np.array([p ** t for t in range(nl + 1)], dtype=np.int64)
        self.brl = np.full((nl, nl, MAXT), -1, dtype=np.int64)
        self.brc = np.zeros((nl, nl, MAXT), dtype=np.int64)
        for (a, b), terms in F.br.items():
            for i, (c, cf) in enumerate(terms):
                self.brl[a, b, i] = c
                self.brc[a, b, i] = cf % p
        self.pml = np.full((nl, MAXT), -1, dtype=np.int64)
        self.pmc = np.zeros((nl, MAXT), dtype=np.int64)
        for a, terms in F.pm.items():
            for i, (c, cf) in enumerate(terms):
                self.pml[a, i] = c
                self.pmc[a, i] = cf % p
        self.ldeg = np.array(F.ldeg, dtype=np.int64)
        self.lwt = np.array(F.lwt, dtype=np.int64).reshape(len(F.ldeg), d)

    def sigma_words(self, orient=1):
        ws = [w for w, c in self.F.sigma(orient)]
        L = max(len(w) for w in ws)
        A = np.zeros((len(ws), L), dtype=np.int64)
        for i, w in enumerate(ws):
            A[i, :len(w)] = w
        return A, np.array([len(w) for w in ws], dtype=np.int64), np.ones(len(ws), dtype=np.int64)

    def decode(self, code):
        l = []
        for t in range(self.nl):
            l.append(int(code // self.P[t] % self.p))
        return tuple(l)

    def encode(self, mono):
        return int(sum(a * int(self.P[t]) for t, a in enumerate(mono)))

    # ---------------------------------------------------------- enumeration
    def codes_by_block(self, nlo, nhi):
        """all monomial codes with principal degree in [nlo, nhi], grouped: dict weight-tuple -> sorted int64 array"""
        p, nl = self.p, self.nl
        degs = self.ldeg
        # mixed-radix DFS in numba
        codes, wts = _enum(self.P[:nl], degs, self.lwt, p, nlo, nhi, self.d)
        out = {}
        if codes.size == 0:
            return out
        key = np.zeros(codes.size, dtype=np.int64)
        K = int(self.lwt.sum(axis=0).max() * (p - 1) + 1)
        for s in range(self.d):
            key = key * K + wts[:, s]
        order = np.argsort(key, kind="stable")
        key = key[order]
        codes = codes[order]
        wts = wts[order]
        b = np.flatnonzero(np.diff(key)) + 1
        starts = np.concatenate([[0], b])
        ends = np.concatenate([b, [key.size]])
        for s, e in zip(starts, ends):
            out[tuple(int(x) for x in wts[s])] = np.sort(codes[s:e])
        return out


@njit(cache=True)
def _enum(P, degs, lwt, p, nlo, nhi, d):
    nl = P.size
    suffix = np.zeros(nl + 1, dtype=np.int64)
    for t in range(nl - 1, -1, -1):
        suffix[t] = suffix[t + 1] + (p - 1) * degs[t]
    a = np.full(nl + 1, -1, dtype=np.int64)     # a[t] = -1 : level t not entered yet
    amax = np.zeros(nl + 1, dtype=np.int64)
    w = np.zeros(d, dtype=np.int64)
    dg = 0
    code = 0
    out_c = np.empty(1024, dtype=np.int64)
    out_w = np.empty((1024, d), dtype=np.int64)
    n = 0
    t = 0
    while t >= 0:
        if t == nl:                               # a complete monomial
            if dg >= nlo:
                if n >= out_c.size:
                    nc = np.empty(out_c.size * 2, dtype=np.int64)
                    nc[:n] = out_c[:n]
                    out_c = nc
                    nw = np.empty((out_w.shape[0] * 2, d), dtype=np.int64)
                    nw[:n] = out_w[:n]
                    out_w = nw
                out_c[n] = code
                for s in range(d):
                    out_w[n, s] = w[s]
                n += 1
            t -= 1
            continue
        if a[t] == -1:                            # enter level t with exponent 0
            if dg + suffix[t] < nlo:              # cannot reach nlo any more
                t -= 1
                continue
            am = (nhi - dg) // degs[t]
            amax[t] = am if am < p - 1 else p - 1
            a[t] = 0
            a[t + 1] = -1
            t += 1
            continue
        if a[t] < amax[t]:                        # next exponent at level t
            a[t] += 1
            dg += degs[t]
            code += P[t]
            for s in range(d):
                w[s] += lwt[t, s]
            a[t + 1] = -1
            t += 1
            continue
        k = a[t]                                  # level t exhausted: undo and go up
        dg -= k * degs[t]
        code -= k * P[t]
        for s in range(d):
            w[s] -= k * lwt[t, s]
        a[t] = -1
        t -= 1
    return out_c[:n], out_w[:n]


@njit(cache=True)
def _hi(S, P):
    t = P.size - 2
    while P[t] > S:
        t -= 1
    return t


@njit(cache=True)
def rmul_word(S0, c0, word, wl, P, p, brl, brc, pml, pmc, outS, outC, nout, stS, stW, stL, stC, W):
    """append monomials (with coefficients) of c0 * S0 * word ; returns new nout, or -1 on overflow"""
    stS[0] = S0
    for i in range(wl):
        stW[0, wl - 1 - i] = word[i]
    stL[0] = wl
    stC[0] = c0
    sp = 1
    while sp > 0:
        sp -= 1
        S = stS[sp]
        L = stL[sp]
        c = stC[sp]
        for i in range(L):
            W[i] = stW[sp, i]
        while True:
            if L == 0:
                if nout >= outS.size:
                    return -1
                outS[nout] = S
                outC[nout] = c
                nout += 1
                break
            z = W[L - 1]
            if S == 0:
                S = P[z]
                L -= 1
                continue
            h = _hi(S, P)
            if z > h:
                S += P[z]
                L -= 1
                continue
            if z == h:
                e = (S // P[h]) % p
                if e + 1 < p:
                    S += P[h]
                    L -= 1
                    continue
                S -= (p - 1) * P[h]
                # replace z by h^[p]
                if pml[h, 0] < 0:
                    break
                for t in range(1, pml.shape[1]):
                    if pml[h, t] < 0:
                        break
                    if sp >= stS.size:
                        return -1
                    stS[sp] = S
                    for i in range(L):
                        stW[sp, i] = W[i]
                    stW[sp, L - 1] = pml[h, t]
                    stL[sp] = L
                    stC[sp] = (c * pmc[h, t]) % p
                    sp += 1
                W[L - 1] = pml[h, 0]
                c = (c * pmc[h, 0]) % p
                continue
            # z < h
            S -= P[h]
            for t in range(brl.shape[2]):
                b = brl[h, z, t]
                if b < 0:
                    break
                if sp >= stS.size:
                    return -1
                stS[sp] = S
                for i in range(L):
                    stW[sp, i] = W[i]
                stW[sp, L - 1] = b
                stL[sp] = L
                stC[sp] = (c * brc[h, z, t]) % p
                sp += 1
            if L + 1 > W.size:
                return -1
            W[L - 1] = h
            W[L] = z
            L += 1
    return nout


@njit(parallel=True, cache=True)
def build_block(rows, cols, words, wlen, wcoef, P, p, brl, brc, pml, pmc, left, MAXOUT, nchunk):
    """dense uint8 matrix of rows*sigma (left=0) or sigma*rows (left=1), columns = sorted cols."""
    nr = rows.size
    nc = cols.size
    M = np.zeros((nr, nc), dtype=np.uint8)
    bad = np.zeros(nchunk, dtype=np.int64)
    csz = (nr + nchunk - 1) // nchunk
    for ch in prange(nchunk):
        outS = np.empty(MAXOUT, dtype=np.int64)
        outC = np.empty(MAXOUT, dtype=np.int64)
        stS = np.empty(MAXST, dtype=np.int64)
        stW = np.empty((MAXST, MAXW), dtype=np.int64)
        stL = np.empty(MAXST, dtype=np.int64)
        stC = np.empty(MAXST, dtype=np.int64)
        Wb = np.empty(MAXW, dtype=np.int64)
        tmpw = np.empty(MAXW, dtype=np.int64)
        acc = np.zeros(nc, dtype=np.int64)
        lo = ch * csz
        hi = min(nr, lo + csz)
        for r in range(lo, hi):
            n = 0
            for w in range(wlen.size):
                if left == 0:
                    n = rmul_word(rows[r], wcoef[w], words[w, :wlen[w]], wlen[w], P, p, brl, brc, pml, pmc,
                                  outS, outC, n, stS, stW, stL, stC, Wb)
                else:
                    L = 0
                    for i in range(wlen[w]):
                        tmpw[L] = words[w, i]
                        L += 1
                    S = rows[r]
                    t = 0
                    while S > 0:
                        e = S % p
                        for _ in range(e):
                            tmpw[L] = t
                            L += 1
                        S //= p
                        t += 1
                    n = rmul_word(0, wcoef[w], tmpw[:L], L, P, p, brl, brc, pml, pmc, outS, outC, n, stS,
                                  stW, stL, stC, Wb)
                if n < 0:
                    break
            if n < 0:
                bad[ch] += 1
                continue
            for i in range(n):
                j = np.searchsorted(cols, outS[i])
                if j >= nc or cols[j] != outS[i]:
                    bad[ch] += 1000000
                    continue
                acc[j] += outC[i]
            for i in range(n):
                j = np.searchsorted(cols, outS[i])
                if j < nc and cols[j] == outS[i] and acc[j] != -1:
                    M[r, j] = acc[j] % p
                    acc[j] = -1
            for i in range(n):
                j = np.searchsorted(cols, outS[i])
                if j < nc and cols[j] == outS[i]:
                    acc[j] = 0
    return M, bad.sum()


@njit(parallel=True, cache=True)
def rank_mod_p(M, p):
    """rank over F_p of a uint8 matrix (destroys M). Forward elimination, rows updated in parallel."""
    nr, nc = M.shape
    inv = np.zeros(p, dtype=np.uint8)
    for x in range(1, p):
        for y in range(1, p):
            if (x * y) % p == 1:
                inv[x] = y
    MAG = np.uint32((65536 + p - 1) // p)
    r = 0
    for c in range(nc):
        if r == nr:
            break
        piv = -1
        for i in range(r, nr):
            if M[i, c] != 0:
                piv = i
                break
        if piv < 0:
            continue
        if piv != r:
            for j in range(c, nc):
                tmp = M[r, j]
                M[r, j] = M[piv, j]
                M[piv, j] = tmp
        iv = inv[M[r, c]]
        if iv != 1:
            for j in range(c, nc):
                M[r, j] = np.uint8((np.uint32(M[r, j]) * iv) % p)
        for i in prange(r + 1, nr):
            f = M[i, c]
            if f != 0:
                g = np.uint32(p - f)
                for j in range(c, nc):
                    t = np.uint32(M[i, j]) + g * np.uint32(M[r, j])
                    q = (t * MAG) >> np.uint32(16)
                    M[i, j] = np.uint8(t - q * np.uint32(p))
        r += 1
    return r


def kernel_by_degree(E, nlo, nhi, orient=1, left=False, verbose=False, maxout=1 << 20):
    """dict n -> (dim, ker) for R_sigma (or L_sigma) on (u_m)_n, n in [nlo, nhi]"""
    d = E.d
    src = E.codes_by_block(nlo, nhi)
    tgt = E.codes_by_block(nlo + d, nhi + d)
    words, wlen, wcoef = E.sigma_words(orient)
    out = {}
    for g in sorted(src, key=lambda g: (sum(g), g)):
        rows = src[g]
        n = sum(g)
        cols = tgt.get(tuple(a + 1 for a in g), np.zeros(0, dtype=np.int64))
        if cols.size:
            nch = max(1, min(256, rows.size // 16))
            M, bad = build_block(rows, cols, words, wlen, wcoef, E.P, E.p, E.brl, E.brc, E.pml, E.pmc,
                                 1 if left else 0, maxout, nch)
            if bad:
                raise RuntimeError(f"overflow/bad column in block {g}: {bad}")
            rk = rank_mod_p(M, E.p)
        else:
            rk = 0
        k = rows.size - rk
        o = out.setdefault(n, [0, 0, 0])
        o[0] += rows.size
        o[1] += k
        o[2] = max(o[2], rows.size)
        if verbose and rows.size > 2000:
            print(f"   block {g}: {rows.size} -> {cols.size}, ker {k}", flush=True)
    return out
