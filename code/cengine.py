r"""Task C: compiled PBW engine for u_m^split = u(L_split / L_{>=m}) over F2, with the Q-grading.

L_split = positive part of F2^3[phi; s] modulo the central t^j = (1,1,1) phi^{3j}  (path algebra of the cyclic 3-quiver
mod cycles-sum); by taskC1_iso.py, L (x) F8 ~= L_split (x) F8 and sigma(F) |-> sigma' (F2 coefficients):
    sigma' = y1.0*y1.1*y1.2 + y1.2*y2.1 + y1.0*y2.2 + y3.1          (yk.p = e_p phi^k; products are products in u)
Letters: (k, p), k = 1..m-1, p in {0,1,2} (3 !| k) or {0,1} (3 | k; e_2 == e_0 + e_1).
PBW basis: monomials = bitmasks; bit position order = DEGREE DESCENDING, then p ascending, and a monomial is the product
of its letters in increasing bit position (so right multiplication by low-degree letters is cheap).
Right multiplication of a monomial by a word is done by a rewriting system (no memo):
  state (S, w1 w2 ...),  h = last letter of S:   z=w1 > h: S|z ;  z == h: S\h, w1 := h^[2] ;
  z < h: (S\h, z h w2 ...) + sum_{c in [h,z]} (S\h, c w2 ...).
Q-weight of e_p phi^k = sum_{s<k} beta_{(p-s) mod 3}.
"""
import numpy as np
from numba import njit, prange

MAXST = 4096
MAXW = 64
MAXOUT = 1 << 18


def wt(k, p):
    w = [0, 0, 0]
    for s in range(k):
        w[(p - s) % 3] += 1
    return tuple(w)


class Split:
    def __init__(self, m):
        self.m = m
        self.letters = [(k, p) for k in range(m - 1, 0, -1) for p in (range(3) if k % 3 else range(2))]
        self.nl = nl = len(self.letters)
        self.pos = {l: i for i, l in enumerate(self.letters)}
        self.ldeg = np.array([k for k, p in self.letters], dtype=np.int64)
        self.lwt = np.array([wt(k, p) for k, p in self.letters], dtype=np.int64)
        self.br = np.full((max(nl, 1), max(nl, 1), 2), -1, dtype=np.int64)
        self.sq = np.full((max(nl, 1), 2), -1, dtype=np.int64)
        for a, (i, p) in enumerate(self.letters):
            for b, (j, q) in enumerate(self.letters):
                if a == b:
                    continue
                v = 0
                if p == (q + i) % 3:
                    v ^= 1 << p
                if q == (p + j) % 3:
                    v ^= 1 << q
                self._store(self.br[a, b], i + j, v)
            v = (1 << p) if i % 3 == 0 else 0
            self._store(self.sq[a], 2 * i, v)
        self.top = int(sum(k for k, p in self.letters))

    def _store(self, arr, k, v):
        if k >= self.m or v == 0:
            return
        if k % 3 == 0 and (v & 4):
            v ^= 7
        t = 0
        for p in range(3):
            if (v >> p) & 1:
                assert not (k % 3 == 0 and p == 2)
                arr[t] = self.pos[(k, p)]
                t += 1

    def letter(self, k, p):
        if k >= self.m:
            return None
        return self.pos[(k, p)]

    def word(self, lets):
        """list of (k,p) -> np array of letter positions, or None if some letter vanishes"""
        out = []
        for k, p in lets:
            if k >= self.m:
                return None
            out.append(self.pos[(k, p)])
        return np.array(out, dtype=np.int64)

    def sigma_words(self):
        """sigma' as list of words (each a product in u)"""
        ws = [[(1, 0), (1, 1), (1, 2)], [(1, 2), (2, 1)], [(1, 0), (2, 2)], [(3, 1)]]
        out = [self.word(w) for w in ws]
        return [w for w in out if w is not None]

    def qdeg(self, mono):
        w = np.zeros(3, dtype=np.int64)
        i = 0
        while mono:
            if mono & 1:
                w += self.lwt[i]
            mono >>= 1
            i += 1
        return tuple(int(x) for x in w)

    def pdeg(self, mono):
        return sum(self.qdeg(mono))

    def mstr(self, mono):
        s = []
        for i in range(self.nl):
            if (mono >> i) & 1:
                k, p = self.letters[i]
                s.append(f"y{k}.{p}")
        return "*".join(s) if s else "1"

    # ---------------------------------------------------------------- products
    def rmul(self, S, word):
        """S * word -> sorted list of monomials (F2 sum)"""
        out = np.empty(MAXOUT, dtype=np.int64)
        n = rmul_word(np.int64(S), np.asarray(word, dtype=np.int64), self.br, self.sq, out, 0, *workbufs())
        if n < 0:
            raise RuntimeError("overflow")
        return cancel(out[:n])

    def mul_monos(self, A, B):
        lets = [i for i in range(self.nl) if (B >> i) & 1]
        return self.rmul(A, lets)

    def mul(self, X, Y):
        acc = {}
        for a in X:
            for b in Y:
                for t in self.mul_monos(a, b):
                    acc[t] = acc.get(t, 0) ^ 1
        return sorted(t for t, c in acc.items() if c)


def workbufs():
    return (np.empty(MAXST, dtype=np.int64), np.empty((MAXST, MAXW), dtype=np.int64),
            np.empty(MAXST, dtype=np.int64), np.empty(MAXW, dtype=np.int64))


def cancel(arr):
    if arr.size == 0:
        return []
    s = np.sort(arr)
    vals, cnt = np.unique(s, return_counts=True)
    return [int(v) for v, c in zip(vals, cnt) if c & 1]


@njit(cache=False)
def _hibit(S):
    b = 62
    while not (S >> b) & 1:
        b -= 1
    return b


@njit(cache=False)
def rmul_word(S0, word, br, sq, out, nout, stS, stW, stL, W):
    """append to out[nout:] the monomials of S0 * word (with multiplicity); returns new nout or -1 on overflow"""
    sp = 0
    L0 = word.size
    stS[0] = S0
    for i in range(L0):            # stored reversed: W[L-1] is the next letter
        stW[0, L0 - 1 - i] = word[i]
    stL[0] = L0
    sp = 1
    while sp > 0:
        sp -= 1
        S = stS[sp]
        L = stL[sp]
        for i in range(L):
            W[i] = stW[sp, i]
        while True:
            if L == 0:
                if nout >= out.size:
                    return -1
                out[nout] = S
                nout += 1
                break
            z = W[L - 1]
            if S == 0:
                S = np.int64(1) << z
                L -= 1
                continue
            h = _hibit(S)
            if z > h:
                S |= np.int64(1) << z
                L -= 1
                continue
            S ^= np.int64(1) << h
            if z == h:
                c0 = sq[h, 0]
                if c0 < 0:
                    break
                c1 = sq[h, 1]
                if c1 >= 0:
                    if sp >= MAXST:
                        return -1
                    stS[sp] = S
                    for i in range(L):
                        stW[sp, i] = W[i]
                    stW[sp, L - 1] = c1
                    stL[sp] = L
                    sp += 1
                W[L - 1] = c0
                continue
            # z < h : bracket terms
            for t in range(2):
                c = br[h, z, t]
                if c < 0:
                    break
                if sp >= MAXST:
                    return -1
                stS[sp] = S
                for i in range(L):
                    stW[sp, i] = W[i]
                stW[sp, L - 1] = c
                stL[sp] = L
                sp += 1
            # main term: S\h * z * h * rest   (reversed storage: ..., h, z)
            if L + 1 > MAXW:
                return -1
            W[L - 1] = h
            W[L] = z
            L += 1
    return nout


@njit(parallel=True, cache=False)
def build_block(rows, words, wlen, br, sq, colidx, key, tkey, ncols, nchunk):
    """packed matrix: row r = rows[r] * sigma' ; columns indexed by colidx[mono]; key[mono] must equal tkey."""
    nr = rows.size
    Wd = (ncols + 63) // 64
    P = np.zeros((nr, Wd), dtype=np.uint64)
    bad = np.zeros(nchunk, dtype=np.int64)
    csz = (nr + nchunk - 1) // nchunk
    for c in prange(nchunk):
        out = np.empty(MAXOUT, dtype=np.int64)
        stS = np.empty(MAXST, dtype=np.int64)
        stW = np.empty((MAXST, MAXW), dtype=np.int64)
        stL = np.empty(MAXST, dtype=np.int64)
        Wb = np.empty(MAXW, dtype=np.int64)
        lo = c * csz
        hi = min(nr, lo + csz)
        for r in range(lo, hi):
            n = 0
            for w in range(wlen.size):
                n = rmul_word(rows[r], words[w, :wlen[w]], br, sq, out, n, stS, stW, stL, Wb)
                if n < 0:
                    bad[c] += 1
                    break
            if n < 0:
                continue
            for i in range(n):
                if key[out[i]] != tkey:
                    bad[c] += 1000000
                    continue
                j = colidx[out[i]]
                P[r, j >> 6] ^= np.uint64(1) << np.uint64(j & 63)
    return P, bad.sum()


def pack_words(ws):
    if not ws:
        return np.zeros((0, 1), dtype=np.int64), np.zeros(0, dtype=np.int64)
    ml = max(len(w) for w in ws)
    A = np.zeros((len(ws), ml), dtype=np.int64)
    for i, w in enumerate(ws):
        A[i, :len(w)] = w
    return A, np.array([len(w) for w in ws], dtype=np.int64)


def all_qdeg(sp):
    """for all 2^nl masks: Q-degree components (int16 arrays)"""
    nl = sp.nl
    N = 1 << nl
    q = [np.zeros(N, dtype=np.int16) for _ in range(3)]
    v = np.arange(N, dtype=np.int64)
    for i in range(nl):
        bit = ((v >> i) & 1).astype(np.int16)
        for s in range(3):
            if sp.lwt[i, s]:
                q[s] += bit * np.int16(sp.lwt[i, s])
    return q


class Blocks:
    """monomials grouped by Q-degree; colidx-style local index arrays"""

    def __init__(self, sp):
        self.sp = sp
        q0, q1, q2 = all_qdeg(sp)
        K = sp.top + 1
        key = (q0.astype(np.int64) * K + q1) * K + q2
        del q0, q1, q2
        order = np.argsort(key, kind="stable")
        ks = key[order]
        bounds = np.flatnonzero(np.diff(ks)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [ks.size]])
        self.K = K
        self.key = key.astype(np.int32)
        self.blocks = {}
        local = np.empty(key.size, dtype=np.int32)
        for s, e in zip(starts, ends):
            kk = int(ks[s])
            g = (kk // (K * K), (kk // K) % K, kk % K)
            self.blocks[g] = order[s:e]          # sorted masks (stable on arange -> increasing)
            local[order[s:e]] = np.arange(e - s, dtype=np.int32)
        self.local = local

    def gkey(self, g):
        return (g[0] * self.K + g[1]) * self.K + g[2]

    def dim(self, g):
        b = self.blocks.get(g)
        return 0 if b is None else b.size
