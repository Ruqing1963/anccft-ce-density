"""Shared constructions for ce_compute: F8 arithmetic, the groups G_m = P/P_m (vectorized),
Schreier voltages theta_{s,x}, and F_m = 1 + fbar_1 fbar_0 fbar_2.

Conventions copied from Ruqing1963/anccft-volume3 code/a2_common.py (TowerRing):
  F8 elements are ints 0..7, bit i = coefficient of alpha^i, alpha^3 = alpha + 1;
  ring elements are tuples (c_0=1, c_1, ..., c_{m-1}) meaning sum c_i phi^i, phi a = a^2 phi;
  normalize: for i = 1..(m-1)//3 ascending, if bit 0 of c_{3i} is set multiply by 1 + t^i.
  theta_{s,x} = normalize(u0^s u_x u0^{-((s+1) mod 3)}),  u_x = 1 + alpha^x phi.
  fbar_s = sum_x theta_{s,x}^{-1};  F = 1 + fbar_1 fbar_0 fbar_2;  C_m = F2[G]/F2[G] F (rows g F).
"""
from collections import Counter

import numpy as np

GIDX = {(s, x): 3 * x + s for s in range(3) for x in range(7)}


def _f8mul(x, y):
    r = 0
    for i in range(3):
        if (y >> i) & 1:
            r ^= x << i
    for d in (4, 3):
        if (r >> d) & 1:
            r ^= 0b1011 << (d - 3)
    return r


F8 = [[_f8mul(x, y) for y in range(8)] for x in range(8)]
FROB = [F8[x][x] for x in range(8)]
ALPHA = [1]
for _ in range(6):
    ALPHA.append(F8[ALPHA[-1]][2])
MUL = np.array(F8, dtype=np.uint8)
FROBA = np.array(FROB, dtype=np.uint8)
FROBP = [np.arange(8, dtype=np.uint8), FROBA, FROBA[FROBA]]   # sigma^0, sigma^1, sigma^2


def sig(a, i):
    for _ in range(i % 3):
        a = FROB[a]
    return a


class TowerRing:
    """Exact port of a2_common.TowerRing."""

    def __init__(self, m):
        self.m = m

    def mul(self, a, b):
        c = [0] * self.m
        for i, ai in enumerate(a):
            if ai:
                for j in range(self.m - i):
                    bj = sig(b[j], i)
                    if bj:
                        c[i + j] ^= F8[ai][bj]
        return tuple(c)

    def one(self):
        return tuple([1] + [0] * (self.m - 1))

    def inv(self, a):
        x = tuple([0] + list(a[1:]))
        res, p = self.one(), self.one()
        for _ in range(self.m):
            p = self.mul(p, x)
            res = tuple(u ^ v for u, v in zip(res, p))
        return res

    def normalize(self, a):
        a = list(a)
        for i in range(1, (self.m - 1) // 3 + 1):
            if a[3 * i] & 1:
                z = [0] * self.m
                z[0] = z[3 * i] = 1
                a = list(self.mul(tuple(a), tuple(z)))
        return tuple(a)

    def nmul(self, a, b):
        return self.normalize(self.mul(a, b))

    def ninv(self, a):
        return self.normalize(self.inv(a))

    def voltages(self):
        def u(x):
            e = [0] * self.m
            e[0] = 1
            if self.m > 1:
                e[1] = ALPHA[x]
            return tuple(e)
        pw = {0: self.one(), 1: u(0), 2: self.mul(u(0), u(0))}
        th = [None] * 21
        for s in range(3):
            for x in range(7):
                th[GIDX[(s, x)]] = self.normalize(
                    self.mul(self.mul(pw[s], u(x)), self.inv(pw[(s + 1) % 3])))
        return th


def nbits(m):
    return 3 * (m - 1) - (m - 1) // 3


def n_of(m):
    return 3 * 2 ** nbits(m)


def layout(m):
    """list of (j, b): code bit -> coefficient c_j bit b (bit 0 dropped when 3 | j)."""
    return [(j, b) for j in range(1, m) for b in range(3) if not (j % 3 == 0 and b == 0)]


def encode(a, m):
    v = 0
    for i, (j, b) in enumerate(layout(m)):
        v |= ((a[j] >> b) & 1) << i
    return v


def decode(v, m):
    e = [0] * m
    e[0] = 1
    for i, (j, b) in enumerate(layout(m)):
        if (v >> i) & 1:
            e[j] |= 1 << b
    return tuple(e)


def all_elements(m):
    """N x m uint8 array: row v = decode(v)."""
    lay = layout(m)
    N = 1 << len(lay)
    v = np.arange(N, dtype=np.int64)
    C = np.zeros((N, m), dtype=np.uint8)
    C[:, 0] = 1
    for i, (j, b) in enumerate(lay):
        C[:, j] |= (((v >> i) & 1) << b).astype(np.uint8)
    return C


def encode_arr(C, m):
    v = np.zeros(C.shape[0], dtype=np.int64)
    for i, (j, b) in enumerate(layout(m)):
        v |= ((C[:, j].astype(np.int64) >> b) & 1) << i
    return v


def normalize_arr(C, m):
    for i in range(1, (m - 1) // 3 + 1):
        mask = (C[:, 3 * i] & 1).astype(bool)
        if mask.any():
            sub = C[mask]
            for k in range(m - 1, 3 * i - 1, -1):
                sub[:, k] ^= sub[:, k - 3 * i]
            C[mask] = sub
    return C


def rmul_arr(C, h, m):
    """C (N x m) times fixed h on the right, normalized."""
    out = np.zeros_like(C)
    for i in range(m):
        Ci = C[:, i]
        for j in range(m - i):
            hj = sig(h[j], i)
            if hj:
                out[:, i + j] ^= MUL[Ci, hj]
    return normalize_arr(out, m)


def lmul_arr(h, C, m):
    out = np.zeros_like(C)
    for i in range(m):
        if h[i]:
            Ci = FROBP[i % 3][C[:, :m - i]]
            out[:, i:] ^= MUL[h[i], Ci]
    return normalize_arr(out, m)


def F_support(m):
    """Support of F_m = 1 + fbar_1 fbar_0 fbar_2 as list of normalized tuples."""
    R = TowerRing(m)
    th = R.voltages()
    one = R.normalize(R.one())
    fbar = [Counter(R.ninv(th[GIDX[(s, x)]]) for x in range(7)) for s in range(3)]

    def gmul(P_, Q_):
        out = Counter()
        for a, ca in P_.items():
            for b, cb in Q_.items():
                out[R.nmul(a, b)] += ca * cb
        return Counter({g: c % 2 for g, c in out.items() if c % 2})
    F = gmul(gmul(fbar[1], fbar[0]), fbar[2])
    F[one] = (F[one] + 1) % 2
    return sorted(g for g, c in F.items() if c % 2)


def F_matrix_packed(m, supp=None):
    """Packed GF(2) matrix (N x ceil(N/64) uint64): row g = the element g F_m."""
    if supp is None:
        supp = F_support(m)
    C = all_elements(m)
    N = C.shape[0]
    assert np.array_equal(encode_arr(C, m), np.arange(N))
    W = (N + 63) // 64
    P = np.zeros((N, W), dtype=np.uint64)
    rows = np.arange(N)
    for h in supp:
        col = encode_arr(rmul_arr(C, h, m), m)
        assert np.unique(col).size == N
        P[rows, col >> 6] |= np.left_shift(np.uint64(1), (col & 63).astype(np.uint64))
    return P, len(supp)
