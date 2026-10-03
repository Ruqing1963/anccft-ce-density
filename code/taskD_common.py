r"""Task D: common helpers.  Elements of (u^s_m)_delta, delta = (1,1,1), as F2 (or F_{2^k}) combinations of the 6 PBW
monomials of Q-weight delta; block-by-block kernels of right multiplication by such elements.

(u^s_m)_delta has the PBW basis (m >= 4)  y3.0, y3.1, y2.q*y1.{q+1} (q = 0,1,2), y1.0*y1.1*y1.2   (6 elements), so there
are only 63 nonzero F2-elements of weight delta; sigma' is one of them.
"""
import numpy as np

from cengine import Split, Blocks, build_block, pack_words
from gf2 import gf2_rank_inplace

DELTA = (1, 1, 1)


def rot(g):
    return (g[2], g[0], g[1])


def orbit_rep(g):
    return min(g, rot(g), rot(rot(g)))


def elem_add(*els):
    acc = {}
    for el in els:
        for t in el:
            acc[t] = acc.get(t, 0) ^ 1
    return sorted(t for t, c in acc.items() if c)


def mono_word(sp, mono):
    """PBW monomial -> list of letter positions (a monomial is the product of its letters in increasing bit position)"""
    return [i for i in range(sp.nl) if (mono >> i) & 1]


def letter_sum(sp, k, p):
    """e_p phi^k as a list of letter positions (e_2 = e_0 + e_1 when 3 | k)"""
    if k >= sp.m:
        return []
    if k % 3 == 0 and p % 3 == 2:
        return [sp.pos[(k, 0)], sp.pos[(k, 1)]]
    return [sp.pos[(k, p % 3)]]


def words_to_elem(sp, words):
    """sum of products; each factor is a list of letter positions (a sum of letters)"""
    acc = []
    for w in words:
        exp = [[]]
        for ls in w:
            exp = [e + [x] for e in exp for x in ls]
        for e in exp:
            acc = elem_add(acc, sp.rmul(0, e))
    return acc


def rho_elem(sp, el, power=1):
    """vertex rotation rho^power applied to an element (list of PBW monomials)"""
    out = el
    for _ in range(power):
        words = []
        for mono in out:
            words.append([letter_sum(sp, *_rot_letter(sp.letters[i])) for i in mono_word(sp, mono)])
        out = words_to_elem(sp, words)
    return out


def _rot_letter(kp):
    k, p = kp
    return (k, (p + 1) % 3)


class DeltaSpace:
    def __init__(self, sp, B):
        self.sp = sp
        self.basis = [int(x) for x in B.blocks[DELTA]]          # PBW monomials of weight delta
        assert len(self.basis) == 6, len(self.basis)
        self.words = [mono_word(sp, b) for b in self.basis]
        self.idx = {b: i for i, b in enumerate(self.basis)}
        sig = words_to_elem(sp, [[[i] for i in w] for w in sp.sigma_words()])
        self.sigma = self.vec(sig)
        # rho as a 6x6 matrix over F2 (columns = images of basis vectors), as bitmask images
        self.rho_img = [self.vec(rho_elem(sp, [b])) for b in self.basis]

    def vec(self, el):
        v = 0
        for t in el:
            v ^= 1 << self.idx[t]
        return v

    def rho(self, v):
        out = 0
        for i in range(6):
            if (v >> i) & 1:
                out ^= self.rho_img[i]
        return out

    def label(self, v):
        return " + ".join(self.sp.mstr(self.basis[i]) for i in range(6) if (v >> i) & 1)

    def words_of(self, v):
        return [self.words[i] for i in range(6) if (v >> i) & 1]


def block_matrix(sp, B, g, wordlist, nchunk=64):
    h = tuple(a + 1 for a in g)
    words, wlen = pack_words([np.array(w, dtype=np.int64) for w in wordlist])
    P, bad = build_block(B.blocks[g], words, wlen, sp.br, sp.sq, B.local, B.key, np.int32(B.gkey(h)), B.dim(h), nchunk)
    assert bad == 0, (g, bad)
    return P


def block_rank_F2(sp, B, g, wordlist):
    h = tuple(a + 1 for a in g)
    if B.dim(h) == 0 or not wordlist:
        return 0
    P = block_matrix(sp, B, g, wordlist)
    return int(gf2_rank_inplace(P, B.dim(h)))


# ------------------------------------------------------------------ F_{2^k} via the regular representation
def gf2k_mulmat(a, k, poly):
    """k x k matrix over F2 of multiplication by a in F_{2^k} = F2[x]/(poly) acting on row vectors (coefficient bits)"""
    M = np.zeros((k, k), dtype=np.uint8)
    for i in range(k):
        # row i = coefficients of x^i * a
        v = a
        for _ in range(i):
            v <<= 1
            if (v >> k) & 1:
                v ^= poly
        for j in range(k):
            M[i, j] = (v >> j) & 1
    return M


def block_rank_F2k(sp, B, g, coeffs, k, poly, mats=None):
    """rank over F_{2^k} of R_x on block g, x = sum_i coeffs[i] * basis_i (coeffs in F_{2^k} as ints).
    Uses the regular representation: rank_F2(big) = k * rank_{F_{2^k}}."""
    h = tuple(a + 1 for a in g)
    if B.dim(h) == 0:
        return 0
    if mats is None:
        mats = {}
    Ps = []
    for i in range(6):
        if coeffs[i]:
            if i not in mats:
                mats[i] = block_matrix(sp, B, g, [_BASIS_WORDS[i]])
            Ps.append((gf2k_mulmat(coeffs[i], k, poly), mats[i]))
    nr, W = Ps[0][1].shape
    big = np.zeros((k * nr, k * W), dtype=np.uint64)
    for Mk, P in Ps:
        for a in range(k):
            for b in range(k):
                if Mk[a, b]:
                    big[a * nr:(a + 1) * nr, b * W:(b + 1) * W] ^= P
    r = int(gf2_rank_inplace(big, k * W * 64))
    assert r % k == 0
    return r // k


_BASIS_WORDS = []


def set_basis_words(ds):
    _BASIS_WORDS.clear()
    _BASIS_WORDS.extend(ds.words)
