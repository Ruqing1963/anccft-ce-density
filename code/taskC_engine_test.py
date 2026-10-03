"""Tests for cengine.py: (1) products agree with ulie.ULie('split') (independent straightening, ascending order);
(2) associativity; (3) Q-homogeneity of products; (4) sigma' is invariant under the vertex rotation rho."""
import random
import sys

from cengine import Split
from ulie import ULie


def to_ulie(sp, U, mono):
    """my PBW monomial (product of letters in increasing bit position) as element of ULie split (set of monos)"""
    cur = {0}
    for i in range(sp.nl):
        if (mono >> i) & 1:
            k, p = sp.letters[i]
            cur = U.mul(cur, {1 << U.lidx[(k, 1 << p)]})
    return cur


def elem_to_ulie(sp, U, monos):
    acc = set()
    for x in monos:
        acc ^= to_ulie(sp, U, x)
    return acc


if __name__ == "__main__":
    rnd = random.Random(3)
    for m in (4, 5, 6, 7):
        sp = Split(m)
        U = ULie("split", Dl=m - 1, maxdeg=sp.top)
        n1 = 0
        for trial in range(300):
            A = rnd.getrandbits(sp.nl)
            B = rnd.getrandbits(sp.nl) & rnd.getrandbits(sp.nl)
            mine = sp.mul_monos(A, B)
            lhs = elem_to_ulie(sp, U, mine)
            rhs = U.mul(to_ulie(sp, U, A), to_ulie(sp, U, B))
            assert lhs == rhs, (m, A, B)
            qa, qb = sp.qdeg(A), sp.qdeg(B)
            for t in mine:
                assert sp.qdeg(t) == tuple(x + y for x, y in zip(qa, qb))
            n1 += 1
        n2 = 0
        for trial in range(200):
            A, B, C = (rnd.getrandbits(sp.nl) & rnd.getrandbits(sp.nl) for _ in range(3))
            l = sp.mul(sp.mul_monos(A, B), [C])
            r = sp.mul([A], sp.mul_monos(B, C))
            assert l == r
            n2 += 1
        # rho-invariance of sigma': rho(e_p phi^k) = e_{p+1} phi^k (e_2 = e_0+e_1 when 3|k)
        def rho_letter(i):
            k, p = sp.letters[i]
            q = (p + 1) % 3
            if k % 3 == 0 and q == 2:
                return [sp.pos[(k, 0)], sp.pos[(k, 1)]]
            return [sp.pos[(k, q)]]

        def rho_word(w):
            terms = [[]]
            for i in w:
                terms = [t + [c] for t in terms for c in rho_letter(i)]
            return terms
        sig = set()
        for w in sp.sigma_words():
            for t in sp.rmul(0, list(w)):
                sig ^= {t}
        rs = set()
        for w in sp.sigma_words():
            for ww in rho_word(list(w)):
                for t in sp.rmul(0, ww):
                    rs ^= {t}
        print(f"m={m}: {n1} products agree with ULie('split') and are Q-homogeneous; {n2} associativity triples OK; "
              f"sigma' = {' + '.join(sp.mstr(t) for t in sorted(sig))};  rho(sigma') == sigma': {rs == sig}", flush=True)
        assert rs == sig
