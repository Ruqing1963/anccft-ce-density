"""F2[[P]] / I^N in the Jennings basis (ungraded straightening).

Letters e_s = g_s - 1, g_{k,b} = 1 + b phi^k (k < N, same letters/order as ulie.ULie);
basis J(S) = e_{s1} e_{s2} ... (s1 < s2 < ...), weight = sum of degrees; I^N = span{J(S): wt(S) >= N}
(Jennings; the filtration P_k is the Jennings series, checked in task 2), and F2[[P]]/I^N = F2[G_N]/I^N.
Exact (ungraded) relations, computed in the group G_N:
  e_z e_y = e_y e_z + (g_z g_y - g_y g_z)          (z > y)
  e_z^2   = g_z^2 - 1
where a group element g has Jennings coordinates {T subset S(g)} (g = prod_{s in S(g)} g_s, ordered).
All correction terms have strictly larger weight, so straightening terminates.
Elements: Python sets of masks (F2 coefficients), only weights < N kept.
"""
import sys
import threading
from collections import Counter

from ulie import ULie
from task2_leading import Jennings

sys.setrecursionlimit(1000000)


class JAlg:
    def __init__(self, N):
        self.N = N
        self.U = ULie("F8", Dl=N - 1, maxdeg=N - 1)
        U = self.U
        self.ldeg = U.ldeg
        self.nl = U.nl
        self.Jn = Jennings(N, U)
        R = self.Jn.R
        g = self.Jn.gel
        self.sq = []
        for z in range(self.nl):
            h = R.nmul(g[z], g[z])
            self.sq.append(self.coords(h) - {0})
        self.cm = {}
        for z in range(self.nl):
            for y in range(z):
                a = self.coords(R.nmul(g[z], g[y]))
                b = self.coords(R.nmul(g[y], g[z]))
                self.cm[(z, y)] = a ^ b
        self.memo = {}

    def coords(self, h):
        return set(self.Jn.coeffs([h], self.N - 1))

    def wt(self, mono):
        d = 0
        while mono:
            low = mono & -mono
            d += self.ldeg[low.bit_length() - 1]
            mono ^= low
        return d

    def lmul(self, z, mono, dmono=None):
        key = (z, mono)
        r = self.memo.get(key)
        if r is not None:
            return r
        if dmono is None:
            dmono = self.wt(mono)
        if self.ldeg[z] + dmono >= self.N:
            r = frozenset()
        elif mono == 0:
            r = frozenset((1 << z,))
        else:
            low = mono & -mono
            y = low.bit_length() - 1
            rest = mono ^ low
            if z < y:
                r = frozenset((mono | (1 << z),))
            elif z == y:
                acc = set()
                for T in self.sq[z]:
                    acc ^= self.mul_mono(T, rest)
                r = frozenset(acc)
            else:
                acc = set()
                below = (low << 1) - 1
                dy = self.ldeg[y]
                for t in self.lmul(z, rest):
                    assert not (t & below)
                    if self.wt(t) + dy < self.N:
                        acc.add(t | low)
                for T in self.cm[(z, y)]:
                    acc ^= self.mul_mono(T, rest)
                r = frozenset(acc)
        self.memo[key] = r
        return r

    def mul_mono(self, A, B):
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
            if not cur:
                break
        return cur

    def mul(self, X, Y):
        out = Counter()
        for a in X:
            for b in Y:
                for t in self.mul_mono(a, b):
                    out[t] ^= 1
        return {t for t, c in out.items() if c}


def run_big_stack(f, *args):
    threading.stack_size(256 * 1024 * 1024 - 4096)
    res = {}

    def tgt():
        res["r"] = f(*args)
    th = threading.Thread(target=tgt)
    th.start()
    th.join()
    return res.get("r")


def _selftest(N=9, npairs=150, seed=3):
    import random
    import time
    from task2_leading import gmul
    t0 = time.time()
    A = JAlg(N)
    U = A.U
    rng = random.Random(seed)
    nb = 0
    for _ in range(npairs):
        a = rng.randint(1, N - 2)
        b = rng.randint(1, N - 1 - a) if N - 1 - a >= 1 else 1
        x = rng.choice(U.monos(a))
        y = rng.choice(U.monos(b))
        # also allow total weight >= N (product must be truncated)
        if rng.random() < 0.3:
            y = rng.choice(U.monos(rng.randint(1, N - 1)))
        prod = gmul(A.Jn.R, A.Jn.J(x), A.Jn.J(y))
        co = A.Jn.coeffs(prod, N - 1)
        mine = A.mul({x}, {y})
        assert co == mine, (U.mono_str(x), U.mono_str(y), len(co), len(mine))
        nb += 1
    # associativity on random triples
    na = 0
    for _ in range(200):
        x, y, z = (rng.choice(U.monos(rng.randint(1, 4))) for _ in range(3))
        assert A.mul(A.mul({x}, {y}), {z}) == A.mul({x}, A.mul({y}, {z}))
        na += 1
    print(f"JAlg selftest N={N}: {nb} products agree with group algebra F2[G_{N}] (Jennings coords, weight < {N}); "
          f"{na} associativity triples ok ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    run_big_stack(_selftest, int(sys.argv[1]) if len(sys.argv) > 1 else 9)
