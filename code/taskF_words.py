r"""Task F: U = u(n^) as a quotient of the free algebra on the Chevalley generators e_i = y1.i (i = 0,1,2).

* sigma' = e0 e1 e2 + e1 e2 e0 + e2 e0 e1 (cyclic sum of the Coxeter word) -- checked here.
* Normal words (complement of the leading-word ideal of the relations) for a deglex order with letter ranks `order`,
  degree by degree and Q-block by Q-block: w (all of whose proper subwords are normal) is normal iff phi(w) is not in the
  span of phi(v), v < w normal.  The count is checked against dim U_gamma.
* Prefix test PF(n): for every normal w of length n, the leading normal word of NF(w sigma') has prefix w.
  PF(n) for all n  ==>  R_sigma' injective on U  (lead(y sigma') = lead(y) c).
usage: python taskF_words.py D order(e.g. 012) [pf]
"""
import sys
import time
from collections import defaultdict

import numpy as np

from cengine import Split, pack_words, rmul_word, workbufs, cancel, MAXOUT
from taskC2_low import blocks_of_degree


class Words:
    def __init__(self, D, order):
        self.D = D
        self.sp = sp = Split(D + 1)
        self.e = [sp.pos[(1, i)] for i in range(3)]
        self.rank = {int(c): r for r, c in enumerate(order)}      # letter -> rank
        self.bufs = workbufs()
        self.out = np.empty(MAXOUT, dtype=np.int64)
        self.blocks = {}
        self.index = {}

    def rmul_vec(self, vec, word):
        acc = []
        w = np.asarray(word, dtype=np.int64)
        for S in vec:
            n = rmul_word(np.int64(S), w, self.sp.br, self.sp.sq, self.out, 0, *self.bufs)
            assert n >= 0
            acc.append(self.out[:n].copy())
        if not acc:
            return np.zeros(0, dtype=np.int64)
        return np.array(cancel(np.concatenate(acc)), dtype=np.int64)

    def blk(self, n):
        if n not in self.blocks:
            B = blocks_of_degree(self.sp, n) if n > 0 else {(0, 0, 0): np.array([0], dtype=np.int64)}
            self.blocks[n] = B
        return self.blocks[n]

    def tobits(self, n, g, vec):
        cols = self.blk(n)[g]
        j = np.searchsorted(cols, vec)
        assert np.all(cols[j] == vec)
        x = 0
        for jj in j:
            x ^= 1 << int(jj)
        return x

    def compute(self, verbose=True):
        """normal words of degree 0..D: self.normal[n][g] = list of (word, phi-vector, bits) sorted by order."""
        self.normal = {0: {(0, 0, 0): [((), np.array([0], dtype=np.int64))]}}
        self.nset = {()}
        for n in range(1, self.D + 1):
            t0 = time.time()
            cands = defaultdict(list)
            for g, L in self.normal[n - 1].items():
                for w, vec in L:
                    for i in range(3):
                        ww = w + (i,)
                        if ww[1:] not in self.nset:
                            continue
                        g2 = list(g)
                        g2[i] += 1
                        cands[tuple(g2)].append(ww)
            # phi of candidates: phi(w) = phi(w[:-1]) * e_last ; cache prefix vectors
            pref = {w: vec for g, L in self.normal[n - 1].items() for w, vec in L}
            res = {}
            tot = 0
            for g, ws in cands.items():
                ws.sort(key=lambda w: tuple(self.rank[c] for c in w))
                ech = {}           # pivot (highest bit) -> row
                keep = []
                for w in ws:
                    vec = self.rmul_vec(pref[w[:-1]], [self.e[w[-1]]])
                    x = self.tobits(n, g, vec) if vec.size else 0
                    while x:
                        h = x.bit_length() - 1
                        if h in ech:
                            x ^= ech[h]
                        else:
                            ech[h] = x
                            break
                    if x:
                        keep.append((w, vec))
                dimg = self.blk(n)[g].size if g in self.blk(n) else 0
                assert len(keep) == dimg, (n, g, len(keep), dimg)
                res[g] = keep
                tot += len(keep)
                for w, v in keep:
                    self.nset.add(w)
            # blocks of U_n with no candidate at all
            for g, cols in self.blk(n).items():
                assert g in res, (n, g)
            self.normal[n] = res
            if verbose:
                print(f"  degree {n}: {tot} normal words  [{time.time() - t0:.1f}s]", flush=True)

    def leading_words(self, n):
        """minimal non-normal words of length n (leading words of a minimal Groebner basis)"""
        out = []
        for g, L in self.normal[n - 1].items():
            for w, vec in L:
                for i in range(3):
                    ww = w + (i,)
                    if ww[1:] in self.nset and ww not in self.nset:
                        out.append(ww)
        return sorted(out, key=lambda w: tuple(self.rank[c] for c in w))

    def solver(self, n, g):
        """echelon of phi(normal words) of block (n,g) with tracking: returns (ech dict pivot->(row, comb))"""
        key = (n, g)
        if not hasattr(self, "_solv"):
            self._solv = {}
        if key in self._solv:
            return self._solv[key]
        L = self.normal[n][g]
        ech = {}
        for k, (w, vec) in enumerate(L):
            x = self.tobits(n, g, vec)
            c = 1 << k
            while x:
                h = x.bit_length() - 1
                if h in ech:
                    x ^= ech[h][0]
                    c ^= ech[h][1]
                else:
                    ech[h] = (x, c)
                    break
            assert x
        self._solv[key] = ech
        return ech

    def nf(self, n, g, vec):
        """coefficients (bitmask over normal words of block (n,g), index = sorted position) of vec"""
        if vec.size == 0:
            return 0
        ech = self.solver(n, g)
        x = self.tobits(n, g, vec)
        c = 0
        while x:
            h = x.bit_length() - 1
            x ^= ech[h][0]
            c ^= ech[h][1]
        return c


def sigma_check(W):
    sp = W.sp
    e = W.e
    tot = []
    for c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        tot.append(W.rmul_vec(np.array([0], dtype=np.int64), [e[c[0]], e[c[1]], e[c[2]]]))
    cyc = np.array(cancel(np.concatenate(tot)), dtype=np.int64)
    words, wlen = pack_words(sp.sigma_words())
    sv = []
    for k in range(wlen.size):
        sv.append(W.rmul_vec(np.array([0], dtype=np.int64), words[k, :wlen[k]]))
    sig = np.array(cancel(np.concatenate(sv)), dtype=np.int64)
    return np.array_equal(np.sort(cyc), np.sort(sig))


def pf_test(W, nmax, verbose=True):
    CYC = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    allok = True
    stats = {}
    for n in range(0, nmax + 1):
        bad = 0
        lastpat = defaultdict(int)
        for g, L in W.normal[n].items():
            g3 = (g[0] + 1, g[1] + 1, g[2] + 1)
            NL = W.normal[n + 3][g3]
            for w, vec in L:
                prod = [W.rmul_vec(vec, [W.e[c[0]], W.e[c[1]], W.e[c[2]]]) for c in CYC]
                v = np.array(cancel(np.concatenate(prod)), dtype=np.int64)
                c = W.nf(n + 3, g3, v)
                if c == 0:
                    bad += 1
                    print("   ZERO PRODUCT for", w)
                    continue
                lead = NL[c.bit_length() - 1][0]
                if lead[:n] != w:
                    bad += 1
                    if bad <= 5:
                        print(f"   n={n}: w={w} lead={lead}")
                else:
                    lastpat[(w[-2:] if n >= 2 else w, lead[n:])] += 1
        stats[n] = (bad, dict(lastpat))
        if verbose:
            print(f"PF({n}): failures {bad}; suffix patterns (last2 -> appended) {dict(lastpat)}", flush=True)
        allok &= bad == 0
    return allok, stats


if __name__ == "__main__":
    D = int(sys.argv[1])
    order = sys.argv[2] if len(sys.argv) > 2 else "012"
    W = Words(D, order)
    print("sigma' == e0e1e2 + e1e2e0 + e2e0e1 :", sigma_check(W), flush=True)
    W.compute()
    for n in range(2, min(D, 12) + 1):
        lw = W.leading_words(n)
        if lw:
            print(f"  leading words of degree {n} ({len(lw)}): {[''.join(map(str, w)) for w in lw[:40]]}", flush=True)
    if len(sys.argv) > 3:
        pf_test(W, D - 3)
