r"""Task E: certify the kernel of R_sigma' at level M from explicit kernels at level a (2a >= M) via the first
connecting map (THEORY §6.5).

I := L_{>=a} / L_{>=M} is an abelian ideal with zero 2-map (2a >= M), u(I) = Lambda(I), and with the PBW order of cengine
(degree descending) the I-letters are the first s letters, so a level-M monomial is  S | (x << s)  with S a set of
I-letters and x a level-a monomial.  Filtration J^i = span{monomials with >= i I-letters} (right ideals, R-stable),
gr^i = Lambda^i(I) (x) u_a, gr R = 1 (x) R^{(a)}.
E_1^i = sum_{|S| = i} S (x) ker R^{(a)}_{gamma - wt S},
d_1(S (x) x) = sum_{S' = S + l} S' (x) [ z_{S'} ]  in  S' (x) coker R^{(a)}_{gamma + delta - wt S'},
where (S x-hat) sigma' = sum_{S'} S' z_{S'} mod J^{i+2}.
Lemma: gr^i(ker R^{(M)}_gamma) is contained in ker(d_1 | E_1^i), hence dim ker R^{(M)}_gamma <= sum_i dim ker(d_1|E_1^i).

usage: python taskE_lift.py M a nmin nmax [ker-pickle-of-level-a]
"""
import itertools
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
from numba import njit, prange

from cengine import Split, pack_words, rmul_word, MAXOUT, MAXST, MAXW
from gf2 import gf2_rank_inplace
from gf2null import left_kernel, _elim
from taskC2_low import blocks_of_degree, build_block_sorted

DELTA = (1, 1, 1)


def rot(g):
    return (g[2], g[0], g[1])


def orbit_rep(g):
    return min(g, rot(g), rot(rot(g)))


def add(g, h):
    return tuple(x + y for x, y in zip(g, h))


def sub(g, h):
    return tuple(x - y for x, y in zip(g, h))


@njit
def _lowbit_row(row):
    for w in range(row.size):
        v = row[w]
        if v:
            b = 0
            while not (v >> np.uint64(b)) & np.uint64(1):
                b += 1
            return w * 64 + b
    return -1


@njit(parallel=True)
def reduce_rows(V, B, piv):
    """V (k x W) reduced in place modulo the row space of B (echelon rows sorted by pivot column piv)."""
    one = np.uint64(1)
    for i in prange(V.shape[0]):
        for r in range(B.shape[0]):
            p = piv[r]
            if (V[i, p >> 6] >> np.uint64(p & 63)) & one:
                w0 = p >> 6
                for w in range(w0, V.shape[1]):
                    V[i, w] ^= B[r, w]


@njit(parallel=True)
def _term_products(monos, words, wlen, br, sq, cnt, flat, offs, fill):
    nt = monos.size
    nchunk = 64
    csz = (nt + nchunk - 1) // nchunk
    for c in prange(nchunk):
        out = np.empty(MAXOUT, dtype=np.int64)
        stS = np.empty(MAXST, dtype=np.int64)
        stW = np.empty((MAXST, MAXW), dtype=np.int64)
        stL = np.empty(MAXST, dtype=np.int64)
        Wb = np.empty(MAXW, dtype=np.int64)
        for t in range(c * csz, min(nt, (c + 1) * csz)):
            n = 0
            for w in range(wlen.size):
                n = rmul_word(monos[t], words[w, :wlen[w]], br, sq, out, n, stS, stW, stL, Wb)
                if n < 0:
                    break
            if n < 0:
                cnt[t] = -1
                continue
            if fill:
                for k in range(n):
                    flat[offs[t] + k] = out[k]
            cnt[t] = n


def products(monos, starts, words, wlen, br, sq):
    """for each vector v (monos[starts[v]:starts[v+1]]) the monomials of v*sigma' (odd multiplicity), as arrays"""
    nt = monos.size
    cnt = np.zeros(nt, dtype=np.int64)
    dummy = np.zeros(1, dtype=np.int64)
    _term_products(monos, words, wlen, br, sq, cnt, dummy, cnt, False)
    assert np.all(cnt >= 0), "rewriting overflow"
    offs = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    flat = np.empty(int(offs[-1]), dtype=np.int64)
    cnt2 = np.zeros(nt, dtype=np.int64)
    _term_products(monos, words, wlen, br, sq, cnt2, flat, offs[:-1].copy(), True)
    outs = []
    for v in range(starts.size - 1):
        seg = flat[offs[starts[v]]:offs[starts[v + 1]]]
        u, c = np.unique(seg, return_counts=True)
        outs.append(u[(c & 1) == 1])
    return outs


class Level:
    def __init__(self, m):
        self.sp = Split(m)
        self.words, self.wlen = pack_words(self.sp.sigma_words())
        self._deg = {}

    def blocks(self, n):
        if n not in self._deg:
            self._deg[n] = blocks_of_degree(self.sp, n) if n >= 0 else {}
        return self._deg[n]

    def block(self, g):
        if min(g) < 0:
            return np.zeros(0, dtype=np.int64)
        return self.blocks(sum(g)).get(g, np.zeros(0, dtype=np.int64))

    def matrix(self, g):
        rows = self.block(g)
        cols = self.block(add(g, DELTA))
        P, bad = build_block_sorted(rows, self.words, self.wlen, self.sp.br, self.sp.sq, cols, 64)
        assert bad == 0
        return rows, cols, P


def kernel_basis(lev, g):
    rows, cols, P = lev.matrix(g)
    if cols.size == 0:
        return [rows[[i]] for i in range(rows.size)]
    K, rk = left_kernel(P, cols.size)
    out = []
    for v in K:
        bits = np.unpackbits(v.view(np.uint8), bitorder="little")[:rows.size]
        out.append(rows[np.nonzero(bits)[0]])
    return out


def image_echelon(lev, tau):
    """echelon basis (rows sorted by pivot) of im R^{(a)} inside block tau, as packed rows over block(tau)"""
    cols = lev.block(tau)
    src = lev.block(sub(tau, DELTA))
    W = max(1, (cols.size + 63) // 64)
    if src.size == 0 or cols.size == 0:
        return cols, np.zeros((0, W), dtype=np.uint64), np.zeros(0, dtype=np.int64)
    P, bad = build_block_sorted(src, lev.words, lev.wlen, lev.sp.br, lev.sp.sq, cols, 64)
    assert bad == 0
    A = P.copy()
    rank, piv, rest = _elim(A, cols.size)
    B = A[piv].copy()
    pc = np.array([_lowbit_row(B[i]) for i in range(B.shape[0])], dtype=np.int64)
    o = np.argsort(pc)
    return cols, B[o].copy(), pc[o].copy()


def image_solver(lev, tau):
    """for im R^{(a)} in block tau: echelon rows of the augmented matrix [R | Id] (sorted by pivot), so that reducing
    [w | 0] gives [NF(w) | p] with w - NF(w) = p R."""
    cols = lev.block(tau)
    src = lev.block(sub(tau, DELTA))
    Wc = max(1, (cols.size + 63) // 64)
    if src.size == 0 or cols.size == 0:
        return cols, src, Wc, np.zeros((0, Wc + 1), dtype=np.uint64), np.zeros(0, dtype=np.int64)
    P, bad = build_block_sorted(src, lev.words, lev.wlen, lev.sp.br, lev.sp.sq, cols, 64)
    assert bad == 0
    r = src.size
    W2 = (r + 63) // 64
    A = np.zeros((r, Wc + W2), dtype=np.uint64)
    A[:, :P.shape[1]] = P
    idx = np.arange(r)
    A[idx, Wc + (idx >> 6)] = np.left_shift(np.uint64(1), (idx & 63).astype(np.uint64))
    rank, piv, rest = _elim(A, Wc * 64)
    B = A[piv].copy()
    pc = np.array([_lowbit_row(B[i, :Wc]) for i in range(B.shape[0])], dtype=np.int64)
    o = np.argsort(pc)
    return cols, src, Wc, B[o].copy(), pc[o].copy()


def to_packed(cols, monos, W):
    v = np.zeros(W, dtype=np.uint64)
    if len(monos):
        z = np.array(sorted(monos), dtype=np.int64)
        j = np.searchsorted(cols, z)
        assert np.all(j < cols.size) and np.all(cols[j] == z), "monomial outside target block"
        for jj in j:
            v[jj >> 6] ^= np.uint64(1) << np.uint64(jj & 63)
    return v


def from_packed(rowsarr, v):
    bits = np.unpackbits(v.view(np.uint8), bitorder="little")[:rowsarr.size]
    return rowsarr[np.nonzero(bits)[0]]


def xor_sets(arrs):
    if not arrs:
        return np.zeros(0, dtype=np.int64)
    c = np.concatenate(arrs)
    u, k = np.unique(c, return_counts=True)
    return u[(k & 1) == 1]


def exact_kernel(hi, lo, s, Iw, subsets, g, ker, kcache, verbose=False):
    """exact dim ker R^{(M)}_g by layer-wise lifting (THEORY §6.5).  Returns (dim, candidate elements)."""
    comps = []                      # (S, Sbits, beta, layer)
    for S, Sb, ws, ds in subsets:
        b = sub(g, ws)
        if min(b) < 0 or lo.block(b).size == 0:
            continue
        comps.append((S, Sb, b, len(S)))
    maxlayer = max(c[3] for c in comps)
    cand = []
    solv = {}
    for i in range(maxlayer + 1):
        # 1. impose the layer-i equations on the existing candidates
        if cand:
            starts = np.concatenate([[0], np.cumsum([c.size for c in cand])]).astype(np.int64)
            outs = products(np.concatenate(cand), starts, hi.words, hi.wlen, hi.sp.br, hi.sp.sq)
            lowmask = (1 << s) - 1
            pieces = []
            tgt = set()
            for o in outs:
                pc = np.zeros(o.size, dtype=np.int64)
                for q in range(s):
                    pc += ((o & lowmask) >> q) & 1
                assert not np.any(pc < i), "candidate does not satisfy lower layers"
                d = defaultdict(list)
                for mono in o[pc == i]:
                    d[int(mono & lowmask)].append(int(mono >> s))
                pieces.append(d)
                tgt.update(d.keys())
            tgt = sorted(tgt)
            if tgt:
                for Sp in tgt:
                    if Sp not in solv:
                        wsp = tuple(sum(Iw[q][c] for q in range(s) if (Sp >> q) & 1) for c in range(3))
                        solv[Sp] = image_solver(lo, sub(add(g, DELTA), wsp))
                # normal forms (with preimage coefficients)
                NFs = {}
                for Sp in tgt:
                    cols, src, Wc, B, pc = solv[Sp]
                    W2 = B.shape[1] - Wc if B.shape[0] else (src.size + 63) // 64 + 0
                    W2 = max(W2, (src.size + 63) // 64, 1)
                    V = np.zeros((len(cand), Wc + W2), dtype=np.uint64)
                    for r, d in enumerate(pieces):
                        if Sp in d:
                            V[r, :Wc] = to_packed(cols, d[Sp], Wc)
                    if B.shape[0]:
                        Bf = np.zeros((B.shape[0], Wc + W2), dtype=np.uint64)
                        Bf[:, :B.shape[1]] = B
                        reduce_rows(V, Bf, pc)
                    NFs[Sp] = V
                # combined NF matrix -> left kernel = combinations satisfying layer i modulo images
                N = np.concatenate([NFs[Sp][:, :solv[Sp][2]] for Sp in tgt], axis=1)
                ncol = N.shape[1] * 64
                Kc, rk = left_kernel(N, ncol)
                new = []
                for kv in Kc:
                    coef = np.nonzero(np.unpackbits(kv.view(np.uint8), bitorder="little")[:len(cand)])[0]
                    parts = [cand[r] for r in coef]
                    for Sp in tgt:
                        cols, src, Wc, B, pc = solv[Sp]
                        acc = np.zeros(NFs[Sp].shape[1], dtype=np.uint64)
                        for r in coef:
                            acc ^= NFs[Sp][r]
                        assert not acc[:Wc].any()
                        p = from_packed(src, acc[Wc:])
                        if p.size:
                            parts.append((p << s) | Sp)
                    new.append(xor_sets(parts))
                cand = new
        # 2. add the kernel vectors of the layer-i diagonal blocks (E_1^i) as new candidates
        for S, Sb, b, layer in comps:
            if layer != i or not ker.get(b, 0):
                continue
            if b not in kcache:
                kcache[b] = kernel_basis(lo, b)
            for x in kcache[b]:
                cand.append(np.sort((x << s) | Sb))
        if verbose:
            print(f"      layer {i}: {len(cand)} candidates", flush=True)
    # final check: all candidates are kernel elements
    if cand:
        starts = np.concatenate([[0], np.cumsum([c.size for c in cand])]).astype(np.int64)
        outs = products(np.concatenate(cand), starts, hi.words, hi.wlen, hi.sp.br, hi.sp.sq)
        assert all(o.size == 0 for o in outs), "final candidates are not in the kernel"
    return len(cand), cand


def main(M, a, nmin, nmax, kerpkl):
    assert 2 * a >= M
    t0 = time.time()
    hi = Level(M)
    lo = Level(a)
    s = sum(1 for (k, p) in hi.sp.letters if k >= a)
    assert all(hi.sp.letters[i][0] >= a for i in range(s))
    assert [hi.sp.letters[i + s] for i in range(lo.sp.nl)] == lo.sp.letters
    Iw = [tuple(int(x) for x in hi.sp.lwt[i]) for i in range(s)]
    Ideg = [int(hi.sp.ldeg[i]) for i in range(s)]
    with open(kerpkl, "rb") as fh:
        D = pickle.load(fh)
    ker = D["ker"]
    ka = min(sum(g) for g, v in ker.items() if v)
    print(f"M={M} from a={a}: {s} I-letters {[hi.sp.letters[i] for i in range(s)]}; k_a = {ka}", flush=True)
    subsets = []
    for r in range(s + 1):
        for S in itertools.combinations(range(s), r):
            subsets.append((S, sum(1 << i for i in S), tuple(sum(Iw[i][c] for i in S) for c in range(3)),
                            sum(Ideg[i] for i in S)))
    gam = set()
    for b, v in ker.items():
        if not v:
            continue
        for S, Sb, ws, ds in subsets:
            g = add(b, ws)
            if nmin <= sum(g) <= nmax:
                gam.add(orbit_rep(g))
    kcache = {}
    summary = defaultdict(lambda: [0, 0, 0, 0])
    results = {}
    kern = {}
    for g in sorted(gam, key=lambda x: (sum(x), x)):
        n = sum(g)
        E1 = defaultdict(list)          # i -> list of (S, Sbits, x monomials)
        for S, Sb, ws, ds in subsets:
            b = sub(g, ws)
            if min(b) < 0 or not ker.get(b, 0):
                continue
            if b not in kcache:
                kb = kernel_basis(lo, b)
                assert len(kb) == ker[b], (b, len(kb), ker[b])
                kcache[b] = kb
            for x in kcache[b]:
                E1[len(S)].append((S, Sb, x))
        tot_e2 = 0
        info = []
        for i in sorted(E1):
            vecs = E1[i]
            monos = np.concatenate([(x << s) | Sb for (S, Sb, x) in vecs])
            starts = np.concatenate([[0], np.cumsum([x.size for (S, Sb, x) in vecs])]).astype(np.int64)
            outs = products(monos, starts, hi.words, hi.wlen, hi.sp.br, hi.sp.sq)
            comps = []                     # per vector: dict S' -> level-a monomials
            targets = set()
            for (S, Sb, x), o in zip(vecs, outs):
                pc = np.zeros(o.size, dtype=np.int64)
                low = o & ((1 << s) - 1)
                for q in range(s):
                    pc += (low >> q) & 1
                assert not np.any(pc < i)
                assert not np.any(pc == i), "level-a product not zero"
                sel = pc == i + 1
                d = defaultdict(list)
                for mono in o[sel]:
                    d[int(mono & ((1 << s) - 1))].append(int(mono >> s))
                comps.append(d)
                targets.update(d.keys())
            targets = sorted(targets)
            if not targets:
                e2 = len(vecs)
                info.append((i, len(vecs), 0))
                tot_e2 += e2
                continue
            offs = {}
            cols_of = {}
            ech = {}
            off = 0
            for Sp in targets:
                wsp = tuple(sum(Iw[q][c] for q in range(s) if (Sp >> q) & 1) for c in range(3))
                tau = sub(add(g, DELTA), wsp)
                cols, B, pc = image_echelon(lo, tau)
                cols_of[Sp] = cols
                ech[Sp] = (B, pc)
                offs[Sp] = off
                off += cols.size
            Wt = (off + 63) // 64
            NF = np.zeros((len(vecs), Wt), dtype=np.uint64)
            for Sp in targets:
                cols = cols_of[Sp]
                B, pc = ech[Sp]
                Wc = max(1, (cols.size + 63) // 64)
                V = np.zeros((len(vecs), Wc), dtype=np.uint64)
                for r, d in enumerate(comps):
                    if Sp in d:
                        z = np.array(sorted(d[Sp]), dtype=np.int64)
                        j = np.searchsorted(cols, z)
                        assert np.all(j < cols.size) and np.all(cols[j] == z)
                        for jj in j:
                            V[r, jj >> 6] ^= np.uint64(1) << np.uint64(jj & 63)
                if B.shape[0]:
                    reduce_rows(V, B, pc)
                # copy bits into NF at offset
                o0 = offs[Sp]
                bits = np.unpackbits(V.view(np.uint8), axis=1, bitorder="little")[:, :cols.size]
                for r in range(len(vecs)):
                    nzb = np.nonzero(bits[r])[0] + o0
                    for jj in nzb:
                        NF[r, jj >> 6] ^= np.uint64(1) << np.uint64(jj & 63)
            rk = int(gf2_rank_inplace(NF, off)) if off else 0
            e2 = len(vecs) - rk
            info.append((i, len(vecs), rk))
            tot_e2 += e2
        exact = 0
        if tot_e2 > 0 and EXACT:
            exact, cand = exact_kernel(hi, lo, s, Iw, subsets, g, ker, kcache)
            assert exact <= tot_e2
            if exact:
                kern[g] = cand
        results[g] = (tot_e2, info, exact)
        summary[n][0] += 1
        summary[n][1] += sum(x[1] for x in info)
        summary[n][2] += tot_e2
        summary[n][3] += exact
        print(f"  gamma={g} n={n}: E1/rank(d1) per i = {info} -> E2 = {tot_e2}, exact ker = {exact}"
              f"   [{time.time() - t0:.0f}s]", flush=True)
    print("summary (orbit representatives): degree: #blocks, dim E1, dim E2, exact dim ker R^(M)", flush=True)
    for n in sorted(summary):
        print(f"  n={n}: {summary[n]}", flush=True)
    with open(f"taskE_lift_M{M}_a{a}_{nmin}_{nmax}.pkl", "wb") as fh:
        pickle.dump({"M": M, "a": a, "results": results, "summary": dict(summary), "kernels": kern}, fh)


EXACT = True

if __name__ == "__main__":
    M, a, nmin, nmax = (int(x) for x in sys.argv[1:5])
    kp = sys.argv[5] if len(sys.argv) > 5 else f"taskC1_m{a}.pkl"
    main(M, a, nmin, nmax, kp)
