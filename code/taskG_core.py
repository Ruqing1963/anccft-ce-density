r"""Task G core: fast exact layer-wise lifting  ker R^{(M)}_gamma  from level a (2a >= M), THEORY 6.5, re-implemented.

Structure used (THEORY 6.5): u_M = Lambda(I) (x) u_a, I = L_{>=a} abelian with zero 2-map, I-letters first in the PBW order.
For a level-a monomial x, (x-hat) sigma' = sum_T T * Z_T(x) (T = set of I-letters, Z_empty = R^{(a)}), and
(S x-hat) sigma' = sum_{T disjoint from S} (S u T) Z_T(x).  So R^{(M)} = sum_T e_T (x) Z_T, block triangular over subsets.
The Z_T are stored once per level-a block as sparse tables ("straightening tables").

Differences to taskE_lift.py (same algorithm, same output):
 * residuals are computed bit-sliced (one straightening per monomial of a level-a block, shared by all candidates);
 * the test "residual in im R^{(a)}" uses precomputed cokernel functionals F_t (basis of {f : R_{t-delta} f = 0}),
   cached on disk per level-a block;
 * kernel bases of R^{(a)} are cached on disk per block;
 * preimages (only for surviving combinations) come from an echelon of [R_b | Id] with recorded pivots, replayed with
   Four-Russians tables.
Every final candidate is re-checked to satisfy y sigma' = 0 at level M (all T including T = empty).
"""
import os
import pickle
import time
from collections import OrderedDict

import numpy as np
from numba import njit, prange

from cengine import Split, pack_words, rmul_word, MAXOUT, MAXST, MAXW
from gf2 import _tables, _apply, _lowbit
from gf2null import left_kernel
from taskC2_low import blocks_of_degree

DELTA = (1, 1, 1)
CACHE = "taskG_cache"


def add(g, h):
    return tuple(x + y for x, y in zip(g, h))


def sub(g, h):
    return tuple(x - y for x, y in zip(g, h))


def rot(g):
    return (g[2], g[0], g[1])


def orbit_rep(g):
    return min(g, rot(g), rot(rot(g)))


def nwords(n):
    return max(1, (n + 63) // 64)


# ----------------------------------------------------------------------------------------------- numba kernels
@njit
def elim_info(A, nw):
    """Gaussian elimination (64-column panels, Four Russians) on the first nw words of A; all words updated.
    Returns rank, pivot rows (in order), panel word and pivot bit per pivot, remaining (non-pivot) rows."""
    R, W = A.shape
    act = np.arange(R)
    nact = R
    rank = 0
    pivrows = np.empty(R, dtype=np.int64)
    pw = np.empty(R, dtype=np.int64)
    plb = np.empty(R, dtype=np.int64)
    PIV = np.zeros((64, W), dtype=np.uint64)
    T = np.zeros((8, 256, W), dtype=np.uint64)
    lb = np.zeros(64, dtype=np.int64)
    pvec = np.zeros(64, dtype=np.uint64)
    pidx = np.zeros(64, dtype=np.int64)
    one = np.uint64(1)
    for w in range(nw):
        if nact == 0:
            break
        npiv = 0
        keep = np.ones(nact, dtype=np.bool_)
        for ii in range(nact):
            if npiv == 64:
                break
            r = act[ii]
            v = A[r, w]
            if v == 0:
                continue
            comb = np.uint64(0)
            for j in range(npiv):
                if (v >> np.uint64(lb[j])) & one:
                    v ^= pvec[j]
                    comb |= one << np.uint64(j)
            if v == 0:
                continue
            for k in range(w, W):
                PIV[npiv, k] = A[r, k]
            for j in range(npiv):
                if (comb >> np.uint64(j)) & one:
                    for k in range(w, W):
                        PIV[npiv, k] ^= PIV[j, k]
            pvec[npiv] = v
            lb[npiv] = _lowbit(v)
            pidx[npiv] = r
            npiv += 1
            keep[ii] = False
        if npiv == 0:
            continue
        for j in range(npiv - 1, -1, -1):
            for k2 in range(j):
                if (PIV[k2, w] >> np.uint64(lb[j])) & one:
                    for k in range(w, W):
                        PIV[k2, k] ^= PIV[j, k]
        for j in range(npiv):
            for k in range(w, W):
                A[pidx[j], k] = PIV[j, k]
            pivrows[rank + j] = pidx[j]
            pw[rank + j] = w
            plb[rank + j] = lb[j]
        rank += npiv
        newact = np.empty(nact - npiv, dtype=np.int64)
        t = 0
        for ii in range(nact):
            if keep[ii]:
                newact[t] = act[ii]
                t += 1
        act = newact
        nact = t
        _tables(PIV, npiv, w, W, T)
        _apply(A, act, nact, w, W, lb, npiv, T)
    return rank, pivrows[:rank], pw[:rank], plb[:rank], act[:nact]


@njit
def replay(X, E, pivrows, pw, plb, Wuse):
    """reduce the rows of X (words [0, Wuse)) by the recorded pivots of an elim_info run on E"""
    n = X.shape[0]
    rank = pivrows.size
    PIV = np.zeros((64, Wuse), dtype=np.uint64)
    T = np.zeros((8, 256, Wuse), dtype=np.uint64)
    lb = np.zeros(64, dtype=np.int64)
    act = np.arange(n)
    i = 0
    while i < rank:
        w = pw[i]
        j = i
        while j < rank and pw[j] == w:
            j += 1
        npiv = j - i
        for q in range(npiv):
            r = pivrows[i + q]
            for k in range(w, Wuse):
                PIV[q, k] = E[r, k]
            lb[q] = plb[i + q]
        _tables(PIV, npiv, w, Wuse, T)
        _apply(X, act, n, w, Wuse, lb, npiv, T)
        i = j


@njit(parallel=True)
def transpose_bits(X, nr, nc):
    """X packed (nr x ceil(nc/64)) -> Y packed (nc x ceil(nr/64))"""
    Wy = max(1, (nr + 63) // 64)
    Y = np.zeros((nc, Wy), dtype=np.uint64)
    Wx = X.shape[1]
    one = np.uint64(1)
    for w in prange(Wx):
        for bi in range(Wy):
            for t in range(64):
                i = bi * 64 + t
                if i >= nr:
                    break
                v = X[i, w]
                while v:
                    b = _lowbit(v)
                    v &= v - one
                    c = w * 64 + b
                    if c < nc:
                        Y[c, bi] |= one << np.uint64(t)
    return Y


@njit(parallel=True)
def matmul(A, kbits, B):
    """C = A . B over GF(2); A packed (n x ceil(k/64)), B packed (k x Wm).  Four Russians, 8-row groups."""
    n = A.shape[0]
    Wm = B.shape[1]
    C = np.zeros((n, Wm), dtype=np.uint64)
    tab = np.zeros((256, Wm), dtype=np.uint64)
    ng = (kbits + 7) // 8
    for g in range(ng):
        base = g * 8
        nb = min(8, kbits - base)
        for idx in range(1, 256):
            lo = idx & (idx - 1)
            t = idx ^ lo
            j = 0
            while t > 1:
                t >>= 1
                j += 1
            if j >= nb:
                for k in range(Wm):
                    tab[idx, k] = 0
            else:
                for k in range(Wm):
                    tab[idx, k] = tab[lo, k] ^ B[base + j, k]
        wd = base >> 6
        sh = np.uint64(base & 63)
        for i in prange(n):
            idx = (A[i, wd] >> sh) & np.uint64(255)
            if idx:
                for k in range(Wm):
                    C[i, k] ^= tab[idx, k]
    return C


@njit
def scatter_apply(Yt, rows, cols, Out):
    """Out[cols[e]] ^= Yt[rows[e]]  (bit-sliced application of a sparse 0/1 matrix)"""
    W = Yt.shape[1]
    for e in range(rows.size):
        r = rows[e]
        c = cols[e]
        for k in range(W):
            Out[c, k] ^= Yt[r, k]


def straighten(sp, monos, words, wlen):
    """for each monomial: the odd-multiplicity monomials of mono*sigma' -> (row index array, monomial array)"""
    nt = monos.size
    cnt = np.zeros(nt, dtype=np.int64)
    dummy = np.zeros(1, dtype=np.int64)
    _call_straighten(sp, monos, words, wlen, cnt, dummy, cnt, False)
    assert np.all(cnt >= 0), "rewriting overflow"
    offs = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    flat = np.empty(int(offs[-1]), dtype=np.int64)
    cnt2 = np.zeros(nt, dtype=np.int64)
    _call_straighten(sp, monos, words, wlen, cnt2, flat, offs[:-1].copy(), True)
    rows = np.repeat(np.arange(nt, dtype=np.int64), cnt)
    return rows, flat


@njit(parallel=True)
def _straighten2(monos, words, wlen, br, sq, cnt, flat, offs, fill):
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
            s = np.sort(out[:n])
            m = 0
            i = 0
            while i < n:
                j = i
                while j < n and s[j] == s[i]:
                    j += 1
                if (j - i) & 1:
                    if fill:
                        flat[offs[t] + m] = s[i]
                    m += 1
                i = j
            cnt[t] = m


def _call_straighten(sp, monos, words, wlen, cnt, flat, offs, fill):
    _straighten2(monos, words, wlen, sp.br, sp.sq, cnt, flat, offs, fill)


@njit(parallel=True)
def pack_pairs(rows, cols, nr, ncol):
    W = max(1, (ncol + 63) // 64)
    P = np.zeros((nr, W), dtype=np.uint64)
    for e in range(rows.size):
        P[rows[e], cols[e] >> 6] ^= np.uint64(1) << np.uint64(cols[e] & 63)
    return P


# ----------------------------------------------------------------------------------------------- level-a data
class LowLevel:
    """level-a blocks, straightening tables at level M, kernel bases and cokernel functionals (disk cached)."""

    def __init__(self, M, a, verbose=True):
        self.M, self.a = M, a
        self.hi = Split(M)
        self.lo = Split(a)
        self.words, self.wlen = pack_words(self.hi.sigma_words())
        self.s = s = sum(1 for (k, p) in self.hi.letters if k >= a)
        assert [self.hi.letters[i + s] for i in range(self.lo.nl)] == self.lo.letters
        self.Iw = [tuple(int(x) for x in self.hi.lwt[i]) for i in range(s)]
        self.Ideg = [int(self.hi.ldeg[i]) for i in range(s)]
        self.wS = {}
        for S in range(1 << s):
            self.wS[S] = tuple(sum(self.Iw[q][c] for q in range(s) if (S >> q) & 1) for c in range(3))
        self._deg = {}
        self.verbose = verbose
        self.tabcache = OrderedDict()
        self.kcache = OrderedDict()
        self.fcache = OrderedDict()
        self.echcache = OrderedDict()
        self.dir = os.path.join(CACHE, f"a{a}")
        os.makedirs(self.dir, exist_ok=True)
        self.tdir = os.path.join(CACHE, f"tab_M{M}_a{a}")
        os.makedirs(self.tdir, exist_ok=True)
        self.stats = {"t_tab": 0.0, "t_ker": 0.0, "t_cok": 0.0, "t_ech": 0.0}

    def block(self, g):
        if min(g) < 0:
            return np.zeros(0, dtype=np.int64)
        n = sum(g)
        if n not in self._deg:
            self._deg[n] = blocks_of_degree(self.lo, n) if n >= 0 else {}
            if len(self._deg) > 60:
                self._deg.pop(next(iter(self._deg)))
        return self._deg[n].get(g, np.zeros(0, dtype=np.int64))

    @staticmethod
    def _lru(cache, key, val, cap):
        cache[key] = val
        cache.move_to_end(key)
        while len(cache) > cap:
            cache.popitem(last=False)

    def fname(self, kind, g, d=None):
        return os.path.join(d or self.dir, f"{kind}_{g[0]}_{g[1]}_{g[2]}.npz")

    # straightening table of block b at level M: dict T -> (rows, cols, target block)
    def table(self, b):
        if b in self.tabcache:
            self.tabcache.move_to_end(b)
            return self.tabcache[b]
        fn = self.fname("tab", b, self.tdir)
        if os.path.exists(fn):
            z = np.load(fn)
            Ts = z["Ts"]
            tab = {}
            for i, T in enumerate(Ts):
                tab[int(T)] = (z[f"r{i}"], z[f"c{i}"])
        else:
            t0 = time.time()
            monos = self.block(b)
            s = self.s
            rows, outs = straighten(self.hi, monos << s, self.words, self.wlen)
            Tm = outs & ((1 << s) - 1)
            xp = outs >> s
            tab = {}
            for T in np.unique(Tm):
                T = int(T)
                sel = Tm == T
                tgt = sub(add(b, DELTA), self.wS[T])
                cols_arr = self.block(tgt)
                z = xp[sel]
                j = np.searchsorted(cols_arr, z)
                assert np.all(j < cols_arr.size) and np.all(cols_arr[np.minimum(j, cols_arr.size - 1)] == z), \
                    ("table target mismatch", b, T)
                tab[T] = (rows[sel].astype(np.int32), j.astype(np.int32))
            Ts = np.array(sorted(tab), dtype=np.int64)
            np.savez(fn, Ts=Ts, **{f"r{i}": tab[int(T)][0] for i, T in enumerate(Ts)},
                     **{f"c{i}": tab[int(T)][1] for i, T in enumerate(Ts)})
            self.stats["t_tab"] += time.time() - t0
        self._lru(self.tabcache, b, tab, 80)
        return tab

    def rmatrix(self, b):
        """packed R^{(a)}_b : rows = monomials of b, cols = monomials of b + delta"""
        db = self.block(b).size
        dt = self.block(add(b, DELTA)).size
        tab = self.table(b)
        if 0 in tab:
            r, c = tab[0]
            return pack_pairs(r.astype(np.int64), c.astype(np.int64), db, dt), dt
        return np.zeros((db, nwords(dt)), dtype=np.uint64), dt

    def echelon(self, b):
        """elimination of [R_b | Id]: returns (Big, Wt, pivrows, pw, plb, rest)"""
        if b in self.echcache:
            self.echcache.move_to_end(b)
            return self.echcache[b]
        t0 = time.time()
        P, dt = self.rmatrix(b)
        db = P.shape[0]
        Wt = nwords(dt)
        Wb = nwords(db)
        Big = np.zeros((db, Wt + Wb), dtype=np.uint64)
        Big[:, :P.shape[1]] = P
        del P
        idx = np.arange(db)
        Big[idx, Wt + (idx >> 6)] = np.left_shift(np.uint64(1), (idx & 63).astype(np.uint64))
        rank, piv, pw, plb, rest = elim_info(Big, Wt)
        ent = (Big, Wt, piv, pw, plb, rest)
        self.stats["t_ech"] += time.time() - t0
        self._lru(self.echcache, b, ent, 1)
        return ent

    def kernel(self, b, need_rank=None):
        """packed basis (k x W_b) of ker R^{(a)}_b (disk cached)"""
        if b in self.kcache:
            return self.kcache[b]
        fn = self.fname("ker", b)
        if os.path.exists(fn):
            K = np.load(fn)["K"]
        else:
            t0 = time.time()
            Big, Wt, piv, pw, plb, rest = self.echelon(b)
            assert not Big[rest][:, :Wt].any()
            K = Big[rest][:, Wt:].copy()
            del Big
            self.echcache.clear()
            np.savez(fn, K=K)
            self.stats["t_ker"] += time.time() - t0
        self._lru(self.kcache, b, K, 4)
        return K

    def cokfun(self, t):
        """packed basis F (c x W_t) of {f : R_{t-delta} f = 0} (functionals vanishing on the image in block t)"""
        if t in self.fcache:
            self.fcache.move_to_end(t)
            return self.fcache[t]
        fn = self.fname("cok", t)
        dt = self.block(t).size
        if os.path.exists(fn):
            F = np.load(fn)["F"]
        else:
            t0 = time.time()
            src = sub(t, DELTA)
            ds = self.block(src).size
            Wt = nwords(dt)
            if ds == 0:
                F = np.zeros((dt, Wt), dtype=np.uint64)
                idx = np.arange(dt)
                F[idx, idx >> 6] = np.left_shift(np.uint64(1), (idx & 63).astype(np.uint64))
            else:
                tab = self.table(src)
                Ws = nwords(ds)
                Big = np.zeros((dt, Ws + Wt), dtype=np.uint64)
                if 0 in tab:
                    r, c = tab[0]
                    PT = pack_pairs(c.astype(np.int64), r.astype(np.int64), dt, ds)
                    Big[:, :PT.shape[1]] = PT
                    del PT
                idx = np.arange(dt)
                Big[idx, Ws + (idx >> 6)] = np.left_shift(np.uint64(1), (idx & 63).astype(np.uint64))
                rank, piv, pw, plb, rest = elim_info(Big, Ws)
                assert not Big[rest][:, :Ws].any()
                F = Big[rest][:, Ws:].copy()
                del Big
            np.savez(fn, F=F)
            self.stats["t_cok"] += time.time() - t0
        self._lru(self.fcache, t, F, 8)
        return F


# ----------------------------------------------------------------------------------------------- the lifting
def popcount(x):
    return bin(x).count("1")


def lift_block(LL, g, kerdims, verbose=False):
    """exact kernel of R^{(M)} on level-M block g.  kerdims: dict level-a block -> dim ker (for skipping).
    Returns (dim, info list, candidates dict S -> packed rows)."""
    s = LL.s
    comps = {}
    for S in range(1 << s):
        b = sub(g, LL.wS[S])
        if min(b) >= 0 and LL.block(b).size:
            comps[S] = b
    tgts = {}
    for S in range(1 << s):
        t = sub(add(g, DELTA), LL.wS[S])
        if min(t) >= 0 and LL.block(t).size:
            tgts[S] = t
    layers = sorted(set(popcount(S) for S in set(comps) | set(tgts)))
    cand = {}          # S -> packed rows (nc x W_{b_S})
    nc = 0
    info = []
    for i in layers:
        Sps = [S for S in tgts if popcount(S) == i]
        if nc > 0 and i > 0 and Sps:
            Wc = nwords(nc)
            # bit-sliced residuals  ResT[S'] : (d_t x Wc)
            ResT = {Sp: np.zeros((LL.block(tgts[Sp]).size, Wc), dtype=np.uint64) for Sp in Sps}
            for S, Y in cand.items():
                b = comps[S]
                tab = LL.table(b)
                Yt = None
                for T, (r, c) in tab.items():
                    if T == 0 or (T & S) or popcount(S | T) != i:
                        continue
                    Sp = S | T
                    assert Sp in ResT
                    if Yt is None:
                        Yt = transpose_bits(Y, nc, LL.block(b).size)
                    scatter_apply(Yt, r, c, ResT[Sp])
            # test with cokernel functionals
            blocksN = []
            for Sp in Sps:
                F = LL.cokfun(tgts[Sp])
                if F.shape[0] == 0:
                    continue
                NT = matmul(F, LL.block(tgts[Sp]).size, ResT[Sp])        # c x Wc
                blocksN.append(transpose_bits(NT, F.shape[0], nc))       # nc x Wc'
            if blocksN:
                N = np.concatenate(blocksN, axis=1)
                Kc, rk = left_kernel(N, N.shape[1] * 64)
            else:
                Kc = np.zeros((nc, Wc), dtype=np.uint64)
                idx = np.arange(nc)
                Kc[idx, idx >> 6] = np.left_shift(np.uint64(1), (idx & 63).astype(np.uint64))
                rk = 0
            nnew = Kc.shape[0]
            info.append((i, nc, int(rk)))
            if verbose:
                print(f"      layer {i}: {nc} candidates, rank {rk}, survivors {nnew}", flush=True)
            if nnew == 0:
                cand = {}
                nc = 0
            else:
                cand = {S: matmul(Kc, nc, Y) for S, Y in cand.items()}
                ResR = {}
                for Sp in Sps:
                    RR = transpose_bits(ResT[Sp], LL.block(tgts[Sp]).size, nc)
                    ResR[Sp] = matmul(Kc, nc, RR)
                del ResT
                # preimages, grouped by level-a block
                byb = {}
                for Sp in Sps:
                    if not ResR[Sp].any():
                        continue
                    assert Sp in comps, "nonzero residual with empty source block"
                    byb.setdefault(comps[Sp], []).append(Sp)
                for b, lst in byb.items():
                    Big, Wt, piv, pw, plb, rest = LL.echelon(b)
                    Wb = Big.shape[1] - Wt
                    for Sp in lst:
                        X = np.zeros((nnew, Wt + Wb), dtype=np.uint64)
                        X[:, :ResR[Sp].shape[1]] = ResR[Sp]
                        replay(X, Big, piv, pw, plb, Wt + Wb)
                        assert not X[:, :Wt].any(), "functional test passed but residual not in image"
                        cand[Sp] = X[:, Wt:].copy()
                nc = nnew
        # add the kernel of the layer-i diagonal blocks
        for Sp in sorted(S for S in comps if popcount(S) == i):
            b = comps[Sp]
            if not kerdims.get(b, 0):
                continue
            K = LL.kernel(b)
            assert K.shape[0] == kerdims[b], (b, K.shape[0], kerdims[b])
            k = K.shape[0]
            newc = {}
            W = nwords(nc + k)
            for S, Y in cand.items():
                Z = np.zeros((nc + k, Y.shape[1]), dtype=np.uint64)
                Z[:nc] = Y
                newc[S] = Z
            if Sp in newc:
                newc[Sp][nc:] = K
            else:
                Z = np.zeros((nc + k, K.shape[1]), dtype=np.uint64)
                Z[nc:] = K
                newc[Sp] = Z
            cand = newc
            nc += k
        if verbose:
            print(f"    after layer {i}: {nc} candidates", flush=True)
    # final verification: full product (all T, including T = empty) vanishes
    if nc:
        Wc = nwords(nc)
        Out = {Sp: np.zeros((LL.block(t).size, Wc), dtype=np.uint64) for Sp, t in tgts.items()}
        for S, Y in cand.items():
            b = comps[S]
            Yt = transpose_bits(Y, nc, LL.block(b).size)
            for T, (r, c) in LL.table(b).items():
                if T & S:
                    continue
                scatter_apply(Yt, r, c, Out[S | T])
        assert all(not o.any() for o in Out.values()), "final candidates are not in the kernel"
        # independence check: rank of the stacked candidate matrix
        stacked = np.concatenate([cand[S] for S in sorted(cand)], axis=1)
        from gf2 import gf2_rank
        assert gf2_rank(stacked) == nc
    return nc, info, cand


MEMDEBUG = bool(os.environ.get("TASKG_MEMDEBUG"))
SPILL_BYTES = int(float(os.environ.get("TASKG_SPILL_GB", "1.0")) * 2 ** 30)
SPILL_DIR = os.environ.get("TASKG_TMP", os.path.join(CACHE, "spill"))
_SPILLED = []


def _spill(arr):
    """copy a packed array into a disk-backed memmap (survivor components of huge blocks), return an ndarray view"""
    os.makedirs(SPILL_DIR, exist_ok=True)
    fn = os.path.join(SPILL_DIR, f"sp_{os.getpid()}_{len(_SPILLED)}_{time.time_ns()}.bin")
    mm = np.memmap(fn, dtype=np.uint64, mode="w+", shape=arr.shape)
    mm[:] = arr
    mm.flush()
    _SPILLED.append((fn, mm))
    return np.asarray(mm)


def _cleanup_spill():
    import gc
    files = [fn for fn, _ in _SPILLED]
    _SPILLED.clear()
    gc.collect()
    for fn in files:
        try:
            os.remove(fn)
        except OSError:
            pass


def _ident(n):
    K = np.zeros((n, nwords(n)), dtype=np.uint64)
    idx = np.arange(n)
    K[idx, idx >> 6] = np.left_shift(np.uint64(1), (idx & 63).astype(np.uint64))
    return K


class Ckpt:
    """intra-block checkpoint: arrays as .npy files + state pickle (written atomically last)"""

    def __init__(self, d, every):
        self.d = d
        self.every = every
        self.last = time.time()
        if d:
            os.makedirs(d, exist_ok=True)
        self.n = 0

    def load(self):
        if not self.d:
            return None
        fn = os.path.join(self.d, "state.pkl")
        if not os.path.exists(fn):
            return None
        st = pickle.load(open(fn, "rb"))
        return st

    def arr(self, name):
        return np.asarray(np.load(os.path.join(self.d, name), mmap_mode="r"))

    def put(self, a):
        self.n += 1
        name = f"a_{time.time_ns()}_{self.n}.npy"
        np.save(os.path.join(self.d, name), np.asarray(a))
        return name

    DEADLINE = float(os.environ.get("TASKG_DEADLINE", "inf"))

    def due(self):
        return self.d and (time.time() - self.last > self.every or time.time() > self.DEADLINE)

    def save(self, st):
        tmp = os.path.join(self.d, "state.tmp")
        pickle.dump(st, open(tmp, "wb"))
        os.replace(tmp, os.path.join(self.d, "state.pkl"))
        self.last = time.time()
        # remove arrays no longer referenced
        used = set()

        def walk(x):
            if isinstance(x, str) and x.startswith("a_") and x.endswith(".npy"):
                used.add(x)
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    walk(v)
        walk(st)
        for f in os.listdir(self.d):
            if f.startswith("a_") and f.endswith(".npy") and f not in used:
                try:
                    os.remove(os.path.join(self.d, f))
                except OSError:
                    pass
        if time.time() > self.DEADLINE:
            print(f"    checkpoint saved, deadline reached -> exit (state {st.get('phase')}, layer index {st.get('li')})",
                  flush=True)
            raise SystemExit(3)

    def clear(self):
        if self.d and os.path.isdir(self.d):
            import shutil
            shutil.rmtree(self.d, ignore_errors=True)


def lift_block2(LL, g, kerdims, verbose=False, ckdir=None, ckevery=1200):
    """Same result as lift_block, memory-lean version: candidates are kept as groups (fresh kernel blocks are not
    padded), layer tests are done target by target with an accumulated combination matrix, residuals are formed
    one target at a time.  Returns (dim, info, cand) with cand a dict S -> packed rows.
    ckdir: optional directory for intra-block checkpoints (resumable at layer / target / preimage-group level)."""
    CK = Ckpt(ckdir, ckevery)
    s = LL.s
    comps = {}
    for S in range(1 << s):
        b = sub(g, LL.wS[S])
        if min(b) >= 0 and LL.block(b).size:
            comps[S] = b
    tgts = {}
    for S in range(1 << s):
        t = sub(add(g, DELTA), LL.wS[S])
        if min(t) >= 0 and LL.block(t).size:
            tgts[S] = t
    layers = sorted(set(popcount(S) for S in set(comps) | set(tgts)))
    groups = []        # list of (r, {S: packed rows r x W_S})
    info = []

    def ncand():
        return sum(r for r, _ in groups)

    def residualT(grp_r, grp_c, Yts, Sp):
        """bit-sliced residual at S' of one group: (d_t x W(r)) or None"""
        out = None
        for S, Y in grp_c.items():
            if S & ~Sp or S == Sp:
                continue
            tab = LL.table(comps[S])
            ent = tab.get(Sp ^ S)
            if ent is None:
                continue
            if out is None:
                out = np.zeros((LL.block(tgts[Sp]).size, nwords(grp_r)), dtype=np.uint64)
            key = id(Y)
            if key not in Yts:
                Ym = Y if type(Y) is np.ndarray and Y.flags.owndata else np.array(Y)   # disk-backed -> RAM first
                Yts[key] = transpose_bits(Ym, grp_r, LL.block(comps[S]).size)
                del Ym
            scatter_apply(Yts[key], ent[0], ent[1], out)
        return out

    def save_groups(grps):
        return [(r, {S: CK.put(Y) for S, Y in cc.items()}) for r, cc in grps]

    def load_groups(gf):
        return [(r, {S: CK.arr(f) for S, f in cc.items()}) for r, cc in gf]

    st = CK.load()
    li0 = 0
    phase = "start"
    if st is not None:
        assert st["g"] == g
        li0, phase, info = st["li"], st["phase"], st["info"]
        groups = load_groups(st["groups"])
        if verbose:
            print(f"    resumed at layer index {li0}, phase {phase}", flush=True)
    for li in range(li0, len(layers)):
        i = layers[li]
        Sps = sorted((S for S in tgts if popcount(S) == i), key=lambda S: LL.block(tgts[S]).size)
        nc = ncand()
        if not (li == li0 and phase in ("test", "pre")):
            phase = "start"
        if nc > 0 and i > 0 and Sps:
            if phase in ("start", "test"):
                Kacc = None
                nrow = nc
                k0 = 0
                if phase == "test":
                    k0 = st["k"]
                    Kacc = None if st["Kacc"] is None else np.array(CK.arr(st["Kacc"]))
                    nrow = st["nrow"]
                    gfiles = st["groups"]
                else:
                    gfiles = None
                for kk in range(k0, len(Sps)):
                    Sp = Sps[kk]
                    if CK.due():
                        if gfiles is None:
                            gfiles = save_groups(groups)
                        CK.save({"g": g, "li": li, "phase": "test", "info": info, "groups": gfiles, "k": kk,
                                 "Kacc": None if Kacc is None else CK.put(Kacc), "nrow": nrow})
                    F = LL.cokfun(tgts[Sp])
                    c = F.shape[0]
                    if c == 0:
                        continue
                    Nsp = np.zeros((nc, nwords(c)), dtype=np.uint64)
                    o = 0
                    anyres = False
                    for r, cc in groups:
                        RT = residualT(r, cc, {}, Sp)
                        if RT is not None:
                            NT = matmul(F, LL.block(tgts[Sp]).size, RT)
                            Nsp[o:o + r] = transpose_bits(NT, c, r)
                            anyres = True
                        o += r
                    if not anyres:
                        continue
                    cur = Nsp if Kacc is None else matmul(Kacc, nc, Nsp)
                    del Nsp
                    if not cur.any():
                        continue
                    Kp, rk = left_kernel(cur, cur.shape[1] * 64)
                    del cur
                    Kacc = Kp if Kacc is None else matmul(Kp, nrow, Kacc)
                    del Kp
                    nrow = Kacc.shape[0]
                    if nrow == 0:
                        break
                if Kacc is None:
                    Kacc = _ident(nc)
                nsurv = Kacc.shape[0]
                info.append((i, nc, nc - nsurv))
                if verbose:
                    print(f"      layer {i}: {nc} candidates, survivors {nsurv}", flush=True)
                if nsurv == 0:
                    groups = []
                    newc = None
                else:
                    KT = transpose_bits(Kacc, nsurv, nc)          # nc x W(nsurv)
                    del Kacc
                    newc = {}
                    est = nsurv * 8 * sum(nwords(LL.block(comps[S]).size)
                                          for S in set(S for _, cc in groups for S in cc))
                    early = est > SPILL_BYTES
                    o = 0
                    for r, cc in groups:
                        Kg = transpose_bits(np.ascontiguousarray(KT[o:o + r]), r, nsurv)   # nsurv x W(r)
                        for S, Y in cc.items():
                            P = matmul(Kg, r, Y)
                            if S in newc:
                                newc[S] ^= P
                            else:
                                newc[S] = _spill(P) if early else P
                            del P
                        o += r
                    del KT
                    done_b = []
                    newc_saved = None
                    groups = None
            else:   # phase == "pre": survivors already formed, some preimage groups done
                nsurv = st["nsurv"]
                newc = {S: CK.arr(f) for S, f in st["newc"].items()}
                newc_saved = dict(st["newc"])
                done_b = st["done_b"]
                early = True          # disk-backed (read-only memmaps); never modified in place
            if newc is not None:
                spill = sum(Y.nbytes for Y in newc.values()) > SPILL_BYTES
                if spill and not early:
                    for S in list(newc):
                        newc[S] = _spill(newc[S])
                groups = [(nsurv, newc)]
                newc_files = newc_saved
                byb = {}
                for Sp in Sps:
                    if Sp not in comps:
                        RT = residualT(nsurv, newc, {}, Sp)
                        assert RT is None or not RT.any(), "nonzero residual with empty source block"
                        continue
                    byb.setdefault(comps[Sp], []).append(Sp)
                for b, lst in byb.items():
                    if b in done_b:
                        continue
                    if CK.due() or (CK.d and newc_files is None):
                        first = newc_files is None
                        newc_files = dict(newc_files or {})
                        for S_, Y_ in newc.items():
                            if S_ not in newc_files:
                                newc_files[S_] = CK.put(Y_)
                        CK.save({"g": g, "li": li, "phase": "pre", "info": info, "groups": [], "nsurv": nsurv,
                                 "newc": newc_files, "done_b": list(done_b)})
                        if first:
                            # continue from the saved (read-only, disk-backed) copies, as after a resume
                            for S_ in list(newc):
                                newc[S_] = CK.arr(newc_files[S_])
                            _cleanup_spill()
                    need = []
                    for Sp in lst:
                        RT = residualT(nsurv, newc, {}, Sp)
                        if RT is None or not RT.any():
                            continue
                        need.append((Sp, RT))
                    if need:
                        Big, Wt, piv, pw, plb, rest = LL.echelon(b)
                        Wb = Big.shape[1] - Wt
                        for Sp, RT in need:
                            X = np.zeros((nsurv, Wt + Wb), dtype=np.uint64)
                            RR = transpose_bits(RT, LL.block(tgts[Sp]).size, nsurv)
                            X[:, :RR.shape[1]] = RR
                            del RR
                            replay(X, Big, piv, pw, plb, Wt + Wb)
                            assert not X[:, :Wt].any(), "functional test passed but residual not in image"
                            newc[Sp] = _spill(X[:, Wt:]) if spill else X[:, Wt:].copy()
                            del X
                        del Big, need
                        LL.echcache.clear()
                    done_b.append(b)
                    if MEMDEBUG:
                        import psutil
                        print(f"      [mem] pre-phase block {b} done ({len(done_b)}/{len(byb)}): private "
                              f"{psutil.Process().memory_info().private / 2**30:.2f} GB, "
                              f"tab {len(LL.tabcache)} ker {len(LL.kcache)} cok {len(LL.fcache)}", flush=True)
        for Sp in sorted(S for S in comps if popcount(S) == i):
            b = comps[Sp]
            if not kerdims.get(b, 0):
                continue
            K = LL.kernel(b)
            assert K.shape[0] == kerdims[b], (b, K.shape[0], kerdims[b])
            groups.append((K.shape[0], {Sp: K}))
        phase = "start"
        if CK.d and li + 1 < len(layers) and (CK.due() or ncand() > 20000):
            CK.save({"g": g, "li": li + 1, "phase": "start", "info": info, "groups": save_groups(groups)})
        if verbose:
            print(f"    after layer {i}: {ncand()} candidates in {len(groups)} groups", flush=True)
    # merge groups into one dense candidate set
    nc = ncand()
    cand = {}
    o = 0
    for r, cc in groups:
        for S, Y in cc.items():
            if S not in cand:
                cand[S] = np.zeros((nc, Y.shape[1]), dtype=np.uint64)
            cand[S][o:o + r] = Y
        o += r
    groups = None
    newc = None
    _cleanup_spill()
    if nc:
        Wc = nwords(nc)
        Out = {Sp: np.zeros((LL.block(t).size, Wc), dtype=np.uint64) for Sp, t in tgts.items()}
        for S, Y in cand.items():
            b = comps[S]
            Yt = transpose_bits(Y, nc, LL.block(b).size)
            for T, (r, c) in LL.table(b).items():
                if T & S:
                    continue
                scatter_apply(Yt, r, c, Out[S | T])
        assert all(not o_.any() for o_ in Out.values()), "final candidates are not in the kernel"
        stacked = np.concatenate([cand[S] for S in sorted(cand)], axis=1)
        from gf2 import gf2_rank
        assert gf2_rank(stacked) == nc
    CK.clear()
    return nc, info, cand


def to_monomials(LL, g, cand, row):
    """explicit level-M monomials of candidate `row`"""
    out = []
    for S, Y in cand.items():
        b = LL.block(sub(g, LL.wS[S]))
        bits = np.unpackbits(Y[row].view(np.uint8), bitorder="little")[:b.size]
        for x in b[np.nonzero(bits)[0]]:
            out.append((int(x) << LL.s) | S)
    return sorted(out)
