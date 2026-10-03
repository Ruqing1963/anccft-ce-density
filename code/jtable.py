"""Structure tables of A_N = F2[[P]]/I^N in the Jennings basis: for every letter z the sparse matrix
of left multiplication y -> e_z y (CSR over basis indices), built from jalg.JAlg.lmul.  Products of
arbitrary elements are then done with numba on dense uint8 vectors:  J(S) y = e_{s1}(e_{s2}(... e_{sk} y)).
Basis order: by weight, then the order of ULie.monos.  Tables are cached in tables/jt_N.npz."""
import os
import sys
import time

import numpy as np
from numba import njit, prange

from jalg import JAlg, run_big_stack


class JTable:
    def __init__(self, N, verbose=True):
        self.N = N
        fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables", f"jt_{N}.npz")
        self.A = None
        from ulie import ULie
        self.U = ULie("F8", Dl=N - 1, maxdeg=N - 1)
        U = self.U
        self.monos = []
        self.wstart = []
        for d in range(N):
            self.wstart.append(len(self.monos))
            self.monos += U.monos(d)
        self.wstart.append(len(self.monos))
        self.D = len(self.monos)
        self.idx = {m: i for i, m in enumerate(self.monos)}
        self.weight = np.zeros(self.D, dtype=np.int64)
        for d in range(N):
            self.weight[self.wstart[d]:self.wstart[d + 1]] = d
        self.nl = U.nl
        self.ldeg = U.ldeg
        if os.path.exists(fn):
            z = np.load(fn)
            self.ptr, self.ind = z["ptr"], z["ind"]
            if verbose:
                print(f"JTable N={N}: loaded {fn}, D={self.D}, nnz={self.ind.size}", flush=True)
            return
        t0 = time.time()
        A = JAlg(N)
        assert A.U.letters == U.letters
        ptr = np.zeros((self.nl, self.D + 1), dtype=np.int64)
        chunks = []
        tot = 0
        for z in range(self.nl):
            dz = self.ldeg[z]
            for i, T in enumerate(self.monos):
                ptr[z, i] = tot
                if self.weight[i] + dz < N:
                    res = A.lmul(z, T)
                    arr = np.fromiter((self.idx[t] for t in res), dtype=np.int32, count=len(res))
                    arr.sort()
                    chunks.append(arr)
                    tot += arr.size
            ptr[z, self.D] = tot
            if verbose:
                print(f"  letter {z} (deg {dz}) done, nnz so far {tot}, memo {len(A.memo)}, {time.time() - t0:.0f}s",
                      flush=True)
        self.ptr = ptr
        self.ind = np.concatenate(chunks) if chunks else np.zeros(0, np.int32)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        np.savez(fn, ptr=self.ptr, ind=self.ind)
        if verbose:
            print(f"JTable N={N}: built D={self.D}, nnz={self.ind.size}, {time.time() - t0:.0f}s", flush=True)

    def build_right(self, verbose=True):
        """tables of right multiplication y -> y e_z (cached in tables/jtR_N.npz)"""
        N = self.N
        fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables", f"jtR_{N}.npz")
        if os.path.exists(fn):
            z = np.load(fn)
            self.rptr, self.rind = z["ptr"], z["ind"]
            return
        t0 = time.time()
        A = JAlg(N)
        ptr = np.zeros((self.nl, self.D + 1), dtype=np.int64)
        chunks = []
        tot = 0
        for z in range(self.nl):
            dz = self.ldeg[z]
            for i, T in enumerate(self.monos):
                ptr[z, i] = tot
                if self.weight[i] + dz < N:
                    res = A.mul_mono(T, 1 << z)
                    arr = np.fromiter((self.idx[t] for t in res), dtype=np.int32, count=len(res))
                    arr.sort()
                    chunks.append(arr)
                    tot += arr.size
            ptr[z, self.D] = tot
        self.rptr = ptr
        self.rind = np.concatenate(chunks)
        np.savez(fn, ptr=self.rptr, ind=self.rind)
        if verbose:
            print(f"JTable N={N}: right tables built, nnz={self.rind.size}, {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- packed column blocks
    def _trie(self, x, reverse):
        trie = {}
        for i in np.nonzero(x)[0]:
            s = self.letters_of(i)
            node = trie
            for z in (reversed(s) if reverse else s):
                node = node.setdefault(z, {})
            node[-1] = True
        return trie

    def apply_block(self, x, Y, side):
        """Y: packed (D, W) uint64, column c = a vector.  Returns packed columns x*col (side 'L') or col*x ('R')."""
        out = np.zeros_like(Y)
        if side == "L":
            trie, ptr, ind = self._trie(x, True), self.ptr, self.ind
        else:
            trie, ptr, ind = self._trie(x, False), self.rptr, self.rind
        stack = [(trie, Y)]
        while stack:
            node, cur = stack.pop()
            for z, child in node.items():
                if z == -1:
                    out ^= cur
                else:
                    nxt = _apply_block(ptr[z], ind, cur)
                    stack.append((child, nxt))
        return out

    def rmul_vec(self, y, x):
        """y * x for dense uint8 vectors"""
        out = np.zeros(self.D, dtype=np.uint8)
        stack = [(self._trie(x, False), y)]
        while stack:
            node, cur = stack.pop()
            for z, child in node.items():
                if z == -1:
                    out ^= cur
                else:
                    nxt = _apply(self.rptr[z], self.rind, cur, self.D)
                    if nxt.any():
                        stack.append((child, nxt))
        return out

    # ---------------------------------------------------------------- conversions
    def vec(self, X):
        v = np.zeros(self.D, dtype=np.uint8)
        for m in X:
            if m in self.idx:
                v[self.idx[m]] ^= 1
        return v

    def elem(self, v):
        return {self.monos[i] for i in np.nonzero(v)[0]}

    def letters_of(self, i):
        m = self.monos[i]
        out = []
        while m:
            low = m & -m
            out.append(low.bit_length() - 1)
            m ^= low
        return out

    # ---------------------------------------------------------------- products
    def lmul_vec(self, x, y):
        """x * y for dense uint8 vectors (x on the left)"""
        out = np.zeros(self.D, dtype=np.uint8)
        nz = np.nonzero(x)[0]
        # group by letter sequences: process each monomial of x
        seqs = [self.letters_of(i) for i in nz]
        # trie on reversed sequences, iterative DFS
        trie = {}
        for s in seqs:
            node = trie
            for z in reversed(s):
                node = node.setdefault(z, {})
            node[-1] = True
        stack = [(trie, y)]
        while stack:
            node, cur = stack.pop()
            for z, child in node.items():
                if z == -1:
                    out ^= cur
                else:
                    nxt = _apply(self.ptr[z], self.ind, cur, self.D)
                    if nxt.any():
                        stack.append((child, nxt))
        return out

    def mul(self, x, y):
        return self.lmul_vec(x, y)

    def wparts(self, v):
        nz = np.nonzero(v)[0]
        return np.bincount(self.weight[nz], minlength=self.N)


@njit(parallel=True)
def _apply_block(ptr, ind, Y):
    D, W = Y.shape
    out = np.zeros_like(Y)
    nchunk = 8
    cs = (W + nchunk - 1) // nchunk
    for c in prange(nchunk):
        lo = c * cs
        hi = min(W, lo + cs)
        for i in range(D):
            for k in range(ptr[i], ptr[i + 1]):
                r = ind[k]
                for w in range(lo, hi):
                    out[r, w] ^= Y[i, w]
    return out


@njit
def _apply(ptr, ind, y, D):
    out = np.zeros(D, dtype=np.uint8)
    for i in range(D):
        if y[i]:
            for k in range(ptr[i], ptr[i + 1]):
                out[ind[k]] ^= 1
    return out


if __name__ == "__main__":
    def go():
        for N in [int(a) for a in sys.argv[1:]]:
            T = JTable(N)
            # self-check against JAlg on random products
            import random
            rng = random.Random(1)
            A = JAlg(N)
            for _ in range(30):
                X = {rng.choice(T.monos[1:T.wstart[min(N, 6)]]) for _ in range(3)}
                Y = {rng.choice(T.monos[1:]) for _ in range(3)}
                assert T.elem(T.mul(T.vec(X), T.vec(Y))) == A.mul(X, Y)
            print(f"N={N}: 30 random products agree with JAlg.mul", flush=True)
    run_big_stack(go)
