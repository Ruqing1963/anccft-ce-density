"""Task A1: Jordan type of right multiplication R_F : x -> xF on F2[G_m].

F^k is computed in the group algebra as a coefficient vector (f_k = f_{k-1} F = XOR of the rows g F of the
matrix of R_F over g in supp f_{k-1}).  The matrix of R_{F^k} (row g = g F^k) is then built by left
translation along a BFS spanning tree of the Cayley graph w.r.t. the generators u_b = 1 + b phi
(b = 1, alpha, alpha^2): row_{u g} = pi_u(row_g), pi_u = left multiplication permutation of the codes.
r_k = rank R_{F^k}; Jordan blocks of size j: b_j = r_{j-1} - 2 r_j + r_{j+1} (r_0 = |G|).
usage: python taskA1_jordan.py 2 3 4 5 6 7
"""
import sys
import time

import numpy as np
from numba import njit, prange

from common import F_matrix_packed, all_elements, encode_arr, lmul_arr, nbits
from gf2 import gf2_rank_inplace


def left_perms(m):
    C = all_elements(m)
    perms = []
    for b in (1, 2, 4):
        h = tuple([1, b] + [0] * (m - 2))
        p = encode_arr(lmul_arr(h, C, m), m)
        assert np.unique(p).size == C.shape[0]
        perms.append(p)
    return np.array(perms, dtype=np.int64)


def bfs_tree(perms, N):
    parent = -np.ones(N, dtype=np.int64)
    gen = -np.ones(N, dtype=np.int64)
    parent[0] = 0          # code 0 = identity
    levels = [np.array([0], dtype=np.int64)]
    seen = np.zeros(N, dtype=bool)
    seen[0] = True
    while True:
        fr = levels[-1]
        nxt = []
        for gi in range(perms.shape[0]):
            img = perms[gi][fr]
            new = ~seen[img]
            img, src = img[new], fr[new]
            img, first = np.unique(img, return_index=True)
            src = src[first]
            seen[img] = True
            parent[img] = src
            gen[img] = gi
            nxt.append(img)
        nxt = np.concatenate(nxt)
        if nxt.size == 0:
            break
        levels.append(nxt)
    assert seen.all(), "u_b do not generate"
    return parent, gen, levels


@njit(parallel=True)
def _translate_level(P, lev, parent, gen, perms, W):
    for ii in prange(lev.size):
        g = lev[ii]
        p = parent[g]
        pm = perms[gen[g]]
        for k in range(W):
            P[g, k] = 0
        for w in range(W):
            v = P[p, w]
            while v:
                low = v & (~v + np.uint64(1))
                b = 0
                t = low
                while t > np.uint64(1):
                    t >>= np.uint64(1)
                    b += 1
                j = pm[w * 64 + b]
                P[g, j >> 6] |= np.uint64(1) << np.uint64(j & 63)
                v ^= low


@njit(parallel=True)
def _times_F(M1, f, W):
    """f (packed vector) times F = XOR of rows M1[g] over set bits g of f"""
    N = M1.shape[0]
    nb = 16
    part = np.zeros((nb, W), dtype=np.uint64)
    chunk = (N + nb - 1) // nb
    for c in prange(nb):
        for g in range(c * chunk, min(N, (c + 1) * chunk)):
            if (f[g >> 6] >> np.uint64(g & 63)) & np.uint64(1):
                for k in range(W):
                    part[c, k] ^= M1[g, k]
    out = np.zeros(W, dtype=np.uint64)
    for c in range(nb):
        for k in range(W):
            out[k] ^= part[c, k]
    return out


def popcount(f):
    return int(np.unpackbits(f.view(np.uint8)).sum())


def jordan(m, verbose=True):
    t0 = time.time()
    M1, _ = F_matrix_packed(m)
    N, W = M1.shape
    perms = left_perms(m)
    parent, gen, levels = bfs_tree(perms, N)
    if verbose:
        print(f"m={m}: N={N}, BFS depth {len(levels) - 1}, setup {time.time() - t0:.1f}s", flush=True)
    f = M1[0].copy()                       # F itself (row of the identity)
    ranks = [N]
    supps = []
    P = np.zeros((N, W), dtype=np.uint64)
    k = 1
    while popcount(f):
        supps.append(popcount(f))
        t1 = time.time()
        P[0] = f
        for lev in levels[1:]:
            _translate_level(P, lev, parent, gen, perms, W)
        t2 = time.time()
        if k == 1:
            assert np.array_equal(P, M1), "translation check failed"
        r = int(gf2_rank_inplace(P, N))
        ranks.append(r)
        if verbose:
            print(f"  k={k:2d}: |supp F^k|={supps[-1]:6d}  rank R_(F^k)={r:6d}   "
                  f"(build {t2 - t1:.1f}s, rank {time.time() - t2:.1f}s)", flush=True)
        f = _times_F(M1, f, W)
        k += 1
    NF = k                                 # first k with F^k = 0
    ranks.append(0)                        # r_{N_F}
    ranks.append(0)
    blocks = {j: ranks[j - 1] - 2 * ranks[j] + ranks[j + 1] for j in range(1, NF + 1)}
    blocks = {j: c for j, c in blocks.items() if c}
    assert sum(j * c for j, c in blocks.items()) == N
    return N, NF, ranks[:NF + 1], blocks, supps


if __name__ == "__main__":
    for m in [int(a) for a in sys.argv[1:]]:
        t0 = time.time()
        N, NF, ranks, blocks, supps = jordan(m)
        top = sum(k * (3 if k % 3 else 2) for k in range(1, m))
        nblk = sum(blocks.values())
        print(f"m={m}: |G_m|={N}  N_F={NF}  Loewy length={top + 1} (ceil(LL/3)={-(-(top + 1) // 3)})")
        print(f"m={m}: ranks r_k (k=0..N_F) = {ranks}")
        print(f"m={m}: |supp F^k| = {supps}")
        print(f"m={m}: Jordan blocks {{size: count}} = {blocks}")
        print(f"m={m}: #blocks = dim C_m = {nblk};  |G|/N_F = {N / NF:.2f};  dim C_m / (|G|/N_F) = {nblk * NF / N:.4f};"
              f"  mean block size = {N / nblk:.3f}   [{time.time() - t0:.1f}s]", flush=True)
