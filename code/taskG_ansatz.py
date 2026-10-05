r"""Task G: search for eta-type kernel elements of R_sigma' at level m in degree d(m)-1 inside a small Ansatz space.

Ansatz (generalising eta at m = 9, THEORY 2.5): PBW monomials in the Q-block wt(omega_j) - beta_p whose multiset of letter
degrees is obtained from the column degree set H_m by changing at most `dist` entries by an L1-total of `dist`
(dist = 1: one degree k -> k-1, k = 1 meaning removal).  The kernel of y -> y*sigma' restricted to the Ansatz span is
computed exactly (any nonzero solution is a genuine kernel element, re-verified by straightening).
A solution proves k_m <= d(m) - 1; no solution proves nothing.
usage: python taskG_ansatz.py m [dist]
"""
import itertools
import sys
import time
from collections import Counter

import numpy as np

from cengine import Split, wt, pack_words
from taskE_lift import products


def Hset(m):
    H = [k for k in range(1, m) if k % 3]
    r = 0
    while 3 * 2 ** r < m:
        H.append(3 * 2 ** r)
        r += 1
    return sorted(H)


def omega_weight(m, j):
    w = [0, 0, 0]
    for k in Hset(m):
        x = wt(k, (k - 1 + j) % 3)
        for c in range(3):
            w[c] += x[c]
    return tuple(w)


def multisets(H, r, m, shift=-1):
    """degree multisets D = H - A + B with A a sub-multiset of H, |A| <= r, B a multiset of degrees in [1, m-1],
    |B| <= r, sum(B) = sum(A) + shift"""
    out = set()
    base = Counter(H)
    for na in range(0, r + 1):
        for A in itertools.combinations(sorted(H), na):
            sB = sum(A) + shift
            if sB < 0:
                continue
            for nb in range(0, r + 1):
                for B in itertools.combinations_with_replacement(range(1, m), nb):
                    if sum(B) != sB:
                        continue
                    c = Counter(base)
                    c.subtract(Counter(A))
                    c.update(Counter(B))
                    out.add(tuple(sorted(c.elements())))
    return out


def monomials_for(sp, degs, target):
    """PBW monomials (masks) with the given multiset of letter degrees and Q-weight target"""
    cnt = Counter(degs)
    choices = []
    for k, c in cnt.items():
        if k >= sp.m:
            return []
        lets = [sp.pos[(k, p)] for p in (range(3) if k % 3 else range(2))]
        if c > len(lets):
            return []
        choices.append(list(itertools.combinations(lets, c)))
    out = []
    for combo in itertools.product(*choices):
        mask = 0
        w = np.zeros(3, dtype=np.int64)
        for grp in combo:
            for i in grp:
                mask |= 1 << i
                w += sp.lwt[i]
        if tuple(int(x) for x in w) == target:
            out.append(mask)
    return out


def left_null(rows):
    """rows: list of python-int bitsets; returns basis of {c : sum c_i rows_i = 0} as lists of row indices"""
    piv = {}
    basis = []
    for i, r in enumerate(rows):
        comb = 1 << i
        while r:
            h = r.bit_length() - 1
            if h in piv:
                pr, pc = piv[h]
                r ^= pr
                comb ^= pc
            else:
                piv[h] = (r, comb)
                break
        if r == 0:
            basis.append([t for t in range(len(rows)) if (comb >> t) & 1])
    return basis


def main(m, dist):
    t0 = time.time()
    sp = Split(m)
    words, wlen = pack_words(sp.sigma_words())
    H = Hset(m)
    d = sum(H)
    print(f"m={m}: H={H}, d(m)={d}", flush=True)
    found = []
    for j in range(1):            # rho-symmetry: column j = 0 suffices (other columns are rotations)
        for p in range(3):
            tw = tuple(x - (1 if c == p else 0) for c, x in enumerate(omega_weight(m, j)))
            monos = set()
            for D in multisets(H, dist, m):
                monos.update(monomials_for(sp, D, tw))
            monos = sorted(monos)
            if not monos:
                print(f"  j={j} p={p} block {tw}: empty Ansatz", flush=True)
                continue
            arr = np.array(monos, dtype=np.int64)
            starts = np.arange(arr.size + 1, dtype=np.int64)
            outs = products(arr, starts, words, wlen, sp.br, sp.sq)
            colid = {}
            rows = []
            for o in outs:
                r = 0
                for x in o:
                    x = int(x)
                    if x not in colid:
                        colid[x] = len(colid)
                    r |= 1 << colid[x]
                rows.append(r)
            null = left_null(rows)
            print(f"  j={j} p={p} block {tw} (n={sum(tw)}): Ansatz dim {len(monos)}, image cols {len(colid)}, "
                  f"kernel in Ansatz = {len(null)}   [{time.time() - t0:.1f}s]", flush=True)
            for comb in null:
                y = arr[comb]
                chk = products(y, np.array([0, y.size], dtype=np.int64), words, wlen, sp.br, sp.sq)[0]
                assert chk.size == 0
                found.append((tw, y))
                print(f"    kernel element ({len(y)} terms, verified y*sigma'=0): " +
                      " + ".join(sp.mstr(int(x)) for x in y[:12]) + (" + ..." if len(y) > 12 else ""), flush=True)
    return found


if __name__ == "__main__":
    m = int(sys.argv[1])
    dist = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(m, dist)
