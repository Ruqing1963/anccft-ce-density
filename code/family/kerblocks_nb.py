"""List the Q-blocks of degree n with nonzero kernel of R_sigma at level m, and print the kernel vectors of the blocks
of dimension <= maxdim.  usage: python kerblocks_nb.py p d m n [maxdim]"""
import sys
import numpy as np
from numba import njit
from fam_nb import FamNB, build_block, rank_mod_p


@njit(cache=True)
def left_null(M, p):
    """basis of {x : x M = 0} over F_p; M uint8 (nr x nc). Returns (k x nr) uint8."""
    nr, nc = M.shape
    A = np.zeros((nr, nc + nr), dtype=np.int64)
    for i in range(nr):
        for j in range(nc):
            A[i, j] = M[i, j]
        A[i, nc + i] = 1
    r = 0
    for c in range(nc):
        piv = -1
        for i in range(r, nr):
            if A[i, c] % p:
                piv = i
                break
        if piv < 0:
            continue
        if piv != r:
            for j in range(nc + nr):
                t = A[r, j]
                A[r, j] = A[piv, j]
                A[piv, j] = t
        iv = 1
        for y in range(1, p):
            if (A[r, c] * y) % p == 1:
                iv = y
        for j in range(nc + nr):
            A[r, j] = (A[r, j] * iv) % p
        for i in range(nr):
            if i != r and A[i, c] % p:
                f = A[i, c]
                for j in range(nc + nr):
                    A[i, j] = (A[i, j] - f * A[r, j]) % p
        r += 1
    out = np.zeros((nr - r, nr), dtype=np.uint8)
    for i in range(r, nr):
        for j in range(nr):
            out[i - r, j] = A[i, nc + j]
    return out


def name(E, code):
    mono = E.decode(code)
    s = []
    for t, a in enumerate(mono):
        if a:
            k, i = E.F.letters[t]
            if E.d == 2:
                nm = ("h" if k % 2 == 0 else ("x" if i == 0 else "f")) + str(k)
            else:
                nm = f"y{k}.{i}"
            s.append(nm + (f"^{a}" if a > 1 else ""))
    return "*".join(s) if s else "1"


def main():
    p, d, m, n = map(int, sys.argv[1:5])
    maxdim = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
    only = None
    if len(sys.argv) > 6:
        only = {tuple(int(x) for x in b.split(",")) for b in sys.argv[6].split(";")}
    E = FamNB(p, d, m)
    src = E.codes_by_block(n, n)
    tgt = E.codes_by_block(n + d, n + d)
    words, wlen, wcoef = E.sigma_words(1)
    for g in sorted(src):
        if only is not None and g not in only:
            continue
        rows = src[g]
        cols = tgt.get(tuple(a + 1 for a in g))
        M, bad = build_block(rows, cols, words, wlen, wcoef, E.P, p, E.brl, E.brc, E.pml, E.pmc, 0, 1 << 20,
                             max(1, min(256, rows.size // 16)))
        assert bad == 0
        rk = rank_mod_p(M.copy(), p)
        k = rows.size - rk
        if only is not None:
            print(f"block {g}: {rows.size} -> {cols.size}, kernel dim {k}", flush=True)
            continue
        if k:
            print(f"block {g}: {rows.size} -> {cols.size}, kernel dim {k}", flush=True)
            if rows.size <= maxdim:
                N = left_null(M, p)
                for v in N:
                    terms = [f"{int(c)}*{name(E, rows[i])}" for i in np.flatnonzero(v) for c in [v[i]]]
                    print(f"   ({len(terms)} terms) " + " + ".join(terms[:40]) + (" ..." if len(terms) > 40 else ""),
                          flush=True)


if __name__ == "__main__":
    main()
