"""Kernel of R_sigma on (u_m)_n for n in [nlo, nhi] only (blockwise, exact).  usage: python thresh.py p d m nlo nhi
appends to out_thresh_p{p}_d{d}_m{m}.txt"""
import sys
import time
import numpy as np
from fam import Fam, rank_mod_p, block_matrix


def main():
    p, d, m, lo, hi = map(int, sys.argv[1:6])
    F = Fam(p, d, m)
    sig = F.sigma(1)
    fn = f"out_thresh_p{p}_d{d}_m{m}.txt"
    for n in range(lo, hi + 1):
        t0 = time.time()
        src = F.monomials_by_block([n])
        tgt = F.monomials_by_block([n + d])
        tot = 0
        ker = 0
        big = 0
        for g, rows in src.items():
            cols = tgt.get(tuple(a + 1 for a in g), [])
            ci = {mo: j for j, mo in enumerate(cols)}
            rk = rank_mod_p(block_matrix(F, rows, ci, sig), p) if cols else 0
            ker += len(rows) - rk
            tot += len(rows)
            big = max(big, len(rows))
        F.memo.clear()
        line = f"m={m} n={n}: dim={tot} maxblock={big} ker={ker} ({time.time()-t0:.0f}s)"
        print(line, flush=True)
        with open(fn, "a", encoding="utf-8") as fo:
            fo.write(line + "\n")


if __name__ == "__main__":
    main()
