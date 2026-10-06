"""numba version of thresh.py: kernel of R_sigma on (u_m)_n for n in [lo, hi].
usage: python thresh_nb.py p d m lo hi [orient]   -> appends to out_threshnb_p{p}_d{d}_m{m}.txt"""
import sys
import time
from fam_nb import FamNB, kernel_by_degree


def main():
    p, d, m, lo, hi = map(int, sys.argv[1:6])
    o = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    E = FamNB(p, d, m)
    fn = f"out_threshnb_p{p}_d{d}_m{m}.txt"
    for n in range(lo, hi + 1):
        t0 = time.time()
        r = kernel_by_degree(E, n, n, orient=o)
        dim, ker, big = r.get(n, [0, 0, 0])
        line = f"m={m} n={n}: dim={dim} maxblock={big} ker={ker} ({time.time()-t0:.0f}s)"
        print(line, flush=True)
        with open(fn, "a", encoding="utf-8") as fo:
            fo.write(line + "\n")


if __name__ == "__main__":
    main()
