r"""Task C4: decomposition of dim ker R_sigma' on u_m into the 'maximal-rank baseline' and the excess.
baseline_m = sum_gamma max(0, dim u_gamma - dim u_{gamma+delta})   (blockwise maximal rank, Q-graded)
baseline1_m = sum_n max(0, h_n - h_{n+3})                           (degreewise maximal rank)
excess = dim ker - baseline (>= 0).  Data from taskC1_m{m}.pkl.  Also: Q-graded baseline for larger m (generating
function only), to extrapolate the density under the 'maximal rank + small excess' model."""
import pickle
import sys
from collections import defaultdict

from cengine import wt


def qpoly(m):
    poly = {(0, 0, 0): 1}
    for k in range(1, m):
        for p in (range(3) if k % 3 else range(2)):
            w = wt(k, p)
            new = defaultdict(int, poly)
            for g, c in poly.items():
                new[(g[0] + w[0], g[1] + w[1], g[2] + w[2])] += c
            poly = new
    return poly


def baselines(poly):
    hb = defaultdict(int)
    for g, c in poly.items():
        hb[sum(g)] += c
    b1 = sum(max(0, hb[n] - hb.get(n + 3, 0)) for n in hb)
    bq = sum(max(0, c - poly.get((g[0] + 1, g[1] + 1, g[2] + 1), 0)) for g, c in poly.items())
    return b1, bq, max(hb.values())


if __name__ == "__main__":
    print(" m |   |G_m|    | dim ker  | ker/|G|  | baseline_deg | baseline_Q | ker-baseline_Q | excess/|G| | excess by degree half (n<mid / n>=mid) | 3h_max/|G|")
    for m in range(2, 11):
        try:
            with open(f"taskC1_m{m}.pkl", "rb") as fh:
                r = pickle.load(fh)
        except FileNotFoundError:
            continue
        poly = qpoly(m)
        assert poly == {g: c for g, c in r["dims"].items()}
        b1, bq, hmax = baselines(poly)
        N = sum(poly.values())
        K = sum(r["ker"].values())
        top = r["top"]
        ex_lo = ex_hi = 0
        for g, k in r["ker"].items():
            d = r["dims"][g]
            e = k - max(0, d - poly.get((g[0] + 1, g[1] + 1, g[2] + 1), 0))
            assert e >= 0
            if 2 * sum(g) < top:
                ex_lo += e
            else:
                ex_hi += e
        print(f"{m:2d} | {N:9d} | {K:8d} | {K / N:.5f} | {b1:9d} | {bq:9d} | {K - bq:8d} | {(K - bq) / N:.5f} | {ex_lo} / {ex_hi} | {3 * hmax / N:.5f}")
    print("\nQ-graded maximal-rank baseline for larger m (model: ker = baseline_Q + excess):")
    for m in range(2, int(sys.argv[1]) + 1 if len(sys.argv) > 1 else 17):
        poly = qpoly(m)
        b1, bq, hmax = baselines(poly)
        N = sum(poly.values())
        print(f"   m={m:2d}: |G|=2^{N.bit_length() - 1}, baseline_Q/|G| = {bq / N:.5f}, baseline_deg/|G| = {b1 / N:.5f}, "
              f"-> /n_m: {bq / (3 * N):.5f}", flush=True)
