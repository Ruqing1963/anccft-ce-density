r"""Task D4: the conditional bound and its numerical size.
Proved (THEORY 5.2): e_m <= (2 * sum_{n < mid} ker_n + ker_mid)/|G_m| <= 2 sum_{n <= mid} ker_n / |G_m|,  mid = (top-3)/2.
Hypothesis H1(C): ker_n <= C h_{n - k_m} for k_m <= n <= mid.  Then e_m <= 2C P(X_m <= mid - k_m), X_m = degree of a
uniformly random PBW monomial, and by Hoeffding P(X_m <= top/2 - t) <= exp(-2 t^2 / sum_{k<m} k^2 d_k), t = k_m + 3/2.
Also the structural (omega) part: <= 6 * 2^{-|H_m|}.
Prints, for m = 5..30: P(X_m <= mid - k_m) with k_m = d(m) (conjectural for m >= 12, d(9)-1 at m=9), the bound 6P, the
Hoeffding bound, 6*2^{-|H_m|}."""
import math

import numpy as np


def hilb(m):
    top = sum(k * (3 if k % 3 else 2) for k in range(1, m))
    h = np.zeros(top + 1, dtype=object)
    h[0] = 1
    for k in range(1, m):
        for _ in range(3 if k % 3 else 2):
            h[k:] = h[k:] + h[:-k].copy()
    return h, top


def Hset(m):
    return [k for k in range(1, m) if k % 3] + [3 * 2 ** r for r in range(20) if 3 * 2 ** r < m]


known_k = {5: 10, 6: 15, 7: 21, 8: 28, 9: 35, 10: 36, 11: 46}
known_e = {5: 0.01465, 6: 0.00952, 7: 0.00247, 8: 0.00217, 9: 0.00267, 10: 0.00411}
print(" m | k_m (d(m)) | mid | P(X<=mid-k) | 6P (=bound under H1(3)) | Hoeffding 6*exp | e_m (exact) | e_m/(2P) | 6*2^-|H_m|")
for m in range(5, 31):
    h, top = hilb(m)
    N = sum(h)
    H = Hset(m)
    k = known_k.get(m, sum(H))
    mid = (top - 3) / 2
    P = sum(h[: int(math.floor(mid)) - k + 1]) / N
    S2 = sum(kk * kk * (3 if kk % 3 else 2) for kk in range(1, m))
    hoef = math.exp(-2 * (k + 1.5) ** 2 / S2)
    e = known_e.get(m)
    es = f"{e:.5f}" if e else "   -   "
    rr = f"{e / (2 * float(P)):.2f}" if e else "  - "
    print(f"{m:2d} | {k:4d} ({sum(H):4d}) | {mid:6.1f} | {float(P):.6f} | {6 * float(P):.5f} | {6 * hoef:.4f} | {es} | {rr} | {6 * 2.0 ** -len(H):.5f}",
          flush=True)
