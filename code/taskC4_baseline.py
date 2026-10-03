r"""Task C4: the maximal-rank baseline B_m = sum_n max(0, h_n - h_{n+3}) / |G_m| (degreewise; the Q-graded version agrees
to 4 digits for m <= 15, see taskC4_model.py), a rigorous LOWER bound for (graded coker)/|G_m| = dim ker R_sigma / |G_m|,
for m up to 90; fitted exponent.  h = coefficients of prod_{k<m} (1+z^k)^{d_k}."""
import math

import numpy as np

res = {}
for m in range(2, 91):
    top = sum(k * (3 if k % 3 else 2) for k in range(1, m))
    h = np.zeros(top + 1, dtype=object)
    h[0] = 1
    for k in range(1, m):
        for _ in range(3 if k % 3 else 2):
            h[k:] = h[k:] + h[:-k].copy() if k <= top else h[k:]
    N = sum(h)
    B = sum(max(0, h[n] - (h[n + 3] if n + 3 <= top else 0)) for n in range(top + 1))
    res[m] = B / N
    if m <= 16 or m % 5 == 0:
        print(f"m={m:3d}: B_m = {float(res[m]):.6f}   B_m/3 = {float(res[m]) / 3:.6f}   m^1.5*B_m = {float(res[m]) * m ** 1.5:.4f}", flush=True)
ms = list(range(30, 91))
A = np.array([[1.0, -math.log(m)] for m in ms])
b = np.array([math.log(float(res[m])) for m in ms])
sol, *_ = np.linalg.lstsq(A, b, rcond=None)
print(f"fit B_m ~ C m^-a on m=30..90: C={math.exp(sol[0]):.3f}, a={sol[1]:.4f}")
