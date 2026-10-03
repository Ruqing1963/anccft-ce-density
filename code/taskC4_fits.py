r"""Task C4: (a) check of the duality rank R(gamma) = rank R(theta(Gamma - gamma - delta)), theta(n0,n1,n2) = (n1,n0,n2)
(Frobenius pairing on u_m + the anti-automorphism Theta'(e_p phi^k) = e_{k-p} phi^k fixing sigma');
(b) density fits with the new exact values m = 9, 10."""
import math
import pickle

import numpy as np

from cengine import Split

for m in range(3, 11):
    r = pickle.load(open(f"taskC1_m{m}.pkl", "rb"))
    sp = Split(m)
    Gam = tuple(int(x) for x in sp.lwt.sum(axis=0))
    bad = 0
    for g, rk in r["rank"].items():
        mu = tuple(Gam[i] - g[i] - 1 for i in range(3))
        th = (mu[1], mu[0], mu[2])
        if r["rank"].get(th, 0) != rk:
            bad += 1
    print(f"m={m}: Gamma={Gam}; duality rank R(g) = rank R(theta(Gamma-g-delta)) fails on {bad} of {len(r['rank'])} blocks")

x = {2: 7 / 8, 3: 37 / 64, 4: 100 / 256, 5: 547 / 2048, 6: 3182 / 16384, 7: 10226 / 65536, 8: 64216 / 524288,
     9: 423412 / 4194304, 10: 1526176 / 16777216}
print("\nx_m = graded coker / |G_m| (= 3 * graded bound on dim C_m / n_m):", {m: round(v, 6) for m, v in x.items()})
print("successive ratios:", {m: round(x[m] / x[m - 1], 4) for m in range(3, 11)})
print("local exponents log(x_{m-1}/x_m)/log(m/(m-1)):", {m: round(math.log(x[m - 1] / x[m]) / math.log(m / (m - 1)), 3) for m in range(3, 11)})


def fit_power(ms):
    A = np.array([[1, -math.log(m)] for m in ms])
    b = np.array([math.log(x[m]) for m in ms])
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = b - A @ sol
    return math.exp(sol[0]), sol[1], float(np.sqrt(np.mean(res ** 2)))


def fit_pc(ms, cs=np.linspace(0, 0.08, 161)):
    best = None
    for c in cs:
        if any(x[m] - c <= 0 for m in ms):
            continue
        A = np.array([[1, -math.log(m)] for m in ms])
        b = np.array([math.log(x[m] - c) for m in ms])
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        pred = c + np.exp(A @ sol)
        rms = float(np.sqrt(np.mean((np.log(pred) - np.log([x[m] for m in ms])) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, c, math.exp(sol[0]), sol[1])
    return best


for ms in ([4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10], [6, 7, 8, 9, 10], [8, 9, 10]):
    C, a, rms = fit_power(ms)
    print(f"power law C m^-a on m={ms}: C={C:.3f}, a={a:.3f}, rms(log)={rms:.4f}; predicts x_9={C * 9 ** -a:.4f}, x_10={C * 10 ** -a:.4f}, x_12={C * 12 ** -a:.4f}")
    if len(ms) >= 4:
        b = fit_pc(ms)
        print(f"   c + C m^-a best: c={b[1]:.4f} (c_e={b[1] / 3:.4f}), C={b[2]:.3f}, a={b[3]:.3f}, rms={b[0]:.4f}")
# out-of-sample check of the A4 fits
print("A4 predictions (fit on exact m=4..7) vs truth: power law C=3.86 a=1.66 -> x_9 =", round(3.86 * 9 ** -1.66, 4),
      " x_10 =", round(3.86 * 10 ** -1.66, 4), "; truth", round(x[9], 4), round(x[10], 4))
