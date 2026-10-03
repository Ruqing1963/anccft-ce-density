"""Task A4: model fits for x_m = dim C_m / |G_m|  (= 3 * dim C_m / n_m).
Data: exact dim C_m for m = 2..7, graded upper bound 64216 for m = 8 (used as a separate data set).
All fits are least squares on log x_m (i.e. relative errors)."""
import numpy as np
from scipy.optimize import curve_fit

G = {2: 8, 3: 64, 4: 256, 5: 2048, 6: 16384, 7: 65536, 8: 524288, 9: 4194304}
C = {2: 7, 3: 37, 4: 100, 5: 547, 6: 3170, 7: 10226}
Cgr = {2: 7, 3: 37, 4: 100, 5: 547, 6: 3182, 7: 10226, 8: 64216}
NF = {2: 2, 3: 3, 4: 5, 5: 10, 6: 12, 7: 16}          # filled from taskA1 output
LL = {m: 1 + sum(k * (3 if k % 3 else 2) for k in range(1, m)) for m in range(2, 10)}


def fit(name, f, ms, y, p0, bounds=(-np.inf, np.inf), inf=True):
    ms = np.array(ms, float)
    ly = np.log(y)
    try:
        p, cov = curve_fit(lambda m, *p: np.log(np.maximum(f(m, *p), 1e-300)), ms, ly, p0=p0, bounds=bounds,
                           maxfev=200000)
    except Exception as e:  # noqa
        print(f"  {name}: fit failed ({e})")
        return None
    res = ly - np.log(f(ms, *p))
    rms = np.sqrt(np.mean(res ** 2))
    err = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else [np.nan] * len(p)
    ps = ", ".join(f"{v:.5g}+-{e:.2g}" for v, e in zip(p, err))
    print(f"  {name:38s} params [{ps}]  rms rel.err {rms:.4f}  max {np.max(np.abs(res)):.4f}  "
          f"pred m=8: {f(8.0, *p):.5f}  m=9: {f(9.0, *p):.5f}  m->inf: {f(1e9, *p) if inf else 0.0:.5f}")
    return p, rms


def run(label, data, ms_list):
    print(f"\n=== {label} ===")
    for ms in ms_list:
        y = np.array([data[m] / G[m] for m in ms])
        print(f" m = {ms}: x_m = {np.round(y, 6).tolist()}  (c_e-scale x/3 = {np.round(y / 3, 6).tolist()})")
        fit("power  C m^-a", lambda m, C, a: C * m ** (-a), ms, y, [1, 1])
        fit("power+const  c + C m^-a (c>=0)", lambda m, c, C, a: c + C * m ** (-a), ms, y, [0.01, 1, 1],
            bounds=([0, 0, 0], [1, 100, 20]))
        fit("exp+const  c + C r^m (c>=0)", lambda m, c, C, r: c + C * r ** m, ms, y, [0.01, 1, 0.7],
            bounds=([0, 0, 0], [1, 100, 1]))
        fit("C / LL(m)  (LL = Loewy length)", lambda m, C: C / np.interp(m, list(LL), list(LL.values())), ms, y, [1], inf=False)
        fit("C / LL(m)^a", lambda m, C, a: C / np.interp(m, list(LL), list(LL.values())) ** a, ms, y, [1, 1], inf=False)
        fit("C / log|G_m| ^a", lambda m, C, a: C / np.interp(m, list(G), np.log2(list(G.values()))) ** a, ms, y,
            [1, 1], inf=False)
        # profile in c: best rms with c fixed
        prof = []
        for c in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]:
            if c >= y.min():
                break
            r = fit(f"   profile c={c:.2f}: c + C m^-a", lambda m, C, a, c=c: c + C * m ** (-a), ms, y, [1, 1])
            if r:
                prof.append((c, r[1]))


if __name__ == "__main__":
    run("exact dim C_m / |G_m|", C, [[2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7], [4, 5, 6, 7]])
    run("graded upper bound coker / |G_m|", Cgr, [[2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8]])
    print("\n=== C_m * N / |G_m| for N = N_F(m) (from taskA1), Loewy length LL ===")
    for m in sorted(C):
        x = C[m] / G[m]
        nf = NF[m]
        print(f" m={m}: x={x:.5f}  x*N_F={x * nf if nf else float('nan'):.4f}  x*LL={x * LL[m]:.4f}  "
              f"x*LL/3={x * LL[m] / 3:.4f}  x*m={x * m:.4f}  x*m^2={x * m * m:.4f}")

    print("\n=== local exponents a_m = log(x_{m-1}/x_m)/log(m/(m-1)), ratios, Aitken on x ===")
    for label, data in (("exact", C), ("graded", Cgr)):
        ms = sorted(data)
        x = {m: data[m] / G[m] for m in ms}
        print(" ", label, "ratios", [round(x[m] / x[m - 1], 4) for m in ms[1:]])
        print(" ", label, "local exponents", [round(np.log(x[m - 1] / x[m]) / np.log(m / (m - 1)), 3) for m in ms[1:]])
        ait = []
        for m in ms[2:]:
            a, b, c = x[m - 2], x[m - 1], x[m]
            ait.append(round((a * c - b * b) / (a + c - 2 * b), 5))
        print(" ", label, "Aitken limits (x units; c_e = x/3)", ait)
