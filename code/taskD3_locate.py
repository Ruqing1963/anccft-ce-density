r"""Task D3: where is the excess?  For m = 5..10 (taskC1_m{m}.pkl, taskD2_m{m}.pkl):
 * excess_gamma = min(d_gamma, d_{gamma+delta}) - rank_gamma  (= ker - max(0, d - d'));  excess_gamma = excess_{gamma*}  (duality)
 * by principal degree, relative to the middle (top-3)/2, and relative to k_m
 * by 'imbalance' of gamma = (n0,n1,n2): spread = max - min, and by sign of d_gamma - d_{gamma+delta}
 * lower-half kernel vs h_{n - k_m} (the hypothesis 'ker_n <= C h_{n-k_m}')
 * Hoeffding tail for the conditional bound."""
import math
import pickle
from collections import defaultdict

for m in range(5, 11):
    with open(f"taskC1_m{m}.pkl", "rb") as fh:
        C = pickle.load(fh)
    dims, ker, top = C["dims"], C["ker"], C["top"]
    N = sum(dims.values())
    h = defaultdict(int)
    for g, d in dims.items():
        h[sum(g)] += d
    kn = defaultdict(int)
    ex = {}
    for g, d in dims.items():
        dh = dims.get((g[0] + 1, g[1] + 1, g[2] + 1), 0)
        ex[g] = ker[g] - max(0, d - dh)
        kn[sum(g)] += ker[g]
    E = sum(ex.values())
    mid = (top - 3) / 2
    km = min(n for n in kn if kn[n] and n + 3 <= top)
    print(f"\n=== m={m}: |G|={N}, top={top}, middle (top-3)/2 = {mid}, k_m = {km}, total excess {E} = {E / N:.5f}|G|")
    # (1) by side of the middle
    lo = sum(e for g, e in ex.items() if sum(g) < mid)
    md = sum(e for g, e in ex.items() if sum(g) == mid)
    hi = sum(e for g, e in ex.items() if sum(g) > mid)
    lok = sum(k for n, k in kn.items() if n < mid)
    print(f"  excess below / at / above middle: {lo} / {md} / {hi};  whole kernel below middle {lok}"
          f"  (bound e_m <= 2*ker_below/|G| + mid = {(2 * lok + md) / N:.5f})")
    # (2) by d vs d'
    s = defaultdict(int)
    for g, e in ex.items():
        d, dh = dims[g], dims.get((g[0] + 1, g[1] + 1, g[2] + 1), 0)
        s["d<d'" if d < dh else ("d=d'" if d == dh else "d>d'")] += e
    print(f"  excess by sign of d_g - d_(g+delta): {dict(s)}")
    # (3) by imbalance spread (after removing the multiple of delta)
    sp = defaultdict(int)
    spn = defaultdict(int)
    for g, e in ex.items():
        spr = max(g) - min(g)
        sp[spr] += e
        spn[spr] += dims[g]
    print("  excess by spread max(g)-min(g): " + ", ".join(f"{k}: {sp[k]} ({sp[k] / max(1, spn[k]) * 100:.2f}% of dim)" for k in sorted(sp) if sp[k]))
    # (4) excess at the middle degree relative to h_mid
    nm = int(math.floor(mid))
    exn = defaultdict(int)
    for g, e in ex.items():
        exn[sum(g)] += e
    print(f"  excess/h at the middle degree n={nm}: {exn[nm]}/{h[nm]} = {exn[nm] / h[nm]:.4f}")
    # (5) lower-half kernel vs h_{n-k_m}
    rat = [(n, kn[n], h[n - km], round(kn[n] / h[n - km], 3)) for n in range(km, nm + 1)]
    print(f"  ker_n / h_(n-k_m) for k_m <= n <= mid: {[r[3] for r in rat]}")
    print(f"     max ratio {max(r[3] for r in rat)}")
    # (6) the bound sum_{n <= mid} h_{n-k_m} / |G| and Hoeffding
    tail = sum(h[n - km] for n in range(km, nm + 1)) / N
    S2 = sum(k * k * (3 if k % 3 else 2) for k in range(1, m))
    t = top / 2 - (nm - km)
    print(f"  P(deg <= mid - k_m) = {tail:.5f};  Hoeffding bound exp(-2 t^2/sum k^2 d_k) with t = top/2 - (mid - k_m) = {t}: "
          f"{math.exp(-2 * t * t / S2):.4f}")
