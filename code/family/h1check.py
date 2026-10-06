"""(H1) check from sweep per-degree data: max over n <= mid of ker_n / h_{n-k_m}; also excess over the degree baseline.
usage: python h1check.py out_sweep_p3_d2.txt [d]"""
import sys
import json

fn = sys.argv[1]
d = int(sys.argv[2]) if len(sys.argv) > 2 else 2
lines = open(fn, encoding="utf-8").read().splitlines()
m = None
for ln in lines:
    if ln.startswith("m="):
        m = int(ln.split(":")[0][2:])
    if ln.startswith("  perdeg"):
        D = {int(k): v for k, v in json.loads(ln[len("  perdeg "):]).items()}
        top = max(D)
        mid = top / 2
        h = {n: D[n][0] for n in D}
        ker = {n: D[n][1] for n in D}
        ks = [n for n in D if ker[n]]
        if not ks:
            continue
        k = min(ks)
        ratios = [(ker[n] / h[n - k], n) for n in D if k <= n <= mid and h.get(n - k, 0)]
        rmax = max(ratios) if ratios else (0, None)
        tot = sum(h.values())
        base = sum(max(0, h[n] - h.get(n + d, 0)) for n in D)
        kt = sum(ker.values())
        print(f"m={m}: k_m={k} mid={mid:.0f} | H1 ratio max={rmax[0]:.3f} at n={rmax[1]} (ratio at n=k: {ker[k]/h[0]:.0f})"
              f" | x_m={kt/tot:.5f} baseline={base/tot:.5f} excess={(kt-base)/tot:.5f}")
