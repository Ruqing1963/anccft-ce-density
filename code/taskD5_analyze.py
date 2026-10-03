r"""Task D5: m = 11, lower half (n <= mid = 72).  Blocks with dim <= 20000 computed exactly (taskC2_low.py 11 0 72 20000),
larger ones bounded by the central-filtration inequality from m = 10 (res['bound'], an upper bound for ker).
excess_gamma = ker_gamma - max(0, d_gamma - d_{gamma+delta});  e_m|G| = 2 sum_{n<72} excess + excess_{n=72}.
Outputs exact excess on computed blocks, rigorous upper bound for e_11, comparison by spread with m = 9, 10."""
import pickle
from collections import defaultdict

from taskC4_model import qpoly

m = 11
P = qpoly(m)
N = sum(P.values())
with open("taskC2_m11.pkl", "rb") as fh:
    R = pickle.load(fh)
top = 147
mid = 72
ex_known = defaultdict(int)
dim_known = defaultdict(int)
ex_ub = defaultdict(int)
dim_all = defaultdict(int)
unb = []
spread_ex = defaultdict(int)
spread_dim = defaultdict(int)
for g, d in P.items():
    n = sum(g)
    if n > mid:
        continue
    dh = P.get((g[0] + 1, g[1] + 1, g[2] + 1), 0)
    base = max(0, d - dh)
    dim_all[n] += d
    if g in R["known"]:
        e = R["ker"][g] - base
        assert e >= 0
        ex_known[n] += e
        ex_ub[n] += e
        dim_known[n] += d
        s = max(g) - min(g)
        spread_ex[s] += e
        spread_dim[s] += d
    else:
        b = R["bound"].get(g)
        if b is None:
            unb.append(g)
            ex_ub[n] += min(d, dh)
        else:
            ex_ub[n] += max(0, min(b, d) - base)
print(f"m=11: |G|={N}; blocks with n <= 72 not exactly known: {sum(1 for g in P if sum(g) <= mid and g not in R['known'])}"
      f" (of which without a filtration bound: {len(unb)})")
print(" n | h_n | dim computed | excess (computed blocks) | upper bound on excess (all blocks)")
for n in range(0, mid + 1):
    if ex_ub[n] or ex_known[n]:
        print(f" {n} | {dim_all[n]} | {dim_known[n]} | {ex_known[n]} | {ex_ub[n]}")
lowE = 2 * sum(ex_known[n] for n in range(mid)) + ex_known[mid]
upE = 2 * sum(ex_ub[n] for n in range(mid)) + ex_ub[mid]
print(f"exact excess on computed blocks (doubled by duality): {lowE} -> e_11 >= {lowE / N:.5f}")
print(f"rigorous upper bound: e_11 <= {upE / N:.5f};  computed fraction of the lower half: "
      f"{sum(dim_known.values()) / sum(dim_all.values()):.3f}")
print("excess / dim by spread on computed blocks (n <= 72): " +
      ", ".join(f"{s}: {spread_ex[s]}/{spread_dim[s]} = {100 * spread_ex[s] / spread_dim[s]:.2f}%" for s in sorted(spread_dim)))
# same statistic for m = 9, 10 restricted to blocks of dim <= 20000 and n <= mid
for mm in (9, 10):
    with open(f"taskC1_m{mm}.pkl", "rb") as fh:
        C = pickle.load(fh)
    se = defaultdict(int)
    sd = defaultdict(int)
    md = (C["top"] - 3) / 2
    for g, d in C["dims"].items():
        if sum(g) > md or d > 20000:
            continue
        dh = C["dims"].get((g[0] + 1, g[1] + 1, g[2] + 1), 0)
        s = max(g) - min(g)
        se[s] += C["ker"][g] - max(0, d - dh)
        sd[s] += d
    print(f"m={mm} (dim <= 20000, n <= mid): " + ", ".join(f"{s}: {100 * se[s] / sd[s]:.2f}%" for s in sorted(sd)))
