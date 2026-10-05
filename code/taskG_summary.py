"""Task G: per-degree summary of a taskG_M{M}_a{a}.pkl checkpoint.  usage: python taskG_summary.py M [a]"""
import pickle
import sys
from collections import defaultdict

from taskG_dims import block_dims
from taskG_core import orbit_rep

M = int(sys.argv[1])
a = int(sys.argv[2]) if len(sys.argv) > 2 else 10
res = pickle.load(open(f"taskG_M{M}_a{a}.pkl", "rb"))
B = res["blocks"]
nmax = max(sum(g) for g in B)
bd = block_dims(M, nmax)
per = defaultdict(lambda: [0, 0, 0, 0.0, 0, 0, 0])
for g, (dim, info, dt, d) in B.items():
    n = sum(g)
    p = per[n]
    p[0] += 1
    p[1] += dim
    p[2] += sum(x[1] for x in info)
    p[3] += dt
    p[4] = max(p[4], d)
    p[5] += sum(x[1] - x[2] for x in info)  # survivors summed over layer tests
    p[6] += d
print("n | reps done / reps | ker (reps) | ker (all blocks, x3) | largest block | sum dims reps | candidates tested | time s")
for n in sorted(per):
    nreps = sum(1 for g in bd if sum(g) == n and orbit_rep(g) == g)
    p = per[n]
    print(f"{n} | {p[0]}/{nreps} | {p[1]} | {3 * p[1]} | {p[4]} | {p[6]} | {p[2]} | {p[3]:.0f}")
for g, ks in res["kernels"].items():
    print("kernel block", g, "n =", sum(g), ":", [len(k) for k in ks], "terms")
