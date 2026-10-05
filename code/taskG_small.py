"""Task G: exact ker R^{(M)} on all rho-orbit representative blocks of degree n whose E_1 (from level a) is <= cap,
in increasing order of E_1, with a time limit; checkpoint taskG_small_M{M}.pkl.
usage: python taskG_small.py M a n cap timelimit"""
import itertools
import pickle
import sys
import time
from collections import defaultdict

from cengine import wt
from taskG_core import LowLevel, lift_block2, to_monomials, orbit_rep
from taskG_dims import block_dims

M, a, n, cap, tlim = (int(x) for x in sys.argv[1:6])
t0 = time.time()
kerdims = pickle.load(open(f"taskC1_m{a}.pkl", "rb"))["ker"]
Il = [(k, p) for k in range(M - 1, a - 1, -1) for p in (range(3) if k % 3 else range(2))]
subs = defaultdict(int)
for r in range(len(Il) + 1):
    for S in itertools.combinations(Il, r):
        w = [0, 0, 0]
        for l in S:
            x = wt(*l)
            for c in range(3):
                w[c] += x[c]
        subs[tuple(w)] += 1
bd = block_dims(M, n)
reps = [g for g in bd if sum(g) == n and orbit_rep(g) == g]
E1 = {g: sum(mu * kerdims.get((g[0] - w[0], g[1] - w[1], g[2] - w[2]), 0) for w, mu in subs.items()) for g in reps}
fn = f"taskG_small_M{M}.pkl"
try:
    res = pickle.load(open(fn, "rb"))
except FileNotFoundError:
    res = {}
LL = LowLevel(M, a)
todo = sorted((g for g in reps if E1[g] <= cap), key=lambda g: E1[g])
print(f"M={M} n={n}: {len(reps)} reps, {len(todo)} with E1 <= {cap}; E1 of the others: "
      f"{sorted(E1[g] for g in reps if E1[g] > cap)}", flush=True)
for g in todo:
    if g in res:
        continue
    if time.time() - t0 > tlim:
        print("time limit", flush=True)
        break
    tb = time.time()
    dim, info, cand = lift_block2(LL, g, kerdims)
    res[g] = (dim, info, time.time() - tb, bd[g], E1[g], [to_monomials(LL, g, cand, r) for r in range(dim)])
    pickle.dump(res, open(fn, "wb"))
    print(f"  {g} dim={bd[g]} E1={E1[g]}: tests {info} -> ker = {dim}  [{time.time() - tb:.1f}s]", flush=True)
