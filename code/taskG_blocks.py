"""Task G: exact ker R^{(M)} on selected Q-blocks (lifting from level a, taskG_core.lift_block2).
usage: python taskG_blocks.py M a "g0,g1,g2" ["h0,h1,h2" ...]   -> appends to taskG_blocks_M{M}.pkl"""
import pickle
import sys
import time

from taskG_core import LowLevel, lift_block2, to_monomials
from taskG_dims import block_dims

M, a = int(sys.argv[1]), int(sys.argv[2])
gs = [tuple(int(x) for x in s.split(",")) for s in sys.argv[3:]]
kerdims = pickle.load(open(f"taskC1_m{a}.pkl", "rb"))["ker"]
LL = LowLevel(M, a)
bd = block_dims(M, max(sum(g) for g in gs))
fn = f"taskG_blocks_M{M}.pkl"
try:
    res = pickle.load(open(fn, "rb"))
except FileNotFoundError:
    res = {}
for g in gs:
    t0 = time.time()
    dim, info, cand = lift_block2(LL, g, kerdims)
    mons = [to_monomials(LL, g, cand, r) for r in range(dim)]
    res[g] = (dim, info, time.time() - t0, bd.get(g), mons)
    pickle.dump(res, open(fn, "wb"))
    print(f"M={M} block {g} n={sum(g)} dim={bd.get(g)}: layer tests {info} -> ker = {dim}  [{time.time() - t0:.1f}s]",
          flush=True)
    for r, ms in enumerate(mons):
        print(f"   kernel vector {r} ({len(ms)} terms): " + " + ".join(LL.hi.mstr(x) for x in ms), flush=True)
