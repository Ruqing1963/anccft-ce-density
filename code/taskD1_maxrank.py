r"""Task D1 (certificate of generic maximal rank).  Rank is lower semicontinuous: for the generic element
x = sum t_i b_i of (u_m)_delta (t_i indeterminates), rank R_x on block g >= rank R_{x(c)} for every specialization c.
So if on every block SOME specialization (F2 element, or F_{2^k} element) has maximal rank min(d_g, d_{g+delta}), the
generic element has maximal rank on every block, i.e. zero excess (and a single element over a large enough field does).
Step 1: per-block best over the 63 F2 elements (taskD1_m{m}_F2.pkl; rotations used: ker_{rho v}(rho g) = ker_v(g)).
Step 2: for the remaining blocks, random F_{2^k} elements (k = 4, then 8) until maximal rank is reached.
usage: python taskD1_maxrank.py m [k1 k2 ...]"""
import pickle
import sys
import time

import numpy as np

from cengine import Split, Blocks
from taskD_common import DeltaSpace, block_rank_F2k, set_basis_words, rot

POLYS = {2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 8: 0b100011011, 16: 0b10001000000001011}

if __name__ == "__main__":
    m = int(sys.argv[1])
    ks = [int(x) for x in sys.argv[2:]] or [4, 8]
    with open(f"taskD1_m{m}_F2.pkl", "rb") as fh:
        D = pickle.load(fh)
    dims = D["dims"]
    best = {}
    who = {}
    for v, ker in D["res"].items():
        for g, k in ker.items():
            for i in range(3):
                gg = g
                for _ in range(i):
                    gg = rot(gg)
                if gg not in best or k < best[gg]:
                    best[gg] = k
                    who[gg] = (v, i)
    bad = []
    for g, d in dims.items():
        dh = dims.get((g[0] + 1, g[1] + 1, g[2] + 1), 0)
        if dh == 0:
            continue
        if best[g] > max(0, d - dh):
            bad.append(g)
    print(f"m={m}: {len(dims)} blocks; best-over-F2 has excess on {len(bad)} blocks: "
          f"{sorted(bad, key=lambda g: dims[g])[:60]}", flush=True)
    if not bad:
        sys.exit(0)
    sp = Split(m)
    B = Blocks(sp)
    ds = DeltaSpace(sp, B)
    set_basis_words(ds)
    rng = np.random.default_rng(77 + m)
    t0 = time.time()
    remaining = sorted(bad, key=lambda g: dims[g])
    for k in ks:
        still = []
        for g in remaining:
            d, dh = dims[g], dims[(g[0] + 1, g[1] + 1, g[2] + 1)]
            target = min(d, dh)
            ok = False
            mats = {}
            for trial in range(4):
                coeffs = [int(c) for c in rng.integers(1, 1 << k, 6)]
                r = block_rank_F2k(sp, B, g, coeffs, k, POLYS[k], mats)
                if r == target:
                    ok = True
                    break
            if not ok:
                still.append((g, d, dh, target - r))
        print(f"  F_2^{k}: blocks still without a maximal-rank specialization: {len(still)}: {still[:40]}  [{time.time() - t0:.0f}s]",
              flush=True)
        remaining = [s[0] for s in still]
        if not remaining:
            break
    if not remaining:
        print(f"m={m}: CERTIFIED: the generic element of weight delta has maximal rank on every Q-block (excess 0).")
