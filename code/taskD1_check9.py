r"""Task D1: the 18 square blocks at m = 9 where no specialization over F2, F16, F256 reached maximal rank:
recheck with 6 random elements over F_{2^16}, and over F_{2^k} with all 6 coefficients random (k = 16).
If the corank stays 1, det of the generic element vanishes identically on these blocks (structural failure of maximal
rank for ALL elements of weight delta)."""
import time
import numpy as np
from cengine import Split, Blocks
from taskD_common import DeltaSpace, block_rank_F2k, set_basis_words

P16 = 0b10001000000001011     # x^16 + x^12 + x^3 + x + 1 (primitive)
m = 9
sp = Split(m)
B = Blocks(sp)
ds = DeltaSpace(sp, B)
set_basis_words(ds)
blocks = [(12, 17, 18), (12, 18, 17), (14, 15, 20), (13, 16, 19), (13, 19, 16)]   # (one per duality/rotation class)
rng = np.random.default_rng(5)
for g in blocks:
    h = (g[0] + 1, g[1] + 1, g[2] + 1)
    mats = {}
    rs = []
    for t in range(6):
        coeffs = [int(c) for c in rng.integers(1, 1 << 16, 6)]
        rs.append(block_rank_F2k(sp, B, g, coeffs, 16, P16, mats))
    print(f"block {g} (n={sum(g)}): dims {B.dim(g)} -> {B.dim(h)}; ranks over F_2^16 (6 random elements): {rs}", flush=True)
