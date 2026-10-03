r"""Task D1: explain the generic corank-1 blocks at m = 9.  Is there y != 0 in the block with y*b = 0 for ALL six basis
elements b of (u_9)_delta (a common kernel => every element of weight delta, over every field, is non-injective there)?
Compute the left kernel of the horizontally stacked matrix [M_1 | ... | M_6]."""
import numpy as np
from cengine import Split, Blocks
from taskD_common import DeltaSpace, block_matrix
from gf2null import left_kernel

m = 9
sp = Split(m)
B = Blocks(sp)
ds = DeltaSpace(sp, B)
for g in [(12, 17, 18), (12, 18, 17), (14, 15, 20), (13, 16, 19), (13, 19, 16)]:
    Ms = [block_matrix(sp, B, g, [w]) for w in ds.words]
    P = np.hstack(Ms)
    K, rk = left_kernel(P, P.shape[1] * 64)
    print(f"block {g}: dim {B.dim(g)}, common left kernel of the 6 basis elements: {K.shape[0]}", flush=True)
    for v in K:
        terms = [sp.mstr(int(B.blocks[g][j])) for j in range(B.dim(g)) if (int(v[j >> 6]) >> (j & 63)) & 1]
        print(f"   y = {' + '.join(terms)}")
    # also: kernels of the individual basis elements
    from gf2 import gf2_rank_inplace
    h = (g[0] + 1, g[1] + 1, g[2] + 1)
    print(f"   individual coranks: {[B.dim(g) - int(gf2_rank_inplace(M.copy(), B.dim(h))) for M in Ms]}")
