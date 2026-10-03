"""Task B2 (graded part): obstruction spaces for lifting v^2 = 0 and v*w = 0 (v, w in V) to F2[[P]].
ad_v = L_v + R_v : u_i -> u_{i+4} (char 2: va + av = [v, a]);  ad_v^2 = ad_{v^2} = 0.
For a lift x~ = v + ..., the leading part r_w of x~^2 automatically lies in ker(ad_v on u_w)
(since [x~, x~^2] = 0), and it can be killed by a correction of weight w-4 iff r_w in im(ad_v on u_{w-4}).
So the obstruction space at weight w is H_w = ker ad_v|u_w / im ad_v|u_{w-4}.
For pairs: (a, b) -> a w + v b, u_{j-4} + u_{j-4} -> u_j; cokernel dims (no automatic constraint known).
usage: python b2_graded.py D"""
import sys
import time

import numpy as np

from ulie import ULie
from zdtools import mult_matrix, elem_from_bits
from gf2 import gf2_rank

if __name__ == "__main__":
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 22
    t0 = time.time()
    U = ULie("F8", Dl=D)
    V = [elem_from_bits(U, 4, g) for g in (4005, 524823, 527794)]
    v, w = V[0], V[1]
    rk = {}
    pair_vv, pair_vw = {}, {}
    for i in range(0, D - 3):
        PL, nc = mult_matrix(U, v, 4, i, "L")
        PR, _ = mult_matrix(U, v, 4, i, "R")
        rk[i] = gf2_rank(PL ^ PR, nc)
        pair_vv[i + 4] = gf2_rank(np.vstack([PL, PR]), nc)
        PRw, _ = mult_matrix(U, w, 4, i, "R")
        pair_vw[i + 4] = gf2_rank(np.vstack([PL, PRw]), nc)
        print(f"i={i:2d}: dim u_i={PL.shape[0]:6d}, rank ad_v: u_i -> u_(i+4) = {rk[i]:6d}, ker = {PL.shape[0] - rk[i]:5d};"
              f"  coker (a,b)->av+vb in u_{i + 4}: {nc - pair_vv[i + 4]};  (a,b)->a w+v b: {nc - pair_vw[i + 4]}"
              f"  [{time.time() - t0:.0f}s]", flush=True)
    print("\nObstruction spaces H_w = ker(ad_v | u_w) / im(ad_v | u_{w-4}):")
    for wt in range(4, D - 3):
        dim_u = len(U.monos(wt))
        ker = dim_u - rk[wt]
        im = rk.get(wt - 4, 0)
        print(f"  w={wt:2d}: dim u_w={dim_u:6d}  dim ker ad_v={ker:5d}  dim im ad_v={im:5d}  dim H_w={ker - im:4d}")
