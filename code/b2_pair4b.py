"""Task B2: lifting graded pairs v*y = 0 / y*v = 0 with v in V (u_4) and y in ker(L_v) / ker(R_v) on u_b, b = 4..9,
with the exact scheme of b2_pair314.py (y-corrections fully free in weights b+1..w-4, v-corrections of weight
w-b only, earlier v-corrections enumerated / sampled).  For b = 4 this complements b2_lift.py dfs-pair.
usage: python b2_pair4b.py N budget branch b1 b2 ..."""
import random
import sys
import time

import b2_pair314 as M
from jtable import JTable
from jalg import run_big_stack
from b2common import get_V
from b2_lift import lowest_weight
from ulie import ULie
from verify_zd import kernel_vectors

if __name__ == "__main__":
    N, budget, branch = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    bs = [int(a) for a in sys.argv[4:]]

    def go():
        T = JTable(N)
        T.build_right()
        V = get_V(T)
        U = ULie("F8", Dl=14)
        for b in bs:
            M.P_, M.Q_ = 4, b
            for side in ("L", "R"):          # L: v*y = 0 (y right annihilator); R: y*v = 0
                K = kernel_vectors(U, V[0], 4, b, side)
                if 2 ** len(K) - 1 > 63:
                    rr = random.Random(b)
                    combos = [rr.randrange(1, 2 ** len(K)) for _ in range(63)]
                else:
                    combos = list(range(1, 2 ** len(K)))
                summary = {}
                for c in combos:
                    Y = set()
                    for i in range(len(K)):
                        if (c >> i) & 1:
                            Y ^= K[i]
                    t1 = time.time()
                    if side == "L":
                        x, y = T.vec(V[0]), T.vec(Y)
                    else:
                        # y*v = 0: treat as pair (y, v) with roles swapped: x = y (deg b), y = v (deg 4)
                        x, y = T.vec(Y), T.vec(V[0])
                        M.P_, M.Q_ = b, 4
                    w0 = lowest_weight(T, T.mul(x, y))
                    st = {"nodes": 0, "visits": {}, "obs": {}, "kA": {}, "sampled": set()}
                    best = M.dfs(T, x, y, w0, random.Random(c), budget, branch, st)
                    M.P_, M.Q_ = 4, b
                    summary[(w0, best)] = summary.get((w0, best), 0) + 1
                    print(f"b={b} side {side} (dim ker {len(K)}) y#{c}: start I^{w0}, max reached I^{best}; nodes {st['nodes']}; "
                          f"visits {st['visits']}; obstructed {st['obs']}; free-kernel dims "
                          f"{({k: sorted(v) for k, v in st['kA'].items()})}; sampled at {sorted(st['sampled'])}  "
                          f"[{time.time() - t1:.0f}s]", flush=True)
                print(f"SUMMARY b={b} side {side}: dim ker {len(K)}, {len(combos)} leading terms y tried; "
                      f"(start, max reached) -> count: {summary}", flush=True)
    run_big_stack(go)
