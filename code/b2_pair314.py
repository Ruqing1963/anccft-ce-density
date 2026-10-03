"""Task B2: lifting the (3,14) graded zero-divisor pair x*y = 0 (x in u_3, y in u_14, y in ker L_x) to A_N.
x~ = x + a (a of weight >= 4), y~ = y + b (b of weight >= 15).  At weight w (x~ y~ in I^w):
  b is kept FULLY free in weights 15..w-3, a free in weights max(4, w-14)..w-14 (= w-14 only); all products a*b
  then lie in I^{w+1}, so the system  r + a y~ + x~ b = 0 (weights <= w) is exactly linear.  Lower a-components
  are fixed by the ancestors; the a-projection of the solution space is enumerated (exhaustively if
  2^dim <= branch, else `branch` random samples).  Every y in the 3-dim kernel (7 nonzero) is tried.
usage: python b2_pair314.py N budget branch"""
import random
import sys
import time

import numpy as np

from jtable import JTable
from jalg import run_big_stack
from b2common import get_x3, unit_block, block_columns_as_ints, vec_to_int, Solver
from b2_lift import lowest_weight
from ulie import ULie
from verify_zd import kernel_vectors

P_, Q_ = 3, 14


def space(T, x, y, w):
    nrows = T.wstart[w + 1]
    r0 = T.wstart[P_ + Q_ + 1]          # all images and the residual vanish below weight P+Q+1
    ca = list(range(T.wstart[w - Q_], T.wstart[w - Q_ + 1]))
    cb = list(range(T.wstart[Q_ + 1], T.wstart[w - P_ + 1]))
    IA = T.apply_block(y, unit_block(T, ca), "R")
    IB = T.apply_block(x, unit_block(T, cb), "L")
    r = T.mul(x, y)
    assert not IA[:r0].any() and not IB[:r0].any() and not r[:r0].any()
    ints = block_columns_as_ints(IA, nrows, len(ca), r0) + block_columns_as_ints(IB, nrows, len(cb), r0)
    S = Solver(ints)
    tag, _ = S.solve(vec_to_int(r, nrows, r0))
    return ca, cb, S, tag


def split(T, ca, cb, tg):
    a = np.zeros(T.D, dtype=np.uint8)
    b = np.zeros(T.D, dtype=np.uint8)
    for k in range(len(ca) + len(cb)):
        if (tg >> k) & 1:
            if k < len(ca):
                a[ca[k]] ^= 1
            else:
                b[cb[k - len(ca)]] ^= 1
    return a, b


def dfs(T, x, y, w, rng, budget, branch, st):
    st["nodes"] += 1
    if w >= T.N or st["nodes"] > budget:
        return w
    ca, cb, S, tag = space(T, x, y, w)
    st["visits"][w] = st["visits"].get(w, 0) + 1
    if tag is None:
        st["obs"][w] = st["obs"].get(w, 0) + 1
        return w
    # a-projection of the kernel
    maskA = (1 << len(ca)) - 1
    pa = {}
    for kv in S.kernel:
        v = kv & maskA
        while v:
            h = v.bit_length() - 1
            if h in pa:
                v ^= pa[h][0]
                kv ^= pa[h][1]
            else:
                pa[h] = (v, kv)
                break
    kA = [kv for (_, kv) in pa.values()]
    st["kA"].setdefault(w, set()).add(len(kA))
    if 2 ** len(kA) <= branch:
        choices = range(2 ** len(kA))
    else:
        choices = [rng.getrandbits(len(kA)) for _ in range(branch)]
        st["sampled"].add(w)
    best = w
    for c in choices:
        tg = tag
        for i, kv in enumerate(kA):
            if (c >> i) & 1:
                tg ^= kv
        a, b = split(T, ca, cb, tg)
        x2, y2 = x ^ a, y ^ b
        nw = lowest_weight(T, T.mul(x2, y2))
        assert nw > w
        best = max(best, dfs(T, x2, y2, nw, rng, budget, branch, st))
        if best >= T.N or st["nodes"] > budget:
            break
    return best


if __name__ == "__main__":
    N, budget, branch = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    def go():
        t0 = time.time()
        T = JTable(N)
        T.build_right()
        X = get_x3(T)
        U = ULie("F8", Dl=17)
        K = kernel_vectors(U, X, 3, 14, "L")
        print(f"x = {sorted(T.U.mono_str(m) for m in X)}; dim ker(L_x on u_14) = {len(K)}  [{time.time() - t0:.0f}s]",
              flush=True)
        for c in range(1, 2 ** len(K)):
            Y = set()
            for i in range(len(K)):
                if (c >> i) & 1:
                    Y ^= K[i]
            x, y = T.vec(X), T.vec(Y)
            w0 = lowest_weight(T, T.mul(x, y))
            st = {"nodes": 0, "visits": {}, "obs": {}, "kA": {}, "sampled": set()}
            t1 = time.time()
            best = dfs(T, x, y, w0, random.Random(c), budget, branch, st)
            print(f"y = kernel combination {c}: start I^{w0}, max reached I^{best}; nodes {st['nodes']}; visits {st['visits']}; "
                  f"obstructed {st['obs']}; a-kernel dims {({k: sorted(v) for k, v in st['kA'].items()})}; "
                  f"sampled at {sorted(st['sampled'])}  [{time.time() - t1:.0f}s]", flush=True)
    run_big_stack(go)
