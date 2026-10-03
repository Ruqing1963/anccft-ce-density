"""Task B2: Hensel/obstruction lifting in A_N = F2[[P]]/I^N (Jennings basis, exact ungraded products).

square mode:  x~ = v + (weight >= 5), try to reach x~^2 in I^N.
pair mode  :  x~ = v + ..., y~ = w + ..., try to reach x~ y~ in I^N.
At weight w (current residual r = x~^2 or x~y~ in I^w) the following linear systems are tried in order:
  'greedy' : corrections of weight exactly w-4 only (graded: r_w in im ad_v, resp. in R_w u + v u);
  'exact'  : corrections of weights ceil((w+1)/2) .. w-4  (all quadratic terms delta^2 / a b lie in I^{w+1},
             so this is an exact linear system: solvable  <=>  some such correction works);
  'full'   : corrections of weights 5 .. w-4 (linearised; the quadratic term may spoil lower weights, so a
             solution is accepted only after checking; up to `trials` random kernel shifts).
Any accepted correction is shifted by a random element of the kernel of the system (seeded), so that
different seeds explore different lifts.
usage: python b2_lift.py N mode(square|pair) seeds..."""
import math
import random
import sys
import time

import numpy as np

from jtable import JTable
from jalg import run_big_stack
from b2common import get_V, unit_block, block_columns_as_ints, vec_to_int, Solver


def lowest_weight(T, r):
    wp = T.wparts(r)
    nz = np.nonzero(wp)[0]
    return int(nz[0]) if nz.size else T.N


def try_step(T, x, y, w, lo, hi, square, rng, trials):
    """find corrections of weights lo..hi making (x+a)(y+b) (or (x+d)^2) lie in I^{w+1}"""
    if lo > hi:
        return None, "empty"
    nrows = T.wstart[w + 1]
    cols = list(range(T.wstart[lo], T.wstart[hi + 1]))
    K = len(cols)
    Y = unit_block(T, cols)
    if square:
        img = T.apply_block(x, Y, "L") ^ T.apply_block(x, Y, "R")
        ints = block_columns_as_ints(img, nrows, K)
    else:
        ints = (block_columns_as_ints(T.apply_block(y, Y, "R"), nrows, K)          # a -> a y
                + block_columns_as_ints(T.apply_block(x, Y, "L"), nrows, K))       # b -> x b
    r = T.mul(x, y if not square else x)
    S = Solver(ints)
    tag, res = S.solve(vec_to_int(r, nrows))
    if tag is None:
        return None, f"not in image (rank {S.rank}, #unknowns {len(ints)}, kernel {len(S.kernel)})"
    for t in range(trials):
        tg = tag
        for kv in S.kernel:
            if rng.random() < 0.5:
                tg ^= kv
        if t == 0 and trials > 1 and lo >= math.ceil((w + 1) / 2):
            pass
        a = np.zeros(T.D, dtype=np.uint8)
        b = np.zeros(T.D, dtype=np.uint8)
        for k in range(len(ints)):
            if (tg >> k) & 1:
                if square or k < K:
                    a[cols[k]] ^= 1
                else:
                    b[cols[k - K]] ^= 1
        x2 = x ^ a
        y2 = (x2 if square else y ^ b)
        r2 = T.mul(x2, y2)
        lw = lowest_weight(T, r2)
        if lw > w:
            return (x2, y2, lw), f"ok (rank {S.rank}, kernel {len(S.kernel)}, trial {t})"
    return None, f"linearised solution spoiled by quadratic term in all {trials} trials"


def solution_space(T, x, y, w, lo, hi, square):
    """affine space of corrections (weights lo..hi) solving the exact linear system at weight w
    (exact when lo >= ceil((w+1)/2)).  Returns (cols, nunk, particular tag, kernel tags) or None."""
    nrows = T.wstart[w + 1]
    cols = list(range(T.wstart[lo], T.wstart[hi + 1]))
    K = len(cols)
    Y = unit_block(T, cols)
    if square:
        ints = block_columns_as_ints(T.apply_block(x, Y, "L") ^ T.apply_block(x, Y, "R"), nrows, K)
    else:
        ints = (block_columns_as_ints(T.apply_block(y, Y, "R"), nrows, K)
                + block_columns_as_ints(T.apply_block(x, Y, "L"), nrows, K))
    r = T.mul(x, x if square else y)
    S = Solver(ints)
    tag, _ = S.solve(vec_to_int(r, nrows))
    if tag is None:
        return None
    return cols, len(ints), tag, S.kernel


def materialize(T, x, y, sol, tg, square):
    cols, nunk, _, _ = sol
    K = len(cols)
    a = np.zeros(T.D, dtype=np.uint8)
    b = np.zeros(T.D, dtype=np.uint8)
    for k in range(nunk):
        if (tg >> k) & 1:
            if square or k < K:
                a[cols[k]] ^= 1
            else:
                b[cols[k - K]] ^= 1
    x2 = x ^ a
    return x2, (x2 if square else y ^ b)


def dfs(T, x, y, w, square, rng, budget, branch, stats, depth=0, path=""):
    """exact backtracking search.  At weight w the correction weights are ceil((w+1)/2)..w-4 (exact linear
    system); its affine solution set is enumerated exhaustively if 2^dim <= branch, else `branch` random
    samples.  Returns the max k reached (residual in I^k)."""
    stats["nodes"] += 1
    if w >= T.N:
        return T.N
    if stats["nodes"] > budget:
        return w
    lo = max(5, math.ceil((w + 1) / 2))
    sol = solution_space(T, x, y, w, lo, w - 4, square)
    stats.setdefault("visits", {}).setdefault(w, 0)
    stats["visits"][w] += 1
    if sol is None:
        stats.setdefault("obstructed", {}).setdefault(w, 0)
        stats["obstructed"][w] += 1
        return w
    _, _, tag, ker = sol
    stats.setdefault("kerdim", {}).setdefault(w, set()).add(len(ker))
    if 2 ** len(ker) <= branch:
        choices = range(2 ** len(ker))
        exhaustive = True
    else:
        choices = [rng.getrandbits(len(ker)) for _ in range(branch)]
        exhaustive = False
    if exhaustive:
        stats.setdefault("exhaustive", set()).add(w)
    else:
        stats.setdefault("sampled", set()).add(w)
    best = w
    for c in choices:
        tg = tag
        for i, kv in enumerate(ker):
            if (c >> i) & 1:
                tg ^= kv
        x2, y2 = materialize(T, x, y, sol, tg, square)
        nw = lowest_weight(T, T.mul(x2, y2))
        assert nw > w
        best = max(best, dfs(T, x2, y2, nw, square, rng, budget, branch, stats, depth + 1))
        if best >= T.N or stats["nodes"] > budget:
            break
    return best


def run(T, x0, y0, square, seed, trials=8, log=print):
    rng = random.Random(seed)
    x, y = T.vec(x0), T.vec(y0)
    r = T.mul(x, x if square else y)
    w = lowest_weight(T, r)
    log(f"  start: residual in I^{w}")
    while w < T.N:
        done = False
        for mode, lo in (("greedy", w - 4), ("exact", max(5, math.ceil((w + 1) / 2))), ("full", 5)):
            res, msg = try_step(T, x, y, w, lo, w - 4, square, rng, 1 if mode != "full" else trials)
            if res is not None:
                x, y, nw = res
                log(f"  w={w:2d}: {mode:6s} {msg} -> residual in I^{nw}")
                w = nw
                done = True
                break
            log(f"  w={w:2d}: {mode:6s} weights {lo}..{w - 4}: {msg}")
        if not done:
            log(f"  OBSTRUCTED at weight {w}: best achieved residual in I^{w}")
            return w, x, y
    log(f"  reached residual in I^{T.N} (= 0 in A_N)")
    return T.N, x, y


if __name__ == "__main__":
    N = int(sys.argv[1])
    mode = sys.argv[2]
    seeds = [int(s) for s in sys.argv[3:]] or [0]

    def go():
        t0 = time.time()
        T = JTable(N)
        T.build_right()
        V = get_V(T)
        # 7 nonzero elements of V
        Vall = []
        for c in range(1, 8):
            e = set()
            for i in range(3):
                if (c >> i) & 1:
                    e ^= V[i]
            Vall.append(e)
        print(f"N={N}, D={T.D}, setup {time.time() - t0:.0f}s", flush=True)
        results = []
        if mode.startswith("dfs"):
            # dfs-square / dfs-pair: exact backtracking search; seeds[0] = budget, seeds[1] = branch
            budget, branch = seeds[0], seeds[1]
            sq = mode == "dfs-square"
            jobs = ([(f"v{c + 1}", Vall[c], Vall[c]) for c in range(7)] if sq else
                    [(f"(v{c + 1},v{d + 1})", Vall[c], Vall[d]) for c in range(7) for d in range(7)])
            if len(seeds) > 2:
                jobs = jobs[:seeds[2]]
            for name, a, b in jobs:
                t1 = time.time()
                x, y = T.vec(a), T.vec(b)
                w0 = lowest_weight(T, T.mul(x, x if sq else y))
                stats = {"nodes": 0}
                best = dfs(T, x, y, w0, sq, random.Random(1), budget, branch, stats)
                print(f"{mode} {name}: start I^{w0}, max reached I^{best}; nodes {stats['nodes']}; "
                      f"visits/weight {stats.get('visits')}; obstructed/weight {stats.get('obstructed')}; "
                      f"kernel dims {({k: sorted(v) for k, v in stats.get('kerdim', {}).items()})}; "
                      f"exhaustive at {sorted(stats.get('exhaustive', []))}, sampled at {sorted(stats.get('sampled', []))}"
                      f"  [{time.time() - t1:.0f}s]", flush=True)
            return
        if mode == "square":
            jobs = [(f"v{c + 1}", Vall[c], Vall[c]) for c in range(7)]
        else:
            jobs = [(f"(v{c + 1},v{d + 1})", Vall[c], Vall[d]) for c in range(7) for d in range(7)]
        for name, a, b in jobs:
            for s in seeds:
                t1 = time.time()
                print(f"{mode} {name} seed {s}:", flush=True)
                reached, _, _ = run(T, a, b, mode == "square", seed=s,
                                    log=lambda m: print(m, flush=True))
                results.append((name, s, reached))
                print(f"  [{time.time() - t1:.0f}s]", flush=True)
        print("\nSUMMARY (max k with residual in I^k):")
        for name, s, k in results:
            print(f"  {name} seed {s}: {k}")
    run_big_stack(go)
