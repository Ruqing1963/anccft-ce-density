r"""Task D1: is the excess specific to sigma'?  For every nonzero F2-element x of (u^s_m)_delta (63 of them; one per
rho-orbit, since rank R_x on block g = rank R_{rho x} on block rho g), and for random F_{2^k} elements, compute
dim ker R_x block by block and compare with the Q-graded maximal-rank baseline max(0, dim u_g - dim u_{g+delta}).
usage: python taskD1_generic.py m [F2|F2k k nsamples] [maxblock]
Output pickle taskD1_m{m}_{mode}.pkl : {x(label/vec): {g: ker}}"""
import pickle
import sys
import time

import numpy as np

from cengine import Split, Blocks
from taskD_common import DeltaSpace, block_rank_F2, block_rank_F2k, set_basis_words, orbit_rep, rot

POLYS = {2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 8: 0b100011011}


def baseline(B, g):
    h = tuple(a + 1 for a in g)
    return max(0, B.dim(g) - B.dim(h))


def all_blocks_kernel(sp, B, rankfun, maxblock=None):
    ker = {}
    for g in sorted(B.blocks, key=lambda g: B.dim(g)):
        if maxblock is not None and B.dim(g) > maxblock:
            continue
        h = tuple(a + 1 for a in g)
        if B.dim(h) == 0:
            ker[g] = B.dim(g)
            continue
        ker[g] = B.dim(g) - rankfun(g)
    return ker


if __name__ == "__main__":
    m = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "F2"
    sp = Split(m)
    B = Blocks(sp)
    ds = DeltaSpace(sp, B)
    set_basis_words(ds)
    N = 1 << sp.nl
    BQ = sum(baseline(B, g) for g in B.blocks)
    print(f"m={m}: |G|={N}, B^Q={BQ} ({BQ / N:.5f}); delta basis: {[sp.mstr(b) for b in ds.basis]}", flush=True)
    print(f"  sigma' = {ds.label(ds.sigma)}  (vec {ds.sigma:06b});  rho images: {[f'{v:06b}' for v in ds.rho_img]}", flush=True)
    inv = [v for v in range(1, 64) if ds.rho(v) == v]
    print(f"  rho-invariant nonzero elements: {len(inv)}: {[ds.label(v) for v in inv]}", flush=True)
    t0 = time.time()
    out = {"m": m, "dims": {g: B.dim(g) for g in B.blocks}, "BQ": BQ, "N": N, "res": {}, "labels": {}}
    if mode == "F2":
        maxblock = int(sys.argv[3]) if len(sys.argv) > 3 else None
        seen = set()
        reps = []
        for v in range(1, 64):
            if v in seen:
                continue
            orb = {v, ds.rho(v), ds.rho(ds.rho(v))}
            seen |= orb
            reps.append(v)
        print(f"  {len(reps)} rho-orbits of nonzero F2-elements", flush=True)
        for v in reps:
            ws = ds.words_of(v)
            ker = all_blocks_kernel(sp, B, lambda g: block_rank_F2(sp, B, g, ws), maxblock)
            K = sum(ker.values())
            ex = sum(ker[g] - baseline(B, g) for g in ker)
            nbad = sum(1 for g in ker if ker[g] > baseline(B, g))
            out["res"][v] = ker
            out["labels"][v] = ds.label(v)
            tag = " <-- sigma'" if v in (ds.sigma, ds.rho(ds.sigma)) else ""
            print(f"  x={v:06b} inv={ds.rho(v) == v} ker={K} ({K / N:.5f}) excess={ex} ({ex / N:.5f}) blocks with excess {nbad}"
                  f"   x = {ds.label(v)}{tag}  [{time.time() - t0:.0f}s]", flush=True)
        with open(f"taskD1_m{m}_F2.pkl", "wb") as fh:
            pickle.dump(out, fh)
    else:
        k = int(sys.argv[3])
        ns = int(sys.argv[4])
        maxblock = int(sys.argv[5]) if len(sys.argv) > 5 else None
        rng = np.random.default_rng(1000 + m)
        for s in range(ns):
            coeffs = [int(c) for c in rng.integers(1, 1 << k, 6)]       # all nonzero, random
            ker = {}
            for g in sorted(B.blocks, key=lambda g: B.dim(g)):
                if maxblock is not None and B.dim(g) > maxblock:
                    continue
                h = tuple(a + 1 for a in g)
                if B.dim(h) == 0:
                    ker[g] = B.dim(g)
                    continue
                ker[g] = B.dim(g) - block_rank_F2k(sp, B, g, coeffs, k, POLYS[k])
            K = sum(ker.values())
            ex = sum(ker[g] - baseline(B, g) for g in ker)
            nbad = sum(1 for g in ker if ker[g] > baseline(B, g))
            out["res"][tuple(coeffs)] = ker
            print(f"  F_2^{k} sample {s} coeffs {coeffs}: ker={K} ({K / N:.5f}) excess={ex} ({ex / N:.5f}) blocks with excess {nbad}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
            bad = sorted((g, ker[g] - baseline(B, g), B.dim(g), B.dim(tuple(a + 1 for a in g))) for g in ker if ker[g] > baseline(B, g))
            print(f"     excess blocks (g, excess, dim g, dim g+delta): {bad[:40]}", flush=True)
        with open(f"taskD1_m{m}_F2k{k}.pkl", "wb") as fh:
            pickle.dump(out, fh)
