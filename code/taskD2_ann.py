r"""Task D2: the 'annihilator part' of ker R_sigma' and the structural prediction.

For each restricted subalgebra K with sigma' in K*u (taskD2_ideals.py), Ann_right(K) = u*omega_K lies in ker R_sigma'.
A := sum_K Ann(K) (all such K, or only the three columns H^(j) with --cols).  Per Q-block gamma:
    a_gamma = dim A_gamma  (computed exactly: rows w*omega_K, w = PBW monomials in letters complementary to K;
                            these are a basis of Ann(K) by PBW, see THEORY 2.3)
By duality (THEORY 1.6(iv)) dim coker R_gamma = dim ker R_{gamma*}, gamma* = theta(Gamma - gamma - delta), so
    rank R_gamma <= min(d_gamma - a_gamma, d_{gamma+delta} - a_{gamma*})
    ker R_gamma >= S_gamma := max(a_gamma, d_gamma - d_{gamma+delta} + a_{gamma*})        (rigorous lower bound)
Compare with the true ker_gamma (taskC1_m{m}.pkl).
usage: python taskD2_ann.py m [--cols]   -> taskD2_m{m}[_cols].pkl"""
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

from cengine import Split, Blocks, rmul_word, MAXOUT, workbufs
from taskC3_lemmas import lie_closure
from gf2 import gf2_rank_inplace

GENSETS = [
    ['y1.0', 'y1.1', 'y2.0'], ['y1.0', 'y1.1', 'y3.0+y3.1'], ['y1.0', 'y1.2', 'y2.2'], ['y1.0', 'y1.2', 'y3.1'],
    ['y1.0', 'y2.1', 'y3.0+y3.1'], ['y1.1', 'y1.2', 'y2.1'], ['y1.1', 'y1.2', 'y3.0'], ['y1.1', 'y2.2', 'y3.0'],
    ['y1.2', 'y2.0', 'y3.1'], ['y1.0', 'y1.1', 'y3.0', 'y3.1'], ['y1.0', 'y1.2', 'y3.0', 'y3.0+y3.1'],
    ['y1.0', 'y2.1', 'y2.2', 'y3.0'], ['y1.0', 'y2.1', 'y3.0', 'y3.1'], ['y1.1', 'y1.2', 'y3.1', 'y3.0+y3.1'],
    ['y1.1', 'y2.0', 'y2.2', 'y3.1'], ['y1.1', 'y2.2', 'y3.1', 'y3.0+y3.1'], ['y1.2', 'y2.0', 'y2.1', 'y3.0+y3.1'],
    ['y1.2', 'y2.0', 'y3.0', 'y3.0+y3.1']]
COLS = [['y1.0', 'y2.1', 'y3.0+y3.1'], ['y1.1', 'y2.2', 'y3.0'], ['y1.2', 'y2.0', 'y3.1']]


def parse(sp, nm):
    return [sp.pos[(int(t[1]), int(t[3]))] for t in nm.split("+")]


def subalg(sp, gs):
    K = lie_closure(sp, [parse(sp, g) for g in gs])
    return [sorted(v) for v in K]


def omega_words(K):
    """omega_K = prod of the basis vectors z (each a sum of letters) -> list of words"""
    exp = [[]]
    for z in K:
        exp = [e + [x] for e in exp for x in z]
    return exp


def contains(sp, K1, K2):
    """span K2 subset of span K1 ?"""
    basis = {}
    for v in K1:
        v = set(v)
        for p, b in basis.items():
            if p in v:
                v ^= b
        if v:
            basis[min(v)] = v
    for v in K2:
        v = set(v)
        changed = True
        while changed:
            changed = False
            for p, b in basis.items():
                if p in v:
                    v ^= b
                    changed = True
        if v:
            return False
    return True


def ann_rows(sp, B, K):
    """dict block -> list of PBW-element rows (each a list of monomials) spanning Ann(K) in that block"""
    pivots = set(min(z) for z in K)
    comp = [i for i in range(sp.nl) if i not in pivots]
    words = omega_words(K)
    wK = tuple(int(x) for x in sum(sp.lwt[z[0]] for z in K))
    out = defaultdict(list)
    # enumerate complement monomials by block: all masks of complement letters
    compmask = 0
    for i in comp:
        compmask |= 1 << i
    bufs = workbufs()
    obuf = np.empty(MAXOUT, dtype=np.int64)
    for g, arr in B.blocks.items():
        tg = (g[0] + wK[0], g[1] + wK[1], g[2] + wK[2])
        if tg not in B.blocks:
            continue
        sel = arr[(arr & ~compmask) == 0]
        for w in sel:
            n = 0
            for wd in words:
                n = rmul_word(np.int64(w), np.array(wd, dtype=np.int64), sp.br, sp.sq, obuf, n, *bufs)
                assert n >= 0
            vals, cnt = np.unique(obuf[:n], return_counts=True)
            el = vals[cnt & 1 == 1]
            assert el.size > 0
            out[tg].append(el)
    return out


if __name__ == "__main__":
    m = int(sys.argv[1])
    cols = "--cols" in sys.argv
    t0 = time.time()
    sp = Split(m)
    B = Blocks(sp)
    with open(f"taskC1_m{m}.pkl", "rb") as fh:
        C = pickle.load(fh)
    Ks = []
    for gs in (COLS if cols else GENSETS):
        K = subalg(sp, gs)
        Ks.append((gs, K))
    # drop K containing another K' (then Ann(K) subset Ann(K'))
    keep = []
    for i, (gs, K) in enumerate(Ks):
        if any(j != i and len(K2) < len(K) and contains(sp, K, K2) for j, (gs2, K2) in enumerate(Ks)):
            continue
        keep.append((gs, K))
    print(f"m={m}: {len(keep)} subalgebras kept: {[(gs, len(K)) for gs, K in keep]}", flush=True)
    rows = defaultdict(list)
    for gs, K in keep:
        r = ann_rows(sp, B, K)
        for g, L in r.items():
            rows[g].extend(L)
        print(f"   {gs}: dim K={len(K)}, rows {sum(len(v) for v in r.values())} [{time.time() - t0:.0f}s]", flush=True)
    a = {}
    for g, L in rows.items():
        d = B.dim(g)
        W = (d + 63) // 64
        P = np.zeros((len(L), W), dtype=np.uint64)
        for i, el in enumerate(L):
            loc = B.local[el]
            assert (B.key[el] == B.gkey(g)).all()
            np.bitwise_xor.at(P[i], loc >> 6, np.left_shift(np.uint64(1), (loc & 63).astype(np.uint64)))
        a[g] = int(gf2_rank_inplace(P, d))
    Gam = tuple(int(x) for x in sum(sp.lwt[i] for i in range(sp.nl)))
    dims = C["dims"]
    ker = C["ker"]
    res = {"m": m, "a": a, "Gamma": Gam}
    tot = defaultdict(int)
    perblock = {}
    for g in dims:
        h = (g[0] + 1, g[1] + 1, g[2] + 1)
        d, dh = dims[g], dims.get(h, 0)
        gs = (Gam[0] - h[0], Gam[1] - h[1], Gam[2] - h[2])
        gstar = (gs[1], gs[0], gs[2])
        ag, ast = a.get(g, 0), a.get(gstar, 0)
        if dh == 0:
            S = d
        else:
            S = max(ag, d - dh + ast)
        base = max(0, d - dh)
        k = ker[g]
        assert k >= S, (g, k, S, ag, ast, d, dh)
        assert k >= ag
        perblock[g] = (d, dh, base, ag, ast, S, k)
        tot["ker"] += k
        tot["base"] += base
        tot["S"] += S
        tot["a"] += ag
        tot["unexpl"] += k - S
        tot["blocks_unexpl"] += (k > S)
        tot["blocks_excess"] += (k > base)
    res["perblock"] = perblock
    N = 1 << sp.nl
    print(f"m={m}: |G|={N} ker={tot['ker']} B^Q={tot['base']} excess={tot['ker'] - tot['base']} ({(tot['ker'] - tot['base']) / N:.5f});"
          f" sum a={tot['a']} ({tot['a'] / N:.5f}); structural bound S={tot['S']} (S-B^Q = {tot['S'] - tot['base']},"
          f" {(tot['S'] - tot['base']) / N:.5f}); unexplained ker-S = {tot['unexpl']} ({tot['unexpl'] / N:.5f});"
          f" blocks with excess {tot['blocks_excess']}, blocks with ker > S: {tot['blocks_unexpl']}  [{time.time() - t0:.0f}s]",
          flush=True)
    # by principal degree
    top = sp.top
    byn = defaultdict(lambda: [0, 0, 0, 0])
    for g, (d, dh, base, ag, ast, S, k) in perblock.items():
        n = sum(g)
        byn[n][0] += k - base
        byn[n][1] += S - base
        byn[n][2] += k - S
        byn[n][3] += ag
    print("   by degree n: (excess, S-B^Q, unexplained, a)")
    for n in sorted(byn):
        if any(byn[n][:3]):
            print(f"     n={n}: {byn[n]}")
    with open(f"taskD2_m{m}{'_cols' if cols else ''}.pkl", "wb") as fh:
        pickle.dump(res, fh)
