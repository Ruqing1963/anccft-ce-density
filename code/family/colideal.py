"""Is sigma' in the right ideal  c_1 U + ... + c_d U  (column j)?  And which weight-delta elements are?
c^{(j)}_k = e_{k-1+j} phi^k  (for d | k reduced mod T).  Work at level m = d+1 (degree d computations are exact)."""
import sys
import itertools
import numpy as np
from fam import Fam, _acc, rank_mod_p


def col_letter(F, k, j):
    """c^{(j)}_k as dict (element of U)"""
    v = [0] * F.d
    v[(k - 1 + j) % F.d] = 1
    terms = F._vec(k, v)
    res = {}
    for pos, cf in terms:
        l = [0] * F.nl
        l[pos] = 1
        _acc(res, {tuple(l): 1}, cf, F.p)
    return res


def span_rows(F, elems, basis):
    idx = {b: i for i, b in enumerate(basis)}
    M = np.zeros((len(elems), len(basis)), dtype=np.int64)
    for r, e in enumerate(elems):
        for mo, v in e.items():
            M[r, idx[mo]] = v
    return M


def main(p, d):
    F = Fam(p, d, d + 1)
    delta = tuple([1] * d)
    blocks = F.monomials_by_block([d])
    basis = blocks[delta]
    sig = {o: F.elem(F.sigma(o)) for o in (1, -1)}
    print(f"(p,d)=({p},{d}): dim U_delta = {len(basis)}")
    for o in (1, -1):
        print(f"  sigma({o:+d}) = " + " + ".join(f"{v}*{F.mstr(k)}" for k, v in sorted(sig[o].items())))
    allmono = F.monomials_by_block(list(range(0, d)))
    for kind, j in [("column", j) for j in range(d)] + [("row", j) for j in range(d)]:
        gens = []
        for k in range(1, d + 1):
            idx = (k - 1 + j) % d if kind == "column" else j % d
            v = [0] * d
            v[idx] = 1
            c = {}
            for pos, cf in F._vec(k, v):
                l = [0] * F.nl
                l[pos] = 1
                _acc(c, {tuple(l): 1}, cf, F.p)
            if not c:
                continue
            wk = F._wt(k, idx)
            need = tuple(a - b for a, b in zip(delta, wk))
            for mo in allmono.get(need, []):
                gens.append(F.mul(c, {mo: 1}))
        G = span_rows(F, gens, basis)
        rG = rank_mod_p(G, p)
        out = []
        for o in (1, -1):
            S = span_rows(F, gens + [sig[o]], basis)
            out.append("in" if rank_mod_p(S, p) == rG else "NOT in")
        print(f"  {kind} {j}: dim(ideal cap U_delta) = {rG};  sigma(+1) {out[0]},  sigma(-1) {out[1]}")
    # intersection over all columns: elements of U_delta lying in every column ideal
    # (computed as the null space approach: dim of intersection via ranks)
    return


if __name__ == "__main__":
    for a in sys.argv[1:]:
        p, d = map(int, a.split(","))
        main(p, d)
