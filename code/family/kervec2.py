"""d = 2: explicit kernel vectors of R_sigma on u_m in degree n.  usage: python kervec2.py p m n
Letters printed as x_k (y_{k,0}, k odd), f_k (y_{k,1}), h_k (y_{k,0}, k even)."""
import sys
import numpy as np
from fam import Fam
from uinj import nullspace_mod_p


def name(F, mono):
    s = []
    for t, a in enumerate(mono):
        if a:
            k, i = F.letters[t]
            nm = ("h" if k % 2 == 0 else ("x" if i == 0 else "f")) + str(k)
            s.append(nm + (f"^{a}" if a > 1 else ""))
    return "*".join(s) if s else "1"


def main(p, m, n):
    F = Fam(p, 2, m)
    src = F.monomials_by_block([n])
    tgt = F.monomials_by_block([n + 2])
    sig = F.elem(F.sigma(1))
    print(f"(p,d)=({p},2) m={m} n={n}; sigma' = " + " + ".join(f"{v}*{name(F, k)}" for k, v in sig.items()))
    for g, rows in sorted(src.items()):
        cols = tgt.get((g[0] + 1, g[1] + 1), [])
        ci = {mo: j for j, mo in enumerate(cols)}
        M = np.zeros((len(rows), len(cols)), dtype=np.int64)
        for r, mo in enumerate(rows):
            for mo2, v in F.mul({mo: 1}, sig).items():
                M[r, ci[mo2]] = v
        N = nullspace_mod_p(M, p) if len(cols) else np.eye(len(rows), dtype=np.int64)
        if len(N):
            print(f" block {g} dim {len(rows)} -> {len(cols)}: kernel dim {len(N)}")
            for v in N:
                terms = [f"{int(c)}*{name(F, rows[i])}" for i, c in enumerate(v) if c]
                print("   " + " + ".join(terms[:30]) + (f" ... ({len(terms)} terms)" if len(terms) > 30 else ""))


if __name__ == "__main__":
    main(*map(int, sys.argv[1:4]))
