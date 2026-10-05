r"""Task F: symbolic structure of Psi_r(sigma') over F2[t_1..t_r]:  N := Psi_r(sigma') - (t_1+...+t_r) I.
Prints N's nonzero entries pattern, nilpotency index of N, and whether N is t-free.
usage: python taskF_symb.py r
"""
import sys
import sympy as sp


def build(r):
    ts = sp.symbols(" ".join(f"t{s+1}" for s in range(r)) + " dummy")[:r]
    n = 3 ** r
    E = [sp.zeros(n, n) for _ in range(3)]
    for s in range(r):
        R = [[(2, 0, ts[s])], [(0, 1, 1)], [(1, 2, 1)]]
        stride = 3 ** (r - 1 - s)
        for idx in range(n):
            d = (idx // stride) % 3
            for i in range(3):
                for (row, col, val) in R[i]:
                    if col == d:
                        E[i][idx + (row - d) * stride, idx] += val
    M = E[0] * E[1] * E[2] + E[1] * E[2] * E[0] + E[2] * E[0] * E[1]
    red = lambda x: sp.Poly(x, *ts, modulus=2).as_expr() if x != 0 else 0
    M = M.applyfunc(red)
    return ts, M, red


if __name__ == "__main__":
    r = int(sys.argv[1])
    ts, M, red = build(r)
    n = 3 ** r
    S = sum(ts)
    N = (M - S * sp.eye(n)).applyfunc(red)
    nz = [(i, j, N[i, j]) for i in range(n) for j in range(n) if N[i, j] != 0]
    print(f"r={r}: N has {len(nz)} nonzero entries; distinct values {sorted(set(str(v) for _, _, v in nz))}")
    P = N
    k = 1
    nil = True
    while any(x != 0 for x in P):
        P = (P * N).applyfunc(red)
        k += 1
        if k > 3 * r + 3:
            nil = False
            break
    print(f"N nilpotent with N^{k} = 0" if nil else f"N^k != 0 for k <= {k - 1} (N is not nilpotent of small index; "
          f"for r = 3 it contains 3-cycles of weight t1 t2 t3)")
    # commutation with S I trivial; check N*M == M*N
    print("first entries:", nz[:12])
