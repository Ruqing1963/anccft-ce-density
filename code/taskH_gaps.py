r"""Task H (Cor. 8.2(c)): C_a cannot hold on long intervals of levels -- unconditional.

Assume C_a holds at every level in [M, M+L].  Then (Cor. 7.4, monotonicity, Thm 7.1, known k_m for m <= 14) the
lower bounds lb[m] below are valid; Theorem 8.1 says C_a at m forces k_{m-a} <= T_a(m) - 3.  The first m in the interval
with lb[m-a] > T_a(m) - 3 is a contradiction, so C_a fails somewhere in [M, m].  We print the worst interval length
Lmax_a = max over M of (m - M) for M in a range.
usage: python taskH_gaps.py Mmax
"""
import sys

from taskH_nogo import T, KNOWN


def base(m):
    return KNOWN.get(m, max(m - 3, 0)) if m < 15 else max(82, m - 3)


def L_of(M, a, cap=100000):
    lb = {j: base(j) for j in range(1, M)}
    m = M
    while True:
        lb[m] = max(lb[m - 1], base(m), lb[m - a] + (m - a))      # C_a at m assumed
        if lb[m - a] > T(a, m) - 3:
            return m - M
        m += 1
        if m - M > cap:
            return None


if __name__ == "__main__":
    Mmax = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    for a in (1, 2, 3, 4, 5, 6):
        Ls = [(L_of(M, a), M) for M in range(a + 2, Mmax + 1)]
        worst = max(Ls)
        tail = max(L for L, M in Ls if M >= Mmax // 2)
        print(f"a={a}: C_a fails somewhere in every interval [M, M+L], L <= {worst[0]} (worst M={worst[1]}), "
              f"for M in [{Mmax // 2},{Mmax}]: L <= {tail}", flush=True)
