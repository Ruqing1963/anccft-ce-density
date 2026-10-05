r"""Task H: numerics for the no-go theorem for the C_a route (THEORY 8.1-8.3).

T_a(m) := top degree of u(K_a) = sum_{m-a <= k < m} k d_k.
Theorem 8.1: C_a holds at m  ==>  k_{m-a} <= T_a(m) - 3.
(1) table of T_a(m) against the known k_{m-a} (exact for m-a <= 14) and against d(m-a) (conjectural value),
    listing for each a the levels m where C_a is PROVABLY false (known k) and where it is false under k = d(m).
(2) smallest a = a*(m) with T_a(m) >= d(m-a) + 3 (the smallest a for which C_a is not excluded, if k = d).
(3) the ceiling of the iterated route: G(m) = max(base(m), max_a [(m-a) + min(G(m-a), T_a(m) - 3)]),
    base(m) = known k_m (m <= 14) or m - 3; G(m) bounds every lower bound obtainable from Cor. 7.4 chains.
usage: python taskH_nogo.py [Mmax]
"""
import sys
import math


def dk(k):
    return 2 if k % 3 == 0 else 3


def H(m):
    s = {k for k in range(1, m) if k % 3}
    r = 0
    while 3 * 2 ** r < m:
        s.add(3 * 2 ** r)
        r += 1
    return s


def d(m):
    return sum(H(m))


_P = [0, 0]
for _k in range(1, 20001):
    _P.append(_P[-1] + _k * dk(_k))          # _P[m] = sum_{k<m} k d_k


def T(a, m):
    return _P[m] - _P[m - a]


KNOWN = {2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21, 8: 28, 9: 35, 10: 36, 11: 46, 12: 57, 13: 69, 14: 82}


def kproved(m):
    """proved lower bound for k_m: exact for m <= 14, monotone + Thm 7.1 beyond"""
    if m in KNOWN:
        return KNOWN[m]
    return max(82, m - 3)


def kconj(m):
    """refined conjecture THEORY 6.8"""
    if m in KNOWN:
        return KNOWN[m]
    if m % 3 == 0 and m >= 9 and (m // 3) & (m // 3 - 1) != 0:
        return d(m) - 1
    return d(m)


def main(Mmax):
    print("(1) levels m <= 40 where C_a is excluded by Theorem 8.1  [T_a(m) <= k_{m-a} + 2]")
    for a in range(1, 7):
        prov = [m for m in range(a + 2, 41) if T(a, m) <= kproved(m - a) + 2]
        conj = [m for m in range(a + 2, 41) if T(a, m) <= kconj(m - a) + 2]
        print(f"  a={a}: excluded with proved lower bounds for k: {prov}")
        print(f"        excluded if k = conjectured value: first m = {conj[0] if conj else None}, all m in [.,40]: {conj}")
    print("\n  first m with C_a excluded under the conjecture, a = 1..12:")
    for a in range(1, 13):
        m = a + 2
        while T(a, m) > kconj(m - a) + 2:
            m += 1
        print(f"    a={a}: m={m}  (T_a={T(a, m)}, k_(m-a)={kconj(m - a)})")
    print("\n(2) a*(m) = least a with T_a(m) >= k_conj(m-a) + 3   (C_a not excluded), and a*/m")
    for m in [10, 15, 20, 30, 40, 60, 100, 200, 400, 1000]:
        a = 1
        while T(a, m) < kconj(m - a) + 3:
            a += 1
        print(f"  m={m}: a*={a}, a*/m={a / m:.3f}")
    print("\n(3) ceiling of the C_a route: G(m) and G(m)/m^1.5")
    G = {}
    for m in range(2, Mmax + 1):
        best = kproved(m) if m <= 14 else m - 3
        for a in range(1, m - 1):
            best = max(best, (m - a) + min(G[m - a], T(a, m) - 3))
        G[m] = best
    for m in [10, 20, 50, 100, 200, 500, 1000, 2000, 4000]:
        if m <= Mmax:
            print(f"  m={m}: G={G[m]}, G/m^1.5={G[m] / m ** 1.5:.3f}, d(m)={d(m)}, G/d={G[m] / d(m):.3f}")
    print("\n(4) under the refined conjecture (k = kconj >= d - 1): levels m in [6, 10000] where C_3 is NOT excluded:")
    ok = [m for m in range(6, 10001) if T(3, m) > kconj(m - 3) + 2]
    print(f"  {ok}")
    ok2 = [m for m in range(6, 10001) if T(3, m) > d(m - 3) + 1]
    print(f"  same with the weaker k_(m-3) >= d(m-3) - 1 for all m: {ok2}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
