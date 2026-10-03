r"""Task E: the connecting map on the column integrals.

For m with m in H_{m+1} (3 !| m or m = 3*2^r): at level m+1, omega_j-hat * sigma' = c^{(j)}_m * zeta_j (THEORY §6.4).
Checks:
 (Z1) omega_0-hat * sigma' = c_m * zeta_0 at level m+1 for the ascending-product lift of omega_0; compared with the
      closed form for the lift X*T^N (proved in THEORY Prop. 6.5, checked in taskE_zeta2.py)
        m = 1 mod 3: zeta_0 = X T^{N - (m-1)/3} (y1.1 y1.2 + y2.2);   m = 2 mod 3: zeta_0 = X T^{N-(m-2)/3} y1.2;
        m = 3*2^r:   zeta_0 = X
      (the two lifts differ by an element of J, so the zeta's may differ by im R^{(m)}; delta is lift-independent)
      where X = prod of the real column letters c_k (k < m, 3 !| k), T^a = prod of c_{3*2^r} over the binary digits of a,
      N = 2^R - 1, R = #{r : 3*2^r < m}.
 (Z2) zeta_0 * Theta'(omega_{j'}) != 0 at level m for some j'  (=> zeta_0 not in im R^{(m)} => delta(omega_0) != 0).
usage: python taskE_zeta.py mmax
"""
import sys

from cengine import Split
from taskC3_lemmas import elem_add, words_to_elem, letter


def Hset(m):
    return sorted(set([k for k in range(1, m) if k % 3] + [3 * 2 ** r for r in range(12) if 3 * 2 ** r < m]))


def col(sp, j, k):
    return letter(sp, k, (k - 1 + j) % 3)


def theta_col(sp, j, k):
    return letter(sp, k, (1 - j) % 3)


def mul(sp, X, Y):
    return sp.mul(X, Y)


def strip_letter(sp, el, pos_list):
    """el = c * zeta with c a (sum of) degree-m letter(s); return dict letterpos -> zeta-part"""
    out = {}
    for t in el:
        hits = [p for p in pos_list if (t >> p) & 1]
        assert len(hits) == 1, "term without / with two c_m letters"
        p = hits[0]
        out.setdefault(p, []).append(t ^ (1 << p))
    return {p: sorted(v) for p, v in out.items()}


def main(mmax):
    for m in range(3, mmax + 1):
        inH = (m % 3 != 0) or (m in [3 * 2 ** r for r in range(12)])
        spB = Split(m + 1)
        sp = Split(m)
        sig = words_to_elem(spB, [list(w) for w in spB.sigma_words()])
        H = Hset(m)
        res = []
        for j in range(3):
            om = words_to_elem(spB, [[col(spB, j, k) for k in H]])
            prod = mul(spB, om, sig)
            if not inH:
                res.append(f"j={j}: omega-hat*sigma' = {'0' if not prod else str(len(prod)) + ' terms'}")
                continue
            cm = [spB.pos[(m, p)] for p in range(3 if m % 3 else 2)]
            parts = strip_letter(spB, prod, cm)
            # closed form (j = 0 only, the others by rotation)
            cf = ""
            if j == 0 and m % 3:
                R = len([r for r in range(12) if 3 * 2 ** r < m])
                N = 2 ** R - 1
                i0 = (m - 1) // 3 if m % 3 == 1 else (m - 2) // 3
                a = N - i0
                T = [3 * 2 ** r for r in range(R) if (a >> r) & 1]
                Xl = [col(spB, 0, k) for k in range(1, m) if k % 3]
                tail = [[letter(spB, 1, 1), letter(spB, 1, 2)], [letter(spB, 2, 2)]] if m % 3 == 1 else [[letter(spB, 1, 2)]]
                ws = [Xl + [col(spB, 0, t) for t in T] + tl for tl in tail]
                zcf = words_to_elem(spB, ws)
                (p, z), = parts.items()
                cf = f" (ascending-product lift; equals the X*T^N closed form of taskE_zeta2: {z == zcf} -- lifts differ by J, delta is lift-independent)"
            # Z2: transport zeta to level m (same letters, all < m) and multiply by Theta'(omega_{j'})
            z2 = []
            for p, z in parts.items():
                zm = elem_add(*[sp.rmul(0, [sp.pos[spB.letters[i]] for i in range(spB.nl) if (t >> i) & 1]) for t in z])
                nz = []
                for jj in range(3):
                    tom = words_to_elem(sp, [[theta_col(sp, jj, k) for k in reversed(H)]])
                    nz.append(len(mul(sp, zm, tom)))
                z2.append((spB.mstr(1 << p), len(z), nz))
            res.append(f"j={j}: omega-hat*sigma' = sum_c c*zeta, (c, #terms zeta, |zeta*Theta'(omega_j')| j'=0,1,2) = {z2}{cf}")
        print(f"m={m} (m in H_(m+1): {inH}), d(m)={sum(H)}:", flush=True)
        for r in res:
            print("   " + r, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))

