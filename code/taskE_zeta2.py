r"""Task E: machine check of the closed form of the connecting map on omega_0 (THEORY §6.5, Prop. 6.6):
at level m+1 (m in H_{m+1}), with the lift  omega-hat = X * T^N  (X = prod of the real column letters c_k, k < m, 3 !| k,
in ascending order; T^N = prod of the tower letters c_{3*2^r} < m), the product omega-hat * sigma' equals
    c_m * X * T^{N - floor(m/3)} * A_m,   A_m = y1.1 y1.2 + y2.2 (m = 1 mod 3), y1.2 (m = 2 mod 3), 1 (m = 3*2^r),
where T^a := prod of c_{3*2^r} over the binary digits r of a.   usage: python taskE_zeta2.py mmax"""
import sys

from cengine import Split
from taskC3_lemmas import words_to_elem, letter


def col(sp, k):
    return letter(sp, k, (k - 1) % 3)


def T(sp, a):
    out = []
    r = 0
    while a >> r:
        if (a >> r) & 1:
            out.append(col(sp, 3 * 2 ** r))
        r += 1
    return out


if __name__ == "__main__":
    for m in range(3, int(sys.argv[1]) + 1):
        if m % 3 == 0 and m not in [3 * 2 ** r for r in range(10)]:
            continue
        sp = Split(m + 1)
        sig = words_to_elem(sp, [list(w) for w in sp.sigma_words()])
        R = len([r for r in range(10) if 3 * 2 ** r < m])
        N = 2 ** R - 1
        X = [col(sp, k) for k in range(1, m) if k % 3]
        lift = words_to_elem(sp, [X + T(sp, N)])
        lhs = sp.mul(lift, sig)
        if m % 3 == 1:
            tails = [[letter(sp, 1, 1), letter(sp, 1, 2)], [letter(sp, 2, 2)]]
        elif m % 3 == 2:
            tails = [[letter(sp, 1, 2)]]
        else:
            tails = [[]]
        a = N - m // 3 if m % 3 else 0
        rhs = words_to_elem(sp, [[col(sp, m)] + X + T(sp, a) + t for t in tails])
        print(f"m={m}: N={N}, a={a}: omega-hat*sigma' == c_m X T^a A_m : {lhs == rhs}  ({len(lhs)} terms)", flush=True)
        assert lhs == rhs
