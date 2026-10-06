"""d = 2, p odd: verify the two explicit kernel families of R_sigma on u_m.
 (A) column:  omega_H = prod x_k^{p-1} (k odd < m) * prod h_{2p^r}^{p-1}        degree odo(m)
 (B) t^2-subalgebra K = span{x_k (k=1 mod 4), f_k (k=3 mod 4), h_k (k=0 mod 4)}:
     omega_K * f_1^{(p-1)/2}                                                       degree kap(m)
and check that the tail exponent (p-1)/2 is the only one that works.  usage: python kmech2.py p mmax"""
import sys
from fam import Fam


def mono_of(F, lets):
    """lets: dict (k,i) -> exponent ; returns PBW monomial tuple (letters in PBW order)"""
    l = [0] * F.nl
    for (k, i), a in lets.items():
        l[F.pos[(k, i)]] = a
    return tuple(l)


def is_zero_after_sigma(F, elem):
    out = F.mul(elem, F.elem(F.sigma(1)))
    return len(out) == 0


def main():
    p, M = int(sys.argv[1]), int(sys.argv[2])
    for m in range(3, M + 1):
        F = Fam(p, 2, m)
        H = {(k, 0): p - 1 for k in range(1, m, 2)}
        r = 0
        while 2 * p ** r < m:
            H[(2 * p ** r, 0)] = p - 1
            r += 1
        K = {}
        for k in range(1, m):
            if k % 4 == 1:
                K[(k, 0)] = p - 1
            elif k % 4 == 3:
                K[(k, 1)] = p - 1
            elif k % 4 == 0:
                K[(k, 0)] = p - 1
        odo = sum(k * a for (k, i), a in H.items())
        kapbase = sum(k * a for (k, i), a in K.items())
        wH = {mono_of(F, H): 1}
        okA = is_zero_after_sigma(F, wH)
        tails = []
        wK = {mono_of(F, K): 1}
        for j in range(0, p):
            z = F.mulword(F.one(), [F.pos[(1, 1)]] * j)
            el = F.mul(wK, z)
            if el and is_zero_after_sigma(F, el):
                tails.append(j)
        print(f"p={p} m={m}: (A) omega_H sigma = 0: {okA}, deg {odo} | (B) tails j with omega_K f1^j in ker: {tails}"
              f" (predicted [{(p-1)//2}]), deg {kapbase}+j | bound min = {min(odo, kapbase + (p-1)//2)}", flush=True)


if __name__ == "__main__":
    main()
