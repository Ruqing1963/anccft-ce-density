r"""Independent check (group algebra F2[G_10] + graded u(L) code, no JAlg/JTable) of the weight-9 obstruction:
for every v in V \ 0 and its Jennings lift v~, the weight-9 part r_9 of v~^2 is NOT in ad_v(u_5) (ad_v = L_v + R_v),
while ad_v(r_9) = 0 in u_13 (consistency).  Since (v~ + a)^2 = v~^2 + [v~, a] + a^2 with a^2 in I^10 for a in I^5,
this proves: no x in F2[[P]] with leading term in V \ 0 has x^2 in I^10."""
import time
from collections import Counter

from ulie import ULie
from task2_leading import Jennings, gmul
from verify_zd import lift
from zdtools import elem_from_bits
from b2common import VBITS
from jtable import JTable

t0 = time.time()
U = ULie("F8", Dl=13)
Vb = [elem_from_bits(U, 4, g) for g in VBITS]
Jn = Jennings(10, U)
T = JTable(16, verbose=False)
idx9 = U.index(9)
for c in range(1, 8):
    v = set()
    for i in range(3):
        if (c >> i) & 1:
            v ^= Vb[i]
    vt = lift(Jn, v)
    sq = gmul(Jn.R, vt, vt)
    co = Jn.coeffs(sq, 9)
    byw = Counter(U.mdeg(m) for m in co)
    assert set(byw) <= {9}, byw
    r9 = {m for m in co if U.mdeg(m) == 9}
    # same residual from the Jennings tables (A_16)?
    rT = T.elem(T.mul(T.vec(v), T.vec(v)))
    same = r9 == {m for m in rT if T.U.mdeg(m) == 9}
    # span of ad_v(u_5) in u_9 + membership test
    piv = {}

    def red(x):
        while x:
            h = x.bit_length() - 1
            if h in piv:
                x ^= piv[h]
            else:
                return x, h
        return 0, None
    rank = 0
    for a in U.monos(5):
        img = U.mul(v, {a}) ^ U.mul({a}, v)
        x = sum(1 << idx9[m] for m in img)
        x, h = red(x)
        if x:
            piv[h] = x
            rank += 1
    rv = sum(1 << idx9[m] for m in r9)
    rr, _ = red(rv)
    adr = U.mul(v, r9) ^ U.mul(r9, v)
    print(f"V element #{c}: |supp v~| = {len(vt)}, |supp v~^2| (G_10) = {len(sq)}, weight-9 part r_9 has {len(r9)} PBW terms "
          f"(equal to JTable value: {same}); rank ad_v(u_5) = {rank}; r_9 in ad_v(u_5)? {rr == 0}; "
          f"ad_v(r_9) = 0 in u_13? {len(adr) == 0}", flush=True)
print(f"[{time.time() - t0:.0f}s]")
