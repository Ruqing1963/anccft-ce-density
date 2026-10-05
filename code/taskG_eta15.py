"""Task G: structure of the level-15 kernel element eta_15 (block (36,32,27), n = 95, from taskG_blocks_M15.pkl).
Checks: (1) eta_15 mod L_14 (drop monomials containing a degree-14 letter) equals y13.p * omega^{(14)}_0 for some p;
(2) eta_15 * sigma' = 0 at level 15 (re-check with the plain engine);  (3) eta_15 is not annihilated by single letters."""
import pickle

import numpy as np

from cengine import Split
from taskG_ansatz import Hset

res = pickle.load(open("taskG_blocks_M15.pkl", "rb"))
eta = res[(36, 32, 27)][4][0]
sp15 = Split(15)
sp14 = Split(14)
assert sp15.letters[3:] == sp14.letters
print("eta_15:", len(eta), "terms")
low = sorted(x >> 3 for x in eta if not (x & 7))
print("terms without degree-14 letter:", len(low))


def esum(sp, k, p):
    """element e_p phi^k as a list of monomials (e_2 phi^{3i} = y.0 + y.1)"""
    if k % 3 == 0 and p == 2:
        return [1 << sp.pos[(k, 0)], 1 << sp.pos[(k, 1)]]
    return [1 << sp.pos[(k, p)]]


def omega(sp, m, j):
    X = [0]
    for k in sorted(Hset(m), reverse=True):
        X = sp.mul(X, esum(sp, k, (k - 1 + j) % 3))
    return X


om = omega(sp14, 14, 0)
print("omega^(14)_0:", len(om), "terms")
for p in range(3):
    prod = sp14.mul(esum(sp14, 13, p), om)
    print(f"  y13.{p} * omega^(14)_0: {len(prod)} terms; equal to eta mod L_14: {sorted(prod) == low}")
# re-check eta * sigma' = 0 with the plain engine
acc = {}
for w in sp15.sigma_words():
    for x in eta:
        for t in sp15.rmul(x, list(w)):
            acc[t] = acc.get(t, 0) ^ 1
print("eta*sigma' = 0:", not any(acc.values()))
nz = []
for i in range(sp15.nl):
    acc = {}
    for x in eta:
        for t in sp15.rmul(x, [i]):
            acc[t] = acc.get(t, 0) ^ 1
    nz.append(any(acc.values()))
print("eta * letter != 0 for all letters:", all(nz), f"({sum(nz)}/{sp15.nl})")
