r"""Task F: print preimages w with w sigma' = omega_{K_a} (a = 3) for small m, and the space of all preimages
(w + ker R).  usage: python taskF_omegaK_show.py m a"""
import sys
import numpy as np
from cengine import Split, pack_words
from gf2null import left_kernel
from taskC2_low import blocks_of_degree, build_block_sorted

m, a = int(sys.argv[1]), int(sys.argv[2])
sp = Split(m)
words, wlen = pack_words(sp.sigma_words())
s = sum(1 for (k, p) in sp.letters if k >= m - a)
omega = (1 << s) - 1
g = sp.qdeg(omega)
n = sum(g)
gm = tuple(x - 1 for x in g)
rows = blocks_of_degree(sp, n - 3)[gm]
cols = blocks_of_degree(sp, n)[g]
P, bad = build_block_sorted(rows, words, wlen, sp.br, sp.sq, cols, 16)
j = int(np.searchsorted(cols, omega))
e = np.zeros((1, P.shape[1]), dtype=np.uint64)
e[0, j >> 6] = np.uint64(1) << np.uint64(j & 63)
K, rk = left_kernel(np.concatenate([P, e]), cols.size)
sols = []
kers = []
for v in K:
    bits = np.unpackbits(v.view(np.uint8), bitorder="little")[:rows.size + 1]
    vec = rows[np.nonzero(bits[:rows.size])[0]]
    (sols if bits[rows.size] else kers).append(vec)
print(f"m={m} a={a}: omega_K = {sp.mstr(omega)} (degree {n}, weight {g}); block {rows.size}; dim ker R there = {len(kers)}")
# choose the sparsest solution by greedy reduction with kernel vectors
best = min(sols, key=len)
print(f"one preimage w ({len(best)} terms):")
for mono in best:
    print("   ", sp.mstr(int(mono)))
# letters of degree >= m-a appearing in the terms
cnt = {}
for mono in best:
    hi = bin(int(mono) & omega).count("1")
    cnt[hi] = cnt.get(hi, 0) + 1
print("number of K-letters per term:", cnt)
