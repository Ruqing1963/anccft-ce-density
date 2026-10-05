"""Task G: estimate E_1 sizes for lifting level M from level a=10 (uses exact level-10 kernels taskC1_m10.pkl).
usage: python taskG_e1est.py M nmin nmax"""
import itertools
import pickle
import sys
from collections import defaultdict

from cengine import wt
from taskG_dims import block_dims

M, nmin, nmax = (int(x) for x in sys.argv[1:4])
a = int(sys.argv[4]) if len(sys.argv) > 4 else 10
D = pickle.load(open(f"taskC1_m{a}.pkl", "rb"))
ker = D["ker"]
dims = D["dims"]
Il = [(k, p) for k in range(M - 1, a - 1, -1) for p in (range(3) if k % 3 else range(2))]
subs = defaultdict(int)
for r in range(len(Il) + 1):
    for S in itertools.combinations(Il, r):
        w = [0, 0, 0]
        for l in S:
            x = wt(*l)
            for c in range(3):
                w[c] += x[c]
        subs[tuple(w)] += 1
bd = block_dims(M, nmax)
for n in range(nmin, nmax + 1):
    tot = 0
    mx = 0
    nb = 0
    e0 = 0
    for g, dg in bd.items():
        if sum(g) != n or g != min(g, (g[2], g[0], g[1]), (g[1], g[2], g[0])):
            continue
        e = 0
        for w, mult in subs.items():
            b = (g[0] - w[0], g[1] - w[1], g[2] - w[2])
            e += mult * ker.get(b, 0)
        e0 += ker.get(g, 0)
        tot += e
        mx = max(mx, e)
        nb += 1
    print(f"M={M} a={a} n={n}: reps={nb} E1 total={tot} (S=empty part {e0}) max block E1={mx}")
