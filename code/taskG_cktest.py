"""Task G: test of the intra-block checkpoint/resume of lift_block2 (run, kill, rerun; compare with a clean run).
usage: python taskG_cktest.py M g0 g1 g2 ckdir ckevery"""
import pickle
import sys
import time

from taskG_core import LowLevel, lift_block2

M = int(sys.argv[1])
g = tuple(int(x) for x in sys.argv[2:5])
ckdir, every = sys.argv[5], float(sys.argv[6])
kerdims = pickle.load(open("taskC1_m10.pkl", "rb"))["ker"]
LL = LowLevel(M, 10)
t = time.time()
dim, info, cand = lift_block2(LL, g, kerdims, verbose=True, ckdir=ckdir, ckevery=every)
print("RESULT", g, dim, info, f"{time.time() - t:.1f}s", flush=True)
