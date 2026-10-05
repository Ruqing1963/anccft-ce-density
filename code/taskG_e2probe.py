"""Task G probe: E_1 / rank(d_1) per layer (Lemma 6.6 bound only, no exact lifting) for M from a, via taskE_lift.main.
usage: python taskG_e2probe.py M a nmin nmax"""
import sys
import taskE_lift

taskE_lift.EXACT = False
M, a, nmin, nmax = (int(x) for x in sys.argv[1:5])
taskE_lift.main(M, a, nmin, nmax, f"taskC1_m{a}.pkl")
