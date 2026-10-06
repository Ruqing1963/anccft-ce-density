"""Validate fam_nb against fam.py results (exact totals of ker R_sigma on u_m)."""
import time
from fam_nb import FamNB, kernel_by_degree

KNOWN = [  # (p, d, m, orient, total ker)
    (2, 3, 2, 1, 7), (2, 3, 3, 1, 37), (2, 3, 4, 1, 100), (2, 3, 5, 1, 547), (2, 3, 6, 1, 3182), (2, 3, 7, 1, 10226),
    (3, 2, 2, 1, 5), (3, 2, 3, 1, 11), (3, 2, 4, 1, 47), (3, 2, 5, 1, 111), (3, 2, 6, 1, 659), (3, 2, 7, 1, 1661),
    (3, 2, 8, 1, 11223), (3, 2, 9, 1, 29751),
    (5, 2, 5, 1, 1347), (5, 2, 6, 1, 22511), (7, 2, 5, 1, 7261),
    (5, 3, 3, 1, 3315), (5, 3, 4, 1, 56407), (3, 3, 5, 1, 59145), (2, 4, 6, -1, 126296),
]
allok = True
for p, d, m, o, tot in KNOWN:
    t0 = time.time()
    E = FamNB(p, d, m)
    top = E.F.top
    res = kernel_by_degree(E, 0, top, orient=o)
    got = sum(v[1] for v in res.values())
    dims = sum(v[0] for v in res.values())
    resL = kernel_by_degree(E, 0, top, orient=o, left=True) if E.p ** E.nl <= 200000 else None
    gotL = sum(v[1] for v in resL.values()) if resL else None
    ok = (got == tot and dims == p ** E.nl and (gotL is None or gotL == tot))
    allok &= ok
    print(f"({p},{d}) m={m} o={o:+d}: ker R={got} (fam.py {tot}), ker L={gotL}, dim={dims}/{p**E.nl} "
          f"{'OK' if ok else 'MISMATCH'}  {time.time()-t0:.1f}s", flush=True)
E = FamNB(5, 2, 7)
r = kernel_by_degree(E, 42, 44)
print("(5,2) m=7 n=42..44:", {n: r[n][1] for n in sorted(r)}, "(thresh.py: 0, 0, 2)")
print("ALL OK" if allok else "SOME MISMATCH")
