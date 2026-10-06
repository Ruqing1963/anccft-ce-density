"""Validate fam.py at (p,d) = (2,3) against the paper: sigma' expansion, total kernel dims, thresholds k_m."""
import time
from fam import Fam, right_kernel_dims

KNOWN_COKER = {2: 7, 3: 37, 4: 100, 5: 547, 6: 3182, 7: 10226}
KNOWN_K = {2: 1, 3: 3, 4: 6, 5: 10, 6: 15, 7: 21}

F = Fam(2, 3, 6)
cyc = F.elem(F.sigma(1))
old = F.elem([([F.y(1, 0), F.y(1, 1), F.y(1, 2)], 1), ([F.y(1, 2), F.y(2, 1)], 1), ([F.y(1, 0), F.y(2, 2)], 1),
              ([F.y(3, 1)], 1)])
print("cyclic sum (+):", {F.mstr(k): v for k, v in cyc.items()})
print("paper sigma'  :", {F.mstr(k): v for k, v in old.items()})
print("equal:", cyc == old)
cm = F.elem(F.sigma(-1))
print("cyclic sum (-):", {F.mstr(k): v for k, v in cm.items()})

for m in range(2, 8):
    t0 = time.time()
    F = Fam(2, 3, m)
    res = right_kernel_dims(F, F.sigma(1))
    tot = sum(v[1] for v in res.values())
    km = min((n for n, v in res.items() if v[1]), default=None)
    print(f"m={m}: dim u={2**F.nl}, total ker={tot} (paper {KNOWN_COKER[m]}), k_m={km} (paper {KNOWN_K[m]}),"
          f" {time.time()-t0:.1f}s", flush=True)
