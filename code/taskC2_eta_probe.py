import sys
sys.path.insert(0, r"C:\Users\LAPPIE\Desktop\KMS\Find New Math\ce_compute")
import numpy as np
from cengine import Split
from taskC2_kernel import kernel_block

m = 9
sp = Split(m)
rows, K = kernel_block(sp, (12, 9, 14))
eta = [int(rows[i]) for i in np.nonzero(K[0])[0]]
print("eta:", " + ".join(sp.mstr(x) for x in eta))


def times(el, word):
    acc = {}
    for x in el:
        for t in sp.rmul(x, word):
            acc[t] = acc.get(t, 0) ^ 1
    return sorted(t for t, c in acc.items() if c)


def lt(k, p):
    if k % 3 == 0 and p == 2:
        return None
    return sp.pos[(k, p)]


for k in range(1, m):
    for p in range(3):
        if k % 3 == 0 and p == 2:
            # e_2 = e_0 + e_1
            r = times(eta, [sp.pos[(k, 0)]])
            r2 = times(eta, [sp.pos[(k, 1)]])
            s = set(r) ^ set(r2)
            print(f"eta * e_2 phi^{k}: {len(s)} terms")
            continue
        r = times(eta, [sp.pos[(k, p)]])
        print(f"eta * e_{p} phi^{k} (wt {sp.lwt[sp.pos[(k, p)]]}): {len(r)} terms" + ("   <-- 0" if not r else ""))
# sigma' pieces for C^(2): x_k = e_{k+1} phi^k
