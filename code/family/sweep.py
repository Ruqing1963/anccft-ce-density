"""Sweep over (p, d, m): kernel of R_sigma on u_m, thresholds vs the odometer prediction.
usage: python sweep.py p d m_max [orient]
writes out_sweep_p{p}_d{d}.txt (appends)"""
import sys
import time
import json
from fam import Fam, right_kernel_dims


def odometer(p, d, m):
    s = sum(k for k in range(1, m) if k % d)
    r = 0
    while d * p ** r < m:
        s += d * p ** r
        r += 1
    return (p - 1) * s


def main():
    p, d, M = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    orient = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    fn = f"out_sweep_p{p}_d{d}" + ("" if orient == 1 else "_minus") + ".txt"
    with open(fn, "a", encoding="utf-8") as fo:
        def say(s):
            print(s, flush=True)
            fo.write(s + "\n")
            fo.flush()
        F = Fam(p, d, max(d + 2, 4))
        sig = F.elem(F.sigma(orient))
        say(f"=== (p,d)=({p},{d}) orient={orient}: sigma' in PBW basis (level {F.m}): "
            + (" + ".join(f"{v}*{F.mstr(k)}" for k, v in sorted(sig.items())) if sig else "0"))
        for m in range(2, M + 1):
            t0 = time.time()
            F = Fam(p, d, m)
            sig = F.elem(F.sigma(orient))
            if not sig:
                say(f"m={m}: sigma' = 0 in u_m")
                continue
            res = right_kernel_dims(F, F.sigma(orient))
            dimu = p ** F.nl
            tot = sum(v[1] for v in res.values())
            ks = sorted(n for n, v in res.items() if v[1])
            km = ks[0] if ks else None
            prof = {n: res[n][1] for n in ks[:4]}
            say(f"m={m}: N={F.nl} dim u={dimu} top={F.top} | ker total={tot} density={tot/dimu:.6f} "
                f"coker/(d*|G|)={tot/(d*dimu):.6f} | k_m={km} odometer={odometer(p, d, m)} "
                f"first kernels={prof} | {time.time()-t0:.1f}s")
            fo.write("  perdeg " + json.dumps({n: res[n] for n in sorted(res)}) + "\n")


if __name__ == "__main__":
    main()
