"""Task G driver: exact ker R^{(M)} on every rho-orbit representative Q-block of degrees nmin..nmax, lifted from level a
(taskG_core.lift_block).  Checkpoints after every block to taskG_M{M}_a{a}.pkl (resumable).  Optional time limit (s).
usage: python taskG_run.py M a nmin nmax [timelimit]"""
import pickle
import sys
import time

import os
from taskG_core import LowLevel, lift_block, lift_block2, to_monomials, orbit_rep
from taskG_dims import block_dims


def main(M, a, nmin, nmax, tlim):
    t0 = time.time()
    kerdims = pickle.load(open(f"taskC1_m{a}.pkl", "rb"))["ker"]
    LL = LowLevel(M, a)
    fn = os.environ.get("TASKG_PKL", f"taskG_M{M}_a{a}.pkl")
    try:
        res = pickle.load(open(fn, "rb"))
    except FileNotFoundError:
        res = {"M": M, "a": a, "blocks": {}, "kernels": {}}
    bd = block_dims(M, nmax)
    print(f"M={M} from a={a}: I-letters {[LL.hi.letters[i] for i in range(LL.s)]}", flush=True)
    for n in range(nmin, nmax + 1):
        reps = sorted((g for g in bd if sum(g) == n and orbit_rep(g) == g), key=lambda g: bd[g])
        tot = 0
        for g in reps:
            if g in res["blocks"]:
                tot += res["blocks"][g][0]
                continue
            if bd[g] > float(os.environ.get("TASKG_MAXDIM", "inf")):
                print(f"  gamma={g} n={n} dim={bd[g]}: skipped (TASKG_MAXDIM)", flush=True)
                continue
            if tlim and time.time() - t0 > tlim:
                print(f"time limit reached before {g}", flush=True)
                return
            tb = time.time()
            if os.environ.get("TASKG_CKPT"):
                ckd = os.path.join(os.environ["TASKG_CKPT"], f"ck_M{M}_{g[0]}_{g[1]}_{g[2]}")
                dim, info, cand = lift_block2(LL, g, kerdims, verbose=True, ckdir=ckd,
                                              ckevery=float(os.environ.get("TASKG_CKEVERY", "1200")))
            else:
                dim, info, cand = (lift_block2 if os.environ.get("TASKG_V2") else lift_block)(LL, g, kerdims)
            dt = time.time() - tb
            res["blocks"][g] = (dim, info, dt, bd[g])
            if dim:
                res["kernels"][g] = [to_monomials(LL, g, cand, r) for r in range(dim)]
            tot += dim
            pickle.dump(res, open(fn, "wb"))
            print(f"  gamma={g} n={n} dim={bd[g]}: layer tests (layer, #cand, rank) = {info} -> ker = {dim}"
                  f"   [{dt:.1f}s, total {time.time() - t0:.0f}s]  {dict((k, round(v)) for k, v in LL.stats.items())}",
                  flush=True)
            if dim:
                for r in range(dim):
                    mons = res["kernels"][g][r]
                    print(f"     kernel vector {r}: {len(mons)} terms: " +
                          " + ".join(LL.hi.mstr(x) for x in mons[:6]) + (" + ..." if len(mons) > 6 else ""),
                          flush=True)
        print(f"M={M} n={n}: {len(reps)} orbit reps, kernel dim (reps) = {tot}, total = 3*... see blocks", flush=True)


if __name__ == "__main__":
    M, a, nmin, nmax = (int(x) for x in sys.argv[1:5])
    tlim = float(sys.argv[5]) if len(sys.argv) > 5 else 0
    main(M, a, nmin, nmax, tlim)

