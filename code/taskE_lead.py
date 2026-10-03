r"""Task E (exploration): is R_sigma' 'triangular' for a natural monomial order below the threshold?
For each Q-block of degree n < k_m and several total orders on PBW monomials, compute the leading monomial of w*sigma'
for each basis monomial w and test whether w -> lead(w sigma') is injective (=> R injective on the block).
Orders (engine bit order: bit 0 = highest-degree letter):
  hi-max : lex, highest-degree letters dominant, take max     (key = bit-reversed mask)
  hi-min : same key, take min
  lo-max : lex, lowest-degree letters dominant, take max      (key = mask)
  lo-min : same key, take min
usage: python taskE_lead.py m nmax
"""
import sys
from collections import Counter

import numpy as np

from cengine import Split
from taskC2_low import blocks_of_degree


def bitrev(x, nl):
    r = 0
    for i in range(nl):
        if (x >> i) & 1:
            r |= 1 << (nl - 1 - i)
    return r


def main(m, nmax):
    sp = Split(m)
    sig = sp.sigma_words()
    nl = sp.nl
    for n in range(1, nmax + 1):
        B = blocks_of_degree(sp, n)
        fails = Counter()
        nb = 0
        for g, rows in B.items():
            nb += 1
            leads = {k: [] for k in ("hi-max", "hi-min", "lo-max", "lo-min")}
            zero = False
            for w in rows:
                acc = {}
                for wd in sig:
                    for t in sp.rmul(int(w), wd):
                        acc[t] = acc.get(t, 0) ^ 1
                sup = [t for t, c in acc.items() if c]
                if not sup:
                    zero = True
                    break
                kh = [bitrev(t, nl) for t in sup]
                leads["hi-max"].append(sup[int(np.argmax(kh))])
                leads["hi-min"].append(sup[int(np.argmin(kh))])
                leads["lo-max"].append(max(sup))
                leads["lo-min"].append(min(sup))
            for k, v in leads.items():
                if zero or len(set(v)) < len(v):
                    fails[k] += 1
        print(f"m={m} n={n}: {nb} blocks; blocks where lead map NOT injective: {dict(fails)}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
