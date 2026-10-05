r"""Task F exploration: sigma' * (t^{-1} z) * omega_{K - z}  versus omega_K  (K = letters of degree m-3..m-1).
usage: python taskF_explore.py m"""
import sys
import numpy as np
from cengine import Split, pack_words, rmul_word, workbufs, cancel, MAXOUT

m = int(sys.argv[1])
sp = Split(m)
out = np.empty(MAXOUT, dtype=np.int64)
bufs = workbufs()


def mul_word(vec, word):
    acc = []
    for S in vec:
        n = rmul_word(np.int64(S), np.asarray(word, dtype=np.int64), sp.br, sp.sq, out, 0, *bufs)
        assert n >= 0
        acc.append(out[:n].copy())
    return cancel(np.concatenate(acc)) if acc else []


def lets(mono):
    return [i for i in range(sp.nl) if (mono >> i) & 1]


def left_sigma(vec):
    res = []
    for w in sp.sigma_words():
        for S in vec:
            n = rmul_word(np.int64(0), np.concatenate([w, np.array(lets(int(S)), dtype=np.int64)]), sp.br, sp.sq,
                          out, 0, *bufs)
            res.append(out[:n].copy())
    return cancel(np.concatenate(res))


s = sum(1 for (k, p) in sp.letters if k >= m - 3)
omegaK = (1 << s) - 1
print("omega_K =", sp.mstr(omegaK))
total = []
for zi in range(s):
    k, p = sp.letters[zi]
    xi = sp.pos[(k - 3, p)]
    rest = omegaK & ~(1 << zi)
    prod = mul_word([1 << xi], lets(rest))            # x * omega_{K-z}  (x has lower degree -> comes after)
    v = left_sigma(prod)
    diff = set(v) ^ {omegaK}
    print(f"z = y{k}.{p}: sigma' x omega_(K-z) has {len(v)} terms; contains omega_K: {omegaK in v}; "
          f"diff has {len(diff)} terms", flush=True)
    total.append(v)
