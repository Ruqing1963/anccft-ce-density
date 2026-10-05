r"""Task F: machine checks for the injectivity theorem (THEORY §7.1).

(1) rho*: L^s -> gl_3(F2[t]),  rho*(x) = (pi(x) + tr(pi(x)) I)^T,  pi(e_p phi^k) = E_pp Pi^k,  Pi = E10 + E21 + t E02.
    Check: rho* is injective and graded on the letters, rho*(YZ) = rho*(Y) rho*(Z) for random PBW monomials Y, Z
    (products from cengine), and rho*(sigma') = t I.
(2) Faithfulness: Psi_r(y) = (rho*_{t_1} (x) ... (x) rho*_{t_r})(Delta^{(r)} y) at a random point of GF(2^16)^r,
    applied to 2 random vectors, is injective on U_n (n <= nmax, r = n)  -- confirms the Nakayama argument numerically.
(3) det Psi_r(sigma') at the point (t, 0, ..., 0) equals t^(3^r)  (the block-triangular induction).
usage: python taskF_faithful.py nmax
"""
import sys
import numpy as np

from cengine import Split, pack_words, rmul_word, workbufs, cancel, MAXOUT
from taskC2_low import blocks_of_degree
from taskF_tensor import EXP, LOG, K, gmul, rank as frank, sigma_op, matmul


# ---------- polynomials over F2 as python ints (bit i = t^i) ----------
def pmul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def mmul(A, B):
    return [[(lambda i, j: __import__("functools").reduce(lambda x, y: x ^ y, [pmul(A[i][k], B[k][j]) for k in range(3)]))(i, j)
             for j in range(3)] for i in range(3)]


def madd(A, B):
    return [[A[i][j] ^ B[i][j] for j in range(3)] for i in range(3)]


I3 = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
Z3 = [[0] * 3 for _ in range(3)]
PI = [[0, 0, 0b10], [1, 0, 0], [0, 1, 0]]     # E10 + E21 + t E02  (rows i, cols j)


def rho_star_letter(k, p):
    P = I3
    for _ in range(k):
        P = mmul(P, PI)
    E = [[1 if (i == j == p) else 0 for j in range(3)] for i in range(3)]
    X = mmul(E, P)
    tr = X[0][0] ^ X[1][1] ^ X[2][2]
    X = madd(X, [[tr if i == j else 0 for j in range(3)] for i in range(3)])
    return [[X[j][i] for j in range(3)] for i in range(3)]          # transpose


def deg_ok(M, d):
    for p in range(3):
        for q in range(3):
            v = M[p][q]
            k = 0
            while v:
                if v & 1 and 3 * k + q - p != d:
                    return False
                v >>= 1
                k += 1
    return True


def check1(m=13, ntests=300):
    sp = Split(m)
    R = [rho_star_letter(k, p) for (k, p) in sp.letters]
    # injective + graded on letters (letters of one degree linearly independent images)
    for i, (k, p) in enumerate(sp.letters):
        assert deg_ok(R[i], k), (k, p)
    for k in range(1, m):
        idx = [i for i, (kk, p) in enumerate(sp.letters) if kk == k]
        vecs = []
        for i in idx:
            flat = 0
            for a in range(3):
                for b in range(3):
                    flat = (flat << 40) | R[i][a][b]
            vecs.append(flat)
        # GF(2) rank of these flattened vectors
        piv = {}
        rk = 0
        for v in vecs:
            while v:
                h = v.bit_length() - 1
                if h in piv:
                    v ^= piv[h]
                else:
                    piv[h] = v
                    rk += 1
                    break
        assert rk == len(idx), k

    def rho_mono(S):
        M = I3
        for i in range(sp.nl):
            if (S >> i) & 1:
                M = mmul(M, R[i])
        return M

    def rho_vec(v):
        M = Z3
        for S in v:
            M = madd(M, rho_mono(int(S)))
        return M
    rng = np.random.default_rng(3)
    out = np.empty(MAXOUT, dtype=np.int64)
    bufs = workbufs()
    nt = 0
    while nt < ntests:
        S = int(rng.integers(0, 1 << sp.nl))
        T = int(rng.integers(0, 1 << sp.nl))
        # keep total degree < m so that the product in u_m is the product in U
        if sp.pdeg(S) + sp.pdeg(T) >= m:
            S &= int(rng.integers(0, 1 << sp.nl)) & int(rng.integers(0, 1 << sp.nl))
            T &= int(rng.integers(0, 1 << sp.nl)) & int(rng.integers(0, 1 << sp.nl))
            if sp.pdeg(S) + sp.pdeg(T) >= m:
                continue
        lets = np.array([i for i in range(sp.nl) if (T >> i) & 1], dtype=np.int64)
        n = rmul_word(np.int64(S), lets, sp.br, sp.sq, out, 0, *bufs)
        prod = cancel(out[:n])
        assert rho_vec(prod) == mmul(rho_mono(S), rho_mono(T)), (S, T)
        nt += 1
    words, wlen = pack_words(sp.sigma_words())
    sv = []
    for w in range(wlen.size):
        n = rmul_word(np.int64(0), words[w, :wlen[w]], sp.br, sp.sq, out, 0, *bufs)
        sv.append(out[:n].copy())
    sig = cancel(np.concatenate(sv))
    assert rho_vec(sig) == [[0b10 if i == j else 0 for j in range(3)] for i in range(3)]
    print(f"(1) rho*: graded, injective on letters of degree < {m}; homomorphism on {ntests} random products; "
          f"rho*(sigma') = t I : OK", flush=True)
    return sp, R


def peval(poly, a):
    """evaluate F2[t]-poly at a in GF(2^K)"""
    r = 0
    pw = 1
    while poly:
        if poly & 1:
            r ^= pw
        pw = gmul(pw, a, EXP, LOG)
        poly >>= 1
    return r


def check2(nmax):
    rng = np.random.default_rng(11)
    for n in range(1, nmax + 1):
        m = n + 1
        sp = Split(m)
        R = [rho_star_letter(k, p) for (k, p) in sp.letters]
        r = n
        dim = 3 ** r
        ts = [int(x) for x in rng.integers(1, 1 << K, r)]
        # site operators per letter: list of (s, 3x3 numeric)
        Rn = [[[[peval(R[i][a][b], ts[s]) for b in range(3)] for a in range(3)] for s in range(r)]
              for i in range(sp.nl)]

        def apply_letter(i, v):
            out = np.zeros_like(v)
            vv = v.reshape((3,) * r)
            for s in range(r):
                M = Rn[i][s]
                moved = np.moveaxis(vv, s, 0)
                res = np.zeros_like(moved)
                for a in range(3):
                    for b in range(3):
                        c = M[a][b]
                        if c:
                            src = moved[b]
                            res[a] ^= np.array([gmul(int(x), c, EXP, LOG) for x in src.ravel()],
                                               dtype=np.int64).reshape(src.shape)
                out ^= np.moveaxis(res, 0, s).reshape(-1)
            return out
        vs = [rng.integers(0, 1 << K, dim).astype(np.int64) for _ in range(2)]
        B = blocks_of_degree(sp, n)
        monos = np.concatenate([B[g] for g in B])
        rows = []
        for S in monos:
            lets = [i for i in range(sp.nl) if (int(S) >> i) & 1]
            row = []
            for v in vs:
                w = v.copy()
                for i in reversed(lets):
                    w = apply_letter(i, w)
                row.append(w)
            rows.append(np.concatenate(row))
        A = np.array(rows, dtype=np.int64)
        rk = int(frank(A, EXP, LOG))
        print(f"(2) n={n}: dim U_n = {monos.size}, rank of y -> (Psi_n(y) v1, Psi_n(y) v2) = {rk}", flush=True)
        assert rk == monos.size


def check3(rmax):
    for r in range(1, rmax + 1):
        t = 12345
        M = sigma_op(r, [t] + [0] * (r - 1))
        # determinant via elimination over GF(2^16)
        A = M.copy()
        n = A.shape[0]
        det = 1
        from taskF_tensor import ginv
        for c in range(n):
            p = next((i for i in range(c, n) if A[i, c]), -1)
            assert p >= 0
            A[[c, p]] = A[[p, c]]
            det = gmul(det, int(A[c, c]), EXP, LOG)
            inv = ginv(int(A[c, c]), EXP, LOG)
            for i in range(c + 1, n):
                if A[i, c]:
                    f = gmul(int(A[i, c]), inv, EXP, LOG)
                    for j in range(c, n):
                        if A[c, j]:
                            A[i, j] ^= gmul(f, int(A[c, j]), EXP, LOG)
        tp = 1
        for _ in range(3 ** r):
            tp = gmul(tp, t, EXP, LOG)
        print(f"(3) r={r}: det Psi_r(sigma')(t,0,..,0) == t^(3^r): {det == tp}", flush=True)
        assert det == tp


if __name__ == "__main__":
    nmax = int(sys.argv[1])
    check1()
    check3(5)
    check2(nmax)
