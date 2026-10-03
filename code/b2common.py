"""Helpers for task B2: packed column blocks over the Jennings basis of A_N = F2[[P]]/I^N, GF(2) solving."""
import numpy as np

from jtable import JTable
from zdtools import elem_from_bits

VBITS = (4005, 524823, 1310958)         # basis of V in u_4 (v = first); 4005^524823 = 527794
X3_STR = "x1.1*x2.1 + x1.2*x2.2 + x1.2*x2.4 + x1.4*x2.4"


def get_V(T):
    return [elem_from_bits(T.U, 4, g) for g in VBITS]


def get_x3(T):
    X = {mm for mm in T.U.monos(3) if T.U.mono_str(mm) in X3_STR.split(" + ")}
    assert len(X) == 4
    return X


def unit_block(T, cols):
    """packed (D, W) block whose c-th column is the basis vector cols[c]"""
    K = len(cols)
    W = max(1, (K + 63) // 64)
    Y = np.zeros((T.D, W), dtype=np.uint64)
    c = np.arange(K)
    Y[np.asarray(cols, dtype=np.int64), c >> 6] |= np.left_shift(np.uint64(1), (c & 63).astype(np.uint64))
    return Y


def block_columns_as_ints(Y, nrows, K, r0=0):
    """column c of Y[r0:nrows] -> python int (bit i = row r0+i)"""
    B = np.unpackbits(Y[r0:nrows].view(np.uint8), axis=1, bitorder="little")[:, :K]   # (nrows-r0, K)
    Bt = np.ascontiguousarray(B.T)
    packed = np.packbits(Bt, axis=1, bitorder="little")
    return [int.from_bytes(packed[c].tobytes(), "little") for c in range(K)]


def vec_to_int(v, nrows, r0=0):
    b = np.packbits(v[r0:nrows].astype(np.uint8), bitorder="little")
    return int.from_bytes(b.tobytes(), "little")


def int_to_vec(x, D):
    v = np.zeros(D, dtype=np.uint8)
    i = 0
    while x:
        if x & 1:
            v[i] = 1
        x >>= 1
        i += 1
    return v


class Solver:
    """span of images img[k] (python ints); solve sum c_k img[k] = target; kernel basis (as tags)."""

    def __init__(self, imgs):
        self.piv = {}
        self.kernel = []
        for k, v in enumerate(imgs):
            tag = 1 << k
            while v:
                h = v.bit_length() - 1
                p = self.piv.get(h)
                if p is None:
                    self.piv[h] = (v, tag)
                    break
                v ^= p[0]
                tag ^= p[1]
            if not v:
                self.kernel.append(tag)
        self.rank = len(self.piv)

    def solve(self, t):
        tag = 0
        while t:
            h = t.bit_length() - 1
            p = self.piv.get(h)
            if p is None:
                return None, t
            t ^= p[0]
            tag ^= p[1]
        return tag, 0
