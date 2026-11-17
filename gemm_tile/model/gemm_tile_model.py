# Golden model for rtl/gemm_tile.sv

from typing import List, NamedTuple

import numpy as np

from requant_model import requant_ref
from pe_model import signed_range


class Beat(NamedTuple):
    """One beat of the result stream."""
    data: List[int]     # N requantized values, index c = column c
    row: int            # m_axis_tuser
    last: int           # m_axis_tlast


def gemm_tile_ref(a, b, bias, mult, shift, zero_point, relu_en,
                  acc_width: int = 32, sat_width: int = 32,
                  data_width: int = 8) -> np.ndarray:
    """requant(A @ B + bias), per output channel (column).

    a is N x K, b is K x N, bias/mult/shift have one entry per column.
    Everything stays int64 until requant: an int8 or int32 intermediate would
    wrap silently and agree with nothing.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    bias = np.asarray(bias, dtype=np.int64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"shapes {a.shape} and {b.shape} do not multiply")
    if not (len(bias) == len(mult) == len(shift) == b.shape[1]):
        raise ValueError("need one bias, mult and shift per column")

    acc = a @ b
    assert acc.dtype == np.int64
    # the PE accumulator wraps at ACC_WIDTH. at the default widths K would have
    # to pass 2**17 for this to change anything.
    half = np.int64(1) << (acc_width - 1)
    acc = ((acc + half) & ((np.int64(1) << acc_width) - 1)) - half

    total = acc + bias[np.newaxis, :]
    assert total.dtype == np.int64

    out = np.empty(total.shape, dtype=np.int64)
    for r in range(total.shape[0]):
        for c in range(total.shape[1]):
            # the bias is already in; requant_ref saturates the sum itself
            out[r, c] = requant_ref(int(total[r, c]), 0, int(mult[c]), int(shift[c]),
                                    zero_point, relu_en, sat_width, data_width)
    return out


class GemmTileModel:
    """Transaction-level model: memory images in, a list of beats out."""

    REQUANT_LATENCY = 4
    SKID_LATENCY = 1

    def __init__(self, n: int = 8, data_width: int = 8, acc_width: int = 32,
                 read_latency: int = 1, b_addr_w: int = 13,
                 p_addr_w: int = 6, bias_width: int = 32, sat_width: int = 32,
                 m_width: int = 16, shift_width: int = 5, k_max: int = 1024):
        if read_latency not in (1, 2):
            raise ValueError(f"READ_LATENCY must be 1 or 2 (got {read_latency})")
        if (1 << p_addr_w) < n:
            raise ValueError("parameter memory is smaller than one tile")
        self.n = n
        self.data_width = data_width
        self.acc_width = acc_width
        self.read_latency = read_latency
        self.k_max = k_max
        self.b_addr_w = b_addr_w
        self.p_addr_w = p_addr_w
        self.bias_width = bias_width
        self.sat_width = sat_width
        self.m_width = m_width
        self.shift_width = shift_width
        self.b_mem = {}     # addr -> N values, index c
        self.p_mem = {}     # addr -> (bias, mult, shift)

    # packing, the same layout the RTL ports use

    def pack_bytes(self, values) -> int:
        if len(values) != self.n:
            raise ValueError(f"need N={self.n} values")
        lo, hi = signed_range(self.data_width)
        word = 0
        for i, v in enumerate(values):
            if not lo <= v <= hi:
                raise ValueError(f"operand {v} outside signed {self.data_width}-bit range")
            word |= (v & ((1 << self.data_width) - 1)) << (i * self.data_width)
        return word

    def unpack_bytes(self, word: int) -> List[int]:
        mask = (1 << self.data_width) - 1
        sign = 1 << (self.data_width - 1)
        vals = [(word >> (i * self.data_width)) & mask for i in range(self.n)]
        return [v - (1 << self.data_width) if v & sign else v for v in vals]

    def pack_param(self, bias: int, mult: int, shift: int) -> int:
        lo, hi = signed_range(self.bias_width)
        if not lo <= bias <= hi:
            raise ValueError(f"bias {bias} outside signed {self.bias_width}-bit range")
        if not 0 <= mult < (1 << self.m_width):
            raise ValueError(f"mult {mult} does not fit M_WIDTH={self.m_width}")
        if not 0 <= shift < (1 << self.shift_width):
            raise ValueError(f"shift {shift} does not fit SHIFT_WIDTH={self.shift_width}")
        return ((shift << (self.m_width + self.bias_width)) | (mult << self.bias_width)
                | (bias & ((1 << self.bias_width) - 1)))

    # memory images

    def write_b(self, addr: int, row) -> None:
        self.pack_bytes(row)
        self.b_mem[addr % (1 << self.b_addr_w)] = list(row)

    def write_p(self, addr: int, bias: int, mult: int, shift: int) -> None:
        self.pack_param(bias, mult, shift)
        self.p_mem[addr % (1 << self.p_addr_w)] = (bias, mult, shift)

    def load_tile(self, b, params, b_base: int, p_base: int) -> None:
        """b: K x N, params: N tuples of (bias, mult, shift)."""
        for j in range(len(b)):
            self.write_b(b_base + j, list(b[j]))
        for c in range(self.n):
            self.write_p(p_base + c, *params[c])

    def _word(self, mem, addr: int, width: int, name: str):
        addr %= 1 << width
        if addr not in mem:
            raise KeyError(f"{name} word {addr} was never written")
        return mem[addr]

    def tile_params(self, p_base: int):
        return [self._word(self.p_mem, p_base + c, self.p_addr_w, "P")
                for c in range(self.n)]

    def run_tile(self, slices, b_base: int, p_base: int,
                 zero_point: int, relu_en: int) -> List[Beat]:
        """The beats one tile puts on the stream, in order. slices is what the
        feeder supplies on the A port: K lists of N values, slice k holding
        column k of A."""
        k = len(slices)
        if not 1 <= k <= self.k_max:
            raise ValueError(f"K={k} outside 1 .. K_MAX={self.k_max}; the RTL ignores that start")
        cols = [list(sl) for sl in slices]
        for sl in cols:
            self.pack_bytes(sl)
        rows = [self._word(self.b_mem, b_base + j, self.b_addr_w, "B") for j in range(k)]
        a = np.array(cols, dtype=np.int64).T       # N x K
        b = np.array(rows, dtype=np.int64)         # K x N
        params = self.tile_params(p_base)
        q = gemm_tile_ref(a, b, [p[0] for p in params], [p[1] for p in params],
                          [p[2] for p in params], zero_point, relu_en,
                          self.acc_width, self.sat_width, self.data_width)
        return [Beat(data=[int(v) for v in q[r]], row=r, last=int(r == 0))
                for r in range(self.n - 1, -1, -1)]

    # latency, in edges after the one that accepts start, with tready high.
    # these are what the testbench measures on the DUT.

    def first_beat_latency(self, k: int) -> int:
        return (self.read_latency + k + 2 * self.n
                + self.REQUANT_LATENCY + self.SKID_LATENCY)

    def last_beat_latency(self, k: int) -> int:
        return self.first_beat_latency(k) + self.n - 1

    def ready_latency(self, k: int) -> int:
        return self.read_latency + k + 3 * self.n + 1

    def tile_period(self, k: int) -> int:
        """Cycles from one accepting edge to the next when tiles do not overlap."""
        return self.ready_latency(k) + 1


if __name__ == "__main__":
    import random

    from systolic_array_model import expected_matmul

    fails = 0

    def chk(name, got, exp):
        global fails
        if got != exp:
            print(f"FAIL {name}\n  got={got}\n  exp={exp}")
            fails += 1
        else:
            print(f"pass {name}")

    # worked by hand. A @ B = [[19, 22], [43, 50]], bias = [1, -50]
    #   column 0: scale 1/2  -> (20, 44)/2          = 10, 22
    #   column 1: scale 3/4  -> (-28, 0) * 3 / 4    = -21, 0
    # zero point 5, no ReLU: [[15, -16], [27, 5]]. with ReLU the -16 becomes 5.
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    got = gemm_tile_ref(a, b, bias=[1, -50], mult=[1, 3], shift=[1, 2],
                        zero_point=5, relu_en=0)
    chk("hand-worked 2x2", got.tolist(), [[15, -16], [27, 5]])
    chk("result dtype is int64", str(got.dtype), "int64")
    got = gemm_tile_ref(a, b, bias=[1, -50], mult=[1, 3], shift=[1, 2],
                        zero_point=5, relu_en=1)
    chk("hand-worked 2x2 with ReLU", got.tolist(), [[15, 5], [27, 5]])

    # int64 throughout: -128 * -128 * K overflows int8 and int16 at once
    k = 300
    got = gemm_tile_ref([[-128] * k], [[-128]] * k, bias=[0], mult=[1], shift=[16],
                        zero_point=0, relu_en=0)
    chk("no narrow intermediate", got.tolist(), [[75]])      # 4915200 / 65536 = 75

    # against a second formulation that shares no code path with the first:
    # pure Python matmul, and the bias handed to requant_ref instead of numpy
    rng = random.Random(1234)
    for n, k, acc_w in [(1, 1, 32), (2, 5, 32), (4, 16, 32), (8, 100, 32), (4, 40, 18)]:
        av = [[rng.randint(-128, 127) for _ in range(k)] for _ in range(n)]
        bv = [[rng.randint(-128, 127) for _ in range(n)] for _ in range(k)]
        bias = [rng.choice((rng.randint(-2**31, 2**31 - 1), rng.randint(-5000, 5000)))
                for _ in range(n)]
        mult = [rng.randrange(1 << 16) for _ in range(n)]
        shift = [rng.randrange(32) for _ in range(n)]
        zp, relu = rng.randint(-128, 127), rng.randint(0, 1)
        acc = expected_matmul(av, bv, acc_w)
        exp = [[requant_ref(acc[r][c], bias[c], mult[c], shift[c], zp, relu)
                for c in range(n)] for r in range(n)]
        got = gemm_tile_ref(av, bv, bias, mult, shift, zp, relu, acc_width=acc_w)
        chk(f"N={n} K={k} ACC_WIDTH={acc_w} two formulations agree", got.tolist(), exp)

    # the memory layout and beat order
    m = GemmTileModel(n=2, read_latency=2)
    m.load_tile(b, [(1, 1, 1), (-50, 3, 2)], b_base=900, p_base=62)
    slices = [[a[r][j] for r in range(2)] for j in range(2)]
    chk("an A slice is a column", slices[1], [2, 4])
    chk("B word is a row", m.b_mem[901], [7, 8])
    chk("P address wraps at the memory size", sorted(m.p_mem), [62, 63])
    chk("byte packing", m.pack_bytes([-1, 2]), 0x02FF)
    chk("byte unpacking", m.unpack_bytes(0x02FF), [-1, 2])
    chk("param packing", m.pack_param(-1, 3, 2), (2 << 48) | (3 << 32) | 0xFFFFFFFF)
    beats = m.run_tile(slices, 900, 62, zero_point=5, relu_en=0)
    chk("beats: row N-1 first, last on row 0",
        beats, [Beat([27, 5], 1, 0), Beat([15, -16], 0, 1)])

    chk("latency: first beat", m.first_beat_latency(2), 2 + 2 + 4 + 5)
    chk("latency: tile period N=8 K=72 L=1", GemmTileModel(n=8).tile_period(72), 99)

    try:
        m.run_tile(slices + [[0, 0]], 900, 62, 0, 0)
        chk("reading an unwritten word is an error", "no error", "KeyError")
    except KeyError:
        chk("reading an unwritten word is an error", "KeyError", "KeyError")

    try:
        GemmTileModel(n=2, k_max=4).run_tile([[0, 0]] * 5, 900, 62, 0, 0)
        chk("K > K_MAX is rejected", "no error", "ValueError")
    except ValueError:
        chk("K > K_MAX is rejected", "ValueError", "ValueError")

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")

    raise SystemExit(1 if fails else 0)
