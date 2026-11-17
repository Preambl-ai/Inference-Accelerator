# Golden model for rtl/im2col_reader.sv, and a stand-in for the activation buffer

import numpy as np

from bram_model import BramModel, UNKNOWN


def pixel_order(y0: int, x0: int, n: int, w_out: int, win_order: int = 0):
    """The N output pixels of one tile, in row order r = 0 .. N-1.

    Raster order walks x and wraps at w_out. Window order visits 2x2 pooling
    windows, four pixels each, and wraps window by window. Pixels past the
    bottom of the output are returned like any other; they are dropped later.
    """
    out = []
    y, x = y0, x0
    if not win_order:
        for _ in range(n):
            out.append((y, x))
            x += 1
            if x >= w_out:
                x, y = 0, y + 1
        return out
    if n % 4 or w_out % 2 or y0 % 2 or x0 % 2:
        raise ValueError("window order needs N % 4 == 0 and even w_out, y0, x0")
    for _ in range(n // 4):
        out += [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)]
        x += 2
        if x >= w_out:
            x, y = 0, y + 2
    return out


def im2col_slices(tensor, y0: int, x0: int, n: int, w_out: int, kh: int, kw: int,
                  stride: int, pad: int, pad_value: int, win_order: int = 0):
    """The K operand slices of one tile, straight from the tensor.

    tensor is H x W x C with C a multiple of N. Slice k = ((ky*kw + kx)*cw + g)*N + j
    holds, for each tile pixel, channel g*N + j of the input pixel under tap
    (ky, kx), or pad_value outside the image. Knows nothing about addresses,
    words or timing.
    """
    t = np.asarray(tensor, dtype=np.int64)
    h, w, c = t.shape
    if c % n:
        raise ValueError(f"C={c} is not a multiple of N={n}")
    pixels = pixel_order(y0, x0, n, w_out, win_order)
    slices = []
    for ky in range(kh):
        for kx in range(kw):
            for ch in range(c):
                col = []
                for (y, x) in pixels:
                    yi, xi = y * stride + ky - pad, x * stride + kx - pad
                    inside = 0 <= yi < h and 0 <= xi < w
                    col.append(int(t[yi, xi, ch]) if inside else pad_value)
                slices.append(col)
    return slices


def out_size(size: int, k: int, stride: int, pad: int) -> int:
    return (size + 2 * pad - k) // stride + 1


def word_addr(base: int, y: int, x: int, g: int, w: int, cw: int) -> int:
    """Where the word for pixel (y, x), channel group g lives. This is the
    layout contract between the reader, the writer and the activation buffer."""
    return base + (y * w + x) * cw + g


def tensor_words(tensor, n: int, base: int = 0):
    """{address: [N channel values]} for a whole tensor."""
    t = np.asarray(tensor, dtype=np.int64)
    h, w, c = t.shape
    if c % n:
        raise ValueError(f"C={c} is not a multiple of N={n}")
    cw = c // n
    return {word_addr(base, y, x, g, w, cw): [int(v) for v in t[y, x, g * n:(g + 1) * n]]
            for y in range(h) for x in range(w) for g in range(cw)}


class ActBufferModel:
    """Python stand-in for the activation buffer block (not written here):
    NBUF word-wide memories built on bram.sv, one read port and one write
    port, a buffer select on each.

        rd_en, rd_sel, rd_addr -> rd_data, READ_LATENCY enabled edges later
        wr_en, wr_sel, wr_addr, wr_data

    rd_en is the read clock enable, exactly as bram.re. The select behaves as
    the top address bits, so this is one BramModel over {sel, addr}. A word is
    kept as a tuple of N channel values; rdata() is UNKNOWN wherever the real
    memory would be undefined.
    """

    def __init__(self, n: int, addr_width: int, read_latency: int = 1, nbuf: int = 3,
                 data_width: int = 8):
        self.n = n
        self.addr_width = addr_width
        self.nbuf = nbuf
        self.data_width = data_width
        sel_w = max(1, (nbuf - 1).bit_length())
        self.mem = BramModel(n * data_width, addr_width + sel_w, read_latency)

    def _pack(self, values) -> int:
        mask = (1 << self.data_width) - 1
        word = 0
        for i, v in enumerate(values):
            word |= (v & mask) << (i * self.data_width)
        return word

    def _key(self, sel: int, addr: int) -> int:
        if not 0 <= sel < self.nbuf:
            raise ValueError(f"buffer select {sel} outside 0 .. {self.nbuf - 1}")
        return (sel << self.addr_width) | (addr % (1 << self.addr_width))

    def load(self, sel: int, words: dict) -> None:
        """Backdoor fill, e.g. with tensor_words()."""
        for addr, values in words.items():
            self.mem.mem[self._key(sel, addr)] = self._pack(values)

    def peek(self, sel: int, addr: int):
        """Backdoor read of one word as N signed values, or UNKNOWN."""
        word = self.mem.mem.get(self._key(sel, addr), UNKNOWN)
        if word is UNKNOWN:
            return UNKNOWN
        mask, sign = (1 << self.data_width) - 1, 1 << (self.data_width - 1)
        vals = [(word >> (i * self.data_width)) & mask for i in range(self.n)]
        return [v - (1 << self.data_width) if v & sign else v for v in vals]

    def rdata(self):
        """rd_data as a packed word, or UNKNOWN."""
        return self.mem.rdata()

    def step(self, rd_en: int = 0, rd_sel: int = 0, rd_addr: int = 0,
             wr_en: int = 0, wr_sel: int = 0, wr_addr: int = 0, wr_data: int = 0):
        return self.mem.step(we=wr_en, waddr=self._key(wr_sel, wr_addr), wdata=wr_data,
                             re=rd_en, raddr=self._key(rd_sel, rd_addr))


class Im2colReaderModel:
    """Timing of the reference reader, for the numbers the testbench measures."""

    def __init__(self, n: int = 8, read_latency: int = 1):
        self.n = n
        self.read_latency = read_latency

    def gather_latency(self) -> int:
        """Edges from the one that starts a group's walk to its first slice
        being valid, as measured on the RTL: N addresses one per cycle, two
        address stages and the memory behind the last of them."""
        return self.n + self.read_latency + 2

    def group_period(self) -> int:
        """Cycles per group of N slices when each is taken at once."""
        return self.gather_latency() + self.n

    def tile_k(self, kh: int, kw: int, cin_words: int) -> int:
        return kh * kw * cin_words * self.n


if __name__ == "__main__":
    import random

    fails = 0

    def chk(name, got, exp):
        global fails
        if got != exp:
            print(f"FAIL {name}\n  got={got}\n  exp={exp}")
            fails += 1
        else:
            print(f"pass {name}")

    # pixel walks
    chk("raster wraps at w_out", pixel_order(1, 2, 4, 4), [(1, 2), (1, 3), (2, 0), (2, 1)])
    chk("window order", pixel_order(0, 0, 8, 4, 1),
        [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    chk("window order wraps by window", pixel_order(0, 2, 8, 4, 1),
        [(0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1)])

    # worked by hand: 3x3 image, one channel pair, value = 10*y + x (+100 for ch 1)
    t = np.array([[[10 * y + x, 100 + 10 * y + x] for x in range(3)] for y in range(3)])
    s = im2col_slices(t, 0, 0, 2, 3, kh=3, kw=3, stride=1, pad=1, pad_value=-7)
    chk("K = kh*kw*C", len(s), 18)
    # tap (0,0): both pixels of the top row look above the image
    chk("top-left tap is all padding", s[0], [-7, -7])
    # tap (1,1) is the pixel itself: (0,0) and (0,1), channel 0 then channel 1
    chk("centre tap ch0", s[(1 * 3 + 1) * 2 + 0], [0, 1])
    chk("centre tap ch1", s[(1 * 3 + 1) * 2 + 1], [100, 101])
    # tap (1,0): one to the left. pixel (0,0) falls off, pixel (0,1) sees (0,0)
    chk("left tap mixes pad and data", s[(1 * 3 + 0) * 2], [-7, 0])
    # tap (2,2): one down, one right
    chk("bottom-right tap", s[(2 * 3 + 2) * 2], [11, 12])

    # stride 2, no padding: output pixel (0,1) reads input x = 2
    s = im2col_slices(t, 0, 0, 2, 1, kh=1, kw=1, stride=2, pad=0, pad_value=-7)
    chk("stride 2 walks to the next row", s[0], [0, 20])

    # against numpy's own windowing on a padded copy
    rng = random.Random(1234)
    for _ in range(40):
        n = rng.choice((2, 4, 8))
        k, stride, pad = rng.choice((1, 3, 5)), rng.choice((1, 2)), rng.choice((0, 1, 2))
        h, w, cw = rng.randint(k, 9), rng.randint(k, 9), rng.randint(1, 2)
        t = np.array([[[rng.randint(-128, 127) for _ in range(cw * n)]
                       for _ in range(w)] for _ in range(h)])
        pv = rng.randint(-128, 127)
        ho, wo = out_size(h, k, stride, pad), out_size(w, k, stride, pad)
        # a border deep enough that pixels past the end of the output still index it
        big = 4 * n + 8
        padded = np.full((h + 2 * big, w + 2 * big, cw * n), pv, dtype=np.int64)
        padded[big:big + h, big:big + w] = t
        y0, x0 = rng.randrange(ho), rng.randrange(wo)
        got = im2col_slices(t, y0, x0, n, wo, k, k, stride, pad, pv)
        exp = []
        for ky in range(k):
            for kx in range(k):
                for ch in range(cw * n):
                    exp.append([int(padded[big + y * stride + ky - pad,
                                           big + x * stride + kx - pad, ch])
                                for (y, x) in pixel_order(y0, x0, n, wo)])
        if got != exp:
            chk(f"random geometry k={k} s={stride} p={pad}", got, exp)
            break
    else:
        chk("40 random geometries against a padded copy", True, True)

    # layout and the buffer stand-in
    t = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    words = tensor_words(t, n=2, base=100)
    chk("word address", word_addr(100, 1, 2, 1, w=3, cw=2), 100 + (1 * 3 + 2) * 2 + 1)
    chk("word contents", words[100 + (1 * 3 + 2) * 2 + 1], [22, 23])
    buf = ActBufferModel(n=2, addr_width=8, read_latency=2, nbuf=3)
    buf.load(1, words)
    chk("peek", buf.peek(1, 111), [22, 23])
    chk("other buffer is separate", buf.peek(0, 111), UNKNOWN)
    seq = [buf.step(rd_en=1, rd_sel=1, rd_addr=111) for _ in range(2)]
    chk("read latency 2", seq, [UNKNOWN, 22 | (23 << 8)])
    buf.step(wr_en=1, wr_sel=2, wr_addr=5, wr_data=buf._pack([-1, 2]))
    chk("write port", buf.peek(2, 5), [-1, 2])

    m = Im2colReaderModel(n=8, read_latency=1)
    chk("K for 3x3, Cin=16", m.tile_k(3, 3, 2), 144)

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")

    raise SystemExit(1 if fails else 0)
