# Golden model for rtl/add_layer.sv
#
#   out = sat8(round(((a - za)*ma + (b - zb)*mb) / 2^shift) + zo), optional ReLU
#   round = (x + 2^(shift-1)) >> shift (arithmetic, round half up); shift 0: x
#
# add_ref is vectorized NumPy on (H, W, C) tensors; add_naive is per element
# in plain Python ints, written independently, for the self-test. The word
# view (add_words) runs add_ref on every lane of the packed act_buffer image,
# padded lanes included, which is what the hardware computes.

from dataclasses import dataclass

import numpy as np

from conv_layer_model import pack_act, unpack_act, words

E_DIM, E_SRC, E_DST, E_SEL, E_HOST = 0, 5, 6, 7, 8
ERR_NAMES = {E_DIM: "E_DIM", E_SRC: "E_SRC", E_DST: "E_DST", E_SEL: "E_SEL", E_HOST: "E_HOST"}


def add_ref(a, b, za, zb, zo, ma, mb, shift, relu_en):
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    acc = (a - za) * ma + (b - zb) * mb
    q = (acc + (1 << (shift - 1))) >> shift if shift else acc
    lo = zo if relu_en else -128
    return np.clip(q + zo, lo, 127)


def add_naive(a, b, za, zb, zo, ma, mb, shift, relu_en):
    out = []
    for x, y in zip(a, b):
        acc = (int(x) - za) * ma + (int(y) - zb) * mb
        if shift:
            acc += 1 << (shift - 1)
            q = acc // (1 << shift)             # floor division == arithmetic shift
        else:
            q = acc
        v = q + zo
        out.append(min(127, max(zo if relu_en else -128, v)))
    return out


@dataclass
class AddCmd:
    h_in: int
    w_in: int
    c_words: int
    src_sel: int = 0
    src_base: int = 0
    src2_sel: int = 0
    src2_base: int = 0
    dst_sel: int = 1
    dst_base: int = 0
    za: int = 0
    zb: int = 0
    zo: int = 0
    ma: int = 1
    mb: int = 1
    shift: int = 0
    relu_en: int = 0

    def as_dict(self):
        return dict(self.__dict__)

    @property
    def n_words(self):
        return self.h_in * self.w_in * self.c_words


def check_command(c: AddCmd, addr_width: int, nbuf: int, host_mode: int = 0) -> int:
    code = (c.h_in == 0 or c.w_in == 0 or c.c_words == 0) << E_DIM
    n = c.n_words
    top = 1 << addr_width
    code |= (c.src_base + n > top or c.src2_base + n > top) << E_SRC
    code |= (c.dst_base + n > top) << E_DST
    code |= (c.dst_sel in (c.src_sel, c.src2_sel) or max(c.src_sel, c.src2_sel, c.dst_sel) >= nbuf) << E_SEL
    code |= int(bool(host_mode)) << E_HOST
    return code


def err_names(code):
    return [ERR_NAMES[b] for b in sorted(ERR_NAMES) if code >> b & 1]


def add_words(a_words, b_words, c: AddCmd, n):
    """packed words in, packed words out, every lane (padding included)"""
    fa = unpack_act(a_words, c.h_in, c.w_in, c.c_words * n, n, with_pad=True)
    fb = unpack_act(b_words, c.h_in, c.w_in, c.c_words * n, n, with_pad=True)
    out = add_ref(fa, fb, c.za, c.zb, c.zo, c.ma, c.mb, c.shift, c.relu_en)
    return pack_act(out, n, 0)


def scales_to_params(sa, sb, so, m_bits=8, max_shift=15):
    """host-side helper: real scales -> (ma, mb, shift) with the largest shift
    that keeps both multipliers in M_BITS (most precision)"""
    for s in range(max_shift, -1, -1):
        ma, mb = round(sa / so * (1 << s)), round(sb / so * (1 << s))
        if ma < (1 << m_bits) and mb < (1 << m_bits):
            return ma, mb, s
    raise ValueError("scale ratio too large for M_BITS")


def random_cmd(rng, n, addr_width, nbuf, max_dim=6, max_cw=3, m_bits=8):
    """a random legal add, with a, b tensors (H, W, C) and their zero points"""
    for _ in range(1000):
        h, w = int(rng.integers(1, max_dim + 1)), int(rng.integers(1, max_dim + 1))
        c = int(rng.integers(1, max_cw * n + 1))
        cw = words(c, n)
        nw = h * w * cw
        if nw > (1 << addr_width):
            continue
        banks = list(range(nbuf))
        dst = int(rng.choice(banks))
        srcs = [b for b in banks if b != dst]
        sa, sb = int(rng.choice(srcs)), int(rng.choice(srcs))
        cmd = AddCmd(h, w, cw, sa, 0, sb, 0, dst, 0,
                     za=int(rng.integers(-128, 128)), zb=int(rng.integers(-128, 128)),
                     zo=int(rng.integers(-128, 128)), ma=int(rng.integers(0, 1 << m_bits)),
                     mb=int(rng.integers(0, 1 << m_bits)), shift=int(rng.integers(0, 16)),
                     relu_en=int(rng.integers(0, 2)))
        top = (1 << addr_width) - nw
        cmd.src_base = int(rng.integers(0, top + 1))
        cmd.dst_base = int(rng.integers(0, top + 1))
        # a and b in one bank must not overlap each other (the host would
        # never do it, and the model reads each from the mirror)
        for _ in range(100):
            cmd.src2_base = int(rng.integers(0, top + 1))
            if sa != sb or abs(cmd.src2_base - cmd.src_base) >= nw:
                break
        else:
            continue
        a = rng.integers(-128, 128, (h, w, c))
        b = rng.integers(-128, 128, (h, w, c))
        return cmd, a, b
    raise RuntimeError("no legal add found")


def estimate_cycles(c: AddCmd, read_latency: int, m_bits=8, dim_w=8, cw_w=7) -> int:
    """start to done: DIM_W + CW_W + 1 check edges, then per word: read a,
    read b, READ_LATENCY until b is in, M_BITS shift-and-add steps, one output
    edge; plus one edge for the last write"""
    return (dim_w + cw_w + 1) + c.n_words * (m_bits + read_latency + 3) + 1


if __name__ == "__main__":
    fails = 0

    def chk(name, got, exp):
        global fails
        ok = np.array_equal(got, exp) if isinstance(got, np.ndarray) or isinstance(exp, np.ndarray) \
            else got == exp
        print(f"{'pass' if ok else 'FAIL'} {name}" + ("" if ok else f"\n   got {got}\n   exp {exp}"))
        fails += not ok

    r = lambda *v: int(add_ref(*v))
    chk("identity: a + b, scales 1", r(3, 4, 0, 0, 0, 1, 1, 0, 0), 7)
    chk("zero points cancel", r(13, 24, 10, 20, 5, 1, 1, 0, 0), 12)
    chk("half and half, 3 + 4 = 7 / 2 = 3.5 -> 4", r(3, 4, 0, 0, 0, 1, 1, 1, 0), 4)
    chk("-3.5 -> -3 (half up)", r(-3, -4, 0, 0, 0, 1, 1, 1, 0), -3)
    chk("saturate high", r(127, 127, 0, 0, 0, 255, 255, 0, 0), 127)
    chk("saturate low", r(-128, -128, 0, 0, 0, 255, 255, 0, 0), -128)
    chk("relu floor at zo", r(-50, 0, 0, 0, 10, 1, 0, 0, 1), 10)
    chk("padded lanes (a=za, b=zb) give zo", r(-7, 33, -7, 33, 19, 200, 77, 5, 0), 19)
    chk("ma = mb = 0 gives zo", r(100, -100, 0, 0, -9, 0, 0, 3, 0), -9)
    # widths: the largest |acc| fits the RTL's ACC_W = M_BITS + 11 = 19 bits
    worst = 2 * 255 * 255
    chk("|acc| < 2^18 (19-bit signed)", worst < (1 << 18), True)
    # numpy against loops, random
    rng = np.random.default_rng(3)
    for i in range(300):
        a = rng.integers(-128, 128, 20)
        b = rng.integers(-128, 128, 20)
        p = [int(rng.integers(-128, 128)) for _ in range(3)] + \
            [int(rng.integers(0, 256)), int(rng.integers(0, 256)), int(rng.integers(0, 16)),
             int(rng.integers(0, 2))]
        if not np.array_equal(add_ref(a, b, *p), add_naive(a, b, *p)):
            chk(f"ref vs naive #{i}", add_ref(a, b, *p).tolist(), add_naive(a, b, *p))
            break
    else:
        chk("ref vs naive, 300 random vectors with random params", True, True)
    # packed view
    for n in (1, 2, 4, 8):
        cmd, a, b = random_cmd(rng, n, 9, 3)
        aw, bw = pack_act(a, n, cmd.za), pack_act(b, n, cmd.zb)
        got = unpack_act(add_words(aw, bw, cmd, n), cmd.h_in, cmd.w_in, a.shape[2], n)
        want = add_ref(a, b, cmd.za, cmd.zb, cmd.zo, cmd.ma, cmd.mb, cmd.shift, cmd.relu_en)
        chk(f"packed vs tensor N={n}", got, want)
        full = unpack_act(add_words(aw, bw, cmd, n), cmd.h_in, cmd.w_in, cmd.c_words * n, n, with_pad=True)
        pad_ok = bool(np.all(full[..., a.shape[2]:] == max(cmd.zo, cmd.zo)))
        chk(f"  padded lanes hold zo N={n}", pad_ok, True)
        chk(f"  random command legal N={n}", check_command(cmd, 9, 3), 0)
    # legality
    base = AddCmd(2, 2, 1, 0, 0, 0, 100, 1, 0)
    chk("same bank for a and b is legal", check_command(base, 9, 3), 0)
    for over, bit in ((dict(c_words=0), E_DIM), (dict(src2_base=509), E_SRC), (dict(dst_base=509), E_DST),
                      (dict(dst_sel=0), E_SEL), (dict(src2_sel=1), E_SEL), (dict(dst_sel=3), E_SEL)):
        c = AddCmd(**{**base.__dict__, **over})
        chk(f"{ERR_NAMES[bit]} from {over}", check_command(c, 9, 3), 1 << bit)
    chk("E_HOST", check_command(base, 9, 3, host_mode=1), 1 << E_HOST)
    ma, mb, s = scales_to_params(0.02, 0.05, 0.04)
    chk("scales_to_params keeps 8 bits and the ratios", (ma < 256, mb < 256, abs(ma / 2**s - 0.5) < 1 / 256,
                                                         abs(mb / 2**s - 1.25) < 1 / 128), (True,) * 4)

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
