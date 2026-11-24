# Golden model for rtl/pool_layer.sv
#
# maxpool2_ref and gavg_ref are straight NumPy on an (H, W, C) int8 tensor;
# they know nothing about words, lanes or addresses. pool_words() applies them
# to the channel-packed act_buffer image (lane j of group g = channel g*N + j,
# unused lanes included), which is what the hardware sees, and the self-tests
# check that the two views agree and that padded lanes keep the zero point.
#
# rounding for GAVG (contract decision 2): (sum + 2^(S-1)) >> S, arithmetic,
# i.e. round half up, the same convention as requant.

from dataclasses import dataclass

import numpy as np

from conv_layer_model import pack_act, unpack_act, words

M_MAX2, M_GAVG = 1, 2
E_DIM, E_SHAPE, E_SRC, E_DST, E_SEL, E_HOST, E_MODE = 0, 1, 5, 6, 7, 8, 9
ERR_NAMES = {E_DIM: "E_DIM", E_SHAPE: "E_SHAPE", E_SRC: "E_SRC", E_DST: "E_DST",
             E_SEL: "E_SEL", E_HOST: "E_HOST", E_MODE: "E_MODE"}


def maxpool2_ref(x):
    """2x2 max, stride 2, floor on odd sizes. (H, W, C) -> (H//2, W//2, C)"""
    x = np.asarray(x, np.int64)
    h2, w2 = x.shape[0] // 2, x.shape[1] // 2
    x = x[:2 * h2, :2 * w2]
    return x.reshape(h2, 2, w2, 2, -1).max(axis=(1, 3))


def gavg_ref(x):
    """global average, H*W = 2^S, round half up. (H, W, C) -> (1, 1, C)"""
    x = np.asarray(x, np.int64)
    hw = x.shape[0] * x.shape[1]
    s = hw.bit_length() - 1
    assert hw == 1 << s, "GAVG needs h*w to be a power of two"
    total = x.sum(axis=(0, 1))
    out = (total + (1 << (s - 1))) >> s if s else total
    return out.reshape(1, 1, -1)


def pool_ref(x, mode):
    return maxpool2_ref(x) if mode == M_MAX2 else gavg_ref(x)


def naive(x, mode):
    """the same, as plain loops, for the self-test"""
    h, w, c = len(x), len(x[0]), len(x[0][0])
    if mode == M_MAX2:
        return np.array([[[max(int(x[2 * oy + dy][2 * ox + dx][ch]) for dy in (0, 1) for dx in (0, 1))
                           for ch in range(c)] for ox in range(w // 2)] for oy in range(h // 2)])
    s = (h * w).bit_length() - 1
    out = []
    for ch in range(c):
        tot = sum(int(x[y][xx][ch]) for y in range(h) for xx in range(w))
        q = tot + (1 << (s - 1)) if s else tot
        out.append(q // (1 << s))            # floor division == arithmetic shift
    return np.array(out).reshape(1, 1, c)


def out_shape(h, w, mode):
    return (h // 2, w // 2) if mode == M_MAX2 else (1, 1)


@dataclass
class PoolCmd:
    mode: int
    h_in: int
    w_in: int
    c_words: int
    src_sel: int = 0
    dst_sel: int = 1
    src_base: int = 0
    dst_base: int = 0

    def as_dict(self):
        return dict(mode=self.mode, h_in=self.h_in, w_in=self.w_in, c_words=self.c_words,
                    src_sel=self.src_sel, dst_sel=self.dst_sel, src_base=self.src_base,
                    dst_base=self.dst_base)


def check_command(c: PoolCmd, addr_width: int, nbuf: int, host_mode: int = 0) -> int:
    """err_code pool_layer must report (0 = runs)"""
    code = 0
    e_dim = c.h_in == 0 or c.w_in == 0 or c.c_words == 0
    e_mode = c.mode not in (M_MAX2, M_GAVG)
    code |= e_dim << E_DIM
    code |= e_mode << E_MODE
    if not e_dim and not e_mode:
        if c.mode == M_MAX2:
            code |= (c.h_in < 2 or c.w_in < 2) << E_SHAPE
        else:
            hw = c.h_in * c.w_in
            code |= (hw & (hw - 1) != 0) << E_SHAPE
    in_words = c.h_in * c.w_in * c.c_words
    out_words = ((c.h_in // 2) * (c.w_in // 2) if c.mode == M_MAX2 else 1) * c.c_words
    code |= (c.src_base + in_words > (1 << addr_width)) << E_SRC
    code |= (c.dst_base + out_words > (1 << addr_width)) << E_DST
    code |= (c.src_sel == c.dst_sel or c.src_sel >= nbuf or c.dst_sel >= nbuf) << E_SEL
    code |= int(bool(host_mode)) << E_HOST
    return code


def err_names(code):
    return [ERR_NAMES[b] for b in sorted(ERR_NAMES) if code >> b & 1]


def pool_words(in_words, h, w, c_words, n, mode):
    """the act_buffer view: packed input words -> packed output words, every
    lane computed (padded lanes included), exactly what the hardware does"""
    full = unpack_act(in_words, h, w, c_words * n, n, with_pad=True)
    out = pool_ref(full, mode)
    oh, ow = out.shape[:2]
    return pack_act(out, n, 0)              # all c_words*n lanes are real here


def random_cmd(rng, n, addr_width, nbuf, mode=None, max_dim=9, max_cw=4):
    """a random legal command and input tensor (H, W, C) with zero point zp
    in the padded lanes. GAVG picks h*w as a power of two."""
    for _ in range(1000):
        m = mode or int(rng.choice([M_MAX2, M_GAVG]))
        if m == M_MAX2:
            h, w = int(rng.integers(2, max_dim + 1)), int(rng.integers(2, max_dim + 1))
        else:
            s = int(rng.integers(0, 7))
            h = 1 << int(rng.integers(0, s + 1))
            w = (1 << s) // h
        c = int(rng.integers(1, max_cw * n + 1))
        cw = words(c, n)
        src, dst = [int(v) for v in rng.choice(nbuf, 2, replace=False)]
        in_words = h * w * cw
        out_words = ((h // 2) * (w // 2) if m == M_MAX2 else 1) * cw
        if in_words > (1 << addr_width) or out_words > (1 << addr_width):
            continue
        cmd = PoolCmd(m, h, w, cw, src, dst,
                      int(rng.integers(0, (1 << addr_width) - in_words + 1)),
                      int(rng.integers(0, (1 << addr_width) - out_words + 1)))
        zp = int(rng.integers(-128, 128))
        x = rng.integers(-128, 128, (h, w, c))
        return cmd, x, zp
    raise RuntimeError("no legal pool command found")


def estimate_cycles(cmd: PoolCmd, read_latency: int, dim_w: int = 8, cw_w: int = 7) -> int:
    """start to done: DIM_W + CW_W + 1 check edges, then one read per cycle
    (4 per output word for MAX2, h*w per output word for GAVG), then the
    READ_LATENCY of the last read and one edge for the write register"""
    reads = (4 * (cmd.h_in // 2) * (cmd.w_in // 2) * cmd.c_words if cmd.mode == M_MAX2
             else cmd.h_in * cmd.w_in * cmd.c_words)
    return (dim_w + cw_w + 1) + reads + read_latency + 1


if __name__ == "__main__":
    fails = 0

    def chk(name, got, exp):
        global fails
        ok = np.array_equal(got, exp) if isinstance(got, np.ndarray) or isinstance(exp, np.ndarray) \
            else got == exp
        print(f"{'pass' if ok else 'FAIL'} {name}" + ("" if ok else f"\n   got {got}\n   exp {exp}"))
        fails += not ok

    rng = np.random.default_rng(7)
    # hand values
    x = np.array([[[1], [-5]], [[3], [-128]]])
    chk("max of a 2x2", maxpool2_ref(x)[0, 0, 0], 3)
    chk("max of all -128", maxpool2_ref(np.full((2, 2, 1), -128))[0, 0, 0], -128)
    chk("odd sizes floor", maxpool2_ref(np.zeros((5, 3, 2))).shape, (2, 1, 2))
    chk("avg of 1,2 rounds 1.5 up to 2", gavg_ref(np.array([[[1], [2]]]))[0, 0, 0], 2)
    chk("avg of -1,-2 rounds -1.5 up to -1", gavg_ref(np.array([[[-1], [-2]]]))[0, 0, 0], -1)
    chk("avg of -3,0 rounds -1.5 up to -1", gavg_ref(np.array([[[-3], [0]]]))[0, 0, 0], -1)
    chk("avg of one pixel is itself", gavg_ref(np.array([[[-77]]]))[0, 0, 0], -77)
    chk("avg of all 127 stays 127", gavg_ref(np.full((16, 16, 1), 127))[0, 0, 0], 127)
    chk("avg of all -128 stays -128", gavg_ref(np.full((4, 8, 1), -128))[0, 0, 0], -128)
    # numpy against loops
    for i in range(30):
        mode = M_MAX2 if i % 2 else M_GAVG
        if mode == M_MAX2:
            h, w = int(rng.integers(2, 8)), int(rng.integers(2, 8))
        else:
            s = int(rng.integers(0, 7)); h = 1 << int(rng.integers(0, s + 1)); w = (1 << s) // h
        x = rng.integers(-128, 128, (h, w, int(rng.integers(1, 6))))
        chk(f"ref vs loops #{i} mode {mode} {h}x{w}", pool_ref(x, mode), naive(x.tolist(), mode))
    # packed view against the tensor view, padded lanes keep the zero point
    for n in (1, 2, 4, 8):
        for i in range(4):
            cmd, x, zp = random_cmd(rng, n, 10, 2)
            got = pool_words(pack_act(x, n, zp), cmd.h_in, cmd.w_in, cmd.c_words, n, cmd.mode)
            oh, ow = out_shape(cmd.h_in, cmd.w_in, cmd.mode)
            want = pack_act(pool_ref(x, cmd.mode), n, zp)
            chk(f"packed vs tensor N={n} #{i} mode {cmd.mode} {cmd.h_in}x{cmd.w_in} c={x.shape[2]}",
                got, want)
            chk(f"  legal", check_command(cmd, 10, 2), 0)
    # legality, one bit at a time
    base = PoolCmd(M_MAX2, 4, 4, 1, 0, 1, 0, 0)
    for over, bit in ((dict(h_in=0), E_DIM), (dict(mode=3), E_MODE), (dict(mode=0), E_MODE),
                      (dict(h_in=1), E_SHAPE), (dict(mode=M_GAVG, h_in=3, w_in=4), E_SHAPE),
                      (dict(src_base=1024 - 15), E_SRC), (dict(dst_base=1024 - 3), E_DST),
                      (dict(dst_sel=0), E_SEL), (dict(src_sel=2), E_SEL)):
        c = PoolCmd(**{**base.__dict__, **over})
        chk(f"{ERR_NAMES[bit]} from {over}", check_command(c, 10, 2), 1 << bit)
    chk("E_HOST", check_command(base, 10, 2, host_mode=1), 1 << E_HOST)
    chk("GAVG 4x4 legal", check_command(PoolCmd(M_GAVG, 4, 4, 2, 0, 1), 10, 2), 0)

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
