# Golden model for rtl/conv_layer.sv
#
# conv_ref is a straight NumPy quantized convolution. It knows nothing about
# tiles, slices, memory words or the zero-point fold: it pads the image with
# the input zero point, computes sum((x - zp_in) * w) + bias in int64 and
# requantizes every value with requant_model.requant_ref, the same bit-exact
# reference the requant block is checked against.
#
# The packer turns tensors into the act_buffer / B / P memory images that the
# contract defines, and that is the only place the zero-point fold happens.
# run_from_images recomputes the layer from those images the way the hardware
# walks them, so the self-tests can show that packer + fold == conv_ref.

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from requant_model import requant_ref

BIAS_WIDTH, M_WIDTH, SHIFT_WIDTH = 32, 16, 5   # gemm_tile's requant widths
DATA_WIDTH = 8

# err_code bits, contract order
E_DIM, E_SHAPE, E_K, E_BMEM, E_PMEM, E_SRC, E_DST, E_SEL, E_HOST = range(9)
ERR_NAMES = ["E_DIM", "E_SHAPE", "E_K", "E_BMEM", "E_PMEM", "E_SRC", "E_DST", "E_SEL", "E_HOST"]


def out_size(size_in: int, k: int, stride: int, pad: int) -> int:
    """Output length along one axis; 0 when the kernel does not fit at all."""
    span = size_in + 2 * pad
    return 0 if k > span else (span - k) // stride + 1


def words(channels: int, n: int) -> int:
    return -(-channels // n)


# ----------------------------------------------------------------------------
# reference convolution
# ----------------------------------------------------------------------------

_requant_vec = np.frompyfunc(
    lambda acc, bias, mult, shift, zp, relu: requant_ref(int(acc), int(bias), int(mult),
                                                         int(shift), int(zp), int(relu)), 6, 1)


def requant_array(acc, bias, mult, shift, zp_out, relu_en):
    """requant_ref applied elementwise; acc is (..., cout), the rest per channel.
    bias is added inside requant_ref, before its 32-bit saturation, as in RTL."""
    out = _requant_vec(acc, bias, mult, shift, zp_out, int(relu_en))
    return out.astype(np.int64)


def conv_ref(x, w, bias, mult, shift, zp_in, zp_out, relu_en, stride, pad):
    """x: (H, W, Cin) int8 values, w: (KH, KW, Cin, Cout), per-channel bias,
    mult, shift. Returns (H_out, W_out, Cout) int8 values as int64."""
    x = np.asarray(x, np.int64)
    w = np.asarray(w, np.int64)
    h, wd, cin = x.shape
    kh, kw, _, cout = w.shape
    ho, wo = out_size(h, kh, stride, pad), out_size(wd, kw, stride, pad)
    xp = np.full((h + 2 * pad, wd + 2 * pad, cin), zp_in, np.int64)
    xp[pad:pad + h, pad:pad + wd] = x
    xc = xp - zp_in
    acc = np.zeros((ho, wo, cout), np.int64)
    for ky in range(kh):
        for kx in range(kw):
            patch = xc[ky:ky + stride * (ho - 1) + 1:stride, kx:kx + stride * (wo - 1) + 1:stride]
            acc += np.einsum("yxi,io->yxo", patch, w[ky, kx])
    return requant_array(acc, np.asarray(bias), np.asarray(mult), np.asarray(shift),
                         zp_out, relu_en)


def conv_naive(x, w, bias, mult, shift, zp_in, zp_out, relu_en, stride, pad):
    """The same layer as seven plain loops (batch of one), for the self-test."""
    h, wd, cin = len(x), len(x[0]), len(x[0][0])
    kh, kw, cout = len(w), len(w[0]), len(w[0][0][0])
    ho, wo = out_size(h, kh, stride, pad), out_size(wd, kw, stride, pad)
    out = [[[0] * cout for _ in range(wo)] for _ in range(ho)]
    for b in range(1):
        for oy in range(ho):
            for ox in range(wo):
                for oc in range(cout):
                    acc = 0
                    for ky in range(kh):
                        for kx in range(kw):
                            for ic in range(cin):
                                iy, ix = oy * stride + ky - pad, ox * stride + kx - pad
                                v = int(x[iy][ix][ic]) if 0 <= iy < h and 0 <= ix < wd else zp_in
                                acc += (v - zp_in) * int(w[ky][kx][ic][oc])
                    out[oy][ox][oc] = requant_ref(acc, int(bias[oc]), int(mult[oc]),
                                                  int(shift[oc]), zp_out, relu_en)
    return np.array(out, np.int64)


# ----------------------------------------------------------------------------
# layer description and memory images
# ----------------------------------------------------------------------------

@dataclass
class Layer:
    """One conv layer: tensors plus where they live. Command field names match
    the RTL ports."""
    x: np.ndarray                 # (h_in, w_in, cin)
    w: np.ndarray                 # (kh, kw, cin, cout)
    bias: np.ndarray              # (cout,) before the zero-point fold
    mult: np.ndarray
    shift: np.ndarray
    zp_in: int = 0
    zp_out: int = 0
    relu_en: int = 0
    stride: int = 1
    pad: int = 0
    src_sel: int = 0
    dst_sel: int = 1
    src_base: int = 0
    dst_base: int = 0
    b_base: int = 0
    p_base: int = 0

    @property
    def cin(self): return self.x.shape[2]

    @property
    def cout(self): return self.w.shape[3]

    def command(self, n: int) -> dict:
        h, wd = self.x.shape[:2]
        kh, kw = self.w.shape[:2]
        return dict(h_in=h, w_in=wd,
                    h_out=out_size(h, kh, self.stride, self.pad),
                    w_out=out_size(wd, kw, self.stride, self.pad),
                    cin_words=words(self.cin, n), cout_words=words(self.cout, n),
                    kh=kh, kw=kw, stride=self.stride, pad=self.pad,
                    zp_in=self.zp_in, zp_out=self.zp_out, relu_en=self.relu_en,
                    src_sel=self.src_sel, dst_sel=self.dst_sel,
                    src_base=self.src_base, dst_base=self.dst_base,
                    b_base=self.b_base, p_base=self.p_base)

    def reference(self):
        return conv_ref(self.x, self.w, self.bias, self.mult, self.shift, self.zp_in,
                        self.zp_out, self.relu_en, self.stride, self.pad)


def pack_word(lanes, dw: int = DATA_WIDTH) -> int:
    """lane 0 in the LSBs, two's complement per lane"""
    word = 0
    for i, v in enumerate(lanes):
        word |= (int(v) & ((1 << dw) - 1)) << (i * dw)
    return word


def unpack_word(word: int, n: int, dw: int = DATA_WIDTH) -> List[int]:
    out = []
    for i in range(n):
        v = (word >> (i * dw)) & ((1 << dw) - 1)
        out.append(v - (1 << dw) if v >> (dw - 1) else v)
    return out


def pack_act(x, n: int, pad_value: int) -> List[int]:
    """(H, W, C) -> words, word (y*W + x)*cw + g holds channels g*N .. g*N+N-1.
    Lanes past C hold pad_value (the zero point), as the contract says."""
    h, wd, c = x.shape
    cw = words(c, n)
    xp = np.full((h, wd, cw * n), pad_value, np.int64)
    xp[:, :, :c] = x
    return [pack_word(xp[y, xx, g * n:(g + 1) * n]) for y in range(h) for xx in range(wd)
            for g in range(cw)]


def unpack_act(img: List[int], h: int, wd: int, c: int, n: int, with_pad: bool = False):
    """Inverse of pack_act. with_pad returns all cw*N lanes."""
    cw = words(c, n)
    out = np.zeros((h, wd, cw * n), np.int64)
    for y in range(h):
        for xx in range(wd):
            for g in range(cw):
                out[y, xx, g * n:(g + 1) * n] = unpack_word(img[(y * wd + xx) * cw + g], n)
    return out if with_pad else out[:, :, :c]


def k_of(kh: int, kw: int, cin_words: int, n: int) -> int:
    return kh * kw * cin_words * n


def pack_b(w, n: int) -> List[int]:
    """B image starting at b_base: group og, row k = ((ky*kw + kx)*cw + g)*N + j
    at og*K + k, byte c = weight from input channel g*N + j to output channel
    og*N + c. Padded channels in either direction get weight 0."""
    kh, kw, cin, cout = w.shape
    cw, ow = words(cin, n), words(cout, n)
    wp = np.zeros((kh, kw, cw * n, ow * n), np.int64)
    wp[:, :, :cin, :cout] = w
    img = []
    for og in range(ow):
        for ky in range(kh):
            for kx in range(kw):
                for ic in range(cw * n):           # = g*N + j, in k order
                    img.append(pack_word(wp[ky, kx, ic, og * n:(og + 1) * n]))
    return img


def fold_bias(w, bias, zp_in: int) -> np.ndarray:
    """bias - zp_in * sum_k w[k][c]: the array multiplies raw x, so the input
    zero point is taken out through the bias. exact in integers."""
    return np.asarray(bias, np.int64) - zp_in * np.asarray(w, np.int64).sum(axis=(0, 1, 2))


def pack_param(bias: int, mult: int, shift: int) -> int:
    lo, hi = -(1 << (BIAS_WIDTH - 1)), (1 << (BIAS_WIDTH - 1)) - 1
    if not lo <= bias <= hi:
        raise ValueError(f"folded bias {bias} does not fit {BIAS_WIDTH} bits")
    if not 0 <= mult < (1 << M_WIDTH) or not 0 <= shift < (1 << SHIFT_WIDTH):
        raise ValueError(f"mult={mult} or shift={shift} out of range")
    return ((shift << (BIAS_WIDTH + M_WIDTH)) | (mult << BIAS_WIDTH)
            | (bias & ((1 << BIAS_WIDTH) - 1)))


def unpack_param(word: int):
    bias = word & ((1 << BIAS_WIDTH) - 1)
    bias -= (1 << BIAS_WIDTH) if bias >> (BIAS_WIDTH - 1) else 0
    mult = (word >> BIAS_WIDTH) & ((1 << M_WIDTH) - 1)
    shift = word >> (BIAS_WIDTH + M_WIDTH)
    return bias, mult, shift


def pack_p(layer: Layer, n: int) -> List[int]:
    """P image starting at p_base: channel og*N + c. padded channels {0,0,0},
    which requantizes to exactly zp_out."""
    ow = words(layer.cout, n)
    fb = fold_bias(layer.w, layer.bias, layer.zp_in)
    img = []
    for oc in range(ow * n):
        if oc < layer.cout:
            img.append(pack_param(int(fb[oc]), int(layer.mult[oc]), int(layer.shift[oc])))
        else:
            img.append(pack_param(0, 0, 0))
    return img


def output_region(layer: Layer, n: int):
    """(first word, number of words) of the output in the destination bank"""
    c = layer.command(n)
    return layer.dst_base, c["h_out"] * c["w_out"] * c["cout_words"]


def expected_output_words(layer: Layer, n: int) -> List[int]:
    """The destination words the hardware must leave: conv_ref, padded lanes zp_out."""
    return pack_act(layer.reference(), n, layer.zp_out)


def run_from_images(act: List[int], b_img: List[int], p_img: List[int], cmd: dict,
                    n: int) -> List[int]:
    """Recompute the layer the way conv_layer walks memory: K slices per pixel
    from the act image (padding = zp_in in every lane), the weight row b_base +
    og*K + k, requant with P word p_base + og*N + c. Returns output words in
    destination order. Independent of conv_ref; the self-test compares them."""
    h, wd, cw = cmd["h_in"], cmd["w_in"], cmd["cin_words"]
    ho, wo, ow = cmd["h_out"], cmd["w_out"], cmd["cout_words"]
    kh, kw, s, p = cmd["kh"], cmd["kw"], cmd["stride"], cmd["pad"]
    K = k_of(kh, kw, cw, n)
    out = [0] * (ho * wo * ow)
    for og in range(ow):
        rows = [unpack_word(b_img[cmd["b_base"] + og * K + k], n) for k in range(K)]
        params = [unpack_param(p_img[cmd["p_base"] + og * n + c]) for c in range(n)]
        for pix in range(ho * wo):
            oy, ox = divmod(pix, wo)
            acc = [0] * n
            for ky in range(kh):
                for kx in range(kw):
                    iy, ix = oy * s + ky - p, ox * s + kx - p
                    inside = 0 <= iy < h and 0 <= ix < wd
                    for g in range(cw):
                        lanes = (unpack_word(act[cmd["src_base"] + (iy * wd + ix) * cw + g], n)
                                 if inside else [cmd["zp_in"]] * n)
                        for j in range(n):
                            row = rows[((ky * kw + kx) * cw + g) * n + j]
                            for c in range(n):
                                acc[c] += lanes[j] * row[c]
            q = [requant_ref(acc[c], params[c][0], params[c][1], params[c][2],
                             cmd["zp_out"], cmd["relu_en"]) for c in range(n)]
            out[pix * ow + og] = pack_word(q)
    return out


# ----------------------------------------------------------------------------
# legality, exactly as the RTL check phase
# ----------------------------------------------------------------------------

@dataclass
class HwParams:
    n: int = 8
    addr_width: int = 12
    nbuf: int = 2
    k_max: int = 1024
    b_addr_w: int = 13
    p_addr_w: int = 6
    dim_w: int = 8
    cw_w: int = 7


def check_command(cmd: dict, hw: HwParams, host_mode: int = 0) -> int:
    """err_code the RTL must report for this command (0 = runs)."""
    n = hw.n
    code = 0
    dims = ("h_in", "w_in", "h_out", "w_out", "cin_words", "cout_words", "kh", "kw", "stride")
    e_dim = any(cmd[d] == 0 for d in dims)
    code |= e_dim << E_DIM
    if not e_dim:
        bad = (cmd["h_out"] != out_size(cmd["h_in"], cmd["kh"], cmd["stride"], cmd["pad"]) or
               cmd["w_out"] != out_size(cmd["w_in"], cmd["kw"], cmd["stride"], cmd["pad"]))
        code |= bad << E_SHAPE
    K = k_of(cmd["kh"], cmd["kw"], cmd["cin_words"], n)
    code |= (K > hw.k_max) << E_K
    code |= (cmd["b_base"] + cmd["cout_words"] * K > (1 << hw.b_addr_w)) << E_BMEM
    code |= (cmd["p_base"] + cmd["cout_words"] * n > (1 << hw.p_addr_w)) << E_PMEM
    code |= (cmd["src_base"] + cmd["h_in"] * cmd["w_in"] * cmd["cin_words"]
             > (1 << hw.addr_width)) << E_SRC
    code |= (cmd["dst_base"] + cmd["h_out"] * cmd["w_out"] * cmd["cout_words"]
             > (1 << hw.addr_width)) << E_DST
    code |= (cmd["src_sel"] == cmd["dst_sel"] or cmd["src_sel"] >= hw.nbuf
             or cmd["dst_sel"] >= hw.nbuf) << E_SEL
    code |= int(bool(host_mode)) << E_HOST
    return code


def err_names(code: int) -> List[str]:
    return [ERR_NAMES[i] for i in range(9) if code >> i & 1]


# ----------------------------------------------------------------------------
# random layers
# ----------------------------------------------------------------------------

def requant_params(rng: np.random.Generator, cout: int, k_eff: int):
    """mult/shift that land a typical accumulator of k_eff terms near +-64, so
    outputs spread over the int8 range instead of all clamping."""
    est = max(1.0, math.sqrt(k_eff) * 60 * 60)
    mult = rng.integers(1 << 12, 1 << M_WIDTH, cout)
    shift = np.array([min(31, max(0, round(math.log2(m * est / 64)))) for m in mult])
    bias = rng.integers(-4000, 4001, cout)
    return bias, mult, shift


def random_layer(rng: np.random.Generator, hw: HwParams, max_dim: int = 9,
                 max_cin: Optional[int] = None, max_cout: Optional[int] = None,
                 **force) -> Layer:
    """A random legal layer for hw. force overrides any Layer/shape field."""
    n = hw.n
    for _ in range(1000):
        kh = force.get("kh", int(rng.integers(1, 4)))
        kw = force.get("kw", int(rng.integers(1, 4)))
        stride = force.get("stride", int(rng.integers(1, 3)))
        pad = force.get("pad", int(rng.integers(0, min(kh, kw))))
        h = force.get("h_in", int(rng.integers(max(1, kh - 2 * pad), max_dim + 1)))
        wd = force.get("w_in", int(rng.integers(max(1, kw - 2 * pad), max_dim + 1)))
        cin = force.get("cin", int(rng.integers(1, (max_cin or 2 * n) + 1)))
        cout = force.get("cout", int(rng.integers(1, (max_cout or 2 * n) + 1)))
        if out_size(h, kh, stride, pad) == 0 or out_size(wd, kw, stride, pad) == 0:
            continue
        x = rng.integers(-128, 128, (h, wd, cin))
        w = rng.integers(-128, 128, (kh, kw, cin, cout))
        bias, mult, shift = requant_params(rng, cout, kh * kw * cin)
        layer = Layer(x=x, w=w, bias=bias, mult=mult, shift=shift,
                      zp_in=force.get("zp_in", int(rng.integers(-20, 21))),
                      zp_out=force.get("zp_out", int(rng.integers(-20, 21))),
                      relu_en=force.get("relu_en", int(rng.integers(0, 2))),
                      stride=stride, pad=pad,
                      src_sel=force.get("src_sel", 0), dst_sel=force.get("dst_sel", 1))
        cmd = layer.command(n)
        K = k_of(kh, kw, cmd["cin_words"], n)
        in_words = h * wd * cmd["cin_words"]
        out_words = cmd["h_out"] * cmd["w_out"] * cmd["cout_words"]
        # random bases that still fit
        if in_words > (1 << hw.addr_width) or out_words > (1 << hw.addr_width):
            continue
        if cmd["cout_words"] * K > (1 << hw.b_addr_w) or cmd["cout_words"] * n > (1 << hw.p_addr_w):
            continue
        layer.src_base = force.get("src_base", int(rng.integers(0, (1 << hw.addr_width) - in_words + 1)))
        layer.dst_base = force.get("dst_base", int(rng.integers(0, (1 << hw.addr_width) - out_words + 1)))
        layer.b_base = force.get("b_base", int(rng.integers(0, (1 << hw.b_addr_w) - cmd["cout_words"] * K + 1)))
        layer.p_base = force.get("p_base", int(rng.integers(0, (1 << hw.p_addr_w) - cmd["cout_words"] * n + 1)))
        if check_command(layer.command(n), hw) == 0:
            return layer
    raise RuntimeError("no legal random layer found; constraints too tight")


# ----------------------------------------------------------------------------
# cycle estimate from the block latency formulas (paper, for the comparison)
# ----------------------------------------------------------------------------

def estimate_cycles(cmd: dict, n: int, read_latency: int) -> int:
    """start to done from the documented gemm_tile / im2col_reader latencies.
    im2col: first group valid N+L+2 after start, each later group N+L+2 after
    the previous group's last take; N slices per group, one per cycle. gemm:
    with no stalls the last beat is L+K+3N+4 after start, and every cycle the
    feeder is late adds one. the stall per group is the N+L+2 refill minus the
    L cycles the B read is already ahead. plus the conv_layer overheads that
    are not in either block: 4 check-phase edges per layer, and per tile 3
    edges after gemm_tile's last beat is valid: the write-back register
    captures it, the write lands (the sequencer sees tlast there), and the
    next tile's start is accepted from S_LAUNCH.

    the first version of this counted 2 per-tile edges (it missed that the
    write itself is an edge after the capture) and came out exactly 1 cycle
    per tile short of simulation at every N and L; the block formulas
    themselves composed exactly."""
    L = read_latency
    K = k_of(cmd["kh"], cmd["kw"], cmd["cin_words"], n)
    groups = K // n
    tiles = -(-cmd["h_out"] * cmd["w_out"] // n) * cmd["cout_words"]
    feed_stall = groups * (n + L + 2) - L
    tile = (L + K + 3 * n + 4) + feed_stall + 3
    return 4 + tiles * tile


def gemm_only_cycles(cmd: dict, n: int, read_latency: int) -> int:
    """what the same layer would take if the feeder never stalled gemm_tile
    (the pipelined reader's target): gemm_tile's own L + K + 3N + 2 start to
    start, tiles back to back. the gap to estimate_cycles is the reference
    reader's refill cost plus the serialization."""
    L = read_latency
    K = k_of(cmd["kh"], cmd["kw"], cmd["cin_words"], n)
    tiles = -(-cmd["h_out"] * cmd["w_out"] // n) * cmd["cout_words"]
    return tiles * (L + K + 3 * n + 2)


# ----------------------------------------------------------------------------
# self-tests
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    fails = 0

    def chk(name, got, exp):
        global fails
        ok = (np.array_equal(got, exp) if isinstance(got, np.ndarray) or isinstance(exp, np.ndarray)
              else got == exp)
        print(f"{'pass' if ok else 'FAIL'} {name}")
        if not ok:
            print(f"     got={got}\n     exp={exp}")
            fails += 1

    rng = np.random.default_rng(2026)

    # hand-computed: 1x1 conv, one channel, identity requant
    one = conv_ref(np.array([[[3]]]), np.array([[[[2]]]]), [1], [1], [0], 0, 0, 0, 1, 0)
    chk("1x1 hand value 3*2+1", int(one[0, 0, 0]), 7)
    # zp_in = 5, pad 1, 3x3 of ones over a 1x1 image of 5: everything is (5-5)*1 = 0
    z = conv_ref(np.array([[[5]]]), np.ones((3, 3, 1, 1), int), [0], [1], [0], 5, 0, 0, 1, 1)
    chk("padding with zp_in contributes zero", int(z[0, 0, 0]), 0)
    # relu clamps at zp_out
    r = conv_ref(np.array([[[-10]]]), np.array([[[[1]]]]), [0], [1], [0], 0, 3, 1, 1, 0)
    chk("relu floors at zp_out", int(r[0, 0, 0]), 3)
    chk("out_size 5,k3,s2,p0", out_size(5, 3, 2, 0), 2)
    chk("out_size 4,k3,s1,p1", out_size(4, 3, 1, 1), 4)
    chk("out_size kernel too big", out_size(1, 3, 1, 0), 0)

    # word packing round trips
    vals = [-128, -1, 0, 1, 127, 55, -77, 3]
    chk("pack/unpack word", unpack_word(pack_word(vals), 8), vals)
    p = pack_param(-123456, 40000, 17)
    chk("pack/unpack param", unpack_param(p), (-123456, 40000, 17))

    for n in (1, 2, 4, 8):
        for c in (1, n, n + 1, 3 * n - 1):
            x = rng.integers(-128, 128, (3, 5, c))
            img = pack_act(x, n, -7)
            chk(f"act round trip N={n} C={c}", unpack_act(img, 3, 5, c, n), x)
            full = unpack_act(img, 3, 5, c, n, with_pad=True)
            chk(f"  padded lanes hold the zero point N={n} C={c}",
                bool(np.all(full[:, :, c:] == -7)), True)
            chk(f"  word count N={n} C={c}", len(img), 15 * words(c, n))

    # conv_ref against the independent naive loops
    for i in range(25):
        kh, kw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        stride, pad = int(rng.integers(1, 4)), int(rng.integers(0, 3))
        h, wd = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        if out_size(h, kh, stride, pad) == 0 or out_size(wd, kw, stride, pad) == 0:
            continue
        cin, cout = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        x = rng.integers(-128, 128, (h, wd, cin))
        w = rng.integers(-128, 128, (kh, kw, cin, cout))
        bias, mult, shift = requant_params(rng, cout, kh * kw * cin)
        if i % 5 == 0:   # saturating params: huge mult, no shift
            mult, shift = np.full(cout, (1 << M_WIDTH) - 1), np.zeros(cout, int)
        zi, zo, relu = int(rng.integers(-128, 128)), int(rng.integers(-128, 128)), i % 2
        a = conv_ref(x, w, bias, mult, shift, zi, zo, relu, stride, pad)
        b = conv_naive(x.tolist(), w.tolist(), bias, mult, shift, zi, zo, relu, stride, pad)
        chk(f"conv_ref vs naive #{i} k{kh}x{kw} s{stride} p{pad} {h}x{wd}x{cin}->{cout}", a, b)

    # packer + zero-point fold + hardware walk == conv_ref
    for n in (1, 2, 4, 8):
        hw = HwParams(n=n, addr_width=10, p_addr_w=max(6, (4 * n).bit_length()))
        for i in range(4):
            layer = random_layer(rng, hw, max_dim=6)
            cmd = layer.command(n)
            act = [0] * (1 << hw.addr_width)
            for a, v in enumerate(pack_act(layer.x, n, layer.zp_in)):
                act[layer.src_base + a] = v
            b_img = [0] * (1 << hw.b_addr_w)
            for a, v in enumerate(pack_b(layer.w, n)):
                b_img[layer.b_base + a] = v
            p_img = [0] * (1 << hw.p_addr_w)
            for a, v in enumerate(pack_p(layer, n)):
                p_img[layer.p_base + a] = v
            got = run_from_images(act, b_img, p_img, cmd, n)
            chk(f"images vs conv_ref N={n} #{i} {cmd['kh']}x{cmd['kw']} s{cmd['stride']} "
                f"p{cmd['pad']} cw{cmd['cin_words']} ow{cmd['cout_words']}",
                got, expected_output_words(layer, n))
            chk(f"  random layer is legal", check_command(cmd, hw), 0)

    # legality checks, one per bit
    hw = HwParams(n=4, addr_width=8, nbuf=3, k_max=64, b_addr_w=8, p_addr_w=4)
    base = dict(h_in=4, w_in=4, h_out=4, w_out=4, cin_words=1, cout_words=1, kh=3, kw=3,
                stride=1, pad=1, zp_in=0, zp_out=0, relu_en=0, src_sel=0, dst_sel=1,
                src_base=0, dst_base=0, b_base=0, p_base=0)
    chk("base command legal", check_command(base, hw), 0)
    cases = [
        (dict(stride=0), E_DIM), (dict(cout_words=0), E_DIM),
        (dict(h_out=3), E_SHAPE), (dict(w_out=5), E_SHAPE), (dict(pad=0, kh=5), E_SHAPE),
        (dict(cin_words=2), E_K),
        (dict(b_base=256 - 35), E_BMEM),
        (dict(p_base=13), E_PMEM),
        (dict(src_base=241), E_SRC),
        (dict(dst_base=241), E_DST),
        (dict(dst_sel=0), E_SEL), (dict(src_sel=3), E_SEL),
    ]
    for over, bit in cases:
        chk(f"{ERR_NAMES[bit]} from {over}", check_command({**base, **over}, hw), 1 << bit)
    chk("E_HOST", check_command(base, hw, host_mode=1), 1 << E_HOST)
    chk("several bits at once", check_command({**base, "dst_sel": 0, "stride": 0}, hw),
        (1 << E_SEL) | (1 << E_DIM))

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
