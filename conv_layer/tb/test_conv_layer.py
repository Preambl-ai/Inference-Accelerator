""" cocotb tb for rtl/conv_layer.sv

Everything goes through the top-level ports: the input image, B and P are
loaded through the host write ports, the layer runs, and every word of the
banks it could touch is read back through the host read port. The expected
contents come from conv_layer_model.conv_ref (a straight NumPy conv) through
the packer; nothing here reuses the per-block models.

The harness keeps a mirror of every act_buffer bank. Each bank starts filled
with random junk (so a wrong read gives a wrong answer, not an X that happens
to be masked) and the mirror is only updated with outputs the contract says
must be written. Comparing whole banks against the mirror therefore checks
both the output region and that nothing outside it changed.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge
import numpy as np

from conv_layer_model import (gemm_only_cycles, E_BMEM, E_DIM, E_DST, E_HOST, E_K, E_PMEM, E_SEL, E_SHAPE,
                              E_SRC, ERR_NAMES, HwParams, Layer, check_command,
                              err_names, estimate_cycles, expected_output_words, k_of,
                              out_size, pack_act, pack_b, pack_p, random_layer,
                              requant_params, words)

CMD_FIELDS = ("h_in", "w_in", "h_out", "w_out", "cin_words", "cout_words", "kh", "kw",
              "stride", "pad", "zp_in", "zp_out", "relu_en", "src_sel", "dst_sel",
              "src_base", "dst_base", "b_base", "p_base")
SIGNED8 = ("zp_in", "zp_out")


def resolve(sig):
    """int value, or None if any bit is X/Z"""
    raw = str(sig.value)
    if "x" in raw.lower() or "z" in raw.lower():
        return None
    return int(sig.value)


class ConvHarness:

    def __init__(self, dut):
        self.dut = dut
        g = lambda name: int(getattr(dut, name).value)
        self.n = g("N")
        self.lat = g("READ_LATENCY")
        self.hw = HwParams(n=self.n, addr_width=g("ADDR_WIDTH"), nbuf=g("NBUF"),
                           k_max=g("K_MAX"), b_addr_w=g("B_ADDR_W"), p_addr_w=g("P_ADDR_W"),
                           dim_w=g("DIM_W"), cw_w=g("CW_W"))
        self.sel_w = g("SEL_WIDTH")
        self.depth = 1 << self.hw.addr_width
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = np.random.default_rng(self.seed)
        self.mem = [None] * self.hw.nbuf          # mirror of each bank
        self.cycle_log = []

    def where(self):
        return (f"(reproduce: make MODULE_NAME=conv_layer N={self.n} "
                f"READ_LATENCY={self.lat} SEED={self.seed})")

    # -- setup -------------------------------------------------------------
    async def start(self):
        d = self.dut
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        self.idle_inputs()
        d.reset.value = 1
        for _ in range(3):
            await RisingEdge(d.clk)
        await FallingEdge(d.clk)
        d.reset.value = 0
        d._log.info(f"N={self.n} READ_LATENCY={self.lat} {self.hw} SEED={self.seed}")

    def idle_inputs(self):
        d = self.dut
        d.start.value = 0
        for f in CMD_FIELDS:
            getattr(d, f).value = 0
        for s in ("b_we", "b_waddr", "b_wdata", "p_we", "p_waddr", "p_wdata", "host_mode",
                  "host_we", "host_waddr", "host_wdata", "host_wr_sel", "host_rd_en",
                  "host_rd_addr", "host_rd_sel"):
            getattr(d, s).value = 0

    def junk(self, count):
        return [int(v) for v in self.rng.integers(0, 1 << (8 * self.n), count, dtype=np.uint64)] \
            if 8 * self.n < 64 else [random.Random(int(self.rng.integers(2**31))).getrandbits(8 * self.n)
                                     for _ in range(count)]

    # -- memory through the host ports --------------------------------------
    async def host_write(self, sel, base, data):
        """host_mode write burst, one word per cycle; updates the mirror"""
        d = self.dut
        d.host_mode.value = 1
        d.host_we.value = 1
        d.host_wr_sel.value = sel
        for i, w in enumerate(data):
            d.host_waddr.value = base + i
            d.host_wdata.value = w
            await FallingEdge(d.clk)
            self.mem[sel][base + i] = w
        d.host_we.value = 0
        d.host_mode.value = 0

    async def fill_bank(self, sel, data=None):
        if self.mem[sel] is None:
            self.mem[sel] = [None] * self.depth
        await self.host_write(sel, 0, data if data is not None else self.junk(self.depth))

    async def read_words(self, sel, base, count):
        """host read port, rd_en held high. a read sampled on an edge shows up
        READ_LATENCY advancing edges later, counting that edge (act_buffer
        contract; checked by test_host_readback)"""
        d = self.dut
        out = []
        d.host_rd_en.value = 1
        d.host_rd_sel.value = sel
        for i in range(count + self.lat - 1):
            d.host_rd_addr.value = base + min(i, count - 1)
            await RisingEdge(d.clk)
            await ReadOnly()
            if i >= self.lat - 1:
                out.append(resolve(d.host_rd_data))
            await FallingEdge(d.clk)
        d.host_rd_en.value = 0
        return out

    async def check_bank(self, sel, what=""):
        got = await self.read_words(sel, 0, self.depth)
        bad = [a for a in range(self.depth) if got[a] != self.mem[sel][a]]
        assert not bad, (f"bank {sel} {what}: {len(bad)} words differ, first at "
                         f"{[(a, hex(got[a]) if got[a] is not None else 'X', hex(self.mem[sel][a])) for a in bad[:4]]} "
                         f"{self.where()}")

    async def write_b(self, base, data):
        d = self.dut
        d.b_we.value = 1
        for i, w in enumerate(data):
            d.b_waddr.value = base + i
            d.b_wdata.value = w
            await FallingEdge(d.clk)
        d.b_we.value = 0

    async def write_p(self, base, data):
        d = self.dut
        d.p_we.value = 1
        for i, w in enumerate(data):
            d.p_waddr.value = base + i
            d.p_wdata.value = w
            await FallingEdge(d.clk)
        d.p_we.value = 0

    async def load_layer(self, layer: Layer, image=True):
        """input image into its bank (the rest of the bank keeps its junk), B, P"""
        n = self.n
        if image:
            await self.host_write(layer.src_sel, layer.src_base, pack_act(layer.x, n, layer.zp_in))
        await self.write_b(layer.b_base, pack_b(layer.w, n))
        await self.write_p(layer.p_base, pack_p(layer, n))

    # -- running a command --------------------------------------------------
    def drive_cmd(self, cmd):
        d = self.dut
        for f in CMD_FIELDS:
            v = int(cmd[f])
            getattr(d, f).value = v & 0xFF if f in SIGNED8 else v

    async def run(self, cmd, host_mode=0, reset_at=None, max_cycles=None):
        """Start cmd and wait for done or err. Edges are counted from the one
        that accepts start. reset_at = t asserts reset for edge t instead and
        returns there. Returns dict(kind=done|err|reset, cycles, err_code)."""
        d = self.dut
        assert int(d.ready.value) == 1, "harness: DUT not ready before start"
        self.drive_cmd(cmd)
        d.host_mode.value = host_mode
        d.start.value = 1
        await RisingEdge(d.clk)
        await ReadOnly()
        assert int(d.busy.value) == 1 and int(d.ready.value) == 0, "start was not accepted"
        await FallingEdge(d.clk)
        d.start.value = 0
        d.host_mode.value = 0
        limit = max_cycles or (2 * estimate_cycles(cmd, self.n, self.lat) + 200)
        t = 0
        # tile launches, counted on the sequencer's tile_start (one cycle per
        # tile). a wasted all-masked tile changes no memory, only this count
        # and the cycle total, so it is checked directly.
        tiles = 0
        while True:
            if reset_at is not None and t + 1 == reset_at:
                d.reset.value = 1
            await RisingEdge(d.clk)
            t += 1
            await ReadOnly()
            done, err = int(d.done.value), int(d.err.value)
            tiles += int(d.tile_start.value)
            if reset_at is not None and t == reset_at:
                assert (int(d.ready.value), int(d.busy.value), done, err) == (1, 0, 0, 0), \
                    f"after reset: ready/busy/done/err = {d.ready.value}/{d.busy.value}/{done}/{err}"
                await FallingEdge(d.clk)
                d.reset.value = 0
                return dict(kind="reset", cycles=t, err_code=None, tiles=tiles)
            assert not (done and err), "done and err together"
            if done or err:
                assert int(d.ready.value) == 1 and int(d.busy.value) == 0, \
                    "ready must return with done/err"
                code = int(d.err_code.value)
                assert (code == 0) == bool(done), f"err_code={code:#x} with done={done}"
                await FallingEdge(d.clk)
                # pulses are one cycle long
                await RisingEdge(d.clk)
                await ReadOnly()
                assert int(d.done.value) == 0 and int(d.err.value) == 0, "done/err longer than a cycle"
                assert int(d.err_code.value) == code, "err_code must hold until the next start"
                await FallingEdge(d.clk)
                return dict(kind="done" if done else "err", cycles=t, err_code=code, tiles=tiles)
            assert int(d.busy.value) == 1, f"busy fell at edge {t} without done or err"
            assert t < limit, f"no done/err after {t} edges {self.where()}"
            await FallingEdge(d.clk)

    async def run_layer(self, layer: Layer, name, check_banks=True):
        """Load, run, check every bank word against the mirror. Returns cycles."""
        n = self.n
        cmd = layer.command(n)
        assert check_command(cmd, self.hw) == 0, f"stimulus bug: {err_names(check_command(cmd, self.hw))}"
        res = await self.run(cmd)
        assert res["kind"] == "done", f"{name}: layer was rejected: {err_names(res['err_code'])}"
        exp_tiles = -(-cmd["h_out"] * cmd["w_out"] // n) * cmd["cout_words"]
        assert res["tiles"] == exp_tiles, \
            f"{name}: {res['tiles']} tiles launched, ceil(pixels/N)*cout_words = {exp_tiles}"
        for a, w in enumerate(expected_output_words(layer, n)):
            self.mem[layer.dst_sel][layer.dst_base + a] = w
        if check_banks:
            for sel in sorted({layer.src_sel, layer.dst_sel}):
                await self.check_bank(sel, f"after {name}")
        est = estimate_cycles(cmd, n, self.lat)
        tiles = -(-cmd["h_out"] * cmd["w_out"] // n) * cmd["cout_words"]
        self.cycle_log.append((name, cmd, tiles, res["cycles"], est))
        self.dut._log.info(
            f"CYCLES {name}: N={n} L={self.lat} {cmd['h_in']}x{cmd['w_in']}->"
            f"{cmd['h_out']}x{cmd['w_out']} k{cmd['kh']}x{cmd['kw']} s{cmd['stride']} p{cmd['pad']} "
            f"cw{cmd['cin_words']} ow{cmd['cout_words']} K={k_of(cmd['kh'], cmd['kw'], cmd['cin_words'], n)} "
            f"tiles={tiles} measured={res['cycles']} estimate={est} "
            f"gemm_bound={gemm_only_cycles(cmd, n, self.lat)} "
            f"per_tile={(res['cycles'] - 4) / tiles:.1f}")
        return res["cycles"]


def make_layer(h, rng, hw, **force):
    return random_layer(rng, hw, **force)


async def fresh(dut):
    h = ConvHarness(dut)
    await h.start()
    for sel in range(h.hw.nbuf):
        await h.fill_bank(sel)
    return h


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------

@cocotb.test()
async def test_host_readback(dut):
    """the host ports alone: junk in every bank reads back. this is what pins
    down the read offset that every other check relies on."""
    h = await fresh(dut)
    for sel in range(h.hw.nbuf):
        await h.check_bank(sel, "after fill")


@cocotb.test()
async def test_1x1_single_tile(dut):
    """1x1 conv = pure GEMM: exactly one tile of N pixels, N in, N out"""
    h = await fresh(dut)
    n = h.n
    layer = random_layer(h.rng, h.hw, kh=1, kw=1, stride=1, pad=0, h_in=1, w_in=n,
                         cin=n, cout=n)
    await h.load_layer(layer)
    await h.run_layer(layer, "1x1 single tile")


@cocotb.test()
async def test_3x3_pad1_stride1(dut):
    h = await fresh(dut)
    layer = random_layer(h.rng, h.hw, kh=3, kw=3, stride=1, pad=1, h_in=5, w_in=6)
    assert layer.command(h.n)["h_out"] == 5 and layer.command(h.n)["w_out"] == 6
    await h.load_layer(layer)
    await h.run_layer(layer, "3x3 pad1 stride1")


@cocotb.test()
async def test_3x3_pad0_stride2(dut):
    h = await fresh(dut)
    layer = random_layer(h.rng, h.hw, kh=3, kw=3, stride=2, pad=0, h_in=7, w_in=8)
    assert (layer.command(h.n)["h_out"], layer.command(h.n)["w_out"]) == (3, 3)
    await h.load_layer(layer)
    await h.run_layer(layer, "3x3 pad0 stride2")


@cocotb.test()
async def test_several_channel_groups(dut):
    """cin_words > 1 and cout_words > 1, with partial last groups both ways"""
    h = await fresh(dut)
    n = h.n
    layer = random_layer(h.rng, h.hw, kh=3, kw=2, stride=1, pad=1, h_in=4, w_in=3,
                         cin=2 * n + (1 if n > 1 else 0), cout=3 * n - (1 if n > 1 else 0))
    c = layer.command(n)
    assert c["cin_words"] >= 2 and c["cout_words"] == 3
    await h.load_layer(layer)
    await h.run_layer(layer, "several channel groups")


@cocotb.test()
async def test_tail_masking(dut):
    """output pixel count not a multiple of N: the last tile's extra rows are
    computed and must not be written. the bank check covers the words right
    after the region, which is where a missing mask would write."""
    h = await fresh(dut)
    n = h.n
    for hw_in, cout in ((3, n), (5, 2 * n), (1, 1)):
        layer = random_layer(h.rng, h.hw, kh=1, kw=1, stride=1, pad=0, h_in=hw_in, w_in=hw_in,
                             cout=cout)
        npix = hw_in * hw_in
        if n > 1 and npix % n == 0:
            continue
        await h.load_layer(layer)
        await h.run_layer(layer, f"tail {npix} pixels")


@cocotb.test()
async def test_requant_corners(dut):
    """relu on and off, nonzero and extreme zero points, and requant params
    that saturate: max mult with no shift clamps nearly everything to the
    int8 rails, shift 31 rounds nearly everything to the zero point"""
    h = await fresh(dut)
    n = h.n
    cases = [
        ("relu off zp 0/0", dict(relu_en=0, zp_in=0, zp_out=0), None),
        ("relu on zp -17/23", dict(relu_en=1, zp_in=-17, zp_out=23), None),
        ("relu off zp 127/-128", dict(relu_en=0, zp_in=127, zp_out=-128), None),
        ("relu on zp -128/127", dict(relu_en=1, zp_in=-128, zp_out=127), None),
        ("saturating mult", dict(relu_en=0, zp_in=5, zp_out=-3), "sat"),
        ("saturating mult relu", dict(relu_en=1, zp_in=5, zp_out=-3), "sat"),
        ("shift 31", dict(relu_en=0, zp_in=0, zp_out=9), "shift31"),
    ]
    for name, force, mode in cases:
        layer = random_layer(h.rng, h.hw, kh=3, kw=3, stride=1, pad=1, h_in=3, w_in=4, **force)
        if mode == "sat":
            layer.mult = np.full(layer.cout, (1 << 16) - 1)
            layer.shift = np.zeros(layer.cout, int)
        elif mode == "shift31":
            layer.shift = np.full(layer.cout, 31)
        ref = layer.reference()
        if mode == "sat":
            rails = np.isin(ref, (-128, 127)).mean() if not force["relu_en"] else \
                np.isin(ref, (force["zp_out"], 127)).mean()
            assert rails > 0.9, f"stimulus: only {rails:.0%} of outputs saturate"
        await h.load_layer(layer)
        await h.run_layer(layer, name)


@cocotb.test()
async def test_random_layers(dut):
    """constrained-random legal layers: shapes, strides, padding, channel
    counts, zero points, relu, requant params, bank and base placement"""
    h = await fresh(dut)
    count = int(os.environ.get("LAYERS") or 6)
    for i in range(count):
        src = int(h.rng.integers(0, h.hw.nbuf))
        dst = int((src + 1 + h.rng.integers(0, h.hw.nbuf - 1)) % h.hw.nbuf)
        # up to 3N output channels: a group-advance bug only shows from the
        # third group on (group 1 is the first that moves anything)
        layer = random_layer(h.rng, h.hw, max_dim=7, max_cout=3 * h.n, src_sel=src, dst_sel=dst)
        await h.load_layer(layer)
        await h.run_layer(layer, f"random #{i}")
    rows = "\n".join(f"  {name:<24} tiles={tiles:<3} measured={meas:<6} estimate={est:<6} "
                     f"ratio={meas / est:.3f}" for name, _, tiles, meas, est in h.cycle_log)
    dut._log.info(f"cycle summary N={h.n} L={h.lat}:\n{rows}")


@cocotb.test()
async def test_ping_pong_chain(dut):
    """three layers back to back, 0 -> 1 -> 0 -> 1. each layer's input is the
    previous layer's output, read from where the hardware wrote it, with
    zp_in = the previous zp_out. weights for all three stay loaded at
    different B/P bases. the final bank is compared with the chained NumPy
    result, so any disagreement about the memory layout between write-back
    and im2col_reader shows up here."""
    h = await fresh(dut)
    n, rng, hw = h.n, h.rng, h.hw
    specs = [dict(kh=3, kw=3, stride=1, pad=1), dict(kh=1, kw=1, stride=1, pad=0),
             dict(kh=3, kw=3, stride=2, pad=1)]
    x = rng.integers(-128, 128, (6, 5, n + 1))
    zp = int(rng.integers(-20, 21))
    b_next, p_next, src, base = 0, 0, 0, 0
    layers = []
    for i, s in enumerate(specs):
        cin, cout = x.shape[2], int(rng.integers(1, 2 * n + 1))
        w = rng.integers(-128, 128, (s["kh"], s["kw"], cin, cout))
        bias, mult, shift = requant_params(rng, cout, s["kh"] * s["kw"] * cin)
        zp_out = int(rng.integers(-20, 21))
        layer = Layer(x=x, w=w, bias=bias, mult=mult, shift=shift, zp_in=zp, zp_out=zp_out,
                      relu_en=i % 2, stride=s["stride"], pad=s["pad"], src_sel=src,
                      dst_sel=1 - src, src_base=base,
                      dst_base=int(rng.integers(0, 64)), b_base=b_next, p_base=p_next)
        cmd = layer.command(n)
        b_next += cmd["cout_words"] * k_of(s["kh"], s["kw"], cmd["cin_words"], n)
        p_next += cmd["cout_words"] * n
        layers.append(layer)
        x, zp, src, base = layer.reference(), zp_out, 1 - src, layer.dst_base
    assert check_command(layers[-1].command(n), hw) == 0 and p_next <= (1 << hw.p_addr_w)
    # the image goes in once; weights for every layer go in before the first runs
    await h.host_write(0, layers[0].src_base, pack_act(layers[0].x, n, layers[0].zp_in))
    for layer in layers:
        await h.load_layer(layer, image=False)
    for i, layer in enumerate(layers):
        # nothing is written to memory between layers: the hardware's output
        # is the next input as it stands
        await h.run_layer(layer, f"chain layer {i}", check_banks=False)
    for sel in range(hw.nbuf):
        await h.check_bank(sel, "after the chain")


@cocotb.test()
async def test_reset_mid_layer(dut):
    """reset at several points of a layer: ready right after, nothing written
    on or after the reset edge, every destination word old or final, nothing
    outside the region changed. then a clean rerun gives the full output.
    one reset lands exactly on the edge that would write the last word."""
    h = await fresh(dut)
    n = h.n
    layer = random_layer(h.rng, h.hw, kh=3, kw=3, stride=1, pad=1, h_in=4, w_in=5,
                         cout=2 * n if n > 1 else 2)
    await h.load_layer(layer)
    cmd = layer.command(n)
    final = expected_output_words(layer, n)
    base, count = layer.dst_base, len(final)
    dst = layer.dst_sel
    # a clean run first, for the length of the layer
    old = list(h.mem[dst])
    t_full = await h.run_layer(layer, "reset: reference run", check_banks=False)
    for t in (2, 4, t_full // 3, (2 * t_full) // 3, t_full - 1, t_full):
        await h.fill_bank(dst)        # fresh junk, so old != final everywhere
        old = list(h.mem[dst])
        res = await h.run(cmd, reset_at=t)
        assert res["kind"] == "reset"
        got = await h.read_words(dst, 0, h.depth)
        written = 0
        for a in range(h.depth):
            if base <= a < base + count and got[a] == final[a - base] and final[a - base] != old[a]:
                written += 1
            else:
                assert got[a] == old[a], (f"reset at edge {t}: word {a} is neither old nor final "
                                          f"{h.where()}")
        if t == t_full:
            # the last write would land on edge t_full; reset there must stop it
            assert written == count - 1, f"reset on the last edge: {written} of {count} words written"
        dut._log.info(f"reset at edge {t} of {t_full}: {written}/{count} output words written")
        # clean rerun of the same command
        res = await h.run(cmd)
        assert res["kind"] == "done" and res["cycles"] == t_full, \
            f"rerun after reset took {res['cycles']} edges, a clean run takes {t_full}"
        for a, w in enumerate(final):
            h.mem[dst][base + a] = w
        await h.check_bank(dst, f"rerun after reset at edge {t}")
    await h.check_bank(layer.src_sel, "source bank after the resets")


@cocotb.test()
async def test_illegal_configs(dut):
    """each check the contract rejects: err pulses once with exactly the bits
    check_command predicts, done never pulses, nothing is written, and the
    next legal layer still runs"""
    h = await fresh(dut)
    n, hw = h.n, h.hw
    good = random_layer(h.rng, hw, kh=3, kw=3, stride=1, pad=1, h_in=4, w_in=4, cin=n, cout=n)
    await h.load_layer(good)
    g = good.command(n)
    K = k_of(3, 3, 1, n)
    k_cw = -(-(hw.k_max + 1) // (49 * n))            # 7x7 kernel, K just past K_MAX
    cases = [
        ("E_DIM cin_words=0", dict(cin_words=0), E_DIM, 0),
        ("E_DIM stride=0", dict(stride=0), E_DIM, 0),
        ("E_DIM h_out=0", dict(h_out=0), E_DIM, 0),
        # one too big and one too small on each axis: both sides of the
        # (h_out-1)*stride + kh <= h_in + 2*pad < h_out*stride + kh window
        ("E_SHAPE h_out+1", dict(h_out=g["h_out"] + 1), E_SHAPE, 0),
        ("E_SHAPE h_out-1", dict(h_out=g["h_out"] - 1), E_SHAPE, 0),
        ("E_SHAPE w_out+1", dict(w_out=g["w_out"] + 1), E_SHAPE, 0),
        ("E_SHAPE w_out-1", dict(w_out=g["w_out"] - 1), E_SHAPE, 0),
        # stride 2 where the floor matters: 7 -> 3 legal, 2 and 4 are not
        ("E_SHAPE stride 2 h_out 2", dict(stride=2, pad=0, h_in=7, h_out=2, w_in=7, w_out=3),
         E_SHAPE, 0),
        ("E_SHAPE stride 2 w_out 4", dict(stride=2, pad=0, h_in=7, h_out=3, w_in=7, w_out=4),
         E_SHAPE, 0),
        ("E_SHAPE kernel > padded image", dict(pad=0, kh=7, h_out=1), E_SHAPE, 0),
        ("E_K", dict(kh=7, kw=7, pad=0, h_in=7, w_in=7, h_out=1, w_out=1, cin_words=k_cw,
                     src_base=0), E_K, 0),
        ("E_BMEM", dict(b_base=(1 << hw.b_addr_w) - K + 1), E_BMEM, 0),
        ("E_PMEM", dict(p_base=(1 << hw.p_addr_w) - n + 1), E_PMEM, 0),
        ("E_SRC", dict(src_base=h.depth - 16 + 1), E_SRC, 0),
        ("E_DST", dict(dst_base=h.depth - 16 + 1), E_DST, 0),
        ("E_SEL src == dst", dict(dst_sel=g["src_sel"]), E_SEL, 0),
        ("E_HOST", dict(), E_HOST, 1),
    ]
    if (1 << h.sel_w) > hw.nbuf:
        cases.append(("E_SEL dst >= NBUF", dict(dst_sel=hw.nbuf), E_SEL, 0))
        cases.append(("E_SEL src >= NBUF", dict(src_sel=hw.nbuf), E_SEL, 0))
    check_cycles = set()
    for name, over, bit, host in cases:
        cmd = {**g, **over}
        exp = check_command(cmd, hw, host_mode=host)
        assert exp >> bit & 1, f"stimulus: {name} does not trip {ERR_NAMES[bit]}"
        res = await h.run(cmd, host_mode=host)
        assert res["kind"] == "err", f"{name}: layer ran instead of being rejected"
        assert res["tiles"] == 0, f"{name}: a rejected command launched a tile"
        assert res["err_code"] == exp, (f"{name}: err_code {err_names(res['err_code'])}, "
                                        f"expected {err_names(exp)}")
        check_cycles.add(res["cycles"])
        dut._log.info(f"{name}: err_code={err_names(res['err_code'])} after {res['cycles']} edges")
    assert len(check_cycles) == 1, f"check phase length varies: {check_cycles}"
    dut._log.info(f"check phase: err visible {check_cycles.pop()} edges after start")
    for sel in range(hw.nbuf):
        await h.check_bank(sel, "after the rejected commands")
    await h.run_layer(good, "legal layer after the rejects")
