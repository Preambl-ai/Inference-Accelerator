""" cocotb tb for rtl/im2col_reader.sv """

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge
import numpy as np

from im2col_model import (ActBufferModel, Im2colReaderModel, im2col_slices,
                          out_size, tensor_words)
from bram_model import UNKNOWN
from pe_model import signed_range


def read_int(sig, name):
    raw = str(sig.value)
    if "x" in raw.lower() or "z" in raw.lower():
        raise AssertionError(f"{name} is not resolvable: {raw}")
    return int(sig.value)


def take_pattern(rng, kind):
    """Endless generator: does the consumer take a valid slice this cycle?"""
    while True:
        if kind == "always":
            yield 1
        elif kind == "random":
            yield int(rng.random() < 0.5)
        elif kind == "rare":
            yield int(rng.random() < 0.1)
        elif kind == "toggle":
            yield 1
            yield 0
        elif kind == "long_low":
            for _ in range(rng.randint(20, 60)):
                yield 0
            for _ in range(rng.randint(1, 5)):
                yield 1
        else:
            raise ValueError(kind)


class Layer:
    """Geometry of one convolution input and the tensor that goes with it."""

    def __init__(self, tensor, kh, kw, stride, pad, pad_value, base, n):
        self.tensor = tensor
        self.h, self.w, c = tensor.shape
        self.cw = c // n
        self.kh, self.kw, self.stride, self.pad = kh, kw, stride, pad
        self.pad_value = pad_value
        self.base = base
        self.h_out = out_size(self.h, kh, stride, pad)
        self.w_out = out_size(self.w, kw, stride, pad)

    def __str__(self):
        return (f"{self.h}x{self.w}x{self.cw} words, {self.kh}x{self.kw} "
                f"stride {self.stride} pad {self.pad} pad_value {self.pad_value} "
                f"base {self.base}")


class ReaderHarness:
    """Drives the reader against a Python model of the activation buffer.

    The buffer block is not written here, so its read port is played by
    ActBufferModel: every cycle the reader's rd_en/rd_addr go into the model
    and the model's rd_data comes back with the read latency. Words outside
    the tensor are undefined in the model and come back as random garbage, so
    a padded pixel that is read from memory by mistake cannot give the right
    answer.

    The slices the reader hands over are compared with im2col_slices(), which
    indexes the tensor directly and knows nothing about addresses.
    """

    def __init__(self, dut):
        self.dut = dut
        g = lambda name: int(getattr(dut, name).value)
        self.n = g("N")
        self.dw = g("DATA_WIDTH")
        self.addr_w = g("ADDR_W")
        self.lat = g("READ_LATENCY")
        self.dim_w = g("DIM_W")
        self.cw_w = g("CW_W")
        self.lo, self.hi = signed_range(self.dw)
        self.model = Im2colReaderModel(self.n, self.lat)
        self.cycle = 0
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = random.Random(self.seed)
        self.rd_sel = self.rng.randrange(3)
        self.buf = ActBufferModel(self.n, self.addr_w, self.lat, nbuf=3, data_width=self.dw)
        self.taken = []         # slices handed over, as lists of N values
        self.held = None        # a_data while a_valid and not yet taken
        self.a_valid = 0        # a_valid for the cycle about to be driven
        self.reads_of_pad = 0

    async def start(self, reset_cycles=3):
        d = self.dut
        d.reset.value = 1
        d.start.value = 0
        d.a_take.value = 0
        d.rd_data.value = 0
        for name in ("y0", "x0", "win_order", "h_in", "w_in", "w_out", "cin_words",
                     "kh", "kw", "stride", "pad", "pad_value", "src_base"):
            getattr(d, name).value = 0
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        # registers are X until the first reset edge
        await self.step(reset=1, check=False)
        for _ in range(reset_cycles - 1):
            await self.step(reset=1)
        d._log.info(
            f"DUT parameters: N={self.n} READ_LATENCY={self.lat} ADDR_W={self.addr_w} "
            f"SEED={self.seed} (reproduce with: make MODULE_NAME=im2col_reader "
            f"N={self.n} READ_LATENCY={self.lat} ADDR_WIDTH={self.addr_w} SEED={self.seed})"
        )

    # stimulus

    def make_layer(self, h=None, w=None, cw=None, k=None, stride=None, pad=None, base=None,
                   kh=None, kw=None):
        r = self.rng
        k = k or r.choice((1, 3, 5))
        kh, kw = kh or k, kw or k
        stride = stride or r.choice((1, 2))
        pad = r.choice((0, 1, 2)) if pad is None else pad
        cw = cw or r.randint(1, 3)
        h = h or r.randint(max(1, kh - 2 * pad), 9)
        w = w or r.randint(max(1, kw - 2 * pad), 9)
        # nonzero everywhere, and a nonzero pad value: the input zero point is
        # rarely 0, and a pad that reads as 0 must not be able to pass
        tensor = np.array([[[r.choice((-1, 1)) * r.randint(1, self.hi)
                             for _ in range(cw * self.n)] for _ in range(w)] for _ in range(h)])
        pad_value = r.choice((-1, 1)) * r.randint(1, self.hi)
        words = h * w * cw
        if words > (1 << self.addr_w):
            raise ValueError("tensor does not fit the address space")
        base = r.randrange((1 << self.addr_w) - words + 1) if base is None else base
        layer = Layer(tensor, kh, kw, stride, pad, pad_value, base, self.n)
        self.buf = ActBufferModel(self.n, self.addr_w, self.lat, nbuf=3, data_width=self.dw)
        self.buf.load(self.rd_sel, tensor_words(tensor, self.n, base))
        return layer

    def expected(self, layer, y0, x0, win=0):
        return im2col_slices(layer.tensor, y0, x0, self.n, layer.w_out, layer.kh, layer.kw,
                             layer.stride, layer.pad, layer.pad_value, win)

    def _unpack(self, word):
        mask, sign = (1 << self.dw) - 1, 1 << (self.dw - 1)
        vals = [(word >> (i * self.dw)) & mask for i in range(self.n)]
        return [v - (1 << self.dw) if v & sign else v for v in vals]

    # one clock cycle

    async def step(self, cmd=None, take=0, reset=0, check=True):
        """Advance one rising edge. cmd = (layer, y0, x0, win) offers a tile
        with start high. take is honoured only while a slice is valid.
        Returns (accepted, slice taken or None)."""
        d = self.dut
        r = self.rng
        await FallingEdge(d.clk)
        d.reset.value = reset
        if cmd is not None:
            layer, y0, x0, win = cmd
            d.start.value = 1
            d.y0.value, d.x0.value, d.win_order.value = y0, x0, win
            d.h_in.value, d.w_in.value, d.w_out.value = layer.h, layer.w, layer.w_out
            d.cin_words.value = layer.cw
            d.kh.value, d.kw.value = layer.kh, layer.kw
            d.stride.value, d.pad.value = layer.stride, layer.pad
            d.pad_value.value = layer.pad_value & ((1 << self.dw) - 1)
            d.src_base.value = layer.base
        else:
            # with start low the command pins carry legal-looking garbage
            d.start.value = 0
            d.y0.value, d.x0.value = r.randrange(1 << self.dim_w), r.randrange(1 << self.dim_w)
            d.win_order.value = r.randint(0, 1)
            d.h_in.value, d.w_in.value = r.randint(1, 255), r.randint(1, 255)
            d.w_out.value = r.randint(1, 255)
            d.cin_words.value = r.randint(1, (1 << self.cw_w) - 1)
            d.kh.value, d.kw.value = r.randint(1, 7), r.randint(1, 7)
            d.stride.value, d.pad.value = r.randint(1, 3), r.randint(0, 3)
            d.pad_value.value = r.randrange(1 << self.dw)
            d.src_base.value = r.randrange(1 << self.addr_w)

        do_take = int(bool(take and self.a_valid and not reset))
        d.a_take.value = do_take
        # the buffer's rd_data: a real word, or garbage where it is undefined
        word = self.buf.rdata()
        d.rd_data.value = r.getrandbits(self.n * self.dw) | 1 if word is UNKNOWN else word

        accepted, got = False, None
        rd_en = rd_addr = 0
        if check:
            # every output is a register, so the pins now are what the buffer
            # and the tile sample at the coming edge
            await ReadOnly()
            rd_en = read_int(d.rd_en, "rd_en")
            rd_addr = read_int(d.rd_addr, "rd_addr") if rd_en else 0
            a_valid = read_int(d.a_valid, "a_valid")
            assert a_valid == self.a_valid
            if reset:
                self.held = None
            else:
                accepted = bool(cmd is not None and read_int(d.ready, "ready"))
                if a_valid:
                    a_data = read_int(d.a_data, "a_data")
                    if self.held is not None:
                        assert a_data == self.held, (
                            f"cycle {self.cycle}: a_data changed while valid and not "
                            f"taken, seed={self.seed}"
                        )
                    if do_take:
                        got = self._unpack(a_data)
                        self.taken.append(got)
                        self.held = None
                    else:
                        self.held = a_data
                else:
                    assert self.held is None, (
                        f"cycle {self.cycle}: a_valid dropped without a take, seed={self.seed}"
                    )

        await RisingEdge(d.clk)
        self.buf.step(rd_en=rd_en, rd_sel=self.rd_sel, rd_addr=rd_addr)
        # leave the caller at the settled values of the new cycle
        await ReadOnly()
        if check:
            self.a_valid = read_int(d.a_valid, "a_valid")
        self.cycle += 1
        return accepted, got

    async def run_tile(self, layer, y0, x0, win=0, take_kind="always", limit=None):
        """Start one tile and collect its slices until the reader is idle."""
        want = self.expected(layer, y0, x0, win)
        take = take_pattern(self.rng, take_kind)
        self.taken = []
        started = False
        limit = limit or 60 * (len(want) + 4 * self.n) + 500
        for _ in range(limit):
            accepted, _ = await self.step(cmd=None if started else (layer, y0, x0, win),
                                          take=next(take))
            started = started or accepted
            if started and len(self.taken) == len(want):
                break
        else:
            raise AssertionError(f"tile did not finish: {len(self.taken)} of "
                                 f"{len(want)} slices, seed={self.seed}")
        self.check_slices(want, f"{layer}, tile at ({y0},{x0}) win={win}")
        # one more cycle and the reader must be idle with nothing on offer
        await self.step()
        assert read_int(self.dut.ready, "ready") == 1, "reader not idle after the last slice"
        assert read_int(self.dut.a_valid, "a_valid") == 0
        return want

    def check_slices(self, want, what):
        assert len(self.taken) == len(want), f"{what}: {len(self.taken)} slices, want {len(want)}"
        for k, (got, exp) in enumerate(zip(self.taken, want)):
            assert got == exp, (
                f"{what}\nslice {k} of {len(want)}: got {got}\n"
                f"{' ' * len(str(k))}             expected {exp}\nseed={self.seed}"
            )

    def random_origin(self, layer, win=0):
        r = self.rng
        if win:
            return 2 * r.randrange(max(1, layer.h_out // 2)), 2 * r.randrange(layer.w_out // 2)
        return r.randrange(layer.h_out), r.randrange(layer.w_out)


def can_window(h):
    return h.n % 4 == 0


# TESTS


@cocotb.test()
async def test_3x3_pad1_directed(dut):
    """The common case, every tile of a small layer: 3x3, stride 1, pad 1."""
    h = ReaderHarness(dut)
    await h.start()
    layer = h.make_layer(h=5, w=6, cw=2, k=3, stride=1, pad=1)
    pixels = layer.h_out * layer.w_out
    for p in range(0, pixels, h.n):
        # the last tile runs past the end of the output; it is still walked
        await h.run_tile(layer, p // layer.w_out, p % layer.w_out)
    dut._log.info(f"{-(-pixels // h.n)} tiles of {layer}")


@cocotb.test()
async def test_latency_from_sim(dut):
    """Measure the gather latency and the cost of a group on the DUT."""
    h = ReaderHarness(dut)
    await h.start()
    layer = h.make_layer(h=6, w=6, cw=3, k=1, stride=1, pad=0)
    cycle = 0
    start_cycle = None
    take_cycles = []
    while len(take_cycles) < 3 * h.n:
        accepted, got = await h.step(cmd=None if start_cycle is not None else (layer, 0, 0, 0),
                                     take=1)
        if accepted:
            start_cycle = cycle
        if got is not None:
            take_cycles.append(cycle)
        cycle += 1
    # a slice taken in cycle c was valid from the edge before it. the first
    # slice of each group is the one that waited for the gather.
    first = [take_cycles[i * h.n] for i in range(3)]
    measured = {
        "first_slice": first[0] - start_cycle - 1,
        "group_period": first[1] - first[0],
        "group_period_2": first[2] - first[1],
        "slices_back_to_back": take_cycles[h.n - 1] - take_cycles[0],
    }
    expect = {
        "first_slice": h.model.gather_latency(),
        "group_period": h.model.group_period(),
        "group_period_2": h.model.group_period(),
        "slices_back_to_back": h.n - 1,
    }
    assert measured == expect, f"measured {measured}\nexpected {expect}"
    dut._log.info(
        f"reader timing: N={h.n} READ_LATENCY={h.lat} -> first slice after "
        f"{measured['first_slice']} edges, {measured['group_period']} cycles per "
        f"group of {h.n} slices ({100 * h.n // measured['group_period']}% of cycles "
        f"deliver a slice)"
    )


@cocotb.test()
async def test_kernels_strides_pads(dut):
    """1x1, 3x3 and 5x5 against stride 1 and 2 and padding 0, 1 and 2."""
    h = ReaderHarness(dut)
    await h.start()
    done = 0
    for k in (1, 3, 5):
        for stride in (1, 2):
            for pad in (0, 1, 2):
                layer = h.make_layer(k=k, stride=stride, pad=pad, cw=h.rng.randint(1, 2))
                for _ in range(2):
                    await h.run_tile(layer, *h.random_origin(layer))
                    done += 1
    dut._log.info(f"{done} tiles over 18 kernel/stride/pad combinations")


@cocotb.test()
async def test_non_square_kernels(dut):
    """kh != kw, both ways round, on images that are not square either."""
    h = ReaderHarness(dut)
    await h.start()
    for kh, kw in ((1, 3), (3, 1), (3, 5), (5, 3), (1, 5), (5, 1)):
        for stride in (1, 2):
            pad = h.rng.choice((0, 1, 2))
            layer = h.make_layer(kh=kh, kw=kw, stride=stride, pad=pad, cw=h.rng.randint(1, 2),
                                 h=h.rng.randint(5, 9), w=h.rng.randint(5, 9))
            want = await h.run_tile(layer, *h.random_origin(layer))
            assert len(want) == kh * kw * layer.cw * h.n


@cocotb.test()
async def test_padding_is_pad_value_not_memory(dut):
    """Borders with a nonzero input zero point. Every corner tile of a padded
    layer, where most of what the taps cover is outside the image."""
    h = ReaderHarness(dut)
    await h.start()
    for k, pad in ((3, 1), (5, 2), (3, 2)):
        for stride in (1, 2):
            layer = h.make_layer(h=4, w=5, k=k, stride=stride, pad=pad, cw=1)
            assert layer.pad_value != 0
            want = []
            for y0, x0 in ((0, 0), (0, max(0, layer.w_out - 1)),
                           (layer.h_out - 1, 0), (layer.h_out - 1, layer.w_out - 1)):
                want += await h.run_tile(layer, y0, x0, take_kind="random")
            flat = [v for s in want for v in s]
            # the test is only worth something if padding really was involved
            assert flat.count(layer.pad_value) > len(flat) // 10


@cocotb.test()
async def test_window_order_tiles(dut):
    """2x2 pooling-window order, including windows that wrap to the next row
    pair, for stride 1 and 2."""
    h = ReaderHarness(dut)
    await h.start()
    if not can_window(h):
        dut._log.info(f"N={h.n}: window order needs N % 4 == 0, not applicable")
        return
    done = 0
    for k, stride, pad in ((3, 1, 1), (1, 1, 0), (3, 2, 1), (5, 1, 2)):
        for _ in range(3):
            layer = None
            while layer is None or layer.w_out % 2 or layer.h_out < 2:
                layer = h.make_layer(k=k, stride=stride, pad=pad, cw=h.rng.randint(1, 2),
                                     h=h.rng.randint(4, 9), w=h.rng.randint(4, 9))
            await h.run_tile(layer, *h.random_origin(layer, win=1), win=1,
                             take_kind=h.rng.choice(("always", "random")))
            # the window that sits at the right-hand edge, to force the wrap
            await h.run_tile(layer, 0, layer.w_out - 2, win=1)
            done += 2
    dut._log.info(f"{done} window-order tiles")


@cocotb.test()
async def test_take_patterns(dut):
    """The consumer takes slices late in every style; none is lost, repeated
    or changed while it waits."""
    h = ReaderHarness(dut)
    await h.start()
    for kind in ("random", "rare", "toggle", "long_low"):
        layer = h.make_layer(k=3, stride=h.rng.choice((1, 2)), pad=1, cw=2)
        await h.run_tile(layer, *h.random_origin(layer), take_kind=kind)


@cocotb.test()
async def test_back_to_back_and_start_while_busy(dut):
    """start held high with the next tile's command on the pins while the
    current one runs. The running tile must not notice."""
    h = ReaderHarness(dut)
    await h.start()
    layer = h.make_layer(h=7, w=8, cw=2, k=3, stride=1, pad=1)
    tiles = [h.random_origin(layer) for _ in range(5)]
    want = [h.expected(layer, y0, x0) for (y0, x0) in tiles]
    h.taken = []
    queue = list(tiles)
    take = take_pattern(h.rng, "random")
    total = sum(len(w) for w in want)
    for _ in range(80 * total):
        # offer the next tile all the time, including while busy
        accepted, _ = await h.step(cmd=(layer, *queue[0], 0) if queue else None,
                                   take=next(take))
        if accepted:
            queue.pop(0)
        if not queue and len(h.taken) == total:
            break
    h.check_slices([s for w in want for s in w], "five tiles back to back")


@cocotb.test()
async def test_reset_mid_tile(dut):
    """Reset while walking, waiting and serving. The reader comes back idle
    with nothing on offer and the next tile is clean."""
    h = ReaderHarness(dut)
    await h.start()
    layer = h.make_layer(h=6, w=6, cw=2, k=3, stride=1, pad=1)
    y0, x0 = h.random_origin(layer)
    group = h.model.group_period()
    for at in sorted({1, 2, h.n // 2 + 1, h.n + 1, h.n + h.lat + 2, h.model.gather_latency() + 1,
                      h.model.gather_latency() + 2, group, group + 2, 2 * group + 3}):
        started = False
        for _ in range(at):
            accepted, _ = await h.step(cmd=None if started else (layer, y0, x0, 0), take=1)
            started = started or accepted
        await h.step(reset=1)
        assert read_int(dut.a_valid, "a_valid") == 0, "a_valid survived reset"
        assert read_int(dut.ready, "ready") == 1, "reader not idle after reset"
        for _ in range(group + 4):
            await h.step(take=1)
            assert read_int(dut.a_valid, "a_valid") == 0, "a slice appeared after reset"
        await h.run_tile(layer, y0, x0, take_kind="random")


@cocotb.test()
async def test_rd_en_covers_reads_in_flight(dut):
    """rd_en is the buffer's read clock enable, so it has to stay high from the
    first address until the last word has arrived, and be low when idle."""
    h = ReaderHarness(dut)
    await h.start()
    layer = h.make_layer(h=5, w=5, cw=1, k=3, stride=1, pad=1)
    await h.step()
    assert read_int(dut.rd_en, "rd_en") == 0, "rd_en high while idle"
    started = False
    low_while_busy = 0
    h.taken = []
    want = h.expected(layer, 1, 1)
    while len(h.taken) < len(want):
        accepted, _ = await h.step(cmd=None if started else (layer, 1, 1, 0), take=1)
        started = started or accepted
        if started and not read_int(dut.ready, "ready"):
            low_while_busy += int(read_int(dut.rd_en, "rd_en") == 0)
    assert low_while_busy == 0, f"rd_en dropped for {low_while_busy} cycles mid-tile"
    h.check_slices(want, "rd_en tile")
    await h.step()
    await h.step()
    assert read_int(dut.rd_en, "rd_en") == 0, "rd_en still high after the tile"


@cocotb.test()
async def test_large_geometry_addresses(dut):
    """Big images, many channel words and a high base, so the address
    arithmetic is exercised well past 8 and 16 bits. Only where the address
    space is big enough; the sweep has a configuration for it."""
    h = ReaderHarness(dut)
    await h.start()
    if h.addr_w < 20:
        dut._log.info(f"ADDR_W={h.addr_w}: covered by the ADDR_WIDTH=22 sweep point")
        return
    for (hh, ww, cw, k, stride, pad) in ((250, 255, 3, 3, 1, 1), (255, 250, 2, 5, 2, 2),
                                         (200, 255, 5, 1, 2, 0)):
        layer = h.make_layer(h=hh, w=ww, cw=cw, k=k, stride=stride, pad=pad)
        for y0, x0 in ((0, 0), (layer.h_out - 1, layer.w_out - 1),
                       (layer.h_out - 1, 0), h.random_origin(layer)):
            await h.run_tile(layer, y0, x0)
    dut._log.info("large-geometry address arithmetic passed")
