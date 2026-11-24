""" cocotb tb for rtl/gemm_tile.sv """

import math
import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer
import numpy as np

from gemm_tile_model import GemmTileModel, Beat
from pe_model import signed_range


def read_int(sig, name):
    raw = str(sig.value)
    if "x" in raw.lower() or "z" in raw.lower():
        raise AssertionError(f"{name} is not resolvable: {raw}")
    return int(sig.value)


def feed_pattern(rng, kind, n):
    """Endless a_valid generator: is the feeder willing to show a slice?"""
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
        elif kind == "gather":
            # the shape a serial reader has: N slices, then a gap to refill
            for _ in range(n):
                yield 1
            for _ in range(n + 3):
                yield 0
        else:
            raise ValueError(kind)


def ready_pattern(rng, kind):
    """Endless m_axis_tready generator for one backpressure style."""
    while True:
        if kind == "always":
            yield 1
        elif kind == "random":
            yield int(rng.random() < 0.5)
        elif kind == "mostly_low":
            yield int(rng.random() < 0.1)
        elif kind == "toggle":
            # single-cycle toggles
            yield 1
            yield 0
        elif kind == "long_low":
            for _ in range(rng.randint(30, 150)):
                yield 0
            for _ in range(rng.randint(1, 6)):
                yield 1
        elif kind == "mixed":
            sub = ready_pattern(rng, rng.choice(("always", "random", "toggle", "long_low")))
            for _ in range(rng.randint(10, 150)):
                yield next(sub)
        else:
            raise ValueError(kind)


class Tile:
    """One tile's command and the beats it has to produce."""

    def __init__(self, k, b_base, p_base, zero_point, relu_en, beats, params, slices):
        self.k = k
        self.slices = slices    # what the feeder supplies: K lists of N values
        self.b_base = b_base
        self.p_base = p_base
        self.zero_point = zero_point
        self.relu_en = relu_en
        self.beats = beats
        self.params = params


class GemmHarness:
    """Drives the DUT and keeps GemmTileModel's memory images in step with it.

    There is no cycle-accurate model at this level: the blocks inside were
    each checked cycle by cycle on their own. Here every memory write goes to
    the DUT and the model together, and every beat that crosses m_axis is
    compared, in order, with what the model says the tile produces. The
    stream is also held to the AXI-Stream rules on every cycle.
    """

    def __init__(self, dut):
        self.dut = dut
        g = lambda name: int(getattr(dut, name).value)
        self.n = g("N")
        self.dw = g("DATA_WIDTH")
        self.aw = g("ACC_WIDTH")
        self.kw = g("K_WIDTH")
        self.lat = g("READ_LATENCY")
        self.b_aw, self.p_aw = g("B_ADDR_W"), g("P_ADDR_W")
        self.k_max = g("K_MAX")
        self.bw, self.sw = g("BIAS_WIDTH"), g("SAT_WIDTH")
        self.mw, self.shw = g("M_WIDTH"), g("SHIFT_WIDTH")
        self.row_w = g("ROW_W")
        self.model = GemmTileModel(self.n, self.dw, self.aw, self.lat,
                                   self.b_aw, self.p_aw, self.bw, self.sw,
                                   self.mw, self.shw, self.k_max)
        self.lo, self.hi = signed_range(self.dw)
        self.cycle = 0
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = random.Random(self.seed)

        self.expected = []      # beats still owed, across tiles, in order
        self.received = 0
        self.blocked = None     # stream payload held while tvalid & ~tready
        self.running = []       # tiles accepted and not yet drained, oldest first
        self.rows_checked = 0   # rows seen entering requant with params verified
        self.accepts = 0
        self.stall_cycles = 0
        self.cover = dict(clamp_hi=0, clamp_lo=0, relu_floor=0, in_range=0)
        # where the next tile's words go. tiles are often loaded several at a
        # time before any of them runs, so they must not land on each other.
        self.next_base = [self.rng.randrange(1 << w) for w in (self.b_aw, self.p_aw)]
        # the A operand feeder: slices owed to the DUT, oldest first, and how
        # willing it is to show the next one
        self.feed = []
        self.feed_gen = feed_pattern(self.rng, "always", self.n)
        self.feed_hold = 0      # cycles to keep a_valid low regardless
        self.feed_stall_cycles = 0
        self.dup_risk_cycles = 0

    # lifecycle

    async def start(self, reset_cycles=3, fill=True):
        d = self.dut
        d.reset.value = 1
        d.start.value = 0
        d.m_axis_tready.value = 0
        for name in ("k_in", "b_base", "p_base", "zero_point", "relu_en",
                     "a_valid", "a_data", "b_we", "b_waddr", "b_wdata",
                     "p_we", "p_waddr", "p_wdata"):
            getattr(d, name).value = 0
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        # registers are X until the first reset edge
        await self.step(reset=1, check=False)
        for _ in range(reset_cycles - 1):
            await self.step(reset=1)
        d._log.info(
            f"DUT parameters: N={self.n} READ_LATENCY={self.lat} K={get_k()} "
            f"M_WIDTH={self.mw} SHIFT_WIDTH={self.shw} SEED={self.seed} (reproduce "
            f"with: make MODULE_NAME=gemm_tile N={self.n} K={get_k()} "
            f"READ_LATENCY={self.lat} SEED={self.seed})"
        )
        # the sweep has to run the configuration that is meant to be built
        assert (self.mw, self.shw) == (16, 5), (
            f"gemm_tile is expected to build requant with M_WIDTH=16 "
            f"SHIFT_WIDTH=5, got {self.mw}/{self.shw}"
        )
        if fill:
            await self.fill_memories()

    async def fill_memories(self):
        """Put nonzero garbage in every word of both memories, so an
        address or timing error reads something wrong rather than a zero that
        would vanish in the sum."""
        r = self.rng
        byte = lambda: r.choice((-1, 1)) * r.randint(1, self.hi)
        for addr in range(1 << max(self.b_aw, self.p_aw)):
            w = {}
            if addr < (1 << self.b_aw):
                w["b"] = (addr, [byte() for _ in range(self.n)])
            if addr < (1 << self.p_aw):
                w["p"] = (addr, (r.randint(1, 1 << 20), r.randint(1, (1 << self.mw) - 1),
                                 r.randint(1, (1 << self.shw) - 1)))
            await self.step(writes=w)

    # one clock cycle

    async def step(self, tile=None, tready=0, writes=None, reset=0, check=True):
        """Advance one rising edge.

        tile, if given, is offered with start high. writes maps 'b'/'p' to
        (addr, payload) and goes to the DUT and the model. Returns
        (accepted, beat) for the edge: whether start was taken and the beat
        that crossed m_axis, if any.
        """
        d = self.dut
        r = self.rng
        writes = writes or {}
        await FallingEdge(d.clk)
        d.reset.value = reset
        d.m_axis_tready.value = tready
        if tile is not None:
            d.start.value = 1
            d.k_in.value = tile.k
            d.b_base.value = tile.b_base
            d.p_base.value = tile.p_base
            d.zero_point.value = tile.zero_point & ((1 << self.dw) - 1)
            d.relu_en.value = tile.relu_en
        else:
            # with start low the command pins carry garbage that must be ignored
            d.start.value = 0
            d.k_in.value = r.randint(1, (1 << self.kw) - 1)
            d.b_base.value = r.randrange(1 << self.b_aw)
            d.p_base.value = r.randrange(1 << self.p_aw)
            d.zero_point.value = r.randrange(1 << self.dw)
            d.relu_en.value = r.randint(0, 1)

        # the feeder behaves like a register: what it shows can only change
        # between cycles, and it shows garbage when it has nothing valid
        if self.feed_hold:
            self.feed_hold -= 1
            a_valid = 0
        else:
            a_valid = int(bool(self.feed) and not reset and next(self.feed_gen))
        d.a_valid.value = a_valid
        d.a_data.value = (self.model.pack_bytes(self.feed[0]) if a_valid
                          else r.getrandbits(self.n * self.dw) | 1)

        for mem in ("b", "p"):
            we = getattr(d, f"{mem}_we")
            if mem in writes:
                addr, payload = writes[mem]
                we.value = 1
                getattr(d, f"{mem}_waddr").value = addr
                if mem == "p":
                    getattr(d, "p_wdata").value = self.model.pack_param(*payload)
                    self.model.write_p(addr, *payload)
                else:
                    d.b_wdata.value = self.model.pack_bytes(payload)
                    self.model.write_b(addr, payload)
            else:
                we.value = 0

        accepted, beat = False, None
        if check:
            # every output is a function of registers only, so what is on the
            # pins now is what the neighbours sample at the coming edge
            await ReadOnly()
            self._was_ready = bool(read_int(d.ready, "ready"))
            if reset:
                self.expected, self.running, self.blocked = [], [], None
                self.feed = []
            else:
                # the DUT ignores a start whose K it cannot run
                accepted = bool(tile is not None and 1 <= tile.k <= self.k_max
                                and read_int(d.ready, "ready"))
                beat = self._watch_stream(tready)
                self._watch_first_row()
                stall = read_int(d.stall, "stall")
                stall_feed = read_int(d.stall_feed, "stall_feed")
                self.stall_cycles += stall
                self.feed_stall_cycles += stall_feed
                # the case the ~stall on the skid buffer input exists for: a
                # row waiting at the requant output, room in the skid buffer,
                # and the tile frozen by the feeder
                self.dup_risk_cycles += int(stall_feed and not read_int(d.stall_out, "stall_out")
                                            and read_int(d.rq_valid, "rq_valid") & 1)
                if read_int(d.a_take, "a_take"):
                    assert a_valid, (
                        f"cycle {self.cycle}: a_take with a_valid low, seed={self.seed}"
                    )
                    self.feed.pop(0)
                if accepted:
                    self.accepts += 1
                    self.running.append(tile)
                    self.expected += tile.beats
                    self.feed += tile.slices

        await RisingEdge(d.clk)
        # leave the caller at the settled values of the new cycle
        await ReadOnly()
        if check:
            # err_k is a one-cycle pulse for exactly the starts that were refused
            refused = bool(tile is not None and not reset and not accepted
                           and not 1 <= tile.k <= self.k_max and self._was_ready)
            assert read_int(d.err_k, "err_k") == int(refused), (
                f"cycle {self.cycle}: err_k={read_int(d.err_k, 'err_k')}, "
                f"expected {int(refused)}, seed={self.seed}"
            )
        self.cycle += 1
        return accepted, beat

    def _watch_stream(self, tready):
        d = self.dut
        valid = read_int(d.m_axis_tvalid, "m_axis_tvalid")
        if not valid:
            assert self.blocked is None, (
                f"cycle {self.cycle}: tvalid dropped before the beat was taken, "
                f"seed={self.seed}"
            )
            return None
        raw = (read_int(d.m_axis_tdata, "m_axis_tdata"),
               read_int(d.m_axis_tuser, "m_axis_tuser"),
               read_int(d.m_axis_tlast, "m_axis_tlast"))
        if self.blocked is not None:
            assert raw == self.blocked, (
                f"cycle {self.cycle}: the beat changed while waiting for tready "
                f"({self.blocked} -> {raw}), seed={self.seed}"
            )
        if not tready:
            self.blocked = raw
            return None
        self.blocked = None
        got = Beat(self.model.unpack_bytes(raw[0]), raw[1], raw[2])
        assert self.expected, (
            f"cycle {self.cycle}: beat {got} came out with nothing owed "
            f"(a repeated beat?), seed={self.seed}"
        )
        exp = self.expected.pop(0)
        assert got == exp, (
            f"cycle {self.cycle}: beat {self.received} is {got}\n"
            f"                    expected {exp}\n"
            f"N={self.n} K={get_k()} READ_LATENCY={self.lat} seed={self.seed}"
        )
        self.received += 1
        if got.last:
            tile = self.running.pop(0)
            for q in (v for b in tile.beats for v in b.data):
                self.cover["clamp_hi"] += int(q == self.hi)
                self.cover["clamp_lo"] += int(q == self.lo and not tile.relu_en)
                self.cover["relu_floor"] += int(bool(tile.relu_en) and q == tile.zero_point)
                self.cover["in_range"] += int(self.lo < q < self.hi and q != tile.zero_point)
        return got

    def _watch_first_row(self):
        """On every cycle a row is handed to requant, all N shadow registers
        must be loaded and must hold this tile's parameters."""
        d = self.dut
        if not read_int(d.u_ctrl.c_valid, "c_valid") or read_int(d.stall, "stall"):
            return
        # the tile draining now is the newest one accepted
        tile = self.running[-1]
        loaded = read_int(d.p_loaded, "p_loaded")
        assert loaded == (1 << self.n) - 1, (
            f"cycle {self.cycle}: a row reached requant with p_loaded={loaded:#b}, "
            f"seed={self.seed}"
        )
        bias, mult, shift = (read_int(d.p_bias, "p_bias"), read_int(d.p_mult, "p_mult"),
                             read_int(d.p_shift, "p_shift"))
        for c, (e_bias, e_mult, e_shift) in enumerate(tile.params):
            got = ((bias >> (c * self.bw)) & ((1 << self.bw) - 1),
                   (mult >> (c * self.mw)) & ((1 << self.mw) - 1),
                   (shift >> (c * self.shw)) & ((1 << self.shw) - 1))
            exp = (e_bias & ((1 << self.bw) - 1), e_mult, e_shift)
            assert got == exp, (
                f"cycle {self.cycle}: channel {c} shadow parameters are {got}, "
                f"the tile's are {exp}, seed={self.seed}"
            )
        self.rows_checked += 1

    # building and running tiles

    def calibrated_params(self, a, b, zero_point):
        """Per-channel (bias, mult, shift) chosen the way calibration would:
        from the range A @ B actually takes in each column of this tile, so
        the outputs spread across the output range whatever K and N are.

        For a column with accumulators in [v_min, v_max] the scale maps that
        span onto a random 60 to 110 % of the output range (over 100 % clips
        the ends a little) and the bias places it at a random spot inside.
        """
        r = self.rng
        acc = np.asarray(a, dtype=np.int64) @ np.asarray(b, dtype=np.int64)
        m_max, s_max = (1 << self.mw) - 1, (1 << self.shw) - 1
        b_lo, b_hi = signed_range(self.bw)
        # requant output is scaled + zero_point, so scaled must land in [lo, hi]
        lo, hi = self.lo - zero_point, self.hi - zero_point
        out = []
        for c in range(self.n):
            v_min, v_max = int(acc[:, c].min()), int(acc[:, c].max())
            # the bias is an integer, so it can only place the column in steps
            # of one accumulator LSB times the scale. a floor on the span keeps
            # that step to a few output LSBs when a column is nearly constant
            # (always the case for N = 1, where a column is a single value).
            span = max(64, v_max - v_min)
            scale = r.uniform(0.6, 1.1) * (hi - lo) / span
            shift = min(s_max, max(0, int(math.floor(math.log2(m_max / scale)))))
            mult = min(m_max, max(1, round(scale * (1 << shift))))
            real = mult / (1 << shift)
            # put the column anywhere it still fits; a single value (N = 1)
            # has the whole range to move in
            slack = max(0.0, 1.0 - (v_max - v_min) * real / (hi - lo))
            centre = (lo + hi) / 2 + r.uniform(-0.5, 0.5) * slack * (hi - lo)
            bias = round(centre / real) - (v_min + v_max) // 2
            out.append((min(max(bias, b_lo), b_hi), mult, shift))
        return out

    def random_params(self, k=None):
        """Per-channel (bias, mult, shift) with no regard for the data: random
        or extreme scales, which mostly drive the output into the clamps."""
        r = self.rng
        b_lo, b_hi = signed_range(self.bw)
        return [(r.choice((b_lo, b_hi, r.randint(b_lo, b_hi), r.randint(-1000, 1000))),
                 r.randrange(1 << self.mw), r.randrange(1 << self.shw))
                for _ in range(self.n)]

    def clamping_params(self, first_high):
        """Largest scale with the bias pinned at alternate extremes, so
        neighbouring channels clamp opposite ways whatever the operands are."""
        b_lo, b_hi = signed_range(self.bw)
        m_max = (1 << self.mw) - 1
        return [(b_hi if (c % 2 == 0) == bool(first_high) else b_lo, m_max, 0)
                for c in range(self.n)]

    def random_bases(self, k):
        """Fresh bases for one tile. Each memory is handed out in order from a
        random starting point, with a random gap, so consecutive tiles never
        overlap and a tile regularly wraps past the top of a memory."""
        bases = tuple(self.next_base)
        for i, (words, width) in enumerate(((k, self.b_aw), (self.n, self.p_aw))):
            gap = self.rng.randint(0, 3) if words * 8 <= (1 << width) else 0
            self.next_base[i] = (self.next_base[i] + words + gap) % (1 << width)
        return bases

    def make_tile(self, k, bases=None, a=None, b=None, params=None,
                  zero_point=None, relu_en=None):
        """Decide a tile's contents. Returns the Tile and its memory writes;
        the beats are filled in by load(), once the model has the data."""
        r = self.rng
        if k > self.k_max or k > (1 << self.b_aw):
            raise ValueError(f"K={k} does not fit this configuration")
        b_base, p_base = bases or self.random_bases(k)
        a = a or [[r.randint(self.lo, self.hi) for _ in range(k)] for _ in range(self.n)]
        b = b or [[r.randint(self.lo, self.hi) for _ in range(self.n)] for _ in range(k)]
        zp = r.randint(self.lo, self.hi) if zero_point is None else zero_point
        relu = r.randint(0, 1) if relu_en is None else relu_en
        if params is None:
            # mostly calibrated to the data; now and then scales that ignore it
            params = (self.random_params() if r.random() < 0.15
                      else self.calibrated_params(a, b, zp))
        writes = []
        for j in range(max(k, self.n)):
            w = {}
            if j < k:
                w["b"] = ((b_base + j) % (1 << self.b_aw), list(b[j]))
            if j < self.n:
                w["p"] = ((p_base + j) % (1 << self.p_aw), params[j])
            writes.append(w)
        slices = [[a[i][j] for i in range(self.n)] for j in range(k)]
        return Tile(k, b_base, p_base, zp, relu, None, params, slices), writes

    async def load(self, tile, writes, tready=0):
        for w in writes:
            await self.step(writes=w, tready=tready)
        tile.beats = self.model.run_tile(tile.slices, tile.b_base, tile.p_base,
                                         tile.zero_point, tile.relu_en)
        return tile

    async def new_tile(self, k, **kw):
        tile, writes = self.make_tile(k, **kw)
        return await self.load(tile, writes)

    async def run(self, tiles, ready_kind="always", feed_kind="always", limit=None):
        """Run tiles one after another and wait for the last beat of the last
        one. Returns the number of cycles from the first accept."""
        ready = ready_pattern(self.rng, ready_kind)
        self.feed_gen = feed_pattern(self.rng, feed_kind, self.n)
        queue = list(tiles)
        owed = self.received + len(self.expected) + sum(len(t.beats) for t in queue)
        limit = limit or 600 * sum(t.k + 3 * self.n + 12 for t in tiles) + 2000
        first_accept = None
        for _ in range(limit):
            accepted, _ = await self.step(tile=queue[0] if queue else None,
                                          tready=next(ready))
            if accepted:
                queue.pop(0)
                if first_accept is None:
                    first_accept = self.cycle - 1
            if not queue and self.received == owed:
                break
        else:
            raise AssertionError(
                f"tiles did not finish ({ready_kind}): {len(self.expected)} beats "
                f"still owed, seed={self.seed}"
            )
        assert not self.expected and not self.running
        return self.cycle - first_accept

    async def settle(self, cycles=4, tready=1):
        """Idle cycles; nothing may come out."""
        for _ in range(cycles):
            _, beat = await self.step(tready=tready)
            assert beat is None, "a beat appeared with no tile running"


def get_k():
    return int(os.environ.get("K") or 8)


# TESTS


@cocotb.test()
async def test_random_tile(dut):
    """One tile, random operands, parameters and base addresses."""
    h = GemmHarness(dut)
    await h.start()
    tile = await h.new_tile(get_k())
    await h.run([tile])
    await h.settle()
    assert h.rows_checked == h.n, "the parameter check did not see every row"


@cocotb.test()
async def test_latency_and_cycles_per_tile(dut):
    """Measure the latencies in the header comment on the DUT, tready high."""
    h = GemmHarness(dut)
    await h.start()
    k = get_k()
    tiles = [await h.new_tile(k) for _ in range(3)]

    seen = {}
    accept_cycles = []
    queue = list(tiles)
    cycle = 0
    while len(accept_cycles) < 3 or h.expected:
        accepted, beat = await h.step(tile=queue[0] if queue else None, tready=1)
        if accepted:
            accept_cycles.append(cycle)
            queue.pop(0)
        # the first N beats are tile 0's; its last rows leave after tile 1
        # has already been accepted
        if beat is not None and h.received <= h.n:
            seen.setdefault("first_beat", cycle)
            seen["last_beat"] = cycle
        cycle += 1
        assert cycle < 50 * (k + 3 * h.n + 20)

    # a signal seen in cycle c went high c - 1 edges after the accepting edge
    # (cycle 0 is the accepting cycle)
    t0 = accept_cycles[0]
    measured = {
        "first_beat": seen["first_beat"] - t0 - 1,
        "last_beat": seen["last_beat"] - t0 - 1,
        "period": accept_cycles[1] - accept_cycles[0],
        "period_2": accept_cycles[2] - accept_cycles[1],
    }
    expect = {
        "first_beat": h.model.first_beat_latency(k),
        "last_beat": h.model.last_beat_latency(k),
        "period": h.model.tile_period(k),
        "period_2": h.model.tile_period(k),
    }
    assert measured == expect, f"measured {measured}\nexpected {expect}"
    dut._log.info(
        f"cycles per tile: N={h.n} K={k} READ_LATENCY={h.lat} -> "
        f"{measured['period']} cycles start to start, first beat after "
        f"{measured['first_beat']} edges, last after {measured['last_beat']}"
    )


@cocotb.test()
async def test_k1_params_loaded_in_time(dut):
    """K=1 is the shortest window for the parameter loader: the first row
    drains soonest after start. The check in step() runs on every row."""
    h = GemmHarness(dut)
    await h.start()
    for kind in ("always", "random", "toggle"):
        tiles = [await h.new_tile(1) for _ in range(3)]
        before = h.rows_checked
        await h.run(tiles, kind)
        assert h.rows_checked - before == 3 * h.n
    dut._log.info(f"parameters loaded before the first row in {h.rows_checked // h.n} K=1 tiles")


@cocotb.test()
async def test_backpressure_patterns(dut):
    """Random tready, long low stretches and single-cycle toggles: no beat
    dropped or repeated, order kept."""
    h = GemmHarness(dut)
    await h.start()
    k = get_k()
    for kind in ("random", "mostly_low", "toggle", "long_low", "mixed"):
        tiles = [await h.new_tile(k) for _ in range(3)]
        got_before, stall_before = h.received, h.stall_cycles
        await h.run(tiles, kind)
        assert h.received - got_before == 3 * h.n
        await h.settle()
        # a stall needs a third beat to arrive while two are held. with N = 1
        # a tile is one beat, so the next one is a whole tile away and the
        # buffer alone absorbs it.
        if kind in ("mostly_low", "long_low") and h.n >= 2:
            assert h.stall_cycles > stall_before, f"{kind} never stalled the datapath"
    assert h.rows_checked == 15 * h.n
    dut._log.info(f"{h.received} beats in order through 5 backpressure styles, "
                  f"{h.stall_cycles} stalled cycles")


@cocotb.test()
async def test_output_held_off_for_whole_tile(dut):
    """tready low from start: the tile parks with two beats in the skid buffer
    and everything behind them frozen, then drains intact."""
    h = GemmHarness(dut)
    await h.start()
    k = get_k()
    tile = await h.new_tile(k)
    accepted = False
    for _ in range(h.model.tile_period(k) + 60):
        took, beat = await h.step(tile=None if accepted else tile, tready=0)
        accepted = accepted or took
        assert beat is None
    assert read_int(dut.m_axis_tvalid, "m_axis_tvalid") == 1
    if h.n >= 2:
        # with N = 1 the single beat fits in the buffer and nothing stalls
        assert read_int(dut.stall, "stall") == 1, "datapath not stalled with the output blocked"
        assert read_int(dut.ready, "ready") == 0, "ready high while stalled"
    assert h.received == 0
    # release it one cycle at a time, then fully
    for _ in range(3):
        await h.step(tready=1)
        await h.step(tready=0)
        await h.step(tready=0)
    for _ in range(h.n + 30):
        await h.step(tready=1)
    assert h.received == h.n and not h.expected, f"got {h.received} of {h.n} beats"


@cocotb.test()
async def test_back_to_back_tiles(dut):
    """start held high through tiles of different K, bases and parameters.

    While one tile runs the command pins already carry the next one, so this
    also proves the latched command cannot change mid-tile.
    """
    h = GemmHarness(dut)
    await h.start()
    k = get_k()
    ks = [k, 1, min(k + 3, h.k_max), 2, k]
    tiles = []
    for i, kk in enumerate(ks):
        # separate regions, so loading one cannot touch another
        region = (1 << h.b_aw) // len(ks)
        bases = (((i + 2) % len(ks)) * region,
                 (i * h.n) % (1 << h.p_aw) if (len(ks) * h.n) <= (1 << h.p_aw) else None)
        if bases[1] is None or kk > region:
            raise ValueError("memories too small for this test")
        tiles.append(await h.new_tile(kk, bases=bases))
    for kind in ("always", "random"):
        await h.run(tiles, kind)
        await h.settle()
    assert h.accepts == 2 * len(ks)
    assert h.rows_checked == 2 * len(ks) * h.n


@cocotb.test()
async def test_reload_same_addresses(dut):
    """Nothing is baked in: rewrite the same words and the result follows."""
    h = GemmHarness(dut)
    await h.start()
    k = get_k()
    bases = h.random_bases(k)
    results = []
    for _ in range(3):
        # each tile must give a different result from the one before, or the
        # rewrite proves nothing. at N=1 a whole tile is one output byte, and
        # random scales often clamp it to a rail, so a fresh draw repeats the
        # previous result now and then; draw again rather than fail (CI hit
        # this once with an unlucky seed). loading a redrawn tile just
        # rewrites the same words.
        for _ in range(50):
            tile = await h.new_tile(k, bases=bases)
            if not results or tile.beats != results[-1]:
                break
        results.append(tile.beats)
        await h.run([tile], "random")
    assert results[0] != results[1] and results[1] != results[2], (
        "three random tiles gave identical results; the test proves nothing"
    )

    # parameters only: same operands, new bias/mult/shift
    tile, writes = h.make_tile(k, bases=bases)
    await h.load(tile, writes)
    await h.run([tile])
    params = h.random_params()
    for c in range(h.n):
        await h.step(writes={"p": ((bases[1] + c) % (1 << h.p_aw), params[c])})
    tile2 = Tile(k, *bases, tile.zero_point, tile.relu_en,
                 h.model.run_tile(tile.slices, *bases, tile.zero_point, tile.relu_en),
                 params, tile.slices)
    await h.run([tile2])


@cocotb.test()
async def test_extreme_values(dut):
    """Signed extremes on every operand, with parameters at their limits."""
    h = GemmHarness(dut)
    await h.start()
    k = get_k()
    b_lo, b_hi = signed_range(h.bw)
    m_max, s_max = (1 << h.mw) - 1, (1 << h.shw) - 1
    corners = [(b_hi, m_max, 0), (b_lo, m_max, 0), (b_hi, 1, s_max), (b_lo, 1, s_max),
               (0, 0, 0), (0, m_max, s_max), (b_hi, m_max, s_max), (b_lo, m_max, s_max)]
    for va, vb in ((h.lo, h.lo), (h.lo, h.hi), (h.hi, h.lo), (h.hi, h.hi)):
        a = [[va] * k for _ in range(h.n)]
        b = [[vb] * h.n for _ in range(k)]
        for relu in (0, 1):
            params = [corners[(c + relu) % len(corners)] for c in range(h.n)]
            tile = await h.new_tile(k, a=a, b=b, params=params, relu_en=relu,
                                    zero_point=h.rng.choice((h.lo, -1, 0, h.hi)))
            await h.run([tile], "random")


@cocotb.test()
async def test_requant_range_is_exercised(dut):
    """Many tiles with realistic scales; the outputs must cover the range and
    not just sit on the clamps."""
    h = GemmHarness(dut)
    await h.start()
    k = get_k()
    r = h.rng
    for i in range(12):
        a = [[r.randint(h.lo, h.hi) for _ in range(k)] for _ in range(h.n)]
        b = [[r.randint(h.lo, h.hi) for _ in range(h.n)] for _ in range(k)]
        zp = r.randint(h.lo, h.hi)
        if i < 10:
            # calibrated to this tile's own accumulator range
            tile = await h.new_tile(k, a=a, b=b, zero_point=zp, relu_en=i % 2,
                                    params=h.calibrated_params(a, b, zp))
        else:
            # deliberately extreme scales: every channel clamps, both ways
            tile = await h.new_tile(k, a=a, b=b, zero_point=zp, relu_en=0,
                                    params=h.clamping_params(first_high=(i == 10)))
        await h.run([tile])
    total = 12 * h.n * h.n
    assert h.cover["in_range"] > total // 4, f"mostly clamped: {h.cover}, seed={h.seed}"
    for name in ("clamp_hi", "clamp_lo"):
        assert h.cover[name] > 0, f"{name} never hit: {h.cover}, seed={h.seed}"
    if total >= 48:
        assert h.cover["relu_floor"] > 0, f"relu_floor never hit: {h.cover}, seed={h.seed}"
    dut._log.info(f"output coverage over {total} values: {h.cover}")


@cocotb.test()
async def test_reset_mid_tile(dut):
    """Reset at points through the tile, including with beats waiting in the
    skid buffer. Nothing leaks out afterwards and memory is untouched: the
    same tile runs correctly again without being reloaded."""
    h = GemmHarness(dut)
    await h.start()
    k = get_k()
    tile = await h.new_tile(k)
    period = h.model.tile_period(k)
    points = sorted({1, 2, h.lat + 1, k, k + h.n, k + 2 * h.n + h.lat,
                     k + 2 * h.n + h.lat + 3, h.model.first_beat_latency(k) + 1,
                     period - 1, period + 3})
    for at in points:
        for tready in (1, 0):
            accepted = False
            for _ in range(at):
                took, _ = await h.step(tile=None if accepted else tile, tready=tready)
                accepted = accepted or took
            await h.step(reset=1, tready=tready)
            assert read_int(dut.m_axis_tvalid, "m_axis_tvalid") == 0
            assert read_int(dut.busy, "busy") == 0
            await h.step(reset=1)
            for _ in range(period + 10):
                _, beat = await h.step(tready=1)
                assert beat is None, f"a beat survived reset at cycle {at}"
            await h.run([tile], h.rng.choice(("always", "random")))
    dut._log.info(f"reset at {len(points)} points x 2, tile re-run from memory each time")


@cocotb.test()
async def test_tready_is_registered(dut):
    """m_axis_tready must not reach stall, ready or the stream within a cycle."""
    h = GemmHarness(dut)
    await h.start()
    k = get_k()
    tile = await h.new_tile(k)
    accepted = False
    for _ in range(h.model.tile_period(k) + 20):
        took, _ = await h.step(tile=None if accepted else tile, tready=0)
        accepted = accepted or took

    def snapshot():
        return tuple(read_int(getattr(dut, s), s) for s in
                     ("stall", "ready", "busy", "m_axis_tvalid", "m_axis_tdata",
                      "m_axis_tuser", "m_axis_tlast"))

    await FallingEdge(dut.clk)
    before = snapshot()
    dut.m_axis_tready.value = 1
    await Timer(1, unit="ns")
    await ReadOnly()
    assert snapshot() == before, "tready reached an output combinationally"
    await FallingEdge(dut.clk)
    # that edge took a beat behind the harness's back; resynchronise
    await h.step(reset=1)


@cocotb.test()
async def test_feeder_gaps(dut):
    """The feeder is late in every style, against every output style. The
    tile has to stall for it (stall_feed) and still deliver every beat once."""
    h = GemmHarness(dut)
    await h.start()
    k = get_k()
    # enough slices per run that a late feeder is certain to be caught out,
    # however short the tiles are
    n_tiles = max(2, -(-24 // k))
    for feed in ("random", "rare", "toggle", "gather"):
        before = h.feed_stall_cycles
        for ready in ("always", "random", "long_low"):
            # loaded in batches small enough that their parameter words fit
            # the parameter memory side by side
            left = n_tiles
            while left:
                batch = min(left, (1 << h.p_aw) // (h.n + 3))
                tiles = [await h.new_tile(k) for _ in range(batch)]
                await h.run(tiles, ready, feed)
                left -= batch
            await h.settle()
        # judged over the three output styles together: with tready always
        # high a regular feeder can stay in step with a regular tile period
        assert h.feed_stall_cycles > before, (
            f"feed={feed} never stalled the tile; the test did not cover stall_feed"
        )
    assert h.rows_checked == 12 * n_tiles * h.n
    dut._log.info(f"{h.feed_stall_cycles} cycles stalled by the feeder, "
                  f"{h.stall_cycles} stalled in total")


@cocotb.test()
async def test_feeder_stall_with_row_at_requant_output(dut):
    """Back-to-back tiles where the next tile's first slice is late. The last
    rows of the previous tile are still coming out of requant at that moment,
    so a row sits at the requant output with the skid buffer open while the
    datapath is frozen by the feeder. It must be handed over exactly once."""
    h = GemmHarness(dut)
    await h.start()
    k = get_k()
    tiles = [await h.new_tile(k) for _ in range(6)]
    queue = list(tiles)
    owed = sum(len(t.beats) for t in tiles)
    late = 0
    for _ in range(40 * sum(t.k + 3 * h.n + 20 for t in tiles)):
        accepted, _ = await h.step(tile=queue[0] if queue else None, tready=1)
        if accepted:
            queue.pop(0)
            # a different lateness each tile, to sweep the overlap
            late += 1
            h.feed_hold = h.lat + late + 2
        if not queue and h.received == owed:
            break
    assert h.received == owed and not h.expected
    # the next tile's first slice is due READ_LATENCY + 1 cycles after its
    # accept, and the previous tile's last row is at the requant output 2
    # cycles after that accept. so the two only meet at READ_LATENCY = 1; at 2
    # the row has always gone one cycle before the feeder can stall anything.
    if h.lat == 1:
        assert h.dup_risk_cycles > 0, (
            "never had a valid requant output, an open skid buffer and a feeder "
            "stall on the same cycle; the test did not reach the case it is for"
        )
    else:
        assert h.dup_risk_cycles == 0, (
            "a row was held at the requant output by stall_feed at READ_LATENCY=2; "
            "the timing argument in this test is wrong"
        )
    dut._log.info(f"{h.dup_risk_cycles} cycles with a row held at the requant output by stall_feed")


@cocotb.test()
async def test_k_at_k_max(dut):
    """The deepest tile the buffers hold. Only where it fits the weight memory
    in this build; the sweep has a configuration for it."""
    h = GemmHarness(dut)
    await h.start(fill=False)
    if h.k_max > (1 << h.b_aw):
        dut._log.info(f"K_MAX={h.k_max} does not fit B_ADDR_W={h.b_aw} here; "
                      f"covered by the B_ADDR_W=13 sweep point")
        return
    tile = await h.new_tile(h.k_max)
    await h.run([tile], "random", "random")
    dut._log.info(f"K = K_MAX = {h.k_max} passed")


@cocotb.test(skip=os.environ.get("ALLOW_K0") != "1")
async def test_bad_k_start_ignored(dut):
    """A start with k_in == 0 or k_in > K_MAX changes nothing and raises err_k.

    The command latches and the parameter loader use the same accept as
    ctrl_fsm, so neither may move. step() checks err_k on every cycle.

    Needs +ctrl_fsm_allow_k0 +gemm_tile_allow_bad_k (make ALLOW_K0=1) to
    switch off the simulation-only $fatal for each case.
    """
    h = GemmHarness(dut)
    await h.start()
    k = get_k()
    tile = await h.new_tile(k)
    await h.run([tile])

    def latched():
        return tuple(read_int(getattr(dut, name), name) for name in
                     ("p_loaded", "p_req", "b_base_q", "p_base_q", "zp_q", "relu_q"))

    before = latched()
    assert before[0] == (1 << h.n) - 1 and before[1] == 0
    bad_ks = [0, h.k_max + 1, (1 << h.kw) - 1]
    for bad_k in bad_ks:
        # every field differs from what the last tile latched
        bogus = Tile(bad_k, tile.b_base ^ 1, tile.p_base ^ 1, tile.zero_point ^ 1,
                     tile.relu_en ^ 1, [], tile.params, [])
        for _ in range(3):
            accepted, _ = await h.step(tile=bogus, tready=1)
            assert not accepted
            assert read_int(dut.err_k, "err_k") == 1, f"no err_k for k_in={bad_k}"
            assert latched() == before, (
                f"a k_in={bad_k} start changed the latched command or restarted "
                f"the parameter loader: {before} -> {latched()}"
            )
        for _ in range(h.model.tile_period(k)):
            _, beat = await h.step(tready=1)
            assert beat is None and read_int(dut.busy, "busy") == 0
            assert latched() == before
    # the same starts while a tile is running are not starts at all: ready is
    # low, so err_k must stay low too (step() checks it on every cycle)
    accepted, _ = await h.step(tile=tile, tready=1)
    assert accepted
    busy_cycles = 0
    while h.expected:
        bogus = Tile(h.rng.choice(bad_ks), tile.b_base ^ 1, tile.p_base ^ 1,
                     tile.zero_point ^ 1, tile.relu_en ^ 1, [], tile.params, [])
        await h.step(tile=bogus if read_int(dut.busy, "busy") else None, tready=1)
        busy_cycles += read_int(dut.busy, "busy")
    assert busy_cycles > k
    await h.settle()
    await h.run([tile], "random")
    dut._log.info(f"starts with k_in in {bad_ks} ignored, err_k raised")
