""" cocotb tb for rtl/skid_buffer.sv """

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer

from skid_buffer_model import SkidBufferModel, SkidState


def read_int(sig, name):
    raw = str(sig.value)
    if "x" in raw.lower() or "z" in raw.lower():
        raise AssertionError(f"{name} is not resolvable: {raw}")
    return int(sig.value)


def ready_pattern(rng, kind):
    """Endless m_ready generator for one backpressure style."""
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
            for _ in range(rng.randint(20, 80)):
                yield 0
            for _ in range(rng.randint(1, 12)):
                yield 1
        elif kind == "mixed":
            sub = ready_pattern(rng, rng.choice(("always", "random", "toggle", "long_low")))
            for _ in range(rng.randint(10, 120)):
                yield next(sub)
        else:
            raise ValueError(kind)


class SkidHarness:
    """Drives the DUT and the golden model from one input vector.

    step() compares every register against SkidBufferModel. Separately, a
    scoreboard records each beat that crosses s_* and each that crosses m_*
    using the DUT's own handshake signals, and the two lists must match:
    that check does not involve the model at all.
    """

    FIELDS = SkidState._fields

    def __init__(self, dut):
        self.dut = dut
        self.dw = int(dut.DATA_WIDTH.value)
        self.d_max = (1 << self.dw) - 1
        self.model = SkidBufferModel(self.dw)
        self.cycle = 0
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = random.Random(self.seed)
        self.sent = []
        self.received = []
        self.max_occupancy = 0
        self.blocked = None     # m_data being held while m_valid & ~m_ready

    async def start(self, reset_cycles=3):
        d = self.dut
        d.reset.value = 1
        d.s_valid.value = 0
        d.s_data.value = 0
        d.m_ready.value = 0
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        # registers are X until the first reset edge
        await self.step(reset=1, check_pre=False)
        for _ in range(reset_cycles - 1):
            await self.step(reset=1)
        d._log.info(f"DUT parameters: DATA_WIDTH={self.dw} SEED={self.seed} (reproduce "
                    f"with: make MODULE_NAME=skid_buffer DATA_WIDTH={self.dw} SEED={self.seed})")

    def _read(self):
        d = self.dut
        return SkidState(**{f: read_int(getattr(d, f), f) for f in self.FIELDS})

    async def step(self, s_valid=0, s_data=None, m_ready=0, reset=0, check_pre=True):
        """Advance one rising edge on both DUT and model, then compare.
        Returns (beat accepted on s, beat delivered on m) for this edge."""
        if s_data is None:
            s_data = self.rng.randint(0, self.d_max)
        d = self.dut
        await FallingEdge(d.clk)
        d.s_valid.value = s_valid
        d.s_data.value = s_data
        d.m_ready.value = m_ready
        d.reset.value = reset

        took = gave = False
        if check_pre:
            # outputs are registers, so what is on the pins now is what both
            # neighbours sample at the coming edge
            await ReadOnly()
            pre = self._read()
            if reset:
                self.sent, self.received, self.blocked = [], [], None
            else:
                took = bool(s_valid and pre.s_ready)
                gave = bool(pre.m_valid and m_ready)
                if took:
                    self.sent.append(s_data)
                if gave:
                    self.received.append(pre.m_data)
                # AXI-Stream: once valid, data holds until the beat is taken
                if self.blocked is not None:
                    assert pre.m_valid and pre.m_data == self.blocked, (
                        f"cycle {self.cycle}: m_* changed while waiting for m_ready "
                        f"(was {self.blocked:#x}, now valid={pre.m_valid} "
                        f"data={pre.m_data:#x}), seed={self.seed}"
                    )
                self.blocked = pre.m_data if (pre.m_valid and not m_ready) else None

        await RisingEdge(d.clk)
        await ReadOnly()
        expected = self.model.step(s_valid=s_valid, s_data=s_data, m_ready=m_ready, reset=reset)
        got = self._read()
        assert got == expected, (
            f"cycle {self.cycle}: dut {got}\n            model {expected}\n"
            f"stimulus: s_valid={s_valid} s_data={s_data:#x} m_ready={m_ready} reset={reset}\n"
            f"seed={self.seed}"
        )
        self.max_occupancy = max(self.max_occupancy, got.m_valid + got.skid_valid)
        self.cycle += 1
        return took, gave

    async def run_stream(self, n_beats, p_valid, ready_kind, limit=200000):
        """Push n_beats through and drain. The source holds each beat until it
        is taken, as an AXI-Stream master must."""
        ready = ready_pattern(self.rng, ready_kind)
        self.sent, self.received = [], []
        beats = [self.rng.randint(0, self.d_max) for _ in range(n_beats)]
        i = 0
        offering = False
        for _ in range(limit):
            if i < n_beats and not offering:
                offering = self.rng.random() < p_valid
            if i < n_beats and offering:
                took, _ = await self.step(s_valid=1, s_data=beats[i], m_ready=next(ready))
                if took:
                    i += 1
                    offering = False
            else:
                await self.step(m_ready=next(ready))
            if i == n_beats and len(self.received) == n_beats:
                break
        else:
            raise AssertionError(f"stream did not finish ({ready_kind}), seed={self.seed}")
        assert self.sent == beats, "the source side lost track of its own beats"
        assert self.received == beats, (
            f"{ready_kind}: beats dropped, repeated or reordered\n"
            f"first difference at index "
            f"{next(k for k, (a, b) in enumerate(zip(self.received, beats)) if a != b)}"
            f", seed={self.seed}"
        )


# TESTS


@cocotb.test()
async def test_reset_and_latency(dut):
    """s_ready timing out of reset, and one edge of latency when empty."""
    h = SkidHarness(dut)
    await h.start()
    assert read_int(dut.s_ready, "s_ready") == 0, "s_ready high during reset"
    await h.step()
    assert read_int(dut.s_ready, "s_ready") == 1, "s_ready not up one edge after reset"
    assert read_int(dut.m_valid, "m_valid") == 0

    took, _ = await h.step(s_valid=1, s_data=h.d_max)
    assert took
    assert (read_int(dut.m_valid, "m_valid"), read_int(dut.m_data, "m_data")) == (1, h.d_max), (
        "beat was not on m_* one edge after it was accepted"
    )
    await h.step(m_ready=1)
    assert read_int(dut.m_valid, "m_valid") == 0, "m_valid lasted after the beat was taken"


@cocotb.test()
async def test_fills_to_two_and_blocks(dut):
    """Blocked output: two beats are held, the third is refused, order kept."""
    h = SkidHarness(dut)
    await h.start()
    await h.step()
    a, b, c = 1, h.d_max, 1 if h.dw == 1 else 2
    assert (await h.step(s_valid=1, s_data=a))[0]
    assert (await h.step(s_valid=1, s_data=b))[0]
    assert read_int(dut.s_ready, "s_ready") == 0, "s_ready still high with both entries full"
    for _ in range(5):
        took, _ = await h.step(s_valid=1, s_data=c)
        assert not took, "a third beat was accepted into a two-entry buffer"
    await h.step(s_valid=1, s_data=c, m_ready=1)
    for _ in range(4):
        await h.step(m_ready=1)
    assert h.received[:2] == [a, b], f"got {h.received}"
    assert h.max_occupancy == 2


@cocotb.test()
async def test_full_throughput(dut):
    """Nothing blocking: one beat per cycle, n beats in n + 1 edges."""
    h = SkidHarness(dut)
    await h.start()
    await h.step()
    n = 200
    for i in range(n):
        took, _ = await h.step(s_valid=1, s_data=(i * 7 + 1) & h.d_max, m_ready=1)
        assert took, f"s_ready dropped at beat {i} with m_ready held high"
    await h.step(m_ready=1)
    assert len(h.received) == n and h.received == h.sent
    assert h.max_occupancy == 1, "the skid entry was used with no backpressure"


@cocotb.test()
async def test_backpressure_patterns(dut):
    """Every backpressure style against several source rates."""
    h = SkidHarness(dut)
    await h.start()
    await h.step()
    for kind in ("always", "random", "mostly_low", "toggle", "long_low", "mixed"):
        for p_valid in (1.0, 0.5, 0.1):
            await h.run_stream(300, p_valid, kind)
    assert h.max_occupancy == 2
    dut._log.info("no beat dropped, repeated or reordered in 18 pattern/rate pairs")


@cocotb.test()
async def test_single_cycle_ready_pulses(dut):
    """m_ready high for exactly one cycle at a time, at every phase."""
    h = SkidHarness(dut)
    await h.start()
    await h.step()
    beats = [h.rng.randint(0, h.d_max) for _ in range(120)]
    i = 0
    gap = 0
    while len(h.received) < len(beats):
        gap = (gap + 1) % 7
        took, _ = await h.step(s_valid=int(i < len(beats)),
                               s_data=beats[min(i, len(beats) - 1)],
                               m_ready=int(gap in (0, 3)))
        i += int(took)
    assert h.received == beats, f"seed={h.seed}"


@cocotb.test()
async def test_reset_drops_contents(dut):
    """Reset with 0, 1 and 2 entries held; nothing comes out afterwards."""
    h = SkidHarness(dut)
    await h.start()
    for held in (0, 1, 2):
        await h.step()
        for _ in range(held):
            await h.step(s_valid=1, s_data=h.d_max)
        await h.step(s_valid=1, s_data=h.d_max, reset=1)
        assert read_int(dut.m_valid, "m_valid") == 0
        assert read_int(dut.s_ready, "s_ready") == 0
        await h.step(reset=1)
        for _ in range(4):
            _, gave = await h.step(m_ready=1)
            assert not gave, f"a beat survived reset (held={held})"
        await h.run_stream(20, 0.7, "random")


@cocotb.test()
async def test_valid_held_through_reset(dut):
    """A source that keeps s_valid high across reset. s_ready is still low on
    the first cycle after release, so that beat must not be taken twice."""
    h = SkidHarness(dut)
    await h.start()
    for m_ready in (1, 0):
        for _ in range(2):
            await h.step(s_valid=1, s_data=h.d_max, m_ready=m_ready, reset=1)
        beats = [h.d_max, 1, h.d_max >> 1 or 1, 1]
        i = 0
        took, _ = await h.step(s_valid=1, s_data=beats[0], m_ready=m_ready)
        assert not took, "a beat was accepted while s_ready was still low after reset"
        assert read_int(dut.m_valid, "m_valid") == 0, (
            "a beat reached m_* that the source was never told had been taken"
        )
        while len(h.received) < len(beats):
            took, _ = await h.step(s_valid=int(i < len(beats)),
                                   s_data=beats[min(i, len(beats) - 1)], m_ready=1)
            i += int(took)
        assert h.received == beats, f"got {h.received}, sent {beats}"


@cocotb.test()
async def test_ready_is_registered(dut):
    """m_ready must not reach s_ready, m_valid or m_data within the cycle."""
    h = SkidHarness(dut)
    await h.start()
    await h.step()
    await h.step(s_valid=1, s_data=h.d_max)
    await h.step(s_valid=1, s_data=1)
    # both entries full, s_ready low: raising m_ready may only act at the edge
    await FallingEdge(dut.clk)
    before = h._read()
    dut.m_ready.value = 1
    await Timer(1, unit="ns")
    await ReadOnly()
    assert h._read() == before, "m_ready reached an output combinationally"
    await RisingEdge(dut.clk)
    h.model.step(m_ready=1)
