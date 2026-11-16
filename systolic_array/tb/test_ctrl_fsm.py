""" cocotb tb for rtl/ctrl_fsm.sv """

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from ctrl_fsm_model import (CtrlFsmModel, CtrlState, TileCosim, STATE_NAMES,
                            S_IDLE, S_STREAM, S_FLUSH, S_DRAIN, S_DONE)
from systolic_array_model import expected_matmul
from pe_model import signed_range


def read_int(sig, name):
    raw = str(sig.value)
    if "x" in raw.lower() or "z" in raw.lower():
        raise AssertionError(f"{name} is not resolvable: {raw}")
    return int(sig.value)


class CtrlHarness:
    """Drives the DUT and the golden model from one input vector.

    Same discipline as PEHarness and ArrayHarness: every cycle goes through
    step(). Each cycle is checked twice against CtrlFsmModel, once before the
    edge (the values the array, feeder and consumer actually sample, with this
    cycle's stall) and once after it.

    The DUT outputs also drive a TileCosim (feeder + mesh model of the array),
    whose rows are judged against expected_matmul. CtrlFsmModel is not in that
    loop, so an error shared by the FSM and its model still shows up there.
    """

    FIELDS = CtrlState._fields

    def __init__(self, dut):
        self.dut = dut
        self.n = int(dut.N.value)
        self.kw = int(dut.K_WIDTH.value)
        self.ml = int(dut.MEM_LATENCY.value)
        self.k_max = (1 << self.kw) - 1
        self.model = CtrlFsmModel(self.n, self.kw, self.ml)
        self.cycle = 0
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = random.Random(self.seed)
        self.cosim = TileCosim(self.n, self.ml, self.rng)
        self.accepts = 0        # start & ready edges, as seen on the DUT pins
        self.states_stalled = set()
        self.states_reset = set()

    # lifecycle

    async def start(self, reset_cycles=3):
        d = self.dut
        d.reset.value = 1
        d.start.value = 0
        d.k_in.value = 0
        d.stall.value = 0
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        # registers are X until the first reset edge, so skip that one compare
        await self.step(reset=1, check_pre=False)
        for _ in range(reset_cycles - 1):
            await self.step(reset=1)
        d._log.info(
            f"DUT parameters: N={self.n} K_WIDTH={self.kw} MEM_LATENCY={self.ml} "
            f"K={get_k()} SEED={self.seed} (reproduce with: make MODULE_NAME=ctrl_fsm "
            f"N={self.n} K={get_k()} K_WIDTH={self.kw} MEM_LATENCY={self.ml} "
            f"SEED={self.seed})"
        )

    def _read(self):
        d = self.dut
        return CtrlState(**{f: read_int(getattr(d, f), f) for f in self.FIELDS})

    def _compare(self, got, exp, when, stim):
        for f in self.FIELDS:
            g, e = getattr(got, f), getattr(exp, f)
            assert g == e, (
                f"cycle {self.cycle} ({when} edge): {f}={g}, expected {e}\n"
                f"stimulus: {stim}\n"
                f"dut   {got}\nmodel {exp}\n"
                f"N={self.n} K={get_k()} MEM_LATENCY={self.ml} seed={self.seed}"
            )

    async def step(self, start=0, k_in=None, stall=0, reset=0, check_pre=True):
        """Advance one rising edge on both DUT and model, comparing both sides
        of it. Returns the DUT outputs as they were during the cycle."""
        if k_in is None:
            # never leave k_in at an idle value the FSM could latch unnoticed
            k_in = self.rng.randint(1, self.k_max)
        stim = f"start={start} k_in={k_in} stall={stall} reset={reset}"

        d = self.dut
        await FallingEdge(d.clk)
        d.start.value = start
        d.k_in.value = k_in
        d.stall.value = stall
        d.reset.value = reset

        await ReadOnly()
        if check_pre:
            pre = self._read()
            self._compare(pre, self.model.view(stall), "before", stim)
            self.cosim.cycle(pre, stall, reset)
            if reset:
                self.states_reset.add((pre.state, stall))
            else:
                self.accepts += int(bool(start and pre.ready))
                if stall:
                    self.states_stalled.add(pre.state)
        else:
            pre = None

        await RisingEdge(d.clk)
        await ReadOnly()
        expected = self.model.step(start=start, k_in=k_in, stall=stall, reset=reset)
        self._compare(self._read(), expected, "after", stim)
        self.cycle += 1
        return pre

    # tile driving

    def random_matrices(self, k):
        lo, hi = signed_range(8)
        a = [[self.rng.randint(lo, hi) for _ in range(k)] for _ in range(self.n)]
        b = [[self.rng.randint(lo, hi) for _ in range(self.n)] for _ in range(k)]
        return a, b

    def begin_tile(self, a_mat, b_mat):
        self.cosim.load(a_mat, b_mat)
        self.cosim.clear_events()
        self.accepts = 0

    async def run_tile(self, a_mat, b_mat, stall_at=None, stall_len=2,
                       stall_prob=0.0, max_cycles=None):
        """Start one tile and run until the FSM is ready again.

        stall_at inserts stall_len stalled cycles before tile cycle stall_at;
        stall_prob stalls each cycle at random. Returns (C, tile cycles), where
        tile cycles counts unstalled cycles from the accepting one onward.
        """
        k = len(a_mat[0])
        self.begin_tile(a_mat, b_mat)
        limit = max_cycles or 50 * (k + 3 * self.n + self.ml + 8)
        started = False
        enabled = 0
        for _ in range(limit):
            if stall_at == enabled:
                for _ in range(stall_len):
                    await self.step(start=int(not started), k_in=k, stall=1)
                stall_at = None
            stall = int(self.rng.random() < stall_prob)
            pre = await self.step(start=int(not started), k_in=k, stall=stall)
            if started and pre.ready:
                break
            if not started and pre.ready:
                started = True
            if started and not stall:
                enabled += 1
        else:
            raise AssertionError(f"tile never finished, seed={self.seed}")
        return self.cosim.result(), enabled

    def check_tile(self, got, a_mat, b_mat, what):
        exp = expected_matmul(a_mat, b_mat)
        assert got == exp, f"{what}\ngot {got}\nexp {exp}\nseed={self.seed}"
        assert self.cosim.done_events == 1, (
            f"{what}: done & ~stall seen {self.cosim.done_events} times, expected 1"
        )
        k = len(a_mat[0])
        assert self.cosim.req_events == list(range(k)), (
            f"{what}: slices requested {self.cosim.req_events[:8]}..., expected 0..{k-1}"
        )
        assert self.accepts == 1, f"{what}: start accepted {self.accepts} times"

    async def run_until_state(self, state, k, limit=None):
        """Start a tile and stop on the first cycle the DUT is in `state`."""
        limit = limit or 4 * (k + 3 * self.n + self.ml + 8)
        started = False
        for _ in range(limit):
            if read_int(self.dut.state, "state") == state and (started or state == S_IDLE):
                return
            pre = await self.step(start=int(not started), k_in=k)
            started = started or bool(pre.ready)
        raise AssertionError(f"never reached {STATE_NAMES[state]}")


def get_k():
    return int(os.environ.get("K") or 8)


def tile_k(h):
    k = get_k()
    if k > h.k_max:
        raise ValueError(f"K={k} does not fit K_WIDTH={h.kw}")
    return k


# TESTS


@cocotb.test()
async def test_random_tile(dut):
    """One tile end to end through the feeder and the mesh model.

    step() compares every port against CtrlFsmModel on both sides of every
    edge. check_tile is the independent half: expected_matmul knows nothing
    about tags, the flush wait or the drain order.
    """
    h = CtrlHarness(dut)
    await h.start()
    k = tile_k(h)
    a, b = h.random_matrices(k)
    got, _ = await h.run_tile(a, b)
    h.check_tile(got, a, b, f"K={k} tile")


@cocotb.test()
async def test_latency_from_sim(dut):
    """Measure every latency in the header comment on the DUT itself.

    The array's LAST_TO_DRAIN_READY comes from the array model, where it was
    pinned by test_drain_latency_boundary; the FSM has to land c_valid on
    exactly that edge whatever MEM_LATENCY is.
    """
    h = CtrlHarness(dut)
    await h.start()
    k = tile_k(h)
    a, b = h.random_matrices(k)
    h.begin_tile(a, b)

    seen = {}
    cycle = 0
    started = False
    while True:
        pre = await h.step(start=int(not started), k_in=k)
        if started and pre.ready:
            seen["ready"] = cycle
            break
        started = True
        for name in ("op_req", "arr_first", "arr_last", "c_valid", "done"):
            if getattr(pre, name):
                seen.setdefault(name, cycle)
                seen["last_" + name] = cycle
        cycle += 1

    # cycle 0 is the accepting cycle. a signal high in cycle c is sampled c
    # edges after the accepting edge, and rose c - 1 edges after it.
    ml, n = h.ml, h.n
    ldr = h.cosim.array.last_to_drain_ready
    last_sampled = ml + k
    expect = {
        "op_req": 1, "last_op_req": k,
        "arr_first": ml + 1, "last_arr_first": ml + 1,
        "arr_last": last_sampled, "last_arr_last": last_sampled,
        # the array holds row N-1 once ldr edges have passed after the one
        # that sampled last, so that is the edge c_valid has to rise on
        "c_valid": last_sampled + ldr + 1, "last_c_valid": last_sampled + ldr + n,
        "done": last_sampled + ldr + n + 1, "last_done": last_sampled + ldr + n + 1,
        "ready": last_sampled + ldr + n + 2,
    }
    assert seen == expect, f"measured {seen}\nexpected {expect}"
    h.check_tile(h.cosim.result(), a, b, "latency tile")
    dut._log.info(f"latencies measured: {seen}")


@cocotb.test()
async def test_k1_coincident_first_last(dut):
    """K=1: first and last are the same slice and STREAM lasts one cycle."""
    h = CtrlHarness(dut)
    await h.start()
    for tile in range(3):
        a, b = h.random_matrices(1)
        got, _ = await h.run_tile(a, b)
        h.check_tile(got, a, b, f"K=1 tile {tile}")


@cocotb.test()
async def test_k_at_width_limit(dut):
    """K = 2**K_WIDTH - 1, the largest depth k_idx can count.

    Only run where that is a sane number of cycles; the sweep overrides
    K_WIDTH to a small value to get here.
    """
    h = CtrlHarness(dut)
    await h.start()
    if h.k_max > 1024:
        dut._log.info(f"K_WIDTH={h.kw}: K max is {h.k_max} cycles, covered by "
                      f"the K_WIDTH=4 sweep point instead")
        return
    a, b = h.random_matrices(h.k_max)
    got, _ = await h.run_tile(a, b)
    h.check_tile(got, a, b, f"K={h.k_max} at K_WIDTH={h.kw}")
    dut._log.info(f"K at the K_WIDTH limit passed: K={h.k_max}")


@cocotb.test()
async def test_stall_in_every_state(dut):
    """A stall before any single cycle of the tile changes nothing.

    Long tiles are sampled rather than swept, but the first and last cycles
    are always included so every state gets stalled at least once.
    """
    h = CtrlHarness(dut)
    await h.start()
    k = tile_k(h)
    a, b = h.random_matrices(k)
    clean, length = await h.run_tile(a, b)
    h.check_tile(clean, a, b, "clean tile")

    points = list(range(length + 1))
    if len(points) > 48:
        edge = points[:4 + h.ml] + points[-(3 * h.n + 4):]
        points = sorted(set(edge + h.rng.sample(points, 12)))
    for at in points:
        a, b = h.random_matrices(k)
        got, _ = await h.run_tile(a, b, stall_at=at, stall_len=h.rng.randint(1, 3))
        h.check_tile(got, a, b, f"stall before tile cycle {at}")

    missing = {S_IDLE, S_STREAM, S_FLUSH, S_DRAIN, S_DONE} - h.states_stalled
    assert not missing, f"never stalled in {[STATE_NAMES[s] for s in missing]}"
    dut._log.info(f"stalled at {len(points)} points across all five states")


@cocotb.test()
async def test_random_stall_done_once(dut):
    """Heavy random stall: done & ~stall happens exactly once per tile."""
    h = CtrlHarness(dut)
    await h.start()
    k = tile_k(h)
    stretched = 0
    for tile in range(4):
        a, b = h.random_matrices(k)
        h.begin_tile(a, b)
        started = False
        forced = False
        done_cycles = 0
        while True:
            stall = int(h.rng.random() < 0.5)
            # always stall the first done cycle, so the stretched pulse that
            # could be double-counted is hit in every tile and not by luck
            if read_int(dut.done, "done") and not forced:
                stall, forced = 1, True
            pre = await h.step(start=int(not started), k_in=k, stall=stall)
            done_cycles += pre.done
            if started and pre.ready:
                break
            started = started or bool(pre.ready)
        h.check_tile(h.cosim.result(), a, b, f"random stall tile {tile}")
        assert done_cycles > 1, "done did not hold under stall"
        stretched += 1
    dut._log.info(f"done & ~stall once per tile, done held under stall in {stretched}/4 tiles")


@cocotb.test()
async def test_reset_in_every_state(dut):
    """Reset from each state, stalled and not, then a clean tile."""
    h = CtrlHarness(dut)
    await h.start()
    k = max(tile_k(h), 2) if h.k_max >= 2 else 1
    for state in (S_IDLE, S_STREAM, S_FLUSH, S_DRAIN, S_DONE):
        for stall in (0, 1):
            await h.run_until_state(state, k)
            # a couple of extra cycles so reset does not always hit the first one
            if state in (S_STREAM, S_FLUSH) and h.rng.random() < 0.5:
                await h.step()
                if read_int(dut.state, "state") != state:
                    await h.run_until_state(S_IDLE, k)
                    await h.run_until_state(state, k)
            await h.step(reset=1, stall=stall)
            s = h._read()
            assert (s.state, s.busy, s.op_req, s.c_valid, s.done, s.arr_first,
                    s.arr_last, s.arr_drain_shift) == (S_IDLE, 0, 0, 0, 0, 0, 0, 0), (
                f"reset from {STATE_NAMES[state]} stall={stall} left {s}"
            )
            await h.step(reset=1)
            a, b = h.random_matrices(k)
            got, _ = await h.run_tile(a, b)
            h.check_tile(got, a, b, f"tile after reset in {STATE_NAMES[state]}")

    want = {(s, st) for s in range(5) for st in (0, 1)}
    assert want <= h.states_reset, f"missed reset points {want - h.states_reset}"


@cocotb.test()
async def test_back_to_back_start(dut):
    """start held high: tiles run one after another with nothing carried over."""
    h = CtrlHarness(dut)
    await h.start()
    k = tile_k(h)
    gaps = []
    for tile in range(4):
        a, b = h.random_matrices(k)
        h.begin_tile(a, b)
        idle = 0
        while True:
            pre = await h.step(start=1, k_in=k)
            if pre.ready:
                break
            idle += 1
        # start stays high for the whole tile and must not retrigger
        while True:
            pre = await h.step(start=1, k_in=k)
            if pre.done:
                break
        gaps.append(idle)
        h.check_tile(h.cosim.result(), a, b, f"back-to-back tile {tile}")
    # after done there is exactly one DONE cycle before the next accept
    assert gaps[1:] == [0] * 3, f"idle cycles between tiles: {gaps}"


@cocotb.test()
async def test_start_while_busy(dut):
    """start pulses with a different k_in while busy are ignored."""
    h = CtrlHarness(dut)
    await h.start()
    k = tile_k(h)
    others = [x for x in range(1, min(h.k_max, 2 * k + 3) + 1) if x != k] or [k]
    for tile in range(3):
        a, b = h.random_matrices(k)
        h.begin_tile(a, b)
        started = False
        while True:
            if not started:
                pre = await h.step(start=1, k_in=k)
            else:
                pre = await h.step(start=int(h.rng.random() < 0.6),
                                   k_in=h.rng.choice(others),
                                   stall=int(h.rng.random() < 0.2))
                assert read_int(dut.k_lat, "k_lat") == k, "k_lat changed mid-tile"
            if started and pre.done and read_int(dut.stall, "stall") == 0:
                break
            started = started or bool(pre.ready)
        # the DONE cycle with start low, so the next loop starts from IDLE
        await h.step()
        h.check_tile(h.cosim.result(), a, b, f"start-while-busy tile {tile}")


@cocotb.test()
async def test_start_pulse_during_stall_not_accepted(dut):
    """A one-cycle start under stall is dropped, and ready said so."""
    h = CtrlHarness(dut)
    await h.start()
    k = tile_k(h)
    a, b = h.random_matrices(k)
    h.begin_tile(a, b)

    await h.step(stall=1)
    pre = await h.step(start=1, k_in=k, stall=1)
    assert pre.ready == 0, "ready was high under stall"
    await h.step(stall=1)
    for _ in range(k + 3 * h.n + h.ml + 6):
        pre = await h.step()
        assert (pre.busy, pre.op_req, pre.state) == (0, 0, S_IDLE), (
            f"a start pulsed under stall was accepted: {pre}"
        )
    assert h.accepts == 0 and h.cosim.req_events == [] and h.cosim.done_events == 0

    # and the FSM is still usable
    got, _ = await h.run_tile(a, b)
    h.check_tile(got, a, b, "tile after the dropped start")


@cocotb.test()
async def test_start_held_through_stall_accepted_once(dut):
    """start held across a stall is taken on the first unstalled edge, once."""
    h = CtrlHarness(dut)
    await h.start()
    k = tile_k(h)
    a, b = h.random_matrices(k)
    h.begin_tile(a, b)

    for _ in range(4):
        pre = await h.step(start=1, k_in=k, stall=1)
        assert (pre.ready, pre.busy) == (0, 0)
    pre = await h.step(start=1, k_in=k)
    assert pre.ready == 1, "ready did not return when the stall ended"
    assert read_int(dut.state, "state") == S_STREAM, "held start was not accepted"

    # keep holding start through the tile and through more stalls
    while True:
        pre = await h.step(start=1, k_in=k, stall=int(h.rng.random() < 0.3))
        if pre.done and read_int(dut.stall, "stall") == 0:
            break
    h.check_tile(h.cosim.result(), a, b, "held-start tile")


@cocotb.test(skip=os.environ.get("ALLOW_K0") != "1")
async def test_k_zero_start_ignored(dut):
    """start with k_in == 0 leaves the FSM idle instead of hanging it.

    Needs +ctrl_fsm_allow_k0 (make ALLOW_K0=1), otherwise the simulation-only
    $fatal in the RTL ends the run, which is what the normal regression wants.
    """
    h = CtrlHarness(dut)
    await h.start()
    k = tile_k(h)
    a, b = h.random_matrices(k)
    h.begin_tile(a, b)

    for _ in range(3):
        pre = await h.step(start=1, k_in=0)
        assert pre.ready == 1
    for _ in range(k + 3 * h.n + h.ml + 6):
        pre = await h.step()
        assert (pre.busy, pre.op_req, pre.state) == (0, 0, S_IDLE), (
            f"k_in == 0 start was accepted: {pre}"
        )
    assert h.cosim.req_events == [] and h.cosim.done_events == 0

    got, _ = await h.run_tile(a, b)
    h.check_tile(got, a, b, "tile after the k_in == 0 start")
    dut._log.info("k_in == 0 start ignored")
