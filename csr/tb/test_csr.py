""" cocotb tb for csr/rtl/csr.sv

The CSR runs against:
  - an AXI4-Lite master BFM that can present AW before W, W before AW or both
    in the same cycle, and hold off BREADY / RREADY while checking that the
    response stays stable;
  - a stub for conv_layer and its act_buffer: the start/busy/done/err
    handshake with a programmable layer length, and the act_buffer read
    pipeline (READ_LATENCY deep, advancing only while host_rd_en is high);
  - csr_model.CsrModel, which predicts every response, every read value and
    every memory write the CSR commits.
Sampling is in the ReadOnly phase; the stub and the BFM drive on the falling
edge.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from csr_model import DECERR, OKAY, SLVERR, CsrModel
from protocol import (A_ACT_WCRC, A_B_WCRC, A_BUILD_ID, A_CMD0, A_CTRL, A_CYCLES, A_ERR_CODE,
                      A_ID, A_LAUNCH, A_P_WCRC, A_PARAMS0, A_PARAMS1, A_SCRATCH, A_STATUS,
                      A_VERSION, ACT_BASE, B_BASE, CMD_FIELDS, P_BASE, P_WIDTH, slot_shift)

REGS = [A_ID, A_VERSION, A_PARAMS0, A_PARAMS1, A_CTRL, A_STATUS, A_ERR_CODE, A_CYCLES,
        A_SCRATCH, A_BUILD_ID, A_ACT_WCRC, A_B_WCRC, A_P_WCRC, A_LAUNCH] + \
       [A_CMD0 + 4 * i for i in range(len(CMD_FIELDS))]


def junk(sel, addr, n):
    """what an act word holds before anyone wrote it (stub and model agree)"""
    return random.Random(sel * 100003 + addr).getrandbits(8 * n)


class CsrHarness:

    def __init__(self, dut):
        self.dut = dut
        g = lambda name: int(getattr(dut, name).value)
        self.n, self.aw, self.lat, self.nbuf = g("N"), g("ADDR_WIDTH"), g("READ_LATENCY"), g("NBUF")
        self.b_aw, self.p_aw = g("B_ADDR_W"), g("P_ADDR_W")
        self.model = CsrModel(n=self.n, addr_width=self.aw, read_latency=self.lat, nbuf=self.nbuf,
                              b_addr_w=self.b_aw, p_addr_w=self.p_aw, k_max=g("K_MAX"),
                              build_id=g("BUILD_ID"))
        self.sel_w = self.model.sel_w
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = random.Random(self.seed)
        # stub state
        self.mem = {}
        self.busy = 0
        self.remaining = 0
        self.done = self.err = 0
        self.err_code = 0
        self.next_layer = (20, 0)       # (edges busy, err_code; 0 = done)
        self.commits = []
        self.rd_runs = []               # [(sel, addr, edges)] for each host_rd_en run
        self._run = None
        self.pipe = [0, 0]
        self.accepts = 0
        self.busy_edges = 0             # edges since the accept while busy = live CYCLES

    def where(self):
        return f"(reproduce: make MODULE_NAME=csr N={self.n} READ_LATENCY={self.lat} SEED={self.seed})"

    def word(self, sel, addr):
        return self.mem.get((sel, addr), junk(sel, addr, self.n))

    async def start(self):
        d = self.dut
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        for s in ("s_awvalid", "s_wvalid", "s_bready", "s_arvalid", "s_rready", "link_err",
                  "s_awaddr", "s_wdata", "s_araddr"):
            getattr(d, s).value = 0
        d.s_wstrb.value = 0xF
        self._drive_stub()
        d.reset.value = 1
        for _ in range(3):
            await RisingEdge(d.clk)
        await FallingEdge(d.clk)
        d.reset.value = 0
        cocotb.start_soon(self._stub())
        d._log.info(f"N={self.n} READ_LATENCY={self.lat} NBUF={self.nbuf} SEED={self.seed}")

    def _drive_stub(self):
        d = self.dut
        d.conv_ready.value = 1 - self.busy
        d.conv_busy.value = self.busy
        d.conv_done.value = self.done
        d.conv_err.value = self.err
        d.conv_err_code.value = self.err_code
        d.host_rd_data.value = self.pipe[self.lat - 1]

    async def _stub(self):
        """conv_layer + act_buffer, one virtual edge per real edge"""
        d = self.dut
        await ReadOnly()
        while True:
            s = {k: int(getattr(d, k).value) for k in (
                "conv_start", "conv_soft_reset", "host_mode", "host_we", "host_waddr",
                "host_wdata", "host_wr_sel", "host_rd_en", "host_rd_addr", "host_rd_sel",
                "b_we", "b_waddr", "b_wdata", "p_we", "p_waddr", "p_wdata")}
            await RisingEdge(d.clk)
            # --- the edge: conv_layer
            self.done = self.err = 0
            if s["conv_soft_reset"]:
                self.busy, self.remaining, self.err_code = 0, 0, 0
            elif s["conv_start"] and not self.busy:
                self.busy, self.accepts = 1, self.accepts + 1
                self.busy_edges = 0
                self.remaining, code = self.next_layer
                self.planned_code = code
                self.err_code = 0
                self.model.accepted_start()
            elif self.busy:
                self.busy_edges += 1
                self.remaining -= 1
                if self.remaining == 0:
                    self.busy = 0
                    self.err_code = self.planned_code
                    if self.planned_code: self.err = 1
                    else:                 self.done = 1
                    self.model.finished(self.next_layer[0], err=bool(self.planned_code))
            # --- the edge: memories (the host write port needs host_mode)
            if s["host_we"]:
                assert s["host_mode"], "host_we without host_mode"
                self.mem[(s["host_wr_sel"], s["host_waddr"])] = s["host_wdata"]
                self.commits.append(("act", s["host_wr_sel"], s["host_waddr"], s["host_wdata"]))
            if s["b_we"]:
                self.commits.append(("b", s["b_waddr"], s["b_wdata"]))
            if s["p_we"]:
                self.commits.append(("p", s["p_waddr"], s["p_wdata"]))
            # --- the edge: act_buffer read pipeline, moves only with rd_en
            if s["host_rd_en"]:
                key = (s["host_rd_sel"], s["host_rd_addr"])
                if self.lat == 2:
                    self.pipe[1] = self.pipe[0]
                self.pipe[0] = self.word(*key)
                if self._run and self._run[0] == key:
                    self._run[1] += 1
                else:
                    self._run = [key, 1]
                    self.rd_runs.append(self._run)
            else:
                self._run = None
            await FallingEdge(d.clk)
            self._drive_stub()
            await ReadOnly()

    # -- AXI4-Lite master --------------------------------------------------------
    # A handshake happens on the edge whose preceding cycle had valid and
    # ready both high, so ready is sampled in the ReadOnly phase *before* the
    # edge (after it, the slave may already have dropped it for the next
    # transfer). Everything is driven on the falling edge.
    async def _cycle(self):
        """ReadOnly view of this cycle, then on through the next edge to the
        falling edge where the master may drive again"""
        d = self.dut
        await ReadOnly()
        view = {k: int(getattr(d, k).value) for k in (
            "s_awvalid", "s_awready", "s_wvalid", "s_wready", "s_bvalid", "s_bready", "s_bresp",
            "s_arvalid", "s_arready", "s_rvalid", "s_rready", "s_rresp", "s_rdata")}
        await RisingEdge(d.clk)
        await FallingEdge(d.clk)
        return view

    async def write(self, addr, data, order="same", gap=0, bdelay=0):
        """order: same | aw (AW first, W gap cycles after it is accepted) | w"""
        d = self.dut
        await FallingEdge(d.clk)
        aw_go = order in ("same", "aw")
        w_go = order in ("same", "w")
        aw_done = w_done = False
        wait_other = None
        if aw_go: d.s_awaddr.value, d.s_awvalid.value = addr, 1
        if w_go:  d.s_wdata.value, d.s_wvalid.value = data, 1
        d.s_bready.value = 0
        while not (aw_done and w_done):
            v = await self._cycle()
            aw_hs = v["s_awvalid"] and v["s_awready"]
            w_hs = v["s_wvalid"] and v["s_wready"]
            if aw_hs: aw_done, d.s_awvalid.value = True, 0
            if w_hs:  w_done, d.s_wvalid.value = True, 0
            if (aw_hs and not w_go) or (w_hs and not aw_go):
                wait_other = gap
            if wait_other is not None:
                if wait_other == 0:
                    if not w_go: d.s_wdata.value, d.s_wvalid.value, w_go = data, 1, True
                    if not aw_go: d.s_awaddr.value, d.s_awvalid.value, aw_go = addr, 1, True
                    wait_other = None
                else:
                    wait_other -= 1
        # response; BREADY held off for bdelay cycles once BVALID is up, and
        # BRESP must not change while it waits
        held = None
        while True:
            d.s_bready.value = 1 if (held is not None and bdelay == 0) else 0
            v = await self._cycle()
            if v["s_bvalid"]:
                if held is not None:
                    assert v["s_bresp"] == held, "BRESP changed while BVALID waited for BREADY"
                held = v["s_bresp"]
                if v["s_bready"]:
                    break
                if bdelay: bdelay -= 1
        d.s_bready.value = 0
        return held

    async def read(self, addr, rdelay=0):
        """returns (resp, data, edges from the AR handshake edge to RVALID)"""
        d = self.dut
        await FallingEdge(d.clk)
        d.s_araddr.value, d.s_arvalid.value = addr, 1
        d.s_rready.value = 0
        while True:
            v = await self._cycle()
            if v["s_arready"]:
                break
        d.s_arvalid.value = 0
        edges = 1                     # the cycle after the AR handshake edge
        held = None
        first = None
        while True:
            d.s_rready.value = 1 if (held is not None and rdelay == 0) else 0
            v = await self._cycle()
            if v["s_rvalid"]:
                now = (v["s_rresp"], v["s_rdata"])
                if held is None:
                    first = edges
                else:
                    assert now == held, "R channel changed while RVALID waited for RREADY"
                held = now
                if v["s_rready"]:
                    break
                if rdelay: rdelay -= 1
            edges += 1
        d.s_rready.value = 0
        return held[0], held[1], first

    # -- checked operations -------------------------------------------------------
    async def chk_write(self, addr, data, **kw):
        exp = self.model.write(addr, data, busy=bool(self.busy))
        got = await self.write(addr, data, **kw)
        assert got == exp, f"write {addr:#08x} <- {data:#x}: bresp {got}, model {exp} {self.where()}"
        await self.idle(2)
        assert self.commits == self.model.commits, \
            f"memory writes differ after {addr:#08x}: rtl {self.commits[-2:]} model {self.model.commits[-2:]}"
        return got

    async def chk_read(self, addr, **kw):
        exp = self.model.read(addr, busy=bool(self.busy), err_code=self.err_code, mem=self.word)
        runs = len(self.rd_runs)
        live = addr == A_CYCLES and self.busy
        lo = self.busy_edges
        resp, data, edges = await self.read(addr, **kw)
        if live:
            # CYCLES counts live while busy: it must lie between the edge
            # counts at the start and the end of this read
            assert resp == OKAY and lo <= data <= self.busy_edges, \
                f"live CYCLES {data} outside [{lo}, {self.busy_edges}]"
            return resp, data, edges
        assert (resp, data) == exp, (f"read {addr:#08x}: ({resp}, {data:#x}), model "
                                     f"({exp[0]}, {exp[1]:#x}) {self.where()}")
        new_runs = self.rd_runs[runs:]
        if (addr >> 20) == 1 and resp == OKAY:
            assert len(new_runs) == 1 and new_runs[0][1] == self.lat, \
                f"host_rd_en runs {new_runs}, expected one run of {self.lat} edges at one address"
        else:
            assert not new_runs, f"host_rd_en moved for a read that is not a good ACT read: {new_runs}"
        return resp, data, edges

    async def idle(self, n):
        for _ in range(n):
            await RisingEdge(self.dut.clk)

    async def pulse_link_err(self):
        d = self.dut
        await FallingEdge(d.clk)
        d.link_err.value = 1
        await FallingEdge(d.clk)
        d.link_err.value = 0
        self.model.link_err()

    def act_addr(self, sel, idx, slot=0):
        return ACT_BASE + ((((sel << self.aw) | idx) << slot_shift(8 * self.n)) + slot) * 4

    def b_addr(self, idx, slot=0):
        return B_BASE + (((idx << slot_shift(8 * self.n)) + slot) * 4)

    def p_addr(self, idx, slot=0):
        return P_BASE + (((idx << slot_shift(P_WIDTH)) + slot) * 4)


async def fresh(dut):
    h = CsrHarness(dut)
    await h.start()
    return h


# ----------------------------------------------------------------------------

@cocotb.test()
async def test_reset_values(dut):
    """every register reads its documented reset value"""
    h = await fresh(dut)
    for a in REGS:
        await h.chk_read(a)
    resp, data, _ = await h.chk_read(A_ID)
    assert data == 0x434E5631


@cocotb.test()
async def test_register_read_write(dut):
    """each RW register keeps exactly its field width; RO registers refuse
    writes; holes and unmapped regions are DECERR"""
    h = await fresh(dut)
    for a in REGS:
        for v in (0xFFFFFFFF, 0, 0xA5A5A5A5, h.rng.getrandbits(32)):
            if a == A_CTRL:
                v &= 0x4            # start/soft reset/clear have their own tests
            await h.chk_write(a, v)
            await h.chk_read(a)
    for a in (0x38, 0x3C, 0x8C, 0xFC, 0x400000, 0xF00000):
        await h.chk_write(a, 1)
        await h.chk_read(a)


@cocotb.test()
async def test_handshake_orderings(dut):
    """AW before W, W before AW, same cycle; B and R held off; back to back"""
    h = await fresh(dut)
    lat = {}
    for order in ("same", "aw", "w"):
        for gap in (0, 1, 4):
            for bdelay in (0, 1, 5):
                v = h.rng.getrandbits(32)
                await h.chk_write(A_SCRATCH, v, order=order, gap=gap, bdelay=bdelay)
                for rdelay in (0, 3):
                    _, data, edges = await h.chk_read(A_SCRATCH, rdelay=rdelay)
                    assert data == v
                    lat.setdefault("reg", set()).add(edges)
    assert lat["reg"] == {1}, f"register read latency varies: {lat}"
    dut._log.info("LATENCY register read: RVALID 1 edge after the AR handshake")


@cocotb.test()
async def test_windows_and_staging(dut):
    """ACT/B/P writes: lower slots stage, the top slot commits only with every
    lower slot written for the same word; link_err clears staging; write
    CRCs follow every commit; ACT needs host_mode; ACT reads go through the
    read pipeline with host_rd_en held READ_LATENCY edges"""
    h = await fresh(dut)
    sa = 1 << slot_shift(8 * h.n)
    await h.chk_write(A_CTRL, 8)                         # clear CRCs
    await h.chk_write(h.act_addr(0, 1, sa - 1), 5)       # no host_mode: refused
    await h.chk_write(A_CTRL, 4)
    for sel in range(h.nbuf):
        for idx in (0, 1, (1 << h.aw) - 1):
            for s in range(sa):
                await h.chk_write(h.act_addr(sel, idx, s), h.rng.getrandbits(32),
                                  order=h.rng.choice(("same", "aw", "w")))
    if sa > 1:
        await h.chk_write(h.act_addr(0, 7, 1), 1)        # top alone: SLVERR
        await h.chk_write(h.act_addr(0, 7, 0), 1)
        await h.chk_write(h.act_addr(0, 8, 1), 1)        # other word: SLVERR
        await h.chk_write(h.act_addr(0, 7, 0), 1)
        await h.pulse_link_err()
        await h.chk_write(h.act_addr(0, 7, 1), 1)        # cleared by link_err: SLVERR
        await h.chk_write(h.act_addr(0, 7, 0), 1)
        await h.chk_write(h.act_addr(0, 9, 0), 2)        # restarts staging on word 9
        await h.chk_write(h.act_addr(0, 9, 1), 3)        # commits word 9
    # unmapped: index past the bank, bank past NBUF
    await h.chk_write(ACT_BASE + ((1 << (h.sel_w + h.aw)) << slot_shift(8 * h.n)) * 4, 1)
    if (1 << h.sel_w) > h.nbuf:
        await h.chk_write(h.act_addr(h.nbuf, 0, sa - 1), 1)
    for idx in (0, 3, (1 << h.b_aw) - 1):
        for s in range(sa):
            await h.chk_write(h.b_addr(idx, s), h.rng.getrandbits(32))
    for idx in (0, 5, (1 << h.p_aw) - 1):
        await h.chk_write(h.p_addr(idx, 0), h.rng.getrandbits(32))
        await h.chk_write(h.p_addr(idx, 1), 0xFFFFFFFF)  # bits above 53 must drop
    await h.chk_write(h.p_addr(2, 1), 7)                 # P top alone: SLVERR
    await h.chk_write(B_BASE + ((1 << h.b_aw) << slot_shift(8 * h.n)) * 4, 0)
    for a in (A_ACT_WCRC, A_B_WCRC, A_P_WCRC):
        await h.chk_read(a)
    await h.chk_read(h.b_addr(0))                         # B is write only
    await h.chk_read(h.p_addr(0))
    await h.chk_read(h.act_addr(0, 0))                    # host_mode: refused
    await h.chk_write(A_CTRL, 0)
    lats = set()
    for sel in range(h.nbuf):
        for idx in (0, 1, 2, (1 << h.aw) - 1):
            for s in range(sa):
                _, _, e = await h.chk_read(h.act_addr(sel, idx, s), rdelay=h.rng.choice((0, 2)))
                lats.add(e)
    assert len(lats) == 1, f"ACT read latency varies: {lats}"
    dut._log.info(f"LATENCY ACT window read at READ_LATENCY={h.lat}: RVALID {lats.pop()} edges "
                  f"after the AR handshake")
    await h.chk_write(A_CTRL, 8)
    for a in (A_ACT_WCRC, A_B_WCRC, A_P_WCRC):
        await h.chk_read(a)


@cocotb.test()
async def test_start_status_cycles(dut):
    """start pulses once and counts in LAUNCH_COUNT; done and err are sticky
    and W1C; CYCLES equals the layer's edges; ERR_CODE follows conv_layer"""
    h = await fresh(dut)
    for edges, code in ((1, 0), (7, 0), (30, 0x81), (5, 0x002)):
        h.next_layer = (edges, code)
        await h.chk_write(A_CTRL, 1)
        while h.busy:
            await h.idle(1)
        await h.idle(2)
        for a in (A_STATUS, A_CYCLES, A_ERR_CODE, A_LAUNCH):
            await h.chk_read(a)
        # W1C one bit at a time: the bit that is not set first (it must
        # leave the set one alone), then the set one. clearing both at once
        # cannot tell the two bits apart; a mutation run showed that.
        first, second = (0x4, 0x8) if code else (0x8, 0x4)
        await h.chk_write(A_STATUS, first)
        await h.chk_read(A_STATUS)
        await h.chk_write(A_STATUS, second)
        await h.chk_read(A_STATUS)
    assert h.accepts == 4


@cocotb.test()
async def test_busy_rules(dut):
    """while busy: register reads, STATUS W1C and soft reset only"""
    h = await fresh(dut)
    sa = 1 << slot_shift(8 * h.n)
    await h.chk_write(A_CTRL, 4)
    await h.chk_write(h.act_addr(0, 3, 0), 9)            # half-staged before the layer
    await h.chk_write(A_CTRL, 0)
    h.next_layer = (100000, 0)
    await h.chk_write(A_CTRL, 1)
    assert h.busy
    for a in REGS:
        await h.chk_read(a)
        await h.chk_write(a, 0 if a == A_CTRL else 0xFFFFFFFF)
    await h.chk_write(A_CTRL, 4)                         # host_mode while busy: refused
    await h.chk_write(A_CTRL, 1)                         # start while busy: refused
    for a in (h.act_addr(0, 3, sa - 1), h.b_addr(1, sa - 1), h.p_addr(1, 1)):
        await h.chk_write(a, 1)
        await h.chk_read(a)
    await h.chk_write(A_STATUS, 0xC)
    assert h.busy, "the stub layer ended early"
    await h.chk_write(A_CTRL, 2)                         # soft reset gets through
    await h.idle(2)
    assert not h.busy
    for a in (A_STATUS, A_CYCLES, A_LAUNCH, A_CTRL):
        await h.chk_read(a)
    # busy *with* host_mode on: a start written together with host_mode keeps
    # conv_layer busy through its check phase (it then rejects with E_HOST).
    # that is the one state where host_mode alone would let an ACT write
    # through, so busy has to refuse it by itself. the stub stays busy long
    # enough to test it; a mutation run found nothing else reached this state.
    h.next_layer = (100000, 0)
    await h.chk_write(A_CTRL, 1 | 4)
    assert h.busy
    for s in range(sa):
        await h.chk_write(h.act_addr(0, 4, s), 0xABCD)
    await h.chk_read(h.act_addr(0, 4, 0))
    await h.chk_write(A_CTRL, 2)
    await h.idle(2)
    await h.chk_write(A_CTRL, 0)


@cocotb.test()
async def test_random_traffic(dut):
    """random mix of every access, orderings, backpressure, host_mode, link
    errors and short layers, all against the model"""
    h = await fresh(dut)
    sa = 1 << slot_shift(8 * h.n)
    addrs = REGS + [0x38, 0x90, 0x7FFFFC]
    for i in range(600):
        kind = h.rng.random()
        kw = dict(order=h.rng.choice(("same", "aw", "w")), gap=h.rng.choice((0, 0, 2)),
                  bdelay=h.rng.choice((0, 0, 3)))
        sel = h.rng.randrange(1 << h.sel_w)
        idx = h.rng.choice((0, 1, 2, (1 << h.aw) - 1))
        if kind < 0.25:
            a = h.rng.choice(addrs)
            v = h.rng.getrandbits(32)
            if a == A_CTRL:
                v &= 0xC | (1 if h.rng.random() < 0.3 else 0)
                h.next_layer = (h.rng.randint(1, 12), h.rng.choice((0, 0, 0, 0x40)))
            await h.chk_write(a, v, **kw)
        elif kind < 0.45:
            await h.chk_read(h.rng.choice(addrs), rdelay=h.rng.choice((0, 0, 2)))
        elif kind < 0.75:
            s = h.rng.randrange(sa)
            await h.chk_write(h.act_addr(sel, idx, s), h.rng.getrandbits(32), **kw)
        elif kind < 0.85:
            await h.chk_read(h.act_addr(sel, idx, h.rng.randrange(sa)))
        elif kind < 0.93:
            if h.rng.random() < 0.5:
                await h.chk_write(h.b_addr(h.rng.randrange(4), h.rng.randrange(sa)), h.rng.getrandbits(32), **kw)
            else:
                await h.chk_write(h.p_addr(h.rng.randrange(4), h.rng.randrange(2)), h.rng.getrandbits(32), **kw)
        elif kind < 0.97:
            await h.pulse_link_err()
        else:
            await h.chk_write(A_CTRL, 4 if h.rng.random() < 0.5 else 0)
        # the busy rules have their own test; here every access sees a settled
        # busy so the model's view of it is exact
        while h.busy:
            await h.idle(1)
    for a in REGS:
        await h.chk_read(a)
