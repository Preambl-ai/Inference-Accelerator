""" cocotb tb for dma/rtl/wdma.sv

wdma is the toplevel; axi4_slave.Axi4Slave plays DDR behind it (latency and
backpressure from AXI_LAT / AXI_STALL, protocol checks on every burst), and
a monitor records every B and P write. Each fetch is compared with
dma_model.fetch: the error code, the exact list of AXI bursts (one per
output group unless the 256-beat or 4 KB rule splits it), and every B and P
write in order. With no backpressure the start-to-done count must equal
dma_model.estimate_cycles. Sampling in ReadOnly, driving on the falling edge.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from axi4_slave import OKAY, SLVERR, Axi4Slave
from dma_model import E_ALIGN, E_DIM, E_RANGE, E_RESP, Fetch, beat, estimate_cycles, fetch

MEM = 1 << 18          # 256 kB of "DDR"


class WdmaTB:

    def __init__(self, dut):
        self.dut = dut
        g = lambda name: int(getattr(dut, name).value)
        self.n, self.b_aw, self.p_aw = g("N"), g("B_ADDR_W"), g("P_ADDR_W")
        self.k_w, self.cw_w = g("K_WIDTH"), g("CW_W")
        self.lat = int(os.environ.get("AXI_LAT") or 3)
        self.stall = float(os.environ.get("AXI_STALL") or 0)
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = random.Random(self.seed)
        self.bad_ranges = []
        self.slave = Axi4Slave(dut, dut.clk, size=MEM, latency=self.lat, ar_stall=self.stall,
                               r_gap=self.stall, aw_stall=self.stall, w_stall=self.stall,
                               b_latency=self.lat, seed=self.seed, bad=self.bad)
        self.slave.mem[:] = bytes(self.rng.getrandbits(8) for _ in range(MEM))
        self.writes = []        # (edge, kind, waddr, data)
        self.edge = 0

    def bad(self, addr):
        return any(lo <= addr < hi for lo, hi in self.bad_ranges)

    def where(self):
        return (f"(reproduce: make MODULE_NAME=wdma N={self.n} AXI_LAT={self.lat} "
                f"AXI_STALL={self.stall} SEED={self.seed})")

    async def start(self):
        d = self.dut
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        for s in ("f_start", "h_req", "abort", "f_b_src", "f_p_src", "f_k", "f_groups",
                  "f_b_dst", "f_p_dst", "h_we", "h_addr", "h_wdata"):
            getattr(d, s).value = 0
        d.reset.value = 1
        for _ in range(3):
            await RisingEdge(d.clk)
        await FallingEdge(d.clk)
        self.slave.start()                      # the master's outputs are known from here
        d.reset.value = 0
        cocotb.start_soon(self._monitor())
        d._log.info(f"N={self.n} B_ADDR_W={self.b_aw} P_ADDR_W={self.p_aw} AXI_LAT={self.lat} "
                    f"AXI_STALL={self.stall} SEED={self.seed}")

    async def _monitor(self):
        d = self.dut
        while True:
            await RisingEdge(d.clk)
            self.edge += 1
            await ReadOnly()
            if int(d.b_we.value):
                self.writes.append((self.edge, "b", int(d.b_waddr.value), int(d.b_wdata.value)))
            if int(d.p_we.value):
                self.writes.append((self.edge, "p", int(d.p_waddr.value), int(d.p_wdata.value)))

    def drive(self, f: Fetch):
        d = self.dut
        d.f_b_src.value = f.b_src
        d.f_p_src.value = f.p_src
        d.f_k.value = f.k
        d.f_groups.value = f.groups
        d.f_b_dst.value = f.b_dst
        d.f_p_dst.value = f.p_dst

    async def until_end(self, abort_at=None, limit=400000):
        """edges from the accepting edge to done/err; abort_at aborts there"""
        d = self.dut
        t = 0
        while True:
            if abort_at is not None and t + 1 == abort_at:
                d.abort.value = 1
            await RisingEdge(d.clk)
            t += 1
            await ReadOnly()
            dn, er = int(d.f_done.value), int(d.f_err.value)
            if abort_at is not None and t >= abort_at:
                assert not (dn or er), f"done/err pulsed after the abort edge {self.where()}"
                await FallingEdge(d.clk)
                d.abort.value = 0
                return "abort", t
            assert not (dn and er), "done and err together"
            if dn or er:
                code = int(d.f_err_code.value)
                assert int(d.f_ready.value), "f_ready must be high with done/err"
                await FallingEdge(d.clk)
                await RisingEdge(d.clk)
                await ReadOnly()
                assert not int(d.f_done.value) and not int(d.f_err.value), "done/err longer than a cycle"
                assert int(d.f_err_code.value) == code, "f_err_code must hold"
                await FallingEdge(d.clk)
                return ("done" if dn else "err", t, code)
            assert int(d.busy.value), f"busy fell at edge {t} without done/err"
            assert t < limit, f"no done/err after {t} edges {self.where()}"
            await FallingEdge(d.clk)            # drive (abort) only off the ReadOnly phase

    async def run(self, f: Fetch, name, abort_at=None):
        d = self.dut
        exp = fetch(self.slave.mem, f, self.n, self.b_aw, self.p_aw, bad=self.bad)
        n_ar, n_w = len(self.slave.ar_log), len(self.writes)
        await FallingEdge(d.clk)
        assert int(d.f_ready.value), "harness: wdma not ready"
        self.drive(f)
        d.f_start.value = 1
        await RisingEdge(d.clk)
        await ReadOnly()
        start_edge = self.edge                  # the monitor has counted this edge by now
        assert int(d.busy.value), "start not accepted"
        await FallingEdge(d.clk)
        d.f_start.value = 0
        res = await self.until_end(abort_at)
        got_b = [(a, v) for _, k, a, v in self.writes[n_w:] if k == "b"]
        got_p = [(a, v) for _, k, a, v in self.writes[n_w:] if k == "p"]
        bursts = self.slave.ar_log[n_ar:]
        if res[0] == "abort":
            abort_edge = start_edge + abort_at - 1
            late = [w for w in self.writes[n_w:] if w[0] > abort_edge]
            assert not late, f"{name}: {len(late)} writes after the abort edge {self.where()}"
            assert got_b == exp.b_writes[:len(got_b)] and got_p == exp.p_writes[:len(got_p)], \
                f"{name}: writes before the abort differ from the model"
            # busy until the burst in flight is drained, then idle and quiet
            for _ in range(20000):
                await RisingEdge(d.clk)
                await ReadOnly()
                if int(d.f_ready.value):
                    break
            assert int(d.f_ready.value), "wdma did not go idle after abort"
            assert self.slave.idle(), "wdma went idle with AXI beats still owed"
            await FallingEdge(d.clk)
            d._log.info(f"ABORT {name}: at edge {abort_at}, {len(got_b)} B + {len(got_p)} P words "
                        f"before it, {len(bursts)} bursts")
            return res
        kind, cycles, code = res
        assert code == exp.err, (f"{name}: err code {code:#x}, model {exp.err:#x} {self.where()}")
        assert (kind == "done") == (exp.err == 0)
        assert [(a, c) for _, a, c in exp.bursts] == bursts, \
            f"{name}: bursts {bursts[:6]}, model {[(a, c) for _, a, c in exp.bursts][:6]} {self.where()}"
        assert got_b == exp.b_writes, f"{name}: B writes differ (got {len(got_b)}, model {len(exp.b_writes)}) {self.where()}"
        assert got_p == exp.p_writes, f"{name}: P writes differ (got {len(got_p)}, model {len(exp.p_writes)}) {self.where()}"
        if kind == "done" and self.stall == 0:
            est = estimate_cycles(exp, self.lat)
            assert cycles == est, f"{name}: {cycles} edges, formula {est} {self.where()}"
        d._log.info(f"FETCH {name}: {kind} {code:#x}, {len(bursts)} bursts, {len(got_b)} B + "
                    f"{len(got_p)} P words, {cycles} edges")
        return res

    async def host(self, we, addr, data=0):
        d = self.dut
        await FallingEdge(d.clk)
        assert int(d.f_ready.value)
        d.h_req.value, d.h_we.value, d.h_addr.value, d.h_wdata.value = 1, we, addr, data
        await RisingEdge(d.clk)
        await FallingEdge(d.clk)
        d.h_req.value = 0
        for t in range(10000):
            await RisingEdge(d.clk)
            await ReadOnly()
            if int(d.h_ack.value):
                r = (int(d.h_resp.value), int(d.h_rdata.value))
                await FallingEdge(d.clk)
                return r
        raise AssertionError("no h_ack")

    def legal(self, k=None, groups=None):
        n = self.n
        k = k or self.rng.randint(1, min(300, (1 << self.k_w) - 1))
        groups = groups or self.rng.randint(1, 4)
        groups = min(groups, ((1 << self.b_aw) // k), (1 << self.p_aw) // n)
        b_dst = self.rng.randint(0, (1 << self.b_aw) - groups * k)
        p_dst = self.rng.randint(0, (1 << self.p_aw) - groups * n)
        b_src = 8 * self.rng.randrange(0, (MEM // 2 - 8 * groups * k) // 8)
        p_src = MEM // 2 + 8 * self.rng.randrange(0, (MEM // 2 - 8 * groups * n) // 8)
        return Fetch(b_src, p_src, k, groups, b_dst, p_dst)


async def fresh(dut):
    t = WdmaTB(dut)
    await t.start()
    return t


@cocotb.test()
async def test_directed_fetches(dut):
    """1 beat x 1 group; a typical 72-word group; exactly 256 beats from a
    4 KB line (one burst); 257 (two); a group across a 4 KB line; K = 1024;
    the last P entries and the last B words of the memories"""
    t = await fresh(dut)
    n, top_b, top_p = t.n, 1 << t.b_aw, 1 << t.p_aw
    cases = [
        ("1 x 1", Fetch(0, MEM // 2, 1, 1, 0, 0)),
        ("72 x 2", Fetch(0x2000, MEM // 2 + 0x100, 72, 2, 100, 3)),
        ("256 from a 4 KB line", Fetch(0x3000, MEM // 2, 256, 1, 0, 0)),
        ("257 from a 4 KB line", Fetch(0x3000, MEM // 2, 257, 1, 0, 0)),
        ("across a 4 KB line", Fetch(0x5000 - 8 * 10, MEM // 2 + 4096 - 8, 40, 3, 7, 1)),
        ("K = 1024 x 2", Fetch(0x8000, MEM // 2, min(1024, (1 << t.k_w) - 1), 2, 0, 0)),
        ("the ends of B and P", Fetch(0x10000, MEM // 2 + 0x2000, 5, 2, top_b - 10, top_p - 2 * n)),
    ]
    for name, f in cases:
        await t.run(f, name)


@cocotb.test()
async def test_random_fetches(dut):
    """constrained-random legal fetches"""
    t = await fresh(dut)
    for i in range(int(os.environ.get("LAYERS") or 20)):
        await t.run(t.legal(), f"random #{i}")


@cocotb.test()
async def test_errors(dut):
    """E_DIM, E_ALIGN, both; E_RANGE on B in the third group and on P;
    E_RESP on a beat mid-group (the burst is drained, the groups before are
    whole); a clean fetch after each"""
    t = await fresh(dut)
    n = t.n
    cases = [
        ("k = 0", Fetch(0, MEM // 2, 0, 1, 0, 0)),
        ("groups = 0", Fetch(0, MEM // 2, 4, 0, 0, 0)),
        ("b_src misaligned", Fetch(4, MEM // 2, 4, 1, 0, 0)),
        ("p_src misaligned and k = 0", Fetch(0, MEM // 2 + 2, 0, 1, 0, 0)),
        ("B overflows in group 2", Fetch(0, MEM // 2, 50, 5, (1 << t.b_aw) - 120, 0)),
        ("P overflows in group 1", Fetch(0, MEM // 2, 3, 3, 0, (1 << t.p_aw) - n - 1)),
    ]
    for name, f in cases:
        await t.run(f, name)
        await t.run(t.legal(), f"clean after {name}")
    t.bad_ranges = [(0x4000 + 8 * 30, 0x4000 + 8 * 31)]
    await t.run(Fetch(0x4000, MEM // 2, 20, 3, 0, 0), "SLVERR on beat 30 (group 1)")
    t.bad_ranges = [(MEM // 2 + 8 * n, MEM // 2 + 8 * n + 8)]
    await t.run(Fetch(0x4000, MEM // 2, 20, 3, 0, 0), "SLVERR on group 1's first P entry")
    t.bad_ranges = []
    await t.run(t.legal(), "clean after the slave errors")


@cocotb.test()
async def test_abort(dut):
    """abort at many points: before the first AR, while an AR waits for
    ARREADY, mid-burst, between bursts. nothing is written after the abort
    edge, the burst in flight is drained (the slave owes nothing when wdma
    goes idle), no done/err follows, and a clean fetch works after"""
    t = await fresh(dut)
    f = Fetch(0x6000 - 8 * 100, MEM // 2, 150, 3, 0, 0)
    exp = fetch(t.slave.mem, f, t.n, t.b_aw, t.p_aw)
    total = estimate_cycles(exp, t.lat)
    points = sorted({1, 2, 3, 4, 5, 6, t.lat + 4, total // 3, total // 2, total - 3, total - 1}
                    | {t.rng.randint(1, total) for _ in range(6)})
    for at in points:
        await t.run(f, f"abort at {at}", abort_at=at)
        await t.run(t.legal(), f"clean after abort at {at}")
    # an AR held back by the slave while the abort comes in: ARVALID must stay
    t.slave.ar_stall = 0.9
    await t.run(f, "abort while ARREADY is low", abort_at=3)
    t.slave.ar_stall = t.stall
    await t.run(t.legal(), "clean after the stalled abort")


@cocotb.test()
async def test_host_access(dut):
    """single-beat writes to both halves of a word (only that half changes),
    reads back, SLVERR from the slave for writes and reads, a fetch after"""
    t = await fresh(dut)
    for i in range(40):
        addr = 4 * t.rng.randrange(0, MEM // 4)
        v = t.rng.getrandbits(32)
        before = beat(t.slave.mem, addr & ~7)
        resp, _ = await t.host(1, addr, v)
        assert resp == OKAY, f"host write {addr:#x}: resp {resp}"
        after = beat(t.slave.mem, addr & ~7)
        sh = 32 if addr & 4 else 0
        keep = ~(0xFFFFFFFF << sh) & (2**64 - 1)
        assert after == (before & keep) | (v << sh), f"host write {addr:#x}: word {after:#x}"
        resp, got = await t.host(0, addr)
        assert (resp, got) == (OKAY, v), f"host read {addr:#x}: ({resp}, {got:#x}), wrote {v:#x}"
    t.bad_ranges = [(0x100, 0x108)]
    before = bytes(t.slave.mem[0x100:0x108])
    assert (await t.host(1, 0x104, 0x1234))[0] == SLVERR
    assert bytes(t.slave.mem[0x100:0x108]) == before, "a refused write changed memory"
    assert (await t.host(0, 0x100))[0] == SLVERR
    t.bad_ranges = []
    await t.run(t.legal(), "fetch after host traffic")


@cocotb.test()
async def test_back_to_back(dut):
    """f_start held high through a fetch while the fields change (ignored);
    the held start is accepted on the first edge after done; an h_req during
    a fetch is ignored"""
    t = await fresh(dut)
    d = dut
    a, b = t.legal(k=9, groups=2), t.legal(k=5, groups=3)
    exp_a = fetch(t.slave.mem, a, t.n, t.b_aw, t.p_aw)
    exp_b = fetch(t.slave.mem, b, t.n, t.b_aw, t.p_aw)
    n_w = len(t.writes)
    await FallingEdge(d.clk)
    t.drive(a)
    d.f_start.value = 1
    await RisingEdge(d.clk)
    await FallingEdge(d.clk)
    t.drive(b)                                  # start stays high
    d.h_req.value, d.h_we.value, d.h_addr.value = 1, 1, 0x40   # ignored while busy
    kind, _, _ = await t.until_end()
    assert kind == "done"
    d.h_req.value = 0
    # until_end returned on the falling edge after done's next edge: the held
    # start was sampled on that edge, right after done
    await ReadOnly()
    assert int(d.busy.value), "the held start was not accepted right after done"
    await FallingEdge(d.clk)
    d.f_start.value = 0
    kind, _, _ = await t.until_end()
    assert kind == "done"
    got = [(k, a_, v) for _, k, a_, v in t.writes[n_w:]]
    want = [("b", x, v) for x, v in exp_a.b_writes] + [("p", x, v) for x, v in exp_a.p_writes] + \
           [("b", x, v) for x, v in exp_b.b_writes] + [("p", x, v) for x, v in exp_b.p_writes]
    assert sorted(got) == sorted(want) and len(got) == len(want), "back-to-back writes differ"
    assert not t.slave.aw_log, "an h_req during a fetch got through"
