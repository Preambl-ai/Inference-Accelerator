"""Host-side driver for an act_buffer behind a layer engine, shared by the
pool_layer, add_layer and layer_engine TBs.

The DUT must have the conv_layer-style host ports:
  host_mode, host_we, host_waddr, host_wdata, host_wr_sel   (write, host_mode = 1)
  host_rd_en, host_rd_addr, host_rd_sel, host_rd_data        (read, while idle)
and a clock named clk. Every bank is mirrored: a bank starts full of random
junk, the mirror only changes where the contract says an engine writes, and
check_bank compares the whole bank, so a wrong answer cannot hide in a word
that happens to hold its reset value, and a stray write anywhere is caught.

All driving happens on the falling edge, all sampling in the ReadOnly phase.
"""

import random

from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


def resolve(sig):
    raw = str(sig.value)
    if "x" in raw.lower() or "z" in raw.lower():
        return None
    return int(sig.value)


class ActHost:

    def __init__(self, dut, n, read_latency, addr_width, nbuf, seed, where):
        self.dut, self.n, self.lat = dut, n, read_latency
        self.depth, self.nbuf = 1 << addr_width, nbuf
        self.rng = random.Random(seed)
        self.where = where
        self.mem = [None] * nbuf

    def idle_host(self):
        d = self.dut
        for s in ("host_mode", "host_we", "host_waddr", "host_wdata", "host_wr_sel",
                  "host_rd_en", "host_rd_addr", "host_rd_sel"):
            getattr(d, s).value = 0

    def junk(self, count):
        return [self.rng.getrandbits(8 * self.n) for _ in range(count)]

    async def host_write(self, sel, base, data):
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

    async def fill_all(self):
        for sel in range(self.nbuf):
            await self.fill_bank(sel)

    async def read_words(self, sel, base, count):
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

    async def check_all(self, what=""):
        for sel in range(self.nbuf):
            await self.check_bank(sel, what)

    def expect(self, sel, base, words):
        for i, w in enumerate(words):
            self.mem[sel][base + i] = w

    async def run(self, fields, start="start", done="done", err="err", err_code="err_code",
                  busy="busy", ready="ready", reset_at=None, limit=200000, host_mode=0,
                  count=None):
        """drive the command fields, pulse start, wait for done or err.
        edges are counted from the accepting one. reset_at = t asserts reset
        for edge t and returns there. count: name of a 1-bit signal to count
        (e.g. a per-tile pulse) while running. returns a dict."""
        d = self.dut
        assert int(getattr(d, ready).value) == 1, "harness: DUT not ready before start"
        for k, v in fields.items():
            getattr(d, k).value = int(v) & ((1 << len(getattr(d, k))) - 1)
        d.host_mode.value = host_mode
        getattr(d, start).value = 1
        await RisingEdge(d.clk)
        await ReadOnly()
        assert int(getattr(d, busy).value) == 1, "start was not accepted"
        await FallingEdge(d.clk)
        getattr(d, start).value = 0
        d.host_mode.value = 0
        t, counted = 0, 0
        while True:
            if reset_at is not None and t + 1 == reset_at:
                d.reset.value = 1
            await RisingEdge(d.clk)
            t += 1
            await ReadOnly()
            if count:
                counted += int(getattr(d, count).value)
            dn, er = int(getattr(d, done).value), int(getattr(d, err).value)
            if reset_at is not None and t == reset_at:
                assert (int(getattr(d, ready).value), int(getattr(d, busy).value), dn, er) == (1, 0, 0, 0), \
                    "ready/busy/done/err wrong right after reset"
                await FallingEdge(d.clk)
                d.reset.value = 0
                return dict(kind="reset", cycles=t, err_code=None, counted=counted)
            assert not (dn and er), "done and err together"
            if dn or er:
                assert int(getattr(d, ready).value) == 1, "ready must return with done/err"
                code = int(getattr(d, err_code).value)
                assert (code == 0) == bool(dn), f"err_code={code:#x} with done={dn}"
                await FallingEdge(d.clk)
                await RisingEdge(d.clk)
                await ReadOnly()
                assert int(getattr(d, done).value) == 0 and int(getattr(d, err).value) == 0, \
                    "done/err longer than one cycle"
                assert int(getattr(d, err_code).value) == code, "err_code must hold"
                await FallingEdge(d.clk)
                return dict(kind="done" if dn else "err", cycles=t, err_code=code, counted=counted)
            assert int(getattr(d, busy).value) == 1, f"busy fell at edge {t} without done/err"
            assert t < limit, f"no done/err after {t} edges {self.where()}"
            await FallingEdge(d.clk)
