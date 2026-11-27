"""AXI4 slave model for cocotb: a byte-addressed memory behind the five
channels, with programmable latency and backpressure, error injection, and
protocol checks on everything the master does.

Stands in for the MIG (and DDR3) in simulation. Used on wdma directly
(dma TB) and on the MIG port inside top (u_ddr), by passing the handle whose
signals carry the AXI names (`araddr`, `arvalid`, ... `bready`).

Timing (all from the master's point of view, edges counted on clk):
  * AR is accepted on an edge where ARVALID and ARREADY are high. ARREADY is
    dropped at random (ar_stall probability per cycle).
  * the first beat of a burst is valid `latency` edges after its AR
    handshake edge at the earliest; later beats follow one per cycle, with a
    random gap (r_gap probability per cycle). up to `max_outstanding` bursts
    are queued.
  * AW and W are taken independently (aw_stall / w_stall); B is offered
    `b_latency` edges after the last W beat and the AW of a burst are in.
Checks (any failure is recorded in .errors and raised):
  * ARBURST/AWBURST INCR, ARSIZE/AWSIZE = 3 (8 bytes, the bus width),
    8-byte aligned addresses, LEN <= 255, no burst across a 4 KB line
  * VALID never drops before its handshake, and the payload under it stays
    stable (AR, AW, W)
  * WLAST exactly on the last beat of the write burst
  * no more ARs than max_outstanding in flight is not an error: ARREADY just
    stays low until there is room
Drives on the falling edge, samples in ReadOnly.
"""

import random
from collections import deque

from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

OKAY, SLVERR, DECERR = 0, 2, 3


class Axi4Slave:

    def __init__(self, bus, clk, size=1 << 20, latency=4, ar_stall=0.0, r_gap=0.0,
                 aw_stall=0.0, w_stall=0.0, b_latency=2, max_outstanding=4, seed=0,
                 bad=None, log=None):
        self.bus, self.clk = bus, clk
        self.mem = bytearray(size)
        self.latency, self.ar_stall, self.r_gap = latency, ar_stall, r_gap
        self.aw_stall, self.w_stall, self.b_latency = aw_stall, w_stall, b_latency
        self.max_out = max_outstanding
        self.rng = random.Random(seed)
        self.bad = bad or (lambda addr: False)       # addr -> True: answer SLVERR
        self.bad_id = None                           # addr -> True: answer RID 1 (OKAY data)
        self.log = log
        self.errors = []
        self.ar_log = []          # (addr, beats) of every accepted AR, in order
        self.aw_log = []
        self.reads = deque()      # [addr, beats_left, ready_edge]
        self.edge = 0
        self.r_cur = None         # (data, resp, last) being presented
        self.aw_q, self.w_q = deque(), deque()    # accepted AW; W beats of the open write
        self.w_beats = []
        self.b_q = deque()        # [resp, ready_edge]
        self.b_cur = None
        self.prev = {}

    # -- memory ----------------------------------------------------------------
    def write_bytes(self, addr, data):
        self.mem[addr:addr + len(data)] = data

    def word(self, addr):
        return int.from_bytes(self.mem[addr:addr + 8], "little")

    def _fail(self, msg):
        self.errors.append(msg)
        raise AssertionError(f"AXI4 slave: {msg}")

    def _g(self, name):
        return int(getattr(self.bus, name).value)

    def _set(self, name, v):
        getattr(self.bus, name).value = v

    # -- run --------------------------------------------------------------------
    def start(self):
        import cocotb
        for s in ("arready", "rvalid", "rlast", "rdata", "rresp", "rid", "awready", "wready",
                  "bvalid", "bresp", "bid"):
            self._set(s, 0)
        cocotb.start_soon(self._run())

    def _burst_ok(self, kind, addr, length, size, burst):
        if burst != 1:
            self._fail(f"{kind}BURST {burst}, only INCR is used")
        if size != 3:
            self._fail(f"{kind}SIZE {size}, the bus is 8 bytes")
        if addr % 8:
            self._fail(f"{kind}ADDR {addr:#x} not 8-byte aligned")
        if (addr % 4096) + 8 * (length + 1) > 4096:
            self._fail(f"{kind} burst {addr:#x} + {length + 1} beats crosses a 4 KB line")

    def _stable(self, ch, now, keys):
        """a VALID that was up and not taken must stay up with the same payload"""
        p = self.prev.get(ch)
        if p is not None and p["pending"]:
            if not now[ch + "valid"]:
                self._fail(f"{ch.upper()}VALID dropped before its handshake")
            for k in keys:
                if now[k] != p[k]:
                    self._fail(f"{k} changed while {ch.upper()}VALID waited")

    async def _run(self):
        b = self.bus
        ar_ready = aw_ready = w_ready = 0
        while True:
            await ReadOnly()
            # payload is only defined (and only read) under its VALID
            s = {k: self._g(k) for k in ("arvalid", "awvalid", "wvalid", "rready", "bready")}
            for ch, keys in (("ar", ("araddr", "arlen", "arsize", "arburst")),
                             ("aw", ("awaddr", "awlen", "awsize", "awburst")),
                             ("w", ("wdata", "wstrb", "wlast"))):
                for k in keys:
                    s[k] = self._g(k) if s[ch + "valid"] else None
            self._stable("ar", s, ("araddr", "arlen", "arsize", "arburst"))
            self._stable("aw", s, ("awaddr", "awlen", "awsize", "awburst"))
            self._stable("w", s, ("wdata", "wstrb", "wlast"))
            ar_hs = s["arvalid"] and ar_ready
            aw_hs = s["awvalid"] and aw_ready
            w_hs = s["wvalid"] and w_ready
            r_hs = self.r_cur is not None and s["rready"]
            b_hs = self.b_cur is not None and s["bready"]
            self.prev["ar"] = dict(s, pending=s["arvalid"] and not ar_hs)
            self.prev["aw"] = dict(s, pending=s["awvalid"] and not aw_hs)
            self.prev["w"] = dict(s, pending=s["wvalid"] and not w_hs)
            await RisingEdge(self.clk)
            self.edge += 1
            # ---- the edge
            if ar_hs:
                self._burst_ok("AR", s["araddr"], s["arlen"], s["arsize"], s["arburst"])
                self.ar_log.append((s["araddr"], s["arlen"] + 1))
                self.reads.append([s["araddr"], s["arlen"] + 1, self.edge + self.latency - 1])
            if r_hs:
                self.r_cur = None
            if aw_hs:
                self._burst_ok("AW", s["awaddr"], s["awlen"], s["awsize"], s["awburst"])
                self.aw_log.append((s["awaddr"], s["awlen"] + 1))
                self.aw_q.append([s["awaddr"], s["awlen"] + 1])
            if w_hs:
                self.w_beats.append((s["wdata"], s["wstrb"], s["wlast"]))
            if b_hs:
                self.b_cur = None
            # a write completes when its AW and all its W beats are in
            if self.aw_q and self.w_beats and len(self.w_beats) >= self.aw_q[0][1]:
                addr, n = self.aw_q.popleft()
                beats, self.w_beats = self.w_beats[:n], self.w_beats[n:]
                resp = OKAY
                for i, (data, strb, last) in enumerate(beats):
                    if last != (i == n - 1):
                        self._fail(f"WLAST {last} on beat {i + 1} of {n}")
                    a = addr + 8 * i
                    if self.bad(a):
                        resp = SLVERR
                        continue
                    for j in range(8):
                        if strb >> j & 1:
                            self.mem[a + j] = (data >> (8 * j)) & 0xFF
                self.b_q.append([resp, self.edge + self.b_latency - 1])
            await FallingEdge(self.clk)
            # ---- drive for the next cycle
            ar_ready = int(len(self.reads) < self.max_out and self.rng.random() >= self.ar_stall)
            self._set("arready", ar_ready)
            if self.r_cur is None and self.reads and self.edge >= self.reads[0][2] \
                    and self.rng.random() >= self.r_gap:
                rd = self.reads[0]
                addr = rd[0]
                rd[0] += 8
                rd[1] -= 1
                last = rd[1] == 0
                if last:
                    self.reads.popleft()
                self.r_cur = (self.word(addr), SLVERR if self.bad(addr) else OKAY, last,
                              int(bool(self.bad_id and self.bad_id(addr))))
            if self.r_cur is not None:
                self._set("rdata", self.r_cur[0])
                self._set("rresp", self.r_cur[1])
                self._set("rlast", int(self.r_cur[2]))
                self._set("rid", self.r_cur[3])
                self._set("rvalid", 1)
            else:
                self._set("rvalid", 0)
                self._set("rlast", 0)
            aw_ready = int(self.rng.random() >= self.aw_stall)
            w_ready = int(self.rng.random() >= self.w_stall)
            self._set("awready", aw_ready)
            self._set("wready", w_ready)
            if self.b_cur is None and self.b_q and self.edge >= self.b_q[0][1]:
                self.b_cur = self.b_q.popleft()[0]
            if self.b_cur is not None:
                self._set("bresp", self.b_cur)
                self._set("bid", 0)
                self._set("bvalid", 1)
            else:
                self._set("bvalid", 0)

    def idle(self):
        return not self.reads and self.r_cur is None and not self.aw_q and not self.w_beats \
            and not self.b_q and self.b_cur is None
