""" cocotb tb for rtl/act_buffer.sv. The contract is memory/act_buffer_contract.md. """

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer

from act_buffer_model import ActBufferModel, pack_lanes, sel_width, unpack_lanes
from bram_model import UNKNOWN


def read_int(sig, name):
    raw = str(sig.value)
    if "x" in raw.lower() or "z" in raw.lower():
        raise AssertionError(f"{name} is not resolvable: {raw}")
    return int(sig.value)


def allow_bad_sel():
    return os.environ.get("ALLOW_BAD_SEL") == "1"


class ActBufferHarness:
    """Drives the DUT and the golden model from one input vector.

    Every cycle goes through step(), which compares rd_data with the model in
    the ReadOnly phase whenever the model knows what it should be. Ports that
    are not in use this cycle get random values, selects included, and those
    selects may name no bank at all: an unused port must be ignored.
    """

    PORTS = ("rd_en", "rd_addr", "rd_sel", "we", "waddr", "wdata", "wr_sel",
             "host_mode", "host_we", "host_waddr", "host_wdata", "host_wr_sel")

    def __init__(self, dut):
        self.dut = dut
        g = lambda name: int(getattr(dut, name).value)
        self.n = g("N")
        self.dw = g("DATA_WIDTH")
        self.aw = g("ADDR_WIDTH")
        self.lat = g("READ_LATENCY")
        self.nbuf = g("NBUF")
        self.sw = g("SEL_WIDTH")
        self.depth = 1 << self.aw
        self.ww = self.n * self.dw
        self.w_max = (1 << self.ww) - 1
        self.allow_bad = allow_bad_sel()
        self.model = ActBufferModel(self.n, self.dw, self.aw, self.lat, self.nbuf,
                                    allow_bad_sel=self.allow_bad)
        self.cycle = 0
        self.checked = 0
        self.unknown = 0
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = random.Random(self.seed)

    async def start(self):
        d = self.dut
        for p in self.PORTS:
            getattr(d, p).value = 0
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        await self.step()
        d._log.info(
            f"DUT parameters: N={self.n} DATA_WIDTH={self.dw} ADDR_WIDTH={self.aw} "
            f"READ_LATENCY={self.lat} NBUF={self.nbuf} SEL_WIDTH={self.sw} SEED={self.seed} "
            f"(reproduce with: make MODULE_NAME=act_buffer N={self.n} DATA_WIDTH={self.dw} "
            f"ADDR_WIDTH={self.aw} READ_LATENCY={self.lat} NBUF={self.nbuf} "
            f"ALLOW_BAD_SEL={int(self.allow_bad)} SEED={self.seed})"
        )

    # stimulus

    def word(self):
        """Random nonzero word, so a read that never happened cannot pass."""
        return self.rng.randint(1, self.w_max)

    def bank(self):
        return self.rng.randrange(self.nbuf)

    def any_sel(self):
        """Any value the select port can carry, banks or not."""
        return self.rng.randrange(1 << self.sw)

    async def step(self, check=True, **v):
        """Advance one rising edge on both DUT and model, then compare.

        Anything not given is random. Selects default to a real bank on the
        side in use and to anything at all on an idle side."""
        v.setdefault("rd_en", 0)
        v.setdefault("we", 0)
        v.setdefault("host_mode", 0)
        v.setdefault("host_we", 0)
        r = self.rng
        host = v["host_mode"]
        rd_used = v["rd_en"] and not host
        v.setdefault("rd_sel", self.bank() if rd_used else self.any_sel())
        v.setdefault("wr_sel", self.bank() if (v["we"] and not host) else self.any_sel())
        v.setdefault("host_wr_sel", self.bank() if (v["host_we"] and host) else self.any_sel())
        for a in ("rd_addr", "waddr", "host_waddr"):
            v.setdefault(a, r.randrange(self.depth))
        for dname in ("wdata", "host_wdata"):
            v.setdefault(dname, self.word())

        d = self.dut
        await FallingEdge(d.clk)
        for p in self.PORTS:
            getattr(d, p).value = v[p]

        await RisingEdge(d.clk)
        await ReadOnly()

        expected = self.model.step(**v)
        if expected is UNKNOWN or not check:
            self.unknown += 1
        else:
            got = read_int(d.rd_data, "rd_data")
            assert got == expected, (
                f"cycle {self.cycle}: rd_data={got:#x}, expected {expected:#x}\n"
                f"  lanes got      {unpack_lanes(got, self.n, self.dw)}\n"
                f"  lanes expected {unpack_lanes(expected, self.n, self.dw)}\n"
                f"stimulus: {v}\nseed={self.seed}"
            )
            self.checked += 1
        self.cycle += 1
        return expected

    def rd_data(self):
        return read_int(self.dut.rd_data, "rd_data")

    async def fill(self, banks=None):
        """Write every word of the given banks (all by default), random order."""
        banks = range(self.nbuf) if banks is None else banks
        image = {}
        cells = [(b, a) for b in banks for a in range(self.depth)]
        self.rng.shuffle(cells)
        for (b, a) in cells:
            image[(b, a)] = self.word()
            await self.step(we=1, wr_sel=b, waddr=a, wdata=image[(b, a)])
        return image

    async def read(self, sel, addr):
        """Read one word and return it once it is on rd_data."""
        await self.step(rd_en=1, rd_sel=sel, rd_addr=addr)
        for _ in range(self.lat - 1):
            await self.step(rd_en=1)
        return self.rd_data()

    async def stream(self, reads, stall_choice=(0,), extra=None):
        """Issue reads back to back with optional stalls between them, and
        return the words as they come out, in order. Checks against the model
        happen in step() as usual; the caller checks the order independently.

        extra(i) may return more port values for the cycle that issues read i.
        Stalls drive rd_en low and leave everything else random."""
        seen = []
        issued = 0
        flush = [(reads[-1][0], reads[-1][1])] * (self.lat - 1)
        for i, (sel, addr) in enumerate(list(reads) + flush):
            kw = extra(i) if (extra and i < len(reads)) else {}
            await self.step(rd_en=1, rd_sel=sel, rd_addr=addr, **kw)
            issued += 1
            if issued >= self.lat:
                seen.append(self.rd_data())
            for _ in range(self.rng.choice(stall_choice)):
                held = self.rd_data() if issued >= self.lat else None
                await self.step(rd_en=0)
                if held is not None:
                    assert self.rd_data() == held, f"rd_data moved with rd_en low, seed={self.seed}"
        return seen[:len(reads)]


# TESTS


@cocotb.test()
async def test_select_width(dut):
    """SEL_WIDTH is at least 1 and the ports are really that wide. NBUF=1 used
    to give [-1:0], a 2-bit port with three selects that name no bank."""
    h = ActBufferHarness(dut)
    want = sel_width(h.nbuf)
    assert h.sw == want, f"SEL_WIDTH={h.sw}, contract says {want} for NBUF={h.nbuf}"
    for p in ("rd_sel", "wr_sel", "host_wr_sel"):
        assert len(getattr(dut, p)) == want, f"{p} is {len(getattr(dut, p))} bits, want {want}"
    for p in ("rd_data", "wdata", "host_wdata"):
        assert len(getattr(dut, p)) == h.ww, f"{p} is {len(getattr(dut, p))} bits, want N*DATA_WIDTH"


@cocotb.test()
async def test_write_then_read_all_banks(dut):
    """Every word of every bank, written in random order, reads back in
    another random order with the bank changing freely."""
    h = ActBufferHarness(dut)
    await h.start()
    image = await h.fill()
    cells = list(image)
    h.rng.shuffle(cells)
    got = await h.stream(cells)
    bad = [c for c, w in zip(cells, got) if image[c] != w]
    assert not bad, f"{len(bad)} words wrong, first at bank/addr {bad[0]}, seed={h.seed}"
    dut._log.info(f"{len(cells)} words over {h.nbuf} banks read back")


@cocotb.test()
async def test_latency_from_sim(dut):
    """Measure the read latency on the DUT in advancing edges."""
    h = ActBufferHarness(dut)
    await h.start()
    marker, filler = h.w_max, 1
    mb, fb = h.nbuf - 1, 0
    await h.step(we=1, wr_sel=mb, waddr=1, wdata=marker)
    await h.step(we=1, wr_sel=fb, waddr=2, wdata=filler)
    for _ in range(3):
        await h.step(rd_en=1, rd_sel=fb, rd_addr=2)
    assert h.rd_data() == filler

    edges = 0
    while h.rd_data() != marker:
        if edges == 0:
            await h.step(rd_en=1, rd_sel=mb, rd_addr=1)
        else:
            await h.step(rd_en=1, rd_sel=fb, rd_addr=2)
        edges += 1
        assert edges < 8, "the word never appeared"
    assert edges == h.lat, f"measured latency {edges}, parameter says {h.lat}"
    if h.lat == 2:
        await h.step(rd_en=1, rd_sel=fb, rd_addr=2)
        assert h.rd_data() == filler, "the read behind the marker did not follow it"
    dut._log.info(f"read latency measured at {edges} advancing edges")


@cocotb.test()
async def test_sel_every_cycle(dut):
    """A different bank on every cycle, same addresses in every bank but
    different data, so a select that slips a stage relative to its data reads
    the right address from the wrong bank and is caught."""
    h = ActBufferHarness(dut)
    await h.start()
    image = await h.fill()
    reads = []
    for i in range(6 * h.depth):
        prev = reads[-1][0] if reads else 0
        sel = (prev + 1 + h.rng.randrange(h.nbuf - 1)) % h.nbuf if h.nbuf > 1 else 0
        reads.append((sel, h.rng.randrange(h.depth)))
    got = await h.stream(reads)
    want = [image[c] for c in reads]
    assert got == want, f"select and data out of step, seed={h.seed}"


@cocotb.test()
async def test_rd_en_stalls_mid_stream(dut):
    """rd_en low at random points holds rd_data, the word in flight and its
    select; the stream comes out with nothing dropped or repeated. Writes to
    other words keep going during the stalls."""
    h = ActBufferHarness(dut)
    await h.start()
    image = await h.fill()
    # the read list leaves address 0 of every bank alone, so writes there
    # during stalls cannot change what the list should return
    reads = [(h.bank(), 1 + h.rng.randrange(h.depth - 1)) for _ in range(400)]
    got = await h.stream(reads, stall_choice=(0, 0, 1, 1, 2, 5))
    assert got == [image[c] for c in reads], f"stall dropped or repeated a word, seed={h.seed}"
    # and the same with writes landing on the stall cycles, in the model too
    for _ in range(200):
        await h.step(rd_en=int(h.rng.random() < 0.5), rd_addr=1 + h.rng.randrange(h.depth - 1),
                     we=1, waddr=0)
    assert h.checked > 400


@cocotb.test()
async def test_random_traffic(dut):
    """Random reads, writes and host_mode together, checked every cycle the
    model can say what rd_data is."""
    h = ActBufferHarness(dut)
    await h.start()
    await h.fill()
    h.checked = h.unknown = 0
    host = 0
    for _ in range(8000):
        if h.rng.random() < 0.03:
            host ^= 1
        await h.step(rd_en=int(h.rng.random() < 0.7), we=int(h.rng.random() < 0.5),
                     host_mode=host, host_we=int(h.rng.random() < 0.5))
    assert h.checked > 5000, f"only {h.checked} cycles were checkable"
    dut._log.info(f"random traffic: {h.checked} cycles checked, {h.unknown} undefined")


@cocotb.test()
async def test_host_mode_entry_exit(dut):
    """host_mode in the middle of a read stream: rd_data holds whatever rd_en,
    rd_addr and rd_sel do, datapath writes are ignored, host writes land
    (including on the word held on rd_data), and the stream resumes with the
    word that was in flight."""
    h = ActBufferHarness(dut)
    await h.start()
    image = await h.fill()
    for rnd in range(20):
        reads = [(h.bank(), h.rng.randrange(h.depth)) for _ in range(12)]
        got = await h.stream(reads[:6])
        assert got == [image[c] for c in reads[:6]]
        # one more read goes in, so at L=2 a word is in flight when host_mode rises
        await h.step(rd_en=1, rd_sel=reads[6][0], rd_addr=reads[6][1])
        held = h.rd_data()

        host_writes, dp_tried = {}, {}
        held_cell = reads[5] if h.lat == 2 else reads[6]
        for k in range(h.rng.randint(1, 30)):
            b, a, w = h.bank(), h.rng.randrange(h.depth), h.word()
            if k == 0:
                b, a = held_cell        # overwrite the word on rd_data
            host_writes[(b, a)] = w
            db, da, dw = h.bank(), h.rng.randrange(h.depth), h.word()
            dp_tried[(db, da)] = dw
            await h.step(host_mode=1, host_we=1, host_wr_sel=b, host_waddr=a, host_wdata=w,
                         we=1, wr_sel=db, waddr=da, wdata=dw,
                         rd_en=1, rd_sel=h.any_sel())
            assert h.rd_data() == held, f"rd_data moved in host_mode, round {rnd}"
        # host_mode with rd_en low too, then out
        await h.step(host_mode=1, rd_en=0)
        assert h.rd_data() == held

        image.update(host_writes)
        for c in dp_tried:
            if c not in host_writes:
                assert h.model.peek(*c) == image[c]
        # resumes: at L=2 the word in flight (reads[6]) comes out on the first
        # advance. it was read before host_mode, so it is the old word even if
        # the host overwrote that cell; step() checks it against the model.
        # stream() only collects from the second advance at L=2, so what it
        # returns lines up with reads[7:] at either latency.
        got = await h.stream(reads[7:])
        assert got == [image[c] for c in reads[7:]], f"stream after host_mode wrong, round {rnd}"
    # every datapath write tried in host_mode is absent, every host write present
    for (b, a), w in image.items():
        assert await h.read(b, a) == w, f"bank {b} addr {a} wrong after host_mode rounds"


@cocotb.test()
async def test_ping_pong(dut):
    """Write bank i while reading bank i-1 on the same cycles, rotating through
    every bank twice. With NBUF=1 the halves of the one bank take turns."""
    h = ActBufferHarness(dut)
    await h.start()
    if h.nbuf == 1:
        half = h.depth // 2
        regions = [[(0, a) for a in range(half)], [(0, a) for a in range(half, h.depth)]]
    else:
        regions = [[(b, a) for a in range(h.depth)] for b in range(h.nbuf)]
    image = {}
    for c in regions[-1]:
        image[c] = h.word()
        await h.step(we=1, wr_sel=c[0], waddr=c[1], wdata=image[c])
    for rnd in range(2 * len(regions)):
        wr_reg = regions[rnd % len(regions)]
        rd_reg = regions[(rnd - 1) % len(regions)]
        new = {c: h.word() for c in wr_reg}
        order = list(rd_reg)
        h.rng.shuffle(order)
        wr_iter = iter(wr_reg)

        def extra(i):
            c = next(wr_iter, None)
            if c is None:
                return {}
            return dict(we=1, wr_sel=c[0], waddr=c[1], wdata=new[c])

        got = await h.stream(order, extra=extra)
        assert got == [image[c] for c in order], f"ping-pong round {rnd} read wrong, seed={h.seed}"
        for c in wr_iter:      # any left over if the write region is larger
            await h.step(we=1, wr_sel=c[0], waddr=c[1], wdata=new[c])
        image.update(new)
    dut._log.info(f"ping-pong over {len(regions)} regions, {h.checked} cycles checked")


@cocotb.test()
async def test_same_cycle_read_write(dut):
    """Read and write of the same bank and address on one edge: the read is
    undefined (bram) and not checked, the write lands, and the reads next to
    it are clean. The same address in another bank is defined: old data."""
    h = ActBufferHarness(dut)
    await h.start()
    image = await h.fill()
    collisions = 0
    for _ in range(64):
        b, a = h.bank(), h.rng.randrange(h.depth)
        image[(b, a)] = h.word()
        exp = await h.step(rd_en=1, rd_sel=b, rd_addr=a,
                           we=1, wr_sel=b, waddr=a, wdata=image[(b, a)])
        if h.lat == 1:
            assert exp is UNKNOWN
        collisions += 1
        other = (a + 1 + h.rng.randrange(h.depth - 1)) % h.depth
        assert await h.read(b, other) == image[(b, other)], "read after a collision was corrupted"
        assert await h.read(b, a) == image[(b, a)], f"collision write to {b}/{a} was lost"
    if h.nbuf > 1:
        for _ in range(64):
            rb, a = h.bank(), h.rng.randrange(h.depth)
            wb = (rb + 1 + h.rng.randrange(h.nbuf - 1)) % h.nbuf
            old = image[(rb, a)]
            image[(wb, a)] = h.word()
            await h.step(rd_en=1, rd_sel=rb, rd_addr=a, we=1, wr_sel=wb, waddr=a, wdata=image[(wb, a)])
            for _ in range(h.lat - 1):
                await h.step(rd_en=1, rd_sel=rb, rd_addr=a)
            assert h.rd_data() == old, "a write to another bank disturbed this read"
            assert await h.read(wb, a) == image[(wb, a)]
    dut._log.info(f"{collisions} same-bank collisions, write landed every time")


@cocotb.test()
async def test_extreme_values(dut):
    """All ones, all zeros, alternating bits, and signed min/max per lane, in
    every bank, so a lane swap, a lost top bit or a shifted lane shows up."""
    h = ActBufferHarness(dut)
    await h.start()
    dw, n = h.dw, h.n
    smin, smax = -(1 << (dw - 1)), (1 << (dw - 1)) - 1
    alt = int("01" * h.ww, 2) & h.w_max
    patterns = [
        h.w_max, 0, alt, alt ^ h.w_max,
        pack_lanes([smin] * n, dw), pack_lanes([smax] * n, dw),
        pack_lanes([smin if i % 2 else smax for i in range(n)], dw),
        pack_lanes([smax if i % 2 else smin for i in range(n)], dw),
        pack_lanes([-1 if i % 2 else 0 for i in range(n)], dw),
        # lane i holds i+1 (mod the lane width), so any lane reordering shows
        pack_lanes([(i + 1) % (1 << dw) for i in range(n)], dw),
        1, 1 << (h.ww - 1),
    ]
    cells = []
    for b in range(h.nbuf):
        for k, p in enumerate(patterns):
            a = (k * 7 + b) % h.depth
            cells.append((b, a, p))
    for b, a, p in cells:
        # the later ones may overwrite earlier ones at small ADDR_WIDTH; only
        # the last write to each cell is checked below
        await h.step(we=1, wr_sel=b, waddr=a, wdata=p)
    final = {(b, a): p for b, a, p in cells}
    for (b, a), p in final.items():
        got = await h.read(b, a)
        assert got == p, (f"bank {b} addr {a}: lanes {unpack_lanes(got, n, dw)} "
                          f"expected {unpack_lanes(p, n, dw)}")
    # and the same patterns through the host port
    for (b, a), p in final.items():
        await h.step(host_mode=1, host_we=1, host_wr_sel=b, host_waddr=a, host_wdata=p ^ h.w_max)
    for (b, a), p in final.items():
        assert await h.read(b, a) == p ^ h.w_max, f"host write of bank {b} addr {a} wrong"


@cocotb.test()
async def test_read_is_registered(dut):
    """rd_addr and rd_sel must not reach rd_data within the cycle."""
    h = ActBufferHarness(dut)
    await h.start()
    b1 = h.nbuf - 1
    await h.step(we=1, wr_sel=0, waddr=0, wdata=1)
    await h.step(we=1, wr_sel=b1, waddr=1, wdata=h.w_max)
    for _ in range(h.lat + 1):
        await h.step(rd_en=1, rd_sel=0, rd_addr=0)
    await FallingEdge(dut.clk)
    dut.rd_addr.value = 1
    dut.rd_sel.value = b1
    dut.rd_en.value = 1
    await Timer(1, unit="ns")
    await ReadOnly()
    assert read_int(dut.rd_data, "rd_data") == 1, "rd_data followed rd_addr/rd_sel combinationally"
    await RisingEdge(dut.clk)
    h.model.step(rd_en=1, rd_sel=b1, rd_addr=1)


@cocotb.test(skip=not allow_bad_sel())
async def test_bad_select(dut):
    """A select that names no bank (sel >= NBUF). Illegal, and normally the
    RTL stops with $fatal; with +act_buffer_allow_bad_sel (make
    ALLOW_BAD_SEL=1) this proves the hardware behaviour the contract states:
    such a write changes no bank, on either write port, and such a read
    returns undefined data for its slot only, with the reads either side of
    it still in step."""
    h = ActBufferHarness(dut)
    bad_sels = list(range(h.nbuf, 1 << h.sw))
    if not bad_sels:
        # NBUF a power of two: no select value names no bank
        dut._log.info(f"NBUF={h.nbuf} uses every select value, nothing to test")
        return
    await h.start()
    image = await h.fill()
    for s in bad_sels:
        for a in h.rng.sample(range(h.depth), min(h.depth, 8)):
            await h.step(we=1, wr_sel=s, waddr=a, wdata=h.word())
            await h.step(host_mode=1, host_we=1, host_wr_sel=s, host_waddr=a, host_wdata=h.word())
    assert h.model.bad_writes == 2 * len(bad_sels) * min(h.depth, 8)
    for (b, a), w in image.items():
        assert await h.read(b, a) == w, f"a bad-select write reached bank {b} addr {a}"

    # bad reads mixed into a stream: the model marks those slots UNKNOWN and
    # step() skips them; the good ones must all still match
    reads = []
    for _ in range(300):
        if h.rng.random() < 0.25:
            reads.append((h.rng.choice(bad_sels), h.rng.randrange(h.depth)))
        else:
            reads.append((h.bank(), h.rng.randrange(h.depth)))
    h.checked = 0
    for sel, addr in reads + [(0, 0)] * h.lat:
        await h.step(rd_en=1, rd_sel=sel, rd_addr=addr)
    good = sum(1 for s, _ in reads if s < h.nbuf)
    assert h.checked >= good, f"{h.checked} reads checked, {good} good reads issued"
    assert h.model.bad_reads > 0
    dut._log.info(f"{h.model.bad_writes} bad writes dropped, {h.model.bad_reads} bad reads "
                  f"skipped, {h.checked} good reads checked around them")
