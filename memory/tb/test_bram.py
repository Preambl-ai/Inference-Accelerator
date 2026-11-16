""" cocotb tb for rtl/bram.sv """

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer

from bram_model import BramModel, UNKNOWN


def read_int(sig, name):
    raw = str(sig.value)
    if "x" in raw.lower() or "z" in raw.lower():
        raise AssertionError(f"{name} is not resolvable: {raw}")
    return int(sig.value)


class BramHarness:
    """Drives the DUT and the golden model from one input vector.

    Every cycle goes through step(). rdata is compared whenever the model
    knows what it should be; where the RTL leaves it undefined (before the
    first read, unwritten words, same-address collisions) nothing is checked.
    """

    def __init__(self, dut):
        self.dut = dut
        self.dw = int(dut.DATA_WIDTH.value)
        self.aw = int(dut.ADDR_WIDTH.value)
        self.lat = int(dut.READ_LATENCY.value)
        self.depth = 1 << self.aw
        self.d_max = (1 << self.dw) - 1
        self.model = BramModel(self.dw, self.aw, self.lat)
        self.cycle = 0
        self.checked = 0
        self.unknown = 0
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = random.Random(self.seed)

    async def start(self):
        d = self.dut
        d.we.value = 0
        d.re.value = 0
        d.waddr.value = 0
        d.raddr.value = 0
        d.wdata.value = 0
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        await self.step()
        d._log.info(
            f"DUT parameters: DATA_WIDTH={self.dw} ADDR_WIDTH={self.aw} "
            f"READ_LATENCY={self.lat} SEED={self.seed} (reproduce with: make "
            f"MODULE_NAME=bram DATA_WIDTH={self.dw} ADDR_WIDTH={self.aw} "
            f"READ_LATENCY={self.lat} SEED={self.seed})"
        )

    def word(self):
        """Random nonzero data, so a read that never happened cannot pass."""
        return self.rng.randint(1, self.d_max)

    async def step(self, we=0, waddr=None, wdata=None, re=0, raddr=None):
        """Advance one rising edge on both DUT and model, then compare.
        Idle ports get random values: a disabled port must be ignored."""
        waddr = self.rng.randrange(self.depth) if waddr is None else waddr
        raddr = self.rng.randrange(self.depth) if raddr is None else raddr
        wdata = self.word() if wdata is None else wdata

        d = self.dut
        await FallingEdge(d.clk)
        d.we.value = we
        d.waddr.value = waddr
        d.wdata.value = wdata
        d.re.value = re
        d.raddr.value = raddr

        await RisingEdge(d.clk)
        await ReadOnly()

        expected = self.model.step(we=we, waddr=waddr, wdata=wdata, re=re, raddr=raddr)
        if expected is UNKNOWN:
            self.unknown += 1
        else:
            got = read_int(d.rdata, "rdata")
            assert got == expected, (
                f"cycle {self.cycle}: rdata={got:#x}, expected {expected:#x}\n"
                f"stimulus: we={we} waddr={waddr} wdata={wdata:#x} re={re} raddr={raddr}\n"
                f"seed={self.seed}"
            )
            self.checked += 1
        self.cycle += 1
        return expected

    async def fill(self):
        """Write every word with distinct-looking random data."""
        image = {}
        for addr in self.rng.sample(range(self.depth), self.depth):
            image[addr] = self.word()
            await self.step(we=1, waddr=addr, wdata=image[addr])
        return image

    async def read(self, addr):
        """Read one word and return it once it is on rdata."""
        await self.step(re=1, raddr=addr)
        for _ in range(self.lat - 1):
            await self.step(re=1)
        return read_int(self.dut.rdata, "rdata")


# TESTS


@cocotb.test()
async def test_write_then_read_all(dut):
    """Every address written in random order reads back, in another order."""
    h = BramHarness(dut)
    await h.start()
    image = await h.fill()
    for addr in h.rng.sample(range(h.depth), h.depth):
        got = await h.read(addr)
        assert got == image[addr], f"addr {addr}: read {got:#x}, wrote {image[addr]:#x}"
    # the corners by name: address decode and data bit errors show up here
    for addr, data in ((0, h.d_max), (h.depth - 1, 1), (0, 1 << (h.dw - 1)), (h.depth - 1, h.d_max)):
        await h.step(we=1, waddr=addr, wdata=data)
        assert await h.read(addr) == data
    dut._log.info(f"{h.depth} words read back")


@cocotb.test()
async def test_latency_from_sim(dut):
    """Measure READ_LATENCY on the DUT in enabled edges."""
    h = BramHarness(dut)
    await h.start()
    marker, filler = h.d_max, 1
    await h.step(we=1, waddr=1, wdata=marker)
    await h.step(we=1, waddr=2, wdata=filler)
    for _ in range(3):
        await h.step(re=1, raddr=2)
    assert read_int(dut.rdata, "rdata") == filler

    edges = 0
    while read_int(dut.rdata, "rdata") != marker:
        await h.step(re=1, raddr=1 if edges == 0 else 2)
        edges += 1
        assert edges < 8, "the word never appeared"
    assert edges == h.lat, f"measured latency {edges}, parameter says {h.lat}"
    if h.lat == 2:
        # address 2 was presented right behind it and must follow one edge later
        await h.step(re=1, raddr=2)
        assert read_int(dut.rdata, "rdata") == filler
    dut._log.info(f"READ_LATENCY measured at {edges} enabled edges")


@cocotb.test()
async def test_streaming_read(dut):
    """A new address every cycle gives a new word every cycle, in order."""
    h = BramHarness(dut)
    await h.start()
    image = await h.fill()
    addrs = [h.rng.randrange(h.depth) for _ in range(4 * h.depth)]
    seen = []
    for i, addr in enumerate(addrs + [0] * h.lat):
        await h.step(re=1, raddr=addr)
        if i >= h.lat - 1:
            seen.append(read_int(dut.rdata, "rdata"))
    assert seen[:len(addrs)] == [image[a] for a in addrs], f"seed={h.seed}"


@cocotb.test()
async def test_re_stall_holds_pipeline(dut):
    """re low holds rdata and the word in flight; nothing presented meanwhile
    gets in, and writes still land."""
    h = BramHarness(dut)
    await h.start()
    image = await h.fill()
    addrs = [h.rng.randrange(h.depth) for _ in range(300)]
    seen = []
    presented = 0
    for addr in addrs:
        await h.step(re=1, raddr=addr)
        presented += 1
        if presented >= h.lat:
            seen.append(read_int(dut.rdata, "rdata"))
        for _ in range(h.rng.choice((0, 0, 1, 3))):
            held = read_int(dut.rdata, "rdata") if presented >= h.lat else None
            # a write to a word not in the read list, so the image stays valid
            await h.step(re=0)
            if held is not None:
                assert read_int(dut.rdata, "rdata") == held, "rdata moved with re low"
    want = [image[a] for a in addrs]
    assert seen == want[:len(seen)], f"stall dropped or repeated a word, seed={h.seed}"

    # writes with re low: every one must be there afterwards
    writes = {a: h.word() for a in h.rng.sample(range(h.depth), min(h.depth, 16))}
    for a, w in writes.items():
        await h.step(we=1, waddr=a, wdata=w, re=0, raddr=a)
    for a, w in writes.items():
        assert await h.read(a) == w, f"write to {a} with re low was lost"


@cocotb.test()
async def test_we_low_does_not_write(dut):
    """we low with live address and data on the pins changes nothing."""
    h = BramHarness(dut)
    await h.start()
    image = await h.fill()
    for addr in range(h.depth):
        await h.step(we=0, waddr=addr, wdata=image[addr] ^ h.d_max or 1)
    for addr in range(h.depth):
        assert await h.read(addr) == image[addr], f"addr {addr} changed with we low"


@cocotb.test()
async def test_random_traffic(dut):
    """Random reads and writes together, checked wherever rdata is defined."""
    h = BramHarness(dut)
    await h.start()
    await h.fill()
    h.checked = h.unknown = 0
    for _ in range(6000):
        await h.step(we=int(h.rng.random() < 0.5), re=int(h.rng.random() < 0.7))
    assert h.checked > 3000, f"only {h.checked} cycles were checkable"
    dut._log.info(f"random traffic: {h.checked} cycles checked, {h.unknown} "
                  f"undefined (collisions in flight)")


@cocotb.test()
async def test_collision_write_lands(dut):
    """A same-address read and write on one edge: the read is undefined and is
    not checked, but the write must land and later reads must be clean."""
    h = BramHarness(dut)
    await h.start()
    image = await h.fill()
    for _ in range(64):
        addr = h.rng.randrange(h.depth)
        image[addr] = h.word()
        exp = await h.step(we=1, waddr=addr, wdata=image[addr], re=1, raddr=addr)
        if h.lat == 1:
            assert exp is UNKNOWN
        other = (addr + 1 + h.rng.randrange(h.depth - 1)) % h.depth
        assert await h.read(other) == image[other], "read after a collision was corrupted"
        assert await h.read(addr) == image[addr], f"collision write to {addr} was lost"


@cocotb.test()
async def test_read_is_registered(dut):
    """raddr must not reach rdata within the cycle."""
    h = BramHarness(dut)
    await h.start()
    await h.step(we=1, waddr=0, wdata=1)
    await h.step(we=1, waddr=1, wdata=h.d_max)
    for _ in range(h.lat + 1):
        await h.step(re=1, raddr=0)
    await FallingEdge(dut.clk)
    dut.raddr.value = 1
    dut.re.value = 1
    await Timer(1, unit="ns")
    await ReadOnly()
    assert read_int(dut.rdata, "rdata") == 1, "rdata followed raddr combinationally"
    # that drive bypassed the model; flush both with reads of a known word
    await RisingEdge(dut.clk)
    h.model.step(re=1, raddr=1)
