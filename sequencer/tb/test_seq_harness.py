""" cocotb tb for sequencer/rtl/sequencer.sv driving the real layer_engine
(sequencer/tb/seq_harness.sv).

Programs come from host/network.py (random LeNet / ResNet-style networks, the
same generator run_network.py uses); descriptors from protocol.pack_descriptor.
Every layer's output region is compared with the NumPy references
(sequencer_model through the layer_engine bank model), every layer's cycle
count in the cycle RAM with the engine formulas, and whole banks with the
mirror, so nothing outside the regions moves.

Phase 4: with HAS_DMA = 1 the harness has a wdma whose AXI4 master talks to
axi4_slave.Axi4Slave (latency and backpressure from AXI_LAT / AXI_STALL), and
networks are built with weights "mixed" (SEQ_WEIGHTS overrides: chip, ddr,
mixed): each conv's B and P come from DDR or were loaded on chip. The
fetch counts in the second half of the cycle RAM are checked against the
wdma formula when the slave does not stall.

test_full_network runs one of the fixed plans at full size (32x32x3 input):
NET_PLAN=lenet, resnet8 or resnet20 (network.LENET / RESNET8 / RESNET20);
without NET_PLAN it does nothing. sweep-net runs all three.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge
import numpy as np

from act_host import ActHost
from axi4_slave import Axi4Slave
from conv_layer_model import HwParams
import network as NET
import protocol as P


DMA_EXTRA = 2       # edges per DDR layer beyond its fetch count (measured: see the contract)


class SeqTB:

    def __init__(self, dut):
        self.dut = dut
        g = lambda name: int(getattr(dut, name).value)
        self.n, self.lat, self.aw, self.nbuf = g("N"), g("READ_LATENCY"), g("ADDR_WIDTH"), g("NBUF")
        self.depth = g("DESC_DEPTH")
        self.m_bits = g("M_BITS")
        self.hw = HwParams(n=self.n, addr_width=self.aw, nbuf=self.nbuf, k_max=g("K_MAX"),
                           b_addr_w=g("B_ADDR_W"), p_addr_w=g("P_ADDR_W"))
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = np.random.default_rng(self.seed)
        self.host = ActHost(dut, self.n, self.lat, self.aw, self.nbuf, self.seed, self.where)
        self.has_dma = g("HAS_DMA")
        self.alat = int(os.environ.get("AXI_LAT") or 3)
        self.stall = float(os.environ.get("AXI_STALL") or 0)
        self.weights = os.environ.get("SEQ_WEIGHTS") or ("mixed" if self.has_dma else "chip")
        self.bad_ranges = []
        self.slave = Axi4Slave(dut, dut.clk, size=1 << 20, latency=self.alat, ar_stall=self.stall,
                               r_gap=self.stall, seed=self.seed,
                               bad=lambda a: any(lo <= a < hi for lo, hi in self.bad_ranges))

    def netb(self, plan=None, desc_depth=None, **kw):
        """network.build with this harness's sizes and weight source"""
        return NET.build(self.rng, self.hw, self.m_bits, plan=plan, read_latency=self.lat,
                         desc_depth=desc_depth or self.depth, weights=self.weights,
                         axi_latency=self.alat, **kw)

    def where(self):
        return (f"(reproduce: make MODULE_NAME=seq_harness N={self.n} READ_LATENCY={self.lat} "
                f"SEQ_HAS_DMA={self.has_dma} AXI_LAT={self.alat} AXI_STALL={self.stall} "
                f"SEQ_WEIGHTS={self.weights} SEED={self.seed})")

    async def start(self):
        d = self.dut
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        self.host.idle_host()
        for s in ("start", "start_index", "desc_we", "desc_waddr", "desc_wdata", "cyc_rd_en",
                  "cyc_rd_addr", "b_we", "b_waddr", "b_wdata", "p_we", "p_waddr", "p_wdata",
                  "abort"):
            getattr(d, s).value = 0
        d.reset.value = 1
        for _ in range(3):
            await RisingEdge(d.clk)
        await FallingEdge(d.clk)
        if self.has_dma:
            self.slave.start()
        d.reset.value = 0
        await self.host.fill_all()
        d._log.info(f"N={self.n} READ_LATENCY={self.lat} NBUF={self.nbuf} DESC_DEPTH={self.depth} "
                    f"HAS_DMA={self.has_dma} weights={self.weights} AXI_LAT={self.alat} "
                    f"AXI_STALL={self.stall} SEED={self.seed}")

    async def dma_idle(self):
        """after a soft reset: wdma drains its burst, then the slave owes nothing"""
        d = self.dut
        for _ in range(100000):
            await RisingEdge(d.clk)
            await ReadOnly()
            if not int(d.dma_busy.value):
                break
        assert not int(d.dma_busy.value), "wdma still busy long after the soft reset"
        assert self.slave.idle(), "wdma idle with AXI beats still owed"
        await FallingEdge(d.clk)

    def expect_with(self, net, **kw):
        """re-run the model on net (after its descriptors were edited)"""
        from sequencer_model import run as seq_run, Step
        import copy
        from layer_engine_model import Banks
        banks = Banks(self.nbuf, 1 << self.aw)
        banks.write(*net.input)
        net.expected = seq_run([Step(x.op, x.fields, copy.deepcopy(x.layer)) for x in net.steps],
                               banks, self.hw, self.lat, self.m_bits, self.depth, ddr=net.ddr,
                               axi_latency=self.alat, **kw)

    async def write_port(self, prefix, base, data):
        d = self.dut
        getattr(d, f"{prefix}_we").value = 1
        for i, w in enumerate(data):
            getattr(d, f"{prefix}_waddr").value = base + i
            getattr(d, f"{prefix}_wdata").value = w
            await FallingEdge(d.clk)
        getattr(d, f"{prefix}_we").value = 0

    async def load(self, net, first=0, descriptors=None):
        await self.host.host_write(*net.input)
        for base, words in net.b_images:
            await self.write_port("b", base, words)
        for base, words in net.p_images:
            await self.write_port("p", base, words)
        for addr, data in net.ddr_images:
            self.slave.write_bytes(addr, data)
        descs = descriptors or net.descriptors
        await self.write_port("desc", first * P.DESC_WORDS, [w for d in descs for w in d])

    async def read_cycles(self, first, count):
        d = self.dut
        out = []
        d.cyc_rd_en.value = 1
        for i in range(count):
            d.cyc_rd_addr.value = first + i
            for _ in range(self.lat):
                await RisingEdge(d.clk)
            await ReadOnly()
            out.append(int(d.cyc_rd_data.value))
            await FallingEdge(d.clk)
        d.cyc_rd_en.value = 0
        return out

    async def run_program(self, first=0, reset_at=None, limit=3_000_000, abort_at=None):
        """start, wait for done/err; returns (kind, edges, engine starts seen)"""
        d = self.dut
        d.start_index.value = first
        d.start.value = 1
        await RisingEdge(d.clk)
        await ReadOnly()
        assert int(d.busy.value), "program did not start"
        await FallingEdge(d.clk)
        d.start.value = 0
        t, starts = 0, 0
        while True:
            if reset_at is not None and t + 1 == reset_at:
                d.reset.value = 1
            if abort_at is not None and t + 1 == abort_at:
                d.abort.value = 1               # soft reset, as top wires it
            if abort_at is not None and t == abort_at:
                d.abort.value = 0
                await RisingEdge(d.clk)
                await ReadOnly()
                assert not int(d.busy.value) and not int(d.eng_busy.value), "busy after soft reset"
                await FallingEdge(d.clk)
                return "abort", t, starts
            await RisingEdge(d.clk)
            t += 1
            await ReadOnly()
            starts += int(d.eng_start_o.value)
            if reset_at is not None and t == reset_at:
                assert not int(d.busy.value) and not int(d.eng_busy.value), "busy after reset"
                await FallingEdge(d.clk)
                d.reset.value = 0
                return "reset", t, starts
            dn, er = int(d.done.value), int(d.err.value)
            assert not (dn and er)
            if dn or er:
                assert not int(d.busy.value), "busy with done/err"
                await FallingEdge(d.clk)
                return ("done" if dn else "err"), t, starts
            assert t < limit, f"program did not finish {self.where()}"
            await FallingEdge(d.clk)

    async def check_network(self, net, first, kind, t, starts, name):
        e = net.expected
        d = self.dut
        assert kind == ("done" if e.err_index is None else "err"), f"{name}: ended {kind}"
        assert int(d.layers_done.value) == e.layers_done, \
            f"{name}: {int(d.layers_done.value)} layers done, expected {e.layers_done}"
        if e.err_index is None:
            # a program that reaches END reports no error, even right after one
            # that failed (a mutation run found a stale code went unseen)
            assert int(d.err_code.value) == 0, \
                f"{name}: err_code {int(d.err_code.value):#x} after a program that reached END"
        if e.err_index is not None:
            assert (int(d.err_index.value), int(d.err_code.value)) == (e.err_index, e.err_code), \
                f"{name}: error at {int(d.err_index.value)} code {int(d.err_code.value):#x}, " \
                f"expected {e.err_index} {e.err_code:#x}"
        for (bank, base, cnt), words in zip(net.outputs, e.outputs):
            self.host.expect(bank, base, words)
        await self.host.check_all(f"after {name}")
        cyc = await self.read_cycles(first, e.layers_done)
        assert cyc == e.cycles, f"{name}: cycle RAM {cyc}, formulas {e.cycles} {self.where()}"
        n_ddr = sum(1 for f in e.fetch_cycles if f)
        fetch = [0] * e.layers_done
        if self.has_dma:
            got = await self.read_cycles(self.depth + first, e.layers_done)
            fetch = got
            assert all(g == 0 for g, f in zip(got, e.fetch_cycles) if not f), \
                f"{name}: an on-chip layer's fetch count is not 0: {got}"
            if self.stall == 0:
                assert fetch == e.fetch_cycles, \
                    f"{name}: fetch cycles {fetch}, wdma formula {e.fetch_cycles} {self.where()}"
            else:
                assert all((g > 0) == (f > 0) for g, f in zip(fetch, e.fetch_cycles))
        overhead = t - sum(cyc) - sum(fetch)
        if kind == "done":
            # measured: per layer descriptor 16 fetch reads + L + decode,
            # launch and next; plus the END descriptor's fetch and decode;
            # plus, per DDR layer, the edges around its weight fetch
            want = e.layers_done * (20 + self.lat) + 17 + self.lat + n_ddr * DMA_EXTRA
            assert overhead == want, \
                f"{name}: sequencer overhead {overhead}, formula {want} ({n_ddr} DDR layers)"
        self.dut._log.info(f"PROGRAM {name}: {e.layers_done} layers {net.names[:e.layers_done]}, "
                           f"{t} edges start to done, layers {sum(cyc)}, sequencer overhead {overhead} "
                           f"({overhead / max(1, e.layers_done + 1):.1f} per descriptor), {starts} engine starts")
        return overhead


async def fresh(dut):
    t = SeqTB(dut)
    await t.start()
    return t


@cocotb.test()
async def test_lenet_and_resnet_program(dut):
    """conv3 -> max2 -> residual block -> conv1 -> gavg -> END: every layer's
    output and cycle count"""
    t = await fresh(dut)
    net = t.netb(["conv3", "max2", "res", "conv1", "gavg"])
    await t.load(net)
    kind, edges, starts = await t.run_program()
    assert starts == len(net.steps) - 1, f"{starts} engine starts for {len(net.steps) - 1} layers"
    ov = await t.check_network(net, 0, kind, edges, starts, "lenet+resnet")
    # overhead per descriptor is fixed: every layer's fetch + launch, plus the END fetch
    per = (ov - 1) / len(net.steps)
    dut._log.info(f"MEASURED sequencer overhead: {ov} edges for {len(net.steps)} descriptors "
                  f"(incl. END) = {per:.2f} per descriptor + 1")


PLANS = {"lenet": NET.LENET, "resnet8": NET.RESNET8, "resnet20": NET.RESNET20}


@cocotb.test()
async def test_full_network(dut):
    """LeNet-5, ResNet-8 or ResNet-20 (NET_PLAN) on a 32x32x3 input, from one
    start: every layer whose output survives to the end matches NumPy, every
    layer's cycle count its formula, every fetch the wdma formula (no
    stall); activation space is reused (keep="last"), so earlier outputs are
    checked through the banks they leave behind"""
    name = os.environ.get("NET_PLAN")
    if not name:
        return
    t = await fresh(dut)
    net = t.netb(PLANS[name], h=32, w=32, c=3, keep="last")
    n_ddr = sum(1 for s in net.steps if s.fields.get("w_ddr"))
    await t.load(net)
    kind, edges, starts = await t.run_program(limit=10_000_000)
    assert starts == len(net.steps) - 1, f"{starts} engine starts for {len(net.steps) - 1} layers"
    ov = await t.check_network(net, 0, kind, edges, starts, name)
    dut._log.info(f"NETWORK {name}: {len(net.steps) - 1} layers, {n_ddr} with DDR weights, "
                  f"{edges} edges start to done (layers {sum(net.expected.cycles)}, fetches "
                  f"{sum(net.expected.fetch_cycles)}, sequencer {ov})")


@cocotb.test()
async def test_random_networks(dut):
    """random LeNet / ResNet-style programs, each from a nonzero start index"""
    t = await fresh(dut)
    first = 0
    for i in range(int(os.environ.get("LAYERS") or 3)):
        await t.host.fill_all()
        try:
            net = t.netb(desc_depth=t.depth - first)
        except MemoryError:
            continue
        await t.load(net, first)
        kind, edges, starts = await t.run_program(first)
        await t.check_network(net, first, kind, edges, starts, f"random #{i} from {first}")
        first = (first + len(net.steps)) % (t.depth - 12)


@cocotb.test()
async def test_error_stops_the_program(dut):
    """an illegal descriptor mid-program: err at that index with the engine's
    code, the layers before it done (outputs and cycles), nothing after it"""
    t = await fresh(dut)
    net = t.netb(["conv3", "max2", "conv1", "gavg"])
    # descriptor 2 (the conv1) reads and writes the same bank: E_SEL
    s = net.steps[2]
    s.fields["dst_sel"] = s.fields["src_sel"]
    s.layer.dst_sel = s.layer.src_sel
    net.descriptors[2] = P.pack_descriptor(s.op, s.fields)
    from sequencer_model import run as seq_run, Step
    import copy
    from layer_engine_model import Banks
    banks = Banks(t.nbuf, 1 << t.aw)
    banks.write(*net.input)
    net.expected = seq_run([Step(x.op, x.fields, copy.deepcopy(x.layer)) for x in net.steps], banks,
                           t.hw, t.lat, t.m_bits, t.depth, ddr=net.ddr, axi_latency=t.alat)
    assert net.expected.err_index == 2 and net.expected.err_code == 1 << 7
    await t.load(net)
    kind, edges, starts = await t.run_program()
    assert starts == 3, f"{starts} engine starts: the program did not stop at the bad descriptor"
    await t.check_network(net, 0, kind, edges, starts, "error at descriptor 2")
    # a good program right after, no reset: it must end with err_code 0
    good = t.netb(["conv3", "conv1"])
    await t.load(good)
    kind, edges, starts = await t.run_program()
    await t.check_network(good, 0, kind, edges, starts, "good program after the error")


@cocotb.test()
async def test_bad_op_and_no_end(dut):
    """an op of 6 is E_MODE; a program at the top of the RAM with no END runs
    its last descriptor and stops with E_NOEND"""
    t = await fresh(dut)
    net = t.netb(["conv3", "max2"])
    # 1) op 6 in place of the END-preceding max2
    bad = [list(d) for d in net.descriptors]
    bad[1][0] = (bad[1][0] & ~7) | 6
    await t.load(net, 0, descriptors=bad)
    kind, edges, starts = await t.run_program()
    assert kind == "err" and int(t.dut.err_index.value) == 1 and int(t.dut.err_code.value) == 1 << 9
    assert int(t.dut.layers_done.value) == 1
    # 2) the two layers at the very top of the RAM, with no END after them
    first = t.depth - 2
    await t.host.fill_all()
    await t.host.host_write(*net.input)
    await t.write_port("desc", first * P.DESC_WORDS, [w for d in net.descriptors[:2] for w in d])
    kind, edges, starts = await t.run_program(first)
    assert kind == "err" and int(t.dut.err_code.value) == 1 << 10, int(t.dut.err_code.value)
    assert int(t.dut.err_index.value) == t.depth - 1 and int(t.dut.layers_done.value) == 2
    for (bank, base, cnt), words in zip(net.outputs, net.expected.outputs):
        t.host.expect(bank, base, words)
    await t.host.check_all("after E_NOEND")
    assert await t.read_cycles(first, 2) == net.expected.cycles[:2]


@cocotb.test()
async def test_reset_mid_program_and_rerun(dut):
    """reset in the middle of the second layer: everything idle right after;
    the same program then runs clean from the start"""
    t = await fresh(dut)
    net = t.netb(["conv3", "conv3", "max2", "gavg"])
    await t.load(net)
    first_layer = net.expected.cycles[0] + net.expected.fetch_cycles[0]
    at = first_layer + 40 + first_layer // 2
    if t.has_dma:
        # the board reset also resets DDR, which the slave model cannot do;
        # what the board does to a running program is the soft reset
        kind, edges, starts = await t.run_program(abort_at=at)
        assert kind == "abort"
        await t.dma_idle()
    else:
        kind, edges, starts = await t.run_program(reset_at=at)
        assert kind == "reset"
    await t.host.fill_all()
    await t.host.host_write(*net.input)
    kind, edges, starts = await t.run_program()
    await t.check_network(net, 0, kind, edges, starts, "rerun after reset")


@cocotb.test()
async def test_back_to_back_programs(dut):
    """two programs in the RAM; the second starts on the cycle after the first
    one's done, and the first one's cycle RAM entries survive"""
    t = await fresh(dut)
    a = t.netb(["conv1", "max2"])
    await t.load(a, 0)
    kind, edges, starts = await t.run_program(0)
    await t.check_network(a, 0, kind, edges, starts, "program A")
    cyc_a = await t.read_cycles(0, a.expected.layers_done)
    b = t.netb(["conv3", "gavg"])
    await t.load(b, 8)
    kind, edges, starts = await t.run_program(8)
    await t.check_network(b, 8, kind, edges, starts, "program B")
    assert await t.read_cycles(0, a.expected.layers_done) == cyc_a, "program B overwrote A's cycles"


@cocotb.test()
async def test_weight_fetch_errors(dut):
    """HAS_DMA = 0: a DDR layer stops the program with E_NODMA at its index.
    HAS_DMA = 1: a misaligned b_src is wdma's E_ALIGN (bit 12), a slave error
    on a B beat is E_RESP (bit 14), each at the failing descriptor with the
    layers before it complete; a good program after each ends clean"""
    t = await fresh(dut)
    net = NET.build(t.rng, t.hw, t.m_bits, plan=["conv3", "max2", "conv1", "gavg"],
                    read_latency=t.lat, desc_depth=t.depth, weights="ddr", axi_latency=t.alat)
    if not t.has_dma:
        t.expect_with(net, has_dma=False)
        assert (net.expected.err_index, net.expected.err_code) == (0, 1 << 15)
        await t.load(net)
        kind, edges, starts = await t.run_program()
        assert starts == 0, "a DDR layer reached the engine in a build without wdma"
        await t.check_network(net, 0, kind, edges, starts, "DDR layer without wdma")
        return
    # misaligned b_src on the second conv (descriptor 2)
    s2 = net.steps[2]
    s2.fields["b_src"] += 4
    net.descriptors[2] = P.pack_descriptor(s2.op, s2.fields)
    t.expect_with(net)
    assert (net.expected.err_index, net.expected.err_code) == (2, 1 << 12)
    await t.load(net)
    kind, edges, starts = await t.run_program()
    await t.check_network(net, 0, kind, edges, starts, "misaligned b_src at descriptor 2")
    good = t.netb(["conv3", "conv1"])
    await t.load(good)
    kind, edges, starts = await t.run_program()
    await t.check_network(good, 0, kind, edges, starts, "good program after E_ALIGN")
    # the slave answers SLVERR on a beat in the middle of the second conv's B
    net = NET.build(t.rng, t.hw, t.m_bits, plan=["conv3", "max2", "conv1", "gavg"],
                    read_latency=t.lat, desc_depth=t.depth, weights="ddr", axi_latency=t.alat)
    f2 = net.steps[2].fields
    bad = f2["b_src"] + 8 * (f2["k_words"] // 2)
    t.bad_ranges = [(bad, bad + 8)]
    t.expect_with(net, bad=lambda a: a == bad)
    assert (net.expected.err_index, net.expected.err_code) == (2, 1 << 14)
    await t.load(net)
    kind, edges, starts = await t.run_program()
    await t.check_network(net, 0, kind, edges, starts, "slave error at descriptor 2")
    t.bad_ranges = []
    await t.load(good)
    kind, edges, starts = await t.run_program()
    await t.check_network(good, 0, kind, edges, starts, "good program after E_RESP")


@cocotb.test()
async def test_soft_reset_mid_fetch(dut):
    """soft reset while wdma is mid-burst: the sequencer stops at once, wdma
    finishes the AXI burst it owes and writes nothing more, and the same
    program then runs clean (HAS_DMA = 1 only)"""
    t = await fresh(dut)
    if not t.has_dma:
        return
    net = NET.build(t.rng, t.hw, t.m_bits, plan=["conv3", "conv3", "gavg"], read_latency=t.lat,
                    desc_depth=t.depth, weights="ddr", axi_latency=t.alat)
    await t.load(net)
    e = net.expected
    # inside the second layer's fetch: after layer 0 (fetch + run) and the
    # second descriptor's fetch from the RAM
    for at in (20 + t.lat + 3 + e.fetch_cycles[0] // 2,
               2 * (20 + t.lat) + e.fetch_cycles[0] + e.cycles[0] + e.fetch_cycles[1] // 3):
        kind, edges, starts = await t.run_program(abort_at=at)
        assert kind == "abort"
        await t.dma_idle()
        await t.host.fill_all()
        await t.host.host_write(*net.input)
    kind, edges, starts = await t.run_program()
    await t.check_network(net, 0, kind, edges, starts, "rerun after soft resets mid-fetch")
