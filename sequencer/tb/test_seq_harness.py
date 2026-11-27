""" cocotb tb for sequencer/rtl/sequencer.sv driving the real layer_engine
(sequencer/tb/seq_harness.sv).

Programs come from host/network.py (random LeNet / ResNet-style networks, the
same generator run_network.py uses); descriptors from protocol.pack_descriptor.
Every layer's output region is compared with the NumPy references
(sequencer_model through the layer_engine bank model), every layer's cycle
count in the cycle RAM with the engine formulas, and whole banks with the
mirror, so nothing outside the regions moves.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge
import numpy as np

from act_host import ActHost
from conv_layer_model import HwParams
import network as NET
import protocol as P


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

    def where(self):
        return f"(reproduce: make MODULE_NAME=seq_harness N={self.n} READ_LATENCY={self.lat} SEED={self.seed})"

    async def start(self):
        d = self.dut
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        self.host.idle_host()
        for s in ("start", "start_index", "desc_we", "desc_waddr", "desc_wdata", "cyc_rd_en",
                  "cyc_rd_addr", "b_we", "b_waddr", "b_wdata", "p_we", "p_waddr", "p_wdata"):
            getattr(d, s).value = 0
        d.reset.value = 1
        for _ in range(3):
            await RisingEdge(d.clk)
        await FallingEdge(d.clk)
        d.reset.value = 0
        await self.host.fill_all()
        d._log.info(f"N={self.n} READ_LATENCY={self.lat} NBUF={self.nbuf} DESC_DEPTH={self.depth} SEED={self.seed}")

    async def write_port(self, prefix, base, data):
        d = self.dut
        getattr(d, f"{prefix}_we").value = 1
        for i, w in enumerate(data):
            getattr(d, f"{prefix}_waddr").value = base + i
            getattr(d, f"{prefix}_wdata").value = w
            await FallingEdge(d.clk)
        getattr(d, f"{prefix}_we").value = 0

    async def load(self, net, first=0):
        await self.host.host_write(*net.input)
        for base, words in net.b_images:
            await self.write_port("b", base, words)
        for base, words in net.p_images:
            await self.write_port("p", base, words)
        await self.write_port("desc", first * P.DESC_WORDS, [w for d in net.descriptors for w in d])

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

    async def run_program(self, first=0, reset_at=None, limit=3_000_000):
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
        overhead = t - sum(cyc)
        if kind == "done":
            # measured: per layer descriptor 16 fetch reads + L + decode,
            # launch and next; plus the END descriptor's fetch and decode
            want = e.layers_done * (20 + self.lat) + 17 + self.lat
            assert overhead == want, f"{name}: sequencer overhead {overhead}, formula {want}"
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
    net = NET.build(t.rng, t.hw, t.m_bits, plan=["conv3", "max2", "res", "conv1", "gavg"],
                    read_latency=t.lat, desc_depth=t.depth)
    await t.load(net)
    kind, edges, starts = await t.run_program()
    assert starts == len(net.steps) - 1, f"{starts} engine starts for {len(net.steps) - 1} layers"
    ov = await t.check_network(net, 0, kind, edges, starts, "lenet+resnet")
    # overhead per descriptor is fixed: every layer's fetch + launch, plus the END fetch
    per = (ov - 1) / len(net.steps)
    dut._log.info(f"MEASURED sequencer overhead: {ov} edges for {len(net.steps)} descriptors "
                  f"(incl. END) = {per:.2f} per descriptor + 1")


@cocotb.test()
async def test_random_networks(dut):
    """random LeNet / ResNet-style programs, each from a nonzero start index"""
    t = await fresh(dut)
    first = 0
    for i in range(int(os.environ.get("LAYERS") or 3)):
        await t.host.fill_all()
        try:
            net = NET.build(t.rng, t.hw, t.m_bits, read_latency=t.lat, desc_depth=t.depth - first)
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
    net = NET.build(t.rng, t.hw, t.m_bits, plan=["conv3", "max2", "conv1", "gavg"],
                    read_latency=t.lat, desc_depth=t.depth)
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
                           t.hw, t.lat, t.m_bits, t.depth)
    assert net.expected.err_index == 2 and net.expected.err_code == 1 << 7
    await t.load(net)
    kind, edges, starts = await t.run_program()
    assert starts == 3, f"{starts} engine starts: the program did not stop at the bad descriptor"
    await t.check_network(net, 0, kind, edges, starts, "error at descriptor 2")
    # a good program right after, no reset: it must end with err_code 0
    good = NET.build(t.rng, t.hw, t.m_bits, plan=["conv3", "conv1"], read_latency=t.lat,
                     desc_depth=t.depth)
    await t.load(good)
    kind, edges, starts = await t.run_program()
    await t.check_network(good, 0, kind, edges, starts, "good program after the error")


@cocotb.test()
async def test_bad_op_and_no_end(dut):
    """an op of 6 is E_MODE; a program at the top of the RAM with no END runs
    its last descriptor and stops with E_NOEND"""
    t = await fresh(dut)
    net = NET.build(t.rng, t.hw, t.m_bits, plan=["conv3", "max2"], read_latency=t.lat,
                    desc_depth=t.depth)
    # 1) op 6 in place of the END-preceding max2
    bad = [list(d) for d in net.descriptors]
    bad[1][0] = (bad[1][0] & ~7) | 6
    await t.host.host_write(*net.input)
    for base, words in net.b_images:
        await t.write_port("b", base, words)
    for base, words in net.p_images:
        await t.write_port("p", base, words)
    await t.write_port("desc", 0, [w for d in bad for w in d])
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
    net = NET.build(t.rng, t.hw, t.m_bits, plan=["conv3", "conv3", "max2", "gavg"],
                    read_latency=t.lat, desc_depth=t.depth)
    await t.load(net)
    first_layer = net.expected.cycles[0]
    kind, edges, starts = await t.run_program(reset_at=first_layer + 40 + first_layer // 2)
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
    a = NET.build(t.rng, t.hw, t.m_bits, plan=["conv1", "max2"], read_latency=t.lat, desc_depth=t.depth)
    await t.load(a, 0)
    kind, edges, starts = await t.run_program(0)
    await t.check_network(a, 0, kind, edges, starts, "program A")
    cyc_a = await t.read_cycles(0, a.expected.layers_done)
    b = NET.build(t.rng, t.hw, t.m_bits, plan=["conv3", "gavg"], read_latency=t.lat, desc_depth=t.depth)
    await t.load(b, 8)
    kind, edges, starts = await t.run_program(8)
    await t.check_network(b, 8, kind, edges, starts, "program B")
    assert await t.read_cycles(0, a.expected.layers_done) == cyc_a, "program B overwrote A's cycles"
