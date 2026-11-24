""" cocotb tb for layer_engine/rtl/layer_engine.sv

conv and pool ops against one shared act_buffer. Expected contents come from
layer_engine_model (conv_ref / pool references over a mirror of every bank);
the mirror is the act_host mirror, and whole banks are compared, so a write
by the wrong engine, to the wrong bank or outside a region is caught. Every
conv and pool command's start-to-done count is compared with its formula.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge
import numpy as np

from act_host import ActHost
import conv_layer_model as CM
from conv_layer_model import HwParams, pack_act, pack_b, pack_p, random_layer, words
from layer_engine_model import (EXTRA_FIELDS, OP_ADD, OP_CONV, OP_GAVG, OP_MAX2, Banks, add_cmd_of,
                                add_fields, apply_add, apply_conv, apply_pool, check, conv_bus,
                                pool_cmd_of, pool_fields)
import add_model as AM
import pool_model as PM

CMD_FIELDS = ("h_in", "w_in", "h_out", "w_out", "cin_words", "cout_words", "kh", "kw", "stride",
              "pad", "zp_in", "zp_out", "relu_en", "src_sel", "dst_sel", "src_base", "dst_base",
              "b_base", "p_base") + EXTRA_FIELDS


class EngineTB:

    def __init__(self, dut):
        self.dut = dut
        g = lambda name: int(getattr(dut, name).value)
        self.n, self.lat, self.aw, self.nbuf = g("N"), g("READ_LATENCY"), g("ADDR_WIDTH"), g("NBUF")
        self.hw = HwParams(n=self.n, addr_width=self.aw, nbuf=self.nbuf, k_max=g("K_MAX"),
                           b_addr_w=g("B_ADDR_W"), p_addr_w=g("P_ADDR_W"))
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = np.random.default_rng(self.seed)
        self.host = ActHost(dut, self.n, self.lat, self.aw, self.nbuf, self.seed, self.where)
        self.banks = Banks(self.nbuf, 1 << self.aw)

    def where(self):
        return f"(reproduce: make MODULE_NAME=layer_engine N={self.n} READ_LATENCY={self.lat} SEED={self.seed})"

    async def start(self):
        d = self.dut
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        self.host.idle_host()
        for s in ("start", "op", "b_we", "p_we", "b_waddr", "b_wdata", "p_waddr", "p_wdata") + CMD_FIELDS:
            getattr(d, s).value = 0
        d.reset.value = 1
        for _ in range(3):
            await RisingEdge(d.clk)
        await FallingEdge(d.clk)
        d.reset.value = 0
        await self.host.fill_all()
        self.banks.b = self.host.mem              # one mirror for host and model
        d._log.info(f"N={self.n} READ_LATENCY={self.lat} ADDR_WIDTH={self.aw} NBUF={self.nbuf} SEED={self.seed}")

    async def write_bp(self, layer):
        d = self.dut
        for name, base, data in (("b", layer.b_base, pack_b(layer.w, self.n)),
                                 ("p", layer.p_base, pack_p(layer, self.n))):
            getattr(d, f"{name}_we").value = 1
            for i, w in enumerate(data):
                getattr(d, f"{name}_waddr").value = base + i
                getattr(d, f"{name}_wdata").value = w
                await FallingEdge(d.clk)
            getattr(d, f"{name}_we").value = 0

    def conv_fields(self, layer):
        return dict(conv_bus(layer, self.n), op=OP_CONV)

    async def conv(self, layer, name, load_x=True, check_banks=False):
        if load_x:
            await self.host.host_write(layer.src_sel, layer.src_base, pack_act(layer.x, self.n, layer.zp_in))
        await self.write_bp(layer)
        f = self.conv_fields(layer)
        assert check(OP_CONV, layer.command(self.n), self.hw) == 0, "stimulus bug"
        res = await self.host.run(f, count="tile_launch")
        assert res["kind"] == "done", f"{name}: rejected {res['err_code']:#x}"
        assert int(self.dut.done_op.value) == OP_CONV
        apply_conv(self.banks, layer, self.n)
        cmd = layer.command(self.n)
        est = CM.estimate_cycles(cmd, self.n, self.lat)
        assert res["cycles"] == est, f"{name}: conv {res['cycles']} edges, formula {est} {self.where()}"
        tiles = -(-cmd["h_out"] * cmd["w_out"] // self.n) * cmd["cout_words"]
        assert res["counted"] == tiles, f"{name}: {res['counted']} tiles, expected {tiles}"
        if check_banks:
            await self.host.check_all(f"after {name}")
        return res

    async def pool(self, op, fields, name, check_banks=False):
        assert check(op, fields, self.hw) == 0, "stimulus bug"
        res = await self.host.run(dict(fields, op=op))
        assert res["kind"] == "done", f"{name}: rejected {res['err_code']:#x}"
        assert int(self.dut.done_op.value) == op
        apply_pool(self.banks, op, fields, self.n)
        est = PM.estimate_cycles(pool_cmd_of(op, fields), self.lat)
        assert res["cycles"] == est, f"{name}: pool {res['cycles']} edges, formula {est} {self.where()}"
        if check_banks:
            await self.host.check_all(f"after {name}")
        return res


    async def add(self, cmd, name, check_banks=False):
        f = add_fields(cmd)
        assert check(OP_ADD, f, self.hw) == 0, f"stimulus bug {check(OP_ADD, f, self.hw):#x}"
        res = await self.host.run(dict(f, op=OP_ADD))
        assert res["kind"] == "done", f"{name}: rejected {res['err_code']:#x}"
        assert int(self.dut.done_op.value) == OP_ADD
        apply_add(self.banks, f, self.n)
        est = AM.estimate_cycles(cmd, self.lat, int(self.dut.M_BITS.value))
        assert res["cycles"] == est, f"{name}: add {res['cycles']} edges, formula {est} {self.where()}"
        if check_banks:
            await self.host.check_all(f"after {name}")
        return res


async def fresh(dut):
    t = EngineTB(dut)
    await t.start()
    return t


@cocotb.test()
async def test_residual_block(dut):
    """a ResNet basic block on the real engine: y = conv(conv(x)), out =
    relu(y + x) with different scales and zero points, whole banks checked.
    two banks: the skip input shares bank 0 with the block's output"""
    t = await fresh(dut)
    n = t.n
    three = t.nbuf >= 3
    x = t.rng.integers(-128, 128, (4, 4, n))
    l1 = random_layer(t.rng, t.hw, kh=3, kw=3, stride=1, pad=1, h_in=4, w_in=4, cin=n, cout=n,
                      src_sel=0, dst_sel=1, src_base=0, dst_base=0, b_base=0, p_base=0)
    l1.x = x
    await t.conv(l1, "block conv1", check_banks=True)
    l2 = random_layer(t.rng, t.hw, kh=3, kw=3, stride=1, pad=1, h_in=4, w_in=4, cin=n, cout=n,
                      src_sel=1, dst_sel=2 if three else 0, src_base=0, dst_base=0 if three else 200,
                      b_base=200, p_base=16)
    l2.zp_in = l1.zp_out
    await t.conv(l2, "block conv2", load_x=False, check_banks=True)
    ma, mb, sh = AM.scales_to_params(0.031, 0.017, 0.04)
    ac = AM.AddCmd(4, 4, 1, l2.dst_sel, l2.dst_base, 0, 0, 1, 300, za=l2.zp_out, zb=l1.zp_in,
                   zo=int(t.rng.integers(-20, 21)), ma=ma, mb=mb, shift=sh, relu_en=1)
    await t.add(ac, "block add (skip from x)", check_banks=True)


@cocotb.test()
async def test_conv_pool_chain(dut):
    """conv -> max2 -> conv -> gavg, each op reading what the last one wrote,
    alternating banks; whole banks checked after every op"""
    t = await fresh(dut)
    n = t.n
    l1 = random_layer(t.rng, t.hw, kh=3, kw=3, stride=1, pad=1, h_in=8, w_in=8, cin=n, cout=n,
                      src_sel=0, dst_sel=1, src_base=0, dst_base=0, b_base=0, p_base=0)
    await t.conv(l1, "conv1", check_banks=True)
    cw1 = words(l1.cout, n)
    await t.pool(OP_MAX2, pool_fields(OP_MAX2, 8, 8, cw1, 1, 0, 0, 200), "max2", check_banks=True)
    # second conv reads the pooled 4x4 map; its x is re-read from the mirror
    l2 = random_layer(t.rng, t.hw, kh=3, kw=3, stride=1, pad=1, h_in=4, w_in=4, cin=l1.cout,
                      cout=n, src_sel=0, dst_sel=1, src_base=200, dst_base=300, b_base=0, p_base=0)
    l2.zp_in = l1.zp_out
    await t.conv(l2, "conv2", load_x=False, check_banks=True)
    await t.pool(OP_GAVG, pool_fields(OP_GAVG, 4, 4, words(l2.cout, n), 1, 0, 300, 0), "gavg",
                 check_banks=True)


@cocotb.test()
async def test_random_ops(dut):
    """a random mix of conv and pool commands on random banks and bases"""
    t = await fresh(dut)
    n = t.n
    for i in range(int(os.environ.get("LAYERS") or 8)):
        src, dst = [int(v) for v in t.rng.choice(t.nbuf, 2, replace=False)]
        if i % 3 == 0:
            layer = random_layer(t.rng, t.hw, max_dim=6, src_sel=src, dst_sel=dst)
            await t.conv(layer, f"random conv #{i}")
        elif i % 3 == 1:
            cmd, a, b = AM.random_cmd(t.rng, n, t.aw, t.nbuf, max_dim=4, max_cw=2,
                                      m_bits=int(t.dut.M_BITS.value))
            await t.host.host_write(cmd.src_sel, cmd.src_base, pack_act(a, n, cmd.za))
            await t.host.host_write(cmd.src2_sel, cmd.src2_base, pack_act(b, n, cmd.zb))
            await t.add(cmd, f"random add #{i}")
        else:
            cmd, x, zp = PM.random_cmd(t.rng, n, t.aw, t.nbuf, max_dim=6, max_cw=2)
            await t.host.host_write(cmd.src_sel, cmd.src_base, pack_act(x, n, zp))
            op = OP_MAX2 if cmd.mode == PM.M_MAX2 else OP_GAVG
            await t.pool(op, pool_fields(op, cmd.h_in, cmd.w_in, cmd.c_words, cmd.src_sel,
                                         cmd.dst_sel, cmd.src_base, cmd.dst_base), f"random pool #{i}")
    await t.host.check_all("after the random ops")


@cocotb.test()
async def test_back_to_back_starts(dut):
    """the next command starts in the cycle right after done: a pool on the
    conv output, then a conv on the pool output"""
    t = await fresh(dut)
    d, n = t.dut, t.n
    l1 = random_layer(t.rng, t.hw, kh=1, kw=1, stride=1, pad=0, h_in=4, w_in=4, cin=n, cout=n,
                      src_sel=0, dst_sel=1, src_base=0, dst_base=0, b_base=0, p_base=0)
    await t.host.host_write(0, 0, pack_act(l1.x, n, l1.zp_in))
    await t.write_bp(l1)
    l2 = random_layer(t.rng, t.hw, kh=1, kw=1, stride=1, pad=0, h_in=2, w_in=2, cin=l1.cout,
                      cout=n, src_sel=0, dst_sel=1, src_base=100, dst_base=100, b_base=64, p_base=8)
    l2.zp_in = l1.zp_out
    await t.write_bp(l2)
    cw = words(l1.cout, n)
    seq = [(OP_CONV, t.conv_fields(l1)),
           (OP_MAX2, pool_fields(OP_MAX2, 4, 4, cw, 1, 0, 0, 100)),
           (OP_CONV, t.conv_fields(l2))]
    for op, f in seq:
        for k in CMD_FIELDS:
            getattr(d, k).value = int(f[k]) & ((1 << len(getattr(d, k))) - 1)
        d.op.value = op
        d.start.value = 1
        await RisingEdge(d.clk)
        await ReadOnly()
        assert int(d.busy.value) == 1, f"op {op} not accepted right after the previous done"
        await FallingEdge(d.clk)
        d.start.value = 0
        while True:
            await RisingEdge(d.clk)
            await ReadOnly()
            if int(d.done.value) or int(d.err.value):
                assert int(d.done.value), f"op {op} failed: {int(d.err_code.value):#x}"
                break
        await FallingEdge(d.clk)                 # start the next one now
    apply_conv(t.banks, l1, n)
    apply_pool(t.banks, OP_MAX2, seq[1][1], n)
    apply_conv(t.banks, l2, n)
    await t.host.check_all("after back-to-back commands")


@cocotb.test()
async def test_start_while_busy_ignored(dut):
    """starts of any op while an engine runs are ignored: one done, and the
    other op's destination is untouched"""
    t = await fresh(dut)
    d, n = t.dut, t.n
    layer = random_layer(t.rng, t.hw, kh=3, kw=3, stride=1, pad=1, h_in=6, w_in=6, cin=n, cout=n,
                         src_sel=0, dst_sel=1, src_base=0, dst_base=0, b_base=0, p_base=0)
    await t.host.host_write(0, 0, pack_act(layer.x, n, layer.zp_in))
    await t.write_bp(layer)
    f = t.conv_fields(layer)
    for k in CMD_FIELDS:
        getattr(d, k).value = int(f[k]) & ((1 << len(getattr(d, k))) - 1)
    d.op.value = OP_CONV
    d.start.value = 1
    await FallingEdge(d.clk)
    dones = 0
    # hammer start with every op (and a pool aimed at bank 0) while conv runs
    for cyc in range(40):
        d.op.value = [OP_MAX2, OP_GAVG, OP_CONV, OP_ADD, 0][cyc % 5]
        d.dst_sel.value = 0
        d.start.value = 1
        await RisingEdge(d.clk)
        await ReadOnly()
        assert int(d.conv_busy.value) and not int(d.pool_busy.value) and not int(d.add_busy.value), \
            "a second engine started"
        dones += int(d.done.value)
        await FallingEdge(d.clk)
    d.start.value = 0
    while int(d.busy.value):
        await RisingEdge(d.clk)
        await ReadOnly()
        dones += int(d.done.value)
        await FallingEdge(d.clk)
    assert dones == 1, f"{dones} done pulses"
    apply_conv(t.banks, layer, n)
    await t.host.check_all("after the ignored starts")


@cocotb.test()
async def test_bad_op_and_error_codes(dut):
    """ops outside 1..3 end in E_MODE one edge after start; conv and pool
    error codes come through unchanged; nothing is written; a good op still
    runs"""
    t = await fresh(dut)
    n = t.n
    for op in (0, 5, 6, 7):
        res = await t.host.run(dict(pool_fields(OP_MAX2, 4, 4, 1, 0, 1, 0, 0), op=op))
        assert res["kind"] == "err" and res["err_code"] == 1 << 9 and res["cycles"] == 1, res
        assert int(t.dut.done_op.value) == op
    # conv E_SEL (src == dst), pool E_SHAPE (GAVG on 3x4), pool E_SEL
    layer = random_layer(t.rng, t.hw, kh=1, kw=1, stride=1, pad=0, h_in=2, w_in=2, cin=n, cout=n,
                         src_sel=0, dst_sel=1, b_base=0, p_base=0)
    f = dict(t.conv_fields(layer), dst_sel=layer.src_sel)
    res = await t.host.run(f)
    assert res["err_code"] == check(OP_CONV, f, t.hw) == 1 << 7, res
    for op, pf in ((OP_GAVG, pool_fields(OP_GAVG, 3, 4, 1, 0, 1, 0, 0)),
                   (OP_MAX2, pool_fields(OP_MAX2, 4, 4, 1, 1, 1, 0, 0))):
        res = await t.host.run(dict(pf, op=op))
        assert res["kind"] == "err" and res["err_code"] == check(op, pf, t.hw), res
    await t.host.check_all("after the rejected commands")
    await t.conv(layer, "good conv after the errors", check_banks=True)


@cocotb.test()
async def test_reset_mid_op(dut):
    """reset during a conv and during a pool: ready right after, every
    destination word old or final, then a clean rerun"""
    t = await fresh(dut)
    n = t.n
    layer = random_layer(t.rng, t.hw, kh=3, kw=3, stride=1, pad=1, h_in=4, w_in=4, cin=n, cout=n,
                         src_sel=0, dst_sel=1, src_base=0, dst_base=0, b_base=0, p_base=0)
    await t.host.host_write(0, 0, pack_act(layer.x, n, layer.zp_in))
    await t.write_bp(layer)
    pf = pool_fields(OP_MAX2, 4, 4, 1, 0, 1, 0, 100)
    for op, f in ((OP_CONV, t.conv_fields(layer)), (OP_MAX2, dict(pf, op=OP_MAX2))):
        full = (CM.estimate_cycles(layer.command(n), n, t.lat) if op == OP_CONV
                else PM.estimate_cycles(pool_cmd_of(op, pf), t.lat))
        for at in (2, full // 2, full):
            await t.host.fill_bank(1)
            old = list(t.host.mem[1])
            res = await t.host.run(f, reset_at=at)
            assert res["kind"] == "reset"
            got = await t.host.read_words(1, 0, t.host.depth)
            mirror = Banks(t.nbuf, t.host.depth)
            mirror.b = [list(b) for b in t.host.mem]
            final = apply_conv(mirror, layer, n) if op == OP_CONV else apply_pool(mirror, op, pf, n)
            base = layer.dst_base if op == OP_CONV else pf["dst_base"]
            for a in range(t.host.depth):
                k = a - base
                if not (0 <= k < len(final) and got[a] == final[k]):
                    assert got[a] == old[a], f"op {op} reset at {at}: word {a} neither old nor final"
        if op == OP_CONV:
            await t.conv(layer, "conv rerun after reset", load_x=False, check_banks=True)
        else:
            await t.pool(op, pf, "pool rerun after reset", check_banks=True)


@cocotb.test()
async def test_host_reads_ignored_while_busy(dut):
    """host_rd_* toggling during a pool and a conv changes nothing: the
    engine owns the read port"""
    t = await fresh(dut)
    d, n = t.dut, t.n
    cmd, x, zp = PM.random_cmd(t.rng, n, t.aw, t.nbuf, mode=PM.M_MAX2, max_dim=8, max_cw=2)
    await t.host.host_write(cmd.src_sel, cmd.src_base, pack_act(x, n, zp))
    pf = pool_fields(OP_MAX2, cmd.h_in, cmd.w_in, cmd.c_words, cmd.src_sel, cmd.dst_sel,
                     cmd.src_base, cmd.dst_base)

    async def wiggle():
        r = random.Random(1)
        while True:
            await FallingEdge(d.clk)
            d.host_rd_en.value = r.randint(0, 1)
            d.host_rd_addr.value = r.randrange(t.host.depth)
            d.host_rd_sel.value = r.randrange(t.nbuf)

    w = cocotb.start_soon(wiggle())
    await t.pool(OP_MAX2, pf, "pool with host reads wiggling")
    w.cancel()
    t.host.idle_host()
    await t.host.check_all("after the wiggled pool")


@cocotb.test()
async def test_idle_write_port_closed(dut):
    """with no engine busy nothing may write except the host in host mode:
    host_we with host_mode low (ignored by contract), and the engines' own
    write ports, must leave every bank unchanged. (a mutation run showed no
    test drove host_we without host_mode)"""
    t = await fresh(dut)
    d = t.dut
    r = random.Random(t.seed)
    for _ in range(300):
        await FallingEdge(d.clk)
        d.host_mode.value = 0
        d.host_we.value = 1
        d.host_waddr.value = r.randrange(t.host.depth)
        d.host_wdata.value = r.getrandbits(8 * t.n)
        d.host_wr_sel.value = r.randrange(t.nbuf)
    t.host.idle_host()
    await t.host.check_all("after host_we without host_mode")
