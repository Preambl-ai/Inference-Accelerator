""" cocotb tb for add/rtl/add_layer.sv through add/tb/add_harness.sv
(add_layer + the real act_buffer + host access).

Expected words come from add_model (NumPy on every lane of the packed words).
Whole banks are compared with a mirror, and every start-to-done count is
compared with the cycle formula.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
import numpy as np

from act_host import ActHost
from add_model import (AddCmd, E_DIM, E_DST, E_HOST, E_SEL, E_SRC, add_words, check_command,
                       err_names, estimate_cycles, random_cmd)
from conv_layer_model import pack_act, words


class AddTB:

    def __init__(self, dut):
        self.dut = dut
        g = lambda name: int(getattr(dut, name).value)
        self.n, self.lat, self.aw, self.nbuf = g("N"), g("READ_LATENCY"), g("ADDR_WIDTH"), g("NBUF")
        self.m_bits = g("M_BITS")
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = np.random.default_rng(self.seed)
        self.host = ActHost(dut, self.n, self.lat, self.aw, self.nbuf, self.seed, self.where)

    def where(self):
        return f"(reproduce: make MODULE_NAME=add_harness N={self.n} READ_LATENCY={self.lat} SEED={self.seed})"

    async def start(self):
        d = self.dut
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        self.host.idle_host()
        d.start.value = 0
        d.reset.value = 1
        for _ in range(3):
            await RisingEdge(d.clk)
        await FallingEdge(d.clk)
        d.reset.value = 0
        await self.host.fill_all()
        d._log.info(f"N={self.n} READ_LATENCY={self.lat} NBUF={self.nbuf} M_BITS={self.m_bits} SEED={self.seed}")

    async def load(self, cmd, a, b):
        await self.host.host_write(cmd.src_sel, cmd.src_base, pack_act(a, self.n, cmd.za))
        await self.host.host_write(cmd.src2_sel, cmd.src2_base, pack_act(b, self.n, cmd.zb))

    def expected(self, cmd):
        nw = cmd.n_words
        aw = self.host.mem[cmd.src_sel][cmd.src_base:cmd.src_base + nw]
        bw = self.host.mem[cmd.src2_sel][cmd.src2_base:cmd.src2_base + nw]
        return add_words(aw, bw, cmd, self.n)

    async def run_add(self, cmd, name, check=True):
        assert check_command(cmd, self.aw, self.nbuf) == 0, f"stimulus bug: {err_names(check_command(cmd, self.aw, self.nbuf))}"
        exp = self.expected(cmd)
        res = await self.host.run(cmd.as_dict())
        assert res["kind"] == "done", f"{name}: rejected {err_names(res['err_code'])}"
        self.host.expect(cmd.dst_sel, cmd.dst_base, exp)
        if check:
            await self.host.check_all(f"after {name}")
        est = estimate_cycles(cmd, self.lat, self.m_bits)
        assert res["cycles"] == est, f"{name}: {res['cycles']} edges, formula {est} {self.where()}"
        self.dut._log.info(f"CYCLES {name}: {cmd.n_words} words measured={res['cycles']} formula={est}")
        return res


async def fresh(dut):
    t = AddTB(dut)
    await t.start()
    return t


@cocotb.test()
async def test_directed_arithmetic(dut):
    """identity scales, zero points that cancel, ties both ways, both rails,
    ReLU, ma = mb = 0, the largest scales, and a and b in the same bank"""
    t = await fresh(dut)
    n = t.n
    top = (1 << t.m_bits) - 1
    cases = [
        ("identity", dict(za=0, zb=0, zo=0, ma=1, mb=1, shift=0, relu_en=0)),
        ("zero points", dict(za=10, zb=-20, zo=5, ma=1, mb=1, shift=0, relu_en=0)),
        ("halves, ties", dict(za=0, zb=0, zo=0, ma=1, mb=1, shift=1, relu_en=0)),
        ("largest scales saturate", dict(za=-128, zb=-128, zo=0, ma=top, mb=top, shift=0, relu_en=0)),
        ("relu", dict(za=3, zb=-3, zo=-20, ma=100, mb=200, shift=7, relu_en=1)),
        ("ma = mb = 0", dict(za=0, zb=0, zo=-9, ma=0, mb=0, shift=4, relu_en=0)),
        ("shift 15", dict(za=-1, zb=2, zo=1, ma=top, mb=top, shift=15, relu_en=0)),
    ]
    for i, (name, p) in enumerate(cases):
        same = i % 2 == 0 or t.nbuf < 3
        cmd = AddCmd(3, 3, 2, 0, 0, 0 if same else 2, 100, 1, 0, **p)
        c = 2 * n - (1 if n > 1 else 0)
        a = t.rng.integers(-128, 128, (3, 3, c))
        b = t.rng.integers(-128, 128, (3, 3, c))
        if name == "halves, ties":                  # odd sums of every sign
            a = np.full((3, 3, c), 3)
            b = np.where(np.arange(c) % 2 == 0, 4, -10) * np.ones((3, 3, c), int)
        await t.load(cmd, a, b)
        await t.run_add(cmd, name)


@cocotb.test()
async def test_random_commands(dut):
    """constrained-random legal adds"""
    t = await fresh(dut)
    for i in range(int(os.environ.get("LAYERS") or 8)):
        cmd, a, b = random_cmd(t.rng, t.n, t.aw, t.nbuf, max_dim=4, max_cw=2, m_bits=t.m_bits)
        await t.load(cmd, a, b)
        await t.run_add(cmd, f"random #{i}", check=(i % 2 == 0))


@cocotb.test()
async def test_chained(dut):
    """the output of one add is an operand of the next (a residual chain)"""
    t = await fresh(dut)
    n = t.n
    x = t.rng.integers(-128, 128, (2, 4, n))
    y = t.rng.integers(-128, 128, (2, 4, n))
    if t.nbuf >= 3:
        # x in bank 0, y in bank 1, s = x + y in bank 2, then s + x into bank 1
        c1 = AddCmd(2, 4, 1, 0, 0, 1, 0, 2, 0, za=0, zb=0, zo=0, ma=128, mb=128, shift=8)
        c2 = AddCmd(2, 4, 1, 2, 0, 0, 0, 1, 50, za=0, zb=0, zo=3, ma=200, mb=60, shift=8, relu_en=1)
        name2 = "(x + y) + x, relu"
    else:
        # two banks: x and y share bank 0, s = x + y in bank 1, then s + s
        # back into bank 0 (x as the second operand would leave no free bank)
        c1 = AddCmd(2, 4, 1, 0, 0, 0, 100, 1, 0, za=0, zb=0, zo=0, ma=128, mb=128, shift=8)
        c2 = AddCmd(2, 4, 1, 1, 0, 1, 0, 0, 300, za=0, zb=0, zo=3, ma=200, mb=60, shift=8, relu_en=1)
        name2 = "(x + y) + (x + y), relu"
    await t.load(c1, x, y)
    await t.run_add(c1, "x + y")
    await t.run_add(c2, name2)


@cocotb.test()
async def test_reset_mid_layer(dut):
    """reset at several points: ready after, each destination word old or
    final, then a clean rerun"""
    t = await fresh(dut)
    n = t.n
    cmd = AddCmd(2, 3, 2, 0, 0, 2 if t.nbuf >= 3 else 0, 100, 1, 5, za=1, zb=-2, zo=3, ma=77,
                 mb=150, shift=6)
    a = t.rng.integers(-128, 128, (2, 3, 2 * n))
    b = t.rng.integers(-128, 128, (2, 3, 2 * n))
    await t.load(cmd, a, b)
    final = t.expected(cmd)
    full = estimate_cycles(cmd, t.lat, t.m_bits)
    for at in (3, 17, full // 2, full - 1, full):
        await t.host.fill_bank(1)
        old = list(t.host.mem[1])
        res = await t.host.run(cmd.as_dict(), reset_at=at)
        assert res["kind"] == "reset"
        got = await t.host.read_words(1, 0, t.host.depth)
        written = 0
        for adr in range(t.host.depth):
            k = adr - cmd.dst_base
            if 0 <= k < len(final) and got[adr] == final[k] and final[k] != old[adr]:
                written += 1
            else:
                assert got[adr] == old[adr], f"reset at {at}: word {adr} neither old nor final"
        if at == full:
            # every word but the last was written; at N=1 a final value can
            # equal the junk already there (1 in 256), so count only the
            # words where final differs from old
            want = sum(final[k] != old[cmd.dst_base + k] for k in range(len(final) - 1))
            assert written == want, f"reset on the last edge: {written} words written, expected {want}"
        await t.run_add(cmd, f"rerun after reset at {at}")


@cocotb.test()
async def test_illegal_commands(dut):
    """every check: err once with exactly the predicted bits, nothing written"""
    t = await fresh(dut)
    depth = t.host.depth
    base = AddCmd(2, 2, 1, 0, 0, 0, 100, 1, 0)
    cases = [("E_DIM", dict(h_in=0), 0), ("E_SRC a", dict(src_base=depth - 3), 0),
             ("E_SRC b", dict(src2_base=depth - 3), 0), ("E_DST", dict(dst_base=depth - 3), 0),
             ("E_SEL dst == a", dict(dst_sel=0), 0), ("E_HOST", dict(), 1)]
    if t.nbuf >= 3:
        cases.append(("E_SEL dst == b", dict(src2_sel=1), 0))
    if (1 << max(1, (t.nbuf - 1).bit_length())) > t.nbuf:
        cases.append(("E_SEL bank >= NBUF", dict(dst_sel=t.nbuf), 0))
    lens = set()
    for name, over, host in cases:
        cmd = AddCmd(**{**base.__dict__, **over})
        exp = check_command(cmd, t.aw, t.nbuf, host_mode=host)
        assert exp, f"stimulus: {name} legal"
        res = await t.host.run(cmd.as_dict(), host_mode=host)
        assert res["kind"] == "err" and res["err_code"] == exp, \
            f"{name}: {err_names(res['err_code'] or 0)} expected {err_names(exp)}"
        lens.add(res["cycles"])
    assert len(lens) == 1, f"check phase length varies {lens}"
    dut._log.info(f"check phase: err {lens.pop()} edges after start")
    await t.host.check_all("after the rejected commands")
