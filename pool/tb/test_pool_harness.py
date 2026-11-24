""" cocotb tb for pool/rtl/pool_layer.sv, through pool/tb/pool_harness.sv
(pool_layer + the real act_buffer + host access).

Expected words come from pool_model (NumPy on the unpacked tensor, every lane
including the padded ones). Whole banks are compared against a mirror, so
stray writes anywhere are caught, and every start-to-done count is compared
with the cycle formula.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
import numpy as np

from act_host import ActHost
from conv_layer_model import pack_act, words
from pool_model import (E_DIM, E_HOST, E_MODE, E_SEL, E_SHAPE, E_SRC, E_DST, M_GAVG, M_MAX2,
                        PoolCmd, check_command, err_names, estimate_cycles, pool_words, random_cmd)


class PoolTB:

    def __init__(self, dut):
        self.dut = dut
        g = lambda name: int(getattr(dut, name).value)
        self.n, self.lat, self.aw, self.nbuf = g("N"), g("READ_LATENCY"), g("ADDR_WIDTH"), g("NBUF")
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = np.random.default_rng(self.seed)
        self.host = ActHost(dut, self.n, self.lat, self.aw, self.nbuf, self.seed, self.where)
        self.cycles = []

    def where(self):
        return f"(reproduce: make MODULE_NAME=pool_harness N={self.n} READ_LATENCY={self.lat} SEED={self.seed})"

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
        d._log.info(f"N={self.n} READ_LATENCY={self.lat} ADDR_WIDTH={self.aw} NBUF={self.nbuf} SEED={self.seed}")

    async def load(self, cmd, x, zp):
        await self.host.host_write(cmd.src_sel, cmd.src_base, pack_act(x, self.n, zp))

    def expected(self, cmd):
        n_in = cmd.h_in * cmd.w_in * cmd.c_words
        src = self.host.mem[cmd.src_sel][cmd.src_base:cmd.src_base + n_in]
        return pool_words(src, cmd.h_in, cmd.w_in, cmd.c_words, self.n, cmd.mode)

    async def run_pool(self, cmd, name, check=True):
        assert check_command(cmd, self.aw, self.nbuf) == 0, f"stimulus bug: {err_names(check_command(cmd, self.aw, self.nbuf))}"
        exp = self.expected(cmd)
        res = await self.host.run(cmd.as_dict())
        assert res["kind"] == "done", f"{name}: rejected {err_names(res['err_code'])}"
        self.host.expect(cmd.dst_sel, cmd.dst_base, exp)
        if check:
            for sel in sorted({cmd.src_sel, cmd.dst_sel}):
                await self.host.check_bank(sel, f"after {name}")
        est = estimate_cycles(cmd, self.lat)
        self.cycles.append((name, res["cycles"], est))
        assert res["cycles"] == est, f"{name}: {res['cycles']} edges, formula {est} {self.where()}"
        self.dut._log.info(f"CYCLES {name}: mode={cmd.mode} {cmd.h_in}x{cmd.w_in} cw={cmd.c_words} "
                           f"measured={res['cycles']} formula={est}")
        return res


async def fresh(dut):
    t = PoolTB(dut)
    await t.start()
    return t


@cocotb.test()
async def test_max2_even_and_odd(dut):
    """2x2 max on even sizes and on odd sizes (the last row/column dropped)"""
    t = await fresh(dut)
    for h, w, c in ((2, 2, 1), (4, 6, t.n), (5, 7, 2 * t.n + 1), (3, 2, 3)):
        x = t.rng.integers(-128, 128, (h, w, c))
        zp = int(t.rng.integers(-128, 128))
        cmd = PoolCmd(M_MAX2, h, w, words(c, t.n), 0, 1, 3, 5)
        await t.load(cmd, x, zp)
        await t.run_pool(cmd, f"max2 {h}x{w}x{c}")


@cocotb.test()
async def test_gavg_sizes(dut):
    """global average over 1, 2, 4 ... pixels, square and not"""
    t = await fresh(dut)
    for h, w in ((1, 1), (1, 2), (2, 1), (2, 2), (4, 4), (2, 8), (8, 8), (1, 32)):
        c = int(t.rng.integers(1, 2 * t.n + 1))
        if h * w * words(c, t.n) > t.host.depth:
            continue
        x = t.rng.integers(-128, 128, (h, w, c))
        cmd = PoolCmd(M_GAVG, h, w, words(c, t.n), 1, 0, 0, 7)
        await t.load(cmd, x, int(t.rng.integers(-128, 128)))
        await t.run_pool(cmd, f"gavg {h}x{w}x{c}")


@cocotb.test()
async def test_signed_extremes_and_ties(dut):
    """all -128, all 127, alternating rails, and GAVG ties in both directions
    (-1.5 must round to -1, +1.5 to +2). destination banks hold junk, so an
    answer equal to a reset value cannot pass by accident"""
    t = await fresh(dut)
    n = t.n
    cases = []
    for v in (-128, 127):
        cases.append(("max2 all %d" % v, M_MAX2, np.full((4, 4, n), v)))
        cases.append(("gavg all %d" % v, M_GAVG, np.full((4, 4, n), v)))
    alt = np.where((np.indices((4, 4, n)).sum(axis=0) % 2) == 0, -128, 127)
    cases.append(("max2 alternating", M_MAX2, alt))
    cases.append(("gavg alternating", M_GAVG, alt))
    # ties: two pixels per channel summing to an odd number
    ties = np.array([[[-1 if c % 2 else 1 for c in range(n)], [-2 if c % 2 else 2 for c in range(n)]]])
    cases.append(("gavg ties", M_GAVG, ties))
    for name, mode, x in cases:
        h, w, c = x.shape
        cmd = PoolCmd(mode, h, w, words(c, n), 0, 1, 0, 0)
        await t.load(cmd, x, 0)
        await t.run_pool(cmd, name)


@cocotb.test()
async def test_random_commands(dut):
    """constrained-random legal commands: modes, sizes, channel counts, zero
    points, banks and bases"""
    t = await fresh(dut)
    for i in range(int(os.environ.get("LAYERS") or 10)):
        cmd, x, zp = random_cmd(t.rng, t.n, t.aw, t.nbuf, max_dim=8, max_cw=3)
        await t.load(cmd, x, zp)
        await t.run_pool(cmd, f"random #{i}", check=(i % 3 == 0))
    await t.host.check_all("after the random commands")


@cocotb.test()
async def test_back_to_back(dut):
    """commands one after another with no reload in between, alternating
    modes: the output of one pool is the input of the next"""
    t = await fresh(dut)
    n = t.n
    x = t.rng.integers(-128, 128, (8, 8, n))
    await t.load(PoolCmd(M_MAX2, 8, 8, 1, 0, 1, 0, 0), x, 0)
    chain = [PoolCmd(M_MAX2, 8, 8, 1, 0, 1, 0, 0), PoolCmd(M_MAX2, 4, 4, 1, 1, 0, 0, 100),
             PoolCmd(M_GAVG, 2, 2, 1, 0, 1, 100, 200)]
    for i, cmd in enumerate(chain):
        await t.run_pool(cmd, f"chain {i}", check=False)
    await t.host.check_all("after the chain")


@cocotb.test()
async def test_reset_mid_layer(dut):
    """reset at several points: ready right after, every destination word old
    or final, nothing outside the region changed, then a clean rerun"""
    t = await fresh(dut)
    n = t.n
    for mode, shape in ((M_MAX2, (6, 6, 2 * n)), (M_GAVG, (4, 4, 2 * n))):
        x = t.rng.integers(-128, 128, shape)
        cmd = PoolCmd(mode, shape[0], shape[1], 2, 0, 1, 0, 3)
        await t.load(cmd, x, 0)
        final = t.expected(cmd)
        full = estimate_cycles(cmd, t.lat)
        for at in (1, 5, full // 2, full - 1, full):
            await t.host.fill_bank(1)
            old = list(t.host.mem[1])
            res = await t.host.run(cmd.as_dict(), reset_at=at)
            assert res["kind"] == "reset"
            got = await t.host.read_words(1, 0, t.host.depth)
            written = 0
            for a in range(t.host.depth):
                k = a - cmd.dst_base
                if 0 <= k < len(final) and got[a] == final[k] and final[k] != old[a]:
                    written += 1
                else:
                    assert got[a] == old[a], f"mode {mode} reset at {at}: word {a} neither old nor final"
            if at == full:
                # every word but the last was written. at N=1 a final value
                # equals the junk already there 1 time in 256, so count only
                # words where final differs from old (the add TB hit this)
                want = sum(final[k] != old[cmd.dst_base + k] for k in range(len(final) - 1))
                assert written == want, f"reset on the last edge: {written} written, expected {want}"
            await t.run_pool(cmd, f"rerun mode {mode} after reset at {at}")


@cocotb.test()
async def test_illegal_commands(dut):
    """every check: err once with exactly the predicted bits, nothing written,
    and a legal command still runs after"""
    t = await fresh(dut)
    base = PoolCmd(M_MAX2, 4, 4, 1, 0, 1, 0, 0)
    depth = t.host.depth
    cases = [("E_DIM", dict(c_words=0), 0), ("E_MODE 0", dict(mode=0), 0), ("E_MODE 3", dict(mode=3), 0),
             ("E_SHAPE max2 h=1", dict(h_in=1), 0), ("E_SHAPE gavg 3x4", dict(mode=M_GAVG, h_in=3), 0),
             ("E_SRC", dict(src_base=depth - 15), 0), ("E_DST", dict(dst_base=depth - 3), 0),
             ("E_DST gavg", dict(mode=M_GAVG, c_words=2, dst_base=depth - 1), 0),
             ("E_SEL same bank", dict(dst_sel=0), 0), ("E_HOST", dict(), 1)]
    if (1 << max(1, (t.nbuf - 1).bit_length())) > t.nbuf:
        cases.append(("E_SEL bank >= NBUF", dict(src_sel=t.nbuf), 0))
    check_len = set()
    for name, over, host in cases:
        cmd = PoolCmd(**{**base.__dict__, **over})
        cmd.dst_base %= depth
        exp = check_command(cmd, t.aw, t.nbuf, host_mode=host)
        assert exp, f"stimulus: {name} is legal"
        res = await t.host.run(cmd.as_dict(), host_mode=host)
        assert res["kind"] == "err" and res["err_code"] == exp, \
            f"{name}: {err_names(res['err_code'] or 0)} expected {err_names(exp)}"
        check_len.add(res["cycles"])
    assert len(check_len) == 1, f"check phase length varies: {check_len}"
    dut._log.info(f"check phase: err {check_len.pop()} edges after start")
    await t.host.check_all("after the rejected commands")
    x = t.rng.integers(-128, 128, (4, 4, t.n))
    await t.load(base, x, 0)
    await t.run_pool(base, "legal after the rejects")
