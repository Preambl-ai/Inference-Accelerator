""" cocotb tb for rtl/requant.sv """

import itertools
import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer

from requant_model import RequantModel, requant_ref
from pe_model import signed_range


def read_int(sig, name):
    raw = str(sig.value)
    if "x" in raw.lower() or "z" in raw.lower():
        raise AssertionError(f"{name} is not resolvable: {raw}")
    return int(sig.value)


def read_signed(sig, name, width):
    raw = read_int(sig, name)
    return raw - (1 << width) if raw >= (1 << (width - 1)) else raw


def to_unsigned(value, width):
    return value & ((1 << width) - 1)


class RequantHarness:
    """Drives the DUT and the golden model from one input vector.

    Every cycle goes through step(), which compares the output port against
    the pipeline model. On top of that a scoreboard holds requant_ref() for
    every sample that was accepted, in order, and checks each output that
    emerges against it. requant_ref knows nothing about stages, so a mistake
    shared by the RTL and the pipeline model still fails here.
    """

    def __init__(self, dut):
        self.dut = dut
        self.aw = int(dut.ACC_WIDTH.value)
        self.bw = int(dut.BIAS_WIDTH.value)
        self.sw = int(dut.SAT_WIDTH.value)
        self.mw = int(dut.M_WIDTH.value)
        self.shw = int(dut.SHIFT_WIDTH.value)
        self.dw = int(dut.DATA_WIDTH.value)
        self.tw = int(dut.TAG_WIDTH.value)
        self.model = RequantModel(self.aw, self.bw, self.sw, self.mw, self.shw,
                                  self.dw, self.tw)
        self.acc_lo, self.acc_hi = signed_range(self.aw)
        self.bias_lo, self.bias_hi = signed_range(self.bw)
        self.sat_lo, self.sat_hi = signed_range(self.sw)
        self.q_lo, self.q_hi = signed_range(self.dw)
        self.m_max = (1 << self.mw) - 1
        self.shift_max = (1 << self.shw) - 1
        self.tag_max = (1 << self.tw) - 1
        self.cycle = 0
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = random.Random(self.seed)
        self.pending = []       # (expected q, tag, sample) per accepted sample
        self.outputs = []       # (q, tag) per emerged sample
        self.cover = dict(sat_hi=0, sat_lo=0, clamp_hi=0, clamp_lo=0,
                          relu_floor=0, in_range=0, tie=0)

    # lifecycle

    async def start(self, reset_cycles=3):
        cocotb.start_soon(Clock(self.dut.clk, 10, unit="ns").start())
        self.dut.reset.value = 1
        self.dut.en.value = 1
        self._drive(0, self.garbage(), 0)
        # reset stays high until the first step() drives it low, so the DUT
        # never sees an edge the model does not
        for _ in range(reset_cycles):
            await RisingEdge(self.dut.clk)
            self.model.step(reset=1)
        self.dut._log.info(
            f"DUT parameters: ACC_WIDTH={self.aw} BIAS_WIDTH={self.bw} "
            f"SAT_WIDTH={self.sw} M_WIDTH={self.mw} SHIFT_WIDTH={self.shw} "
            f"DATA_WIDTH={self.dw} TAG_WIDTH={self.tw} SEED={self.seed} "
            f"(reproduce with: make MODULE_NAME=requant ACC_WIDTH={self.aw} "
            f"BIAS_WIDTH={self.bw} SAT_WIDTH={self.sw} M_WIDTH={self.mw} "
            f"SHIFT_WIDTH={self.shw} DATA_WIDTH={self.dw} TAG_WIDTH={self.tw} "
            f"SEED={self.seed})"
        )

    # stimulus

    def sample(self, acc=0, bias=0, mult=1, shift=0, zp=0, relu=0):
        return dict(in_acc=acc, bias=bias, mult=mult, shift=shift,
                    zero_point=zp, relu_en=relu)

    def split(self, total):
        """(acc, bias) with acc + bias == total, or None if out of reach."""
        if not self.acc_lo + self.bias_lo <= total <= self.acc_hi + self.bias_hi:
            return None
        bias = min(max(total // 2, self.bias_lo), self.bias_hi)
        acc = min(max(total - bias, self.acc_lo), self.acc_hi)
        return acc, total - acc

    def garbage(self):
        """Nonzero values for cycles that carry no sample, so an invalid or
        stalled input that leaks into the pipeline cannot look like idle."""
        r = self.rng
        return dict(
            in_acc=r.choice((-1, 1)) * r.randint(1, self.acc_hi),
            bias=r.choice((-1, 1)) * r.randint(1, self.bias_hi),
            mult=r.randint(1, self.m_max),
            shift=r.randint(0, self.shift_max),
            zero_point=r.randint(self.q_lo, self.q_hi),
            relu_en=r.randint(0, 1),
        )

    def random_sample(self):
        r = self.rng
        shift = r.randint(0, self.shift_max)
        # a multiplier sized to the shift lands the result near the output
        # range; a fully random one almost always just clamps
        near = min(self.m_max, r.randint(0, 3) << max(shift - self.dw, 0))
        small = (1 << self.dw) * 4
        return dict(
            in_acc=r.choice((self.acc_lo, self.acc_hi, r.randint(self.acc_lo, self.acc_hi),
                             r.randint(-small, small), r.randint(-small, small))),
            bias=r.choice((self.bias_lo, self.bias_hi, r.randint(self.bias_lo, self.bias_hi),
                           r.randint(-small, small), r.randint(-small, small))),
            mult=r.choice((r.randint(0, self.m_max), near, near)),
            shift=shift,
            zero_point=r.randint(self.q_lo, self.q_hi),
            relu_en=r.randint(0, 1),
        )

    def _drive(self, in_valid, s, tag):
        d = self.dut
        d.in_valid.value = in_valid
        d.in_acc.value = to_unsigned(s["in_acc"], self.aw)
        d.in_tag.value = tag
        d.bias.value = to_unsigned(s["bias"], self.bw)
        d.mult.value = s["mult"]
        d.shift.value = s["shift"]
        d.zero_point.value = to_unsigned(s["zero_point"], self.dw)
        d.relu_en.value = s["relu_en"]

    def _classify(self, s, q):
        total = s["in_acc"] + s["bias"]
        sat = min(max(total, self.sat_lo), self.sat_hi)
        self.cover["sat_hi"] += int(total > self.sat_hi)
        self.cover["sat_lo"] += int(total < self.sat_lo)
        prod = sat * s["mult"]
        sh = s["shift"]
        self.cover["tie"] += int(sh > 0 and prod % (1 << sh) == (1 << (sh - 1)))
        pre = ((prod + ((1 << sh) >> 1)) >> sh) + s["zero_point"]
        self.cover["clamp_hi"] += int(pre > self.q_hi)
        self.cover["clamp_lo"] += int(pre < self.q_lo and not s["relu_en"])
        self.cover["relu_floor"] += int(bool(s["relu_en"]) and pre < s["zero_point"])
        self.cover["in_range"] += int(self.q_lo < q < self.q_hi and q != s["zero_point"])

    async def step(self, s=None, tag=None, en=1, reset=0):
        """Advance one rising edge on both DUT and model, then compare.

        s is a sample dict to push with in_valid = 1; None pushes garbage with
        in_valid = 0. Returns the model's view of the output port.
        """
        in_valid = int(s is not None)
        if s is None:
            s = self.garbage()
        if tag is None:
            tag = self.rng.randint(0, self.tag_max)

        d = self.dut
        await FallingEdge(d.clk)
        self._drive(in_valid, s, tag)
        d.en.value = en
        d.reset.value = reset

        await RisingEdge(d.clk)
        await ReadOnly()

        expected = self.model.step(in_valid=in_valid, in_tag=tag, en=en, reset=reset, **s)
        got = (read_int(d.out_valid, "out_valid"),
               read_signed(d.out_q, "out_q", self.dw),
               read_int(d.out_tag, "out_tag"))
        assert got == tuple(expected), (
            f"cycle {self.cycle}: (out_valid, out_q, out_tag)={got}, "
            f"expected {tuple(expected)}\n"
            f"stimulus: in_valid={in_valid} tag={tag} en={en} reset={reset} {s}\n"
            f"seed={self.seed}"
        )
        # whitebox: out_q only shows DATA_WIDTH bits of each stage, so an
        # off-by-one in a saturation bound or a dropped product bit can hide
        # behind the output clamp. compare the stage registers directly.
        m = self.model
        p_w = m.p_w
        for name, width, exp in (("s1_sum", self.sw, m.s1["sum"]),
                                 ("s2_prod", p_w, m.s2["prod"]),
                                 ("s3_scaled", p_w, m.s3["scaled"])):
            reg = read_signed(getattr(d, name), name, width)
            assert reg == exp, (
                f"cycle {self.cycle}: {name}={reg}, expected {exp}\n"
                f"stimulus: in_valid={in_valid} en={en} reset={reset} {s}\n"
                f"seed={self.seed}"
            )

        if reset:
            self.pending = []
        elif en:
            if got[0]:
                assert self.pending, f"an output emerged that nobody sent, seed={self.seed}"
                exp_q, exp_tag, src = self.pending.pop(0)
                assert (got[1], got[2]) == (exp_q, exp_tag), (
                    f"cycle {self.cycle}: out_q={got[1]} out_tag={got[2]}, reference "
                    f"says q={exp_q} tag={exp_tag}\nsample: {src}\nseed={self.seed}"
                )
                self.outputs.append((got[1], got[2]))
            if in_valid:
                q = requant_ref(s["in_acc"], s["bias"], s["mult"], s["shift"],
                                s["zero_point"], s["relu_en"], self.sw, self.dw)
                self.pending.append((q, tag, s))
                self._classify(s, q)
        self.cycle += 1
        return expected

    async def flush(self):
        """Run garbage through until everything accepted has emerged."""
        for _ in range(self.model.LATENCY + 2):
            await self.step()
        assert not self.pending, (
            f"{len(self.pending)} samples never emerged, seed={self.seed}"
        )

    async def run_vectors(self, vectors):
        """Push the samples back to back and return their out_q values."""
        self.outputs = []
        for s in vectors:
            await self.step(s)
        await self.flush()
        assert len(self.outputs) == len(vectors)
        return [q for q, _ in self.outputs]


# TESTS


@cocotb.test()
async def test_latency_from_sim(dut):
    """Measure LATENCY on the DUT: one sample, counted in enabled edges."""
    h = RequantHarness(dut)
    await h.start()
    for _ in range(3):
        await h.step()

    await h.step(h.sample(acc=5, mult=1, zp=1), tag=h.tag_max)
    edges = 1
    while not read_int(dut.out_valid, "out_valid"):
        await h.step()
        edges += 1
        assert edges < 20, "the sample never emerged"
    assert edges == h.model.LATENCY == 4, (
        f"measured latency {edges}, model says {h.model.LATENCY}"
    )
    assert read_signed(dut.out_q, "out_q", h.dw) == 6
    assert read_int(dut.out_tag, "out_tag") == h.tag_max
    await h.step()
    assert read_int(dut.out_valid, "out_valid") == 0, "out_valid lasted more than one edge"
    dut._log.info(f"LATENCY measured at {edges} enabled edges")


@cocotb.test()
async def test_bias_saturation_both_ways(dut):
    """acc + bias beyond SAT_WIDTH pins at the bound instead of wrapping.

    Where SAT_WIDTH is the full sum width nothing can saturate, and the same
    vectors then prove the sum is carried without loss.
    """
    h = RequantHarness(dut)
    await h.start()
    # scale the saturated sum down to about a quarter of the output range
    shift = min(h.shift_max, max(0, h.sw - h.dw + 1))
    vectors = [
        h.sample(acc=h.acc_hi, bias=h.bias_hi, shift=shift),
        h.sample(acc=h.acc_lo, bias=h.bias_lo, shift=shift),
        h.sample(acc=h.acc_hi, bias=h.bias_hi, shift=shift, mult=3),
        h.sample(acc=h.acc_lo, bias=h.bias_lo, shift=shift, mult=3),
    ]
    # one LSB either side of each bound, wherever acc + bias can get there.
    # the output cannot show that LSB; the s1_sum compare in step() does.
    for bound in (h.sat_hi, h.sat_lo):
        for delta in (-1, 0, 1):
            parts = h.split(bound + delta)
            if parts:
                vectors.append(h.sample(acc=parts[0], bias=parts[1], shift=shift))
    got = await h.run_vectors(vectors)

    can_saturate = h.acc_hi + h.bias_hi > h.sat_hi
    if can_saturate:
        assert h.cover["sat_hi"] >= 2 and h.cover["sat_lo"] >= 2, h.cover
        # sat(max) = 2^(SAT-1) - 1, shifted right by SAT-1 with rounding = 1
        if shift == h.sw - h.dw + 1:
            quarter = 1 << (h.dw - 2)
            assert got[:2] == [quarter, -quarter], f"saturated results {got[:2]}"
    else:
        assert h.cover["sat_hi"] == 0 and h.cover["sat_lo"] == 0
    dut._log.info(f"saturation {'exercised' if can_saturate else 'unreachable at these widths'}: "
                  f"{h.cover['sat_hi']} high, {h.cover['sat_lo']} low")


@cocotb.test()
async def test_shift_zero_and_max(dut):
    """shift = 0 passes the product through, the largest shift still rounds."""
    h = RequantHarness(dut)
    await h.start()
    smax = h.shift_max
    vectors = [h.sample(acc=a, mult=m, shift=0, zp=z)
               for a in (-3, -1, 0, 1, 3, h.q_hi, h.q_lo)
               for m in (0, 1, 2) for z in (0, -1, 1)]
    got = await h.run_vectors(vectors)
    for s, q in zip(vectors, got):
        raw = s["in_acc"] * s["mult"] + s["zero_point"]
        assert q == min(max(raw, h.q_lo), h.q_hi), f"shift=0: {s} gave {q}"

    # at the largest shift the product has to straddle half an output LSB,
    # 2^(smax-1). take the largest sums and the multipliers either side of it.
    half = 1 << (smax - 1)
    vectors = []
    # the largest sums acc + bias can actually reach inside the saturation range
    top = min(h.sat_hi, h.acc_hi + h.bias_hi)
    bottom = max(h.sat_lo, h.acc_lo + h.bias_lo)
    for total in (top, top - 1, bottom, bottom + 1):
        acc, bias = h.split(total)
        base = half // abs(total)
        for mult in (base - 1, base, base + 1, base + 2, 3 * base, 3 * base + 3, h.m_max):
            if 0 <= mult <= h.m_max:
                vectors.append(h.sample(acc=acc, bias=bias, mult=mult, shift=smax))
    got = await h.run_vectors(vectors)
    # the straddle has to produce results on both sides of each boundary
    assert {-1, 0, 1} <= set(got), (
        f"max-shift vectors only produced {sorted(set(got))}; they do not "
        f"straddle the rounding point at these widths"
    )
    dut._log.info(f"shift=0 and shift={smax} passed")


@cocotb.test()
async def test_round_half_ties(dut):
    """Exact .5 ties round toward +infinity for positive and negative values."""
    h = RequantHarness(dut)
    await h.start()
    for shift in sorted({1, 2, 3, min(7, h.shift_max), h.shift_max}):
        if shift == 0:
            continue
        half = 1 << (shift - 1)
        if half > h.m_max:
            continue
        odd = [n for n in (-7, -5, -3, -1, 1, 3, 5, 7)]
        vectors = [h.sample(acc=n, mult=half, shift=shift) for n in odd]
        before = h.cover["tie"]
        got = await h.run_vectors(vectors)
        assert h.cover["tie"] - before == len(odd), "vectors were not ties"
        for n, q in zip(odd, got):
            # n/2 with n odd: half up is (n + 1) / 2 for both signs
            assert q == (n + 1) // 2, (
                f"shift={shift}: {n}/2 rounded to {q}, expected {(n + 1) // 2}"
            )
    dut._log.info(f"{h.cover['tie']} ties rounded half up")


@cocotb.test()
async def test_clamp_and_relu(dut):
    """Output clamp both ways, and ReLU on/off for the same sample."""
    h = RequantHarness(dut)
    await h.start()
    big = 4 << h.dw
    zps = (h.q_lo, -3, 0, 5, h.q_hi)
    vectors = [h.sample(acc=a, zp=z, relu=r)
               for a in (-big, h.q_lo - 1, h.q_lo, -2, -1, 0, 1, 2, h.q_hi, h.q_hi + 1, big)
               for z in zps for r in (0, 1)]
    got = await h.run_vectors(vectors)
    for s, q in zip(vectors, got):
        raw = s["in_acc"] + s["zero_point"]
        floor = s["zero_point"] if s["relu_en"] else h.q_lo
        assert q == min(max(raw, floor), h.q_hi), f"{s} gave {q}"
    for name in ("clamp_hi", "clamp_lo", "relu_floor"):
        assert h.cover[name] > 0, f"{name} never exercised: {h.cover}"
    dut._log.info(f"clamp and ReLU passed: {h.cover}")


@cocotb.test()
async def test_corner_cross(dut):
    """Every combination of the extreme value of every input."""
    h = RequantHarness(dut)
    await h.start()
    vectors = [h.sample(acc=a, bias=b, mult=m, shift=sh, zp=z, relu=r)
               for a, b, m, sh, z, r in itertools.product(
                   (h.acc_lo, -1, 0, 1, h.acc_hi), (h.bias_lo, 0, h.bias_hi),
                   (0, 1, h.m_max), (0, 1, h.shift_max), (h.q_lo, 0, h.q_hi), (0, 1))]
    await h.run_vectors(vectors)
    dut._log.info(f"{len(vectors)} corner combinations passed")


@cocotb.test()
async def test_random_stream(dut):
    """Constrained-random samples with gaps; parameters change every sample."""
    h = RequantHarness(dut)
    await h.start()
    for _ in range(6000):
        await h.step(h.random_sample() if h.rng.random() < 0.7 else None)
    await h.flush()
    # a random run that only ever clamps would prove very little
    for name in ("clamp_hi", "clamp_lo", "relu_floor", "in_range"):
        assert h.cover[name] > 50, f"{name} barely exercised: {h.cover}, seed={h.seed}"
    if h.acc_hi + h.bias_hi > h.sat_hi:
        assert h.cover["sat_hi"] > 50 and h.cover["sat_lo"] > 50, h.cover
    dut._log.info(f"random stream passed: {len(h.outputs)} outputs, coverage {h.cover}")


@cocotb.test()
async def test_tag_alignment(dut):
    """Tags stay with their data when neighbours differ in every parameter."""
    h = RequantHarness(dut)
    await h.start()
    sent = []
    tag = 1
    for i in range(200):
        if h.rng.random() < 0.75:
            s = h.random_sample()
            await h.step(s, tag=tag)
            sent.append(tag)
            tag = (tag + 1) & h.tag_max
        else:
            # a gap carries a tag too; it must not be the one that comes out
            await h.step(None, tag=(tag + 1) & h.tag_max)
    await h.flush()
    assert [t for _, t in h.outputs] == sent, f"tags reordered or dropped, seed={h.seed}"


@cocotb.test()
async def test_stall_mid_pipeline(dut):
    """en low holds all four stages, and nothing driven meanwhile gets in."""
    h = RequantHarness(dut)
    await h.start()
    vectors = [h.random_sample() for _ in range(h.model.LATENCY)]
    for depth in range(1, h.model.LATENCY + 1):
        # depth samples in flight, then a stall with valid garbage on the pins
        for s in vectors[:depth]:
            await h.step(s)
        held = (read_int(dut.out_valid, "out_valid"),
                read_signed(dut.out_q, "out_q", h.dw),
                read_int(dut.out_tag, "out_tag"))
        for _ in range(3):
            await h.step(h.random_sample(), en=0)
            now = (read_int(dut.out_valid, "out_valid"),
                   read_signed(dut.out_q, "out_q", h.dw),
                   read_int(dut.out_tag, "out_tag"))
            assert now == held, f"output moved under stall: {held} -> {now}"
        await h.flush()

    # heavy random stall over a stream
    h.outputs = []
    sent = 0
    for _ in range(3000):
        en = int(h.rng.random() < 0.6)
        s = h.random_sample() if h.rng.random() < 0.7 else None
        await h.step(s, en=en)
        sent += int(en and s is not None)
    await h.flush()
    assert len(h.outputs) == sent, f"sent {sent}, got {len(h.outputs)}, seed={h.seed}"
    dut._log.info(f"stall mid-pipeline passed, {sent} samples through 40% stall")


@cocotb.test()
async def test_reset_mid_pipeline(dut):
    """Reset drops everything in flight, wins over en, and leaves it usable."""
    h = RequantHarness(dut)
    await h.start()
    for en in (1, 0):
        for depth in range(1, h.model.LATENCY + 1):
            for _ in range(depth):
                await h.step(h.random_sample())
            await h.step(h.random_sample(), en=en, reset=1)
            assert read_int(dut.out_valid, "out_valid") == 0
            assert read_signed(dut.out_q, "out_q", h.dw) == 0
            for _ in range(h.model.LATENCY + 2):
                await h.step()
                assert read_int(dut.out_valid, "out_valid") == 0, (
                    f"a sample survived reset (depth={depth}, en={en})"
                )
            got = await h.run_vectors([h.sample(acc=7, mult=3, shift=1, zp=-2)])
            assert got == [9], f"wrong result after reset: {got}"   # 10.5 -> 11, - 2


@cocotb.test()
async def test_no_combinational_path(dut):
    """No input may reach the output port within the same cycle."""
    h = RequantHarness(dut)
    await h.start()
    for _ in range(h.model.LATENCY + 1):
        await h.step()
    await FallingEdge(dut.clk)
    before = (read_int(dut.out_valid, "out_valid"), read_int(dut.out_q, "out_q"),
              read_int(dut.out_tag, "out_tag"))
    h._drive(1, h.sample(acc=h.q_hi, zp=0), h.tag_max)
    await Timer(1, unit="ns")
    await ReadOnly()
    after = (read_int(dut.out_valid, "out_valid"), read_int(dut.out_q, "out_q"),
             read_int(dut.out_tag, "out_tag"))
    assert after == before, "an input reached the output combinationally"
    # the harness did not see that drive, so resynchronise through reset
    await h.step(reset=1)
