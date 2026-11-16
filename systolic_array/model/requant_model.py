# Golden model for rtl/requant.sv

from typing import NamedTuple

from pe_model import signed_range


class RequantState(NamedTuple):
    """What is visible on the output port at the ReadOnly point after an edge."""
    out_valid: int
    out_q: int
    out_tag: int


def clamp(value: int, lo: int, hi: int) -> int:
    return lo if value < lo else hi if value > hi else value


def requant_ref(acc: int, bias: int, mult: int, shift: int, zero_point: int,
                relu_en: int, sat_width: int = 32, data_width: int = 8) -> int:
    """Independent reference in unbounded integers. Deliberately knows nothing
    about pipeline stages or bit widths beyond the two saturation points.

        q = clamp(round_half_up(sat(acc + bias) * mult / 2**shift) + zero_point)

    round_half_up is floor(x + 1/2): ties go toward +infinity for both signs.
    ReLU raises the lower clamp to the zero point.
    """
    sat_lo, sat_hi = signed_range(sat_width)
    out_lo, out_hi = signed_range(data_width)
    s = clamp(acc + bias, sat_lo, sat_hi)
    p = s * mult
    # Python's >> on a negative int floors, same as >>> on a signed vector
    scaled = (p + (1 << (shift - 1))) >> shift if shift > 0 else p
    lo = zero_point if relu_en else out_lo
    return clamp(scaled + zero_point, lo, out_hi)


class RequantModel:
    """Cycle-accurate model of the 4-stage epilogue."""

    LATENCY = 4

    def __init__(self, acc_width: int = 32, bias_width: int = 32,
                 sat_width: int = 32, m_width: int = 32, shift_width: int = 6,
                 data_width: int = 8, tag_width: int = 4):
        self.acc_width = acc_width
        self.bias_width = bias_width
        self.sat_width = sat_width
        self.m_width = m_width
        self.shift_width = shift_width
        self.data_width = data_width
        self.tag_width = tag_width

        sum_full_w = max(acc_width, bias_width) + 1
        self.p_w = sat_width + m_width + 1
        # the same parameter checks the RTL makes with $fatal
        if sat_width > sum_full_w:
            raise ValueError(f"SAT_WIDTH={sat_width} exceeds the natural sum width {sum_full_w}")
        if sat_width < 2 or m_width < 1 or data_width < 2:
            raise ValueError("degenerate width parameter")
        if (1 << shift_width) - 1 > self.p_w - 2:
            raise ValueError(f"SHIFT_WIDTH={shift_width} allows shifts beyond the "
                             f"product width {self.p_w}")
        self._clear()

    def _clear(self) -> None:
        # Register names match the RTL. Data registers run on every enabled
        # edge whether or not the sample is valid, exactly as the RTL does.
        self.s1 = dict(valid=0, sum=0, mult=0, shift=0, zp=0, relu=0, tag=0)
        self.s2 = dict(valid=0, prod=0, shift=0, zp=0, relu=0, tag=0)
        self.s3 = dict(valid=0, scaled=0, zp=0, relu=0, tag=0)
        self.out_valid = 0
        self.out_q = 0
        self.out_tag = 0

    def _check(self, name: str, value: int, lo: int, hi: int) -> None:
        if not lo <= value <= hi:
            raise ValueError(f"{name}={value} outside [{lo}, {hi}]. "
                             f"this is a stimulus bug, not a DUT bug")

    def state(self) -> RequantState:
        return RequantState(self.out_valid, self.out_q, self.out_tag)

    def step(self, in_valid: int = 0, in_acc: int = 0, in_tag: int = 0,
             bias: int = 0, mult: int = 0, shift: int = 0, zero_point: int = 0,
             relu_en: int = 0, en: int = 1, reset: int = 0) -> RequantState:
        """Advance one rising edge and return the ReadOnly view after it."""
        self._check("in_acc", in_acc, *signed_range(self.acc_width))
        self._check("bias", bias, *signed_range(self.bias_width))
        self._check("mult", mult, 0, (1 << self.m_width) - 1)
        self._check("shift", shift, 0, (1 << self.shift_width) - 1)
        self._check("zero_point", zero_point, *signed_range(self.data_width))
        self._check("in_tag", in_tag, 0, (1 << self.tag_width) - 1)

        if reset:
            self._clear()
            return self.state()
        if not en:
            return self.state()

        s1, s2, s3 = self.s1, self.s2, self.s3

        # stage 4: zero point, clamp, optional ReLU
        out_lo, out_hi = signed_range(self.data_width)
        with_zp = s3["scaled"] + s3["zp"]
        lo = s3["zp"] if s3["relu"] else out_lo
        self.out_valid = s3["valid"]
        self.out_q = clamp(with_zp, lo, out_hi)
        self.out_tag = s3["tag"]

        # stage 3: round half up, arithmetic shift
        rnd = (1 << (s2["shift"] - 1)) if s2["shift"] else 0
        self.s3 = dict(valid=s2["valid"], scaled=(s2["prod"] + rnd) >> s2["shift"],
                       zp=s2["zp"], relu=s2["relu"], tag=s2["tag"])

        # stage 2: multiply
        self.s2 = dict(valid=s1["valid"], prod=s1["sum"] * s1["mult"],
                       shift=s1["shift"], zp=s1["zp"], relu=s1["relu"], tag=s1["tag"])

        # stage 1: bias add, saturate
        sat_lo, sat_hi = signed_range(self.sat_width)
        self.s1 = dict(valid=int(bool(in_valid)), sum=clamp(in_acc + bias, sat_lo, sat_hi),
                       mult=mult, shift=shift, zp=zero_point,
                       relu=int(bool(relu_en)), tag=in_tag)

        return self.state()

    def reset_cycles(self, cycles: int = 3) -> RequantState:
        for _ in range(cycles):
            self.step(reset=1)
        return self.state()

    def run_one(self, **sample) -> int:
        """Push one valid sample and return out_q once it emerges."""
        self.step(in_valid=1, **sample)
        for _ in range(self.LATENCY - 1):
            s = self.step()
        if not s.out_valid:
            raise AssertionError("sample did not emerge after LATENCY edges")
        return s.out_q


if __name__ == "__main__":
    import random

    fails = 0

    def chk(name, got, exp):
        global fails
        if got != exp:
            print(f"FAIL {name:<40} got={got} exp={exp}")
            fails += 1
        else:
            print(f"pass {name:<40} = {got}")

    def q(acc, bias=0, mult=1, shift=0, zp=0, relu=0, **kw):
        return requant_ref(acc, bias, mult, shift, zp, relu, **kw)

    # hand-computed values, not derived from either implementation
    chk("identity", q(5), 5)
    chk("bias add", q(100, bias=-90), 10)
    chk("scale 3/4 of 10 = 7.5 -> 8", q(10, mult=3, shift=2), 8)
    chk("tie +0.5 -> 1", q(1, shift=1), 1)
    chk("tie +1.5 -> 2", q(3, shift=1), 2)
    chk("tie -0.5 -> 0", q(-1, shift=1), 0)
    chk("tie -1.5 -> -1", q(-3, shift=1), -1)
    chk("tie -2.5 -> -2", q(-5, shift=1), -2)
    chk("-0.75 -> -1", q(-3, shift=2), -1)
    chk("-0.25 -> 0", q(-1, shift=2), 0)
    chk("clamp high", q(1000), 127)
    chk("clamp low", q(-1000), -128)
    chk("zero point shifts", q(10, zp=-20), -10)
    chk("relu floors at zero point", q(-50, zp=-20, relu=1), -20)
    chk("relu off keeps negative", q(-50, zp=-20), -70)
    chk("relu passes positive", q(50, zp=-20, relu=1), 30)
    chk("relu still clamps high", q(1000, zp=100, relu=1), 127)
    chk("sat high: 2^31-1 >> 31", q(2**31 - 1, bias=2**31 - 1, shift=31), 1)
    chk("sat low: -2^31 >> 31", q(-2**31, bias=-2**31, shift=31), -1)
    chk("no sat at 33 bits", q(2**31 - 1, bias=2**31 - 1, shift=31, sat_width=33), 2)
    chk("mult = 0", q(12345, mult=0, zp=7), 7)
    chk("max mult, max shift", q(2**31 - 1, mult=2**32 - 1, shift=63), 1)
    chk("max shift rounds -1/2 up to 0", q(-2**31, mult=2**31, shift=63), 0)
    chk("16-bit output clamp", q(10**6, data_width=16), 32767)

    # the pipeline model against the reference, over several width sets
    rng = random.Random(1234)
    configs = [
        dict(),
        dict(sat_width=24, shift_width=5),
        dict(sat_width=33),
        dict(m_width=16, shift_width=5),
        dict(acc_width=24, bias_width=16, sat_width=20, m_width=12, shift_width=5),
        dict(data_width=16, tag_width=8),
    ]
    for cfg in configs:
        m = RequantModel(**cfg)
        m.reset_cycles(2)
        acc_lo, acc_hi = signed_range(m.acc_width)
        b_lo, b_hi = signed_range(m.bias_width)
        z_lo, z_hi = signed_range(m.data_width)
        expected = []
        mismatches = 0
        outputs = 0
        for i in range(4000):
            en = int(rng.random() > 0.2)
            valid = int(rng.random() > 0.3)
            shift = rng.randrange(1 << m.shift_width)
            sample = dict(
                in_acc=rng.choice((acc_lo, acc_hi, rng.randint(acc_lo, acc_hi),
                                   rng.randint(-300, 300))),
                bias=rng.choice((b_lo, b_hi, rng.randint(b_lo, b_hi), rng.randint(-300, 300))),
                # half the time pick a multiplier that lands the result near the
                # output range, otherwise nearly everything just clamps
                mult=rng.choice((rng.randrange(1 << m.m_width),
                                 min((1 << m.m_width) - 1, rng.randint(0, 3) << max(shift - 8, 0)))),
                shift=shift,
                zero_point=rng.randint(z_lo, z_hi),
                relu_en=rng.randint(0, 1),
            )
            tag = rng.randrange(1 << m.tag_width)
            if en and valid:
                expected.append((requant_ref(sample["in_acc"], sample["bias"], sample["mult"],
                                             sample["shift"], sample["zero_point"],
                                             sample["relu_en"], m.sat_width, m.data_width), tag))
            s = m.step(in_valid=valid, in_tag=tag, en=en, **sample)
            if en and s.out_valid:
                outputs += 1
                mismatches += int((s.out_q, s.out_tag) != expected.pop(0))
        chk(f"pipeline vs reference {cfg or 'default'}", mismatches, 0)
        chk("  outputs seen", outputs > 2000 and len(expected) <= m.LATENCY, True)

    m = RequantModel()
    m.reset_cycles(2)
    chk("run_one", m.run_one(in_acc=10, mult=3, shift=2), 8)
    m.step(in_valid=1, in_acc=99, mult=1)
    chk("reset clears valid", m.step(reset=1).out_valid, 0)
    chk("nothing emerges after reset", [m.step().out_valid for _ in range(6)], [0] * 6)

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")

    raise SystemExit(1 if fails else 0)
