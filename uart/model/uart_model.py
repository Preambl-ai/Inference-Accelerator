"""Model of the UART line and of uart_rx's sampling, for the uart TB.

waveform() turns bytes into line levels per clock, with an optional rate
error (the transmitter's bit period is clks_per_bit * (1 + rate_error)),
idle gaps and deliberately bad stop bits. reference_rx() is a cycle-level
model of uart_rx (2-flop synchronizer, half-bit start check, mid-bit
sampling, stop check, wait-for-high after a framing error). The self-test
shows where its tolerance ends, so the TB's +-2 % is known to be inside it.
"""

from typing import List, Sequence, Tuple


def frame_levels(byte: int, stop: int = 1) -> List[int]:
    """start, 8 data bits LSB first, stop"""
    return [0] + [(byte >> i) & 1 for i in range(8)] + [stop]


def waveform(data: Sequence[int], clks_per_bit: int, rate_error: float = 0.0,
             gap_bits: float = 0.0, bad_stop: Sequence[int] = (), lead_idle: int = 4) -> List[int]:
    """line level for every clock. bit edges fall at round(k * period), so a
    fractional period spreads its error evenly instead of all on one bit."""
    period = clks_per_bit * (1.0 + rate_error)
    levels: List[Tuple[int, float]] = []
    for i, b in enumerate(data):
        for lv in frame_levels(b, 0 if i in bad_stop else 1):
            levels.append((lv, period))
        if gap_bits:
            levels.append((1, gap_bits * period))
    out = [1] * lead_idle
    t = 0.0
    for lv, dur in levels:
        start, t = round(t), t + dur
        out += [lv] * (round(t) - start)
    return out + [1] * (2 * clks_per_bit)


def reference_rx(line: Sequence[int], clks_per_bit: int):
    """what uart_rx does, cycle by cycle. returns (bytes, frame_errors)"""
    half = (clks_per_bit - 1) // 2
    s1 = s2 = 1
    state, cnt, bit, shift = "idle", 0, 0, 0
    got, ferr = [], 0
    for lv in line:
        rx = s2                          # what the state machine sees this cycle
        if state == "idle":
            if not rx:
                state, cnt = "start", 0
        elif state == "start":
            if cnt == half:
                if not rx:
                    state, cnt, bit = "data", 0, 0
                else:
                    state = "idle"
            else:
                cnt += 1
        elif state == "data":
            if cnt == clks_per_bit - 1:
                cnt = 0
                shift = (shift >> 1) | (rx << 7)
                if bit == 7:
                    state = "stop"
                else:
                    bit += 1
            else:
                cnt += 1
        elif state == "stop":
            if cnt == clks_per_bit - 1:
                cnt = 0
                if rx:
                    got.append(shift)
                    state = "idle"
                else:
                    ferr += 1
                    state = "wait_high"
            else:
                cnt += 1
        elif state == "wait_high":
            if rx:
                state = "idle"
        s2, s1 = s1, lv
    return got, ferr


def tolerance(clks_per_bit: int, n_bytes: int = 64, seed: int = 1) -> Tuple[float, float]:
    """largest +- rate error (in 0.5 % steps) the reference receiver survives
    on back-to-back random bytes"""
    import random
    rng = random.Random(seed)
    data = [rng.randrange(256) for _ in range(n_bytes)]
    lim = []
    for sign in (+1, -1):
        e = 0.0
        while True:
            got, ferr = reference_rx(waveform(data, clks_per_bit, sign * (e + 0.005)), clks_per_bit)
            if got != data or ferr:
                break
            e += 0.005
            if e > 0.2:
                break
        lim.append(sign * e)
    return tuple(lim)


if __name__ == "__main__":
    import random
    fails = 0

    def chk(name, ok, info=""):
        global fails
        print(f"{'pass' if ok else 'FAIL'} {name} {info}")
        fails += not ok

    chk("frame of 0xA5", frame_levels(0xA5) == [0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    w = waveform([0x00], 10)
    chk("exact rate: 10 bits x 10 clocks between the idles", w[4:4 + 90] == [0] * 90 and w[94] == 1)
    w = waveform([0x55], 10, rate_error=0.02)
    chk("+2 % stretches a frame to 102 clocks", len(w) - 4 - 20 == 102, str(len(w) - 24))

    rng = random.Random(7)
    for cpb in (8, 16, 100):
        data = [rng.randrange(256) for _ in range(200)]
        for err in (0.0, 0.02, -0.02):
            got, ferr = reference_rx(waveform(data, cpb, err), cpb)
            chk(f"cpb={cpb} rate {err:+.0%} back to back", got == data and ferr == 0)
        got, ferr = reference_rx(waveform(data, cpb, 0.0, gap_bits=1.5), cpb)
        chk(f"cpb={cpb} with idle gaps", got == data and ferr == 0)
        # a bad stop bit in a back-to-back stream also loses the framing of
        # what follows (the next start bit looks like data); the receiver is
        # only expected to drop that byte, flag it, and recover once the line
        # has been idle. the packet CRC deals with the rest.
        line = waveform(data[:4], cpb, bad_stop=[3]) + waveform(data[4:10], cpb, lead_idle=0)
        got, ferr = reference_rx(line, cpb)
        chk(f"cpb={cpb} bad stop bit: byte dropped, one frame error, recovers after idle",
            got == data[:3] + data[4:10] and ferr == 1, f"got {len(got)} bytes, {ferr} errors")
        hi, lo = tolerance(cpb)
        chk(f"cpb={cpb} tolerance covers +-2 %", hi >= 0.02 and lo <= -0.02,
            f"(+{hi:.1%} / {lo:.1%})")
    # and it really does fail somewhere: the TB's +-2 % is not trivially easy
    data = [rng.randrange(256) for _ in range(64)]
    got, _ = reference_rx(waveform(data, 16, 0.08), 16)
    chk("+8 % at cpb=16 breaks, as it should", got != data)
    # a held-low line is one framing error, not a stream of zero bytes
    line = [1] * 8 + [0] * 400 + [1] * 40
    got, ferr = reference_rx(line, 16)
    chk("break: one frame error, no bytes", got == [] and ferr == 1, f"{got} {ferr}")

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
