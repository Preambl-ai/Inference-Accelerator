""" cocotb tb for uart_tx and uart_rx, through uart/tb/uart_loopback.sv

Every check is against uart_model: waveform() for what the transmitter must
put on the line clock by clock, reference_rx() for what the receiver must
deliver from a given line. Outputs are sampled in the ReadOnly phase.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from uart_model import reference_rx, waveform


class UartHarness:

    def __init__(self, dut):
        self.dut = dut
        self.cpb = int(dut.CLKS_PER_BIT.value)
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = random.Random(self.seed)
        self.rx_log = []        # (cycle, byte) for every rx_valid
        self.ferr_log = []      # cycle of every rx_frame_err
        self.tx_line = []       # tx_line every cycle
        self.cycle = 0

    async def start(self):
        d = self.dut
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        d.reset.value = 1
        d.loop.value = 0
        d.rx_line.value = 1
        d.tx_valid.value = 0
        d.tx_data.value = 0
        for _ in range(3):
            await RisingEdge(d.clk)
        await FallingEdge(d.clk)
        d.reset.value = 0
        cocotb.start_soon(self._monitor())
        d._log.info(f"CLKS_PER_BIT={self.cpb} SEED={self.seed}")

    async def _monitor(self):
        d = self.dut
        while True:
            await RisingEdge(d.clk)
            await ReadOnly()
            self.cycle += 1
            self.tx_line.append(int(d.tx_line.value))
            if int(d.rx_valid.value):
                self.rx_log.append((self.cycle, int(d.rx_data.value)))
            if int(d.rx_frame_err.value):
                self.ferr_log.append(self.cycle)

    async def drive_line(self, levels):
        """one level per clock on rx_line, changed at the falling edge"""
        d = self.dut
        for lv in levels:
            await FallingEdge(d.clk)
            d.rx_line.value = lv

    async def send(self, data, hold_valid=True):
        """push bytes into uart_tx as fast as it takes them"""
        d = self.dut
        for b in data:
            await FallingEdge(d.clk)
            d.tx_data.value = b
            d.tx_valid.value = 1
            while True:
                await RisingEdge(d.clk)
                await ReadOnly()
                if int(d.tx_ready.value):
                    break
            # the edge just past was the accepting one
        await FallingEdge(d.clk)
        d.tx_valid.value = 0

    async def idle(self, cycles):
        for _ in range(cycles):
            await RisingEdge(self.dut.clk)


def where(h):
    return f"(reproduce: make MODULE_NAME=uart_loopback CLKS_PER_BIT={h.cpb} SEED={h.seed})"


@cocotb.test()
async def test_loopback_random_back_to_back(dut):
    """uart_tx -> uart_rx with valid held: every byte arrives, in order, and
    the line is exactly the model waveform: 10 bits of CLKS_PER_BIT clocks
    each, the next start bit right after the previous stop bit"""
    h = UartHarness(dut)
    await h.start()
    dut.loop.value = 1
    data = [h.rng.randrange(256) for _ in range(200)] + [0x00, 0xFF, 0x55, 0xAA]
    t0 = len(h.tx_line)
    await h.send(data)
    await h.idle(12 * h.cpb)
    got = [b for _, b in h.rx_log]
    assert got == data, f"loopback lost or changed bytes {where(h)}"
    assert not h.ferr_log, "framing error in loopback"
    line = h.tx_line[t0:]
    first = line.index(0)
    want = waveform(data, h.cpb, lead_idle=0)
    assert line[first:first + len(want)] == want[:len(line) - first], \
        f"tx line differs from the model waveform {where(h)}"
    # throughput: back to back means exactly 10 bit times per byte
    assert len(want) - 2 * h.cpb == 10 * h.cpb * len(data)


@cocotb.test()
async def test_rx_rate_error(dut):
    """receiver fed by a transmitter 2 % fast and 2 % slow, back to back and
    with idle gaps, compared byte for byte with the reference receiver"""
    h = UartHarness(dut)
    await h.start()
    for err in (0.02, -0.02):
        for gap in (0.0, 0.7):
            data = [h.rng.randrange(256) for _ in range(120)]
            line = waveform(data, h.cpb, rate_error=err, gap_bits=gap)
            ref, ref_ferr = reference_rx(line, h.cpb)
            assert ref == data and ref_ferr == 0, "stimulus outside the model's tolerance"
            n0, f0 = len(h.rx_log), len(h.ferr_log)
            await h.drive_line(line)
            await h.idle(4)
            got = [b for _, b in h.rx_log[n0:]]
            assert got == data, (f"rate {err:+.0%} gap {gap}: {sum(a != b for a, b in zip(got, data))} "
                                 f"bytes wrong, {len(got)} of {len(data)} received {where(h)}")
            assert len(h.ferr_log) == f0, "framing error at +-2 %"


@cocotb.test()
async def test_framing_error(dut):
    """a stop bit sampled low: frame_err pulses once, the byte is not
    delivered, and the receiver recovers once the line has been idle. a line
    held low afterwards is one more framing error, not a stream of 0x00."""
    h = UartHarness(dut)
    await h.start()
    data = [h.rng.randrange(256) for _ in range(6)]
    # the bad byte ends its packet of bytes; the next ones start after the
    # idle tail waveform() leaves (the line has to be idle to resync)
    line = (waveform(data[:2], h.cpb, bad_stop=[1]) +
            waveform(data[2:], h.cpb, lead_idle=0) +
            [1] * 5 + [0] * (30 * h.cpb) + [1] * (3 * h.cpb))
    ref, ref_ferr = reference_rx(line, h.cpb)
    await h.drive_line(line)
    await h.idle(4)
    got = [b for _, b in h.rx_log]
    assert (got, len(h.ferr_log)) == (ref, ref_ferr), \
        f"rtl {got} / {len(h.ferr_log)} frame errors, model {ref} / {ref_ferr}"
    assert got == [data[0]] + data[2:] and ref_ferr == 2, "unexpected model result"


@cocotb.test()
async def test_rx_glitch_ignored(dut):
    """a low pulse shorter than half a bit is not a start bit"""
    h = UartHarness(dut)
    await h.start()
    glitch = max(1, h.cpb // 2 - 3)
    line = [1] * 20 + [0] * glitch + [1] * (3 * h.cpb) + waveform([0x3C], h.cpb)
    ref, _ = reference_rx(line, h.cpb)
    await h.drive_line(line)
    await h.idle(4)
    got = [b for _, b in h.rx_log]
    assert got == ref == [0x3C] and not h.ferr_log, f"glitch was taken as data: {got}"


@cocotb.test()
async def test_rx_matches_reference_cycle_for_cycle(dut):
    """random traffic with random gaps, rate errors inside +-2 % and the odd
    bad stop bit: the receiver's bytes and framing errors equal the cycle
    model's, and each byte appears at the same clock"""
    h = UartHarness(dut)
    await h.start()
    line = []
    for _ in range(40):
        data = [h.rng.randrange(256) for _ in range(h.rng.randint(1, 6))]
        bad = [h.rng.randrange(len(data))] if h.rng.random() < 0.2 else []
        line += waveform(data, h.cpb, rate_error=h.rng.uniform(-0.02, 0.02),
                         gap_bits=h.rng.choice((0, 0, 0.3, 2)), bad_stop=bad,
                         lead_idle=h.rng.randint(0, 3))
    ref, ref_ferr = reference_rx(line, h.cpb)
    await h.drive_line(line)
    await h.idle(4)
    got = [b for _, b in h.rx_log]
    assert got == ref and len(h.ferr_log) == ref_ferr, \
        f"rtl {len(got)} bytes / {len(h.ferr_log)} errors, model {len(ref)} / {ref_ferr} {where(h)}"
