""" cocotb tb for top/rtl/top.sv, end to end through the UART pins

The only things touching the DUT are the clock, the reset button and the two
UART lines. A UART BFM drives uart_txd_in and listens on uart_rxd_out, and
every byte it sends comes from host/protocol.py, the same code the real host
driver uses: the sequences in protocol.py run here through run_async over a
cocotb transport, and over pyserial on the bench. Expected results come from
the conv_layer NumPy model (conv_ref through host/layers.py).

Fault injection lives in the transport: a hook can flip or drop one byte of a
chosen request, or garble a response, and the tests check that the bridge's
answer and the host's retry logic deal with it.
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge
from cocotb.utils import get_sim_time
import numpy as np

import protocol as P
from conv_layer_model import (E_HOST, E_SEL, Layer, check_command, err_names, estimate_cycles,
                              k_of, random_layer, requant_params)
from layers import expected, hw_params, layer_images


class UartTransport:
    """async transport for protocol.run_async, bit-level on the DUT's pins"""

    def __init__(self, dut, cpb, timeout_bits):
        self.dut, self.cpb, self.timeout_bits = dut, cpb, timeout_bits
        self.rx = []                 # (byte, sim cycle its stop bit ended)
        self.inject = None           # fn(packet_number, bytes) -> bytes to really send
        self.garble = None           # fn(packet_number, response bytes) -> bytes the host sees
        self.packets = 0
        self.log = []                # (packet, response) as seen on the wire

    def start(self):
        self.dut.uart_txd_in.value = 1
        cocotb.start_soon(self._monitor())

    def now(self):
        return int(get_sim_time(unit="ns")) // 10

    async def _monitor(self):
        d = self.dut
        clk = d.CLK100MHZ
        while True:
            await FallingEdge(d.uart_rxd_out)
            await ClockCycles(clk, self.cpb // 2)
            if int(d.uart_rxd_out.value):
                continue                       # not a start bit
            b = 0
            for i in range(8):
                await ClockCycles(clk, self.cpb)
                b |= int(d.uart_rxd_out.value) << i
            await ClockCycles(clk, self.cpb)
            assert int(d.uart_rxd_out.value) == 1, "board sent a byte with a bad stop bit"
            self.rx.append((b, self.now()))

    async def send(self, data):
        d = self.dut
        for b in data:
            for lv in [0] + [(b >> i) & 1 for i in range(8)] + [1]:
                await FallingEdge(d.CLK100MHZ)
                d.uart_txd_in.value = lv
                await ClockCycles(d.CLK100MHZ, self.cpb - 1, rising=False)
        await FallingEdge(d.CLK100MHZ)

    async def xfer(self, packet, resp_len):
        self.packets += 1
        wire = self.inject(self.packets, packet) if self.inject else packet
        self.rx.clear()
        await self.send(wire)
        self.last_sent = self.now()
        # wait for the answer: give up once nothing has come for longer than
        # the bridge's own timeout (so its TIMEOUT answer still gets here)
        quiet_limit = (2 * self.timeout_bits + 30) * self.cpb
        quiet, seen = 0, 0
        while len(self.rx) < resp_len and quiet < quiet_limit:
            await ClockCycles(self.dut.CLK100MHZ, 4 * self.cpb)
            if len(self.rx) != seen:
                seen, quiet = len(self.rx), 0
            else:
                quiet += 4 * self.cpb
        resp = bytes(b for b, _ in self.rx[:resp_len])
        self.first_resp_at = self.rx[0][1] if self.rx else None
        self.log.append((wire, resp))
        return self.garble(self.packets, resp) if self.garble else resp

    async def wait_idle(self):
        await ClockCycles(self.dut.CLK100MHZ, (2 * self.timeout_bits + 20) * self.cpb)
        self.rx.clear()


class TopHarness:

    def __init__(self, dut):
        self.dut = dut
        g = lambda name: int(getattr(dut, name).value)
        self.n, self.lat, self.cpb = g("N"), g("READ_LATENCY"), g("CLKS_PER_BIT")
        self.timeout_bits = g("TIMEOUT_BITS")
        self.seed = int(os.environ.get("SEED") or random.randrange(2**31))
        self.rng = np.random.default_rng(self.seed)
        self.t = UartTransport(dut, self.cpb, self.timeout_bits)
        self.stats = P.LinkStats()

    def where(self):
        return f"(reproduce: make MODULE_NAME=top N={self.n} SEED={self.seed})"

    async def start(self):
        d = self.dut
        cocotb.start_soon(Clock(d.CLK100MHZ, 10, unit="ns").start())
        d.btn_rst.value = 0
        self.t.start()
        await ClockCycles(d.CLK100MHZ, 60)          # power-on reset runs out
        self.hw, self.build = await self.run(P.identify(self.stats))
        self.mirror = {}
        d._log.info(f"N={self.n} L={self.lat} CLKS_PER_BIT={self.cpb} {self.hw} SEED={self.seed}")

    async def run(self, gen):
        return await P.run_async(gen, self.t)

    async def raw(self, packet, read_words=0):
        """one packet, no retry: (status, words) or (None, []) if no good answer"""
        resp = await self.t.xfer(packet, P.response_len(read_words))
        try:
            return P.parse_response(resp, read_words)
        except P.ProtocolError:
            return None, []

    # -- act_buffer contents through the UART ----------------------------------
    async def fill_bank(self, bank, words=None):
        depth = 1 << self.hw.addr_width
        words = words if words is not None else [int(v) for v in self.rng.integers(0, 1 << (8 * self.n), depth)]
        imgs = P.memory_images(self.hw, act=[(bank, 0, words)])
        await self.run(P.load(self.hw, imgs, self.stats))
        for i, w in enumerate(words):
            self.mirror[(bank, i)] = w

    async def check_bank(self, bank, what):
        depth = 1 << self.hw.addr_width
        got = await self.run(P.read_act(self.hw, bank, 0, depth, self.stats))
        bad = [i for i in range(depth) if got[i] != self.mirror.get((bank, i))]
        assert not bad, f"bank {bank} {what}: {len(bad)} words differ, first {bad[:4]} {self.where()}"

    def note_output(self, layer):
        for i, w in enumerate(expected(layer, self.n)):
            self.mirror[(layer.dst_sel, layer.dst_base + i)] = w

    def note_input(self, layer):
        from conv_layer_model import pack_act
        for i, w in enumerate(pack_act(layer.x, self.n, layer.zp_in)):
            self.mirror[(layer.src_sel, layer.src_base + i)] = w

    async def run_layer(self, layer, name, with_act=True, check_banks=True):
        cmd = layer.command(self.n)
        assert check_command(cmd, hw_params(self.hw)) == 0, "stimulus bug: illegal layer"
        exp = expected(layer, self.n)
        res = await self.run(P.run_layer(self.hw, cmd, layer_images(self.hw, layer, with_act),
                                         len(exp), self.stats))
        assert res.status & P.ST_DONE and not res.status & P.ST_ERR, \
            f"{name}: STATUS {res.status:#x} ERR_CODE {err_names(res.err_code)}"
        assert res.output == exp, f"{name}: output differs from the NumPy conv {self.where()}"
        est = estimate_cycles(cmd, self.n, self.lat)
        assert res.cycles == est, f"{name}: CYCLES {res.cycles}, conv_layer formula {est}"
        if with_act:
            self.note_input(layer)
        self.note_output(layer)
        if check_banks:
            for bank in sorted({layer.src_sel, layer.dst_sel}):
                await self.check_bank(bank, f"after {name}")
        self.dut._log.info(f"{name}: output matches, CYCLES={res.cycles} (formula {est}), "
                           f"{res.polls} STATUS polls, link {self.stats}")
        return res


async def fresh(dut):
    h = TopHarness(dut)
    await h.start()
    return h


def small_layer(h, **force):
    hw = hw_params(h.hw)
    return random_layer(h.rng, hw, max_dim=5, max_cin=2 * h.n, max_cout=2 * h.n, **force)


# ----------------------------------------------------------------------------

@cocotb.test()
async def test_identify_and_link(dut):
    """ID, VERSION, PARAMS match the build; SCRATCH round trip; how long the
    bridge takes from the end of a request to the start of its answer"""
    h = await fresh(dut)
    assert (h.hw.n, h.hw.read_latency) == (h.n, h.lat)
    assert h.build == int(dut.BUILD_ID.value)
    for v in (0, 0xFFFFFFFF, 0x12345678):
        await h.run(P.write32(P.A_SCRATCH, v, h.stats))
        last_byte_end = h.t.last_sent
        first = h.t.first_resp_at
        got = await h.run(P.read32(P.A_SCRATCH, h.stats))
        assert got == v
    gap = first - last_byte_end
    dut._log.info(f"LATENCY bridge write32: first response start bit {gap - h.cpb * 10} clocks "
                  f"after the request's last stop bit ends (CLKS_PER_BIT={h.cpb})")
    assert h.stats.retries == 0


@cocotb.test()
async def test_layer_end_to_end(dut):
    """random layers loaded over the UART, run, read back and compared with
    the NumPy conv; whole banks compared, so nothing outside the output moved"""
    h = await fresh(dut)
    for bank in range(h.hw.nbuf):
        await h.fill_bank(bank)
    for i in range(int(os.environ.get("LAYERS") or 3)):
        # any source bank, any other bank as the destination (1 - src
        # assumed two banks; with three it gave dst = -1 and E_SEL)
        src = int(h.rng.integers(0, h.hw.nbuf))
        dst = (src + int(h.rng.integers(1, h.hw.nbuf))) % h.hw.nbuf
        layer = small_layer(h, src_sel=src, dst_sel=dst)
        await h.run_layer(layer, f"layer {i}")
    assert dut.led.value.to_unsigned() & 0b0100, "done LED is off"


@cocotb.test()
async def test_ping_pong_chain(dut):
    """three layers 0 -> 1 -> 0 -> 1, each reading what the previous wrote,
    weights for all three loaded up front, final bank against chained NumPy"""
    h = await fresh(dut)
    n, rng = h.n, h.rng
    x = rng.integers(-128, 128, (4, 5, n + 1))
    zp = int(rng.integers(-20, 21))
    specs = [dict(kh=3, kw=3, stride=1, pad=1), dict(kh=1, kw=1, stride=1, pad=0),
             dict(kh=3, kw=3, stride=2, pad=1)]
    layers, b_next, p_next, src, base = [], 0, 0, 0, 0
    for i, s in enumerate(specs):
        cin, cout = x.shape[2], int(rng.integers(1, 2 * n + 1))
        w = rng.integers(-128, 128, (s["kh"], s["kw"], cin, cout))
        bias, mult, shift = requant_params(rng, cout, s["kh"] * s["kw"] * cin)
        zp_out = int(rng.integers(-20, 21))
        layer = Layer(x=x, w=w, bias=bias, mult=mult, shift=shift, zp_in=zp, zp_out=zp_out,
                      relu_en=i % 2, stride=s["stride"], pad=s["pad"], src_sel=src,
                      dst_sel=1 - src, src_base=base, dst_base=int(rng.integers(0, 16)),
                      b_base=b_next, p_base=p_next)
        cmd = layer.command(n)
        b_next += cmd["cout_words"] * k_of(s["kh"], s["kw"], cmd["cin_words"], n)
        p_next += cmd["cout_words"] * n
        layers.append(layer)
        x, zp, src, base = layer.reference(), zp_out, 1 - src, layer.dst_base
    for bank in range(h.hw.nbuf):
        await h.fill_bank(bank)
    from conv_layer_model import pack_act, pack_b, pack_p
    first = layers[0]
    imgs = P.memory_images(
        h.hw, act=[(first.src_sel, first.src_base, pack_act(first.x, n, first.zp_in))],
        b=[(l.b_base, pack_b(l.w, n)) for l in layers], p=[(l.p_base, pack_p(l, n)) for l in layers])
    await h.run(P.load(h.hw, imgs, h.stats))
    h.note_input(first)
    for i, layer in enumerate(layers):
        cmd = layer.command(n)
        await h.run(P.write_command(cmd, h.stats))
        await h.run(P.start(h.stats))
        st, _ = await h.run(P.poll_done(h.stats))
        assert st & P.ST_DONE, f"chain layer {i}: STATUS {st:#x}"
        cyc = await h.run(P.read32(P.A_CYCLES, h.stats))
        assert cyc == estimate_cycles(cmd, n, h.lat)
        h.note_output(layer)
    for bank in range(h.hw.nbuf):
        await h.check_bank(bank, "after the chain")


@cocotb.test()
async def test_err_flags(dut):
    """illegal commands end in STATUS.err with conv_layer's err_code; the
    layer still counts as launched; nothing is written"""
    h = await fresh(dut)
    for bank in range(h.hw.nbuf):
        await h.fill_bank(bank)
    good = small_layer(h)
    await h.run(P.load(h.hw, layer_images(h.hw, good), h.stats))
    h.note_input(good)
    # src == dst
    cmd = dict(good.command(h.n), dst_sel=good.src_sel)
    await h.run(P.write_command(cmd, h.stats))
    before = await h.run(P.read32(P.A_LAUNCH, h.stats))
    await h.run(P.start(h.stats))
    st, _ = await h.run(P.poll_done(h.stats))
    code = await h.run(P.read32(P.A_ERR_CODE, h.stats))
    cyc = await h.run(P.read32(P.A_CYCLES, h.stats))
    assert st & P.ST_ERR and not st & P.ST_DONE and code == 1 << E_SEL, (hex(st), err_names(code))
    assert cyc == 4, f"rejected command CYCLES={cyc}, the check phase is 4 edges"
    assert await h.run(P.read32(P.A_LAUNCH, h.stats)) == before + 1
    assert dut.led.value.to_unsigned() & 0b1000, "err LED is off"
    # host_mode high at start: conv_layer's E_HOST
    await h.run(P.write_command(good.command(h.n), h.stats))
    await h.run(P.write32(P.A_CTRL, P.CTRL_START | P.CTRL_HOST_MODE, h.stats))
    st, _ = await h.run(P.poll_done(h.stats))
    code = await h.run(P.read32(P.A_ERR_CODE, h.stats))
    assert st & P.ST_ERR and code == 1 << E_HOST, (hex(st), err_names(code))
    await h.run(P.write32(P.A_CTRL, 0, h.stats))
    for bank in range(h.hw.nbuf):
        await h.check_bank(bank, "after the rejected commands")
    # W1C clears err; a good layer still runs
    await h.run(P.write32(P.A_STATUS, P.ST_ERR, h.stats))
    assert not (await h.run(P.read32(P.A_STATUS, h.stats))) & P.ST_ERR
    await h.run_layer(good, "good layer after the errors", with_act=False)
    # W1C one bit at a time against the real sticky bits: with done set,
    # clearing err must leave done alone, and clearing done must clear it
    # (clearing both at once cannot tell the bits apart; a mutation run
    # showed the end-to-end test missing a swapped bit)
    await h.run(P.write32(P.A_STATUS, P.ST_ERR, h.stats))
    st = await h.run(P.read32(P.A_STATUS, h.stats))
    assert st & P.ST_DONE, f"W1C of err cleared done (STATUS {st:#x})"
    await h.run(P.write32(P.A_STATUS, P.ST_DONE, h.stats))
    st = await h.run(P.read32(P.A_STATUS, h.stats))
    assert not st & P.ST_DONE, f"W1C of done did not clear it (STATUS {st:#x})"


@cocotb.test()
async def test_link_errors(dut):
    """a corrupted or dropped byte is never written silently: the bridge
    answers HCRC / DCRC / TIMEOUT and the register is unchanged; the host's
    retries then get every packet through"""
    h = await fresh(dut)
    await h.run(P.write32(P.A_SCRATCH, 0x11111111, h.stats))

    def flip(at):
        return lambda k, b: b[:at] + bytes((b[at] ^ 0x10,)) + b[at + 1:]

    def drop(at):
        return lambda k, b: b[:at] + b[at + 1:]

    pkt = P.pkt_write32(P.A_SCRATCH, 0x22222222)
    for name, hook, want in (("data byte flipped", flip(10), P.ST_DCRC),
                             ("address byte flipped", flip(3), P.ST_HCRC),
                             ("header CRC byte flipped", flip(8), P.ST_HCRC),
                             ("data byte dropped", drop(11), P.ST_TIMEOUT),
                             ("header byte dropped", drop(4), None)):
        h.t.inject = hook
        st, _ = await h.raw(pkt)
        h.t.inject = None
        await h.t.wait_idle()
        assert st != P.ST_OK, f"{name}: a broken packet was accepted"
        if want is not None:
            assert st == want, f"{name}: status {P.STATUS_NAMES.get(st, st)}, expected {P.STATUS_NAMES[want]}"
        got = await h.run(P.read32(P.A_SCRATCH, h.stats))
        assert got == 0x11111111, f"{name}: SCRATCH changed to {got:#x}"
        dut._log.info(f"{name}: answer {P.STATUS_NAMES.get(st, 'none')}, SCRATCH unchanged")
    # every link error must throw away a half-staged wide word. P words are 53
    # bits, so they take two slots at any N: stage slot 0, break a packet,
    # then the top slot must be refused. (at N <= 4 an ACT or B word is one
    # slot, so without this nothing at top level would notice a bridge that
    # never raises link_err; a mutation run showed exactly that.)
    lo, hi = h.hw.p_addr(5, 0), h.hw.p_addr(5, 1)
    st, _ = await h.raw(P.pkt_write32(lo, 0x1234))
    st2, _ = await h.raw(P.pkt_write32(hi, 0x5))
    assert (st, st2) == (P.ST_OK, P.ST_OK), "control: a clean two-slot P write must commit"
    for name, hook in (("DCRC", flip(10)), ("HCRC", flip(3)), ("TIMEOUT", drop(11))):
        st, _ = await h.raw(P.pkt_write32(lo, 0x1234))
        assert st == P.ST_OK
        h.t.inject = hook
        await h.raw(P.pkt_write32(P.A_SCRATCH, 0))
        h.t.inject = None
        await h.t.wait_idle()
        st, _ = await h.raw(P.pkt_write32(hi, 0x5))
        assert st == P.ST_SLVERR, (f"after a {name} error the staged P slot survived: "
                                   f"top slot answered {P.STATUS_NAMES.get(st, st)}")
    # bad opcode and bad LEN
    for bad in (P.header(0x07, P.A_SCRATCH, 0), P.header(P.OP_WRITE32, P.A_SCRATCH, 3)):
        st, _ = await h.raw(bad)
        await h.t.wait_idle()
        assert st == P.ST_BADOP
    # a read packet that fails before the bus gets the 4-byte short answer,
    # and the host must read it as HCRC / BADOP, not as a garbled response
    h.t.inject = flip(3)
    st, words = await h.raw(P.pkt_burst_read(P.A_SCRATCH, 4), 4)
    h.t.inject = None
    await h.t.wait_idle()
    assert (st, words) == (P.ST_HCRC, []), f"bad-header burst read: {st}, {words}"
    assert len(h.t.log[-1][1]) == 4, "a short answer is exactly 4 bytes"
    st, words = await h.raw(P.header(P.OP_READ32, P.A_SCRATCH, 2), 1)
    await h.t.wait_idle()
    assert (st, words) == (P.ST_BADOP, []), "read32 with LEN 2 must get the short BADOP answer"
    # a garbled *response* to a read: the host retries, gets the right value
    once = {"left": 1}

    def garble(k, r):
        if once["left"] and len(r) > 3:
            once["left"] -= 1
            return r[:2] + bytes((r[2] ^ 1,)) + r[3:]
        return r
    h.t.garble = garble
    got = await h.run(P.read32(P.A_SCRATCH, h.stats))
    h.t.garble = None
    assert got == 0x11111111 and h.stats.protocol_errors >= 1
    # out of range address and a write-only window
    st, _ = await h.raw(P.pkt_write32(0x01000000, 0))
    assert st == P.ST_DECERR
    st, words = await h.raw(P.pkt_burst_read(P.B_BASE, 3), 3)
    assert st == P.ST_SLVERR and words == [0, 0, 0]
    # every error above was the host's packet going wrong, never the host
    # talking over an answer: the bridge must drain before it answers, so
    # none of those packets' tails may have been counted as DROPPED
    st = await h.run(P.check_dropped(h.stats))
    assert P.dropped_count(st) == 0, "an error answer overlapped the rest of the host's packet"


@cocotb.test()
async def test_bus_errors_and_refusals(dut):
    """the error paths end to end, against the real conv_layer and act_buffer:
    a bus error stops a burst at that word (writes after it never happen,
    reads after it are zeros); ACT reads are refused in host mode and ACT
    writes outside it; a P word's top slot without its staged lower slot is
    refused and commits nothing"""
    h = await fresh(dut)
    # -- writes: a refused word stops the burst. SCRATCH (ok), then BUILD_ID
    # (read only: SLVERR), ... up to H_IN at 0x40, which must stay untouched
    await h.run(P.write_command(dict.fromkeys(P.CMD_FIELDS, 1), h.stats))
    st, _ = await h.raw(P.pkt_burst_write(P.A_SCRATCH, [0x5C5C] + [0xEE] * 8))
    assert st == P.ST_SLVERR, f"burst over a read-only register: {P.STATUS_NAMES.get(st, st)}"
    assert await h.run(P.read32(P.A_SCRATCH, h.stats)) == 0x5C5C, "the word before the error was lost"
    assert await h.run(P.read32(P.A_CMD0, h.stats)) == 1, "a burst kept writing after SLVERR"
    # unmapped first word (0x38): DECERR, and H_IN/W_IN two words on stay untouched
    st, _ = await h.raw(P.pkt_burst_write(0x38, [7, 7, 9, 9]))
    assert st == P.ST_DECERR
    assert await h.run(P.read32(P.A_CMD0, h.stats)) == 1 and \
        await h.run(P.read32(P.A_CMD0 + 4, h.stats)) == 1, "a burst kept writing after DECERR"
    # -- reads: LAUNCH_COUNT (ok), 0x38 (DECERR), 0x3C, H_IN: zeros from the error on
    launches = await h.run(P.read32(P.A_LAUNCH, h.stats))
    st, words = await h.raw(P.pkt_burst_read(P.A_LAUNCH, 4), 4)
    assert (st, words) == (P.ST_DECERR, [launches, 0, 0, 0]), \
        f"DECERR mid burst read: {P.STATUS_NAMES.get(st, st)} {words}"
    # -- ACT refusals by host_mode, on the real act_buffer
    word = [int(h.rng.integers(0, 1 << (8 * h.n)))]
    st, _ = await h.raw(P.pkt_burst_write(h.hw.act_addr(1, 3), word))
    assert st == P.ST_SLVERR, "ACT write without host_mode was accepted"
    await h.run(P.write32(P.A_CTRL, P.CTRL_HOST_MODE, h.stats))
    await h.run(P.burst_write(h.hw.act_addr(1, 3), word, h.stats))
    st, words = await h.raw(P.pkt_burst_read(h.hw.act_addr(1, 3), 1), 1)
    assert (st, words) == (P.ST_SLVERR, [0]), "ACT read in host mode was accepted (reads are frozen then)"
    await h.run(P.write32(P.A_CTRL, 0, h.stats))
    got = await h.run(P.read_act(h.hw, 1, 3, 1, h.stats))
    assert got == word, "the ACT word written in host mode did not read back"
    # -- stale partial commit on the real P memory: the layer would use the
    # wrong parameters, so check by running one. a good P word first...
    layer = small_layer(h, kh=1, kw=1, stride=1, pad=0, h_in=1, w_in=h.n, cin=h.n, cout=h.n,
                        p_base=0)
    await h.run_layer(layer, "reference layer", check_banks=False)
    # ...then a top slot on its own, and a top slot for a different word than
    # the one staged: both refused. 0x1F_FFFF in the top slot is shift = 31,
    # mult = 0xFFFF, which would change channel 0's output if it landed
    st, _ = await h.raw(P.pkt_write32(h.hw.p_addr(0, 1), 0x1F_FFFF))
    assert st == P.ST_SLVERR, "P top slot alone was committed"
    st1, _ = await h.raw(P.pkt_write32(h.hw.p_addr(1, 0), 0xFFFFFFFF))
    st2, _ = await h.raw(P.pkt_write32(h.hw.p_addr(0, 1), 0x1F_FFFF))
    assert (st1, st2) == (P.ST_OK, P.ST_SLVERR), "P top slot committed with another word's staged slot"
    # rerun WITHOUT reloading P: the output only matches if P word 0 is untouched
    cmd = layer.command(h.n)
    await h.run(P.start(h.stats))
    st, _ = await h.run(P.poll_done(h.stats))
    exp = expected(layer, h.n)
    out = await h.run(P.read_act(h.hw, cmd["dst_sel"], cmd["dst_base"], len(exp), h.stats))
    assert st & P.ST_DONE and out == exp, "a refused P write changed the parameters"


@cocotb.test()
async def test_dropped_bytes_mid_answer(dut):
    """bytes sent while the board is answering are dropped, but counted:
    STATUS.DROPPED and DROPPED_COUNT show exactly how many, the answer itself
    is intact, and the host reports a protocol fault instead of retrying"""
    h = await fresh(dut)
    t, clk = h.t, dut.CLK100MHZ
    st = await h.run(P.check_dropped(h.stats))
    assert not st & P.ST_DROPPED, "DROPPED set on a clean link"
    words_in = [int(v) for v in h.rng.integers(0, 1 << (8 * h.n), 64)]
    await h.run(P.write32(P.A_CTRL, P.CTRL_HOST_MODE, h.stats))
    await h.run(P.burst_write(h.hw.act_addr(0, 0), words_in, h.stats))
    await h.run(P.write32(P.A_CTRL, 0, h.stats))
    # a long answer (64 words), and 6 bytes sent into the middle of it; the
    # bytes look like the start of a packet, so a parser that took them would
    # be visibly derailed
    pkt = P.pkt_burst_read(h.hw.act_addr(0, 0), 64)
    want = P.response_len(64)
    extra = bytes((0xA5, P.OP_WRITE32, 0x20, 0, 0, 0))
    t.rx.clear()
    await t.send(pkt)
    while len(t.rx) < 3:
        await ClockCycles(clk, 4 * h.cpb)
    await t.send(extra)
    assert len(t.rx) < want - 8, "the extra bytes did not land inside the answer"
    while len(t.rx) < want:
        await ClockCycles(clk, 4 * h.cpb)
    status, words = P.parse_response(bytes(b for b, _ in t.rx[:want]), 64)
    assert (status, words) == (P.ST_OK, words_in), "the answer was disturbed by the dropped bytes"
    await t.wait_idle()
    st = await h.run(P.read32(P.A_STATUS, h.stats))
    assert st & P.ST_DROPPED and P.dropped_count(st) == len(extra), \
        f"STATUS {st:#x}: expected DROPPED with count {len(extra)}"
    # the host side: any check is a ProtocolFault carrying the count...
    try:
        await h.run(P.check_dropped(h.stats))
        assert False, "check_dropped did not raise"
    except P.ProtocolFault as e:
        assert e.count == len(extra)
    # ...and so is a failed exchange: the host probes STATUS before retrying
    # and stops, instead of retrying into a string of CRC errors
    packets = h.stats.packets
    first = t.packets + 1               # corrupt only this write, not the STATUS probe after it
    h.t.inject = lambda k, b: b[:10] + bytes((b[10] ^ 1,)) + b[11:] if k == first else b
    try:
        await h.run(P.write32(P.A_SCRATCH, 1, h.stats))
        assert False, "a failed exchange with DROPPED set was retried"
    except P.ProtocolFault as e:
        assert e.count == len(extra) and h.stats.packets == packets + 1
    h.t.inject = None
    # W1C clears flag and count together; the link is then clean again
    await h.run(P.write32(P.A_STATUS, P.ST_DROPPED, h.stats))
    st = await h.run(P.check_dropped(h.stats))
    assert P.dropped_count(st) == 0
    dut._log.info(f"{len(extra)} bytes sent mid-answer: answer intact, DROPPED_COUNT={len(extra)}, "
                  f"host raised ProtocolFault")


@cocotb.test()
async def test_corrupted_burst_in_a_layer_load(dut):
    """a flipped byte and a dropped byte inside burst writes of a real layer
    load: DCRC / TIMEOUT answers, retries, the write-CRC check forces a clean
    reload, and the layer still matches NumPy byte for byte"""
    h = await fresh(dut)
    for bank in range(h.hw.nbuf):
        await h.fill_bank(bank)
    layer = small_layer(h, kh=3, kw=3, stride=1, pad=1, h_in=4, w_in=4, cin=h.n, cout=h.n)
    burst_packets = []

    def hook(k, b):
        if b[1] == P.OP_BURST_WRITE and len(b) > 20:
            burst_packets.append(k)
            if len(burst_packets) == 1:
                return b[:15] + bytes((b[15] ^ 0x40,)) + b[16:]      # flip a data byte
            if len(burst_packets) == 3:
                return b[:14] + b[15:]                              # drop a data byte
        return b

    h.t.inject = hook
    before = dict(h.stats.statuses)
    await h.run_layer(layer, "layer with corrupted bursts")
    h.t.inject = None
    assert h.stats.statuses.get(P.ST_DCRC, 0) > before.get(P.ST_DCRC, 0), "no DCRC answer seen"
    assert h.stats.statuses.get(P.ST_TIMEOUT, 0) > before.get(P.ST_TIMEOUT, 0), "no TIMEOUT answer seen"
    assert h.stats.reloads >= 1, "the write-CRC check did not force a reload"


@cocotb.test()
async def test_start_exactly_once(dut):
    """the start packet's answer is lost (LAUNCH_COUNT moved: no resend), or
    the start packet itself is corrupted (it did not move: resend). either
    way the layer is launched exactly once"""
    h = await fresh(dut)
    layer = small_layer(h)
    await h.run(P.load(h.hw, layer_images(h.hw, layer), h.stats))
    cmd = layer.command(h.n)
    await h.run(P.write_command(cmd, h.stats))
    start_pkt = P.pkt_write32(P.A_CTRL, P.CTRL_START)
    for mode in ("lose the answer", "corrupt the request"):
        before = await h.run(P.read32(P.A_LAUNCH, h.stats))
        log0 = len(h.t.log)
        if mode == "lose the answer":
            h.t.garble = lambda k, r: b"" if h.t.log[-1][0] == start_pkt else r
        else:
            state = {"done": False}

            def once(k, b):
                if b == start_pkt and not state["done"]:
                    state["done"] = True
                    return b[:10] + bytes((b[10] ^ 1,)) + b[11:]
                return b
            h.t.inject = once
        await h.run(P.start(h.stats))
        h.t.garble = h.t.inject = None
        st, _ = await h.run(P.poll_done(h.stats))
        after = await h.run(P.read32(P.A_LAUNCH, h.stats))
        assert st & P.ST_DONE and after == before + 1, f"{mode}: LAUNCH_COUNT {before} -> {after}"
        wire = [pkt for pkt, _ in h.t.log[log0:]]
        good = sum(1 for pkt in wire if pkt == start_pkt)
        broken = sum(1 for pkt in wire if pkt != start_pkt and pkt[:9] == start_pkt[:9])
        # a lost answer must not cause a resend; a broken request must
        assert good == 1 and broken == (mode == "corrupt the request"), (mode, good, broken)
        dut._log.info(f"{mode}: launched once; on the wire {good} good start packet(s), "
                      f"{broken} corrupted")


@cocotb.test()
async def test_busy_refusals_and_soft_reset(dut):
    """while a layer runs the CSR refuses window and command writes (SLVERR),
    status reads work; a soft reset mid-layer clears busy, done and CYCLES
    without touching the link, and a clean rerun matches NumPy"""
    h = await fresh(dut)
    for bank in range(h.hw.nbuf):
        await h.fill_bank(bank)
    # a layer long enough to talk to while it runs
    layer = small_layer(h, kh=3, kw=3, stride=1, pad=1, h_in=8, w_in=8, cin=3 * h.n, cout=3 * h.n,
                        src_sel=0, dst_sel=1)
    await h.run(P.load(h.hw, layer_images(h.hw, layer), h.stats))
    h.note_input(layer)
    cmd = layer.command(h.n)
    await h.run(P.write_command(cmd, h.stats))
    await h.run(P.start(h.stats))
    st = await h.run(P.read32(P.A_STATUS, h.stats))
    assert st & P.ST_BUSY, "layer finished before the busy checks"
    for pkt in (P.pkt_write32(P.A_CMD0, 1), P.pkt_write32(P.A_CTRL, P.CTRL_HOST_MODE),
                P.pkt_burst_write(h.hw.b_addr(0), [1, 2]), P.pkt_write32(P.A_CTRL, P.CTRL_START)):
        s, _ = await h.raw(pkt)
        assert s == P.ST_SLVERR, f"accepted while busy: {pkt[1]:#x} -> {P.STATUS_NAMES.get(s, s)}"
    s, _ = await h.raw(P.pkt_burst_read(h.hw.act_addr(0, 0), 2), 2)
    assert s == P.ST_SLVERR
    launches = await h.run(P.read32(P.A_LAUNCH, h.stats))
    await h.run(P.write32(P.A_CTRL, P.CTRL_SOFT_RESET, h.stats))
    st = await h.run(P.read32(P.A_STATUS, h.stats))
    cyc = await h.run(P.read32(P.A_CYCLES, h.stats))
    assert (st & (P.ST_BUSY | P.ST_DONE | P.ST_ERR), cyc) == (0, 0), (hex(st), cyc)
    assert await h.run(P.read32(P.A_LAUNCH, h.stats)) == launches, "soft reset touched LAUNCH_COUNT"
    assert await h.run(P.read32(P.A_CMD0, h.stats)) == cmd["h_in"], "soft reset touched the command"
    # the destination may now hold any mix of old and new words; a clean
    # rerun must leave exactly the right output and nothing else changed
    await h.run_layer(layer, "rerun after soft reset", with_act=False)
