"""Host <-> board protocol, shared by the real host driver and the simulations.

Everything that decides which bytes go on the wire lives here, once:
  - crc16, the one CRC-16/CCITT-FALSE implementation (check value 0x29B1)
  - packet builders and the response parser
  - the CSR map (addresses, command field order, memory windows and slots)
  - the layer sequences (load, verify write CRCs, start, poll, read back)

Contract: top/host_interface_contract.md.

The sequences are generators. They yield transport requests and get the
results sent back in, so the same code runs over pyserial (host/board.py,
plain loop) and over the cocotb UART BFM (top/tb, async loop):

    Xfer(packet, resp_len) -> the response bytes that arrived (may be short)
    WaitIdle()             -> None; let the bridge's timeout pass, drop input

Run one with run_sync(gen, transport) or `await run_async(gen, transport)`.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# CRC-16/CCITT-FALSE
# ---------------------------------------------------------------------------

def crc16(data, crc: int = 0xFFFF) -> int:
    """poly 0x1021, init 0xFFFF, MSB first, no reflection, no final XOR"""
    for b in data:
        crc ^= (b & 0xFF) << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


# ---------------------------------------------------------------------------
# packets
# ---------------------------------------------------------------------------

SYNC_IN, SYNC_OUT = 0xA5, 0x5A
OP_WRITE32, OP_READ32, OP_BURST_WRITE, OP_BURST_READ = 0x01, 0x02, 0x03, 0x04
MAX_BURST = 256

ST_OK, ST_SLVERR, ST_DECERR, ST_HCRC, ST_DCRC, ST_BADOP, ST_TIMEOUT = range(7)
STATUS_NAMES = {ST_OK: "OK", ST_SLVERR: "SLVERR", ST_DECERR: "DECERR", ST_HCRC: "HCRC",
                ST_DCRC: "DCRC", ST_BADOP: "BADOP", ST_TIMEOUT: "TIMEOUT"}
# link errors are worth a retry; bus errors are deterministic and are not
LINK_ERRORS = (ST_HCRC, ST_DCRC, ST_BADOP, ST_TIMEOUT)


def _le16(v): return bytes((v & 0xFF, (v >> 8) & 0xFF))
def _le32(v): return bytes((v >> (8 * i)) & 0xFF for i in range(4))


def header(op: int, addr: int, length: int) -> bytes:
    """A5 OP ADDR[4] LEN HCRC[2]; length is LEN = count - 1"""
    body = bytes((op,)) + _le32(addr) + bytes((length,))
    return bytes((SYNC_IN,)) + body + _le16(crc16(body))


def data_phase(words: Sequence[int]) -> bytes:
    raw = b"".join(_le32(w & 0xFFFFFFFF) for w in words)
    return raw + _le16(crc16(raw))


def pkt_write32(addr: int, value: int) -> bytes:
    return header(OP_WRITE32, addr, 0) + data_phase([value])


def pkt_read32(addr: int) -> bytes:
    return header(OP_READ32, addr, 0)


def pkt_burst_write(addr: int, words: Sequence[int]) -> bytes:
    assert 1 <= len(words) <= MAX_BURST
    return header(OP_BURST_WRITE, addr, len(words) - 1) + data_phase(words)


def pkt_burst_read(addr: int, count: int) -> bytes:
    assert 1 <= count <= MAX_BURST
    return header(OP_BURST_READ, addr, count - 1)


def response_len(read_words: int = 0) -> int:
    """5A [DATA] STATUS RCRC[2]"""
    return 1 + 4 * read_words + 1 + 2


def build_response(status: int, words: Sequence[int] = ()) -> bytes:
    """what a correct bridge sends; used by the models and self-tests"""
    body = b"".join(_le32(w) for w in words) + bytes((status,))
    return bytes((SYNC_OUT,)) + body + _le16(crc16(body))


class ProtocolError(Exception):
    """a response that is short, unsynchronized or fails its CRC"""


def parse_response(raw: Optional[bytes], read_words: int = 0) -> Tuple[int, List[int]]:
    want = response_len(read_words)
    if raw is None or len(raw) < want:
        raise ProtocolError(f"short response: {0 if raw is None else len(raw)} of {want} bytes")
    raw = bytes(raw[:want])
    if raw[0] != SYNC_OUT:
        raise ProtocolError(f"bad sync byte {raw[0]:#04x}")
    body, rcrc = raw[1:-2], raw[-2] | (raw[-1] << 8)
    if crc16(body) != rcrc:
        raise ProtocolError(f"response CRC {rcrc:#06x}, computed {crc16(body):#06x}")
    words = [int.from_bytes(body[4 * i:4 * i + 4], "little") for i in range(read_words)]
    return body[-1], words


# ---------------------------------------------------------------------------
# CSR map
# ---------------------------------------------------------------------------

ID_VALUE = 0x434E5631          # "CNV1"
VERSION = 0x00010000

A_ID, A_VERSION, A_PARAMS0, A_PARAMS1 = 0x00, 0x04, 0x08, 0x0C
A_CTRL, A_STATUS, A_ERR_CODE, A_CYCLES = 0x10, 0x14, 0x18, 0x1C
A_SCRATCH, A_BUILD_ID, A_ACT_WCRC, A_B_WCRC, A_P_WCRC, A_LAUNCH = 0x20, 0x24, 0x28, 0x2C, 0x30, 0x34
A_CMD0 = 0x40
CMD_FIELDS = ("h_in", "w_in", "h_out", "w_out", "cin_words", "cout_words", "kh", "kw",
              "stride", "pad", "zp_in", "zp_out", "relu_en", "src_sel", "dst_sel",
              "src_base", "dst_base", "b_base", "p_base")

CTRL_START, CTRL_SOFT_RESET, CTRL_HOST_MODE, CTRL_CLEAR_WCRC = 1, 2, 4, 8
ST_BUSY, ST_READY, ST_DONE, ST_ERR, ST_HOSTMODE = 1, 2, 4, 8, 16

ACT_BASE, B_BASE, P_BASE = 0x100000, 0x200000, 0x300000
P_WIDTH = 53


def slot_shift(width: int) -> int:
    slots = (width + 31) // 32
    return (slots - 1).bit_length() if slots > 1 else 0


def to_slots(word: int, width: int) -> List[int]:
    return [(word >> (32 * i)) & 0xFFFFFFFF for i in range(1 << slot_shift(width))]


def from_slots(slots: Sequence[int]) -> int:
    return sum((s & 0xFFFFFFFF) << (32 * i) for i, s in enumerate(slots))


def word_bytes(word: int, width: int) -> bytes:
    """the bytes a commit feeds into *_WCRC: ceil(width/8), LSB first"""
    return (word & ((1 << width) - 1)).to_bytes((width + 7) // 8, "little")


@dataclass
class HwInfo:
    """what PARAMS0/PARAMS1 say the bitstream was built with"""
    n: int
    read_latency: int
    addr_width: int
    nbuf: int
    b_addr_w: int
    p_addr_w: int
    k_max: int

    @classmethod
    def decode(cls, p0: int, p1: int) -> "HwInfo":
        return cls(n=p0 & 0xFF, read_latency=(p0 >> 8) & 0xF, addr_width=(p0 >> 12) & 0xFF,
                   nbuf=(p0 >> 20) & 0xF, b_addr_w=(p0 >> 24) & 0xFF, p_addr_w=p1 & 0xFF,
                   k_max=(p1 >> 8) & 0xFFFF)

    @property
    def word_w(self): return 8 * self.n

    @property
    def sel_width(self): return (self.nbuf - 1).bit_length() if self.nbuf > 1 else 1

    def act_addr(self, bank: int, index: int, slot: int = 0) -> int:
        word = (bank << self.addr_width) | index
        return ACT_BASE + (((word << slot_shift(self.word_w)) + slot) << 2)

    def b_addr(self, index: int, slot: int = 0) -> int:
        return B_BASE + (((index << slot_shift(self.word_w)) + slot) << 2)

    def p_addr(self, index: int, slot: int = 0) -> int:
        return P_BASE + (((index << slot_shift(P_WIDTH)) + slot) << 2)


# ---------------------------------------------------------------------------
# transport requests and drivers
# ---------------------------------------------------------------------------

@dataclass
class Xfer:
    packet: bytes
    resp_len: int


@dataclass
class WaitIdle:
    pass


class LinkError(Exception):
    """the link failed RETRIES times in a row"""


class BusError(Exception):
    """the CSR answered SLVERR or DECERR"""

    def __init__(self, status, addr):
        super().__init__(f"{STATUS_NAMES.get(status, status)} at {addr:#08x}")
        self.status, self.addr = status, addr


def run_sync(gen, transport):
    """drive a sequence over a blocking transport with .xfer() and .wait_idle()"""
    try:
        req = next(gen)
        while True:
            res = transport.xfer(req.packet, req.resp_len) if isinstance(req, Xfer) \
                else transport.wait_idle()
            req = gen.send(res)
    except StopIteration as stop:
        return stop.value


async def run_async(gen, transport):
    """the same, for an async transport (cocotb)"""
    try:
        req = next(gen)
        while True:
            res = await transport.xfer(req.packet, req.resp_len) if isinstance(req, Xfer) \
                else await transport.wait_idle()
            req = gen.send(res)
    except StopIteration as stop:
        return stop.value


RETRIES = 4


@dataclass
class LinkStats:
    """what went wrong on the way; the TB checks that injected errors show up here"""
    packets: int = 0
    retries: int = 0
    statuses: Dict[int, int] = field(default_factory=dict)
    protocol_errors: int = 0
    reloads: int = 0

    def saw(self, status):
        self.statuses[status] = self.statuses.get(status, 0) + 1


def transact(packet: bytes, read_words: int, stats: LinkStats, retry: bool = True):
    """one packet with retries on link errors. returns (status, words)"""
    for attempt in range(RETRIES if retry else 1):
        stats.packets += 1
        raw = yield Xfer(packet, response_len(read_words))
        try:
            status, words = parse_response(raw, read_words)
        except ProtocolError:
            stats.protocol_errors += 1
            yield WaitIdle()
            stats.retries += 1
            continue
        stats.saw(status)
        if status in LINK_ERRORS:
            yield WaitIdle()            # the bridge drains until the line is quiet
            stats.retries += 1
            continue
        return status, words
    if retry:
        raise LinkError(f"no good response after {RETRIES} tries")
    return None, []


def write32(addr, value, stats, check=True):
    status, _ = yield from transact(pkt_write32(addr, value), 0, stats)
    if check and status != ST_OK:
        raise BusError(status, addr)
    return status


def read32(addr, stats, check=True):
    status, words = yield from transact(pkt_read32(addr), 1, stats)
    if check and status != ST_OK:
        raise BusError(status, addr)
    return words[0] if status == ST_OK else status


def burst_write(addr, words, stats, check=True):
    """any length; split into MAX_BURST-word packets (all slot counts divide it)"""
    for i in range(0, len(words), MAX_BURST):
        chunk = words[i:i + MAX_BURST]
        status, _ = yield from transact(pkt_burst_write(addr + 4 * i, chunk), 0, stats)
        if check and status != ST_OK:
            raise BusError(status, addr + 4 * i)
    return ST_OK


def burst_read(addr, count, stats):
    out = []
    for i in range(0, count, MAX_BURST):
        n = min(MAX_BURST, count - i)
        status, words = yield from transact(pkt_burst_read(addr + 4 * i, n), n, stats)
        if status != ST_OK:
            raise BusError(status, addr + 4 * i)
        out += words
    return out


# ---------------------------------------------------------------------------
# layer sequences
# ---------------------------------------------------------------------------

def identify(stats):
    """first thing on any board: the ID register, then what it was built with"""
    ident = yield from read32(A_ID, stats)
    if ident != ID_VALUE:
        raise RuntimeError(f"ID register reads {ident:#010x}, expected {ID_VALUE:#010x} "
                           f"(wrong board or bitstream)")
    version = yield from read32(A_VERSION, stats)
    if version >> 16 != VERSION >> 16:
        raise RuntimeError(f"CSR map version {version:#010x}, this host speaks {VERSION:#010x}")
    p0 = yield from read32(A_PARAMS0, stats)
    p1 = yield from read32(A_PARAMS1, stats)
    build = yield from read32(A_BUILD_ID, stats)
    return HwInfo.decode(p0, p1), build


@dataclass
class MemImages:
    """what one layer writes, as (start address, 32-bit words) bursts, plus the
    write CRCs the board must end up with"""
    act: List[Tuple[int, List[int]]]
    b: List[Tuple[int, List[int]]]
    p: List[Tuple[int, List[int]]]
    act_crc: int
    b_crc: int
    p_crc: int


def memory_images(hw: HwInfo, act=(), b=(), p=()) -> MemImages:
    """act: [(bank, first word, words)], b and p: [(first word, words)], each
    written in the order given. the write CRCs run over all of it in that
    order, which is the order the board commits it."""
    def flat(words, width):
        out = []
        for w in words:
            out += to_slots(w, width)
        return out

    def crc_of(segments, width):
        c = 0xFFFF
        for words in segments:
            for w in words:
                c = crc16(word_bytes(w, width), c)
        return c

    return MemImages(
        act=[(hw.act_addr(bank, base), flat(words, hw.word_w)) for bank, base, words in act if words],
        b=[(hw.b_addr(base), flat(words, hw.word_w)) for base, words in b if words],
        p=[(hw.p_addr(base), flat(words, P_WIDTH)) for base, words in p if words],
        act_crc=crc_of([w for _, _, w in act], hw.word_w),
        b_crc=crc_of([w for _, w in b], hw.word_w),
        p_crc=crc_of([w for _, w in p], P_WIDTH))


LOAD_TRIES = 3


def load(hw: HwInfo, images: MemImages, stats):
    """clear the write CRCs, write ACT (in host mode), B and P, then check the
    CRCs the board computed against the host's.

    a burst that failed its DCRC and was retried has already committed its
    bad words, and the running CRC has those bytes in it for good. so a
    retry anywhere in a load makes the CRCs disagree even though memory is
    now right; the load is then repeated from a cleared CRC until one pass
    goes through clean."""
    for attempt in range(LOAD_TRIES):
        yield from write32(A_CTRL, CTRL_CLEAR_WCRC, stats)
        if images.act:
            yield from write32(A_CTRL, CTRL_HOST_MODE, stats)
            for addr, words in images.act:
                yield from burst_write(addr, words, stats)
            yield from write32(A_CTRL, 0, stats)
        for addr, words in images.b + images.p:
            yield from burst_write(addr, words, stats)
        got = []
        for a in (A_ACT_WCRC, A_B_WCRC, A_P_WCRC):
            got.append((yield from read32(a, stats)))
        want = [images.act_crc, images.b_crc, images.p_crc]
        if got == want:
            return attempt + 1
        stats.reloads += 1
    raise RuntimeError(f"write CRCs (act, b, p) on the board {[hex(g) for g in got]}, "
                       f"host computed {[hex(w) for w in want]}, after {LOAD_TRIES} loads")


def write_command(cmd: dict, stats):
    """all 19 fields in one burst: one round trip instead of 19 (FTDI latency)"""
    yield from burst_write(A_CMD0, [cmd[f] & 0xFFFFFFFF for f in CMD_FIELDS], stats)


def start(stats):
    """start exactly once. LAUNCH_COUNT before and after decides a retry."""
    before = yield from read32(A_LAUNCH, stats)
    for _ in range(RETRIES):
        status, _ = yield from transact(pkt_write32(A_CTRL, CTRL_START), 0, stats, retry=False)
        if status == ST_OK:
            return before + 1
        if status in (ST_SLVERR, ST_DECERR):
            raise BusError(status, A_CTRL)
        yield WaitIdle()
        after = yield from read32(A_LAUNCH, stats)
        if after != before:
            return after                 # it took; only the answer was lost
        stats.retries += 1
    raise LinkError("start did not take")


def poll_done(stats, max_polls: int = 100000):
    """STATUS until done or err. returns (status word, polls)"""
    for i in range(max_polls):
        st = yield from read32(A_STATUS, stats)
        if st & (ST_DONE | ST_ERR):
            return st, i + 1
    raise RuntimeError("layer did not finish")


def read_act(hw: HwInfo, bank: int, base: int, count: int, stats) -> List[int]:
    """count act words from bank, via burst reads of their slots"""
    s = 1 << slot_shift(hw.word_w)
    flat = yield from burst_read(hw.act_addr(bank, base), count * s, stats)
    return [from_slots(flat[i * s:(i + 1) * s]) for i in range(count)]


@dataclass
class LayerResult:
    status: int
    err_code: int
    cycles: int
    output: List[int]
    polls: int


def run_layer(hw: HwInfo, cmd: dict, images: MemImages, out_count: int, stats):
    """load, command, start, wait, read back. images may be partial (a chained
    layer loads no ACT); out_count = 0 skips the readback."""
    yield from load(hw, images, stats)
    yield from write_command(cmd, stats)
    yield from start(stats)
    st, polls = yield from poll_done(stats)
    err_code = yield from read32(A_ERR_CODE, stats)
    cycles = yield from read32(A_CYCLES, stats)
    out = []
    if out_count and st & ST_DONE:
        out = yield from read_act(hw, cmd["dst_sel"], cmd["dst_base"], out_count, stats)
    return LayerResult(st, err_code, cycles, out, polls)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fails = 0

    def chk(name, got, exp):
        global fails
        ok = got == exp
        print(f"{'pass' if ok else 'FAIL'} {name}" + ("" if ok else f": got {got!r}, expected {exp!r}"))
        fails += not ok

    chk("crc16 check value", crc16(b"123456789"), 0x29B1)
    chk("crc16 empty", crc16(b""), 0xFFFF)
    chk("crc16 incremental", crc16(b"6789", crc16(b"12345")), 0x29B1)
    h = header(OP_WRITE32, 0x00123456, 0)
    chk("header layout", list(h[:7]), [0xA5, 0x01, 0x56, 0x34, 0x12, 0x00, 0x00])
    chk("header crc covers op..len", h[7] | (h[8] << 8), crc16(h[1:7]))
    p = pkt_burst_write(0x100000, [1, 2, 3])
    chk("burst write length", len(p), 9 + 12 + 2)
    chk("burst LEN field", p[6], 2)
    r = build_response(ST_OK, [0xDEADBEEF, 7])
    chk("response round trip", parse_response(r, 2), (ST_OK, [0xDEADBEEF, 7]))
    bad = bytearray(r); bad[3] ^= 1
    try:
        parse_response(bytes(bad), 2); chk("corrupt response rejected", False, True)
    except ProtocolError:
        chk("corrupt response rejected", True, True)
    try:
        parse_response(r[:-1], 2); chk("short response rejected", False, True)
    except ProtocolError:
        chk("short response rejected", True, True)
    chk("slot shift 16/32/64/53", [slot_shift(w) for w in (16, 32, 64, 53)], [0, 0, 1, 1])
    chk("slots round trip", from_slots(to_slots(0x1122334455667788, 64)), 0x1122334455667788)
    chk("P bytes", len(word_bytes((1 << 53) - 1, P_WIDTH)), 7)
    hw = HwInfo.decode((13 << 24) | (2 << 20) | (12 << 12) | (1 << 8) | 8, (1024 << 8) | 6)
    chk("params decode", (hw.n, hw.read_latency, hw.addr_width, hw.nbuf, hw.b_addr_w,
                          hw.p_addr_w, hw.k_max), (8, 1, 12, 2, 13, 6, 1024))
    chk("act address bank 1 word 3 slot 1", hw.act_addr(1, 3, 1),
        ACT_BASE + ((((1 << 12) | 3) * 2 + 1) * 4))
    chk("command field count", len(CMD_FIELDS), 19)

    # the retry logic, against a scripted transport
    class Scripted:
        def __init__(self, answers): self.answers, self.sent, self.idles = list(answers), [], 0
        def xfer(self, pkt, n): self.sent.append(pkt); return self.answers.pop(0)
        def wait_idle(self): self.idles += 1

    st = LinkStats()
    t = Scripted([build_response(ST_DCRC), None, build_response(ST_OK)])
    run_sync(write32(A_SCRATCH, 5, st), t)
    chk("retry after DCRC and a lost answer", (len(t.sent), t.idles, st.retries), (3, 2, 2))
    chk("retries resend identical bytes", len(set(t.sent)), 1)
    st = LinkStats()
    t = Scripted([build_response(ST_SLVERR)])
    try:
        run_sync(write32(A_CTRL, 1, st), t); chk("SLVERR raises", False, True)
    except BusError:
        chk("SLVERR raises, no retry", len(t.sent), 1)
    # start: answer lost but LAUNCH_COUNT moved -> no second start
    st = LinkStats()
    t = Scripted([build_response(ST_OK, [5]), None, build_response(ST_OK, [6])])
    run_sync(start(st), t)
    starts = sum(1 for pkt in t.sent if pkt == pkt_write32(A_CTRL, CTRL_START))
    chk("start not repeated when LAUNCH_COUNT moved", starts, 1)
    st = LinkStats()
    t = Scripted([build_response(ST_OK, [5]), None, build_response(ST_OK, [5]), build_response(ST_OK)])
    run_sync(start(st), t)
    starts = sum(1 for pkt in t.sent if pkt == pkt_write32(A_CTRL, CTRL_START))
    chk("start repeated when LAUNCH_COUNT did not move", starts, 2)

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
