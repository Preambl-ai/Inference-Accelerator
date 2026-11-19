# Golden model for rtl/act_buffer.sv. The contract is memory/act_buffer_contract.md.

from bram_model import BramModel, UNKNOWN


def sel_width(nbuf: int) -> int:
    """Width of the select ports: at least 1, as in the RTL."""
    return max(1, (nbuf - 1).bit_length())


def pack_lanes(values, data_width: int) -> int:
    """N lane values (signed or unsigned) to one word, lane 0 in the LSBs."""
    mask = (1 << data_width) - 1
    word = 0
    for i, v in enumerate(values):
        word |= (int(v) & mask) << (i * data_width)
    return word


def unpack_lanes(word: int, n: int, data_width: int, signed: bool = True):
    """One word to N lane values, lane 0 from the LSBs."""
    mask, sign = (1 << data_width) - 1, 1 << (data_width - 1)
    vals = [(word >> (i * data_width)) & mask for i in range(n)]
    if signed:
        vals = [v - (1 << data_width) if v & sign else v for v in vals]
    return vals


class ActBufferModel:
    """Cycle-accurate model of the activation buffer.

    One BramModel per bank, so each bank keeps bram's own rules: a read that
    advances on the edge where that bank writes the same address is UNKNOWN,
    the write lands, and nothing else interacts. Every bank sees the same
    read enable and address, exactly as in the RTL; sel_pipe carries the
    select beside the data and moves on the same enable.

    rd_data is UNKNOWN before the first read has come through, for unwritten
    words, for collisions, and for a read whose select was not a bank.

    A select >= NBUF on the side in use is illegal and raises ValueError,
    standing in for the RTL's $fatal. With allow_bad_sel (the RTL's
    +act_buffer_allow_bad_sel) it is modelled as the hardware does it: the
    write is dropped and the read slot is UNKNOWN.
    """

    def __init__(self, n: int = 8, data_width: int = 8, addr_width: int = 8,
                 read_latency: int = 1, nbuf: int = 3, allow_bad_sel: bool = False):
        if n < 1 or data_width < 1:
            raise ValueError("N and DATA_WIDTH must be >= 1")
        if nbuf < 1:
            raise ValueError(f"NBUF must be >= 1 (got {nbuf})")
        self.n = n
        self.data_width = data_width
        self.word_width = n * data_width
        self.addr_width = addr_width
        self.read_latency = read_latency
        self.nbuf = nbuf
        self.sel_width = sel_width(nbuf)
        self.allow_bad_sel = allow_bad_sel
        # BramModel checks READ_LATENCY and the widths
        self.banks = [BramModel(self.word_width, addr_width, read_latency) for _ in range(nbuf)]
        self.sel_pipe = [UNKNOWN] * read_latency
        self.bad_reads = 0     # advancing reads with a select that is not a bank
        self.bad_writes = 0    # writes dropped for the same reason

    def rdata(self):
        sel = self.sel_pipe[-1]
        if sel is UNKNOWN or sel >= self.nbuf:
            return UNKNOWN
        return self.banks[sel].rdata()

    def peek(self, sel: int, addr: int):
        """Backdoor read of a stored word, or UNKNOWN."""
        return self.banks[sel].mem.get(addr, UNKNOWN)

    def _check_sel(self, name: str, sel: int) -> bool:
        """True if sel is a bank. Raises for an illegal select unless allowed."""
        if not 0 <= sel < (1 << self.sel_width):
            raise ValueError(f"{name}={sel} does not fit SEL_WIDTH={self.sel_width}. "
                             f"this is a stimulus bug, not a DUT bug")
        if sel < self.nbuf:
            return True
        if not self.allow_bad_sel:
            raise ValueError(f"{name}={sel} is not a bank (NBUF={self.nbuf}); "
                             f"the RTL stops with $fatal here")
        return False

    def step(self, rd_en: int = 0, rd_addr: int = 0, rd_sel: int = 0,
             we: int = 0, waddr: int = 0, wdata: int = 0, wr_sel: int = 0,
             host_mode: int = 0, host_we: int = 0, host_waddr: int = 0,
             host_wdata: int = 0, host_wr_sel: int = 0):
        """Advance one rising edge and return rd_data after it."""
        # the host_mode mux: only the side in use is looked at, the other may
        # carry anything (its select is not even range-checked, as in the RTL)
        if host_mode:
            w_en, w_addr, w_data, w_sel = host_we, host_waddr, host_wdata, host_wr_sel
        else:
            w_en, w_addr, w_data, w_sel = we, waddr, wdata, wr_sel
        advance = bool(rd_en) and not host_mode

        w_ok = self._check_sel("write sel", w_sel) if w_en else False
        if w_en and not w_ok:
            self.bad_writes += 1
        if advance and not self._check_sel("rd_sel", rd_sel):
            self.bad_reads += 1

        for b, bank in enumerate(self.banks):
            bank.step(we=int(w_ok and w_sel == b), waddr=w_addr, wdata=w_data,
                      re=int(advance), raddr=rd_addr)
        if advance:
            self.sel_pipe = [rd_sel] + self.sel_pipe[:-1]
        return self.rdata()


if __name__ == "__main__":
    import random

    fails = 0

    def chk(name, got, exp):
        global fails
        if got != exp:
            print(f"FAIL {name:<52} got={got} exp={exp}")
            fails += 1
        else:
            print(f"pass {name:<52} = {got}")

    def fmt(v):
        return v if v is UNKNOWN else hex(v)

    # packing: lane 0 in the LSBs, signed min/max per lane, odd widths
    chk("pack lane 0 is LSBs", pack_lanes([1, 2], 8), 0x0201)
    chk("pack signed", pack_lanes([-1, -128, 127], 8), 0x7F80FF)
    chk("unpack signed", unpack_lanes(0x7F80FF, 3, 8), [-1, -128, 127])
    chk("odd width round trip", unpack_lanes(pack_lanes([-16, 15, -1, 0], 5), 4, 5), [-16, 15, -1, 0])
    chk("sel width NBUF=1,2,3,4,5", [sel_width(b) for b in (1, 2, 3, 4, 5)], [1, 1, 2, 2, 3])

    for lat in (1, 2):
        m = ActBufferModel(n=2, data_width=8, addr_width=3, read_latency=lat, nbuf=3)
        chk(f"L={lat} rd_data unknown at start", m.rdata(), UNKNOWN)
        for b in range(3):
            for a in range(4):
                m.step(we=1, wr_sel=b, waddr=a, wdata=0x1000 * (b + 1) + a)

        # latency, and the select travelling with its word
        seq = [m.step(rd_en=1, rd_sel=1, rd_addr=3)] + \
              [m.step(rd_en=1, rd_sel=2, rd_addr=0) for _ in range(2)]
        chk(f"L={lat} latency", fmt(seq[lat - 1]), hex(0x2003))
        chk(f"L={lat} select follows its word", fmt(seq[lat]), hex(0x3000))

        # a new bank every cycle
        out = [m.step(rd_en=1, rd_sel=i % 3, rd_addr=i % 4) for i in range(6 + lat)]
        chk(f"L={lat} select every cycle", [fmt(v) for v in out[lat - 1:lat - 1 + 6]],
            [hex(0x1000 * (i % 3 + 1) + i % 4) for i in range(6)])

        # rd_en low: rd_data and the word in flight hold
        m.step(rd_en=1, rd_sel=0, rd_addr=1)
        held = m.step(rd_en=1, rd_sel=1, rd_addr=2)
        chk(f"L={lat} stall holds", [m.step(rd_en=0, rd_sel=2, rd_addr=3) for _ in range(3)], [held] * 3)
        chk(f"L={lat} resume in order", fmt(m.step(rd_en=1, rd_sel=2, rd_addr=3)),
            hex(0x3003 if lat == 1 else 0x2002))

        # host_mode: reads freeze, datapath write ignored, host write lands
        before = [m.step(rd_en=1, rd_sel=0, rd_addr=1) for _ in range(lat)][-1]
        during = [m.step(host_mode=1, rd_en=1, rd_sel=1, rd_addr=2,
                         we=1, wr_sel=0, waddr=1, wdata=0xDEAD,
                         host_we=1, host_wr_sel=0, host_waddr=2, host_wdata=0xBEEF)
                  for _ in range(2)]
        chk(f"L={lat} host_mode holds rd_data", during, [before, before])
        chk(f"L={lat} datapath write ignored in host_mode", m.peek(0, 1), 0x1001)
        chk(f"L={lat} host write lands", m.peek(0, 2), 0xBEEF)
        chk(f"L={lat} host port ignored outside host_mode",
            (m.step(host_we=1, host_wr_sel=0, host_waddr=3, host_wdata=0x1111), m.peek(0, 3))[1], 0x1003)

        # same bank, same address: read unknown, write lands
        got = [m.step(rd_en=1, rd_sel=0, rd_addr=1, we=1, wr_sel=0, waddr=1, wdata=0x7777)]
        got += [m.step(rd_en=1, rd_sel=0, rd_addr=1) for _ in range(lat)]
        chk(f"L={lat} collision read unknown", got[lat - 1], UNKNOWN)
        chk(f"L={lat} collision write lands", fmt(got[lat]), hex(0x7777))
        # same address, other bank: defined, old data
        got = [m.step(rd_en=1, rd_sel=0, rd_addr=2, we=1, wr_sel=1, waddr=2, wdata=0x5555)]
        got += [m.step(rd_en=1, rd_sel=1, rd_addr=2) for _ in range(lat)]
        chk(f"L={lat} other-bank same address is defined", fmt(got[lat - 1]), hex(0xBEEF))
        chk(f"L={lat} other-bank write lands", fmt(got[lat]), hex(0x5555))

        # illegal select raises unless allowed
        try:
            m.step(rd_en=1, rd_sel=3)
            chk(f"L={lat} bad rd_sel raises", "no error", "ValueError")
        except ValueError:
            chk(f"L={lat} bad rd_sel raises", "ValueError", "ValueError")
        try:
            m.step(host_mode=1, host_we=1, host_wr_sel=3)
            chk(f"L={lat} bad host sel raises", "no error", "ValueError")
        except ValueError:
            chk(f"L={lat} bad host sel raises", "ValueError", "ValueError")
        # junk on the idle side is fine
        m.step(we=1, wr_sel=3, host_mode=1)
        m.step(host_we=1, host_wr_sel=3)

        a = ActBufferModel(n=1, data_width=4, addr_width=2, read_latency=lat, nbuf=3, allow_bad_sel=True)
        for b in range(3):
            a.step(we=1, wr_sel=b, waddr=0, wdata=b + 1)
        a.step(we=1, wr_sel=3, waddr=0, wdata=0xF)
        chk(f"L={lat} bad-sel write dropped", [a.peek(b, 0) for b in range(3)], [1, 2, 3])
        out = [a.step(rd_en=1, rd_sel=s, rd_addr=0) for s in (0, 3, 1)]
        out += [a.step(rd_en=1, rd_sel=2, rd_addr=0) for _ in range(lat)]
        chk(f"L={lat} bad-sel read unknown, neighbours aligned",
            out[lat - 1:lat + 2], [1, UNKNOWN, 2])

        one = ActBufferModel(n=2, data_width=8, addr_width=2, read_latency=lat, nbuf=1)
        one.step(we=1, wr_sel=0, waddr=1, wdata=0xA5A5)
        chk(f"L={lat} NBUF=1 reads its one bank",
            fmt([one.step(rd_en=1, rd_sel=0, rd_addr=1) for _ in range(lat)][-1]), hex(0xA5A5))

    # random traffic against a plain dict per bank and an explicit delay line
    rng = random.Random(1234)
    for nbuf in (1, 2, 3, 4):
        for lat in (1, 2):
            m = ActBufferModel(n=3, data_width=5, addr_width=3, read_latency=lat, nbuf=nbuf)
            ref = [dict() for _ in range(nbuf)]
            pipe, frozen_ok, bad = [UNKNOWN] * lat, True, 0
            for _ in range(4000):
                host = int(rng.random() < 0.15)
                v = dict(rd_en=int(rng.random() < 0.7), rd_addr=rng.randrange(8), rd_sel=rng.randrange(nbuf),
                         we=int(rng.random() < 0.5), waddr=rng.randrange(8), wdata=rng.randrange(1 << 15),
                         wr_sel=rng.randrange(nbuf), host_mode=host, host_we=int(rng.random() < 0.5),
                         host_waddr=rng.randrange(8), host_wdata=rng.randrange(1 << 15),
                         host_wr_sel=rng.randrange(nbuf))
                pre = "host_" if host else ""
                w_en, w_sel, w_a, w_d = v[pre + "we"], v[pre + "wr_sel"], v[pre + "waddr"], v[pre + "wdata"]
                if v["rd_en"] and not host:
                    collide = w_en and w_sel == v["rd_sel"] and w_a == v["rd_addr"]
                    pipe = [UNKNOWN if collide else ref[v["rd_sel"]].get(v["rd_addr"])] + pipe[:-1]
                if w_en:
                    ref[w_sel][w_a] = w_d
                bad += int(m.step(**v) != pipe[-1])
            chk(f"NBUF={nbuf} L={lat} random traffic mismatches", bad, 0)

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")

    raise SystemExit(1 if fails else 0)
