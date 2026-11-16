# Golden model for rtl/bram.sv

UNKNOWN = None   # a value the RTL does not define; the testbench must not check it


class BramModel:
    """Cycle-accurate model of the simple dual-port RAM.

    Unwritten words, rdata before the first read has come through, and a read
    that collides with a write to the same address are all UNKNOWN.
    """

    def __init__(self, data_width: int = 64, addr_width: int = 10,
                 read_latency: int = 1):
        if read_latency not in (1, 2):
            raise ValueError(f"READ_LATENCY must be 1 or 2 (got {read_latency})")
        if data_width < 1 or addr_width < 1:
            raise ValueError("degenerate width parameter")
        self.data_width = data_width
        self.addr_width = addr_width
        self.read_latency = read_latency
        self.depth = 1 << addr_width
        self.mem = {}
        # regs[0] is the read register, regs[1] the optional output register
        self.regs = [UNKNOWN] * read_latency

    def rdata(self):
        return self.regs[-1]

    def step(self, we: int = 0, waddr: int = 0, wdata: int = 0,
             re: int = 0, raddr: int = 0):
        """Advance one rising edge and return rdata after it."""
        for name, addr in (("waddr", waddr), ("raddr", raddr)):
            if not 0 <= addr < self.depth:
                raise ValueError(f"{name}={addr} outside the memory. "
                                 f"this is a stimulus bug, not a DUT bug")
        if not 0 <= wdata < (1 << self.data_width):
            raise ValueError(f"wdata={wdata} does not fit DATA_WIDTH={self.data_width}")

        if re:
            # the read sees the array as it was before this edge's write
            word = UNKNOWN if (we and waddr == raddr) else self.mem.get(raddr, UNKNOWN)
            self.regs = [word] + self.regs[:-1]
        if we:
            self.mem[waddr] = wdata
        return self.rdata()


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

    for lat in (1, 2):
        m = BramModel(data_width=8, addr_width=4, read_latency=lat)
        chk(f"L={lat} rdata unknown at start", m.rdata(), UNKNOWN)
        m.step(we=1, waddr=3, wdata=0xA5)
        m.step(we=1, waddr=4, wdata=0x5A)
        chk(f"L={lat} unwritten word unknown",
            [m.step(re=1, raddr=9) for _ in range(lat)][-1], UNKNOWN)

        # latency: the word appears READ_LATENCY enabled edges after raddr
        seq = [m.step(re=1, raddr=3), m.step(re=1, raddr=4), m.step(re=1, raddr=4)]
        chk(f"L={lat} latency", seq[lat - 1], 0xA5)
        chk(f"L={lat} next word follows", seq[lat], 0x5A)

        # re low holds rdata and the word in flight
        m.step(re=1, raddr=3)
        held = m.step(re=1, raddr=4)
        chk(f"L={lat} hold", [m.step(re=0, raddr=9) for _ in range(3)], [held] * 3)
        # L=1: the new read lands at once. L=2: the word that was in flight
        # (address 4) comes out first, so the stall did not drop it.
        chk(f"L={lat} resumes in order", m.step(re=1, raddr=3), 0xA5 if lat == 1 else 0x5A)

        # a write while re is low still lands
        m.step(we=1, waddr=7, wdata=0x77, re=0, raddr=7)
        chk(f"L={lat} write with re low",
            [m.step(re=1, raddr=7) for _ in range(lat)][-1], 0x77)

        # collision is unknown, but the write lands
        got = [m.step(we=1, waddr=7, wdata=0x11, re=1, raddr=7)]
        got += [m.step(re=1, raddr=7) for _ in range(lat)]
        chk(f"L={lat} collision read unknown", got[lat - 1], UNKNOWN)
        chk(f"L={lat} collision write lands", got[lat], 0x11)

    # random traffic against a plain dict and an explicit delay
    rng = random.Random(1234)
    for lat in (1, 2):
        m = BramModel(data_width=16, addr_width=5, read_latency=lat)
        ref, pipe, bad = {}, [UNKNOWN] * lat, 0
        for _ in range(5000):
            we, re = rng.random() < 0.5, rng.random() < 0.7
            wa, ra, wd = rng.randrange(32), rng.randrange(32), rng.randrange(1 << 16)
            if re:
                pipe = [UNKNOWN if (we and wa == ra) else ref.get(ra)] + pipe[:-1]
            if we:
                ref[wa] = wd
            bad += int(m.step(we=we, waddr=wa, wdata=wd, re=re, raddr=ra) != pipe[-1])
        chk(f"L={lat} random traffic mismatches", bad, 0)

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")

    raise SystemExit(1 if fails else 0)
