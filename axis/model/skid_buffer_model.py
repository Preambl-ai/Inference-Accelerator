# Golden model for rtl/skid_buffer.sv

from typing import NamedTuple


class SkidState(NamedTuple):
    """What is visible at the ReadOnly point after one rising edge."""
    s_ready: int
    m_valid: int
    m_data: int
    skid_valid: int
    skid_data: int


class SkidBufferModel:
    """Cycle-accurate model of the two-entry register slice."""

    def __init__(self, data_width: int = 8):
        if data_width < 1:
            raise ValueError(f"DATA_WIDTH must be >= 1 (got {data_width})")
        self.data_width = data_width
        self._clear()

    def _clear(self) -> None:
        # Register names match the RTL
        self.s_ready = 0
        self.m_valid = 0
        self.m_data = 0
        self.skid_valid = 0
        self.skid_data = 0

    def state(self) -> SkidState:
        return SkidState(self.s_ready, self.m_valid, self.m_data,
                         self.skid_valid, self.skid_data)

    def occupancy(self) -> int:
        return self.m_valid + self.skid_valid

    def step(self, s_valid: int = 0, s_data: int = 0, m_ready: int = 0,
             reset: int = 0) -> SkidState:
        """Advance one rising edge and return the ReadOnly view after it."""
        if not 0 <= s_data < (1 << self.data_width):
            raise ValueError(f"s_data={s_data} does not fit DATA_WIDTH={self.data_width}")
        if reset:
            self._clear()
            return self.state()

        accept = bool(s_valid and self.s_ready)
        if not self.m_valid or m_ready:
            if self.skid_valid:
                self.m_valid = 1
                self.m_data = self.skid_data
                self.skid_valid = 0
            else:
                self.m_valid = int(accept)
                if accept:
                    self.m_data = s_data
            self.s_ready = 1
        elif accept:
            self.skid_valid = 1
            self.skid_data = s_data
            self.s_ready = 0
        else:
            self.s_ready = int(not self.skid_valid)
        return self.state()

    def reset_cycles(self, cycles: int = 3) -> SkidState:
        for _ in range(cycles):
            self.step(reset=1)
        return self.state()


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

    m = SkidBufferModel(8)
    m.reset_cycles(2)
    chk("s_ready low in reset", m.state().s_ready, 0)
    chk("s_ready one edge after reset", m.step().s_ready, 1)

    # latency 1 into an empty buffer
    s = m.step(s_valid=1, s_data=11)
    chk("beat on m after one edge", (s.m_valid, s.m_data), (1, 11))

    # output blocked: second beat parks, door closes, third is refused
    s = m.step(s_valid=1, s_data=22, m_ready=0)
    chk("second beat parked", (s.skid_valid, s.skid_data, s.s_ready), (1, 22, 0))
    s = m.step(s_valid=1, s_data=33, m_ready=0)
    chk("third beat refused", (s.m_data, s.skid_data, s.s_ready), (11, 22, 0))
    chk("holds while blocked", m.step(s_valid=1, s_data=33, m_ready=0), s)

    # release: oldest first, then the parked one, door reopens
    s = m.step(s_valid=1, s_data=33, m_ready=1)
    chk("parked beat moves up", (s.m_valid, s.m_data, s.skid_valid, s.s_ready), (1, 22, 0, 1))
    s = m.step(s_valid=1, s_data=33, m_ready=1)
    chk("third beat follows", (s.m_valid, s.m_data), (1, 33))
    s = m.step(m_ready=1)
    chk("drains empty", (s.m_valid, s.s_ready), (0, 1))

    # full throughput: n beats in n + 1 edges when nothing blocks
    m.reset_cycles(2)
    m.step()
    out = []
    for i in range(1, 21):
        before = m.state()
        if before.m_valid:
            out.append(before.m_data)
        m.step(s_valid=1, s_data=i, m_ready=1)
    chk("one beat per cycle", out, list(range(1, 20)))

    # random traffic against a plain queue: nothing lost, repeated or reordered
    rng = random.Random(1234)
    for p_valid, p_ready in ((0.5, 0.5), (0.9, 0.2), (0.2, 0.9), (1.0, 0.5)):
        m = SkidBufferModel(16)
        m.reset_cycles(2)
        sent, got, max_occ = [], [], 0
        nxt, pending = 1, None
        for _ in range(5000):
            if pending is None and rng.random() < p_valid:
                pending, nxt = nxt, nxt + 1
            ready = int(rng.random() < p_ready)
            before = m.state()
            if before.m_valid and ready:
                got.append(before.m_data)
            took = pending is not None and before.s_ready
            m.step(s_valid=int(pending is not None), s_data=pending or 0, m_ready=ready)
            if took:
                sent.append(pending)
                pending = None
            max_occ = max(max_occ, m.occupancy())
        chk(f"p_valid={p_valid} p_ready={p_ready} order", got, sent[:len(got)])
        chk("  nothing stuck beyond two entries", len(sent) - len(got) <= 2, True)
        chk("  both entries used", max_occ, 2)

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")

    raise SystemExit(1 if fails else 0)
