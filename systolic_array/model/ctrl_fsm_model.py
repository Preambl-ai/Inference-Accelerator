# Golden model for rtl/ctrl_fsm.sv

from typing import List, NamedTuple

from systolic_array_model import DelayLine, SystolicArrayModel, expected_matmul

S_IDLE, S_STREAM, S_FLUSH, S_DRAIN, S_DONE = range(5)
STATE_NAMES = ("IDLE", "STREAM", "FLUSH", "DRAIN", "DONE")


class CtrlState(NamedTuple):
    """Everything visible on the ports for a given stall, plus whitebox state."""
    ready: int
    busy: int
    done: int
    op_req: int
    k_idx: int
    arr_en: int
    arr_first: int
    arr_last: int
    arr_drain_shift: int
    c_valid: int
    c_row: int
    state: int
    k_lat: int
    last_seen: int
    wait_cnt: int


class CtrlFsmModel:
    """Cycle-accurate model of the tile sequencer."""

    def __init__(self, n: int = 8, k_width: int = 16, mem_latency: int = 1):
        if n < 1:
            raise ValueError(f"N must be >= 1 (got {n})")
        if mem_latency < 0:
            raise ValueError(f"MEM_LATENCY must be >= 0 (got {mem_latency})")
        if k_width < 1:
            raise ValueError(f"K_WIDTH must be >= 1 (got {k_width})")
        self.n = n
        self.k_width = k_width
        self.mem_latency = mem_latency
        self.wait_cycles = 2 * n
        self.wait_w = (self.wait_cycles).bit_length()      # $clog2(WAIT_CYCLES + 1)
        self.row_w = (n - 1).bit_length() if n > 1 else 1  # $clog2(N), min 1

        # delay.sv with DEPTH == 0 is a wire, so there is no line to model
        self.first_dly = DelayLine(mem_latency) if mem_latency else None
        self.last_dly = DelayLine(mem_latency) if mem_latency else None
        self._clear()

    def _clear(self) -> None:
        # Register names match the RTL so a waveform and a model trace line up
        self.state = S_IDLE
        self.k_lat = 0
        self.k_idx = 0
        self.op_req = 0
        self.raw_first = 0
        self.raw_last = 0
        self.last_seen = 0
        self.wait_cnt = 0
        self.arr_drain_shift = 0
        self.c_valid = 0
        self.c_row = 0
        self.done = 0

    def _first_dly(self) -> int:
        return self.first_dly.out() if self.first_dly else self.raw_first

    def _last_dly(self) -> int:
        return self.last_dly.out() if self.last_dly else self.raw_last

    def view(self, stall: int = 0) -> CtrlState:
        """Port values for the current registers and the given stall input.

        ready and arr_en are combinational in stall, so the ports are only
        defined once the stall for that cycle is known.
        """
        return CtrlState(
            ready=int(self.state == S_IDLE and not stall),
            busy=int(self.state != S_IDLE),
            done=self.done,
            op_req=self.op_req,
            k_idx=self.k_idx,
            arr_en=int(not stall),
            arr_first=self._first_dly(),
            arr_last=self._last_dly(),
            arr_drain_shift=self.arr_drain_shift,
            c_valid=self.c_valid,
            c_row=self.c_row,
            state=self.state,
            k_lat=self.k_lat,
            last_seen=self.last_seen,
            wait_cnt=self.wait_cnt,
        )

    def step(self, start: int = 0, k_in: int = 0, stall: int = 0,
             reset: int = 0) -> CtrlState:
        """Advance one rising edge and return the ReadOnly view after it."""
        if not 0 <= k_in < (1 << self.k_width):
            raise ValueError(f"k_in={k_in} does not fit K_WIDTH={self.k_width}. "
                             f"this is a stimulus bug, not a DUT bug")

        if reset:
            self._clear()
            for line in (self.first_dly, self.last_dly):
                if line:
                    line.step(0, reset=1)
            return self.view(stall)

        if stall:
            # every register holds, including the tag delay lines
            return self.view(stall)

        # all right-hand sides read the pre-edge values
        state = self.state
        k_idx = self.k_idx
        k_lat = self.k_lat
        c_row = self.c_row
        last_seen = self.last_seen
        wait_cnt = self.wait_cnt
        raw_first = self.raw_first
        raw_last = self.raw_last
        last_dly = self._last_dly()

        # wavefront timer
        if state != S_IDLE and last_dly:
            self.last_seen = 1
            self.wait_cnt = 0
        elif last_seen:
            self.wait_cnt = (wait_cnt + 1) & ((1 << self.wait_w) - 1)

        self.done = 0

        if state == S_IDLE:
            if start and k_in != 0:
                self.k_lat = k_in
                self.k_idx = 0
                self.op_req = 1
                self.raw_first = 1
                self.raw_last = int(k_in == 1)
                self.state = S_STREAM

        elif state == S_STREAM:
            self.raw_first = 0
            if k_idx == k_lat - 1:
                self.op_req = 0
                self.raw_last = 0
                self.state = S_FLUSH
            else:
                self.k_idx = (k_idx + 1) & ((1 << self.k_width) - 1)
                self.raw_last = int(k_idx + 1 == k_lat - 1)

        elif state == S_FLUSH:
            if last_seen and wait_cnt == self.wait_cycles - 1:
                self.state = S_DRAIN
                self.c_valid = 1
                self.c_row = (self.n - 1) & ((1 << self.row_w) - 1)
                self.arr_drain_shift = int(self.n > 1)
                # the case arm is later in the always block, so it wins over the timer
                self.last_seen = 0
                self.wait_cnt = 0

        elif state == S_DRAIN:
            if c_row == 0:
                self.c_valid = 0
                self.arr_drain_shift = 0
                self.done = 1
                self.state = S_DONE
            else:
                self.c_row = c_row - 1
                self.arr_drain_shift = int(c_row - 1 != 0)

        elif state == S_DONE:
            self.state = S_IDLE

        else:
            self.state = S_IDLE

        if self.first_dly:
            self.first_dly.step(raw_first)
            self.last_dly.step(raw_last)

        return self.view(stall)

    def reset_cycles(self, cycles: int = 3) -> CtrlState:
        for _ in range(cycles):
            self.step(reset=1)
        return self.view()


class OperandFeeder:
    """Stand-in for the operand memories: a lookup by k_idx followed by
    MEM_LATENCY stall-gated registers.

    Outside the stream it returns fresh nonzero garbage every cycle, so tags
    that arrive on the wrong edge pull a wrong value into the accumulators
    instead of a harmless zero.
    """

    def __init__(self, n: int, mem_latency: int, rng, data_width: int = 8):
        self.n = n
        self.mem_latency = mem_latency
        self.rng = rng
        self.hi = (1 << (data_width - 1)) - 1
        self.a_mat = None
        self.b_mat = None
        self.regs = [self._garbage() for _ in range(mem_latency)]

    def load(self, a_mat, b_mat) -> None:
        self.a_mat = a_mat
        self.b_mat = b_mat

    def _garbage(self):
        pick = lambda: self.rng.choice((-1, 1)) * self.rng.randint(1, self.hi)
        return ([pick() for _ in range(self.n)], [pick() for _ in range(self.n)])

    def _lookup(self, op_req: int, k_idx: int):
        if op_req and self.a_mat is not None and k_idx < len(self.a_mat[0]):
            return ([self.a_mat[r][k_idx] for r in range(self.n)],
                    [self.b_mat[k_idx][c] for c in range(self.n)])
        return self._garbage()

    def out(self, op_req: int, k_idx: int):
        """(a_in, b_in) at the array boundary for this cycle."""
        if self.mem_latency == 0:
            return self._lookup(op_req, k_idx)
        return self.regs[-1]

    def step(self, op_req: int, k_idx: int, stall: int = 0) -> None:
        if stall or self.mem_latency == 0:
            return
        self.regs = [self._lookup(op_req, k_idx)] + self.regs[:-1]


class TileCosim:
    """Feeder + mesh model of the array, driven by a controller's outputs.

    The controller can be the RTL or CtrlFsmModel; either way the results are
    judged against expected_matmul, which knows nothing about tags, flush
    waits or drain order. That is what catches an error the FSM and its own
    model share.
    """

    def __init__(self, n: int, mem_latency: int, rng,
                 data_width: int = 8, acc_width: int = 32):
        self.n = n
        self.array = SystolicArrayModel(n, data_width, acc_width)
        self.feeder = OperandFeeder(n, mem_latency, rng, data_width)
        self.rows = []          # (c_row, c_out) per c_valid & ~stall cycle
        self.done_events = 0    # done & ~stall cycles
        self.req_events = []    # k_idx per op_req & ~stall cycle

    def load(self, a_mat, b_mat) -> None:
        self.feeder.load(a_mat, b_mat)

    def clear_events(self) -> None:
        self.rows = []
        self.done_events = 0
        self.req_events = []

    def cycle(self, o, stall: int = 0, reset: int = 0) -> None:
        """One clock cycle. `o` holds the controller outputs during the cycle,
        i.e. what the array, feeder and result consumer sample at its edge."""
        if not reset and not stall:
            # events only count on an unstalled cycle
            if o.c_valid:
                self.rows.append((o.c_row, self.array.c_out()))
            if o.done:
                self.done_events += 1
            if o.op_req:
                self.req_events.append(o.k_idx)
        a_in, b_in = self.feeder.out(o.op_req, o.k_idx)
        self.array.step(a_in=a_in, b_in=b_in, first_in=o.arr_first,
                        last_in=o.arr_last, drain_shift=o.arr_drain_shift,
                        en=o.arr_en, reset=reset)
        if not reset:
            self.feeder.step(o.op_req, o.k_idx, stall)

    def result(self) -> List[List[int]]:
        """C as N x N, rebuilt from the tagged rows. Checks the drain order."""
        order = [row for row, _ in self.rows]
        if order != list(range(self.n - 1, -1, -1)):
            raise AssertionError(f"rows came out as {order}, expected N-1 down to 0")
        c = [None] * self.n
        for row, vals in self.rows:
            c[row] = list(vals)
        return c


def run_tile(fsm: CtrlFsmModel, cosim: TileCosim, a_mat, b_mat,
             stall_pattern=None, max_cycles: int = 1_000_000):
    """Model-only tile: start, run to ready, return (C, cycles)."""
    k = len(a_mat[0])
    cosim.load(a_mat, b_mat)
    cosim.clear_events()
    pattern = list(stall_pattern) if stall_pattern else []
    started = False
    for cycle in range(max_cycles):
        stall = int(pattern[cycle]) if cycle < len(pattern) else 0
        start = int(not started)
        o = fsm.view(stall)
        if started and o.ready:
            return cosim.result(), cycle
        cosim.cycle(o, stall)
        if start and o.ready:
            started = True
        fsm.step(start=start, k_in=k, stall=stall)
    raise AssertionError("tile never finished")


if __name__ == "__main__":
    import random

    fails = 0

    def chk(name, got, exp):
        global fails
        if got != exp:
            print(f"FAIL {name}\n  got={got}\n  exp={exp}")
            fails += 1
        else:
            print(f"pass {name}")

    rng = random.Random(1234)

    def mats(n, k):
        a = [[rng.randint(-128, 127) for _ in range(k)] for _ in range(n)]
        b = [[rng.randint(-128, 127) for _ in range(n)] for _ in range(k)]
        return a, b

    # end to end against the plain matmul, clean and stalled
    for n in (1, 2, 4, 8):
        for k in (1, 2, 7, 33):
            for ml in (0, 1, 2):
                a, b = mats(n, k)
                fsm = CtrlFsmModel(n=n, mem_latency=ml)
                cosim = TileCosim(n, ml, rng)
                c, cycles = run_tile(fsm, cosim, a, b)
                chk(f"N={n} K={k} ML={ml} tile", c, expected_matmul(a, b))
                # start accepted on cycle 0, ready again MEM_LATENCY+K+3N+1 edges later
                chk(f"N={n} K={k} ML={ml} length", cycles, ml + k + 3 * n + 2)
                chk(f"N={n} K={k} ML={ml} slices", cosim.req_events, list(range(k)))
                chk(f"N={n} K={k} ML={ml} done once", cosim.done_events, 1)

                a, b = mats(n, k)
                pattern = [rng.random() < 0.4 for _ in range(4 * (k + 3 * n + ml + 4))]
                c, _ = run_tile(fsm, cosim, a, b, stall_pattern=pattern)
                chk(f"N={n} K={k} ML={ml} stalled tile", c, expected_matmul(a, b))
                chk(f"N={n} K={k} ML={ml} stalled done once", cosim.done_events, 1)

    # hand-traced port sequence: N=2, K=2, MEM_LATENCY=1
    m = CtrlFsmModel(n=2, mem_latency=1)
    m.reset_cycles(2)
    chk("idle ready", m.view().ready, 1)
    chk("ready low under stall", m.view(stall=1).ready, 0)
    trace = [m.step(start=1, k_in=2)] + [m.step() for _ in range(10)]
    chk("op_req", [s.op_req for s in trace], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    chk("k_idx", [s.k_idx for s in trace[:2]], [0, 1])
    chk("arr_first", [s.arr_first for s in trace], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    chk("arr_last", [s.arr_last for s in trace], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    chk("c_valid", [s.c_valid for s in trace], [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
    chk("c_row", [s.c_row for s in trace[7:9]], [1, 0])
    chk("drain_shift", [s.arr_drain_shift for s in trace], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    chk("done", [s.done for s in trace], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    chk("busy", [s.busy for s in trace], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])

    # k_in == 0 is ignored, and so is a start during stall or while busy
    m = CtrlFsmModel(n=2, mem_latency=0)
    m.reset_cycles(2)
    chk("k_in=0 ignored", m.step(start=1, k_in=0).state, S_IDLE)
    chk("start under stall ignored", m.step(start=1, k_in=3, stall=1).state, S_IDLE)
    m.step(start=1, k_in=3)
    chk("start while busy keeps k_lat", m.step(start=1, k_in=9).k_lat, 3)

    # stall holds everything, reset wins over stall
    before = m.view(stall=1)
    chk("stall holds", m.step(stall=1), before)
    chk("reset beats stall", m.step(stall=1, reset=1).state, S_IDLE)

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")

    raise SystemExit(1 if fails else 0)
