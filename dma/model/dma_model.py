"""Golden model of wdma (dma/rtl/wdma.sv, contract dma/dma_contract.md).

fetch(mem, cmd, n, b_aw, p_aw) returns what the fetch must do: the error
code, the AXI bursts in order, and every B and P write. `mem` is the AXI
memory as a bytes-like object (or anything indexable by byte address).

The burst plan is computed by a beat-by-beat walk (plan_bursts) and checked
in the self-tests against an independent closed form (split_closed), so the
TB's expectation does not share code with itself.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

E_DIM, E_ALIGN, E_RANGE, E_RESP = 1, 2, 4, 8
OKAY, SLVERR, DECERR = 0, 2, 3
P_WIDTH = 53


@dataclass
class Fetch:
    b_src: int
    p_src: int
    k: int
    groups: int
    b_dst: int
    p_dst: int


@dataclass
class Result:
    err: int = 0
    bursts: List[Tuple[str, int, int]] = field(default_factory=list)   # (kind, addr, beats)
    b_writes: List[Tuple[int, int]] = field(default_factory=list)      # (waddr, word)
    p_writes: List[Tuple[int, int]] = field(default_factory=list)
    groups_done: int = 0


def plan_bursts(addr, beats):
    """split `beats` 8-byte beats from `addr` into INCR bursts: at most 256
    beats, never crossing a 4 KB line. walks one beat at a time"""
    out, start, n = [], addr, 0
    for i in range(beats):
        a = addr + 8 * i
        if n and (n == 256 or a % 4096 == 0):
            out.append((start, n))
            start, n = a, 0
        n += 1
    if n:
        out.append((start, n))
    return out


def split_closed(addr, beats):
    """the same split in closed form (what the RTL computes per burst)"""
    out = []
    while beats:
        b = min(beats, 256, (4096 - addr % 4096) // 8)
        out.append((addr, b))
        addr, beats = addr + 8 * b, beats - b
    return out


def beat(mem, addr):
    return int.from_bytes(bytes(mem[addr:addr + 8]), "little")


def fetch(mem, cmd: Fetch, n, b_aw, p_aw, bad=lambda addr: False) -> Result:
    """bad(addr) -> True where the slave answers a beat with an error"""
    r = Result()
    if cmd.groups == 0 or cmd.k == 0:
        r.err |= E_DIM
    if cmd.b_src % 8 or cmd.p_src % 8:
        r.err |= E_ALIGN
    if r.err:
        return r
    b_addr, p_addr, b_ptr, p_ptr = cmd.b_src, cmd.p_src, cmd.b_dst, cmd.p_dst
    for og in range(cmd.groups):
        if b_ptr + cmd.k > 1 << b_aw or p_ptr + n > 1 << p_aw:
            r.err = E_RANGE
            return r
        for kind, beats in (("b", cmd.k), ("p", n)):
            addr = b_addr if kind == "b" else p_addr
            for a, cnt in split_closed(addr, beats):
                r.bursts.append((kind, a, cnt))
                failed = False
                for i in range(cnt):
                    ba = a + 8 * i
                    if bad(ba):
                        failed = True
                    if failed:
                        continue                 # drained, never written
                    v = beat(mem, ba)
                    if kind == "b":
                        r.b_writes.append((b_ptr + (ba - b_addr) // 8, v & ((1 << (8 * n)) - 1)))
                    else:
                        r.p_writes.append((p_ptr + (ba - p_addr) // 8, v & ((1 << P_WIDTH) - 1)))
                if failed:
                    r.err = E_RESP
                    return r
            if kind == "b":
                b_addr += 8 * beats
                b_ptr += beats
            else:
                p_addr += 8 * beats
                p_ptr += n
        r.groups_done += 1
    return r


def estimate_cycles(res: Result, latency):
    """edges from the accepting edge to done, with a slave that takes AR at
    once and puts the first beat up `latency` edges after the AR handshake,
    then one per cycle. measured in the TB; see the contract"""
    per_burst = sum(1 + latency + beats for _, _, beats in res.bursts)   # SEG edge, AR..first beat, beats
    return 1 + res.groups_done + per_burst


if __name__ == "__main__":
    import random
    fails = 0

    def chk(name, got, exp):
        global fails
        ok = got == exp
        print(f"{'pass' if ok else 'FAIL'} {name}" + ("" if ok else f": got {got}, expected {exp}"))
        fails += not ok

    chk("one burst when it fits", plan_bursts(0x1000, 72), [(0x1000, 72)])
    chk("256-beat limit", plan_bursts(0, 300), [(0, 256), (2048, 44)])
    chk("4 KB line", plan_bursts(4096 - 16, 5), [(4080, 2), (4096, 3)])
    chk("256 limit then 4 KB line", plan_bursts(8, 600), [(8, 256), (2056, 255), (4096, 89)])
    rng = random.Random(1)
    for i in range(2000):
        a, b = 8 * rng.randrange(0, 4096), rng.randint(1, 1100)
        if plan_bursts(a, b) != split_closed(a, b):
            chk(f"walk and closed form agree at {a:#x}+{b}", plan_bursts(a, b), split_closed(a, b))
            break
    else:
        chk("walk and closed form agree on 2000 random splits", True, True)
    for a, b in ((0, 1), (4088, 1), (4088, 2), (0, 1024), (2048, 512)):
        bs = plan_bursts(a, b)
        chk(f"bursts legal at {a:#x}+{b}", all(c <= 256 and (s % 4096) + 8 * c <= 4096 for s, c in bs)
            and sum(c for _, c in bs) == b, True)

    mem = bytes(rng.getrandbits(8) for _ in range(1 << 16))
    n = 8
    f = Fetch(b_src=0x100, p_src=0x8000, k=9, groups=3, b_dst=5, p_dst=8)
    r = fetch(mem, f, n, 13, 8)
    chk("fetch ok", r.err, 0)
    chk("B words land at b_dst + og*K + i", [a for a, _ in r.b_writes], list(range(5, 5 + 27)))
    chk("B word i is beat i", r.b_writes[10][1], beat(mem, 0x100 + 80))
    chk("P entries land at p_dst + og*N + j", [a for a, _ in r.p_writes], list(range(8, 8 + 24)))
    chk("P keeps 53 bits", r.p_writes[0][1], beat(mem, 0x8000) & ((1 << 53) - 1))
    chk("one B and one P burst per group", [(k, c) for k, _, c in r.bursts], [("b", 9), ("p", 8)] * 3)
    chk("N=2 B word is 2 bytes", fetch(mem, Fetch(0, 64, 4, 1, 0, 0), 2, 13, 8).b_writes[1][1],
        beat(mem, 8) & 0xFFFF)
    chk("E_DIM", fetch(mem, Fetch(0, 0, 0, 1, 0, 0), n, 13, 8).err, E_DIM)
    chk("E_DIM groups", fetch(mem, Fetch(0, 0, 4, 0, 0, 0), n, 13, 8).err, E_DIM)
    chk("E_ALIGN", fetch(mem, Fetch(4, 0, 4, 1, 0, 0), n, 13, 8).err, E_ALIGN)
    chk("E_DIM | E_ALIGN", fetch(mem, Fetch(0, 12, 0, 1, 0, 0), n, 13, 8).err, E_DIM | E_ALIGN)
    r = fetch(mem, Fetch(0, 0x8000, 100, 5, 8192 - 250, 0), n, 13, 8)
    chk("E_RANGE in group 2: groups 0 and 1 written", (r.err, r.groups_done, len(r.b_writes)),
        (E_RANGE, 2, 200))
    chk("E_RANGE on P", fetch(mem, Fetch(0, 0, 4, 2, 0, 250), n, 13, 8).err, E_RANGE)
    r = fetch(mem, Fetch(0, 0x8000, 20, 3, 0, 0), n, 13, 8, bad=lambda a: a == 8 * 25)
    chk("E_RESP: group 0 whole, group 1 up to the bad beat", (r.err, r.groups_done, len(r.b_writes)),
        (E_RESP, 1, 25))
    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
