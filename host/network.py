"""Random multi-layer networks for the descriptor sequencer.

Used by host/run_network.py (the board), the sequencer TB and the top-level
end-to-end TB, so all three build programs the same way. Random int8 weights
and activations only; no trained model.

A network is a chain of layers on one running tensor:
  conv    3x3 / 1x1 (same padding), stride 1 or 2, any channel count
  conv5   5x5, no padding (LeNet-5)
  fc      a fully connected layer: a valid conv whose kernel is the whole
          h x w input, so the output is 1x1 (LeNet's F5/F6/output)
  max2    2x2 max pool
  res     a ResNet basic block: conv3x3 -> conv3x3 -> add(skip), ReLU
  gavg    global average pool (only when h*w is a power of two), always last

Every layer writes a fresh region that no later layer overwrites, so every
layer's output can be read back and compared at the end. The expected words
come from sequencer_model.run (conv_ref / pool / add references through the
layer_engine bank model).
"""

import copy
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

import add_model as AM
from conv_layer_model import HwParams, Layer, k_of, pack_act, pack_b, pack_p, requant_params, words
from layer_engine_model import (OP_ADD, OP_CONV, OP_GAVG, OP_MAX2, Banks, add_fields, conv_bus,
                                pool_fields)
import protocol as P
from sequencer_model import OP_END, SeqResult, Step, run


@dataclass
class Network:
    steps: List[Step]
    names: List[str]
    descriptors: List[List[int]]
    input: Tuple[int, int, list]                    # (bank, base, words)
    b_images: List[Tuple[int, list]] = field(default_factory=list)
    p_images: List[Tuple[int, list]] = field(default_factory=list)
    outputs: List[Tuple[int, int, int]] = field(default_factory=list)   # (bank, base, count) per layer
    expected: SeqResult = None


class _Alloc:
    """bump allocator per act_buffer bank, and for B and P"""

    def __init__(self, hw: HwParams):
        self.hw = hw
        self.act = [0] * hw.nbuf
        self.b = 0
        self.p = 0

    def act_region(self, bank, count):
        base = self.act[bank]
        if base + count > (1 << self.hw.addr_width):
            raise MemoryError("act_buffer bank full")
        self.act[bank] += count
        return base

    def wp(self, b_count, p_count):
        if self.b + b_count > (1 << self.hw.b_addr_w) or self.p + p_count > (1 << self.hw.p_addr_w):
            raise MemoryError("B or P full")
        b, p = self.b, self.p
        self.b += b_count
        self.p += p_count
        return b, p


def build(rng, hw: HwParams, m_bits: int = 8, plan=None, h=8, w=8, c=None, read_latency=1,
          desc_depth=32, cout=None) -> Network:
    """plan: list of 'conv3', 'conv1', 'conv3s2', 'conv5', 'fc', 'max2', 'res', 'gavg'.
    None picks a random one. LENET and RESNET8 below are fixed plans. cout fixes
    every conv's output channels (default: random 1..2N per conv)."""
    n = hw.n
    if hw.nbuf < 3:
        raise ValueError("networks with residual adds need NBUF >= 3")
    c = c or n
    if plan is None:
        plan = ["conv3"]
        for _ in range(int(rng.integers(2, 5))):
            plan.append(str(rng.choice(["conv3", "conv1", "max2", "res", "conv3s2"])))
        plan.append("gavg")
    al = _Alloc(hw)
    x = rng.integers(-128, 128, (h, w, c))
    zp = int(rng.integers(-20, 21))
    cur = dict(bank=0, base=0, h=h, w=w, c=c, zp=zp)
    cur["base"] = al.act_region(0, h * w * words(c, n))
    net = Network([], [], [], (0, cur["base"], pack_act(x, n, zp)))

    def width():
        return want or cout or int(rng.integers(1, 2 * n + 1))

    def other(*banks):
        return next(b for b in range(hw.nbuf) if b not in banks)

    def add_conv(k, stride, cout, relu=None, src=None, pad=None):
        s = src or cur
        pad = k // 2 if pad is None else pad
        ho, wo = (s["h"] + 2 * pad - k) // stride + 1, (s["w"] + 2 * pad - k) // stride + 1
        cw_in, cw_out = words(s["c"], n), words(cout, n)
        K = k_of(k, k, cw_in, n)
        b_base, p_base = al.wp(cw_out * K, cw_out * n)
        dst = other(s["bank"])
        dst_base = al.act_region(dst, ho * wo * cw_out)
        wts = rng.integers(-128, 128, (k, k, s["c"], cout))
        bias, mult, shift = requant_params(rng, cout, k * k * s["c"])
        layer = Layer(x=np.zeros((s["h"], s["w"], s["c"]), int), w=wts, bias=bias, mult=mult,
                      shift=shift, zp_in=s["zp"], zp_out=int(rng.integers(-20, 21)),
                      relu_en=int(rng.integers(0, 2)) if relu is None else relu, stride=stride,
                      pad=pad, src_sel=s["bank"], dst_sel=dst, src_base=s["base"],
                      dst_base=dst_base, b_base=b_base, p_base=p_base)
        net.b_images.append((b_base, pack_b(wts, n)))
        net.p_images.append((p_base, pack_p(layer, n)))
        net.steps.append(Step(OP_CONV, conv_bus(layer, n), layer))
        net.outputs.append((dst, dst_base, ho * wo * cw_out))
        return dict(bank=dst, base=dst_base, h=ho, w=wo, c=cout, zp=layer.zp_out)

    for item in plan:
        kind, _, w_s = item.partition(":")          # "conv5:6" fixes this conv's width
        want = int(w_s) if w_s else None
        if kind in ("conv3", "conv1", "conv3s2"):
            k, st = (3, 1) if kind == "conv3" else (1, 1) if kind == "conv1" else (3, 2)
            cur = add_conv(k, st, width())
            net.names.append(kind)
        elif kind == "conv5":
            if cur["h"] < 5 or cur["w"] < 5:
                continue
            cur = add_conv(5, 1, width(), pad=0)
            net.names.append(kind)
        elif kind == "fc":
            if cur["h"] != cur["w"] or cur["h"] > 7:
                raise ValueError(f"fc needs a square input of at most 7x7, got {cur['h']}x{cur['w']}")
            cur = add_conv(cur["h"], 1, width(), pad=0)
            net.names.append(kind)
        elif kind == "max2":
            if cur["h"] < 2 or cur["w"] < 2:
                continue
            cw = words(cur["c"], n)
            dst = other(cur["bank"])
            ho, wo = cur["h"] // 2, cur["w"] // 2
            db = al.act_region(dst, ho * wo * cw)
            net.steps.append(Step(OP_MAX2, pool_fields(OP_MAX2, cur["h"], cur["w"], cw, cur["bank"],
                                                       dst, cur["base"], db)))
            net.outputs.append((dst, db, ho * wo * cw))
            net.names.append("max2")
            cur = dict(cur, bank=dst, base=db, h=ho, w=wo)
        elif kind == "res":
            skip = dict(cur)
            mid = add_conv(3, 1, cur["c"], relu=1)
            # the second conv must not land in the skip's bank either, so the
            # add (reading skip and branch) has a third bank to write
            y = add_conv(3, 1, cur["c"], relu=0, src=mid)
            if y["bank"] == skip["bank"]:
                # move the branch's output to the bank that is neither
                layer = net.steps[-1].layer
                nb = other(mid["bank"], skip["bank"])
                al.act[y["bank"]] -= y["h"] * y["w"] * words(y["c"], n)
                layer.dst_sel = nb
                layer.dst_base = al.act_region(nb, y["h"] * y["w"] * words(y["c"], n))
                net.steps[-1] = Step(OP_CONV, conv_bus(layer, n), layer)
                net.outputs[-1] = (nb, layer.dst_base, net.outputs[-1][2])
                y = dict(y, bank=nb, base=layer.dst_base)
            cw = words(cur["c"], n)
            dst = other(y["bank"], skip["bank"])
            db = al.act_region(dst, cur["h"] * cur["w"] * cw)
            sa, sb, so = [float(v) for v in rng.uniform(0.01, 0.05, 3)]
            ma, mb, sh = AM.scales_to_params(sa, sb, so, m_bits)
            ac = AM.AddCmd(cur["h"], cur["w"], cw, y["bank"], y["base"], skip["bank"], skip["base"],
                           dst, db, za=y["zp"], zb=skip["zp"], zo=int(rng.integers(-20, 21)),
                           ma=ma, mb=mb, shift=sh, relu_en=1)
            net.steps.append(Step(OP_ADD, add_fields(ac)))
            net.outputs.append((dst, db, cur["h"] * cur["w"] * cw))
            net.names += ["res.conv_a", "res.conv_b", "res.add"]
            cur = dict(cur, bank=dst, base=db, zp=ac.zo)
        elif kind == "gavg":
            hw_px = cur["h"] * cur["w"]
            if hw_px & (hw_px - 1):
                continue                     # not a power of two: no head here
            cw = words(cur["c"], n)
            dst = other(cur["bank"])
            db = al.act_region(dst, cw)
            net.steps.append(Step(OP_GAVG, pool_fields(OP_GAVG, cur["h"], cur["w"], cw, cur["bank"],
                                                       dst, cur["base"], db)))
            net.outputs.append((dst, db, cw))
            net.names.append("gavg")
            cur = dict(cur, bank=dst, base=db, h=1, w=1)
    net.steps.append(Step(OP_END, {}))
    if len(net.steps) > desc_depth:
        raise ValueError("network longer than the descriptor RAM")
    net.descriptors = [P.pack_descriptor(s.op, s.fields) for s in net.steps]
    # expected: run the model on a copy (Layer objects get their x re-read
    # from the bank mirror, so give it private copies)
    banks = Banks(hw.nbuf, 1 << hw.addr_width)
    banks.write(net.input[0], net.input[1], net.input[2])
    steps = [Step(s.op, s.fields, copy.deepcopy(s.layer)) for s in net.steps]
    net.expected = run(steps, banks, hw, read_latency, m_bits, desc_depth)
    return net


@dataclass
class NetRun:
    program: "P.ProgramResult"
    outputs: List[list]             # words read back from every layer's output region
    loads: int                      # load passes (1 = no reload)


def images(hw: "P.HwInfo", net: Network) -> "P.MemImages":
    return P.memory_images(hw, act=[net.input], b=net.b_images, p=net.p_images)


def board_run(hw: "P.HwInfo", net: Network, first: int, stats):
    """protocol sequence: load input, B and P (CRC checked), load the program
    at descriptor `first` (CRC checked), run it once, then read every layer's
    output region back. the host does nothing per layer."""
    loads = yield from P.load(hw, images(hw, net), stats)
    yield from P.load_program(net.descriptors, first, stats)
    prog = yield from P.run_program(first, stats)
    outs = []
    for bank, base, count in net.outputs[:prog.layers_done]:
        outs.append((yield from P.read_act(hw, bank, base, count, stats)))
    return NetRun(prog, outs, loads)


def compare(net: Network, run: NetRun):
    """(ok, lines): every layer's words and cycle count against the model"""
    e = net.expected
    lines, ok = [], True
    if run.program.err_code or run.program.layers_done != e.layers_done:
        ok = False
        lines.append(f"program: {run.program.layers_done} layers done, error "
                     f"{run.program.err_code:#x} at descriptor {run.program.err_index}; model "
                     f"{e.layers_done} layers, no error")
    for i, name in enumerate(net.names[:run.program.layers_done]):
        got, exp = run.outputs[i], e.outputs[i]
        cyc, ecyc = run.program.cycles[i], e.cycles[i]
        good = got == exp and cyc == ecyc
        ok &= good
        msg = f"layer {i:2d} {name:12s} {len(exp):5d} words  cycles {cyc:7d} (model {ecyc:7d})"
        if got != exp:
            bad = [j for j, (a, b) in enumerate(zip(got, exp)) if a != b] or [min(len(got), len(exp))]
            msg += f"  {len(bad)} words differ, first at {bad[0]}"
        lines.append(("ok   " if good else "FAIL ") + msg)
    return ok, lines


# fixed plans: LeNet-5 (32x32 input: C1 5x5, S2, C3 5x5, S4, F5 as a 5x5 fc,
# F6 and the output as 1x1 fc) and a ResNet-8-style stack (stem, three basic
# blocks with stride-2 downsampling convs between them, global average)
LENET = ["conv5:6", "max2", "conv5:16", "max2", "fc:120", "fc:84", "fc:10"]
RESNET8 = ["conv3:16", "res", "conv3s2:16", "res", "conv3s2:16", "res", "gavg"]


if __name__ == "__main__":
    fails = 0

    def chk(name, ok):
        global fails
        print(f"{'pass' if ok else 'FAIL'} {name}")
        fails += not ok

    for n in (2, 4, 8):
        hw = HwParams(n=n, addr_width=12, nbuf=3, b_addr_w=13, p_addr_w=6 if n < 8 else 7)
        rng = np.random.default_rng(n)
        for i in range(5):
            try:
                net = build(rng, hw, plan=None if i else ["conv3", "max2", "res", "conv1", "gavg"])
            except MemoryError:
                continue
            e = net.expected
            chk(f"N={n} #{i} {net.names}: runs to END, one output per layer",
                e.err_index is None and e.layers_done == len(net.steps) - 1 == len(net.outputs)
                and all(len(o) == cnt for o, (_, _, cnt) in zip(e.outputs, net.outputs)))
            chk(f"  descriptors round-trip", all(P.unpack_descriptor(d)[0] == s.op
                                                 for d, s in zip(net.descriptors, net.steps)))
            regions = sorted((b, base, base + cnt) for b, base, cnt in net.outputs)
            chk(f"  output regions never overlap",
                all(not (a[0] == b[0] and b[1] < a[2]) for a, b in zip(regions, regions[1:])))
    for n in (2, 4, 8):
        hw = HwParams(n=n, addr_width=12, nbuf=3, b_addr_w=13, p_addr_w=6 if n < 8 else 7)
        rng = np.random.default_rng(10 + n)
        hw = HwParams(n=n, addr_width=12, nbuf=3, b_addr_w=13, p_addr_w=8)   # the board's sizes
        for name, plan, h in (("LeNet-5", LENET, 32), ("ResNet-8", RESNET8, 16)):
            try:
                net = build(rng, hw, plan=plan, h=h, w=h, c=3)
            except MemoryError as e:
                # B is 2^13 words of N bytes: LeNet-5's ~60 kB of weights need N=8
                chk(f"N={n} {name} does not fit on chip ({e}): only expected below N=8", n < 8)
                continue
            e = net.expected
            chk(f"N={n} {name} {net.names}: runs to END", e.err_index is None and
                e.layers_done == len(net.outputs))
            if name == "LeNet-5":
                f = net.steps[-2].fields
                chk(f"  LeNet-5 ends in a 1x1 output", (f["h_out"], f["w_out"]) == (1, 1))
    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
