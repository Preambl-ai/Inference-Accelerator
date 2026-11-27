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
    ddr: bytearray = None                            # phase 4: the DDR image (weights of DDR layers)
    ddr_images: List[Tuple[int, bytes]] = field(default_factory=list)   # (DDR byte address, bytes)
    checked: List[int] = field(default_factory=list)  # layers whose output region survives to the end


class _Alloc:
    """act_buffer regions per bank, and bump allocation for B and P.
    keep="all": every output gets fresh space (all of them can be read back).
    keep="last": first fit around the tensors still live (`live`), so a deep
    network fits; earlier outputs get overwritten"""

    def __init__(self, hw: HwParams, keep="all"):
        self.hw, self.keep = hw, keep
        self.act = [0] * hw.nbuf
        self.live = []                  # (bank, base, count) still needed
        self.b = 0
        self.p = 0

    def act_region(self, bank, count):
        depth = 1 << self.hw.addr_width
        if self.keep == "all":
            base = self.act[bank]
            if base + count > depth:
                raise MemoryError("act_buffer bank full")
            self.act[bank] += count
            return base
        taken = sorted((b0, b0 + c) for bk, b0, c in self.live if bk == bank)
        base = 0
        for lo, hi in taken:
            if base + count <= lo:
                break
            base = max(base, hi)
        if base + count > depth:
            raise MemoryError("act_buffer bank full")
        return base

    def wp(self, b_count, p_count):
        if self.b + b_count > (1 << self.hw.b_addr_w) or self.p + p_count > (1 << self.hw.p_addr_w):
            raise MemoryError("B or P full")
        b, p = self.b, self.p
        self.b += b_count
        self.p += p_count
        return b, p


def build(rng, hw: HwParams, m_bits: int = 8, plan=None, h=8, w=8, c=None, read_latency=1,
          desc_depth=32, cout=None, weights="chip", axi_latency=1, ddr_size=1 << 20,
          keep="all") -> Network:
    """plan: list of 'conv3', 'conv1', 'conv3s2', 'conv5', 'fc', 'max2', 'res', 'gavg'.
    None picks a random one. LENET and RESNET8 below are fixed plans. cout fixes
    every conv's output channels (default: random 1..2N per conv).

    weights: "chip" (B and P loaded on chip before the program), "ddr" (every
    conv fetches its B and P from the DDR image through wdma) or "mixed"
    (each conv picks at random). a DDR conv's on-chip copy goes top-aligned in
    B and P (a one-layer cache), so it never lands on an on-chip layer's
    weights, which grow from 0. DDR images start at random 8-byte addresses,
    so groups cross 4 KB lines."""
    n = hw.n
    if hw.nbuf < 3:
        raise ValueError("networks with residual adds need NBUF >= 3")
    c = c or n
    if plan is None:
        plan = ["conv3"]
        for _ in range(int(rng.integers(2, 5))):
            plan.append(str(rng.choice(["conv3", "conv1", "max2", "res", "conv3s2"])))
        plan.append("gavg")
    al = _Alloc(hw, keep)
    ddr = bytearray(ddr_size)
    ddr_next = [0]
    drng = np.random.default_rng(int(rng.integers(0, 2**31)))     # drawn whatever `weights` is
    # (placement and the mixed choice draw from drng, so the weights and
    # activations of a seed are the same whatever `weights` is)
    cache_floor = [1 << hw.b_addr_w, 1 << hw.p_addr_w]      # lowest on-chip slot a DDR layer uses

    def ddr_put(words, width_bytes=8):
        """place 8-byte beats at a random-ish 8-aligned address; returns it"""
        addr = ddr_next[0] + 8 * int(drng.integers(0, 64))
        data = b"".join(int(v).to_bytes(width_bytes, "little") for v in words)
        if addr + len(data) > len(ddr):
            raise MemoryError("DDR image full")
        ddr[addr:addr + len(data)] = data
        ddr_next[0] = addr + len(data)
        return addr
    x = rng.integers(-128, 128, (h, w, c))
    zp = int(rng.integers(-20, 21))
    cur = dict(bank=0, base=0, h=h, w=w, c=c, zp=zp)
    cur["base"] = al.act_region(0, h * w * words(c, n))
    net = Network([], [], [], (0, cur["base"], pack_act(x, n, zp)))

    def width():
        return want or cout or int(rng.integers(1, 2 * n + 1))

    def other(*banks):
        return next(b for b in range(hw.nbuf) if b not in banks)

    def reg(t):
        return (t["bank"], t["base"], t["h"] * t["w"] * words(t["c"], n))

    def add_conv(k, stride, cout, relu=None, src=None, pad=None, avoid=(), live=()):
        """avoid: banks the output must not go to; live: tensors (besides
        the source) that must survive this layer"""
        s = src or cur
        al.live = [reg(s)] + [reg(t) for t in live]
        pad = k // 2 if pad is None else pad
        ho, wo = (s["h"] + 2 * pad - k) // stride + 1, (s["w"] + 2 * pad - k) // stride + 1
        cw_in, cw_out = words(s["c"], n), words(cout, n)
        K = k_of(k, k, cw_in, n)
        from_ddr = weights == "ddr" or (weights == "mixed" and bool(drng.integers(0, 2)))
        if from_ddr:
            b_base, p_base = (1 << hw.b_addr_w) - cw_out * K, (1 << hw.p_addr_w) - cw_out * n
            if b_base < 0 or p_base < 0:
                raise MemoryError("one layer's B or P is larger than the on-chip memory")
            cache_floor[0], cache_floor[1] = min(cache_floor[0], b_base), min(cache_floor[1], p_base)
        else:
            b_base, p_base = al.wp(cw_out * K, cw_out * n)
        dst = other(s["bank"], *avoid)
        dst_base = al.act_region(dst, ho * wo * cw_out)
        wts = rng.integers(-128, 128, (k, k, s["c"], cout))
        bias, mult, shift = requant_params(rng, cout, k * k * s["c"])
        layer = Layer(x=np.zeros((s["h"], s["w"], s["c"]), int), w=wts, bias=bias, mult=mult,
                      shift=shift, zp_in=s["zp"], zp_out=int(rng.integers(-20, 21)),
                      relu_en=int(rng.integers(0, 2)) if relu is None else relu, stride=stride,
                      pad=pad, src_sel=s["bank"], dst_sel=dst, src_base=s["base"],
                      dst_base=dst_base, b_base=b_base, p_base=p_base)
        fields = conv_bus(layer, n)
        if from_ddr:
            b_src = ddr_put(pack_b(wts, n))
            p_src = ddr_put(pack_p(layer, n))
            net.ddr_images += [(b_src, bytes(ddr[b_src:b_src + 8 * cw_out * K])),
                               (p_src, bytes(ddr[p_src:p_src + 8 * cw_out * n]))]
            fields.update(w_ddr=1, k_words=K, b_src=b_src, p_src=p_src)
        else:
            net.b_images.append((b_base, pack_b(wts, n)))
            net.p_images.append((p_base, pack_p(layer, n)))
        net.steps.append(Step(OP_CONV, fields, layer))
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
            al.live = [reg(cur)]
            db = al.act_region(dst, ho * wo * cw)
            net.steps.append(Step(OP_MAX2, pool_fields(OP_MAX2, cur["h"], cur["w"], cw, cur["bank"],
                                                       dst, cur["base"], db)))
            net.outputs.append((dst, db, ho * wo * cw))
            net.names.append("max2")
            cur = dict(cur, bank=dst, base=db, h=ho, w=wo)
        elif kind in ("res", "resd"):
            # res: identity skip. resd: the ResNet downsampling block, a
            # stride-2 3x3 on the branch and a stride-2 1x1 projection as the
            # skip (width from "resd:64", default the input's)
            c_out = want or cur["c"]
            if kind == "resd":
                skip = add_conv(1, 2, c_out, relu=0, pad=0)
                mid = add_conv(3, 2, c_out, relu=1, live=(skip,))
            else:
                skip = dict(cur)
                mid = add_conv(3, 1, c_out, relu=1)
            # the second conv must not land in the skip's bank either, so the
            # add (reading skip and branch) has a third bank to write
            y = add_conv(3, 1, c_out, relu=0, src=mid, avoid=(skip["bank"],), live=(skip,))
            cw = words(c_out, n)
            dst = other(y["bank"], skip["bank"])
            al.live = [reg(y), reg(skip)]
            db = al.act_region(dst, y["h"] * y["w"] * cw)
            sa, sb, so = [float(v) for v in rng.uniform(0.01, 0.05, 3)]
            ma, mb, sh = AM.scales_to_params(sa, sb, so, m_bits)
            ac = AM.AddCmd(y["h"], y["w"], cw, y["bank"], y["base"], skip["bank"], skip["base"],
                           dst, db, za=y["zp"], zb=skip["zp"], zo=int(rng.integers(-20, 21)),
                           ma=ma, mb=mb, shift=sh, relu_en=1)
            net.steps.append(Step(OP_ADD, add_fields(ac)))
            net.outputs.append((dst, db, y["h"] * y["w"] * cw))
            net.names += (["resd.skip"] if kind == "resd" else []) + \
                [f"{kind}.conv_a", f"{kind}.conv_b", f"{kind}.add"]
            cur = dict(y, bank=dst, base=db, zp=ac.zo)
        elif kind == "gavg":
            hw_px = cur["h"] * cur["w"]
            if hw_px & (hw_px - 1):
                continue                     # not a power of two: no head here
            cw = words(cur["c"], n)
            dst = other(cur["bank"])
            al.live = [reg(cur)]
            db = al.act_region(dst, cw)
            net.steps.append(Step(OP_GAVG, pool_fields(OP_GAVG, cur["h"], cur["w"], cw, cur["bank"],
                                                       dst, cur["base"], db)))
            net.outputs.append((dst, db, cw))
            net.names.append("gavg")
            cur = dict(cur, bank=dst, base=db, h=1, w=1)
    net.steps.append(Step(OP_END, {}))
    if al.b > cache_floor[0] or al.p > cache_floor[1]:
        raise MemoryError("on-chip weights reach into the DDR layers' cache region")
    net.ddr = ddr
    net.checked = list(range(len(net.outputs))) if keep == "all" else [len(net.outputs) - 1]
    if len(net.steps) > desc_depth:
        raise ValueError("network longer than the descriptor RAM")
    net.descriptors = [P.pack_descriptor(s.op, s.fields) for s in net.steps]
    # expected: run the model on a copy (Layer objects get their x re-read
    # from the bank mirror, so give it private copies)
    banks = Banks(hw.nbuf, 1 << hw.addr_width)
    banks.write(net.input[0], net.input[1], net.input[2])
    steps = [Step(s.op, s.fields, copy.deepcopy(s.layer)) for s in net.steps]
    net.expected = run(steps, banks, hw, read_latency, m_bits, desc_depth, ddr=ddr,
                       axi_latency=axi_latency)
    return net


@dataclass
class NetRun:
    program: "P.ProgramResult"
    outputs: dict                   # layer -> words read back from its output region (net.checked)
    loads: int                      # load passes (1 = no reload)


def images(hw: "P.HwInfo", net: Network) -> "P.MemImages":
    return P.memory_images(hw, act=[net.input], b=net.b_images, p=net.p_images)


def board_run(hw: "P.HwInfo", net: Network, first: int, stats):
    """protocol sequence: load input, B and P (CRC checked), load the program
    at descriptor `first` (CRC checked), run it once, then read every layer's
    output region back. the host does nothing per layer."""
    loads = yield from P.load(hw, images(hw, net), stats)
    if net.ddr_images:
        yield from P.load_ddr(net.ddr_images, stats)
    yield from P.load_program(net.descriptors, first, stats)
    prog = yield from P.run_program(first, stats, desc_depth=hw.desc_depth if hw.has_dma else 0)
    outs = {}
    for i in net.checked:
        if i < prog.layers_done:
            bank, base, count = net.outputs[i]
            outs[i] = yield from P.read_act(hw, bank, base, count, stats)
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
        got, exp = run.outputs.get(i), e.outputs[i]
        cyc, ecyc = run.program.cycles[i], e.cycles[i]
        good = (got is None or got == exp) and cyc == ecyc
        ok &= good
        msg = f"layer {i:2d} {name:12s} {len(exp):5d} words  cycles {cyc:7d} (model {ecyc:7d})"
        if run.program.fetch_cycles and run.program.fetch_cycles[i]:
            msg += f"  weight fetch {run.program.fetch_cycles[i]:6d}"
        if got is None:
            msg += "  (overwritten later, not read back)"
        elif got != exp:
            bad = [j for j, (a, b) in enumerate(zip(got, exp)) if a != b] or [min(len(got), len(exp))]
            msg += f"  {len(bad)} words differ, first at {bad[0]}"
        lines.append(("ok   " if good else "FAIL ") + msg)
    return ok, lines


# fixed plans:
#   LeNet-5 (32x32 input): C1 5x5, S2, C3 5x5, S4, F5 as a 5x5 fc, F6, output
#   RESNET8_CHIP: a width-16 ResNet-style stack whose weights all fit on chip
#     (phase 3; stride-2 convs between identity blocks)
#   ResNet-8 / ResNet-20 (CIFAR layout, 32x32): stem 3x3x16, then 1 or 3 basic
#     blocks per stage at widths 16 / 32 / 64, stages 2 and 3 opened by a
#     downsampling block (stride-2 branch, 1x1 stride-2 projection skip),
#     global average, fc-10. Their weights need DDR (weights="ddr") and, for a
#     full 32x32 input, keep="last"
LENET = ["conv5:6", "max2", "conv5:16", "max2", "fc:120", "fc:84", "fc:10"]
RESNET8_CHIP = ["conv3:16", "res", "conv3s2:16", "res", "conv3s2:16", "res", "gavg"]
RESNET8 = ["conv3:16", "res", "resd:32", "resd:64", "gavg", "fc:10"]
RESNET20 = ["conv3:16", "res", "res", "res", "resd:32", "res", "res", "resd:64", "res", "res",
            "gavg", "fc:10"]


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
        for name, plan, h in (("LeNet-5", LENET, 32), ("ResNet-8 on chip", RESNET8_CHIP, 16)):
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
    # phase 4: weights from DDR
    for n in (2, 4, 8):
        hw = HwParams(n=n, addr_width=12, nbuf=3, b_addr_w=13, p_addr_w=8)
        rng = np.random.default_rng(20 + n)
        for weights in ("ddr", "mixed"):
            net = build(rng, hw, plan=["conv3", "max2", "res", "conv1", "gavg"], weights=weights,
                        axi_latency=4)
            e = net.expected
            nd = sum(1 for s in net.steps if s.fields.get("w_ddr"))
            chk(f"N={n} {weights}: {nd} DDR convs, runs to END, fetch cycles only on them",
                e.err_index is None and (nd > 0 or weights == "mixed") and
                [c > 0 for c in e.fetch_cycles] ==
                [bool(s.fields.get("w_ddr")) for s in net.steps if s.op != OP_END])
            # the model's output does not depend on where the weights live:
            # build the same network on chip from the same seed
            twin = build(np.random.default_rng(99), hw, plan=["conv3", "res"], weights="chip")
            twin_d = build(np.random.default_rng(99), hw, plan=["conv3", "res"], weights="ddr")
            chk(f"  same seed, chip or DDR weights: identical outputs",
                twin.expected.outputs == twin_d.expected.outputs)
    # ResNet-8 / ResNet-20 at full size: 32x32x3, DDR weights, act space reused
    hw = HwParams(n=8, addr_width=12, nbuf=3, b_addr_w=13, p_addr_w=8)
    for name, plan in (("ResNet-8", RESNET8), ("ResNet-20", RESNET20)):
        net = build(np.random.default_rng(5), hw, plan=plan, h=32, w=32, c=3, weights="ddr",
                    keep="last", desc_depth=64)
        e = net.expected
        convs = sum(1 for s in net.steps if s.op == 1)
        chk(f"{name}: {convs} convs, {len(net.steps) - 1} layers, runs to END, 1x1x10 output",
            e.err_index is None and e.layers_done == len(net.steps) - 1 and net.outputs[-1][2] == 2)
        chk(f"  {name}: B and P of every layer in DDR ({len(net.ddr_images) // 2} layers)",
            len(net.ddr_images) == 2 * convs and not net.b_images)
    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
