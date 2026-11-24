# Golden model for rtl/sequencer.sv: what a program of descriptors does to the
# act_buffer banks, the per-layer cycle counts, and the first error.
#
# A program is a list of steps (op, fields, layer); `layer` carries the conv
# weights the descriptor only points at (b_base, p_base). Ops are applied
# through layer_engine_model over a bank mirror; cycle counts come from the
# engines' measured formulas. The descriptor words themselves come from
# host/protocol.py (pack_descriptor), the one packer the board, the TB and
# run_network.py all use.

from dataclasses import dataclass, field
from typing import List, Optional

import add_model as AM
import conv_layer_model as CM
from layer_engine_model import (OP_ADD, OP_CONV, OP_GAVG, OP_MAX2, Banks, add_cmd_of, apply_add,
                                apply_conv, apply_pool, check, pool_cmd_of)
import pool_model as PM

OP_END = 0
E_NOEND = 10


@dataclass
class Step:
    op: int
    fields: dict
    layer: Optional[object] = None          # conv_layer_model.Layer for CONV


@dataclass
class SeqResult:
    layers_done: int = 0
    cycles: List[int] = field(default_factory=list)
    err_index: Optional[int] = None
    err_code: int = 0
    outputs: List[list] = field(default_factory=list)   # words each finished layer wrote


def layer_cycles(op, fields, n, read_latency, m_bits):
    if op == OP_CONV:
        return CM.estimate_cycles(fields, n, read_latency)
    if op in (OP_MAX2, OP_GAVG):
        return PM.estimate_cycles(pool_cmd_of(op, fields), read_latency)
    return AM.estimate_cycles(add_cmd_of(fields), read_latency, m_bits)


def run(steps: List[Step], banks: Banks, hw: CM.HwParams, read_latency: int, m_bits: int = 8,
        desc_depth: int = 32, first: int = 0) -> SeqResult:
    """steps[i] is descriptor first + i. stops at END, at the first error, or
    after the last descriptor of the RAM (E_NOEND)"""
    r = SeqResult()
    n = hw.n
    for i, s in enumerate(steps):
        idx = first + i
        if s.op == OP_END:
            return r
        code = check(s.op, s.fields, hw)
        if code:
            r.err_index, r.err_code = idx, code
            return r
        if s.op == OP_CONV:
            out = apply_conv(banks, s.layer, n)
        elif s.op in (OP_MAX2, OP_GAVG):
            out = apply_pool(banks, s.op, s.fields, n)
        else:
            out = apply_add(banks, s.fields, n)
        r.outputs.append(out)
        r.cycles.append(layer_cycles(s.op, s.fields, n, read_latency, m_bits))
        r.layers_done += 1
        if idx == desc_depth - 1:
            r.err_index, r.err_code = idx, 1 << E_NOEND
            return r
    raise ValueError("program has no END and does not reach the end of the RAM")


if __name__ == "__main__":
    import numpy as np
    from conv_layer_model import conv_ref, pack_act, random_layer, unpack_act, words
    from layer_engine_model import add_fields, conv_bus, pool_fields
    from pool_model import maxpool2_ref, gavg_ref

    fails = 0

    def chk(name, got, exp):
        global fails
        ok = (np.array_equal(got, exp) if isinstance(got, np.ndarray) or isinstance(exp, np.ndarray)
              else got == exp)
        print(f"{'pass' if ok else 'FAIL'} {name}" + ("" if ok else f": {got} vs {exp}"))
        fails += not ok

    rng = np.random.default_rng(11)
    n = 4
    hw = CM.HwParams(n=n, addr_width=10, nbuf=3)
    # conv(8x8) -> max2 -> conv -> add(skip = pooled) -> gavg -> END
    x = rng.integers(-128, 128, (8, 8, n))
    l1 = random_layer(rng, hw, kh=3, kw=3, stride=1, pad=1, h_in=8, w_in=8, cin=n, cout=n,
                      src_sel=0, dst_sel=1, src_base=0, dst_base=0, b_base=0, p_base=0)
    l1.x = x
    mp = pool_fields(OP_MAX2, 8, 8, 1, 1, 2, 0, 0)
    l2 = random_layer(rng, hw, kh=3, kw=3, stride=1, pad=1, h_in=4, w_in=4, cin=n, cout=n,
                      src_sel=2, dst_sel=0, src_base=0, dst_base=200, b_base=100, p_base=8)
    l2.zp_in = l1.zp_out
    ad = add_fields(AM.AddCmd(4, 4, 1, 0, 200, 2, 0, 1, 300, za=l2.zp_out, zb=l1.zp_out, zo=0,
                              ma=128, mb=128, shift=8))
    gp = pool_fields(OP_GAVG, 4, 4, 1, 1, 0, 300, 400)
    steps = [Step(OP_CONV, conv_bus(l1, n), l1), Step(OP_MAX2, mp), Step(OP_CONV, conv_bus(l2, n), l2),
             Step(OP_ADD, ad), Step(OP_GAVG, gp), Step(OP_END, {})]
    banks = Banks(3, 1 << 10)
    banks.write(0, 0, pack_act(x, n, l1.zp_in))
    r = run(steps, banks, hw, read_latency=1)
    chk("five layers done, no error", (r.layers_done, r.err_index, r.err_code), (5, None, 0))
    # direct NumPy pipeline
    y1 = conv_ref(x, l1.w, l1.bias, l1.mult, l1.shift, l1.zp_in, l1.zp_out, l1.relu_en, 1, 1)
    p1 = maxpool2_ref(y1)
    y2 = conv_ref(p1, l2.w, l2.bias, l2.mult, l2.shift, l2.zp_in, l2.zp_out, l2.relu_en, 1, 1)
    s = AM.add_ref(y2, p1, l2.zp_out, l1.zp_out, 0, 128, 128, 8, 0)
    g = gavg_ref(s)
    chk("program output == NumPy pipeline", unpack_act(banks.read(0, 400, 1), 1, 1, n, n), g)
    chk("cycles per layer from the formulas", r.cycles,
        [CM.estimate_cycles(l1.command(n), n, 1), PM.estimate_cycles(pool_cmd_of(OP_MAX2, mp), 1),
         CM.estimate_cycles(l2.command(n), n, 1), AM.estimate_cycles(add_cmd_of(ad), 1),
         PM.estimate_cycles(pool_cmd_of(OP_GAVG, gp), 1)])
    # an illegal descriptor stops the program there
    bad = dict(gp, dst_sel=1)                      # src == dst
    r = run(steps[:4] + [Step(OP_GAVG, bad), Step(OP_END, {})], Banks(3, 1 << 10), hw, 1)
    chk("error stops at descriptor 4 with E_SEL", (r.layers_done, r.err_index, r.err_code), (4, 4, 1 << 7))
    r = run([Step(OP_MAX2, mp), Step(6, {}), Step(OP_END, {})], Banks(3, 1 << 10), hw, 1)
    chk("bad op is E_MODE", (r.layers_done, r.err_index, r.err_code), (1, 1, 1 << 9))
    # no END: the last descriptor of the RAM runs, then E_NOEND
    r = run([Step(OP_MAX2, mp)] * 2, Banks(3, 1 << 10), hw, 1, desc_depth=4, first=2)
    chk("E_NOEND after the last descriptor", (r.layers_done, r.err_index, r.err_code), (2, 3, 1 << E_NOEND))

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
