# Golden model for rtl/layer_engine.sv: what the act_buffer banks hold after a
# sequence of ops, built from the per-engine NumPy references (conv_ref via
# conv_layer_model, maxpool2_ref / gavg_ref via pool_model). It is also the
# network reference the sequencer and run_network.py compare against.
#
# Banks is a mirror of every act_buffer bank as lists of packed words. Each op
# reads its input from the mirror (so chained ops use exactly what the
# previous op left) and writes its expected output words into it.

import numpy as np

from conv_layer_model import (HwParams, Layer, check_command as conv_check, expected_output_words,
                              pack_act, unpack_act, words)
from pool_model import M_GAVG, M_MAX2, PoolCmd, check_command as pool_check, pool_words

OP_CONV, OP_MAX2, OP_GAVG = 1, 2, 3
OPS = (OP_CONV, OP_MAX2, OP_GAVG)
E_MODE = 9


class Banks:
    def __init__(self, nbuf, depth, fill=None):
        self.depth = depth
        self.b = [list(fill[i]) if fill else [0] * depth for i in range(nbuf)]

    def write(self, sel, base, data):
        for i, w in enumerate(data):
            self.b[sel][base + i] = w

    def read(self, sel, base, count):
        return self.b[sel][base:base + count]


def pool_cmd_of(op, fields):
    return PoolCmd(mode=M_MAX2 if op == OP_MAX2 else M_GAVG, h_in=fields["h_in"],
                   w_in=fields["w_in"], c_words=fields["cin_words"], src_sel=fields["src_sel"],
                   dst_sel=fields["dst_sel"], src_base=fields["src_base"], dst_base=fields["dst_base"])


def check(op, fields, hw: HwParams, host_mode=0):
    """err_code layer_engine reports for (op, fields): 0 means it runs"""
    if op == OP_CONV:
        return conv_check(fields, hw, host_mode)
    if op in (OP_MAX2, OP_GAVG):
        return pool_check(pool_cmd_of(op, fields), hw.addr_width, hw.nbuf, host_mode)
    return 1 << E_MODE


def apply_conv(banks: Banks, layer: Layer, n: int, from_banks: bool = True):
    """conv from the mirror: the input tensor is re-read from the source bank
    (so it is exactly what the previous op left), unpacked with its padded
    lanes, and convolved; the zero-weight padded input lanes do not matter"""
    cmd = layer.command(n)
    if from_banks:
        src = banks.read(layer.src_sel, layer.src_base, cmd["h_in"] * cmd["w_in"] * cmd["cin_words"])
        layer.x = unpack_act(src, cmd["h_in"], cmd["w_in"], layer.x.shape[2], n)
    out = expected_output_words(layer, n)
    banks.write(layer.dst_sel, layer.dst_base, out)
    return out


def apply_pool(banks: Banks, op, fields, n: int):
    c = pool_cmd_of(op, fields)
    src = banks.read(c.src_sel, c.src_base, c.h_in * c.w_in * c.c_words)
    out = pool_words(src, c.h_in, c.w_in, c.c_words, n, c.mode)
    banks.write(c.dst_sel, c.dst_base, out)
    return out


def pool_fields(op, h, w, cw, src_sel, dst_sel, src_base, dst_base):
    """the command-bus fields a pool op uses (the rest are don't-care, 0)"""
    f = dict.fromkeys(("h_out", "w_out", "cout_words", "kh", "kw", "stride", "pad", "zp_in",
                       "zp_out", "relu_en", "b_base", "p_base"), 0)
    f.update(h_in=h, w_in=w, cin_words=cw, src_sel=src_sel, dst_sel=dst_sel,
             src_base=src_base, dst_base=dst_base)
    return f


if __name__ == "__main__":
    from conv_layer_model import conv_ref, random_layer
    from pool_model import maxpool2_ref, gavg_ref

    fails = 0

    def chk(name, got, exp):
        global fails
        ok = (np.array_equal(got, exp) if isinstance(got, np.ndarray) or isinstance(exp, np.ndarray)
              else got == exp)
        print(f"{'pass' if ok else 'FAIL'} {name}")
        fails += not ok

    rng = np.random.default_rng(5)
    for n in (2, 4, 8):
        hw = HwParams(n=n, addr_width=10, nbuf=2)
        # conv -> max2 -> gavg, through the mirror, equals the plain NumPy pipeline
        layer = random_layer(rng, hw, kh=3, kw=3, stride=1, pad=1, h_in=8, w_in=8,
                             src_sel=0, dst_sel=1, src_base=0, dst_base=0)
        banks = Banks(2, 1 << 10)
        banks.write(0, 0, pack_act(layer.x, n, layer.zp_in))
        apply_conv(banks, layer, n)
        cw = words(layer.cout, n)
        apply_pool(banks, OP_MAX2, pool_fields(OP_MAX2, 8, 8, cw, 1, 0, 0, 500), n)
        apply_pool(banks, OP_GAVG, pool_fields(OP_GAVG, 4, 4, cw, 0, 1, 500, 900), n)
        direct = gavg_ref(maxpool2_ref(conv_ref(layer.x, layer.w, layer.bias, layer.mult, layer.shift,
                                                layer.zp_in, layer.zp_out, layer.relu_en, 1, 1)))
        got = unpack_act(banks.read(1, 900, cw), 1, 1, layer.cout, n)
        chk(f"N={n} conv->max2->gavg through the mirror == NumPy pipeline", got, direct)
        full = unpack_act(banks.read(1, 900, cw), 1, 1, cw * n, n, with_pad=True)
        chk(f"N={n} padded lanes still hold zp_out", bool(np.all(full[..., layer.cout:] == layer.zp_out)),
            True)
        chk(f"N={n} legal ops check 0", (check(OP_CONV, layer.command(n), hw),
                                         check(OP_MAX2, pool_fields(OP_MAX2, 8, 8, cw, 1, 0, 0, 500), hw)),
            (0, 0))
    hw = HwParams(n=4, addr_width=10, nbuf=2)
    chk("op 0 is E_MODE", check(0, {}, hw), 1 << E_MODE)
    chk("op 5 is E_MODE", check(5, {}, hw), 1 << E_MODE)
    chk("pool E_SEL through the bus", check(OP_GAVG, pool_fields(OP_GAVG, 2, 2, 1, 1, 1, 0, 0), hw), 1 << 7)

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
