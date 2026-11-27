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
from add_model import AddCmd, add_words, check_command as add_check

OP_CONV, OP_MAX2, OP_GAVG, OP_ADD = 1, 2, 3, 4
OPS = (OP_CONV, OP_MAX2, OP_GAVG, OP_ADD)
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


def add_cmd_of(fields):
    return AddCmd(h_in=fields["h_in"], w_in=fields["w_in"], c_words=fields["cin_words"],
                  src_sel=fields["src_sel"], src_base=fields["src_base"],
                  src2_sel=fields["src2_sel"], src2_base=fields["src2_base"],
                  dst_sel=fields["dst_sel"], dst_base=fields["dst_base"], za=fields["zp_in"],
                  zb=fields["zp_in2"], zo=fields["zp_out"], ma=fields["add_ma"],
                  mb=fields["add_mb"], shift=fields["add_shift"], relu_en=fields["relu_en"])


def check(op, fields, hw: HwParams, host_mode=0):
    """err_code layer_engine reports for (op, fields): 0 means it runs"""
    if op == OP_CONV:
        return conv_check(fields, hw, host_mode)
    if op in (OP_MAX2, OP_GAVG):
        return pool_check(pool_cmd_of(op, fields), hw.addr_width, hw.nbuf, host_mode)
    if op == OP_ADD:
        return add_check(add_cmd_of(fields), hw.addr_width, hw.nbuf, host_mode)
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


EXTRA_FIELDS = ("src2_sel", "src2_base", "zp_in2", "add_ma", "add_mb", "add_shift")


def pool_fields(op, h, w, cw, src_sel, dst_sel, src_base, dst_base):
    """the command-bus fields a pool op uses (the rest are don't-care, 0)"""
    f = dict.fromkeys(("h_out", "w_out", "cout_words", "kh", "kw", "stride", "pad", "zp_in",
                       "zp_out", "relu_en", "b_base", "p_base") + EXTRA_FIELDS, 0)
    f.update(h_in=h, w_in=w, cin_words=cw, src_sel=src_sel, dst_sel=dst_sel,
             src_base=src_base, dst_base=dst_base)
    return f


def add_fields(c: AddCmd):
    """the command-bus fields of an add"""
    f = pool_fields(OP_ADD, c.h_in, c.w_in, c.c_words, c.src_sel, c.dst_sel, c.src_base, c.dst_base)
    f.update(src2_sel=c.src2_sel, src2_base=c.src2_base, zp_in=c.za, zp_in2=c.zb, zp_out=c.zo,
             add_ma=c.ma, add_mb=c.mb, add_shift=c.shift, relu_en=c.relu_en)
    return f


def apply_add(banks: Banks, fields, n: int):
    c = add_cmd_of(fields)
    aw = banks.read(c.src_sel, c.src_base, c.n_words)
    bw = banks.read(c.src2_sel, c.src2_base, c.n_words)
    out = add_words(aw, bw, c, n)
    banks.write(c.dst_sel, c.dst_base, out)
    return out


def conv_bus(layer, n):
    """a conv layer's command plus the add-only fields, zero"""
    f = dict(layer.command(n))
    f.update(dict.fromkeys(EXTRA_FIELDS, 0))
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
    # a residual block: y = conv2(conv1(x)); out = add(y, x), checked against
    # plain NumPy (conv_ref twice, add_ref) through the mirror
    from add_model import add_ref
    for n in (2, 4):
        hw = HwParams(n=n, addr_width=10, nbuf=3)
        x = rng.integers(-128, 128, (4, 4, n))
        l1 = random_layer(rng, hw, kh=3, kw=3, stride=1, pad=1, h_in=4, w_in=4, cin=n, cout=n,
                          src_sel=0, dst_sel=1, src_base=0, dst_base=0)
        l1.x = x
        l2 = random_layer(rng, hw, kh=3, kw=3, stride=1, pad=1, h_in=4, w_in=4, cin=n, cout=n,
                          src_sel=1, dst_sel=2, src_base=0, dst_base=0)
        l2.zp_in = l1.zp_out
        banks = Banks(3, 1 << 10)
        banks.write(0, 0, pack_act(x, n, l1.zp_in))
        apply_conv(banks, l1, n)
        apply_conv(banks, l2, n)
        ac = AddCmd(4, 4, 1, 2, 0, 0, 0, 1, 100, za=l2.zp_out, zb=l1.zp_in, zo=3, ma=150, mb=90,
                    shift=8, relu_en=1)
        apply_add(banks, add_fields(ac), n)
        y1 = conv_ref(x, l1.w, l1.bias, l1.mult, l1.shift, l1.zp_in, l1.zp_out, l1.relu_en, 1, 1)
        y2 = conv_ref(y1, l2.w, l2.bias, l2.mult, l2.shift, l2.zp_in, l2.zp_out, l2.relu_en, 1, 1)
        want = add_ref(y2, x, l2.zp_out, l1.zp_in, 3, 150, 90, 8, 1)
        got = unpack_act(banks.read(1, 100, 16), 4, 4, n, n)
        chk(f"N={n} residual block conv->conv->add(x) == NumPy", got, want)
        chk(f"N={n} add legal through the bus", check(OP_ADD, add_fields(ac), hw), 0)
    hw = HwParams(n=4, addr_width=10, nbuf=2)
    chk("op 0 is E_MODE", check(0, {}, hw), 1 << E_MODE)
    chk("op 5 is E_MODE", check(5, {}, hw), 1 << E_MODE)
    chk("add dst == a is E_SEL", check(OP_ADD, add_fields(AddCmd(2, 2, 1, 0, 0, 0, 50, 0, 100)), hw),
        1 << 7)
    chk("pool E_SEL through the bus", check(OP_GAVG, pool_fields(OP_GAVG, 2, 2, 1, 1, 1, 0, 0), hw), 1 << 7)

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
