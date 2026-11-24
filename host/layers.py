"""Glue between the conv_layer golden model and the protocol: a model Layer
becomes the memory images, command and expected output the board needs.
Used by host/run_layer.py and by the top-level TB, so the board and the
simulation load identical images."""

from conv_layer_model import (HwParams, Layer, expected_output_words, pack_act, pack_b, pack_p)
from protocol import HwInfo, MemImages, memory_images


def hw_params(hw: HwInfo) -> HwParams:
    """the model's view of the bitstream, from PARAMS0/PARAMS1"""
    return HwParams(n=hw.n, addr_width=hw.addr_width, nbuf=hw.nbuf, k_max=hw.k_max,
                    b_addr_w=hw.b_addr_w, p_addr_w=hw.p_addr_w)


def layer_images(hw: HwInfo, layer: Layer, with_act: bool = True) -> MemImages:
    """input image (unless a previous layer left it in place), B and P"""
    n = hw.n
    act = [(layer.src_sel, layer.src_base, pack_act(layer.x, n, layer.zp_in))] if with_act else []
    return memory_images(hw, act=act, b=[(layer.b_base, pack_b(layer.w, n))],
                         p=[(layer.p_base, pack_p(layer, n))])


def expected(layer: Layer, n: int):
    """the destination words the board must hold afterwards"""
    return expected_output_words(layer, n)
