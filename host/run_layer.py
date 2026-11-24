#!/usr/bin/env python3
"""Run one INT8 conv layer on the Arty and compare it byte for byte with NumPy.

    python host/run_layer.py --port /dev/tty.usbserial-XXXX1 --seed 1
    python host/run_layer.py --port COM4 --h 16 --w 16 --cin 64 --cout 64 --kh 3 --pad 1
    python host/run_layer.py --dry-run --seed 1        # no board: build and size the traffic

What it does, in order:
  1. reads the ID register (stops at once on a wrong board or bitstream), the
     map version, PARAMS0/1 and BUILD_ID
  2. builds the layer from the seed with the conv_layer golden model
     (random int8 tensors, requant parameters sized to the layer)
  3. loads input image, B and P, checks the board's write CRCs
  4. writes the command in one burst, starts it exactly once (LAUNCH_COUNT)
  5. polls STATUS, reads ERR_CODE and CYCLES, reads the output back
  6. compares with conv_ref, and CYCLES with the conv_layer cycle formula

Exit code 0 on PASS, 1 on FAIL. The packets come from host/protocol.py, the
same code the cocotb end-to-end test drives the simulated board with.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("host", "conv_layer/model", "systolic_array/model", "memory/model"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import numpy as np                                                  # noqa: E402

import protocol as P                                                # noqa: E402
from conv_layer_model import (Layer, check_command, err_names, estimate_cycles,   # noqa: E402
                              out_size, requant_params, words)
from layers import expected, hw_params, layer_images                # noqa: E402


def build_layer(args, hw: P.HwInfo, rng) -> Layer:
    x = rng.integers(-128, 128, (args.h, args.w, args.cin))
    w = rng.integers(-128, 128, (args.kh, args.kw, args.cin, args.cout))
    bias, mult, shift = requant_params(rng, args.cout, args.kh * args.kw * args.cin)
    layer = Layer(x=x, w=w, bias=bias, mult=mult, shift=shift, zp_in=args.zp_in,
                  zp_out=args.zp_out, relu_en=int(args.relu), stride=args.stride, pad=args.pad,
                  src_sel=0, dst_sel=1, src_base=0, dst_base=0, b_base=0, p_base=0)
    code = check_command(layer.command(hw.n), hw_params(hw))
    if code:
        raise SystemExit(f"this layer does not fit the bitstream ({hw}): {err_names(code)}")
    return layer


def traffic_bytes(images: P.MemImages, out_words: int, hw: P.HwInfo) -> int:
    """bytes on the wire for load + readback, burst overhead included"""
    total = 0
    for _, ws in images.act + images.b + images.p:
        for i in range(0, len(ws), P.MAX_BURST):
            n = min(P.MAX_BURST, len(ws) - i)
            total += len(P.pkt_burst_write(0, [0] * n)) + P.response_len()
    slots = out_words << P.slot_shift(hw.word_w)
    for i in range(0, slots, P.MAX_BURST):
        n = min(P.MAX_BURST, slots - i)
        total += len(P.pkt_burst_read(0, n)) + P.response_len(n)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port of the Arty's FT2232H (the higher-numbered one)")
    ap.add_argument("--baud", type=int, default=1_000_000, help="must match the bitstream's BAUD")
    ap.add_argument("--timeout-bits", type=int, default=1000, help="the bitstream's TIMEOUT_BITS")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--w", type=int, default=8)
    ap.add_argument("--cin", type=int, default=16)
    ap.add_argument("--cout", type=int, default=16)
    ap.add_argument("--kh", type=int, default=3)
    ap.add_argument("--kw", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--zp-in", type=int, default=-3)
    ap.add_argument("--zp-out", type=int, default=5)
    ap.add_argument("--relu", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="no board: assume N=8 default parameters, build everything, report sizes")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    stats = P.LinkStats()

    if args.dry_run:
        hw = P.HwInfo(n=8, read_latency=1, addr_width=12, nbuf=2, b_addr_w=13, p_addr_w=6, k_max=1024)
        layer = build_layer(args, hw, rng)
        cmd = layer.command(hw.n)
        imgs = layer_images(hw, layer)
        exp = expected(layer, hw.n)
        nbytes = traffic_bytes(imgs, len(exp), hw)
        print(f"dry run, assumed {hw}")
        print(f"layer {args.h}x{args.w}x{args.cin} -> {cmd['h_out']}x{cmd['w_out']}x{args.cout}, "
              f"k{args.kh}x{args.kw} s{args.stride} p{args.pad}, seed {args.seed}")
        print(f"load + readback: {nbytes} bytes on the wire, {nbytes * 10 / args.baud:.2f} s at "
              f"{args.baud} baud; layer {estimate_cycles(cmd, hw.n, hw.read_latency)} cycles")
        print(f"write CRCs act/b/p: {imgs.act_crc:#06x} {imgs.b_crc:#06x} {imgs.p_crc:#06x}")
        return 0

    if not args.port:
        ap.error("--port is required (or use --dry-run)")
    from board import SerialTransport
    t = SerialTransport(args.port, args.baud, args.timeout_bits)
    try:
        hw, build = P.run_sync(P.identify(stats), t)          # first: is this our board?
        print(f"board: ID ok, build {build:#09x}, {hw}")
        if stats.stale_dropped:
            print(f"note: STATUS.DROPPED held {stats.stale_dropped} from an earlier session "
                  f"(another program talking over the board?); cleared")
        layer = build_layer(args, hw, rng)
        cmd = layer.command(hw.n)
        exp = expected(layer, hw.n)
        imgs = layer_images(hw, layer)
        t0 = time.monotonic()
        try:
            res = P.run_sync(P.run_layer(hw, cmd, imgs, len(exp), stats), t)
        except P.ProtocolFault as e:
            print(f"PROTOCOL FAULT: {e}")
            print("FAIL")
            return 1
        dt = time.monotonic() - t0
    finally:
        t.close()

    est = estimate_cycles(cmd, hw.n, hw.read_latency)
    ok = bool(res.status & P.ST_DONE) and res.output == exp and res.cycles == est
    print(f"layer {args.h}x{args.w}x{args.cin} -> {cmd['h_out']}x{cmd['w_out']}x{args.cout} "
          f"k{args.kh}x{args.kw} s{args.stride} p{args.pad} seed {args.seed}")
    print(f"STATUS {res.status:#x}  ERR_CODE {err_names(res.err_code) or 'none'}")
    print(f"CYCLES {res.cycles} (formula {est}) = {res.cycles / 100e6 * 1e3:.3f} ms at 100 MHz")
    if res.output and res.output != exp:
        bad = [i for i, (a, b) in enumerate(zip(res.output, exp)) if a != b]
        print(f"output: {len(bad)} of {len(exp)} words differ, first at word {bad[0]}: "
              f"board {res.output[bad[0]]:#x}, NumPy {exp[bad[0]]:#x}")
    print(f"link: {t.bytes_out} bytes out, {t.bytes_in} in, {dt:.2f} s, {stats.retries} retries, "
          f"{stats.reloads} reloads, statuses {stats.statuses}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
