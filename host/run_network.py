#!/usr/bin/env python3
"""Run a random multi-layer CNN on the Arty with the descriptor sequencer and
compare every layer byte for byte with NumPy.

    python host/run_network.py --port /dev/tty.usbserial-XXXX1 --seed 1
    python host/run_network.py --port COM4 --plan conv3,max2,res,res,gavg --h 16 --w 16
    python host/run_network.py --port COM4 --plan lenet --h 32 --w 32 --c 3
    python host/run_network.py --port COM4 --plan resnet8 --h 16 --w 16 --c 3 --cout 8
    python host/run_network.py --dry-run --seed 1      # no board: build, size and model it

What it does, in order:
  1. reads ID, VERSION and PARAMS0/1/2; stops unless the map is 2.x and the
     bitstream has the sequencer and NBUF >= 3
  2. builds the network from the seed (host/network.py: random int8 weights
     and activations, conv / max pool / ResNet basic block / global average)
     and runs the golden models for every layer's expected words and cycles
  3. loads the input image, all B and P, and the descriptor program, each
     checked with the board's write CRCs
  4. starts the program exactly once; the board runs every layer to END with
     no host involvement and records each layer's cycle count
  5. polls SEQ_STATUS, reads SEQ_ERR, the cycle window and every layer's
     output region, and compares all of it with the model

Exit code 0 on PASS, 1 on FAIL. Random weights only: this checks the
hardware, not a trained model.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("host", "conv_layer/model", "systolic_array/model", "memory/model", "pool/model",
            "add/model", "layer_engine/model", "sequencer/model"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import numpy as np                                                  # noqa: E402

import network as NET                                               # noqa: E402
import protocol as P                                                # noqa: E402
from layers import hw_params                                        # noqa: E402
from run_layer import traffic_bytes                                 # noqa: E402
from sequencer_model import OP_END                                  # noqa: E402


def build(args, hw: P.HwInfo, rng) -> NET.Network:
    fixed = {"lenet": NET.LENET, "resnet8": NET.RESNET8}
    plan = fixed.get(args.plan) or (args.plan.split(",") if args.plan else None)
    return NET.build(rng, hw_params(hw), m_bits=hw.m_bits, plan=plan, h=args.h, w=args.w,
                     c=args.c, read_latency=hw.read_latency, desc_depth=hw.desc_depth,
                     cout=args.cout)


def describe(net: NET.Network):
    shown = [s for s in net.steps if s.op != OP_END]
    return f"{len(shown)} layers: " + " ".join(net.names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port of the Arty's FT2232H (the higher-numbered one)")
    ap.add_argument("--baud", type=int, default=1_000_000, help="must match the bitstream's BAUD")
    ap.add_argument("--timeout-bits", type=int, default=1000, help="the bitstream's TIMEOUT_BITS")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plan", help="lenet, resnet8, or a comma list of conv3, conv1, conv3s2, "
                                   "conv5, fc, max2, res, gavg (default: random from the seed)")
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--w", type=int, default=8)
    ap.add_argument("--c", type=int, default=None, help="input channels (default N)")
    ap.add_argument("--cout", type=int, default=None,
                    help="output channels of every conv (default random 1..2N; all B and P "
                         "must fit on chip: P holds 2^P_ADDR_W / N output-channel words")
    ap.add_argument("--first", type=int, default=0, help="descriptor index to load the program at")
    ap.add_argument("--dry-run", action="store_true",
                    help="no board: assume the default N=8 map-2.0 bitstream, build and model it")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    stats = P.LinkStats()

    if args.dry_run:
        hw = P.HwInfo(n=8, read_latency=1, addr_width=12, nbuf=3, b_addr_w=13, p_addr_w=8,
                      k_max=1024, version=P.VERSION, m_bits=8, desc_depth=32, has_dma=0)
        net = build(args, hw, rng)
        imgs = NET.images(hw, net)
        out_words = sum(c for _, _, c in net.outputs)
        nbytes = traffic_bytes(imgs, out_words, hw)
        nbytes += len(P.pkt_burst_write(0, [0] * P.DESC_WORDS)) * len(net.descriptors)
        e = net.expected
        print(f"dry run, assumed {hw}")
        print(f"network, seed {args.seed}: {describe(net)}")
        for i, (name, (bank, base, cnt)) in enumerate(zip(net.names, net.outputs)):
            print(f"  layer {i:2d} {name:12s} -> bank {bank} @ {base:5d}, {cnt:5d} words, "
                  f"{e.cycles[i]:7d} cycles (model)")
        total = sum(e.cycles)
        print(f"layer cycles {total} = {total / 100e6 * 1e3:.3f} ms at 100 MHz (sequencer overhead "
              f"is about {20 + hw.read_latency} cycles per layer on top)")
        print(f"load + program + readback: about {nbytes} bytes, {nbytes * 10 / args.baud:.2f} s "
              f"at {args.baud} baud")
        print(f"write CRCs act/b/p: {imgs.act_crc:#06x} {imgs.b_crc:#06x} {imgs.p_crc:#06x}")
        return 0

    if not args.port:
        ap.error("--port is required (or use --dry-run)")
    from board import SerialTransport
    t = SerialTransport(args.port, args.baud, args.timeout_bits)
    try:
        hw, build_id = P.run_sync(P.identify(stats), t)
        print(f"board: ID ok, build {build_id:#09x}, {hw}")
        if hw.version >> 16 < 2 or hw.desc_depth == 0:
            print("this bitstream has no descriptor sequencer (map 1.x): rebuild from this tree")
            return 1
        if hw.nbuf < 3:
            print(f"NBUF={hw.nbuf}: residual blocks need three act_buffer banks")
            return 1
        net = build(args, hw, rng)
        if args.first + len(net.descriptors) > hw.desc_depth:
            print(f"{len(net.descriptors)} descriptors from index {args.first} do not fit "
                  f"DESC_DEPTH={hw.desc_depth}")
            return 1
        print(f"network, seed {args.seed}: {describe(net)}")
        t0 = time.monotonic()
        try:
            res = P.run_sync(NET.board_run(hw, net, args.first, stats), t)
        except P.ProtocolFault as e:
            print(f"PROTOCOL FAULT: {e}")
            print("FAIL")
            return 1
        dt = time.monotonic() - t0
    finally:
        t.close()

    ok, lines = NET.compare(net, res)
    print("\n".join(lines))
    total = sum(res.program.cycles)
    print(f"SEQ_STATUS {res.program.status:#x}, {res.program.layers_done} layers, "
          f"{total} layer cycles = {total / 100e6 * 1e3:.3f} ms at 100 MHz")
    print(f"link: {t.bytes_out} bytes out, {t.bytes_in} in, {dt:.2f} s, {stats.retries} retries, "
          f"{stats.reloads} reloads, statuses {stats.statuses}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
