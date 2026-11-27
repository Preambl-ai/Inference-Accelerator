#!/usr/bin/env python3
"""Run a random multi-layer CNN on the Arty with the descriptor sequencer and
compare every layer byte for byte with NumPy.

    python host/run_network.py --port /dev/tty.usbserial-XXXX1 --seed 1
    python host/run_network.py --port COM4 --plan conv3,max2,res,res,gavg --h 16 --w 16
    python host/run_network.py --port COM4 --plan lenet --h 32 --w 32 --c 3
    python host/run_network.py --port COM4 --plan resnet8 --h 32 --w 32 --c 3      # DDR weights
    python host/run_network.py --port COM4 --plan resnet20 --h 32 --w 32 --c 3
    python host/run_network.py --port COM4 --plan conv3,res,gavg --weights mixed
    python host/run_network.py --dry-run --seed 1      # no board: build, size and model it

What it does, in order:
  1. reads ID, VERSION and PARAMS0/1/2; stops unless the map is 2.x and the
     bitstream has the sequencer and NBUF >= 3
  2. builds the network from the seed (host/network.py: random int8 weights
     and activations, conv / max pool / ResNet basic block / global average)
     and runs the golden models for every layer's expected words and cycles
  3. loads the input image, the on-chip B and P, the DDR weight images (map
     2.1, through the DDR window) and the descriptor program, each checked
     with the board's write CRCs
  4. starts the program exactly once; the board runs every layer to END with
     no host involvement (fetching DDR layers' weights itself) and records
     each layer's cycle count and weight-fetch time
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
            "add/model", "layer_engine/model", "sequencer/model", "dma/model"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import numpy as np                                                  # noqa: E402

import network as NET                                               # noqa: E402
import protocol as P                                                # noqa: E402
from layers import hw_params                                        # noqa: E402
from run_layer import traffic_bytes                                 # noqa: E402
from sequencer_model import OP_END                                  # noqa: E402


FIXED = {"lenet": NET.LENET, "resnet8": NET.RESNET8, "resnet20": NET.RESNET20}
RESNETS = ("resnet8", "resnet20")


def resolve(args):
    """defaults that depend on the plan: the ResNets keep their weights in DDR
    and reuse activation space (they do not fit on chip otherwise)"""
    if args.weights is None:
        args.weights = "ddr" if args.plan in RESNETS else "chip"
    if args.keep is None:
        args.keep = "last" if args.plan in RESNETS else "all"


def build(args, hw: P.HwInfo, rng) -> NET.Network:
    plan = FIXED.get(args.plan) or (args.plan.split(",") if args.plan else None)
    return NET.build(rng, hw_params(hw), m_bits=hw.m_bits, plan=plan, h=args.h, w=args.w,
                     c=args.c, read_latency=hw.read_latency, desc_depth=hw.desc_depth,
                     cout=args.cout, weights=args.weights, keep=args.keep)


def describe(net: NET.Network):
    shown = [s for s in net.steps if s.op != OP_END]
    return f"{len(shown)} layers: " + " ".join(net.names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port of the Arty's FT2232H (the higher-numbered one)")
    ap.add_argument("--baud", type=int, default=1_000_000, help="must match the bitstream's BAUD")
    ap.add_argument("--timeout-bits", type=int, default=1000, help="the bitstream's TIMEOUT_BITS")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plan", help="lenet, resnet8, resnet20, or a comma list of conv3, conv1, "
                                   "conv3s2, conv5, fc, max2, res, resd, gavg (default: random "
                                   "from the seed)")
    ap.add_argument("--weights", choices=("chip", "ddr", "mixed"), default=None,
                    help="where each conv's B and P live: loaded on chip, fetched from DDR by "
                         "the board (needs a map 2.1 bitstream with DMA), or a random mix "
                         "(default: ddr for the ResNets, chip otherwise)")
    ap.add_argument("--keep", choices=("all", "last"), default=None,
                    help="all: every layer's output stays in the act_buffer and is checked; "
                         "last: activation space is reused, only surviving outputs are read "
                         "back (default: last for the ResNets, all otherwise)")
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--w", type=int, default=8)
    ap.add_argument("--c", type=int, default=None, help="input channels (default N)")
    ap.add_argument("--cout", type=int, default=None,
                    help="output channels of every conv (default random 1..2N; all B and P "
                         "must fit on chip: P holds 2^P_ADDR_W / N output-channel words")
    ap.add_argument("--first", type=int, default=0, help="descriptor index to load the program at")
    ap.add_argument("--dry-run", action="store_true",
                    help="no board: assume the default N=8 bitstream (DESC_DEPTH 64, DMA when "
                         "--weights is not chip), build and model it")
    args = ap.parse_args()
    resolve(args)
    rng = np.random.default_rng(args.seed)
    stats = P.LinkStats()

    if args.dry_run:
        hw = P.HwInfo(n=8, read_latency=1, addr_width=12, nbuf=3, b_addr_w=13, p_addr_w=8,
                      k_max=1024, version=P.VERSION, m_bits=8, desc_depth=64,
                      has_dma=int(args.weights != "chip"))
        net = build(args, hw, rng)
        imgs = NET.images(hw, net)
        out_words = sum(net.outputs[i][2] for i in net.checked)
        nbytes = traffic_bytes(imgs, out_words, hw)
        nbytes += len(P.pkt_burst_write(0, [0] * P.DESC_WORDS)) * len(net.descriptors)
        ddr_bytes = sum(len(d) for _, d in net.ddr_images)
        nbytes += ddr_bytes * len(P.pkt_burst_write(0, [0] * 64)) // 256
        e = net.expected
        print(f"dry run, assumed {hw}")
        print(f"network, seed {args.seed}: {describe(net)}")
        for i, (name, (bank, base, cnt)) in enumerate(zip(net.names, net.outputs)):
            fetch = f", weights from DDR in {e.fetch_cycles[i]:6d}" if e.fetch_cycles[i] else ""
            print(f"  layer {i:2d} {name:12s} -> bank {bank} @ {base:5d}, {cnt:5d} words, "
                  f"{e.cycles[i]:7d} cycles (model){fetch}")
        total, fetch = sum(e.cycles), sum(e.fetch_cycles)
        print(f"layer cycles {total} + weight fetch {fetch} (AXI latency 1, no stalls) = "
              f"{(total + fetch) / 100e6 * 1e3:.3f} ms at 100 MHz (sequencer overhead is about "
              f"{20 + hw.read_latency} cycles per layer on top)")
        if ddr_bytes:
            print(f"DDR weight images: {len(net.ddr_images)} regions, {ddr_bytes} bytes")
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
        if args.weights != "chip" and not hw.has_dma:
            print(f"--weights {args.weights} needs a bitstream with DMA (PARAMS2[16]); this one "
                  f"has none: use --weights chip, or rebuild with HAS_DMA=1 and the MIG")
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
    fetch = sum(res.program.fetch_cycles or [])
    print(f"SEQ_STATUS {res.program.status:#x}, {res.program.layers_done} layers, "
          f"{total} layer cycles + {fetch} weight-fetch cycles = "
          f"{(total + fetch) / 100e6 * 1e3:.3f} ms at 100 MHz")
    print(f"link: {t.bytes_out} bytes out, {t.bytes_in} in, {dt:.2f} s, {stats.retries} retries, "
          f"{stats.reloads} reloads, statuses {stats.statuses}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
