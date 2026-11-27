# sequencer contract

`sequencer/rtl/sequencer.sv` runs a whole network without the host: it
fetches descriptors from a descriptor RAM, drives each one onto
layer_engine's command bus, waits for done, records the layer's cycle count,
and moves on until an END descriptor. The host's only per-network work is
loading the memories and the descriptors, starting the program, and reading
results. Every timing figure is measured in simulation.

## Decisions made under rule 9

### 1. Descriptor storage: wide RAM or 32-bit RAM fetched over 16 cycles

| | 512-bit-wide descriptor RAM | 32-bit-wide RAM, 16 reads per descriptor (**chosen**) |
| --- | --- | --- |
| BRAM for 32 descriptors | 512 / 72 -> 8 BRAM36 | 512 words x 32 b -> 1 BRAM18 |
| fetch time | 1 read | 16 reads + READ_LATENCY (~18 cycles) |
| CSR write path | needs staging to 512 bits | one CSR word = one RAM word, no staging |

**Chosen: 32-bit RAM.** About 18 cycles of fetch per layer is noise next to
a layer (hundreds to millions of cycles), and it saves 7 BRAM36. That
matters now that BRAM is the second-tightest resource (~40 of 50).

### 2. Descriptor size: 16 words

8 words would hold every phase-3 field. 16 leaves the weight-source fields
of phase 4 (DDR addresses for B and P) room without changing the format or
the RAM geometry again. Words 7-15 are reserved and must be 0 in phase 3.

### 3. The legacy single-conv command

The single-conv CSR command and CTRL.start stay exactly as they are (op =
CONV), so `host/run_layer.py` and every existing host-path test still pass.
While the sequencer is idle, the legacy command drives layer_engine. While
the sequencer runs, it owns the bus and the CSR refuses CTRL.start.

## Descriptor format (16 little-endian 32-bit words)

| word | bits | field |
| ---- | ---- | ----- |
| 0 | [2:0] | op: 0 END, 1 CONV, 2 MAX2, 3 GAVG, 4 ADD, 5-7 illegal (layer_engine E_MODE) |
| 0 | [3] | relu_en |
| 0 | [15:8] | zp_in (signed) |
| 0 | [23:16] | zp_out (signed) |
| 0 | [31:24] | zp_in2 (signed, the add's second zero point) |
| 1 | [7:0], [15:8], [23:16], [31:24] | h_in, w_in, h_out, w_out |
| 2 | [6:0] | cin_words |
| 2 | [14:8] | cout_words |
| 2 | [18:16], [22:20] | kh, kw |
| 2 | [25:24], [29:28] | stride, pad |
| 3 | [3:0], [7:4], [11:8] | src_sel, dst_sel, src2_sel |
| 3 | [31:16] | src_base |
| 4 | [15:0] | dst_base |
| 4 | [31:16] | src2_base |
| 5 | [15:0] | b_base |
| 5 | [23:16] | p_base |
| 5 | [27:24] | add_shift |
| 6 | [15:0], [31:16] | add_ma, add_mb |
| 7 | [0] | w_ddr (phase 4): this CONV's B and P come from DDR through wdma before it runs. Ignored for other ops |
| 7 | [31:16] | k_words: K, the B words per output group (kh*kw*cin_words*N; the host computes it, so the sequencer needs no multiplier) |
| 8 | [31:0] | b_src: DDR byte address of output group 0's B (8-byte aligned) |
| 9 | [31:0] | p_src: DDR byte address of output channel 0's P entry (8-byte aligned) |
| 10-15 | | reserved, 0 |

Each field is truncated to layer_engine's port width (ADDR_WIDTH, SEL_WIDTH,
B_ADDR_W, P_ADDR_W, M_BITS), exactly as the CSR command registers are.
`host/protocol.py` builds these words (`pack_descriptor`); the TB and
`run_network.py` use that one packer.

### 4. Where the weights of a whole program live (and P_ADDR_W)

A program keeps every layer's B and P resident at once, since the host does
nothing between layers. P holds one 53-bit requant entry per output channel
(rounded up to N) of every conv in the program, so the board's P_ADDR_W = 6
(64 entries) caps a program at 64 output channels in total: not even a narrow
ResNet-8 (9 convs x 16 = 144).

| | keep P_ADDR_W = 6, stream P per layer from the host | **P_ADDR_W = 8 (256 entries, chosen)** |
| --- | --- | --- |
| cost | none in RTL; the host works per layer, which the goal forbids | 256 x 53 b = one BRAM36 (64 x 53 b was LUTRAM) |
| fits on chip | 64 channels | LeNet-5 (6+16+120+84+10 -> 248 at N=8), ResNet-8 at width 16 (144) |
| ceiling | - | the descriptor's p_base field is 8 bits, so 8 is the most without a format change |

**Chosen: P_ADDR_W = 8 in top.** B (8192 words of N bytes, 64 kB at N=8) is
then the limit: LeNet-5's weights (~8100 words at N=8) just fit, ResNet-8/20
at their usual widths (16/32/64) do not. Those are what phase 4's DMA is for:
B and P come from DDR per output group, and the on-chip B/P become a cache
of one layer.

### 5. DESC_DEPTH on the board (phase 4)

ResNet-20 is 32 layers once each residual block is split into its convs and
its add (the downsampling blocks add a 1x1 skip conv). With END that is 33
descriptors, one more than the 32 of phase 3.

| | keep 32, run ResNet-20 as two programs | **64 (chosen)** |
| --- | --- | --- |
| cost | none in RTL | descriptor RAM 1024 x 32 b = one BRAM36 (was one BRAM18): +0.5 BRAM36; cycle RAM 128 x 32 b, still LUTRAM |
| host work | the host starts the second half: a per-network step, against the goal | none |
| ceiling | - | PARAMS2 reports DESC_DEPTH in 8 bits: 128 without a map change |

**Chosen: DESC_DEPTH = 64 in top.** The sequencer itself is unchanged
(IDX_W follows the parameter), and the CSR's DESC and CYC windows size from
it. A host reads DESC_DEPTH from PARAMS2 and does not assume 32.

## Ports

| group | ports |
| ----- | ----- |
| control | `start`, `start_index[IDX_W-1:0]`, `busy`, `done` (pulse), `err` (pulse) |
| status | `cur_index`, `layers_done` (count of descriptors finished), `err_index`, `err_code[15:0]`: [9:0] from layer_engine, [10] E_NOEND, [14:11] wdma's E_DIM/E_ALIGN/E_RANGE/E_RESP, [15] E_NODMA (w_ddr set in a build without wdma) |
| descriptor RAM | write port from the CSR: `desc_we`, `desc_waddr[IDX_W+3:0]`, `desc_wdata[31:0]` |
| cycle RAM | read port to the CSR: `cyc_rd_en`, `cyc_rd_addr[IDX_W:0]`, `cyc_rd_data[31:0]` (READ_LATENCY like any bram). Entries [0, DESC_DEPTH) are layer cycles, [DESC_DEPTH, 2*DESC_DEPTH) the weight-fetch cycles of the same descriptor |
| wdma (phase 4) | `dma_start`, `dma_b_src`, `dma_p_src`, `dma_k` out (groups = cout_words, destinations = b_base and p_base, from the bus); `dma_ready`, `dma_done`, `dma_err`, `dma_err_code[3:0]` in. Unused when HAS_DMA = 0 |
| engine | the layer_engine command bus (`eng_start`, `eng_op` and every field), and the engine's `ready`, `done`, `err`, `err_code` |

`IDX_W = log2(DESC_DEPTH)`; the default DESC_DEPTH is 32.

## Behaviour

    start (idle): idx = start_index, layers_done = 0
    loop:
      FETCH   read words idx*16 .. idx*16+15 into the descriptor register
      if op == END: done, stop
      DMA     (CONV with w_ddr only) start wdma: B from b_src -> B[b_base ..], P from p_src ->
              P[p_base ..], one output group at a time; count edges from the accepting edge to
              its done -> cycle RAM[DESC_DEPTH + idx]
              wdma err -> err_index = idx, err_code[14:11] = its code, stop
              (HAS_DMA = 0: err with E_NODMA instead)
      LAUNCH  drive the fields, pulse eng_start (the engine is ready by construction)
      RUN     count edges from the accepting edge to the engine's done or err
              done -> cycle RAM[idx] = count, layers_done += 1
              err  -> err_index = idx, err_code = engine's code, err pulse, stop
      idx += 1; if idx == DESC_DEPTH: err with E_NOEND (bit 10), err_index = idx - 1, stop

* `cur_index` shows the descriptor being fetched or run.
* The cycle count uses exactly the CSR CYCLES rule: edges from the accepting
  edge to done. For each layer it therefore equals the engine formula (conv,
  pool or add).
* A descriptor's cycle-RAM word is written only when it finishes with done.
  Its fetch word (second half) is written when the fetch finishes, so a
  layer that fails after a good fetch still shows the fetch time.
* A descriptor without w_ddr runs exactly as in phase 3: same states, same
  overhead. With w_ddr, the on-chip B and P are a one-layer cache, so every
  DDR layer of a program can use b_base = p_base = 0.

## Illegal use

* **start while busy:** ignored (the CSR refuses it first, with SLVERR).
* **A descriptor with illegal fields:** the engine rejects it (its own
  checks and error bits). The sequencer stops there with that code. Every
  layer before it has run and its cycle counts are valid.
* **No END within the RAM:** E_NOEND after the last descriptor runs.
* **Reset** is synchronous. It clears the FSM and the counters, and the
  engine's reset rules apply to the layer in flight. The descriptor RAM keeps
  its contents.

## Measured (filled in from simulation)

From `sequencer/tb/test_seq_harness.py` (sweep-seq: N = 2, 4, 8 x
READ_LATENCY = 1, 2, plus DESC_DEPTH = 16), asserted on every program that
reaches END:

    start to done = sum(layer cycles) + layers * (20 + READ_LATENCY) + 17 + READ_LATENCY

* per layer descriptor, 20 + L edges outside the layer itself: 16 fetch reads
  plus L to drain the read pipeline, then decode, launch and the move to the
  next descriptor. At L = 1 that is 21 edges, under 1 % of the smallest layer
  in the test networks (a 2x2 global average, ~26 cycles, is the exception).
* the END descriptor costs 17 + L: its fetch and decode.
* each layer's cycle-RAM word equals the engine formula for that layer
  (conv_layer, pool_layer, add_layer), so per-layer counts read back through
  the CYC window are directly comparable with the models.
* through the board (top, sweep-top and sweep-net): the same per-layer counts
  read back over the UART for random networks, a LeNet-class plan and a
  ResNet basic block, byte-exact outputs for every layer.

Example, from `host/run_network.py --dry-run --plan lenet --h 32 --w 32 --c 3`
at N = 8, L = 1: seven layers, 82 455 layer cycles + 7 x 21 + 18 = 82 620
edges, 0.83 ms at 100 MHz.
