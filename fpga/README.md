# Arty A7-35T bring-up

The first on-hardware result: run one INT8 conv layer on the board and match
it byte for byte against the NumPy model.

    PC (host/run_layer.py, pyserial) -> USB-UART -> top.sv (uart, bridge, csr) -> conv_layer

The protocol and the CSR map are in `top/host_interface_contract.md`. Every
part of this path is simulated end to end through the UART pins by
`make sweep-top`, using the same packet code as the host script
(`host/protocol.py`). The Vivado steps below have **not** been run as part of
that work (the development machine cannot run Vivado). The resource and
timing figures are a paper estimate until you run them.

## 1. Build the bitstream (Windows or Linux, Vivado 2020.2 or later)

From the repository root:

    vivado -mode batch -source fpga/build.tcl -tclargs 8 1 1 1000000

The arguments are N, READ_LATENCY, CLK_DIV and BAUD (the defaults are the
values shown). The part defaults to `xc7a35ticsg324-1L`, the one Digilent's
Arty A7-35 board files use. If the chip on your board reads XC7A35T-1CSG324C,
set `PART=xc7a35tcsg324-1` in the environment first. This writes `fpga/build/top.bit` and reports into
`fpga/reports/`. The script exits with status 1 and prints `TIMING NOT MET`
if setup or hold slack is negative. In that case, see section 5.

Check before using the bitstream:
* `fpga/reports/timing_summary.rpt`: "All user specified timing constraints are met"
* `fpga/reports/utilization.rpt`: DSPs ≤ 90, Block RAM Tile ≤ 50
* `fpga/reports/drc.rpt`: no errors

## 2. Program the board over USB

Connect the Arty's micro-USB port (J10). Then either use the GUI:

    vivado &    # Open Hardware Manager -> Open Target -> Auto Connect
                # -> Program Device -> fpga/build/top.bit

or batch mode:

    vivado -mode tcl
    open_hw_manager; connect_hw_server; open_hw_target
    set_property PROGRAM.FILE fpga/build/top.bit [current_hw_device]
    program_hw_devices [current_hw_device]

Once programmed, LD4 blinks (heartbeat, about 0.75 Hz at 100 MHz). LD5 is
busy, LD6 done, and LD7 err. Done and err are sticky until the next start.
BTN0 resets the whole design.

## 3. Find the serial port

The FT2232H shows up as two ports: channel A is JTAG, channel B is the UART.
Use the **second** (higher-numbered) port.

* macOS: `/dev/tty.usbserial-XXXXXXXX1` (the one ending in 1)
* Linux: `/dev/ttyUSB1` (add yourself to the `dialout` group)
* Windows: the higher of the two COM ports in Device Manager

## 4. Run a layer

    pip install pyserial numpy
    python host/run_layer.py --port /dev/tty.usbserial-XXXXXXXX1 --seed 1

The script works in this order:
1. Reads the ID register first, and stops at once if it is not 0x434E5631
   (wrong board or bitstream).
2. Prints BUILD_ID (the git hash the bitstream was built from) and the
   parameters it was built with.
3. Builds the layer from the seed, loads it, checks the board's write CRCs,
   starts it exactly once, polls, and reads back.
4. Prints `PASS` or `FAIL`, the cycle count from the CSR, and the cycle count
   the conv_layer formula predicts. They must be equal.

Other layers, for example the realistic one from the contract (0.7 s of link
time at 1 Mbaud):

    python host/run_layer.py --port ... --h 16 --w 16 --cin 64 --cout 64 --kh 3 --kw 3 --pad 1
    python host/run_layer.py --dry-run --h 16 --w 16 --cin 64 --cout 64   # no board: sizes only

### Baud rate and the FTDI latency timer

* Baud defaults to 1,000,000: 100 MHz / 100 and the FT2232H's 3 MHz / 3 are
  both exact. If a USB hub or cable gives trouble, rebuild with `BAUD=115200`
  and pass `--baud 115200`. Both sides have to match.
* With `CLK_DIV=2` the divisor is 50 at 1 Mbaud, which is still exact.
* **Latency timer: we batch instead of tuning it.** The FTDI driver holds
  received bytes for up to its latency timer (16 ms by default) before sending
  them to the PC, so every request/response round trip can cost 16 ms.
  pyserial cannot change that timer portably; on macOS there is no knob for
  it at all. The host therefore keeps round trips few: all 19 command
  registers go in one burst, and memory goes in 256-word bursts. A whole layer
  is a few dozen round trips. If you want it faster on Linux:

      echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB1/latency_timer

  On Windows, set Device Manager -> the COM port -> Port Settings -> Advanced
  -> Latency Timer to 1 ms.

## 5. Paper resource and timing estimate (N = 8, XC7A35T)

These figures are counted from the RTL, not from a Vivado run. Check them
against `fpga/reports/utilization.rpt`.

| resource | estimate | available | notes |
| -------- | -------- | --------- | ----- |
| DSP48E1 | **~86-90** | 90 | 64 PE multiply-accumulates (8x8 into 32 bits, one per PE). 16 for the 8 requants (32x17 products, two DSPs each; that is why M_WIDTH is 16). 2 for the im2col address multiplies (y*w_in, then *cin_words). About 4-6 for conv_layer's check phase (h*w, (h*w)*c_words, cout*K). **At risk.** |
| BRAM36 | ~32 | 50 | act_buffer: 2 banks of 4096 x 64 b = 8 BRAM36 each. B: 8192 x 64 b = 16. P: 64 x 53 b goes to LUTRAM or one BRAM18 (phase 3 makes it 256 x 53 b, below) |
| LUT | ~7-10 k | 20 800 | the bulk is requant (shift and clamp per channel), im2col (N-word slice buffer and its address logic), the systolic array's skew registers, the CSR read mux and its 64-bit CRC XOR trees |
| FF | ~9 k | 41 600 | PE accumulators (64 x 32 b) and requant pipeline registers dominate |

**Added by phase 1 (pooling).** These are estimates until you run Vivado.

| block | DSP | BRAM36 | LUT | FF | notes |
| ----- | --- | ------ | --- | -- | ----- |
| pool_layer | 0 | 0 | ~0.7-0.9 k | ~0.5 k | No multipliers. Per lane at N=8: an 8-bit signed compare and a 24-bit adder with its rounding shifter. Bit-serial shift-and-add check phase. 24-bit lane accumulators (192 FF), a 64-bit write register, counters and pointers |
| layer_engine | 0 | 0 | ~0.15 k | ~0.01 k | A 3:1 mux on the act_buffer read and write ports (64 b data + 12 b address), dispatch, and the bad-op flag |

Running total at N=8: **DSP unchanged (~86-90)**, BRAM unchanged (~32),
LUT ~8-11 k, FF ~9.5 k. The new timing paths are short: the pool compare
and add sit between registers, and the owner mux sits in front of the BRAM
ports, which the act_buffer read already crossed through conv_layer's host
mux.

**Added by phase 2 (residual add).** These are estimates until you run Vivado.

| block | DSP | BRAM36 | LUT | FF | notes |
| ----- | --- | ------ | --- | -- | ----- |
| add_layer | 0 | 0 | ~0.7-0.9 k | ~0.7 k | No multipliers: the rescale is bit-serial. Per lane at N=8: a 19-bit three-input add, and a round, zero point and clamp stage. `a - za` and `b - zb` are held in shifting registers (no barrel shifter). 3 x 19-bit registers per lane, plus a 64-bit a-word and the write register |
| act_buffer, third bank (top NBUF 2 -> 3) | 0 | **+8** | ~0.05 k | 0 | 4096 x 64 b per bank at N=8. ResNet needs the input (skip), the branch and the sum in three banks |
| layer_engine, 4-way owner mux | 0 | 0 | ~0.05 k | 0 | |

Running total at N=8: **DSP unchanged (~86-90)**, **BRAM ~40 of 50**, LUT
~9-12 k, FF ~10.5 k. BRAM is now the second resource to watch, after DSP.
If it runs short, the first step is ADDR_WIDTH 12 -> 11 (each bank halves,
-12 BRAM36), which still holds a 32x32x16 ResNet-20 map at N = 8 (2048 words).

**Added by phase 3 (descriptor sequencer).** These are estimates until you run Vivado.

| block | DSP | BRAM36 | LUT | FF | notes |
| ----- | --- | ------ | --- | -- | ----- |
| sequencer | 0 | **+0.5** (one BRAM18) | ~0.3-0.4 k | ~0.35 k | No multipliers: a descriptor's address is idx << 4. Descriptor RAM 512 x 32 b (32 descriptors) is one BRAM18. The cycle RAM, 32 x 32 b, goes to LUTRAM. A 16 x 32-bit descriptor register (only the ~150 bits in use survive synthesis), a 32-bit edge counter, and a 5-bit tag pipeline |
| P memory, P_ADDR_W 6 -> 8 | 0 | **+1** | ~-0.1 k | 0 | 256 x 53 b moves from LUTRAM to one BRAM36, so a whole LeNet-5 keeps its requant entries on chip (sequencer contract, decision 4) |
| csr, map 2.0 | 0 | 0 | ~0.2 k | ~0.1 k | Five more registers on the read mux, a 32-bit CRC-16 XOR tree for DESC_WCRC, and the cycle-window read through the existing READ_LATENCY path |
| top, engine bus mux | 0 | 0 | ~0.1 k | 0 | Sequencer or legacy CSR command onto layer_engine, switched by the sequencer's busy |

Running total at N=8: **DSP unchanged (~86-90)**, **BRAM ~41.5 of 50**, LUT
~10-13 k, FF ~11 k. The sequencer is off the datapath: its fields are
registered and sit on the engine bus for many cycles before eng_start, so it
adds no critical path.

**Added by phase 4 (weights from DDR).** These are estimates until you run Vivado.
The default build (`HAS_DMA` = 0) gets only the descriptor RAM change. wdma,
the CSR's DDR window and the MIG come in with `HAS_DMA` = 1 (section 6).

| block | DSP | BRAM36 | LUT | FF | notes |
| ----- | --- | ------ | --- | -- | ----- |
| descriptor RAM, DESC_DEPTH 32 -> 64 | 0 | **+0.5** | ~0.05 k | ~0.01 k | 1024 x 32 b is one BRAM36 (was one BRAM18). ResNet-20 is 33 descriptors with END. The cycle RAM doubles twice (64 entries, plus the fetch half): 128 x 32 b, still LUTRAM |
| wdma (HAS_DMA = 1) | 0 | 0 | ~0.4-0.6 k | ~0.45 k | No multipliers. The burst planner is adds, compares and shifts. Four 32-bit address counters, the 64-bit R beat into the B write register, the 53-bit P word, and the range checks at their own widths (RB_W/RP_W). One AR outstanding, so no FIFO |
| csr, map 2.1 (HAS_DMA = 1) | 0 | 0 | ~0.15 k | ~0.12 k | DDR_BASE page, the DDR window's request/ack handshake, DDR_WCRC (another 32-bit CRC-16 XOR tree) |
| top, B/P write mux | 0 | 0 | ~0.1 k | 0 | wdma or the CSR onto the B and P write ports, switched by wdma's busy |
| MIG 7-series, DDR3L x16 (when added) | 0 | ~0-1 | **~5-6 k** | **~6-7 k** | Typical for one 16-bit DDR3 with an AXI4 slave; Vivado's report after generating it is the number to trust. It also takes an MMCM/PLL and the DDR3 bank's IO |

Running total at N=8: default build **DSP ~86-90, BRAM ~42 of 50**, LUT ~10-13 k,
FF ~11 k. With HAS_DMA = 1 and the MIG: BRAM ~42-43, **LUT ~16-20 k of
20.8 k**, FF ~18 k. **LUT becomes the resource to watch.** If it does not fit,
build N=4 with DMA: N=4 frees about a quarter of the datapath LUTs and
three-quarters of the PE DSPs. wdma is off the datapath. Its only path into
the engine is the B/P write mux, which sits in front of the BRAM write
ports, register to register.

**DSP is the resource that decides whether N=8 fits.** If utilization shows
more than 90, or placement fails on DSPs, try these in order:
1. **Cap synthesis:** `-tclargs 8 1 1 1000000 84`. This passes
   `-max_dsp 84`, and Vivado moves the cheapest multipliers to fabric. Those
   are the check-phase and address multiplies, which run once per layer or
   once per tile and are nowhere near critical.
2. **Drop N to 4:** `-tclargs 4 1 1 1000000`. That is 16 PEs + 8 requant DSPs.
   N is the parameter to give up first for fit; the host reads N from PARAMS0
   and adapts.

**Timing at 100 MHz.** Paths at risk, most likely first:
1. **gemm_tile's `stall`** (skid buffer full OR feeder late) is the clock
   enable of the whole array, both memory read ports and every requant stage.
   It is a few-thousand-load net from two flops, so routing on that net is
   the likely critical path.
2. **The act_buffer read at READ_LATENCY=1:** BRAM clock-to-out with no
   output register, then the bank mux, then the N-way write into im2col's
   slice buffer.
3. **The requant 32x17 multiply** (two cascaded DSPs) between stages 1 and 2,
   if Vivado does not pull the stage registers into the DSPs.

If timing fails, change these first:
1. **READ_LATENCY=2** turns on the BRAM output registers. It costs a few
   cycles per slice group, which the cycle formula and the TB already cover.
2. **CLK_DIV=2** (50 MHz through the MMCM). This is close to free here: a
   realistic layer computes in about 3.6 ms at 100 MHz but takes about 700 ms
   on the UART at 1 Mbaud, so halving the clock adds about 1 % to the time you
   wait.

Either way, N is the parameter that buys fit. Clock speed and READ_LATENCY
buy timing.

## 6. DDR3 through the MIG (weights from DDR, HAS_DMA = 1)

Phase 4 adds wdma, which fetches a layer's B and P from DDR over AXI4 before
the layer runs. That is how ResNet-8 and ResNet-20 run: their weights do not
fit on chip. The RTL and its verification are complete against an AXI4 slave
model (dma/tb/axi4_slave.py), which checks every AXI rule wdma relies on, with
latency and backpressure. **What is not in the repository is the MIG itself.**
`dma/rtl/ddr_port.sv` marks its place. In simulation the slave model drives
it; in synthesis it stops a `HAS_DMA = 1` build with a `$fatal` until it is
replaced. The steps, in Vivado, none of them verified here:

1. **Generate the MIG** (IP catalog, Memory Interface Generator, 7 Series):
   DDR3 SDRAM, the Arty A7's part (MT41K128M16JT-125, 16 bits, 256 MB). Digilent's
   Arty A7 board files fill in the pins and timing; without them, use
   Digilent's published MIG project file (.prj) for the board. Memory clock
   325-333 MHz (4:1), **AXI4 interface, data width 64, ID width 4**. Those
   are wdma's widths: `AXI_IDW` = 4 and 8-byte beats, and the MIG's AXI
   width converter handles the 128-bit native side. Address width 28 (256
   MB).
2. **Clocks.** The MIG wants a system clock (166.67 MHz on the Arty) and a
   200 MHz reference clock. Make both from CLK100MHZ with one MMCM, outside
   the MIG. The AXI side of the MIG runs on its `ui_clk` (memory clock / 4,
   about 81-83 MHz), not on the design's clock. Pick one of:
   - **Run the whole design on `ui_clk`** (simplest: one clock domain, as
     now). Replace `clk` in top with `ui_clk`, hold `sys_reset` while
     `init_calib_complete` is low, and set `CLKS_PER_BIT` from the real
     frequency. At 1 Mbaud an 83.33 MHz clock is not an integer divisor; use
     a baud rate that divides it, and pass it to the host as `--baud`.
   - Keep the design at 100 MHz and put an **AXI Clock Converter** IP between
     wdma and the MIG. This costs ~1 k LUT more and adds a real clock-domain
     crossing to constrain.
3. **Replace ddr_port.** Instantiate the generated MIG in place of `u_ddr` in
   `top.sv`'s `g_dma` branch: the same AXI4 signal names, its `aresetn` from
   `!sys_reset`, and its DDR3 pins (`ddr3_dq`, `ddr3_dqs_p/n`, `ddr3_addr`,
   `ddr3_ba`, `ddr3_ras_n`, `ddr3_cas_n`, `ddr3_we_n`, `ddr3_reset_n`,
   `ddr3_ck_p/n`, `ddr3_cke`, `ddr3_cs_n`, `ddr3_dm`, `ddr3_odt`) brought out
   as ports of top. The MIG's own XDC sets their pins and IO standards; add
   it to build.tcl with `read_ip` or `read_xdc`. The MIG's write channel is
   wired already: the host reaches DDR through the CSR's DDR window as single
   AXI beats.
4. **Build** with `-tclargs 8 1 1 <baud> -1 1` (the sixth argument is
   HAS_DMA). Check `PARAMS2[16]` reads 1 on the board, then run
   `python host/run_network.py --port ... --plan resnet8 --h 32 --w 32 --c 3`.
   A first DDR test without the CNN: write and read back a few words through
   the DDR window (`protocol.load_ddr` checks the write CRC).

Things to expect on the board that the simulation cannot show. Fetch times
depend on refresh and the MIG's real latency. The cycle window's second half
reports them per layer, and the ResNets are compute-bound anyway: weight
fetch is 2.0 % of ResNet-20's cycles at AXI latency 1, and 2.5 % at 30
(network.py's model, 32x32x3, N=8). Calibration also has to finish before the first DDR access.
`init_calib_complete` belongs in the reset, as above.
