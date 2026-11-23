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
| BRAM36 | ~32 | 50 | act_buffer: 2 banks of 4096 x 64 b = 8 BRAM36 each. B: 8192 x 64 b = 16. P: 64 x 53 b goes to LUTRAM or one BRAM18 |
| LUT | ~7-10 k | 20 800 | the bulk is requant (shift and clamp per channel), im2col (N-word slice buffer and its address logic), the systolic array's skew registers, the CSR read mux and its 64-bit CRC XOR trees |
| FF | ~9 k | 41 600 | PE accumulators (64 x 32 b) and requant pipeline registers dominate |

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
