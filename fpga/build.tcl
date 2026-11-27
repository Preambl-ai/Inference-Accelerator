# Non-project Vivado flow for the Arty A7-35T (Artix-7 XC7A35T, CSG324 package).
#
#   vivado -mode batch -source fpga/build.tcl -tclargs [N] [READ_LATENCY] [CLK_DIV] [BAUD] [MAX_DSP] [HAS_DMA]
#
#   N             tile size, default 8
#   READ_LATENCY  1 or 2, default 1 (2 turns on the BRAM output registers)
#   CLK_DIV       1 = 100 MHz straight from the board clock (default),
#                 2 = 50 MHz through an MMCM, for when timing does not close
#   BAUD          default 1000000; the host's --baud must match
#   MAX_DSP       optional cap passed to synth_design -max_dsp (see README); -1 = none
#   HAS_DMA       0 (default) builds without wdma: weights on chip only. 1 adds
#                 wdma and the DDR port, and needs the MIG in place of
#                 dma/rtl/ddr_port.sv first (README, "DDR3 through the MIG");
#                 without it synthesis stops at ddr_port's $fatal on purpose
#
# Run it from the repository root. Outputs:
#   fpga/build/top.bit
#   fpga/reports/{utilization,utilization_hier,timing_summary,drc,clock_utilization}.rpt
# BUILD_ID is the git short hash of HEAD (0 if git is not available), and is
# readable over the UART at CSR 0x24, so the host can tell which build is on
# the board.

set N            [expr {[llength $argv] > 0 ? [lindex $argv 0] : 8}]
set READ_LATENCY [expr {[llength $argv] > 1 ? [lindex $argv 1] : 1}]
set CLK_DIV      [expr {[llength $argv] > 2 ? [lindex $argv 2] : 1}]
set BAUD         [expr {[llength $argv] > 3 ? [lindex $argv 3] : 1000000}]
set MAX_DSP      [expr {[llength $argv] > 4 ? [lindex $argv 4] : -1}]
set HAS_DMA      [expr {[llength $argv] > 5 ? [lindex $argv 5] : 0}]
# Digilent's Arty A7-35 board files name xc7a35ticsg324-1L; boards marked
# XC7A35T-1CSG324C build the same way with PART=xc7a35tcsg324-1 in the environment.
set PART [expr {[info exists ::env(PART)] ? $::env(PART) : "xc7a35ticsg324-1L"}]

set root    [file normalize [file join [file dirname [info script]] ..]]
set outdir  [file join $root fpga build]
set repdir  [file join $root fpga reports]
file mkdir $outdir $repdir

# git short hash -> 32-bit BUILD_ID (7 hex digits fit in 28 bits)
if {[catch {exec git -C $root rev-parse --short=7 HEAD} hash]} { set hash 0 }
set BUILD_ID [format "%08x" [expr {"0x$hash"}]]
set clk_mhz [expr {100 / $CLK_DIV}]
puts "build: N=$N READ_LATENCY=$READ_LATENCY CLK_DIV=$CLK_DIV (${clk_mhz} MHz) BAUD=$BAUD HAS_DMA=$HAS_DMA BUILD_ID=$BUILD_ID"
if {100000000 % ($CLK_DIV * $BAUD) != 0} {
    puts "WARNING: ${clk_mhz} MHz / $BAUD baud is not an integer divisor; the rate will be off"
}

# --- sources: everything between the UART pins and the systolic array -------
set srcs {
    top/rtl/top.sv
    csr/rtl/csr.sv
    uart/rtl/uart_rx.sv
    uart/rtl/uart_tx.sv
    uart/rtl/uart_bridge.sv
    layer_engine/rtl/layer_engine.sv
    pool/rtl/pool_layer.sv
    add/rtl/add_layer.sv
    sequencer/rtl/sequencer.sv
    dma/rtl/wdma.sv
    dma/rtl/ddr_port.sv
    conv_layer/rtl/conv_core.sv
    gemm_tile/rtl/gemm_tile.sv
    im2col/rtl/im2col_reader.sv
    memory/rtl/act_buffer.sv
    memory/rtl/bram.sv
    systolic_array/rtl/ctrl_fsm.sv
    systolic_array/rtl/systolic_array.sv
    systolic_array/rtl/pe.sv
    systolic_array/rtl/delay.sv
    systolic_array/rtl/requant.sv
    axis/rtl/skid_buffer.sv
}
# ddr_port is only elaborated when HAS_DMA = 1 (top's generate), so reading
# it always is harmless; it is where the MIG goes
foreach f $srcs { read_verilog -sv [file join $root $f] }
read_xdc [file join $root fpga arty_a7_35t.xdc]

# --- synthesis --------------------------------------------------------------
# SYNTHESIS keeps the simulation-only $fatal checks out and selects the MMCM.
set synth_args [list -top top -part $PART -verilog_define SYNTHESIS \
    -generic N=$N -generic READ_LATENCY=$READ_LATENCY -generic CLK_DIV=$CLK_DIV \
    -generic BAUD=$BAUD -generic HAS_DMA=$HAS_DMA -generic "BUILD_ID=32'h$BUILD_ID"]
if {$MAX_DSP >= 0} { lappend synth_args -max_dsp $MAX_DSP }
synth_design {*}$synth_args
report_utilization -file [file join $repdir post_synth_utilization.rpt]

# --- implementation ---------------------------------------------------------
opt_design
place_design
phys_opt_design
route_design

report_utilization                 -file [file join $repdir utilization.rpt]
report_utilization -hierarchical   -file [file join $repdir utilization_hier.rpt]
report_timing_summary -max_paths 20 -report_unconstrained -file [file join $repdir timing_summary.rpt]
report_drc                         -file [file join $repdir drc.rpt]
report_clock_utilization           -file [file join $repdir clock_utilization.rpt]

# --- verdict ----------------------------------------------------------------
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "timing: WNS $wns ns, WHS $whs ns at ${clk_mhz} MHz"

write_bitstream -force [file join $outdir top.bit]

if {$wns < 0 || $whs < 0} {
    puts "TIMING NOT MET: the bitstream was written but must not be trusted."
    puts "Try READ_LATENCY=2 first, then CLK_DIV=2 (see fpga/README.md)."
    exit 1
}
puts "OK: fpga/build/top.bit, reports in fpga/reports/"
