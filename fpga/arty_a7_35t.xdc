## Arty A7-35T constraints for top/rtl/top.sv.
##
## Every PACKAGE_PIN / IOSTANDARD / create_clock line below is copied, by a
## script, from Digilent's official master file
##   https://github.com/Digilent/digilent-xdc  Arty-A7-35-Master.xdc
##   (master at commit c8644a9fe31c), only uncommented, with btn[0] renamed
##   to btn_rst. Nothing here was typed from memory.
##
## led[0..3] are LD4..LD7: heartbeat, busy, done (sticky), err (sticky).
## btn_rst is BTN0 (active high). uart_txd_in / uart_rxd_out are named from
## the FT2232H's side, as on Digilent's schematic: txd_in is our RX.

set_property -dict { PACKAGE_PIN E3    IOSTANDARD LVCMOS33 } [get_ports { CLK100MHZ }]; #IO_L12P_T1_MRCC_35 Sch=gclk[100]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports { CLK100MHZ }];
set_property -dict { PACKAGE_PIN H5    IOSTANDARD LVCMOS33 } [get_ports { led[0] }]; #IO_L24N_T3_35 Sch=led[4]
set_property -dict { PACKAGE_PIN J5    IOSTANDARD LVCMOS33 } [get_ports { led[1] }]; #IO_25_35 Sch=led[5]
set_property -dict { PACKAGE_PIN T9    IOSTANDARD LVCMOS33 } [get_ports { led[2] }]; #IO_L24P_T3_A01_D17_14 Sch=led[6]
set_property -dict { PACKAGE_PIN T10   IOSTANDARD LVCMOS33 } [get_ports { led[3] }]; #IO_L24N_T3_A00_D16_14 Sch=led[7]
set_property -dict { PACKAGE_PIN D9    IOSTANDARD LVCMOS33 } [get_ports { btn_rst }]; #IO_L6N_T0_VREF_16 Sch=btn[0]
set_property -dict { PACKAGE_PIN D10   IOSTANDARD LVCMOS33 } [get_ports { uart_rxd_out }]; #IO_L19N_T3_VREF_16 Sch=uart_rxd_out
set_property -dict { PACKAGE_PIN A9    IOSTANDARD LVCMOS33 } [get_ports { uart_txd_in }]; #IO_L14N_T2_SRCC_16 Sch=uart_txd_in

## Configuration bank voltage, as for every Arty A7 design
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

## The button and the UART RX line are asynchronous to the clock and each
## goes through a 2-flop synchronizer in the RTL (top.sv, uart_rx.sv); the
## LEDs and UART TX are slow outputs with no timing relationship to anything
## off chip. None of them should be timed against CLK100MHZ.
set_false_path -from [get_ports btn_rst]
set_false_path -from [get_ports uart_txd_in]
set_false_path -to   [get_ports {led[*]}]
set_false_path -to   [get_ports uart_rxd_out]
