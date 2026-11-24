// uart_loopback: test harness for uart_tx and uart_rx. with loop = 1 the
// transmitter's line feeds the receiver; with loop = 0 the testbench drives
// the receiver's line directly (rate errors, bad stop bits, glitches).
// tx_line is always visible so the TB can check the transmitter's bit timing.

module uart_loopback #(
    parameter int CLKS_PER_BIT = 16
)(
    input  logic       clk,
    input  logic       reset,
    input  logic       loop,
    input  logic       rx_line,
    input  logic [7:0] tx_data,
    input  logic       tx_valid,
    output logic       tx_ready,
    output logic       tx_line,
    output logic [7:0] rx_data,
    output logic       rx_valid,
    output logic       rx_frame_err
);

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .reset(reset), .data(tx_data), .valid(tx_valid),
        .ready(tx_ready), .tx(tx_line)
    );

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .reset(reset), .rx(loop ? tx_line : rx_line),
        .data(rx_data), .valid(rx_valid), .frame_err(rx_frame_err)
    );

endmodule
