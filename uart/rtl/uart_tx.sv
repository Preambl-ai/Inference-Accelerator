// uart_tx: 8N1 transmitter, LSB first, idle high.
//
// valid/ready handshake: a byte is taken on an edge with valid & ready. ready
// is high whenever the transmitter is idle, including the cycle the stop bit
// ends, so a stream of bytes goes out back to back with no idle bit between
// them (start bit directly after the previous stop bit).
//
// tx comes straight from a flop, so it never glitches between bits.

module uart_tx #(
    parameter int CLKS_PER_BIT = 100
)(
    input  logic       clk,
    input  logic       reset,       // synchronous
    input  logic [7:0] data,
    input  logic       valid,
    output logic       ready,
    output logic       tx
);

    localparam int CNT_W = $clog2(CLKS_PER_BIT + 1);

    initial begin
        if (CLKS_PER_BIT < 4)
            $fatal(1, "uart_tx: CLKS_PER_BIT must be >= 4 (got %0d)", CLKS_PER_BIT);
    end

    // {stop, data[7:0], start}, shifted out from bit 0
    logic [9:0] frame;
    logic [3:0] bits_left;
    logic [CNT_W-1:0] cnt;

    // the last bit's final cycle counts as idle, so the next byte's start bit
    // begins exactly one bit period after the stop bit did
    assign ready = (bits_left == 0) ||
                   (bits_left == 4'd1 && cnt == CNT_W'(CLKS_PER_BIT - 1));

    always_ff @(posedge clk) begin
        if (reset) begin
            frame     <= '1;
            bits_left <= '0;
            cnt       <= '0;
            tx        <= 1'b1;
        end else if (valid && ready) begin
            frame     <= {1'b1, data, 1'b0};
            tx        <= 1'b0;                 // start bit
            bits_left <= 4'd10;
            cnt       <= '0;
        end else if (bits_left != 0) begin
            if (cnt == CNT_W'(CLKS_PER_BIT - 1)) begin
                cnt       <= '0;
                bits_left <= bits_left - 1'b1;
                frame     <= {1'b1, frame[9:1]};
                tx        <= (bits_left == 4'd1) ? 1'b1 : frame[1];
            end else begin
                cnt <= cnt + 1'b1;
            end
        end
    end

endmodule
