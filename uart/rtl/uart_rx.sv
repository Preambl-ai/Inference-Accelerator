// uart_rx: 8N1 receiver, LSB first, idle high.
//
// rx is asynchronous to clk; it goes through a 2-flop synchronizer first, so
// everything after it sees a clean, 2-cycle-late copy. reset value 1 = idle,
// so a reset never looks like a start bit.
//
// sampling: on the first low sample the receiver waits half a bit and looks
// again. still low = a real start bit (a glitch shorter than half a bit is
// ignored); from there every bit is sampled one full bit period later, so
// each sample lands in the middle of its bit. the error at the stop bit is
// 9.5 bit periods' worth of rate mismatch plus one clock of quantization,
// which keeps a +-2 % rate error well inside the half bit there is to spare.
//
// the stop bit is sampled in its middle and the receiver goes straight back
// to looking for the next start bit, so back-to-back bytes with no idle time
// are received, and a transmitter that is slightly fast still gets its next
// start edge seen.
//
// framing error: a stop bit sampled low. the byte is dropped (valid does not
// pulse), frame_err pulses, and the receiver waits for the line to go high
// before it looks for a start bit again. without that wait a line held low
// (a break, or a cable pulled) would be read as an endless run of 0x00 bytes.
//
// outputs: data is valid on the cycle valid pulses and holds until the next
// byte. one byte per valid pulse.

module uart_rx #(
    parameter int CLKS_PER_BIT = 100     // clk / baud, >= 4
)(
    input  logic       clk,
    input  logic       reset,            // synchronous
    input  logic       rx,               // asynchronous serial input
    output logic [7:0] data,
    output logic       valid,
    output logic       frame_err
);

    localparam int CNT_W = $clog2(CLKS_PER_BIT + 1);
    localparam int HALF  = (CLKS_PER_BIT - 1) / 2;

    initial begin
        if (CLKS_PER_BIT < 4)
            $fatal(1, "uart_rx: CLKS_PER_BIT must be >= 4 (got %0d)", CLKS_PER_BIT);
    end

    // input synchronizer, the only CDC in this block
    logic rx_s1, rx_s2;
    always_ff @(posedge clk) begin
        if (reset) begin
            rx_s1 <= 1'b1;
            rx_s2 <= 1'b1;
        end else begin
            rx_s1 <= rx;
            rx_s2 <= rx_s1;
        end
    end

    typedef enum logic [2:0] {S_IDLE, S_START, S_DATA, S_STOP, S_WAIT_HIGH} state_t;
    state_t state;
    logic [CNT_W-1:0] cnt;
    logic [2:0] bit_idx;
    logic [7:0] shift;

    always_ff @(posedge clk) begin
        if (reset) begin
            state     <= S_IDLE;
            cnt       <= '0;
            bit_idx   <= '0;
            shift     <= '0;
            data      <= '0;
            valid     <= 1'b0;
            frame_err <= 1'b0;
        end else begin
            valid     <= 1'b0;
            frame_err <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (!rx_s2) begin
                        cnt   <= '0;
                        state <= S_START;
                    end
                end
                // middle of the start bit: confirm it
                S_START: begin
                    if (cnt == CNT_W'(HALF)) begin
                        if (!rx_s2) begin
                            cnt     <= '0;
                            bit_idx <= '0;
                            state   <= S_DATA;
                        end else begin
                            state <= S_IDLE;     // glitch
                        end
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end
                // one full bit after the previous sample = middle of this bit
                S_DATA: begin
                    if (cnt == CNT_W'(CLKS_PER_BIT - 1)) begin
                        cnt   <= '0;
                        shift <= {rx_s2, shift[7:1]};      // LSB first
                        if (bit_idx == 3'd7) state <= S_STOP;
                        else                 bit_idx <= bit_idx + 1'b1;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end
                S_STOP: begin
                    if (cnt == CNT_W'(CLKS_PER_BIT - 1)) begin
                        cnt <= '0;
                        if (rx_s2) begin
                            data  <= shift;
                            valid <= 1'b1;
                            state <= S_IDLE;
                        end else begin
                            frame_err <= 1'b1;
                            state     <= S_WAIT_HIGH;
                        end
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end
                S_WAIT_HIGH: begin
                    if (rx_s2) state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
