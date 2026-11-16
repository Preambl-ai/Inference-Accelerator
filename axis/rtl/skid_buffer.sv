// skid_buffer: two-entry AXI-Stream register slice.
//
// every output is a register, so m_ready only reaches the flops in this
// module and s_ready can be used upstream as a stall source without putting
// the downstream ready on a long combinational path.
//
// interface
//   s_valid/s_ready/s_data : upstream. a beat moves on an edge where
//                            s_valid & s_ready.
//   m_valid/m_ready/m_data : downstream, same handshake. m_valid and m_data
//                            hold until the beat is taken.
//
// latency: a beat accepted into an empty buffer is on m_* after 1 edge.
// throughput: one beat per cycle. s_ready only drops when both entries are
//   full, i.e. the cycle after a beat arrives while m_* is blocked.
// order: beats leave in the order they arrived, none dropped or repeated.
//
// reset is synchronous and drops both entries. s_ready is low during reset
// and rises one edge after it is released.
//
// there is no en input: this block is where a stall comes from, it is never
// the thing being stalled.

module skid_buffer #(
    parameter int DATA_WIDTH = 8
)(
    input  logic clk,
    input  logic reset,    // synchronous

    input  logic s_valid,
    output logic s_ready,
    input  logic [DATA_WIDTH-1:0] s_data,

    output logic m_valid,
    input  logic m_ready,
    output logic [DATA_WIDTH-1:0] m_data
);

    initial begin
        if (DATA_WIDTH < 1)
            $fatal(1, "skid_buffer: DATA_WIDTH must be >= 1 (got %0d)", DATA_WIDTH);
    end

    // second entry: catches the beat that arrives while m_* is blocked
    logic skid_valid;
    logic [DATA_WIDTH-1:0] skid_data;

    always_ff @(posedge clk) begin
        if (reset) begin
            s_ready    <= 1'b0;
            m_valid    <= 1'b0;
            m_data     <= '0;
            skid_valid <= 1'b0;
            skid_data  <= '0;
        end else begin
            if (!m_valid || m_ready) begin
                // output register is free or being emptied this edge
                if (skid_valid) begin
                    // oldest beat first. s_ready is low while the skid entry
                    // is full, so nothing can arrive on this edge.
                    m_valid    <= 1'b1;
                    m_data     <= skid_data;
                    skid_valid <= 1'b0;
                end else begin
                    m_valid <= s_valid & s_ready;
                    if (s_valid && s_ready) m_data <= s_data;
                end
                s_ready <= 1'b1;
            end else if (s_valid && s_ready) begin
                // output blocked: park the beat and close the door
                skid_valid <= 1'b1;
                skid_data  <= s_data;
                s_ready    <= 1'b0;
            end else begin
                s_ready <= ~skid_valid;
            end
        end
    end

endmodule
