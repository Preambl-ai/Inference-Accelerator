// bram: simple dual-port RAM, one clock, written so Vivado infers block RAM.
//
// interface
//   we/waddr/wdata : synchronous write. never gated, the host can load at any
//                    time.
//   re/raddr/rdata : registered read. rdata holds the word at raddr
//                    READ_LATENCY enabled edges after raddr is presented.
//                    re is the clock enable of the whole read path, so with
//                    re low rdata and the word in flight both hold. that is
//                    what lets a stalled datapath keep its operands aligned.
//
// READ_LATENCY = 1 is the plain BRAM read register. 2 adds the BRAM output
// register, which is the one to use when the read data path is timing critical.
//
// there is no reset. the array cannot be reset in a block RAM, and the read
// registers do not need one: rdata is undefined until the first read finishes.
// contents are whatever was last written; nothing is initialised.
//
// reading and writing the same address on the same edge is undefined. real
// 7-series BRAM does not promise old or new data in every mode, so simulation
// returns X for that read to stop anything relying on it. the write itself
// always lands.

module bram #(
    parameter int DATA_WIDTH   = 64,
    parameter int ADDR_WIDTH   = 10,
    parameter int READ_LATENCY = 1
)(
    input  logic clk,

    // write port
    input  logic we,
    input  logic [ADDR_WIDTH-1:0] waddr,
    input  logic [DATA_WIDTH-1:0] wdata,

    // read port
    input  logic re,
    input  logic [ADDR_WIDTH-1:0] raddr,
    output logic [DATA_WIDTH-1:0] rdata
);

    initial begin
        if (READ_LATENCY != 1 && READ_LATENCY != 2)
            $fatal(1, "bram: READ_LATENCY must be 1 or 2 (got %0d)", READ_LATENCY);
        if (DATA_WIDTH < 1 || ADDR_WIDTH < 1)
            $fatal(1, "bram: degenerate width parameter");
    end

    logic [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];
    logic [DATA_WIDTH-1:0] rd_q;

    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
    end

    always_ff @(posedge clk) begin
        if (re) begin
            rd_q <= mem[raddr];
`ifndef SYNTHESIS
            if (we && (waddr == raddr)) rd_q <= 'x;
`endif
        end
    end

    generate
        if (READ_LATENCY == 2) begin : g_out_reg
            logic [DATA_WIDTH-1:0] rd_q2;
            // same enable as the read register, or a stall would let the word
            // in flight overwrite the one being held
            always_ff @(posedge clk) begin
                if (re) rd_q2 <= rd_q;
            end
            assign rdata = rd_q2;
        end else begin : g_no_out_reg
            assign rdata = rd_q;
        end
    endgenerate

endmodule
