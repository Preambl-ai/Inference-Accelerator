// act_buffer: NBUF activation memories ("banks") on bram.sv, one shared read
// port, one datapath write port and a host write port. the full contract is
// memory/act_buffer_contract.md; the parts a user of the block needs are here.
//
// word: N lanes of DATA_WIDTH bits, lane i at [i*DATA_WIDTH +: DATA_WIDTH], so
//   lane 0 is in the LSBs (im2col_reader reads byte j as channel g*N + j). the
//   buffer only stores bits; lanes are two's complement by convention.
//
// interface
//   rd_en/rd_addr/rd_sel : a read advances on an edge with rd_en & !host_mode.
//                          rd_data is the word at {rd_sel, rd_addr} exactly
//                          READ_LATENCY advancing edges later (measured, 1 and
//                          2). rd_sel may change every cycle. without an
//                          advance, rd_data and the word in flight both hold.
//   we/waddr/wdata/wr_sel: datapath write, used while host_mode = 0.
//   host_*               : host write, used while host_mode = 1.
//   host_mode            : 1 gives the host the write port and freezes reads.
//                          a read asked for during host_mode is dropped, not
//                          queued. raise it only between layers: the reader's
//                          own tag pipeline does not freeze with this one.
//
// rd_sel/wr_sel/host_wr_sel >= NBUF on the side in use is illegal (simulation
// $fatal, +act_buffer_allow_bad_sel switches it off). the hardware then drops
// such a write and returns undefined data for such a read; neighbouring reads
// stay aligned.
//
// there is no reset. rd_data is undefined until READ_LATENCY advancing edges
// after the first advancing read. a read that advances on the edge where the
// same bank writes the same address is undefined (inherited from bram); the
// write lands. the same address in another bank is fine.

module act_buffer #(
    parameter int N = 8,
    parameter int DATA_WIDTH = 8,
    parameter int ADDR_WIDTH = 8,
    parameter int READ_LATENCY = 1,
    parameter int NBUF = 3,
    // derived, do not override. at least 1: $clog2(1) = 0 made the select
    // ports [-1:0], which is 2 bits wide, not 0, and gave NBUF = 1 three
    // selects that index no bank. same idiom as im2col_reader's ROW_W.
    parameter int SEL_WIDTH = (NBUF > 1) ? $clog2(NBUF) : 1
)(
    input  logic                        clk,

    // Read port
    input  logic                        rd_en,
    input  logic [ADDR_WIDTH-1:0]       rd_addr,
    input  logic [SEL_WIDTH-1:0]        rd_sel,
    output logic [N*DATA_WIDTH-1:0]     rd_data,

    // Write port
    input  logic                        we,
    input  logic [ADDR_WIDTH-1:0]       waddr,
    input  logic [N*DATA_WIDTH-1:0]     wdata,
    input  logic [SEL_WIDTH-1:0]        wr_sel,

    // Host write port
    input  logic                        host_mode,
    input  logic                        host_we,
    input  logic [ADDR_WIDTH-1:0]       host_waddr,
    input  logic [N*DATA_WIDTH-1:0]     host_wdata,
    input  logic [SEL_WIDTH-1:0]        host_wr_sel
);

    localparam int WORD_WIDTH = N * DATA_WIDTH;
    localparam int SEL_W_DERIVED = (NBUF > 1) ? $clog2(NBUF) : 1;

    initial begin
        if (N < 1 || DATA_WIDTH < 1)
            $fatal(1, "act_buffer: N and DATA_WIDTH must be >= 1 (got %0d, %0d)", N, DATA_WIDTH);
        if (NBUF < 1)
            $fatal(1, "act_buffer: NBUF must be >= 1 (got %0d)", NBUF);
        // too narrow cuts off select bits; too wide adds selects that match
        // no bank. either way the ports stop meaning what the contract says.
        if (SEL_WIDTH != SEL_W_DERIVED)
            $fatal(1, "act_buffer: SEL_WIDTH is derived, do not override it (got %0d, want %0d)",
                   SEL_WIDTH, SEL_W_DERIVED);
        // READ_LATENCY and ADDR_WIDTH are checked by bram
    end

    logic [WORD_WIDTH-1:0] bank_rdata [0:NBUF-1];

    logic [SEL_WIDTH-1:0] sel_pipe [0:READ_LATENCY-1];

    logic                 write_en;
    logic [ADDR_WIDTH-1:0] write_addr;
    logic [WORD_WIDTH-1:0] write_data;
    logic [SEL_WIDTH-1:0]  write_sel;
    logic                 read_advance;

    // Host has exclusive write access between layers
    assign write_en   = host_mode ? host_we      : we;
    assign write_addr = host_mode ? host_waddr   : waddr;
    assign write_data = host_mode ? host_wdata   : wdata;
    assign write_sel  = host_mode ? host_wr_sel  : wr_sel;

    assign read_advance = rd_en && !host_mode;

    // One BRAM per buffer. a select >= NBUF matches no bank's compare, so
    // that write is dropped rather than landing somewhere.
    genvar i;
    generate
        for (i = 0; i < NBUF; i = i + 1) begin : GEN_BRAM

            bram #(
                .DATA_WIDTH(WORD_WIDTH),
                .ADDR_WIDTH(ADDR_WIDTH),
                .READ_LATENCY(READ_LATENCY)
            ) u_bram (
                .clk(clk),
                .we(write_en && (write_sel == i)),
                .waddr(write_addr),
                .wdata(write_data),
                .re(read_advance),
                .raddr(rd_addr),
                .rdata(bank_rdata[i])
            );

        end
    endgenerate

    // Track which buffer each read belongs to. same enable as the bram read
    // registers, so a select meets its data at the mux after any stall.
    integer j;
    always_ff @(posedge clk) begin
        if (read_advance) begin
            sel_pipe[0] <= rd_sel;

            for (j = 1; j < READ_LATENCY; j = j + 1)
                sel_pipe[j] <= sel_pipe[j-1];
        end
    end

    // Select the correct BRAM output. no range guard: a select >= NBUF is
    // illegal, and a guard would sit on the read data path that
    // READ_LATENCY = 2 exists to keep short.
    assign rd_data = bank_rdata[sel_pipe[READ_LATENCY-1]];

    // Simulation-only checks. only the side of the host_mode mux in use is
    // checked, so a junk select on an idle port stays legal.
`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (!$test$plusargs("act_buffer_allow_bad_sel")) begin
            if (read_advance && (rd_sel >= NBUF))
                $fatal(1, "act_buffer: rd_sel=%0d is not a bank (NBUF=%0d)", rd_sel, NBUF);
            if (write_en && (write_sel >= NBUF))
                $fatal(1, "act_buffer: %s write sel=%0d is not a bank (NBUF=%0d)",
                       host_mode ? "host" : "datapath", write_sel, NBUF);
        end
    end
`endif

endmodule
