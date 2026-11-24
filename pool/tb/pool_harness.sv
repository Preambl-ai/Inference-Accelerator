// pool_harness: test top for pool_layer. the real act_buffer, with the same
// host access conv_layer has (host_mode + host write port while idle, host
// read port muxed on busy), so the pool runs against the actual buffer and
// its read latency rather than a model of it.

module pool_harness #(
    parameter int N            = 8,
    parameter int ADDR_WIDTH   = 9,
    parameter int READ_LATENCY = 1,
    parameter int NBUF         = 2,
    parameter int DIM_W        = 8,
    parameter int CW_W         = 7,
    parameter int SEL_WIDTH    = (NBUF > 1) ? $clog2(NBUF) : 1
)(
    input  logic clk,
    input  logic reset,
    input  logic start,
    output logic ready,
    output logic busy,
    output logic done,
    output logic err,
    output logic [9:0] err_code,
    input  logic [1:0] mode,
    input  logic [DIM_W-1:0] h_in,
    input  logic [DIM_W-1:0] w_in,
    input  logic [CW_W-1:0] c_words,
    input  logic [SEL_WIDTH-1:0] src_sel,
    input  logic [SEL_WIDTH-1:0] dst_sel,
    input  logic [ADDR_WIDTH-1:0] src_base,
    input  logic [ADDR_WIDTH-1:0] dst_base,
    input  logic host_mode,
    input  logic host_we,
    input  logic [ADDR_WIDTH-1:0] host_waddr,
    input  logic [N*8-1:0] host_wdata,
    input  logic [SEL_WIDTH-1:0] host_wr_sel,
    input  logic host_rd_en,
    input  logic [ADDR_WIDTH-1:0] host_rd_addr,
    input  logic [SEL_WIDTH-1:0] host_rd_sel,
    output logic [N*8-1:0] host_rd_data
);

    logic p_rd_en, p_we;
    logic [ADDR_WIDTH-1:0] p_rd_addr, p_waddr;
    logic [SEL_WIDTH-1:0] p_rd_sel, p_wr_sel;
    logic [N*8-1:0] p_wdata, rdata;

    pool_layer #(
        .N(N), .ADDR_WIDTH(ADDR_WIDTH), .READ_LATENCY(READ_LATENCY), .NBUF(NBUF),
        .DIM_W(DIM_W), .CW_W(CW_W)
    ) u_pool (
        .clk(clk), .reset(reset), .start(start), .ready(ready), .busy(busy), .done(done),
        .err(err), .err_code(err_code), .mode(mode), .h_in(h_in), .w_in(w_in),
        .c_words(c_words), .src_sel(src_sel), .dst_sel(dst_sel), .src_base(src_base),
        .dst_base(dst_base), .host_mode(host_mode),
        .rd_en(p_rd_en), .rd_addr(p_rd_addr), .rd_sel(p_rd_sel), .rd_data(rdata),
        .we(p_we), .waddr(p_waddr), .wdata(p_wdata), .wr_sel(p_wr_sel)
    );

    act_buffer #(
        .N(N), .DATA_WIDTH(8), .ADDR_WIDTH(ADDR_WIDTH), .READ_LATENCY(READ_LATENCY), .NBUF(NBUF)
    ) u_act (
        .clk(clk),
        .rd_en(busy ? p_rd_en : host_rd_en),
        .rd_addr(busy ? p_rd_addr : host_rd_addr),
        .rd_sel(busy ? p_rd_sel : host_rd_sel),
        .rd_data(rdata),
        .we(p_we), .waddr(p_waddr), .wdata(p_wdata), .wr_sel(p_wr_sel),
        .host_mode(host_mode), .host_we(host_we), .host_waddr(host_waddr),
        .host_wdata(host_wdata), .host_wr_sel(host_wr_sel)
    );

    assign host_rd_data = rdata;

endmodule
