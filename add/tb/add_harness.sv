// add_harness: test top for add_layer, with the real act_buffer and the same
// host access conv_layer has (host write in host_mode while idle, host read
// port muxed on busy).

module add_harness #(
    parameter int N            = 8,
    parameter int ADDR_WIDTH   = 9,
    parameter int READ_LATENCY = 1,
    parameter int NBUF         = 3,
    parameter int DIM_W        = 8,
    parameter int CW_W         = 7,
    parameter int M_BITS       = 8,
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
    input  logic [DIM_W-1:0] h_in,
    input  logic [DIM_W-1:0] w_in,
    input  logic [CW_W-1:0] c_words,
    input  logic [SEL_WIDTH-1:0] src_sel,
    input  logic [ADDR_WIDTH-1:0] src_base,
    input  logic [SEL_WIDTH-1:0] src2_sel,
    input  logic [ADDR_WIDTH-1:0] src2_base,
    input  logic [SEL_WIDTH-1:0] dst_sel,
    input  logic [ADDR_WIDTH-1:0] dst_base,
    input  logic [7:0] za,
    input  logic [7:0] zb,
    input  logic [7:0] zo,
    input  logic [M_BITS-1:0] ma,
    input  logic [M_BITS-1:0] mb,
    input  logic [3:0] shift,
    input  logic relu_en,
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

    logic e_rd_en, e_we;
    logic [ADDR_WIDTH-1:0] e_rd_addr, e_waddr;
    logic [SEL_WIDTH-1:0] e_rd_sel, e_wr_sel;
    logic [N*8-1:0] e_wdata, rdata;

    add_layer #(
        .N(N), .ADDR_WIDTH(ADDR_WIDTH), .READ_LATENCY(READ_LATENCY), .NBUF(NBUF),
        .DIM_W(DIM_W), .CW_W(CW_W), .M_BITS(M_BITS)
    ) u_add (
        .clk(clk), .reset(reset), .start(start), .ready(ready), .busy(busy), .done(done),
        .err(err), .err_code(err_code), .h_in(h_in), .w_in(w_in), .c_words(c_words),
        .src_sel(src_sel), .src_base(src_base), .src2_sel(src2_sel), .src2_base(src2_base),
        .dst_sel(dst_sel), .dst_base(dst_base), .za(za), .zb(zb), .zo(zo), .ma(ma), .mb(mb),
        .shift(shift), .relu_en(relu_en), .host_mode(host_mode),
        .rd_en(e_rd_en), .rd_addr(e_rd_addr), .rd_sel(e_rd_sel), .rd_data(rdata),
        .we(e_we), .waddr(e_waddr), .wdata(e_wdata), .wr_sel(e_wr_sel)
    );

    act_buffer #(
        .N(N), .DATA_WIDTH(8), .ADDR_WIDTH(ADDR_WIDTH), .READ_LATENCY(READ_LATENCY), .NBUF(NBUF)
    ) u_act (
        .clk(clk),
        .rd_en(busy ? e_rd_en : host_rd_en),
        .rd_addr(busy ? e_rd_addr : host_rd_addr),
        .rd_sel(busy ? e_rd_sel : host_rd_sel),
        .rd_data(rdata),
        .we(e_we), .waddr(e_waddr), .wdata(e_wdata), .wr_sel(e_wr_sel),
        .host_mode(host_mode), .host_we(host_we), .host_waddr(host_waddr),
        .host_wdata(host_wdata), .host_wr_sel(host_wr_sel)
    );

    assign host_rd_data = rdata;

endmodule
