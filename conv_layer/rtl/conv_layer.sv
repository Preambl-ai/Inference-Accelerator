// conv_layer: one complete INT8 convolution layer out of on-chip memory.
//
//   act_buffer bank src_sel -> im2col_reader -> gemm_tile -> write-back -> bank dst_sel
//
// the full contract is conv_layer/conv_layer_contract.md. the layer itself is
// conv_core; this wrapper adds the act_buffer and the host's access to it,
// exactly as the block did before the split, so its ports, behaviour and
// timing are unchanged (the conv_layer TB runs on this module untouched).
// layer_engine uses conv_core directly and shares its act_buffer with the
// other engines.
//
// ownership: while idle the host writes the act_buffer through host_mode and
//   the host_* write port, reads it through host_rd_* (with host_mode low),
//   and writes B and P. while the layer runs none of that is allowed
//   (simulation $fatal in conv_core); host_rd_* are ignored.
//
// reset is synchronous. no act_buffer write lands on or after the reset edge,
// every destination word is left either old or final, ready is high after it.

module conv_layer #(
    parameter int N            = 8,
    parameter int DATA_WIDTH   = 8,
    parameter int ADDR_WIDTH   = 12,
    parameter int READ_LATENCY = 1,
    parameter int NBUF         = 2,
    parameter int K_MAX        = 1024,
    parameter int K_WIDTH      = 11,
    parameter int B_ADDR_W     = 13,
    parameter int P_ADDR_W     = 6,
    parameter int DIM_W        = 8,
    parameter int CW_W         = 7,
    // derived, do not override
    parameter int SEL_WIDTH    = (NBUF > 1) ? $clog2(NBUF) : 1,
    parameter int ROW_W        = (N > 1) ? $clog2(N) : 1,
    parameter int P_WIDTH      = 5 + 16 + 32     // gemm_tile's SHIFT + M + BIAS
)(
    input  logic clk,
    input  logic reset,    // synchronous

    // layer command
    input  logic start,
    output logic ready,
    output logic busy,
    output logic done,
    output logic err,
    output logic [8:0] err_code,
    input  logic [DIM_W-1:0] h_in,
    input  logic [DIM_W-1:0] w_in,
    input  logic [DIM_W-1:0] h_out,
    input  logic [DIM_W-1:0] w_out,
    input  logic [CW_W-1:0] cin_words,
    input  logic [CW_W-1:0] cout_words,
    input  logic [2:0] kh,
    input  logic [2:0] kw,
    input  logic [1:0] stride,
    input  logic [1:0] pad,
    input  logic signed [DATA_WIDTH-1:0] zp_in,
    input  logic signed [DATA_WIDTH-1:0] zp_out,
    input  logic relu_en,
    input  logic [SEL_WIDTH-1:0] src_sel,
    input  logic [SEL_WIDTH-1:0] dst_sel,
    input  logic [ADDR_WIDTH-1:0] src_base,
    input  logic [ADDR_WIDTH-1:0] dst_base,
    input  logic [B_ADDR_W-1:0] b_base,
    input  logic [P_ADDR_W-1:0] p_base,

    // B and P memory write ports (gemm_tile's), idle only
    input  logic b_we,
    input  logic [B_ADDR_W-1:0] b_waddr,
    input  logic [N*DATA_WIDTH-1:0] b_wdata,
    input  logic p_we,
    input  logic [P_ADDR_W-1:0] p_waddr,
    input  logic [P_WIDTH-1:0] p_wdata,

    // act_buffer host write port, idle only
    input  logic host_mode,
    input  logic host_we,
    input  logic [ADDR_WIDTH-1:0] host_waddr,
    input  logic [N*DATA_WIDTH-1:0] host_wdata,
    input  logic [SEL_WIDTH-1:0] host_wr_sel,

    // act_buffer host read port, idle and host_mode low only. data comes
    // READ_LATENCY host_rd_en edges later, as act_buffer defines it.
    input  logic host_rd_en,
    input  logic [ADDR_WIDTH-1:0] host_rd_addr,
    input  logic [SEL_WIDTH-1:0] host_rd_sel,
    output logic [N*DATA_WIDTH-1:0] host_rd_data
);

    localparam int WORD_W = N * DATA_WIDTH;

    logic core_rd_en, core_we;
    logic [ADDR_WIDTH-1:0] core_rd_addr, core_waddr;
    logic [SEL_WIDTH-1:0] core_rd_sel, core_wr_sel;
    logic [WORD_W-1:0] core_wdata, act_rdata;
    // the tile launch pulse keeps its old name here: the conv_layer TB
    // counts tiles on dut.tile_start
    logic tile_start;

    conv_core #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .READ_LATENCY(READ_LATENCY),
        .NBUF(NBUF), .K_MAX(K_MAX), .K_WIDTH(K_WIDTH), .B_ADDR_W(B_ADDR_W),
        .P_ADDR_W(P_ADDR_W), .DIM_W(DIM_W), .CW_W(CW_W)
    ) u_core (
        .clk(clk), .reset(reset),
        .start(start), .ready(ready), .busy(busy), .done(done), .err(err), .err_code(err_code),
        .h_in(h_in), .w_in(w_in), .h_out(h_out), .w_out(w_out),
        .cin_words(cin_words), .cout_words(cout_words), .kh(kh), .kw(kw),
        .stride(stride), .pad(pad), .zp_in(zp_in), .zp_out(zp_out), .relu_en(relu_en),
        .src_sel(src_sel), .dst_sel(dst_sel), .src_base(src_base), .dst_base(dst_base),
        .b_base(b_base), .p_base(p_base),
        .b_we(b_we), .b_waddr(b_waddr), .b_wdata(b_wdata),
        .p_we(p_we), .p_waddr(p_waddr), .p_wdata(p_wdata),
        .host_mode(host_mode),
        .act_rd_en(core_rd_en), .act_rd_addr(core_rd_addr), .act_rd_sel(core_rd_sel),
        .act_rd_data(act_rdata),
        .act_we(core_we), .act_waddr(core_waddr), .act_wdata(core_wdata), .act_wr_sel(core_wr_sel),
        .tile_launch(tile_start)
    );

    // the read port belongs to the reader while busy and to the host
    // otherwise. muxing on busy, not on the reader's rd_en, keeps a host
    // read from slipping in during the check phase or between tiles.
    logic act_rd_en;
    logic [ADDR_WIDTH-1:0] act_rd_addr;
    logic [SEL_WIDTH-1:0] act_rd_sel;
    assign act_rd_en    = busy ? core_rd_en   : host_rd_en;
    assign act_rd_addr  = busy ? core_rd_addr : host_rd_addr;
    assign act_rd_sel   = busy ? core_rd_sel  : host_rd_sel;
    assign host_rd_data = act_rdata;

    act_buffer #(
        .N            (N),
        .DATA_WIDTH   (DATA_WIDTH),
        .ADDR_WIDTH   (ADDR_WIDTH),
        .READ_LATENCY (READ_LATENCY),
        .NBUF         (NBUF)
    ) u_act (
        .clk         (clk),
        .rd_en       (act_rd_en),
        .rd_addr     (act_rd_addr),
        .rd_sel      (act_rd_sel),
        .rd_data     (act_rdata),
        .we          (core_we),
        .waddr       (core_waddr),
        .wdata       (core_wdata),
        .wr_sel      (core_wr_sel),
        .host_mode   (host_mode),
        .host_we     (host_we),
        .host_waddr  (host_waddr),
        .host_wdata  (host_wdata),
        .host_wr_sel (host_wr_sel)
    );

endmodule
