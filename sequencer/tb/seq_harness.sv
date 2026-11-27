// seq_harness: test top for the sequencer. the sequencer drives the real
// layer_engine (conv, pool, add on one act_buffer); the TB loads memories
// through the engine's host ports, writes descriptors, starts programs and
// reads the cycle RAM. with HAS_DMA = 1 a wdma sits beside it, its AXI4
// master on the harness ports (the TB's AXI4 slave model plays DDR), and the
// B/P write ports belong to wdma while it is busy and to the TB otherwise.

module seq_harness #(
    parameter int N            = 4,
    parameter int ADDR_WIDTH   = 10,
    parameter int READ_LATENCY = 1,
    parameter int NBUF         = 3,
    parameter int B_ADDR_W     = 13,
    parameter int P_ADDR_W     = 8,
    parameter int K_MAX        = 1024,
    parameter int K_WIDTH      = 11,
    parameter int M_BITS       = 8,
    parameter int DESC_DEPTH   = 32,
    parameter int HAS_DMA      = 0,
    parameter int SEL_WIDTH    = (NBUF > 1) ? $clog2(NBUF) : 1,
    parameter int IDX_W        = $clog2(DESC_DEPTH)
)(
    input  logic clk,
    input  logic reset,
    input  logic abort,                 // soft reset as top gives it to wdma
    // sequencer
    input  logic start,
    input  logic [IDX_W-1:0] start_index,
    output logic busy,
    output logic done,
    output logic err,
    output logic [IDX_W-1:0] cur_index,
    output logic [IDX_W:0] layers_done,
    output logic [IDX_W-1:0] err_index,
    output logic [15:0] err_code,
    input  logic desc_we,
    input  logic [IDX_W+3:0] desc_waddr,
    input  logic [31:0] desc_wdata,
    input  logic cyc_rd_en,
    input  logic [IDX_W:0] cyc_rd_addr,
    output logic [31:0] cyc_rd_data,
    // engine status (for the TB)
    output logic eng_busy,
    output logic eng_start_o,
    output logic dma_busy,
    // memories
    input  logic b_we,
    input  logic [B_ADDR_W-1:0] b_waddr,
    input  logic [N*8-1:0] b_wdata,
    input  logic p_we,
    input  logic [P_ADDR_W-1:0] p_waddr,
    input  logic [52:0] p_wdata,
    input  logic host_mode,
    input  logic host_we,
    input  logic [ADDR_WIDTH-1:0] host_waddr,
    input  logic [N*8-1:0] host_wdata,
    input  logic [SEL_WIDTH-1:0] host_wr_sel,
    input  logic host_rd_en,
    input  logic [ADDR_WIDTH-1:0] host_rd_addr,
    input  logic [SEL_WIDTH-1:0] host_rd_sel,
    output logic [N*8-1:0] host_rd_data,
    // AXI4 master of wdma (HAS_DMA = 1)
    output logic [31:0] araddr,
    output logic [7:0]  arlen,
    output logic [2:0]  arsize,
    output logic [1:0]  arburst,
    output logic [3:0]  arid,
    output logic        arvalid,
    input  logic        arready,
    input  logic [63:0] rdata,
    input  logic [1:0]  rresp,
    input  logic        rlast,
    input  logic [3:0]  rid,
    input  logic        rvalid,
    output logic        rready,
    output logic [31:0] awaddr,
    output logic [7:0]  awlen,
    output logic [2:0]  awsize,
    output logic [1:0]  awburst,
    output logic [3:0]  awid,
    output logic        awvalid,
    input  logic        awready,
    output logic [63:0] wdata,
    output logic [7:0]  wstrb,
    output logic        wlast,
    output logic        wvalid,
    input  logic        wready,
    input  logic [1:0]  bresp,
    input  logic [3:0]  bid,
    input  logic        bvalid,
    output logic        bready
);

    logic eng_start, eng_ready, eng_done, eng_err;
    logic [2:0] eng_op;
    logic [9:0] eng_code;
    logic [7:0] h_in, w_in, h_out, w_out;
    logic [6:0] cin_words, cout_words;
    logic [2:0] kh, kw;
    logic [1:0] stride, pad;
    logic signed [7:0] zp_in, zp_out, zp_in2;
    logic relu_en;
    logic [SEL_WIDTH-1:0] src_sel, dst_sel, src2_sel;
    logic [ADDR_WIDTH-1:0] src_base, dst_base, src2_base;
    logic [B_ADDR_W-1:0] b_base;
    logic [P_ADDR_W-1:0] p_base;
    logic [M_BITS-1:0] add_ma, add_mb;
    logic [3:0] add_shift;

    logic dma_start, dma_ready, dma_done, dma_err;
    logic [3:0] dma_code;
    logic [31:0] dma_b_src, dma_p_src;
    logic [K_WIDTH-1:0] dma_k;

    sequencer #(
        .ADDR_WIDTH(ADDR_WIDTH), .NBUF(NBUF), .B_ADDR_W(B_ADDR_W), .P_ADDR_W(P_ADDR_W),
        .M_BITS(M_BITS), .DESC_DEPTH(DESC_DEPTH), .READ_LATENCY(READ_LATENCY),
        .HAS_DMA(HAS_DMA), .K_WIDTH(K_WIDTH), .AXI_AW(32)
    ) u_seq (
        .clk(clk), .reset(reset || abort), .start(start), .start_index(start_index), .busy(busy),
        .done(done), .err(err), .cur_index(cur_index), .layers_done(layers_done),
        .err_index(err_index), .err_code(err_code),
        .desc_we(desc_we), .desc_waddr(desc_waddr), .desc_wdata(desc_wdata),
        .cyc_rd_en(cyc_rd_en), .cyc_rd_addr(cyc_rd_addr), .cyc_rd_data(cyc_rd_data),
        .dma_start(dma_start), .dma_b_src(dma_b_src), .dma_p_src(dma_p_src), .dma_k(dma_k),
        .dma_ready(dma_ready), .dma_done(dma_done), .dma_err(dma_err), .dma_err_code(dma_code),
        .eng_start(eng_start), .eng_op(eng_op), .h_in(h_in), .w_in(w_in), .h_out(h_out),
        .w_out(w_out), .cin_words(cin_words), .cout_words(cout_words), .kh(kh), .kw(kw),
        .stride(stride), .pad(pad), .zp_in(zp_in), .zp_out(zp_out), .zp_in2(zp_in2),
        .relu_en(relu_en), .src_sel(src_sel), .dst_sel(dst_sel), .src2_sel(src2_sel),
        .src_base(src_base), .dst_base(dst_base), .src2_base(src2_base), .b_base(b_base),
        .p_base(p_base), .add_ma(add_ma), .add_mb(add_mb), .add_shift(add_shift),
        .eng_ready(eng_ready), .eng_done(eng_done), .eng_err(eng_err), .eng_err_code(eng_code)
    );

    assign eng_start_o = eng_start;

    // the on-chip B/P write ports: wdma while it fetches, the TB otherwise
    logic d_b_we, d_p_we;
    logic [B_ADDR_W-1:0] d_b_waddr;
    logic [N*8-1:0] d_b_wdata;
    logic [P_ADDR_W-1:0] d_p_waddr;
    logic [52:0] d_p_wdata;

    generate
        if (HAS_DMA != 0) begin : g_dma
            wdma #(
                .N(N), .B_ADDR_W(B_ADDR_W), .P_ADDR_W(P_ADDR_W), .K_WIDTH(K_WIDTH), .CW_W(7),
                .AXI_AW(32), .AXI_IDW(4)
            ) u_dma (
                .clk(clk), .reset(reset), .abort(abort),
                .f_start(dma_start), .f_b_src(dma_b_src), .f_p_src(dma_p_src), .f_k(dma_k),
                .f_groups(cout_words), .f_b_dst(b_base), .f_p_dst(p_base),
                .f_ready(dma_ready), .busy(dma_busy), .f_done(dma_done), .f_err(dma_err),
                .f_err_code(dma_code),
                .h_req(1'b0), .h_we(1'b0), .h_addr('0), .h_wdata('0), .h_ack(), .h_resp(),
                .h_rdata(),
                .b_we(d_b_we), .b_waddr(d_b_waddr), .b_wdata(d_b_wdata),
                .p_we(d_p_we), .p_waddr(d_p_waddr), .p_wdata(d_p_wdata),
                .araddr(araddr), .arlen(arlen), .arsize(arsize), .arburst(arburst), .arid(arid),
                .arvalid(arvalid), .arready(arready), .rdata(rdata), .rresp(rresp),
                .rlast(rlast), .rid(rid), .rvalid(rvalid), .rready(rready),
                .awaddr(awaddr), .awlen(awlen), .awsize(awsize), .awburst(awburst), .awid(awid),
                .awvalid(awvalid), .awready(awready), .wdata(wdata), .wstrb(wstrb),
                .wlast(wlast), .wvalid(wvalid), .wready(wready), .bresp(bresp), .bid(bid),
                .bvalid(bvalid), .bready(bready)
            );
        end else begin : g_no_dma
            assign {dma_ready, dma_busy, dma_done, dma_err, dma_code} = '0;
            assign {d_b_we, d_p_we, d_b_waddr, d_b_wdata, d_p_waddr, d_p_wdata} = '0;
            assign {araddr, arlen, arsize, arburst, arid, arvalid, rready} = '0;
            assign {awaddr, awlen, awsize, awburst, awid, awvalid} = '0;
            assign {wdata, wstrb, wlast, wvalid, bready} = '0;
        end
    endgenerate

    layer_engine #(
        .N(N), .ADDR_WIDTH(ADDR_WIDTH), .READ_LATENCY(READ_LATENCY), .NBUF(NBUF),
        .K_MAX(K_MAX), .B_ADDR_W(B_ADDR_W), .P_ADDR_W(P_ADDR_W), .M_BITS(M_BITS)
    ) u_engine (
        .clk(clk), .reset(reset || abort),
        .start(eng_start), .op(eng_op), .ready(eng_ready), .busy(eng_busy), .done(eng_done),
        .err(eng_err), .err_code(eng_code), .done_op(),
        .h_in(h_in), .w_in(w_in), .h_out(h_out), .w_out(w_out),
        .cin_words(cin_words), .cout_words(cout_words), .kh(kh), .kw(kw),
        .stride(stride), .pad(pad), .zp_in(zp_in), .zp_out(zp_out), .relu_en(relu_en),
        .src_sel(src_sel), .dst_sel(dst_sel), .src_base(src_base), .dst_base(dst_base),
        .b_base(b_base), .p_base(p_base),
        .src2_sel(src2_sel), .src2_base(src2_base), .zp_in2(zp_in2), .add_ma(add_ma),
        .add_mb(add_mb), .add_shift(add_shift),
        .conv_busy(), .pool_busy(), .add_busy(), .tile_launch(),
        .b_we(dma_busy ? d_b_we : b_we), .b_waddr(dma_busy ? d_b_waddr : b_waddr),
        .b_wdata(dma_busy ? d_b_wdata : b_wdata),
        .p_we(dma_busy ? d_p_we : p_we), .p_waddr(dma_busy ? d_p_waddr : p_waddr),
        .p_wdata(dma_busy ? d_p_wdata : p_wdata),
        .host_mode(host_mode), .host_we(host_we), .host_waddr(host_waddr),
        .host_wdata(host_wdata), .host_wr_sel(host_wr_sel),
        .host_rd_en(host_rd_en), .host_rd_addr(host_rd_addr), .host_rd_sel(host_rd_sel),
        .host_rd_data(host_rd_data)
    );

endmodule
