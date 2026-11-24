// top: Arty A7-35T board top. PC <-> USB-UART <-> uart_rx/uart_tx <->
// uart_bridge (AXI4-Lite master) <-> csr <-> layer_engine (conv, pool, add on
// one act_buffer). the engine takes commands from either the CSR's legacy
// single-layer command (a conv) or the descriptor sequencer, which runs a
// whole program of layers loaded through the CSR's descriptor window.
// contract: top/host_interface_contract.md.
//
// one clock domain. CLK_DIV = 1 runs everything on the 100 MHz board clock;
// CLK_DIV = 2 puts an MMCME2_BASE in front for 50 MHz (synthesis only, for
// when timing does not close). CLKS_PER_BIT follows the clock, so at 1 Mbaud
// it is 100 or 50; simulations override it with a small value.
//
// reset: BTN0 is asynchronous, so it goes through two flops; a power-on
// counter and (with the MMCM) "not locked" join it, and the result is
// stretched to RST_HOLD cycles of synchronous reset for the whole design.
// the only other asynchronous input is the UART line, synchronized inside
// uart_rx. there are no other clock domain crossings.
//
// layer_engine and the sequencer additionally get the CSR's soft reset. the
// UART, bridge and CSR never see it: they are in the middle of answering the
// write that asked for it.
//
// LEDs: LD4 heartbeat, LD5 busy, LD6 done (sticky), LD7 err (sticky).

module top #(
    parameter int N            = 8,
    parameter int READ_LATENCY = 1,
    parameter int CLK_DIV      = 1,                    // 1 = 100 MHz, 2 = 50 MHz (MMCM)
    parameter int BAUD         = 1_000_000,
    parameter int CLKS_PER_BIT = 100_000_000 / CLK_DIV / BAUD,
    parameter int TIMEOUT_BITS = 1000,
    parameter logic [31:0] BUILD_ID = 32'h0,
    parameter int ADDR_WIDTH   = 12,
    // three banks: a ResNet block needs its input (the skip), the branch and
    // the sum in three different banks (add_layer reads two, writes a third)
    parameter int NBUF         = 3,
    parameter int K_MAX        = 1024,
    parameter int K_WIDTH      = 11,
    parameter int B_ADDR_W     = 13,
    // 256 requant entries (one per output channel of every layer of a program
    // kept on chip): LeNet-5 needs ~250. 256 x 53 b is one BRAM36. the
    // descriptor's p_base field is 8 bits, so this is also the ceiling.
    parameter int P_ADDR_W     = 8,
    parameter int M_BITS       = 8,                    // add rescale multiplier width
    parameter int DESC_DEPTH   = 32,                   // descriptors (one BRAM18)
    parameter int HB_BIT       = 26,                   // heartbeat ~0.75 Hz at 100 MHz
    parameter int RST_HOLD     = 16
)(
    input  logic       CLK100MHZ,
    input  logic       btn_rst,          // BTN0, active high, asynchronous
    input  logic       uart_txd_in,      // FT2232H TXD -> FPGA (named from the FTDI side)
    output logic       uart_rxd_out,     // FPGA -> FT2232H RXD
    output logic [3:0] led
);

    localparam int DIM_W = 8;
    localparam int CW_W  = 7;
    localparam int SEL_WIDTH = (NBUF > 1) ? $clog2(NBUF) : 1;
    localparam int AXI_ADDR_W = 24;
    localparam int IDX_W = (DESC_DEPTH > 1) ? $clog2(DESC_DEPTH) : 1;

    // ------------------------------------------------------------------
    // clock
    // ------------------------------------------------------------------
    logic clk, mmcm_locked;

    generate
        if (CLK_DIV == 1) begin : g_clk_direct
            assign clk         = CLK100MHZ;
            assign mmcm_locked = 1'b1;
        end else begin : g_clk_mmcm
`ifdef SYNTHESIS
            // VCO = 100 MHz * 10 = 1 GHz, out = 1 GHz / (10 * CLK_DIV)
            logic clk_fb, clk_fb_buf, clk_out;
            MMCME2_BASE #(
                .CLKIN1_PERIOD    (10.0),
                .CLKFBOUT_MULT_F  (10.0),
                .CLKOUT0_DIVIDE_F (10.0 * CLK_DIV)
            ) u_mmcm (
                .CLKIN1   (CLK100MHZ),
                .CLKFBIN  (clk_fb_buf),
                .CLKFBOUT (clk_fb),
                .CLKOUT0  (clk_out),
                .LOCKED   (mmcm_locked),
                .PWRDWN   (1'b0),
                .RST      (1'b0),
                .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(),
                .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6()
            );
            BUFG u_bufg_fb  (.I(clk_fb),  .O(clk_fb_buf));
            BUFG u_bufg_out (.I(clk_out), .O(clk));
`else
            initial $fatal(1, "top: CLK_DIV > 1 needs the MMCM, which is synthesis only");
`endif
        end
    endgenerate

    // ------------------------------------------------------------------
    // reset
    // ------------------------------------------------------------------
    // power-on: these flops start at 0 from configuration (initial values
    // are honoured on 7-series), so reset is held for the first cycles
    logic [4:0] por_cnt = 5'd0;
    logic       por_done;
    assign por_done = &por_cnt;
    always_ff @(posedge clk)
        if (!por_done) por_cnt <= por_cnt + 1'b1;

    // BTN0 synchronizer (async input, the only CDC besides UART RX)
    logic btn_s1 = 1'b1, btn_s2 = 1'b1;
    always_ff @(posedge clk) begin
        btn_s1 <= btn_rst;
        btn_s2 <= btn_s1;
    end

    // stretch any request to RST_HOLD cycles
    logic rst_req;
    logic [$clog2(RST_HOLD + 1)-1:0] rst_cnt = '0;
    logic sys_reset = 1'b1;
    assign rst_req = btn_s2 || !por_done || !mmcm_locked;
    always_ff @(posedge clk) begin
        if (rst_req) begin
            rst_cnt   <= '0;
            sys_reset <= 1'b1;
        end else if (int'(rst_cnt) != RST_HOLD) begin
            rst_cnt   <= rst_cnt + 1'b1;
            sys_reset <= 1'b1;
        end else begin
            sys_reset <= 1'b0;
        end
    end

    // ------------------------------------------------------------------
    // UART and bridge
    // ------------------------------------------------------------------
    logic [7:0] rx_data, tx_data;
    logic rx_valid, rx_frame_err, tx_valid, tx_ready, link_err, rx_dropped;

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .reset(sys_reset), .rx(uart_txd_in),
        .data(rx_data), .valid(rx_valid), .frame_err(rx_frame_err)
    );

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .reset(sys_reset), .data(tx_data), .valid(tx_valid),
        .ready(tx_ready), .tx(uart_rxd_out)
    );

    logic [AXI_ADDR_W-1:0] awaddr, araddr;
    logic awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready;
    logic [31:0] wdata, rdata;
    logic [3:0]  wstrb;
    logic [1:0]  bresp, rresp;

    uart_bridge #(
        .CLKS_PER_BIT (CLKS_PER_BIT),
        .TIMEOUT_BITS (TIMEOUT_BITS),
        .ADDR_W       (AXI_ADDR_W)
    ) u_bridge (
        .clk(clk), .reset(sys_reset),
        .rx_data(rx_data), .rx_valid(rx_valid),
        .tx_data(tx_data), .tx_valid(tx_valid), .tx_ready(tx_ready),
        .link_err(link_err), .rx_dropped(rx_dropped),
        .m_awaddr(awaddr), .m_awvalid(awvalid), .m_awready(awready),
        .m_wdata(wdata), .m_wstrb(wstrb), .m_wvalid(wvalid), .m_wready(wready),
        .m_bresp(bresp), .m_bvalid(bvalid), .m_bready(bready),
        .m_araddr(araddr), .m_arvalid(arvalid), .m_arready(arready),
        .m_rdata(rdata), .m_rresp(rresp), .m_rvalid(rvalid), .m_rready(rready)
    );

    // ------------------------------------------------------------------
    // CSR, sequencer and layer_engine
    // ------------------------------------------------------------------
    // the engine bus: csr_* is the legacy CSR command, seq_* the sequencer's,
    // eng_* what layer_engine sees
    logic soft_reset, csr_start, eng_ready, eng_busy, eng_done, eng_err;
    logic [9:0] eng_err_code;
    logic [DIM_W-1:0] csr_h_in, csr_w_in, csr_h_out, csr_w_out;
    logic [CW_W-1:0] csr_cin_words, csr_cout_words;
    logic [2:0] csr_kh, csr_kw;
    logic [1:0] csr_stride, csr_pad;
    logic signed [7:0] csr_zp_in, csr_zp_out;
    logic csr_relu_en;
    logic [SEL_WIDTH-1:0] csr_src_sel, csr_dst_sel, host_wr_sel, host_rd_sel;
    logic [ADDR_WIDTH-1:0] csr_src_base, csr_dst_base, host_waddr, host_rd_addr;
    logic [B_ADDR_W-1:0] csr_b_base, b_waddr;
    logic [P_ADDR_W-1:0] csr_p_base, p_waddr;
    logic host_mode, host_we, host_rd_en, b_we, p_we;
    logic [N*8-1:0] host_wdata, host_rd_data, b_wdata;
    logic [52:0] p_wdata;
    logic done_sticky, err_sticky;

    logic seq_start, seq_busy, seq_done, seq_err;
    logic [IDX_W-1:0] seq_start_index, seq_cur_index, seq_err_index;
    logic [IDX_W:0] seq_layers_done;
    logic [10:0] seq_err_code;
    logic desc_we, cyc_rd_en;
    logic [IDX_W+3:0] desc_waddr;
    logic [31:0] desc_wdata, cyc_rd_data;
    logic [IDX_W-1:0] cyc_rd_addr;

    logic seq_eng_start;
    logic [2:0] seq_op;
    logic [DIM_W-1:0] seq_h_in, seq_w_in, seq_h_out, seq_w_out;
    logic [CW_W-1:0] seq_cin_words, seq_cout_words;
    logic [2:0] seq_kh, seq_kw;
    logic [1:0] seq_stride, seq_pad;
    logic signed [7:0] seq_zp_in, seq_zp_out, seq_zp_in2;
    logic seq_relu_en;
    logic [SEL_WIDTH-1:0] seq_src_sel, seq_dst_sel, seq_src2_sel;
    logic [ADDR_WIDTH-1:0] seq_src_base, seq_dst_base, seq_src2_base;
    logic [B_ADDR_W-1:0] seq_b_base;
    logic [P_ADDR_W-1:0] seq_p_base;
    logic [M_BITS-1:0] seq_add_ma, seq_add_mb;
    logic [3:0] seq_add_shift;

    csr #(
        .N(N), .ADDR_WIDTH(ADDR_WIDTH), .READ_LATENCY(READ_LATENCY), .NBUF(NBUF),
        .B_ADDR_W(B_ADDR_W), .P_ADDR_W(P_ADDR_W), .K_MAX(K_MAX), .DIM_W(DIM_W),
        .CW_W(CW_W), .BUILD_ID(BUILD_ID), .AXI_ADDR_W(AXI_ADDR_W), .M_BITS(M_BITS),
        .DESC_DEPTH(DESC_DEPTH), .HAS_DMA(0)
    ) u_csr (
        .clk(clk), .reset(sys_reset),
        .s_awaddr(awaddr), .s_awvalid(awvalid), .s_awready(awready),
        .s_wdata(wdata), .s_wstrb(wstrb), .s_wvalid(wvalid), .s_wready(wready),
        .s_bresp(bresp), .s_bvalid(bvalid), .s_bready(bready),
        .s_araddr(araddr), .s_arvalid(arvalid), .s_arready(arready),
        .s_rdata(rdata), .s_rresp(rresp), .s_rvalid(rvalid), .s_rready(rready),
        .link_err(link_err), .link_dropped(rx_dropped),
        .conv_soft_reset(soft_reset), .conv_start(csr_start), .conv_ready(eng_ready),
        .conv_busy(eng_busy), .conv_done(eng_done), .conv_err(eng_err),
        .conv_err_code(eng_err_code[8:0]),
        .h_in(csr_h_in), .w_in(csr_w_in), .h_out(csr_h_out), .w_out(csr_w_out),
        .cin_words(csr_cin_words), .cout_words(csr_cout_words), .kh(csr_kh), .kw(csr_kw),
        .stride(csr_stride), .pad(csr_pad), .zp_in(csr_zp_in), .zp_out(csr_zp_out),
        .relu_en(csr_relu_en), .src_sel(csr_src_sel), .dst_sel(csr_dst_sel),
        .src_base(csr_src_base), .dst_base(csr_dst_base),
        .b_base(csr_b_base), .p_base(csr_p_base),
        .host_mode(host_mode), .host_we(host_we), .host_waddr(host_waddr),
        .host_wdata(host_wdata), .host_wr_sel(host_wr_sel),
        .host_rd_en(host_rd_en), .host_rd_addr(host_rd_addr), .host_rd_sel(host_rd_sel),
        .host_rd_data(host_rd_data),
        .b_we(b_we), .b_waddr(b_waddr), .b_wdata(b_wdata),
        .p_we(p_we), .p_waddr(p_waddr), .p_wdata(p_wdata),
        .done_sticky(done_sticky), .err_sticky(err_sticky),
        .seq_start(seq_start), .seq_start_index(seq_start_index), .seq_busy(seq_busy),
        .seq_done(seq_done), .seq_err(seq_err), .seq_cur_index(seq_cur_index),
        .seq_layers_done(seq_layers_done), .seq_err_index(seq_err_index),
        .seq_err_code(seq_err_code),
        .desc_we(desc_we), .desc_waddr(desc_waddr), .desc_wdata(desc_wdata),
        .cyc_rd_en(cyc_rd_en), .cyc_rd_addr(cyc_rd_addr), .cyc_rd_data(cyc_rd_data)
    );

    // the sequencer: soft reset stops a program, like it stops a layer
    sequencer #(
        .ADDR_WIDTH(ADDR_WIDTH), .NBUF(NBUF), .B_ADDR_W(B_ADDR_W), .P_ADDR_W(P_ADDR_W),
        .DIM_W(DIM_W), .CW_W(CW_W), .M_BITS(M_BITS), .DESC_DEPTH(DESC_DEPTH),
        .READ_LATENCY(READ_LATENCY)
    ) u_seq (
        .clk(clk), .reset(sys_reset || soft_reset),
        .start(seq_start), .start_index(seq_start_index), .busy(seq_busy),
        .done(seq_done), .err(seq_err), .cur_index(seq_cur_index),
        .layers_done(seq_layers_done), .err_index(seq_err_index), .err_code(seq_err_code),
        .desc_we(desc_we), .desc_waddr(desc_waddr), .desc_wdata(desc_wdata),
        .cyc_rd_en(cyc_rd_en), .cyc_rd_addr(cyc_rd_addr), .cyc_rd_data(cyc_rd_data),
        .eng_start(seq_eng_start), .eng_op(seq_op), .h_in(seq_h_in), .w_in(seq_w_in),
        .h_out(seq_h_out), .w_out(seq_w_out), .cin_words(seq_cin_words),
        .cout_words(seq_cout_words), .kh(seq_kh), .kw(seq_kw), .stride(seq_stride),
        .pad(seq_pad), .zp_in(seq_zp_in), .zp_out(seq_zp_out), .zp_in2(seq_zp_in2),
        .relu_en(seq_relu_en), .src_sel(seq_src_sel), .dst_sel(seq_dst_sel),
        .src2_sel(seq_src2_sel), .src_base(seq_src_base), .dst_base(seq_dst_base),
        .src2_base(seq_src2_base), .b_base(seq_b_base), .p_base(seq_p_base),
        .add_ma(seq_add_ma), .add_mb(seq_add_mb), .add_shift(seq_add_shift),
        .eng_ready(eng_ready), .eng_done(eng_done), .eng_err(eng_err),
        .eng_err_code(eng_err_code)
    );

    // layer_engine: conv, pool and add on one act_buffer. the command bus
    // belongs to the sequencer while a program runs, otherwise to the CSR's
    // legacy command, which is always a conv with the add fields at 0. the
    // CSR refuses CTRL.start while a program runs and SEQ_CTRL.start while
    // anything runs, so the two starts never meet. E_MODE cannot come from the
    // legacy command, so the CSR's 9-bit ERR_CODE still holds all of its bits.
    logic [DIM_W-1:0] h_in, w_in, h_out, w_out;
    logic [CW_W-1:0] cin_words, cout_words;
    logic [2:0] kh, kw;
    logic [1:0] stride, pad;
    logic signed [7:0] zp_in, zp_out;
    logic relu_en;
    logic [SEL_WIDTH-1:0] src_sel, dst_sel;
    logic [ADDR_WIDTH-1:0] src_base, dst_base;
    logic [B_ADDR_W-1:0] b_base;
    logic [P_ADDR_W-1:0] p_base;

    always_comb begin
        if (seq_busy) begin
            {h_in, w_in, h_out, w_out} = {seq_h_in, seq_w_in, seq_h_out, seq_w_out};
            {cin_words, cout_words}    = {seq_cin_words, seq_cout_words};
            {kh, kw, stride, pad}      = {seq_kh, seq_kw, seq_stride, seq_pad};
            {zp_in, zp_out, relu_en}   = {seq_zp_in, seq_zp_out, seq_relu_en};
            {src_sel, dst_sel}         = {seq_src_sel, seq_dst_sel};
            {src_base, dst_base}       = {seq_src_base, seq_dst_base};
            {b_base, p_base}           = {seq_b_base, seq_p_base};
        end else begin
            {h_in, w_in, h_out, w_out} = {csr_h_in, csr_w_in, csr_h_out, csr_w_out};
            {cin_words, cout_words}    = {csr_cin_words, csr_cout_words};
            {kh, kw, stride, pad}      = {csr_kh, csr_kw, csr_stride, csr_pad};
            {zp_in, zp_out, relu_en}   = {csr_zp_in, csr_zp_out, csr_relu_en};
            {src_sel, dst_sel}         = {csr_src_sel, csr_dst_sel};
            {src_base, dst_base}       = {csr_src_base, csr_dst_base};
            {b_base, p_base}           = {csr_b_base, csr_p_base};
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk)
        if (!sys_reset && csr_start && seq_busy)
            $fatal(1, "top: legacy start while a program runs (the CSR must refuse it)");
`endif

    layer_engine #(
        .N(N), .ADDR_WIDTH(ADDR_WIDTH), .READ_LATENCY(READ_LATENCY), .NBUF(NBUF),
        .K_MAX(K_MAX), .K_WIDTH(K_WIDTH), .B_ADDR_W(B_ADDR_W), .P_ADDR_W(P_ADDR_W),
        .DIM_W(DIM_W), .CW_W(CW_W), .M_BITS(M_BITS)
    ) u_engine (
        .clk(clk), .reset(sys_reset || soft_reset),
        .start(seq_busy ? seq_eng_start : csr_start), .op(seq_busy ? seq_op : 3'd1),
        .ready(eng_ready), .busy(eng_busy), .done(eng_done),
        .err(eng_err), .err_code(eng_err_code), .done_op(),
        .conv_busy(), .pool_busy(), .add_busy(), .tile_launch(),
        .src2_sel(seq_busy ? seq_src2_sel : '0), .src2_base(seq_busy ? seq_src2_base : '0),
        .zp_in2(seq_busy ? seq_zp_in2 : 8'sd0), .add_ma(seq_busy ? seq_add_ma : '0),
        .add_mb(seq_busy ? seq_add_mb : '0), .add_shift(seq_busy ? seq_add_shift : 4'd0),
        .h_in(h_in), .w_in(w_in), .h_out(h_out), .w_out(w_out),
        .cin_words(cin_words), .cout_words(cout_words), .kh(kh), .kw(kw),
        .stride(stride), .pad(pad), .zp_in(zp_in), .zp_out(zp_out), .relu_en(relu_en),
        .src_sel(src_sel), .dst_sel(dst_sel), .src_base(src_base), .dst_base(dst_base),
        .b_base(b_base), .p_base(p_base),
        .b_we(b_we), .b_waddr(b_waddr), .b_wdata(b_wdata),
        .p_we(p_we), .p_waddr(p_waddr), .p_wdata(p_wdata),
        .host_mode(host_mode), .host_we(host_we), .host_waddr(host_waddr),
        .host_wdata(host_wdata), .host_wr_sel(host_wr_sel),
        .host_rd_en(host_rd_en), .host_rd_addr(host_rd_addr), .host_rd_sel(host_rd_sel),
        .host_rd_data(host_rd_data)
    );

    // ------------------------------------------------------------------
    // LEDs
    // ------------------------------------------------------------------
    logic [HB_BIT:0] hb_cnt;
    always_ff @(posedge clk)
        if (sys_reset) hb_cnt <= '0;
        else           hb_cnt <= hb_cnt + 1'b1;

    // LD5 shows a program or a legacy layer running; LD6/LD7 are the legacy
    // command's sticky bits (a program reports through SEQ_STATUS)
    assign led = {err_sticky, done_sticky, eng_busy || seq_busy, hb_cnt[HB_BIT]};

endmodule
