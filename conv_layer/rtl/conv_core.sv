// conv_core: conv_layer without its act_buffer. everything that computes a
// layer (sequencer, check phase, im2col_reader, gemm_tile, write-back) is
// here; the act_buffer and the mux that hands its ports to the host live
// outside, in conv_layer (the original block, now a thin wrapper) or in
// layer_engine (which shares one act_buffer between conv, pool and add).
// split out so a second engine can use the same buffer without touching
// this datapath; behaviour is unchanged, which the unchanged conv_layer TB
// checks on the wrapper.
//
//   act_buffer bank src_sel -> im2col_reader -> gemm_tile -> write-back -> bank dst_sel
//
// the full contract is conv_layer/conv_layer_contract.md; the parts a user of
// the block needs are here. this is the reference version: tiles are strictly
// serialized, the next one starts only after the previous one's last output
// word has been written.
//
// memory layout (host packs it, see conv_layer_model.py):
//   act_buffer : pixel (y, x), channel group g at base + (y*w + x)*c_words + g,
//                lane j = channel g*N + j. unused input lanes hold zp_in.
//   B          : output-channel group og, row k at b_base + og*K + k,
//                K = kh*kw*cin_words*N, byte c = weight of output channel og*N + c.
//   P          : output channel og*N + c at p_base + og*N + c, {shift, mult, bias}.
//                bias already has zp_in folded in: bias - zp_in * sum_k w[k][c].
//                unused output channels: {0, 0, 0}, so they come out as zp_out.
//
// loop order: output-channel group outer, pixel tile inner. a tile is N output
//   pixels in raster order; tile t covers pixels t*N .. t*N+N-1 and output row
//   r of gemm_tile is pixel t*N + r. it is written to
//   dst_base + (t*N + r)*cout_words + og, and only if t*N + r < h_out*w_out.
//
// command: latched on start & ready. a check phase follows (CHECK_CYCLES
//   edges, measured); on a failure err pulses, err_code holds the failing
//   checks and nothing is read or written. otherwise the layer runs and done
//   pulses on the edge where the last output word is written.
//
// ownership: while idle the host writes the act_buffer through host_mode and
//   the host_* write port, reads it through host_rd_* (with host_mode low),
//   and writes B and P. while the layer runs none of that is allowed
//   (simulation $fatal); host_rd_* are ignored.
//
// reset is synchronous. no act_buffer write lands on or after the reset edge,
// every destination word is left either old or final, ready is high after it.

module conv_core #(
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

    // host_mode is only observed here: E_HOST at start, $fatal while running.
    // the act_buffer it really controls lives outside.
    input  logic host_mode,

    // act_buffer read port. driven all the time; the owner mux outside
    // passes it to the buffer only while this core is busy
    output logic act_rd_en,
    output logic [ADDR_WIDTH-1:0] act_rd_addr,
    output logic [SEL_WIDTH-1:0] act_rd_sel,
    input  logic [N*DATA_WIDTH-1:0] act_rd_data,

    // act_buffer datapath write port (the write-back)
    output logic act_we,
    output logic [ADDR_WIDTH-1:0] act_waddr,
    output logic [N*DATA_WIDTH-1:0] act_wdata,
    output logic [SEL_WIDTH-1:0] act_wr_sel,

    // one-cycle pulse per tile launched (status; the TBs count tiles with it)
    output logic tile_launch
);

    localparam int WORD_W = N * DATA_WIDTH;
    // the check phase works in one generous width so no product can wrap:
    // the largest is h*w*c_words, 2*DIM_W + CW_W bits, plus a base address
    localparam int CHK_W  = 2*DIM_W + 2*CW_W + 3 + 3 + ROW_W + 16;
    // pixel indices go up to h_out*w_out plus one tile
    localparam int PIX_W  = 2*DIM_W + ROW_W + 1;

    initial begin
        if (DATA_WIDTH != 8)
            $fatal(1, "conv_layer: DATA_WIDTH must be 8 (got %0d)", DATA_WIDTH);
        if (NBUF < 2)
            $fatal(1, "conv_layer: NBUF must be >= 2, src and dst banks differ (got %0d)", NBUF);
        if (SEL_WIDTH != ((NBUF > 1) ? $clog2(NBUF) : 1) || ROW_W != ((N > 1) ? $clog2(N) : 1)
            || P_WIDTH != 53)
            $fatal(1, "conv_layer: SEL_WIDTH, ROW_W and P_WIDTH are derived, do not override them");
    end

    // ------------------------------------------------------------------
    // sequencer
    // ------------------------------------------------------------------
    typedef enum logic [2:0] {
        S_IDLE,
        S_CHK1,      // products of the latched fields
        S_CHK2,      // K, end addresses
        S_CHK3,      // B end, all compares registered
        S_DECIDE,    // err or run
        S_LAUNCH,    // wait for both blocks ready, then start the tile
        S_RUN        // wait for the tile's last write
    } state_t;
    state_t state;

    // latched command
    logic [DIM_W-1:0] h_in_q, w_in_q, h_out_q, w_out_q;
    logic [CW_W-1:0] cin_q, cout_q;
    logic [2:0] kh_q, kw_q;
    logic [1:0] stride_q, pad_q;
    logic signed [DATA_WIDTH-1:0] zp_in_q, zp_out_q;
    logic relu_q;
    logic [SEL_WIDTH-1:0] src_sel_q, dst_sel_q;
    logic [ADDR_WIDTH-1:0] src_base_q, dst_base_q;
    logic [B_ADDR_W-1:0] b_base_q;
    logic [P_ADDR_W-1:0] p_base_q;
    logic host_at_start_q;

    // check phase registers
    logic [CHK_W-1:0] kk_c, inpx_c, outpx_c, h_lo_c, h_hi_c, w_lo_c, w_hi_c;
    logic [CHK_W-1:0] k_c, src_end_c, dst_end_c, p_end_c, b_end_c;
    logic [8:0] chk_code;

    // layer position
    logic [CW_W-1:0] og;                    // output-channel group
    logic [PIX_W-1:0] pix0;                 // first pixel of the tile
    logic [PIX_W-1:0] n_pix;                // h_out * w_out
    logic [DIM_W-1:0] y0, x0;               // tile origin for im2col_reader
    logic [ROW_W:0] step_cnt;               // raster steps still owed to (y0, x0)
    logic [ADDR_WIDTH-1:0] grp_base;        // dst_base + og
    logic [ADDR_WIDTH-1:0] row_base;        // dst_base + pix0*cout_words + og
    logic [B_ADDR_W-1:0] b_cur;             // b_base + og*K
    logic [P_ADDR_W-1:0] p_cur;             // p_base + og*N
    logic [K_WIDTH-1:0] k_q;                // K for gemm_tile's k_in

    // tile handshake
    logic tile_start;
    logic rd_ready, g_ready, g_busy, g_err_k;

    // write-back register
    logic wb_valid, wb_last, wb_inrange;
    logic [ADDR_WIDTH-1:0] wb_addr;
    logic [WORD_W-1:0] wb_data;
    logic wb_we;

    logic running;
    assign running = (state == S_LAUNCH) || (state == S_RUN);

    assign ready = (state == S_IDLE);
    assign busy  = !ready;

    // both blocks idle and the next tile origin fully stepped. im2col_reader
    // and gemm_tile are always idle here in practice (the reader finishes
    // before the array drains, gemm_tile raises ready before its last rows
    // leave), but waiting on them costs nothing and makes it obvious.
    assign tile_start = (state == S_LAUNCH) && rd_ready && g_ready && (step_cnt == '0);
    assign tile_launch = tile_start;

    // the last write of a tile lands on the edge where wb_last is registered
    // and the sequencer sees it; tlast is row 0, which is always in range
    logic tile_end, last_tile, last_group;
    assign tile_end   = (state == S_RUN) && wb_valid && wb_last;
    assign last_tile  = (pix0 + PIX_W'(N)) >= n_pix;
    assign last_group = (og == cout_q - 1'b1);

    // the checks, from registered values only. E_SHAPE is only meaningful
    // once no dimension is zero (h_out - 1 and the stride would underflow).
    localparam logic [CHK_W-1:0] ONE = 1;
    logic e_dim, e_shape, e_k, e_b, e_p, e_src, e_dst, e_sel;
    logic [CHK_W-1:0] h_span, w_span;
    assign h_span = CHK_W'(h_in_q) + 2*CHK_W'(pad_q);
    assign w_span = CHK_W'(w_in_q) + 2*CHK_W'(pad_q);
    assign e_dim  = (h_in_q == 0) || (w_in_q == 0) || (h_out_q == 0) || (w_out_q == 0) ||
                    (cin_q == 0) || (cout_q == 0) || (kh_q == 0) || (kw_q == 0) || (stride_q == 0);
    // h_out == floor((h_span - kh)/stride) + 1, kh <= h_span
    //   <=> (h_out-1)*stride + kh <= h_span < h_out*stride + kh
    assign e_shape = !e_dim && !((h_lo_c <= h_span) && (h_span < h_hi_c) &&
                                 (w_lo_c <= w_span) && (w_span < w_hi_c));
    assign e_k    = k_c > CHK_W'(K_MAX);
    assign e_b    = b_end_c > (ONE << B_ADDR_W);
    assign e_p    = p_end_c > (ONE << P_ADDR_W);
    assign e_src  = src_end_c > (ONE << ADDR_WIDTH);
    assign e_dst  = dst_end_c > (ONE << ADDR_WIDTH);
    assign e_sel  = (src_sel_q == dst_sel_q) || (CHK_W'(src_sel_q) >= CHK_W'(NBUF)) ||
                    (CHK_W'(dst_sel_q) >= CHK_W'(NBUF));

    always_ff @(posedge clk) begin
        if (reset) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            err      <= 1'b0;
            err_code <= '0;
            step_cnt <= '0;
            og       <= '0;
            pix0     <= '0;
            y0       <= '0;
            x0       <= '0;
        end else begin
            done <= 1'b0;
            err  <= 1'b0;

            // raster stepper: owes N steps after each tile start. it moves the
            // origin one pixel per cycle, which avoids a divide by w_out; a
            // tile takes far more than N cycles, so it never delays a launch.
            if (step_cnt != '0) begin
                step_cnt <= step_cnt - 1'b1;
                if (x0 == w_out_q - 1'b1) begin
                    x0 <= '0;
                    y0 <= y0 + 1'b1;
                end else begin
                    x0 <= x0 + 1'b1;
                end
            end

            case (state)

                S_IDLE: begin
                    if (start) begin
                        h_in_q   <= h_in;
                        w_in_q   <= w_in;
                        h_out_q  <= h_out;
                        w_out_q  <= w_out;
                        cin_q    <= cin_words;
                        cout_q   <= cout_words;
                        kh_q     <= kh;
                        kw_q     <= kw;
                        stride_q <= stride;
                        pad_q    <= pad;
                        zp_in_q  <= zp_in;
                        zp_out_q <= zp_out;
                        relu_q   <= relu_en;
                        src_sel_q  <= src_sel;
                        dst_sel_q  <= dst_sel;
                        src_base_q <= src_base;
                        dst_base_q <= dst_base;
                        b_base_q <= b_base;
                        p_base_q <= p_base;
                        host_at_start_q <= host_mode;
                        err_code <= '0;
                        state    <= S_CHK1;
                    end
                end

                S_CHK1: begin
                    kk_c    <= CHK_W'(kh_q) * CHK_W'(kw_q);
                    inpx_c  <= CHK_W'(h_in_q) * CHK_W'(w_in_q);
                    outpx_c <= CHK_W'(h_out_q) * CHK_W'(w_out_q);
                    h_lo_c  <= (CHK_W'(h_out_q) - ONE) * CHK_W'(stride_q) + CHK_W'(kh_q);
                    h_hi_c  <= CHK_W'(h_out_q) * CHK_W'(stride_q) + CHK_W'(kh_q);
                    w_lo_c  <= (CHK_W'(w_out_q) - ONE) * CHK_W'(stride_q) + CHK_W'(kw_q);
                    w_hi_c  <= CHK_W'(w_out_q) * CHK_W'(stride_q) + CHK_W'(kw_q);
                    state   <= S_CHK2;
                end

                S_CHK2: begin
                    k_c       <= kk_c * CHK_W'(cin_q) * CHK_W'(N);
                    src_end_c <= CHK_W'(src_base_q) + inpx_c * CHK_W'(cin_q);
                    dst_end_c <= CHK_W'(dst_base_q) + outpx_c * CHK_W'(cout_q);
                    p_end_c   <= CHK_W'(p_base_q) + CHK_W'(cout_q) * CHK_W'(N);
                    state     <= S_CHK3;
                end

                S_CHK3: begin
                    b_end_c  <= CHK_W'(b_base_q) + CHK_W'(cout_q) * k_c;
                    chk_code <= {host_at_start_q, e_sel, e_dst, e_src, e_p, 1'b0, e_k, e_shape, e_dim};
                    state    <= S_DECIDE;
                end

                S_DECIDE: begin
                    if (chk_code != '0 || e_b) begin
                        err      <= 1'b1;
                        err_code <= chk_code | (9'(e_b) << 3);
                        state    <= S_IDLE;
                    end else begin
                        // first group, first tile
                        k_q      <= K_WIDTH'(k_c);
                        n_pix    <= PIX_W'(outpx_c);
                        og       <= '0;
                        pix0     <= '0;
                        y0       <= '0;
                        x0       <= '0;
                        grp_base <= dst_base_q;
                        row_base <= dst_base_q;
                        b_cur    <= b_base_q;
                        p_cur    <= p_base_q;
                        state    <= S_LAUNCH;
                    end
                end

                S_LAUNCH: begin
                    if (tile_start) begin
                        step_cnt <= (ROW_W+1)'(N);
                        state    <= S_RUN;
                    end
                end

                S_RUN: begin
                    if (tile_end) begin
                        if (!last_tile) begin
                            // next tile of this group; (y0, x0) is already stepped
                            pix0     <= pix0 + PIX_W'(N);
                            row_base <= row_base + ADDR_WIDTH'(N * cout_q);
                            state    <= S_LAUNCH;
                        end else if (!last_group) begin
                            // next output-channel group: same pixels again,
                            // next N weight columns and parameter words
                            og       <= og + 1'b1;
                            pix0     <= '0;
                            y0       <= '0;
                            x0       <= '0;
                            grp_base <= grp_base + 1'b1;
                            row_base <= grp_base + 1'b1;
                            b_cur    <= b_cur + B_ADDR_W'(k_q);
                            p_cur    <= p_cur + P_ADDR_W'(N);
                            state    <= S_LAUNCH;
                        end else begin
                            // the last word lands on this edge
                            done  <= 1'b1;
                            state <= S_IDLE;
                        end
                    end
                end

                default: state <= S_IDLE;

            endcase
        end
    end

    // ------------------------------------------------------------------
    // datapath
    // ------------------------------------------------------------------
    logic [WORD_W-1:0] act_rdata;
    logic a_valid, a_take;
    logic [WORD_W-1:0] a_data;

    // the reader drives the read port directly; the bank is always the
    // layer's source. the write port carries the write-back to the layer's
    // destination bank.
    assign act_rdata  = act_rd_data;
    assign act_rd_sel = src_sel_q;
    assign act_we     = wb_we;
    assign act_waddr  = wb_addr;
    assign act_wdata  = wb_data;
    assign act_wr_sel = dst_sel_q;

    im2col_reader #(
        .N            (N),
        .DATA_WIDTH   (DATA_WIDTH),
        .ADDR_W       (ADDR_WIDTH),
        .READ_LATENCY (READ_LATENCY),
        .DIM_W        (DIM_W),
        .CW_W         (CW_W)
    ) u_reader (
        .clk       (clk),
        .reset     (reset),
        .start     (tile_start),
        .ready     (rd_ready),
        .y0        (y0),
        .x0        (x0),
        .win_order (1'b0),          // raster only; pooling order is out of scope
        .h_in      (h_in_q),
        .w_in      (w_in_q),
        .w_out     (w_out_q),
        .cin_words (cin_q),
        .kh        (kh_q),
        .kw        (kw_q),
        .stride    (stride_q),
        .pad       (pad_q),
        .pad_value (zp_in_q),       // padded pixels must be the input zero point
        .src_base  (src_base_q),
        .rd_en     (act_rd_en),
        .rd_addr   (act_rd_addr),
        .rd_data   (act_rdata),
        .a_valid   (a_valid),
        .a_data    (a_data),
        .a_take    (a_take)
    );

    logic m_valid, m_last;
    logic [WORD_W-1:0] m_data;
    logic [ROW_W-1:0] m_row;

    gemm_tile #(
        .N            (N),
        .DATA_WIDTH   (DATA_WIDTH),
        .K_MAX        (K_MAX),
        .K_WIDTH      (K_WIDTH),
        .READ_LATENCY (READ_LATENCY),
        .B_ADDR_W     (B_ADDR_W),
        .P_ADDR_W     (P_ADDR_W)
    ) u_gemm (
        .clk           (clk),
        .reset         (reset),
        .start         (tile_start),
        .k_in          (k_q),
        .b_base        (b_cur),
        .p_base        (p_cur),
        .zero_point    (zp_out_q),
        .relu_en       (relu_q),
        .ready         (g_ready),
        .busy          (g_busy),
        .err_k         (g_err_k),
        .a_valid       (a_valid),
        .a_data        (a_data),
        .a_take        (a_take),
        .b_we          (b_we),
        .b_waddr       (b_waddr),
        .b_wdata       (b_wdata),
        .p_we          (p_we),
        .p_waddr       (p_waddr),
        .p_wdata       (p_wdata),
        .m_axis_tvalid (m_valid),
        // the write port takes a word every cycle and nothing else uses it
        // while busy, so there is never a reason to push back
        .m_axis_tready (1'b1),
        .m_axis_tdata  (m_data),
        .m_axis_tuser  (m_row),
        .m_axis_tlast  (m_last)
    );

    // write-back: one register stage, so the row*cout_words multiply and the
    // add are off the path into the BRAM write port. rows past the end of the
    // output (last tile only) are captured but never written.
    always_ff @(posedge clk) begin
        if (reset) begin
            wb_valid <= 1'b0;
            wb_last  <= 1'b0;
            wb_inrange <= 1'b0;
        end else begin
            wb_valid   <= m_valid && running;
            wb_last    <= m_last;
            wb_inrange <= (pix0 + PIX_W'(m_row)) < n_pix;
            wb_addr    <= row_base + ADDR_WIDTH'(m_row * cout_q);
            wb_data    <= m_data;
        end
    end

    // !reset: gemm_tile's skid buffer can be holding a beat on the reset edge,
    // and the contract promises that no write lands on or after it
    assign wb_we = wb_valid && wb_inrange && !reset;

    // Simulation-only checks
`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (!reset && running) begin
            if (host_mode)
                $fatal(1, "conv_layer: host_mode raised while the layer runs");
            if (b_we || p_we)
                $fatal(1, "conv_layer: B/P memory written while the layer runs");
        end
        if (!reset && g_err_k)
            $fatal(1, "conv_layer: gemm_tile rejected k_in=%0d, the check phase let it through", k_q);
    end
`endif

endmodule
