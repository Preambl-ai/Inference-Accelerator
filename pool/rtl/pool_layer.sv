// pool_layer: one pooling layer out of on-chip memory, between two act_buffer
// banks, while conv_layer is idle. contract: pool/pool_contract.md.
//
//   mode 1, MAX2 : 2x2 max pool, stride 2, floor on odd sizes
//   mode 2, GAVG : global average over h*w = 2^S pixels, round half up
//
// no multipliers anywhere: sizes in the check phase come from bit-serial
// shift-and-add, and run-time addresses from pointers that only add a
// constant. the datapath is a signed compare (MAX2) or an add (GAVG) per lane.
//
// one read per cycle with rd_en held high. each read carries a tag {first,
// last, final, output address} through a READ_LATENCY-deep delay, which meets
// the data as it comes out of the act_buffer; the lane accumulators load on
// first, combine otherwise, and hand the result to a write register on last.
//
// the act_buffer ports are only meaningful while busy: layer_engine grants
// them to the busy engine.

module pool_layer #(
    parameter int N            = 8,
    parameter int DATA_WIDTH   = 8,
    parameter int ADDR_WIDTH   = 12,
    parameter int READ_LATENCY = 1,
    parameter int NBUF         = 2,
    parameter int DIM_W        = 8,
    parameter int CW_W         = 7,
    // derived, do not override
    parameter int SEL_WIDTH    = (NBUF > 1) ? $clog2(NBUF) : 1
)(
    input  logic clk,
    input  logic reset,                 // synchronous

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

    // act_buffer read port
    output logic rd_en,
    output logic [ADDR_WIDTH-1:0] rd_addr,
    output logic [SEL_WIDTH-1:0] rd_sel,
    input  logic [N*DATA_WIDTH-1:0] rd_data,

    // act_buffer datapath write port
    output logic we,
    output logic [ADDR_WIDTH-1:0] waddr,
    output logic [N*DATA_WIDTH-1:0] wdata,
    output logic [SEL_WIDTH-1:0] wr_sel
);

    localparam int WORD_W = N * DATA_WIDTH;
    localparam int ACC_W  = DATA_WIDTH + 2*DIM_W;          // sum of up to 2^(2*DIM_W) int8
    localparam int HW_W   = 2*DIM_W;
    // one width for the check arithmetic: the largest product is h*w*c_words,
    // and it is compared with 2^ADDR_WIDTH
    localparam int CHK_W  = ((2*DIM_W + CW_W > ADDR_WIDTH) ? 2*DIM_W + CW_W : ADDR_WIDTH) + 2;
    localparam logic [1:0] M_MAX2 = 2'd1, M_GAVG = 2'd2;

    initial begin
        if (DATA_WIDTH != 8)
            $fatal(1, "pool_layer: DATA_WIDTH must be 8 (got %0d)", DATA_WIDTH);
        if (NBUF < 2)
            $fatal(1, "pool_layer: NBUF must be >= 2, src and dst banks differ (got %0d)", NBUF);
        if (SEL_WIDTH != ((NBUF > 1) ? $clog2(NBUF) : 1))
            $fatal(1, "pool_layer: SEL_WIDTH is derived, do not override it");
        if (READ_LATENCY < 1)
            $fatal(1, "pool_layer: READ_LATENCY must be >= 1");
    end

    typedef enum logic [2:0] {S_IDLE, S_MUL1, S_MUL2, S_DECIDE, S_RUN, S_DRAIN} state_t;
    state_t state;

    assign ready = (state == S_IDLE);
    assign busy  = !ready;

    // latched command
    logic [1:0] mode_q;
    logic [DIM_W-1:0] h_q, w_q;
    logic [CW_W-1:0] cw_q;
    logic [SEL_WIDTH-1:0] src_sel_q, dst_sel_q;
    logic [ADDR_WIDTH-1:0] src_base_q, dst_base_q;
    logic host_q;

    // ------------------------------------------------------------------
    // check phase: bit-serial products, shifts and adds only
    // ------------------------------------------------------------------
    logic [$clog2(DIM_W + CW_W + 1)-1:0] mi;        // step of the current round
    logic [CHK_W-1:0] row_c;                        // w * c_words
    logic [CHK_W-1:0] hw_c;                         // h * w
    logic [CHK_W-1:0] opx_c;                        // (h/2) * (w/2)
    logic [CHK_W-1:0] inw_c;                        // h * w * c_words
    logic [CHK_W-1:0] outw_c;                       // output words
    logic [DIM_W-1:0] h2, w2;
    assign h2 = h_q >> 1;
    assign w2 = w_q >> 1;

    // shift amount for GAVG: the position of the one set bit of h*w
    logic [4:0] s_c;
    logic pow2;
    always_comb begin
        s_c = '0;
        for (int i = 0; i < HW_W; i++)
            if (hw_c[i]) s_c = 5'(i);
        pow2 = (hw_c != 0) && ((hw_c & (hw_c - 1'b1)) == 0);
    end

    logic e_dim, e_mode, e_shape, e_src, e_dst, e_sel;
    localparam logic [CHK_W-1:0] ONE = 1;
    assign e_dim   = (h_q == 0) || (w_q == 0) || (cw_q == 0);
    assign e_mode  = (mode_q != M_MAX2) && (mode_q != M_GAVG);
    assign e_shape = !e_dim && !e_mode &&
                     ((mode_q == M_MAX2) ? ((h_q < 2) || (w_q < 2)) : !pow2);
    assign e_src   = CHK_W'(src_base_q) + inw_c > (ONE << ADDR_WIDTH);
    assign e_dst   = CHK_W'(dst_base_q) + ((mode_q == M_MAX2) ? outw_c : CHK_W'(cw_q))
                     > (ONE << ADDR_WIDTH);
    assign e_sel   = (src_sel_q == dst_sel_q) || (int'(src_sel_q) >= NBUF) ||
                     (int'(dst_sel_q) >= NBUF);

    // ------------------------------------------------------------------
    // issue: one read per cycle while running
    // ------------------------------------------------------------------
    logic [4:0] shift_q;                            // S for GAVG
    logic [ADDR_WIDTH-1:0] row_q;                   // w * c_words, as an address step
    logic [HW_W-1:0] hw_q;                          // h * w
    // MAX2 counters and pointers
    logic [DIM_W-1:0] oy, ox;
    logic [CW_W-1:0] g;
    logic [1:0] tap;
    logic [ADDR_WIDTH-1:0] row_ptr;                 // pixel (2oy, 0), group 0
    logic [ADDR_WIDTH-1:0] px_ptr;                  // pixel (2oy, 2ox), group 0
    logic [ADDR_WIDTH-1:0] out_ptr;                 // next output word
    // GAVG counters
    logic [HW_W-1:0] p;
    logic [ADDR_WIDTH-1:0] g_ptr;                   // src_base + g
    logic [ADDR_WIDTH-1:0] rd_ptr;                  // src_base + p*c_words + g

    logic [ADDR_WIDTH-1:0] iss_addr, iss_out, tap_off;
    logic iss_first, iss_last, iss_final;
    always_comb begin
        case (tap)
            2'd0:    tap_off = '0;
            2'd1:    tap_off = ADDR_WIDTH'(cw_q);
            2'd2:    tap_off = row_q;
            default: tap_off = row_q + ADDR_WIDTH'(cw_q);
        endcase
        if (mode_q == M_MAX2) begin
            iss_addr  = px_ptr + ADDR_WIDTH'(g) + tap_off;
            iss_first = (tap == 2'd0);
            iss_last  = (tap == 2'd3);
            iss_out   = out_ptr;
            iss_final = iss_last && (g == cw_q - 1'b1) && (ox == w2 - 1'b1) && (oy == h2 - 1'b1);
        end else begin
            iss_addr  = rd_ptr;
            iss_first = (p == '0);
            iss_last  = (p == hw_q - 1'b1);
            iss_out   = dst_base_q + ADDR_WIDTH'(g);
            iss_final = iss_last && (g == cw_q - 1'b1);
        end
    end

    assign rd_en   = (state == S_RUN) || (state == S_DRAIN);
    assign rd_addr = iss_addr;
    assign rd_sel  = src_sel_q;

    // tag pipeline: meets the data READ_LATENCY edges after the read is
    // issued. rd_en is high the whole time, so the buffer's pipeline and this
    // one advance together.
    logic t_valid, t_first, t_last, t_final;
    logic [ADDR_WIDTH-1:0] t_out;
    delay #(.DATA_WIDTH(4 + ADDR_WIDTH), .DEPTH(READ_LATENCY)) u_tag (
        .clk      (clk),
        .reset    (reset),
        .en       (1'b1),
        .in_data  ({(state == S_RUN), iss_first, iss_last, iss_final, iss_out}),
        .out_data ({t_valid, t_first, t_last, t_final, t_out})
    );

    // ------------------------------------------------------------------
    // combine and write
    // ------------------------------------------------------------------
    logic [ACC_W-1:0] acc [0:N-1];
    logic wb_valid, wb_final;
    logic [ADDR_WIDTH-1:0] wb_addr;
    logic [WORD_W-1:0] wb_data;

    // blocking temporaries, only read in the pass that writes them
    logic signed [ACC_W-1:0] d_ext, cur, rounded;
    logic signed [DATA_WIDTH-1:0] d8, m8;

    assign we     = wb_valid && !reset;    // nothing lands on or after a reset edge
    assign waddr  = wb_addr;
    assign wdata  = wb_data;
    assign wr_sel = dst_sel_q;

    always_ff @(posedge clk) begin
        if (reset) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            err      <= 1'b0;
            err_code <= '0;
            wb_valid <= 1'b0;
            wb_final <= 1'b0;
            for (int j = 0; j < N; j++) acc[j] <= '0;
        end else begin
            done     <= 1'b0;
            err      <= 1'b0;
            wb_valid <= 1'b0;

            // ---- combine the word that arrives this cycle
            if (t_valid) begin
                for (int j = 0; j < N; j++) begin
                    d8    = rd_data[j*DATA_WIDTH +: DATA_WIDTH];
                    d_ext = ACC_W'(d8);
                    if (mode_q == M_MAX2) begin
                        m8  = acc[j][DATA_WIDTH-1:0];
                        cur = (t_first || d8 > m8) ? d_ext : ACC_W'(m8);
                        acc[j] <= cur;
                        wb_data[j*DATA_WIDTH +: DATA_WIDTH] <= cur[DATA_WIDTH-1:0];
                    end else begin
                        cur = t_first ? d_ext : $signed(acc[j]) + d_ext;
                        acc[j] <= cur;
                        // round half up: + 2^(S-1), then arithmetic shift by S.
                        // the constant is made signed on purpose: one unsigned
                        // operand would make the sum unsigned and >>> logical
                        rounded = (shift_q == 0) ? cur
                                  : ((cur + $signed(ACC_W'(1) << (shift_q - 1'b1))) >>> shift_q);
                        wb_data[j*DATA_WIDTH +: DATA_WIDTH] <= rounded[DATA_WIDTH-1:0];
                    end
                end
                if (t_last) begin
                    wb_valid <= 1'b1;
                    wb_final <= t_final;
                    wb_addr  <= t_out;
                end
            end

            case (state)
                S_IDLE: begin
                    if (start) begin
                        mode_q     <= mode;
                        h_q        <= h_in;
                        w_q        <= w_in;
                        cw_q       <= c_words;
                        src_sel_q  <= src_sel;
                        dst_sel_q  <= dst_sel;
                        src_base_q <= src_base;
                        dst_base_q <= dst_base;
                        host_q     <= host_mode;
                        err_code   <= '0;
                        row_c      <= '0;
                        hw_c       <= '0;
                        opx_c      <= '0;
                        inw_c      <= '0;
                        outw_c     <= '0;
                        mi         <= '0;
                        state      <= S_MUL1;
                    end
                end

                // round 1: w*cw, h*w and (h/2)*(w/2), one multiplier bit per step
                S_MUL1: begin
                    if (w_q[mi]) row_c <= row_c + (CHK_W'(cw_q) << mi);
                    if (h_q[mi]) hw_c  <= hw_c  + (CHK_W'(w_q)  << mi);
                    if (h2[mi])  opx_c <= opx_c + (CHK_W'(w2)   << mi);
                    if (int'(mi) == DIM_W - 1) begin mi <= '0; state <= S_MUL2; end
                    else mi <= mi + 1'b1;
                end

                // round 2: (h*w)*cw and (h/2*w/2)*cw
                S_MUL2: begin
                    if (cw_q[mi]) begin
                        inw_c  <= inw_c  + (hw_c  << mi);
                        outw_c <= outw_c + (opx_c << mi);
                    end
                    if (int'(mi) == CW_W - 1) state <= S_DECIDE;
                    else mi <= mi + 1'b1;
                end

                S_DECIDE: begin
                    if (e_dim || e_mode || e_shape || e_src || e_dst || e_sel || host_q) begin
                        err      <= 1'b1;
                        err_code <= {e_mode, host_q, e_sel, e_dst, e_src, 3'b000, e_shape, e_dim};
                        state    <= S_IDLE;
                    end else begin
                        shift_q <= s_c;
                        row_q   <= ADDR_WIDTH'(row_c);
                        hw_q    <= HW_W'(hw_c);
                        oy      <= '0;
                        ox      <= '0;
                        g       <= '0;
                        tap     <= '0;
                        row_ptr <= src_base_q;
                        px_ptr  <= src_base_q;
                        out_ptr <= dst_base_q;
                        p       <= '0;
                        g_ptr   <= src_base_q;
                        rd_ptr  <= src_base_q;
                        state   <= S_RUN;
                    end
                end

                S_RUN: begin
                    if (mode_q == M_MAX2) begin
                        if (tap != 2'd3) begin
                            tap <= tap + 1'b1;
                        end else begin
                            tap     <= '0;
                            out_ptr <= out_ptr + 1'b1;
                            if (g != cw_q - 1'b1) begin
                                g <= g + 1'b1;
                            end else begin
                                g <= '0;
                                if (ox != w2 - 1'b1) begin
                                    ox     <= ox + 1'b1;
                                    px_ptr <= px_ptr + ADDR_WIDTH'({cw_q, 1'b0});
                                end else begin
                                    ox      <= '0;
                                    oy      <= oy + 1'b1;
                                    // next output row starts two input rows on
                                    row_ptr <= row_ptr + (row_q << 1);
                                    px_ptr  <= row_ptr + (row_q << 1);
                                end
                            end
                        end
                    end else begin
                        if (p != hw_q - 1'b1) begin
                            p      <= p + 1'b1;
                            rd_ptr <= rd_ptr + ADDR_WIDTH'(cw_q);
                        end else begin
                            p      <= '0;
                            g      <= g + 1'b1;
                            g_ptr  <= g_ptr + 1'b1;
                            rd_ptr <= g_ptr + 1'b1;
                        end
                    end
                    if (iss_final) state <= S_DRAIN;
                end

                // the reads are all issued; wait for the last word to be
                // written. done rises on the edge where it lands.
                S_DRAIN: begin
                    if (wb_valid && wb_final) begin
                        done  <= 1'b1;
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (!reset && (state == S_RUN || state == S_DRAIN) && host_mode)
            $fatal(1, "pool_layer: host_mode raised while the layer runs");
    end
`endif

endmodule
