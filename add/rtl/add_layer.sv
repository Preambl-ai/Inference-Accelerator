// add_layer: residual add of two INT8 tensors with their own zero points and
// scales, out of on-chip memory. contract: add/add_contract.md.
//
//   out = sat8(round(((a - za)*ma + (b - zb)*mb) / 2^shift) + zo), optional ReLU
//
// no multipliers: ma and mb are applied bit-serially, one bit of each per
// cycle, as shift-and-add into a per-lane accumulator (M_BITS cycles a word).
// sizes in the check phase are bit-serial too.
//
// one act_buffer read port: for word i, a[i] is read, then b[i] on the next
// cycle (rd_en held high the whole time); a tag {valid, is_b} rides a
// READ_LATENCY-deep delay to meet each word. the result goes to a write
// register and lands one edge later. words are serialized in this reference
// version.

module add_layer #(
    parameter int N            = 8,
    parameter int DATA_WIDTH   = 8,
    parameter int ADDR_WIDTH   = 12,
    parameter int READ_LATENCY = 1,
    parameter int NBUF         = 3,
    parameter int DIM_W        = 8,
    parameter int CW_W         = 7,
    parameter int M_BITS       = 8,
    // derived, do not override
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
    input  logic signed [DATA_WIDTH-1:0] za,
    input  logic signed [DATA_WIDTH-1:0] zb,
    input  logic signed [DATA_WIDTH-1:0] zo,
    input  logic [M_BITS-1:0] ma,
    input  logic [M_BITS-1:0] mb,
    input  logic [3:0] shift,
    input  logic relu_en,
    input  logic host_mode,

    output logic rd_en,
    output logic [ADDR_WIDTH-1:0] rd_addr,
    output logic [SEL_WIDTH-1:0] rd_sel,
    input  logic [N*DATA_WIDTH-1:0] rd_data,

    output logic we,
    output logic [ADDR_WIDTH-1:0] waddr,
    output logic [N*DATA_WIDTH-1:0] wdata,
    output logic [SEL_WIDTH-1:0] wr_sel
);

    localparam int WORD_W = N * DATA_WIDTH;
    localparam int R_W    = DATA_WIDTH + 1;                // a - za
    localparam int ACC_W  = M_BITS + R_W + 2;              // 2 * |ra| * (2^M - 1) fits
    localparam int CHK_W  = ((2*DIM_W + CW_W > ADDR_WIDTH) ? 2*DIM_W + CW_W : ADDR_WIDTH) + 2;
    localparam int CNT_W  = 2*DIM_W + CW_W;

    initial begin
        if (DATA_WIDTH != 8)
            $fatal(1, "add_layer: DATA_WIDTH must be 8 (got %0d)", DATA_WIDTH);
        if (NBUF < 2)
            $fatal(1, "add_layer: NBUF must be >= 2 (got %0d)", NBUF);
        if (M_BITS < 1 || M_BITS > 16)
            $fatal(1, "add_layer: M_BITS must be 1..16 (got %0d)", M_BITS);
        if (SEL_WIDTH != ((NBUF > 1) ? $clog2(NBUF) : 1))
            $fatal(1, "add_layer: SEL_WIDTH is derived, do not override it");
    end

    typedef enum logic [3:0] {
        S_IDLE, S_MUL1, S_MUL2, S_DECIDE, S_RDA, S_RDB, S_WAIT, S_STEP, S_OUT, S_DRAIN
    } state_t;
    state_t state;

    assign ready = (state == S_IDLE);
    assign busy  = !ready;

    // latched command
    logic [DIM_W-1:0] h_q, w_q;
    logic [CW_W-1:0] cw_q;
    logic [SEL_WIDTH-1:0] sa_q, sb_q, sd_q;
    logic [ADDR_WIDTH-1:0] a_base_q, b_base_q, d_base_q;
    logic signed [DATA_WIDTH-1:0] za_q, zb_q, zo_q;
    logic [M_BITS-1:0] ma_q, mb_q;
    logic [3:0] shift_q;
    logic relu_q, host_q;

    // ------------------------------------------------------------------
    // check phase: words = h * w * c_words, bit-serial
    // ------------------------------------------------------------------
    logic [$clog2(DIM_W + CW_W + 1)-1:0] mi;
    logic [CHK_W-1:0] hw_c, n_c;
    localparam logic [CHK_W-1:0] ONE = 1;
    logic e_dim, e_src, e_dst, e_sel;
    assign e_dim = (h_q == 0) || (w_q == 0) || (cw_q == 0);
    assign e_src = (CHK_W'(a_base_q) + n_c > (ONE << ADDR_WIDTH)) ||
                   (CHK_W'(b_base_q) + n_c > (ONE << ADDR_WIDTH));
    assign e_dst = CHK_W'(d_base_q) + n_c > (ONE << ADDR_WIDTH);
    assign e_sel = (sd_q == sa_q) || (sd_q == sb_q) || (int'(sa_q) >= NBUF) ||
                   (int'(sb_q) >= NBUF) || (int'(sd_q) >= NBUF);

    // ------------------------------------------------------------------
    // run
    // ------------------------------------------------------------------
    logic [ADDR_WIDTH-1:0] a_ptr, b_ptr, d_ptr;
    logic [CNT_W-1:0] left;                         // words still to start, this one included
    logic [$clog2(M_BITS + 1)-1:0] bit_i;

    assign rd_en   = (state == S_RDA) || (state == S_RDB) || (state == S_WAIT) ||
                     (state == S_STEP) || (state == S_OUT) || (state == S_DRAIN);
    assign rd_addr = (state == S_RDB) ? b_ptr : a_ptr;
    assign rd_sel  = (state == S_RDB) ? sb_q : sa_q;

    logic t_valid, t_is_b;
    delay #(.DATA_WIDTH(2), .DEPTH(READ_LATENCY)) u_tag (
        .clk(clk), .reset(reset), .en(1'b1),
        .in_data({(state == S_RDA) || (state == S_RDB), (state == S_RDB)}),
        .out_data({t_valid, t_is_b})
    );

    logic [WORD_W-1:0] a_word;
    logic signed [R_W-1:0] ra [0:N-1];
    logic signed [R_W-1:0] rb [0:N-1];
    logic signed [ACC_W-1:0] acc [0:N-1];

    logic wb_valid, wb_final;
    logic [ADDR_WIDTH-1:0] wb_addr;
    logic [WORD_W-1:0] wb_data;
    assign we     = wb_valid && !reset;
    assign waddr  = wb_addr;
    assign wdata  = wb_data;
    assign wr_sel = sd_q;

    // blocking temporaries
    logic signed [ACC_W-1:0] q;
    logic signed [ACC_W:0] v, lo;
    logic signed [DATA_WIDTH-1:0] av, bv;

    always_ff @(posedge clk) begin
        if (reset) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            err      <= 1'b0;
            err_code <= '0;
            wb_valid <= 1'b0;
            wb_final <= 1'b0;
        end else begin
            done     <= 1'b0;
            err      <= 1'b0;
            wb_valid <= 1'b0;

            // the a word arrives first and waits here for b
            if (t_valid && !t_is_b) a_word <= rd_data;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        h_q <= h_in; w_q <= w_in; cw_q <= c_words;
                        sa_q <= src_sel; sb_q <= src2_sel; sd_q <= dst_sel;
                        a_base_q <= src_base; b_base_q <= src2_base; d_base_q <= dst_base;
                        za_q <= za; zb_q <= zb; zo_q <= zo;
                        ma_q <= ma; mb_q <= mb; shift_q <= shift; relu_q <= relu_en;
                        host_q   <= host_mode;
                        err_code <= '0;
                        hw_c     <= '0;
                        n_c      <= '0;
                        mi       <= '0;
                        state    <= S_MUL1;
                    end
                end

                S_MUL1: begin
                    if (h_q[mi]) hw_c <= hw_c + (CHK_W'(w_q) << mi);
                    if (int'(mi) == DIM_W - 1) begin mi <= '0; state <= S_MUL2; end
                    else mi <= mi + 1'b1;
                end

                S_MUL2: begin
                    if (cw_q[mi]) n_c <= n_c + (hw_c << mi);
                    if (int'(mi) == CW_W - 1) state <= S_DECIDE;
                    else mi <= mi + 1'b1;
                end

                S_DECIDE: begin
                    if (e_dim || e_src || e_dst || e_sel || host_q) begin
                        err      <= 1'b1;
                        err_code <= {1'b0, host_q, e_sel, e_dst, e_src, 4'b0000, e_dim};
                        state    <= S_IDLE;
                    end else begin
                        a_ptr <= a_base_q;
                        b_ptr <= b_base_q;
                        d_ptr <= d_base_q;
                        left  <= CNT_W'(n_c);
                        state <= S_RDA;
                    end
                end

                S_RDA: state <= S_RDB;
                S_RDB: state <= S_WAIT;

                // b arrives READ_LATENCY edges after its read: form a - za
                // and b - zb per lane and clear the accumulators
                S_WAIT: begin
                    if (t_valid && t_is_b) begin
                        for (int j = 0; j < N; j++) begin
                            av = a_word[j*DATA_WIDTH +: DATA_WIDTH];
                            bv = rd_data[j*DATA_WIDTH +: DATA_WIDTH];
                            ra[j]  <= R_W'(av) - R_W'(za_q);
                            rb[j]  <= R_W'(bv) - R_W'(zb_q);
                            acc[j] <= '0;
                        end
                        bit_i <= '0;
                        state <= S_STEP;
                    end
                end

                // one bit of ma and one bit of mb per cycle, LSB first
                S_STEP: begin
                    for (int j = 0; j < N; j++)
                        acc[j] <= acc[j]
                                  + (ma_q[bit_i] ? (ACC_W'(ra[j]) <<< bit_i) : ACC_W'(0))
                                  + (mb_q[bit_i] ? (ACC_W'(rb[j]) <<< bit_i) : ACC_W'(0));
                    if (int'(bit_i) == M_BITS - 1) state <= S_OUT;
                    else bit_i <= bit_i + 1'b1;
                end

                // round half up, add zo, clamp (ReLU raises the floor to zo)
                S_OUT: begin
                    for (int j = 0; j < N; j++) begin
                        q  = (shift_q == 0) ? acc[j]
                             : ((acc[j] + $signed(ACC_W'(1) << (shift_q - 1'b1))) >>> shift_q);
                        v  = (ACC_W+1)'(q) + (ACC_W+1)'(zo_q);
                        lo = relu_q ? (ACC_W+1)'(zo_q) : -(ACC_W+1)'(128);
                        if (v < lo)                   v = lo;
                        else if (v > (ACC_W+1)'(127)) v = (ACC_W+1)'(127);
                        wb_data[j*DATA_WIDTH +: DATA_WIDTH] <= v[DATA_WIDTH-1:0];
                    end
                    wb_valid <= 1'b1;
                    wb_addr  <= d_ptr;
                    wb_final <= (left == 1);
                    if (left == 1) begin
                        state <= S_DRAIN;
                    end else begin
                        left  <= left - 1'b1;
                        a_ptr <= a_ptr + 1'b1;
                        b_ptr <= b_ptr + 1'b1;
                        d_ptr <= d_ptr + 1'b1;
                        state <= S_RDA;
                    end
                end

                // the last word lands on this edge
                S_DRAIN: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (!reset && state != S_IDLE && state != S_MUL1 && state != S_MUL2 &&
            state != S_DECIDE && host_mode)
            $fatal(1, "add_layer: host_mode raised while the layer runs");
    end
`endif

endmodule
