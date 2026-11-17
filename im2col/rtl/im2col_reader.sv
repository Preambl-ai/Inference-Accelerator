// im2col_reader: A operand feeder for gemm_tile. turns one tile of a
// convolution into its K operand slices by reading the activation buffer
// directly; no im2col matrix is ever stored.
//
// this is the reference version: one slice buffer. it gathers N words, hands
// out N slices, and only then fetches the next N words, so the tile stalls on
// it (gemm_tile's stall_feed) once per group. the fast path prefetches.
//
// activation layout: one word is N channels of one pixel, byte j = channel
//   g*N + j. the word for pixel (y, x), channel group g is at
//   src_base + (y*w_in + x)*cin_words + g. channel counts are padded up to a
//   multiple of N by the host.
//
// a tile is N output pixels, r = 0 .. N-1, starting at (y0, x0):
//   win_order = 0 : raster order. x steps by 1 and wraps at w_out to the next
//                   row.
//   win_order = 1 : 2x2 pooling windows, 4 pixels each in the order
//                   (y,x) (y,x+1) (y+1,x) (y+1,x+1); windows step by 2 in x
//                   and wrap at w_out to y+2. needs N % 4 == 0, w_out even
//                   and (y0, x0) even.
//   pixels past the end of the output are still walked; whatever they read is
//   dropped downstream.
//
// slice order, k = ((ky*kw + kx)*cin_words + g)*N + j. slice k holds, for each
//   pixel r, channel g*N + j of input pixel
//   (y_r*stride + ky - pad, x_r*stride + kx - pad), or pad_value where that
//   falls outside the input. pad_value is the layer's input zero point; a
//   padded pixel is never read from memory. a tile has K = kh*kw*cin_words*N
//   slices, which is what gemm_tile must be given as k_in.
//
// interface
//   start/ready      : ready is high when idle. the command inputs are latched
//                      on start & ready and need not be held. start while busy
//                      is ignored.
//   rd_en/rd_addr    : activation buffer read port. rd_en is the buffer's read
//                      clock enable and is held high for the whole tile, so a
//                      word in flight always arrives. rd_data is expected
//                      READ_LATENCY edges after rd_addr.
//   a_valid/a_data   : the next slice, from registers. byte r belongs to pixel r.
//   a_take           : that slice is consumed on this edge. only legal with
//                      a_valid high.
//
// there is no en/stall input: this block is a stall source for the tile, not
// part of what stalls. it runs ahead on its own and waits in a_valid.
//
// latency, measured in simulation (L = READ_LATENCY): the first slice is valid
//   N + L + 2 edges after the one that accepts start. after the last slice of
//   a group is taken the next group is valid N + L + 2 edges later, so a group
//   costs 2N + L + 2 cycles when slices are taken as soon as they are valid.
//
// reset is synchronous. it drops the tile in progress and clears a_valid.

module im2col_reader #(
    parameter int N            = 8,
    parameter int DATA_WIDTH   = 8,
    parameter int ADDR_W       = 12,
    parameter int READ_LATENCY = 1,
    parameter int DIM_W        = 8,    // image height and width
    parameter int CW_W         = 7,    // channel words per pixel
    parameter int ROW_W        = (N > 1) ? $clog2(N) : 1
)(
    input  logic clk,
    input  logic reset,    // synchronous

    // tile command
    input  logic start,
    output logic ready,
    input  logic [DIM_W-1:0] y0,
    input  logic [DIM_W-1:0] x0,
    input  logic win_order,
    input  logic [DIM_W-1:0] h_in,
    input  logic [DIM_W-1:0] w_in,
    input  logic [DIM_W-1:0] w_out,
    input  logic [CW_W-1:0] cin_words,
    input  logic [2:0] kh,
    input  logic [2:0] kw,
    input  logic [1:0] stride,
    input  logic [1:0] pad,
    input  logic signed [DATA_WIDTH-1:0] pad_value,
    input  logic [ADDR_W-1:0] src_base,

    // activation buffer read port
    output logic rd_en,
    output logic [ADDR_W-1:0] rd_addr,
    input  logic [N*DATA_WIDTH-1:0] rd_data,

    // A operand port of gemm_tile
    output logic a_valid,
    output logic [N*DATA_WIDTH-1:0] a_data,
    input  logic a_take
);

    localparam int COORD_W = DIM_W + 4;     // signed input coordinate
    localparam int LIN_W   = 2*DIM_W;       // y*w_in + x
    localparam int WORD_W  = LIN_W + CW_W;  // (y*w_in + x)*cin_words
    localparam int TAG_W   = ROW_W + 2;     // {valid, pad, r}

    initial begin
        if (N < 1)
            $fatal(1, "im2col_reader: N must be >= 1 (got %0d)", N);
        if (READ_LATENCY < 1)
            $fatal(1, "im2col_reader: READ_LATENCY must be >= 1 (got %0d)", READ_LATENCY);
    end

    localparam logic [1:0] S_IDLE  = 2'd0,
                           S_WALK  = 2'd1,   // one pixel address per cycle
                           S_WAIT  = 2'd2,   // words still in flight
                           S_SERVE = 2'd3;   // slices on a_data
    logic [1:0] state;

    // latched command
    logic [DIM_W-1:0] y0_q, x0_q, h_q, w_q, wo_q;
    logic win_q;
    logic [CW_W-1:0] cw_q;
    logic [2:0] kh_q, kw_q;
    logic [1:0] stride_q, pad_q;
    logic signed [DATA_WIDTH-1:0] padv_q;
    logic [ADDR_W-1:0] base_q;

    // position in the tile: tap (ky, kx), channel group g, pixel r
    logic [2:0] ky, kx;
    logic [CW_W-1:0] g;
    logic [ROW_W-1:0] r;
    logic [DIM_W-1:0] wy, wx;      // pixel, or window origin when win_q
    logic [ROW_W-1:0] j;           // slices served from the current group

    // current output pixel and the input pixel this tap needs from it
    logic [DIM_W:0] cur_y, cur_x;
    logic signed [COORD_W-1:0] y_in, x_in;
    logic in_pad;
    logic step_px;
    logic [DIM_W:0] wx_next;

    assign cur_y = wy + DIM_W'(win_q & r[ROW_W > 1 ? 1 : 0] & (ROW_W > 1));
    assign cur_x = wx + DIM_W'(win_q & r[0]);
    // operands are widened before the multiply: a cast around the product
    // would be applied after it had already been truncated
    assign y_in  = $signed(COORD_W'(cur_y) * COORD_W'(stride_q)) + $signed(COORD_W'(ky)) - $signed(COORD_W'(pad_q));
    assign x_in  = $signed(COORD_W'(cur_x) * COORD_W'(stride_q)) + $signed(COORD_W'(kx)) - $signed(COORD_W'(pad_q));
    assign in_pad = (y_in < 0) || (x_in < 0) ||
                    (y_in >= $signed(COORD_W'(h_q))) || (x_in >= $signed(COORD_W'(w_q)));

    // raster: every pixel moves on. windows: move on after the 4th pixel.
    assign step_px = win_q ? (r[0] & r[ROW_W > 1 ? 1 : 0]) : 1'b1;
    assign wx_next = wx + (win_q ? 2'd2 : 2'd1);

    // address pipeline: two stages, so each has one multiply
    logic s1_valid, s1_pad;
    logic [ROW_W-1:0] s1_r;
    logic [LIN_W-1:0] s1_lin;
    logic s2_valid, s2_pad;
    logic [ROW_W-1:0] s2_r;

    // the tag of a word travels beside the memory and meets it at rd_data
    logic arr_valid, arr_pad;
    logic [ROW_W-1:0] arr_r;

    delay #(.DATA_WIDTH(TAG_W), .DEPTH(READ_LATENCY)) u_tag_dly (
        .clk      (clk),
        .reset    (reset),
        .en       (1'b1),
        .in_data  ({s2_valid, s2_pad, s2_r}),
        .out_data ({arr_valid, arr_pad, arr_r})
    );

    // slice buffer: N words, one per pixel. the low byte of each is the slice
    // on offer; a take shifts every word down one channel.
    logic [N*DATA_WIDTH-1:0] word [0:N-1];

    genvar gi;
    generate
        for (gi = 0; gi < N; gi++) begin : g_a_data
            assign a_data[gi*DATA_WIDTH +: DATA_WIDTH] = word[gi][DATA_WIDTH-1:0];
        end
    endgenerate

    assign ready = (state == S_IDLE);
    assign rd_en = (state != S_IDLE);

    logic last_g, last_kx, last_ky;
    assign last_g  = (g  == cw_q - 1'b1);
    assign last_kx = (kx == kw_q - 1'b1);
    assign last_ky = (ky == kh_q - 1'b1);

    always_ff @(posedge clk) begin
        if (reset) begin
            state    <= S_IDLE;
            y0_q     <= '0;
            x0_q     <= '0;
            h_q      <= '0;
            w_q      <= '0;
            wo_q     <= '0;
            win_q    <= 1'b0;
            cw_q     <= '0;
            kh_q     <= '0;
            kw_q     <= '0;
            stride_q <= '0;
            pad_q    <= '0;
            padv_q   <= '0;
            base_q   <= '0;
            ky       <= '0;
            kx       <= '0;
            g        <= '0;
            r        <= '0;
            wy       <= '0;
            wx       <= '0;
            j        <= '0;
            s1_valid <= 1'b0;
            s1_pad   <= 1'b0;
            s1_r     <= '0;
            s1_lin   <= '0;
            s2_valid <= 1'b0;
            s2_pad   <= 1'b0;
            s2_r     <= '0;
            rd_addr  <= '0;
            a_valid  <= 1'b0;
            for (int i = 0; i < N; i++) word[i] <= '0;
        end else begin
            // stage 1: which input pixel, and is it padding
            s1_valid <= (state == S_WALK);
            s1_pad   <= in_pad;
            s1_r     <= r;
            s1_lin   <= LIN_W'(y_in) * LIN_W'(w_q) + LIN_W'(x_in);

            // stage 2: its address. a padded pixel still occupies its slot in
            // the pipeline, the address is just not used.
            s2_valid <= s1_valid;
            s2_pad   <= s1_pad;
            s2_r     <= s1_r;
            rd_addr  <= base_q + ADDR_W'(WORD_W'(s1_lin) * WORD_W'(cw_q)) + ADDR_W'(g);

            // arrival
            if (arr_valid)
                word[arr_r] <= arr_pad ? {N{padv_q}} : rd_data;

            case (state)

                S_IDLE: begin
                    if (start) begin
                        y0_q     <= y0;
                        x0_q     <= x0;
                        h_q      <= h_in;
                        w_q      <= w_in;
                        wo_q     <= w_out;
                        win_q    <= win_order;
                        cw_q     <= cin_words;
                        kh_q     <= kh;
                        kw_q     <= kw;
                        stride_q <= stride;
                        pad_q    <= pad;
                        padv_q   <= pad_value;
                        base_q   <= src_base;
                        ky       <= '0;
                        kx       <= '0;
                        g        <= '0;
                        r        <= '0;
                        wy       <= y0;
                        wx       <= x0;
                        state    <= S_WALK;
                    end
                end

                S_WALK: begin
                    if (step_px) begin
                        if (wx_next >= {1'b0, wo_q}) begin
                            wx <= '0;
                            wy <= wy + (win_q ? 2'd2 : 2'd1);
                        end else begin
                            wx <= wx_next[DIM_W-1:0];
                        end
                    end
                    if (r == N - 1) state <= S_WAIT;
                    else            r <= r + 1'b1;
                end

                S_WAIT: begin
                    // the last word lands on the edge that raises a_valid
                    if (arr_valid && (arr_r == N - 1)) begin
                        a_valid <= 1'b1;
                        j       <= '0;
                        state   <= S_SERVE;
                    end
                end

                S_SERVE: begin
                    if (a_take) begin
                        for (int i = 0; i < N; i++)
                            word[i] <= word[i] >> DATA_WIDTH;
                        if (j == N - 1) begin
                            a_valid <= 1'b0;
                            // every group walks the same N pixels again
                            r  <= '0;
                            wy <= y0_q;
                            wx <= x0_q;
                            if (!last_g) begin
                                g     <= g + 1'b1;
                                state <= S_WALK;
                            end else if (!last_kx) begin
                                g     <= '0;
                                kx    <= kx + 1'b1;
                                state <= S_WALK;
                            end else if (!last_ky) begin
                                g     <= '0;
                                kx    <= '0;
                                ky    <= ky + 1'b1;
                                state <= S_WALK;
                            end else begin
                                state <= S_IDLE;
                            end
                        end else begin
                            j <= j + 1'b1;
                        end
                    end
                end

                default: state <= S_IDLE;

            endcase
        end
    end

    // Simulation-only checks
`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (!reset && (state == S_IDLE) && start) begin
            if (kh == 0 || kw == 0 || stride == 0 || cin_words == 0)
                $fatal(1, "im2col_reader: kh, kw, stride and cin_words must be >= 1");
            if (win_order && ((N % 4) != 0 || w_out[0] || y0[0] || x0[0]))
                $fatal(1, "im2col_reader: win_order needs N %% 4 == 0 and even w_out, y0, x0");
        end
        if (!reset && a_take && !a_valid)
            $fatal(1, "im2col_reader: a_take without a_valid");
    end
`endif

endmodule
