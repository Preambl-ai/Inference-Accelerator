// gemm_tile: one N x K by K x N matrix product, requantized, out on AXI-Stream.
//
//   C[r][c] = requant(sum_k A[r][k] * B[k][c] + bias[c])     r, c = 0 .. N-1
//
// ctrl_fsm sequences the tile. the B operands (weights) and the per-channel
// parameters come out of block RAMs, the A operands (activations) come in on
// a port from a feeder, and each drained row goes through N requant instances
// (one per column, so one per output channel) and a skid buffer.
//
// nothing about the layer is fixed here. weights and parameters are written
// into memory at runtime and K, the base addresses, zero point and ReLU come
// with each start.
//
// memory layout, written through the *_we ports (never stalled or reset):
//   B  word b_base + k : row k of B. byte c is B[k][c].
//   P  word p_base + c : {shift, mult, bias} for output channel c.
//   a word must not be written on the same edge it is read (see bram.sv), so
//   load a tile's words before starting it.
//
// A operand port: the feeder supplies the tile's K slices in order, k = 0 up.
//   a_valid : the next slice is on a_data. must come from a register.
//   a_data  : column k of A. byte r is A[r][k].
//   a_take  : the slice is consumed on this edge; show the next one after it.
//   a_valid may be high early or go low between slices. whenever the array
//   needs a slice that is not there, the whole tile stalls (stall_feed).
//
// tile command: start/ready is the ctrl_fsm handshake. on the accepting edge
//   k_in, b_base, p_base, zero_point and relu_en are latched, so they only
//   need to be valid with start. a start with k_in == 0 or k_in > K_MAX is
//   ignored and raises err_k for one cycle. ready returns as soon as the
//   array is free, which is before the last rows have left requant; that is
//   safe because requant carries its parameters down the pipeline with the
//   data.
//
// output stream: one beat per row, rows N-1 down to 0.
//   m_axis_tdata : N requantized bytes, byte c is C[row][c]
//   m_axis_tuser : the row index
//   m_axis_tlast : high on row 0, the last beat of the tile
//
// stall: stall = stall_out | stall_feed. stall_out is the skid buffer's
//   registered "full", so m_axis_tready only reaches the skid buffer's own
//   flops. stall_feed is high when the array is due a slice and a_valid is
//   low; both of those are registers too. stall freezes ctrl_fsm, the array,
//   both memory read paths, the parameter loader and the requants. a requant
//   output is handed to the skid buffer only on an unstalled cycle, so it
//   moves exactly once whichever source raised the stall.
//
// latency with m_axis_tready high and a_valid never late, in edges after the
// one that accepts start (L = READ_LATENCY), measured in simulation:
//   first beat valid on m_axis : L + K + 2N + 5
//   last beat valid on m_axis  : L + K + 3N + 4
//   ready rises again          : L + K + 3N + 1, so a tile takes L + K + 3N + 2
//                                cycles start to start when tiles do not overlap
//
// reset is synchronous and wins over stall. it clears control, the array, the
// requants and the skid buffer, so a tile cut off by reset can end the stream
// without tlast. memory contents survive reset.

module gemm_tile #(
    parameter int N            = 8,
    parameter int DATA_WIDTH   = 8,
    parameter int ACC_WIDTH    = 32,
    parameter int K_MAX        = 1024, // deepest tile the operand buffers hold
    parameter int K_WIDTH      = 11,   // has to hold K_MAX itself
    parameter int READ_LATENCY = 1,    // 1 or 2, also ctrl_fsm's MEM_LATENCY
    parameter int B_ADDR_W     = 13,
    parameter int P_ADDR_W     = 6,
    parameter int BIAS_WIDTH   = 32,
    parameter int SAT_WIDTH    = 32,
    // 16 keeps each requant multiply to a 32x17, two DSP48E1s instead of four
    parameter int M_WIDTH      = 16,
    parameter int SHIFT_WIDTH  = 5,
    parameter int ROW_W        = (N > 1) ? $clog2(N) : 1,
    parameter int P_WIDTH      = SHIFT_WIDTH + M_WIDTH + BIAS_WIDTH
)(
    input  logic clk,
    input  logic reset,    // synchronous

    // tile command
    input  logic start,
    input  logic [K_WIDTH-1:0] k_in,
    input  logic [B_ADDR_W-1:0] b_base,
    input  logic [P_ADDR_W-1:0] p_base,
    input  logic signed [DATA_WIDTH-1:0] zero_point,
    input  logic relu_en,
    output logic ready,
    output logic busy,
    output logic err_k,

    // A operand feeder
    input  logic a_valid,
    input  logic [N*DATA_WIDTH-1:0] a_data,
    output logic a_take,

    // memory write ports
    input  logic b_we,
    input  logic [B_ADDR_W-1:0] b_waddr,
    input  logic [N*DATA_WIDTH-1:0] b_wdata,
    input  logic p_we,
    input  logic [P_ADDR_W-1:0] p_waddr,
    input  logic [P_WIDTH-1:0] p_wdata,

    // result stream
    output logic m_axis_tvalid,
    input  logic m_axis_tready,
    output logic [N*DATA_WIDTH-1:0] m_axis_tdata,
    output logic [ROW_W-1:0] m_axis_tuser,
    output logic m_axis_tlast
);

    localparam int TAG_W  = ROW_W + 1;                // {last, row}
    localparam int BEAT_W = TAG_W + N*DATA_WIDTH;

    initial begin
        if (N < 1)
            $fatal(1, "gemm_tile: N must be >= 1 (got %0d)", N);
        if (READ_LATENCY != 1 && READ_LATENCY != 2)
            $fatal(1, "gemm_tile: READ_LATENCY must be 1 or 2 (got %0d)", READ_LATENCY);
        if (K_MAX < 1 || K_MAX >= (1 << K_WIDTH))
            $fatal(1, "gemm_tile: K_MAX=%0d does not fit K_WIDTH=%0d", K_MAX, K_WIDTH);
        if ((1 << P_ADDR_W) < N)
            $fatal(1, "gemm_tile: parameter memory holds %0d words, a tile needs N=%0d",
                   1 << P_ADDR_W, N);
        if (P_WIDTH != SHIFT_WIDTH + M_WIDTH + BIAS_WIDTH)
            $fatal(1, "gemm_tile: P_WIDTH is derived, do not override it");
    end

    // stall, an OR of sources
    logic stall;
    logic stall_out;     // output skid buffer is full
    logic stall_feed;    // the array is due a slice the feeder has not got
    logic skid_s_ready;
    logic a_need;

    assign stall_out  = ~skid_s_ready;
    assign stall_feed = a_need & ~a_valid;
    assign stall      = stall_out | stall_feed;
    assign a_take     = a_need & ~stall;

    // a tile deeper than the operand buffers would read past its own words.
    // ctrl_fsm never sees that start; it drops k_in == 0 by itself.
    logic k_ok;
    assign k_ok = (k_in <= K_WIDTH'(K_MAX));

    // tile sequencer
    logic accept;
    logic done;
    logic op_req;
    logic [K_WIDTH-1:0] k_idx;
    logic arr_en, arr_first, arr_last, arr_drain_shift;
    logic c_valid;
    logic [ROW_W-1:0] c_row;

    ctrl_fsm #(
        .N           (N),
        .K_WIDTH     (K_WIDTH),
        .MEM_LATENCY (READ_LATENCY)
    ) u_ctrl (
        .clk             (clk),
        .reset           (reset),
        .start           (start & k_ok),
        .k_in            (k_in),
        .ready           (ready),
        .busy            (busy),
        .done            (done),
        .stall           (stall),
        .op_req          (op_req),
        .k_idx           (k_idx),
        .arr_en          (arr_en),
        .arr_first       (arr_first),
        .arr_last        (arr_last),
        .arr_drain_shift (arr_drain_shift),
        .c_valid         (c_valid),
        .c_row           (c_row)
    );

    // same condition ctrl_fsm uses to take the tile, so the latches below can
    // never disagree with it about whether a tile started
    assign accept = start & ready & k_ok & (k_in != '0);

    // a_need marks the cycle the B word for a slice reaches the array, which
    // is the cycle the A slice has to be there too. it is op_req seen through
    // the same delay as the memory, frozen by the same stall.
    delay #(.DATA_WIDTH(1), .DEPTH(READ_LATENCY)) u_need_dly (
        .clk      (clk),
        .reset    (reset),
        .en       (~stall),
        .in_data  (op_req),
        .out_data (a_need)
    );

    always_ff @(posedge clk) begin
        if (reset) err_k <= 1'b0;
        else       err_k <= start & ready & ~(k_ok & (k_in != '0));
    end

    // per-tile command latches and the parameter loader
    logic [B_ADDR_W-1:0] b_base_q;
    logic [P_ADDR_W-1:0] p_base_q;
    logic signed [DATA_WIDTH-1:0] zp_q;
    logic relu_q;

    logic p_req;                     // a parameter word is being requested
    logic [ROW_W-1:0] p_idx;
    logic p_wr;                      // its data is on p_rdata
    logic [ROW_W-1:0] p_wr_idx;
    logic [P_WIDTH-1:0] p_rdata;

    // shadow copy of the tile's N parameter words. the drain needs all N at
    // once and a block RAM gives one word per cycle, so they are fetched one
    // by one while the operands stream; the last lands N + READ_LATENCY edges
    // after the accept and the first row is not sampled before K + 2N + 1.
    logic [N*BIAS_WIDTH-1:0]  p_bias;
    logic [N*M_WIDTH-1:0]     p_mult;
    logic [N*SHIFT_WIDTH-1:0] p_shift;
    logic [N-1:0]             p_loaded;

    delay #(.DATA_WIDTH(ROW_W + 1), .DEPTH(READ_LATENCY)) u_p_dly (
        .clk      (clk),
        .reset    (reset),
        .en       (~stall),
        .in_data  ({p_req, p_idx}),
        .out_data ({p_wr, p_wr_idx})
    );

    always_ff @(posedge clk) begin
        if (reset) begin
            b_base_q <= '0;
            p_base_q <= '0;
            zp_q     <= '0;
            relu_q   <= 1'b0;
            p_req    <= 1'b0;
            p_idx    <= '0;
            p_bias   <= '0;
            p_mult   <= '0;
            p_shift  <= '0;
            p_loaded <= '0;
        end else if (!stall) begin
            if (accept) begin
                b_base_q <= b_base;
                p_base_q <= p_base;
                zp_q     <= zero_point;
                relu_q   <= relu_en;
                p_req    <= 1'b1;
                p_idx    <= '0;
                p_loaded <= '0;
            end else if (p_req) begin
                if (p_idx == N - 1) p_req <= 1'b0;
                else                p_idx <= p_idx + 1'b1;
            end

            if (p_wr) begin
                p_bias  [p_wr_idx*BIAS_WIDTH  +: BIAS_WIDTH]  <= p_rdata[0 +: BIAS_WIDTH];
                p_mult  [p_wr_idx*M_WIDTH     +: M_WIDTH]     <= p_rdata[BIAS_WIDTH +: M_WIDTH];
                p_shift [p_wr_idx*SHIFT_WIDTH +: SHIFT_WIDTH] <= p_rdata[BIAS_WIDTH + M_WIDTH +: SHIFT_WIDTH];
                p_loaded[p_wr_idx] <= 1'b1;
            end
        end
    end

    // memories. the read enable is ~stall so the words in flight hold with
    // the array they are headed for.
    logic [N*DATA_WIDTH-1:0] b_rdata;

    bram #(
        .DATA_WIDTH   (N*DATA_WIDTH),
        .ADDR_WIDTH   (B_ADDR_W),
        .READ_LATENCY (READ_LATENCY)
    ) u_b_mem (
        .clk   (clk),
        .we    (b_we),
        .waddr (b_waddr),
        .wdata (b_wdata),
        .re    (~stall),
        .raddr (b_base_q + B_ADDR_W'(k_idx)),
        .rdata (b_rdata)
    );

    bram #(
        .DATA_WIDTH   (P_WIDTH),
        .ADDR_WIDTH   (P_ADDR_W),
        .READ_LATENCY (READ_LATENCY)
    ) u_p_mem (
        .clk   (clk),
        .we    (p_we),
        .waddr (p_waddr),
        .wdata (p_wdata),
        .re    (~stall),
        .raddr (p_base_q + P_ADDR_W'(p_idx)),
        .rdata (p_rdata)
    );

    // the array
    logic [N*ACC_WIDTH-1:0] c_out;

    systolic_array #(
        .N          (N),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_array (
        .clk         (clk),
        .reset       (reset),
        .en          (arr_en),
        .a_in        (a_data),
        .b_in        (b_rdata),
        .first_in    (arr_first),
        .last_in     (arr_last),
        .drain_shift (arr_drain_shift),
        .c_out       (c_out)
    );

    // epilogue: one requant per column. every instance sees the same valid and
    // tag, so only column 0's copies are used.
    logic [N-1:0] rq_valid;
    logic [N*DATA_WIDTH-1:0] rq_q;
    logic [TAG_W-1:0] rq_tag [0:N-1];

    genvar c;
    generate
        for (c = 0; c < N; c++) begin : g_requant
            requant #(
                .ACC_WIDTH   (ACC_WIDTH),
                .BIAS_WIDTH  (BIAS_WIDTH),
                .SAT_WIDTH   (SAT_WIDTH),
                .M_WIDTH     (M_WIDTH),
                .SHIFT_WIDTH (SHIFT_WIDTH),
                .DATA_WIDTH  (DATA_WIDTH),
                .TAG_WIDTH   (TAG_W)
            ) u_requant (
                .clk        (clk),
                .reset      (reset),
                .en         (~stall),
                .in_valid   (c_valid),
                .in_acc     (c_out[c*ACC_WIDTH +: ACC_WIDTH]),
                .in_tag     ({c_row == '0, c_row}),
                .bias       (p_bias [c*BIAS_WIDTH  +: BIAS_WIDTH]),
                .mult       (p_mult [c*M_WIDTH     +: M_WIDTH]),
                .shift      (p_shift[c*SHIFT_WIDTH +: SHIFT_WIDTH]),
                .zero_point (zp_q),
                .relu_en    (relu_q),
                .out_valid  (rq_valid[c]),
                .out_q      (rq_q[c*DATA_WIDTH +: DATA_WIDTH]),
                .out_tag    (rq_tag[c])
            );
        end
    endgenerate

    // output register slice. the ~stall on s_valid matters once stall_feed is
    // real: a requant output frozen by another stall source would otherwise be
    // taken again on every cycle the skid buffer still has room.
    skid_buffer #(
        .DATA_WIDTH (BEAT_W)
    ) u_skid (
        .clk     (clk),
        .reset   (reset),
        .s_valid (rq_valid[0] & ~stall),
        .s_ready (skid_s_ready),
        .s_data  ({rq_tag[0], rq_q}),
        .m_valid (m_axis_tvalid),
        .m_ready (m_axis_tready),
        .m_data  ({m_axis_tlast, m_axis_tuser, m_axis_tdata})
    );

    // Simulation-only checks
`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (!reset && !stall && c_valid && !(&p_loaded))
            $fatal(1, "gemm_tile: a row reached requant before all parameters were loaded");
        // +gemm_tile_allow_bad_k lets the testbench prove such a start is ignored
        if (!reset && start && ready && !k_ok && !$test$plusargs("gemm_tile_allow_bad_k"))
            $fatal(1, "gemm_tile: k_in=%0d exceeds K_MAX=%0d", k_in, K_MAX);
    end
`endif

endmodule
