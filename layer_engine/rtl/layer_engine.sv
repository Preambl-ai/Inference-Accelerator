// layer_engine: the act_buffer plus every engine that computes from it,
// with exactly one owner of the buffer's ports at a time.
// contract: layer_engine/layer_engine_contract.md.
//
//   op 1 CONV -> conv_core      op 2 MAX2, op 3 GAVG -> pool_layer
//
// one start and one command bus: the op picks the engine, which latches the
// fields it uses. two engines can therefore never be started together; the
// only overlap left (two engines busy at once) is a simulation $fatal no
// legal input reaches.
//
// owner = conv_busy ? CONV : pool_busy ? POOL : HOST. the read port follows
// the owner (host_rd_* when nobody is busy, as on conv_layer); the datapath
// write port follows it too and is held off when nobody is busy.
//
// no register stage is added: every engine's latency is its own.

module layer_engine #(
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
    parameter int P_WIDTH      = 5 + 16 + 32
)(
    input  logic clk,
    input  logic reset,

    // command
    input  logic start,
    input  logic [2:0] op,
    output logic ready,
    output logic busy,
    output logic done,
    output logic err,
    output logic [9:0] err_code,
    output logic [2:0] done_op,
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

    // status
    output logic conv_busy,
    output logic pool_busy,
    output logic tile_launch,

    // B and P write ports, idle only
    input  logic b_we,
    input  logic [B_ADDR_W-1:0] b_waddr,
    input  logic [N*DATA_WIDTH-1:0] b_wdata,
    input  logic p_we,
    input  logic [P_ADDR_W-1:0] p_waddr,
    input  logic [P_WIDTH-1:0] p_wdata,

    // act_buffer host ports, as on conv_layer
    input  logic host_mode,
    input  logic host_we,
    input  logic [ADDR_WIDTH-1:0] host_waddr,
    input  logic [N*DATA_WIDTH-1:0] host_wdata,
    input  logic [SEL_WIDTH-1:0] host_wr_sel,
    input  logic host_rd_en,
    input  logic [ADDR_WIDTH-1:0] host_rd_addr,
    input  logic [SEL_WIDTH-1:0] host_rd_sel,
    output logic [N*DATA_WIDTH-1:0] host_rd_data
);

    localparam int WORD_W = N * DATA_WIDTH;
    localparam logic [2:0] OP_CONV = 3'd1, OP_MAX2 = 3'd2, OP_GAVG = 3'd3;

    // ------------------------------------------------------------------
    // dispatch
    // ------------------------------------------------------------------
    logic go, op_ok, bad_q;
    logic [2:0] op_q;                       // op of the command in flight
    assign op_ok = (op == OP_CONV) || (op == OP_MAX2) || (op == OP_GAVG);
    assign busy  = conv_busy || pool_busy || bad_q;
    assign ready = !busy;
    assign go    = start && ready;

    logic conv_done, conv_err, pool_done, pool_err;
    logic [8:0] conv_code;
    logic [9:0] pool_code;

    // a bad op is rejected like any other illegal command: busy for the
    // accepting edge, err with E_MODE on the next. the engines' own pulses
    // and their held err_codes pass straight through, so the engine adds no
    // edge to any latency.
    logic bad_err_q, bad_code_q;
    always_ff @(posedge clk) begin
        if (reset) begin
            bad_q      <= 1'b0;
            bad_err_q  <= 1'b0;
            bad_code_q <= 1'b0;
            op_q       <= '0;
        end else begin
            bad_err_q <= 1'b0;
            if (go) begin
                op_q       <= op;
                bad_q      <= !op_ok;
                bad_code_q <= 1'b0;
            end
            if (bad_q) begin
                bad_q      <= 1'b0;
                bad_err_q  <= 1'b1;
                bad_code_q <= 1'b1;
            end
        end
    end

    assign done    = conv_done || pool_done;
    assign err     = conv_err || pool_err || bad_err_q;
    assign done_op = op_q;
    // each engine holds its code until its next start; show the code of the
    // engine the last command went to
    always_comb begin
        if (bad_code_q)            err_code = 10'b10_0000_0000;     // E_MODE
        else if (op_q == OP_CONV)  err_code = {1'b0, conv_code};
        else if (op_q == OP_MAX2 || op_q == OP_GAVG) err_code = pool_code;
        else                       err_code = '0;
    end

    // ------------------------------------------------------------------
    // engines
    // ------------------------------------------------------------------
    logic [WORD_W-1:0] act_rdata;
    logic c_rd_en, c_we, p_rd_en, p_we_a;
    logic [ADDR_WIDTH-1:0] c_rd_addr, c_waddr, p_rd_addr, p_waddr_a;
    logic [SEL_WIDTH-1:0] c_rd_sel, c_wr_sel, p_rd_sel, p_wr_sel;
    logic [WORD_W-1:0] c_wdata, p_wdata_a;
    logic conv_ready, pool_ready;

    conv_core #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .READ_LATENCY(READ_LATENCY),
        .NBUF(NBUF), .K_MAX(K_MAX), .K_WIDTH(K_WIDTH), .B_ADDR_W(B_ADDR_W),
        .P_ADDR_W(P_ADDR_W), .DIM_W(DIM_W), .CW_W(CW_W)
    ) u_conv (
        .clk(clk), .reset(reset),
        .start(go && op == OP_CONV), .ready(conv_ready), .busy(conv_busy),
        .done(conv_done), .err(conv_err), .err_code(conv_code),
        .h_in(h_in), .w_in(w_in), .h_out(h_out), .w_out(w_out),
        .cin_words(cin_words), .cout_words(cout_words), .kh(kh), .kw(kw),
        .stride(stride), .pad(pad), .zp_in(zp_in), .zp_out(zp_out), .relu_en(relu_en),
        .src_sel(src_sel), .dst_sel(dst_sel), .src_base(src_base), .dst_base(dst_base),
        .b_base(b_base), .p_base(p_base),
        .b_we(b_we), .b_waddr(b_waddr), .b_wdata(b_wdata),
        .p_we(p_we), .p_waddr(p_waddr), .p_wdata(p_wdata),
        .host_mode(host_mode),
        .act_rd_en(c_rd_en), .act_rd_addr(c_rd_addr), .act_rd_sel(c_rd_sel),
        .act_rd_data(act_rdata),
        .act_we(c_we), .act_waddr(c_waddr), .act_wdata(c_wdata), .act_wr_sel(c_wr_sel),
        .tile_launch(tile_launch)
    );

    pool_layer #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .READ_LATENCY(READ_LATENCY),
        .NBUF(NBUF), .DIM_W(DIM_W), .CW_W(CW_W)
    ) u_pool (
        .clk(clk), .reset(reset),
        .start(go && (op == OP_MAX2 || op == OP_GAVG)), .ready(pool_ready), .busy(pool_busy),
        .done(pool_done), .err(pool_err), .err_code(pool_code),
        .mode((op == OP_MAX2) ? 2'd1 : 2'd2),
        .h_in(h_in), .w_in(w_in), .c_words(cin_words),
        .src_sel(src_sel), .dst_sel(dst_sel), .src_base(src_base), .dst_base(dst_base),
        .host_mode(host_mode),
        .rd_en(p_rd_en), .rd_addr(p_rd_addr), .rd_sel(p_rd_sel), .rd_data(act_rdata),
        .we(p_we_a), .waddr(p_waddr_a), .wdata(p_wdata_a), .wr_sel(p_wr_sel)
    );

    // ------------------------------------------------------------------
    // the act_buffer and its owner mux
    // ------------------------------------------------------------------
    logic a_rd_en, a_we;
    logic [ADDR_WIDTH-1:0] a_rd_addr, a_waddr;
    logic [SEL_WIDTH-1:0] a_rd_sel, a_wr_sel;
    logic [WORD_W-1:0] a_wdata;

    always_comb begin
        if (conv_busy) begin
            a_rd_en = c_rd_en;  a_rd_addr = c_rd_addr;  a_rd_sel = c_rd_sel;
            a_we    = c_we;     a_waddr   = c_waddr;    a_wdata  = c_wdata;   a_wr_sel = c_wr_sel;
        end else if (pool_busy) begin
            a_rd_en = p_rd_en;  a_rd_addr = p_rd_addr;  a_rd_sel = p_rd_sel;
            a_we    = p_we_a;   a_waddr   = p_waddr_a;  a_wdata  = p_wdata_a; a_wr_sel = p_wr_sel;
        end else begin
            a_rd_en = host_rd_en; a_rd_addr = host_rd_addr; a_rd_sel = host_rd_sel;
            a_we    = 1'b0;       a_waddr   = '0;           a_wdata  = '0;       a_wr_sel = '0;
        end
    end
    assign host_rd_data = act_rdata;

    act_buffer #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .ADDR_WIDTH(ADDR_WIDTH), .READ_LATENCY(READ_LATENCY),
        .NBUF(NBUF)
    ) u_act (
        .clk(clk),
        .rd_en(a_rd_en), .rd_addr(a_rd_addr), .rd_sel(a_rd_sel), .rd_data(act_rdata),
        .we(a_we), .waddr(a_waddr), .wdata(a_wdata), .wr_sel(a_wr_sel),
        .host_mode(host_mode), .host_we(host_we), .host_waddr(host_waddr),
        .host_wdata(host_wdata), .host_wr_sel(host_wr_sel)
    );

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (!reset) begin
            if (conv_busy && pool_busy)
                $fatal(1, "layer_engine: conv and pool busy in the same cycle");
            if ((c_we && !conv_busy) || (p_we_a && !pool_busy))
                $fatal(1, "layer_engine: an engine that does not own the port raised its write enable");
        end
    end
`endif

endmodule
