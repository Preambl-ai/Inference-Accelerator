// csr: AXI4-Lite slave in front of conv_layer. register map, memory windows
// and access rules: top/host_interface_contract.md sections 2 to 4.
//
// write channel: AW and W are taken independently (either order, or the same
// cycle) into holding registers; the write is performed on the cycle both
// are held, and B is offered the cycle after. AWREADY/WREADY stay low while a
// response is waiting, so a master that holds off BREADY simply stalls the
// next write instead of losing a response.
//
// read channel: one read at a time. a register read answers the cycle after
// AR is accepted. an ACT window read holds conv_layer's host_rd_en high with
// a fixed address for READ_LATENCY edges (the act_buffer read pipeline only
// moves while rd_en is high, so after those edges it holds exactly this
// word), captures host_rd_data on the next edge and raises RVALID with it.
// the master just waits; read latency never shows outside the handshake.
//
// every refused access answers SLVERR and changes nothing; unmapped
// addresses answer DECERR. while conv_layer is busy only register reads,
// STATUS W1C and CTRL.soft_reset are accepted, so conv_layer's own
// "memory touched while running" $fatals are unreachable from the bus.

module csr #(
    parameter int N            = 8,
    parameter int ADDR_WIDTH   = 12,
    parameter int READ_LATENCY = 1,
    parameter int NBUF         = 2,
    parameter int B_ADDR_W     = 13,
    parameter int P_ADDR_W     = 6,
    parameter int K_MAX        = 1024,
    parameter int DIM_W        = 8,
    parameter int CW_W         = 7,
    parameter logic [31:0] BUILD_ID = 32'h0,
    parameter int AXI_ADDR_W   = 24,
    parameter int M_BITS       = 8,        // add rescale width, reported in PARAMS2
    parameter int DESC_DEPTH   = 32,       // sequencer descriptors
    parameter int HAS_DMA      = 0,        // PARAMS2[16] (phase 4)
    // derived, do not override
    parameter int SEL_WIDTH    = (NBUF > 1) ? $clog2(NBUF) : 1,
    parameter int P_WIDTH      = 53,
    parameter int IDX_W        = (DESC_DEPTH > 1) ? $clog2(DESC_DEPTH) : 1
)(
    input  logic clk,
    input  logic reset,

    // AXI4-Lite slave
    input  logic [AXI_ADDR_W-1:0] s_awaddr,
    input  logic                  s_awvalid,
    output logic                  s_awready,
    input  logic [31:0]           s_wdata,
    input  logic [3:0]            s_wstrb,      // 32-bit accesses only; ignored
    input  logic                  s_wvalid,
    output logic                  s_wready,
    output logic [1:0]            s_bresp,
    output logic                  s_bvalid,
    input  logic                  s_bready,
    input  logic [AXI_ADDR_W-1:0] s_araddr,
    input  logic                  s_arvalid,
    output logic                  s_arready,
    output logic [31:0]           s_rdata,
    output logic [1:0]            s_rresp,
    output logic                  s_rvalid,
    input  logic                  s_rready,

    // from the bridge: any CRC / timeout / opcode error
    input  logic link_err,
    // from the bridge: a byte arrived while an answer was going out
    input  logic link_dropped,

    // to and from conv_layer
    output logic conv_soft_reset,
    output logic conv_start,
    input  logic conv_ready,
    input  logic conv_busy,
    input  logic conv_done,
    input  logic conv_err,
    input  logic [8:0] conv_err_code,
    output logic [DIM_W-1:0] h_in, w_in, h_out, w_out,
    output logic [CW_W-1:0] cin_words, cout_words,
    output logic [2:0] kh, kw,
    output logic [1:0] stride, pad,
    output logic signed [7:0] zp_in, zp_out,
    output logic relu_en,
    output logic [SEL_WIDTH-1:0] src_sel, dst_sel,
    output logic [ADDR_WIDTH-1:0] src_base, dst_base,
    output logic [B_ADDR_W-1:0] b_base,
    output logic [P_ADDR_W-1:0] p_base,
    output logic host_mode,
    output logic host_we,
    output logic [ADDR_WIDTH-1:0] host_waddr,
    output logic [N*8-1:0] host_wdata,
    output logic [SEL_WIDTH-1:0] host_wr_sel,
    output logic host_rd_en,
    output logic [ADDR_WIDTH-1:0] host_rd_addr,
    output logic [SEL_WIDTH-1:0] host_rd_sel,
    input  logic [N*8-1:0] host_rd_data,
    output logic b_we,
    output logic [B_ADDR_W-1:0] b_waddr,
    output logic [N*8-1:0] b_wdata,
    output logic p_we,
    output logic [P_ADDR_W-1:0] p_waddr,
    output logic [P_WIDTH-1:0] p_wdata,

    // to and from the sequencer (map 2.0)
    output logic seq_start,
    output logic [IDX_W-1:0] seq_start_index,
    input  logic seq_busy,
    input  logic seq_done,
    input  logic seq_err,
    input  logic [IDX_W-1:0] seq_cur_index,
    input  logic [IDX_W:0] seq_layers_done,
    input  logic [IDX_W-1:0] seq_err_index,
    input  logic [10:0] seq_err_code,
    output logic desc_we,
    output logic [IDX_W+3:0] desc_waddr,
    output logic [31:0] desc_wdata,
    output logic cyc_rd_en,
    output logic [IDX_W-1:0] cyc_rd_addr,
    input  logic [31:0] cyc_rd_data,

    // for the board LEDs
    output logic done_sticky,
    output logic err_sticky
);

    // ------------------------------------------------------------------
    // map
    // ------------------------------------------------------------------
    localparam logic [31:0] ID_VALUE = 32'h434E_5631;     // "CNV1"
    // 2.0: the descriptor sequencer. every 1.x register is unchanged
    localparam logic [31:0] VERSION  = 32'h0002_0000;

    localparam logic [7:0] A_ID = 8'h00, A_VERSION = 8'h04, A_PARAMS0 = 8'h08, A_PARAMS1 = 8'h0C,
                           A_CTRL = 8'h10, A_STATUS = 8'h14, A_ERR_CODE = 8'h18, A_CYCLES = 8'h1C,
                           A_SCRATCH = 8'h20, A_BUILD_ID = 8'h24, A_ACT_WCRC = 8'h28,
                           A_B_WCRC = 8'h2C, A_P_WCRC = 8'h30, A_LAUNCH = 8'h34,
                           A_PARAMS2 = 8'h38, A_CMD0 = 8'h40,
                           A_SEQ_CTRL = 8'h90, A_SEQ_STATUS = 8'h94, A_SEQ_ERR = 8'h98,
                           A_DESC_WCRC = 8'h9C;
    localparam int NCMD = 19;

    // a word of W bits takes S = ceil(W/32) slots, padded to a power of two
    localparam int WORD_W   = N * 8;
    localparam int A_SLOTS  = (WORD_W + 31) / 32;
    localparam int A_SHIFT  = (A_SLOTS > 1) ? $clog2(A_SLOTS) : 0;
    localparam int A_S      = 1 << A_SHIFT;
    localparam int P_SLOTS  = (P_WIDTH + 31) / 32;
    localparam int P_SHIFT  = (P_SLOTS > 1) ? $clog2(P_SLOTS) : 0;
    localparam int P_S      = 1 << P_SHIFT;
    localparam int AI_W     = SEL_WIDTH + ADDR_WIDTH;      // ACT word index {bank, addr}
    localparam int WORD_BYTES = WORD_W / 8;
    localparam int P_BYTES  = (P_WIDTH + 7) / 8;

    initial begin
        if (N > 16)
            $fatal(1, "csr: N must be <= 16 so an act word fits the staging (got %0d)", N);
        if (AXI_ADDR_W != 24)
            $fatal(1, "csr: the map assumes a 24-bit AXI address");
        if (P_SLOTS != 2)
            $fatal(1, "csr: a P word is expected to take two slots");
    end

    // CRC-16/CCITT-FALSE, one byte (reference: host/protocol.py crc16)
    function automatic logic [15:0] crc16_byte(input logic [15:0] c, input logic [7:0] b);
        logic [15:0] x;
        x = c ^ {b, 8'h00};
        for (int i = 0; i < 8; i++)
            x = x[15] ? ((x << 1) ^ 16'h1021) : (x << 1);
        return x;
    endfunction

    // the staged lower slots plus the top slot just written, as one word
    function automatic logic [A_S*32-1:0] assemble_a(input logic [A_S*32-1:0] stg, input logic [31:0] top);
        logic [A_S*32-1:0] w;
        w = stg;
        w[(A_S-1)*32 +: 32] = top;
        return w;
    endfunction

    // every lower slot valid (the top slot is the one being written now)
    function automatic logic lower_valid_a(input logic [A_S-1:0] v);
        return (v | (A_S'(1) << (A_S - 1))) == {A_S{1'b1}};
    endfunction

    // width of each command field, in register order
    function automatic int cmd_w(input int i);
        case (i)
            0, 1, 2, 3: return DIM_W;
            4, 5:       return CW_W;
            6, 7:       return 3;
            8, 9:       return 2;
            10, 11:     return 8;
            12:         return 1;
            13, 14:     return SEL_WIDTH;
            15, 16:     return ADDR_WIDTH;
            17:         return B_ADDR_W;
            default:    return P_ADDR_W;
        endcase
    endfunction

    // ------------------------------------------------------------------
    // state
    // ------------------------------------------------------------------
    logic [31:0] cmd [0:NCMD-1];
    logic [31:0] scratch, cycles, launch_count;
    logic [15:0] act_wcrc, b_wcrc, p_wcrc;
    // bytes the bridge dropped mid-answer: sticky flag and a count that
    // saturates at 0xFFFF instead of wrapping back to a harmless-looking 0
    logic        dropped_sticky;
    logic [15:0] dropped_count;
    // sequencer: sticky done/err of the last program, and the CRC of every
    // descriptor word written
    logic        seq_done_sticky, seq_err_sticky;
    logic [15:0] desc_wcrc;
    // busy for the host rules: a layer running, or a program between layers
    logic        busy;
    assign busy = conv_busy || seq_busy;

    assign h_in       = cmd[0][DIM_W-1:0];
    assign w_in       = cmd[1][DIM_W-1:0];
    assign h_out      = cmd[2][DIM_W-1:0];
    assign w_out      = cmd[3][DIM_W-1:0];
    assign cin_words  = cmd[4][CW_W-1:0];
    assign cout_words = cmd[5][CW_W-1:0];
    assign kh         = cmd[6][2:0];
    assign kw         = cmd[7][2:0];
    assign stride     = cmd[8][1:0];
    assign pad        = cmd[9][1:0];
    assign zp_in      = cmd[10][7:0];
    assign zp_out     = cmd[11][7:0];
    assign relu_en    = cmd[12][0];
    assign src_sel    = cmd[13][SEL_WIDTH-1:0];
    assign dst_sel    = cmd[14][SEL_WIDTH-1:0];
    assign src_base   = cmd[15][ADDR_WIDTH-1:0];
    assign dst_base   = cmd[16][ADDR_WIDTH-1:0];
    assign b_base     = cmd[17][B_ADDR_W-1:0];
    assign p_base     = cmd[18][P_ADDR_W-1:0];

    // wide-word staging, one set per window. valid bits cover the lower
    // slots only; the top slot is the commit.
    logic [A_S*32-1:0] act_stg, b_stg;
    logic [P_S*32-1:0] p_stg;
    logic [A_S-1:0]    act_vld, b_vld;
    logic [P_S-1:0]    p_vld;
    logic [AI_W-1:0]   act_stg_idx;
    logic [B_ADDR_W-1:0] b_stg_idx;
    logic [P_ADDR_W-1:0] p_stg_idx;

    // ------------------------------------------------------------------
    // address decode, shared by reads and writes
    // ------------------------------------------------------------------
    typedef enum logic [2:0] {R_REG, R_ACT, R_B, R_P, R_DESC, R_CYC, R_NONE} region_t;

    function automatic region_t region_of(input logic [AXI_ADDR_W-1:0] a);
        if (a[23:8] == 16'h0)  return R_REG;
        case (a[23:20])
            4'h1:    return R_ACT;
            4'h2:    return R_B;
            4'h3:    return R_P;
            4'h4:    return R_DESC;     // 2.0: descriptor words, write only
            4'h5:    return R_CYC;      // 2.0: per-descriptor cycle counts, read only
            default: return R_NONE;
        endcase
    endfunction

    // word index and slot inside a window, and whether the index is mapped
    function automatic logic [19:0] win_index(input logic [AXI_ADDR_W-1:0] a, input int shift);
        return a[19:2] >> shift;
    endfunction

    // ------------------------------------------------------------------
    // write channel
    // ------------------------------------------------------------------
    logic aw_held, w_held;
    logic [AXI_ADDR_W-1:0] aw_q;
    logic [31:0] w_q;

    assign s_awready = !aw_held && !s_bvalid;
    assign s_wready  = !w_held  && !s_bvalid;

    // decode of the held write
    region_t w_region;
    logic [7:0]  w_off;
    logic [19:0] w_idx_a, w_idx_b, w_idx_p;
    logic [A_SHIFT > 0 ? A_SHIFT-1 : 0:0] w_slot_a;
    logic [P_SHIFT > 0 ? P_SHIFT-1 : 0:0] w_slot_p;
    logic        w_top_a, w_top_p;
    assign w_region = region_of(aw_q);
    assign w_off    = aw_q[7:0];
    assign w_idx_a  = win_index(aw_q, A_SHIFT);
    assign w_idx_b  = win_index(aw_q, A_SHIFT);
    assign w_idx_p  = win_index(aw_q, P_SHIFT);
    assign w_slot_a = (A_SHIFT > 0) ? aw_q[2 +: (A_SHIFT > 0 ? A_SHIFT : 1)] : '0;
    assign w_slot_p = (P_SHIFT > 0) ? aw_q[2 +: (P_SHIFT > 0 ? P_SHIFT : 1)] : '0;
    assign w_top_a  = (A_SHIFT == 0) || (int'(w_slot_a) == A_S - 1);
    assign w_top_p  = (P_SHIFT == 0) || (int'(w_slot_p) == P_S - 1);

    logic do_write;
    assign do_write = aw_held && w_held && !s_bvalid;

    // ------------------------------------------------------------------
    // read channel
    // ------------------------------------------------------------------
    typedef enum logic [1:0] {RD_IDLE, RD_MEM, RD_CAP} rd_state_t;
    rd_state_t rd_state;
    logic [AXI_ADDR_W-1:0] ar_q;
    logic [$clog2(READ_LATENCY + 1)-1:0] rd_cnt;

    assign s_arready = (rd_state == RD_IDLE) && !s_rvalid;

    region_t r_region;
    logic [19:0] r_idx_a;
    logic [A_SHIFT > 0 ? A_SHIFT-1 : 0:0] r_slot_a;
    assign r_region = region_of(s_araddr);
    assign r_idx_a  = win_index(s_araddr, A_SHIFT);

    logic [AI_W-1:0] rd_idx_q;
    logic [A_SHIFT > 0 ? A_SHIFT-1 : 0:0] rd_slot_q;
    logic rd_is_cyc;                        // the read in flight is the cycle RAM, not ACT
    logic [IDX_W-1:0] rd_cyc_q;
    assign host_rd_en   = (rd_state == RD_MEM) && !rd_is_cyc;
    assign host_rd_addr = rd_idx_q[ADDR_WIDTH-1:0];
    assign host_rd_sel  = rd_idx_q[AI_W-1 -: SEL_WIDTH];
    // the cycle RAM is a bram with the same READ_LATENCY: the same
    // hold-rd_en-for-READ_LATENCY-edges path reads it
    assign cyc_rd_en    = (rd_state == RD_MEM) && rd_is_cyc;
    assign cyc_rd_addr  = rd_cyc_q;

    // register read value as {mapped, value}
    function automatic logic [32:0] reg_value(input logic [7:0] off);
        case (off)
            A_ID:       return {1'b1, ID_VALUE};
            A_VERSION:  return {1'b1, VERSION};
            A_PARAMS0:  return {1'b1, {8'(B_ADDR_W), 4'(NBUF), 8'(ADDR_WIDTH), 4'(READ_LATENCY), 8'(N)}};
            A_PARAMS1:  return {1'b1, {8'h0, 16'(K_MAX), 8'(P_ADDR_W)}};
            A_CTRL:     return {1'b1, {29'h0, host_mode, 2'b00}};
            // busy and ready cover a running program too (2.0)
            A_STATUS:   return {1'b1, {dropped_count, 10'h0, dropped_sticky, host_mode, err_sticky,
                                       done_sticky, !busy, busy}};
            A_ERR_CODE: return {1'b1, {23'h0, conv_err_code}};
            A_CYCLES:   return {1'b1, cycles};
            A_SCRATCH:  return {1'b1, scratch};
            A_BUILD_ID: return {1'b1, BUILD_ID};
            A_ACT_WCRC: return {1'b1, {16'h0, act_wcrc}};
            A_B_WCRC:   return {1'b1, {16'h0, b_wcrc}};
            A_P_WCRC:   return {1'b1, {16'h0, p_wcrc}};
            A_LAUNCH:   return {1'b1, launch_count};
            A_PARAMS2:  return {1'b1, {15'h0, 1'(HAS_DMA), 8'(DESC_DEPTH), 8'(M_BITS)}};
            A_SEQ_CTRL: return {1'b1, 32'h0};
            A_SEQ_STATUS: return {1'b1, {16'(seq_layers_done), 8'(seq_cur_index), 5'h0,
                                         seq_err_sticky, seq_done_sticky, seq_busy}};
            A_SEQ_ERR:  return {1'b1, {16'(seq_err_index), 5'h0, seq_err_code}};
            A_DESC_WCRC: return {1'b1, {16'h0, desc_wcrc}};
            default: begin
                if (off >= A_CMD0 && off < A_CMD0 + 8'(4*NCMD) && off[1:0] == 2'b00) begin
                    // ZP_IN and ZP_OUT read back sign-extended
                    if (off == A_CMD0 + 8'd40 || off == A_CMD0 + 8'd44)
                        return {1'b1, {{24{cmd[(off - A_CMD0) >> 2][7]}}, cmd[(off - A_CMD0) >> 2][7:0]}};
                    return {1'b1, cmd[(off - A_CMD0) >> 2]};
                end
                return 33'h0;
            end
        endcase
    endfunction

    // ------------------------------------------------------------------
    // the one sequential block
    // ------------------------------------------------------------------
    // blocking temporaries, only ever read in the same pass that writes them
    logic [A_S*32-1:0] act_word;
    logic [P_S*32-1:0] p_word;
    logic [15:0] crc_t;
    logic reg_ok;
    logic [31:0] reg_rd;

    always_ff @(posedge clk) begin
        if (reset) begin
            aw_held <= 1'b0; w_held <= 1'b0; aw_q <= '0; w_q <= '0;
            s_bvalid <= 1'b0; s_bresp <= 2'b00;
            s_rvalid <= 1'b0; s_rresp <= 2'b00; s_rdata <= '0;
            rd_state <= RD_IDLE; ar_q <= '0; rd_cnt <= '0; rd_idx_q <= '0; rd_slot_q <= '0;
            rd_is_cyc <= 1'b0; rd_cyc_q <= '0;
            seq_start <= 1'b0; seq_start_index <= '0;
            seq_done_sticky <= 1'b0; seq_err_sticky <= 1'b0;
            desc_we <= 1'b0; desc_waddr <= '0; desc_wdata <= '0; desc_wcrc <= 16'hFFFF;
            for (int i = 0; i < NCMD; i++) cmd[i] <= '0;
            scratch <= '0; cycles <= '0; launch_count <= '0;
            act_wcrc <= 16'hFFFF; b_wcrc <= 16'hFFFF; p_wcrc <= 16'hFFFF;
            host_mode <= 1'b0; conv_start <= 1'b0; conv_soft_reset <= 1'b0;
            done_sticky <= 1'b0; err_sticky <= 1'b0;
            dropped_sticky <= 1'b0; dropped_count <= '0;
            host_we <= 1'b0; host_waddr <= '0; host_wdata <= '0; host_wr_sel <= '0;
            b_we <= 1'b0; b_waddr <= '0; b_wdata <= '0;
            p_we <= 1'b0; p_waddr <= '0; p_wdata <= '0;
            act_stg <= '0; b_stg <= '0; p_stg <= '0;
            act_vld <= '0; b_vld <= '0; p_vld <= '0;
            act_stg_idx <= '0; b_stg_idx <= '0; p_stg_idx <= '0;
        end else begin
            // one-cycle strobes
            conv_start      <= 1'b0;
            conv_soft_reset <= 1'b0;
            host_we <= 1'b0;
            b_we    <= 1'b0;
            p_we    <= 1'b0;
            seq_start <= 1'b0;
            desc_we   <= 1'b0;

            // status from conv_layer. start (accepted on this edge, because
            // the CSR only starts when ready) and soft reset both clear the
            // sticky bits and the counter; the pulses set the sticky bits.
            // these describe the legacy CTRL.start command only: while a
            // program runs they do not move (2.0; programs use SEQ_* and CYC)
            if (conv_start && conv_ready) begin
                cycles       <= '0;
                launch_count <= launch_count + 1'b1;
                done_sticky  <= 1'b0;
                err_sticky   <= 1'b0;
            end else if (conv_soft_reset) begin
                cycles      <= '0;
                done_sticky <= 1'b0;
                err_sticky  <= 1'b0;
            end else if (!seq_busy) begin
                if (conv_busy) cycles      <= cycles + 1'b1;
                if (conv_done) done_sticky <= 1'b1;
                if (conv_err)  err_sticky  <= 1'b1;
            end

            // the sequencer's sticky bits: cleared by a program start and by
            // a soft reset (which also resets the sequencer), set by its pulses
            if (seq_start || conv_soft_reset) begin
                seq_done_sticky <= 1'b0;
                seq_err_sticky  <= 1'b0;
            end else begin
                if (seq_done) seq_done_sticky <= 1'b1;
                if (seq_err)  seq_err_sticky  <= 1'b1;
            end

            // link state, not layer state: start and soft reset leave it
            // alone; only W1C (below, which wins on the same edge) and the
            // board reset clear it
            if (link_dropped) begin
                dropped_sticky <= 1'b1;
                if (dropped_count != 16'hFFFF) dropped_count <= dropped_count + 1'b1;
            end

            // ---------------- write channel ----------------
            if (s_awvalid && s_awready) begin aw_q <= s_awaddr; aw_held <= 1'b1; end
            if (s_wvalid  && s_wready)  begin w_q  <= s_wdata;  w_held  <= 1'b1; end
            if (s_bvalid && s_bready) s_bvalid <= 1'b0;

            if (do_write) begin
                aw_held  <= 1'b0;
                w_held   <= 1'b0;
                s_bvalid <= 1'b1;
                s_bresp  <= 2'b00;
                case (w_region)
                    R_REG: begin
                        if (w_off == A_CTRL) begin
                            if (busy) begin
                                // only a soft reset gets through while busy
                                if (w_q[1]) conv_soft_reset <= 1'b1;
                                else        s_bresp <= 2'b10;
                            end else begin
                                conv_start      <= w_q[0];
                                conv_soft_reset <= w_q[1];
                                host_mode       <= w_q[2];
                                if (w_q[3]) begin
                                    act_wcrc  <= 16'hFFFF;
                                    b_wcrc    <= 16'hFFFF;
                                    p_wcrc    <= 16'hFFFF;
                                    desc_wcrc <= 16'hFFFF;
                                end
                            end
                        end else if (w_off == A_SEQ_CTRL) begin
                            // a program start, refused while anything runs
                            if (busy) s_bresp <= 2'b10;
                            else if (w_q[0]) begin
                                seq_start       <= 1'b1;
                                seq_start_index <= w_q[8 +: IDX_W];
                            end
                        end else if (w_off == A_SEQ_STATUS) begin
                            // W1C of the program's sticky bits, allowed while busy
                            if (w_q[1]) seq_done_sticky <= 1'b0;
                            if (w_q[2]) seq_err_sticky  <= 1'b0;
                        end else if (w_off == A_STATUS) begin
                            // W1C, allowed while busy
                            if (w_q[2]) done_sticky <= 1'b0;
                            if (w_q[3]) err_sticky  <= 1'b0;
                            if (w_q[5]) begin
                                dropped_sticky <= 1'b0;
                                dropped_count  <= '0;
                            end
                        end else if (w_off == A_SCRATCH) begin
                            if (busy) s_bresp <= 2'b10;
                            else      scratch <= w_q;
                        end else if (w_off >= A_CMD0 && w_off < A_CMD0 + 8'(4*NCMD) && w_off[1:0] == 2'b00) begin
                            if (busy) s_bresp <= 2'b10;
                            else begin
                                for (int i = 0; i < NCMD; i++)
                                    if (w_off == A_CMD0 + 8'(4*i))
                                        cmd[i] <= w_q & ((32'h1 << cmd_w(i)) - 1);
                            end
                        end else begin
                            // read-only registers refuse writes; holes are unmapped
                            {reg_ok, reg_rd} = reg_value(w_off);
                            s_bresp <= reg_ok ? 2'b10 : 2'b11;
                        end
                    end

                    R_ACT: begin
                        if ((w_idx_a >> AI_W) != 0 ||
                            int'(w_idx_a[AI_W-1 -: SEL_WIDTH]) >= NBUF) s_bresp <= 2'b11;
                        else if (busy || !host_mode)              s_bresp <= 2'b10;
                        else if (!w_top_a) begin
                            act_stg[w_slot_a*32 +: 32] <= w_q;
                            act_stg_idx <= w_idx_a[AI_W-1:0];
                            if (act_stg_idx == w_idx_a[AI_W-1:0]) act_vld[w_slot_a] <= 1'b1;
                            else act_vld <= A_S'(1) << w_slot_a;
                        end else begin
                            act_vld <= '0;
                            if (A_S > 1 && (!lower_valid_a(act_vld) || act_stg_idx != w_idx_a[AI_W-1:0]))
                                s_bresp <= 2'b10;
                            else begin
                                act_word = assemble_a(act_stg, w_q);
                                host_we     <= 1'b1;
                                host_waddr  <= w_idx_a[ADDR_WIDTH-1:0];
                                host_wr_sel <= w_idx_a[AI_W-1 -: SEL_WIDTH];
                                host_wdata  <= act_word[WORD_W-1:0];
                                crc_t = act_wcrc;
                                for (int k = 0; k < WORD_BYTES; k++)
                                    crc_t = crc16_byte(crc_t, act_word[k*8 +: 8]);
                                act_wcrc <= crc_t;
                            end
                        end
                    end

                    R_B: begin
                        if ((w_idx_b >> B_ADDR_W) != 0) s_bresp <= 2'b11;
                        else if (busy)                  s_bresp <= 2'b10;
                        else if (!w_top_a) begin
                            b_stg[w_slot_a*32 +: 32] <= w_q;
                            b_stg_idx <= w_idx_b[B_ADDR_W-1:0];
                            if (b_stg_idx == w_idx_b[B_ADDR_W-1:0]) b_vld[w_slot_a] <= 1'b1;
                            else b_vld <= A_S'(1) << w_slot_a;
                        end else begin
                            b_vld <= '0;
                            if (A_S > 1 && (!lower_valid_a(b_vld) || b_stg_idx != w_idx_b[B_ADDR_W-1:0]))
                                s_bresp <= 2'b10;
                            else begin
                                act_word = assemble_a(b_stg, w_q);
                                b_we    <= 1'b1;
                                b_waddr <= w_idx_b[B_ADDR_W-1:0];
                                b_wdata <= act_word[WORD_W-1:0];
                                crc_t = b_wcrc;
                                for (int k = 0; k < WORD_BYTES; k++)
                                    crc_t = crc16_byte(crc_t, act_word[k*8 +: 8]);
                                b_wcrc <= crc_t;
                            end
                        end
                    end

                    R_P: begin
                        if ((w_idx_p >> P_ADDR_W) != 0) s_bresp <= 2'b11;
                        else if (busy)                  s_bresp <= 2'b10;
                        else if (!w_top_p) begin
                            p_stg[w_slot_p*32 +: 32] <= w_q;
                            p_stg_idx <= w_idx_p[P_ADDR_W-1:0];
                            if (p_stg_idx == w_idx_p[P_ADDR_W-1:0]) p_vld[w_slot_p] <= 1'b1;
                            else p_vld <= P_S'(1) << w_slot_p;
                        end else begin
                            p_vld <= '0;
                            // P is always two slots (checked at elaboration)
                            if (!p_vld[0] || p_stg_idx != w_idx_p[P_ADDR_W-1:0])
                                s_bresp <= 2'b10;
                            else begin
                                // bits above P_WIDTH are dropped, so they are 0 in the CRC too
                                p_word = {w_q, p_stg[31:0]} & ((64'h1 << P_WIDTH) - 1);
                                p_we    <= 1'b1;
                                p_waddr <= w_idx_p[P_ADDR_W-1:0];
                                p_wdata <= p_word[P_WIDTH-1:0];
                                crc_t = p_wcrc;
                                for (int k = 0; k < P_BYTES; k++)
                                    crc_t = crc16_byte(crc_t, p_word[k*8 +: 8]);
                                p_wcrc <= crc_t;
                            end
                        end
                    end

                    // 2.0: one 32-bit descriptor word per write, no staging
                    R_DESC: begin
                        if (int'(aw_q[19:2]) >= DESC_DEPTH * 16) s_bresp <= 2'b11;
                        else if (busy)                          s_bresp <= 2'b10;
                        else begin
                            desc_we    <= 1'b1;
                            desc_waddr <= aw_q[2 +: IDX_W + 4];
                            desc_wdata <= w_q;
                            crc_t = desc_wcrc;
                            for (int k = 0; k < 4; k++)
                                crc_t = crc16_byte(crc_t, w_q[k*8 +: 8]);
                            desc_wcrc <= crc_t;
                        end
                    end

                    // the cycle window is read only
                    R_CYC: s_bresp <= (int'(aw_q[19:2]) >= DESC_DEPTH) ? 2'b11 : 2'b10;

                    default: s_bresp <= 2'b11;
                endcase
            end

            // any link error throws away every half-staged word
            if (link_err) begin
                act_vld <= '0;
                b_vld   <= '0;
                p_vld   <= '0;
            end

            // ---------------- read channel ----------------
            if (s_rvalid && s_rready) s_rvalid <= 1'b0;

            case (rd_state)
                RD_IDLE: begin
                    if (s_arvalid && s_arready) begin
                        ar_q <= s_araddr;
                        case (r_region)
                            R_REG: begin
                                {reg_ok, reg_rd} = reg_value(s_araddr[7:0]);
                                s_rdata  <= reg_ok ? reg_rd : 32'h0;
                                s_rresp  <= reg_ok ? 2'b00 : 2'b11;
                                s_rvalid <= 1'b1;
                            end
                            R_ACT: begin
                                if ((r_idx_a >> AI_W) != 0 ||
                                    int'(r_idx_a[AI_W-1 -: SEL_WIDTH]) >= NBUF) begin
                                    s_rdata <= '0; s_rresp <= 2'b11; s_rvalid <= 1'b1;
                                end else if (busy || host_mode) begin
                                    s_rdata <= '0; s_rresp <= 2'b10; s_rvalid <= 1'b1;
                                end else begin
                                    rd_idx_q  <= r_idx_a[AI_W-1:0];
                                    rd_slot_q <= (A_SHIFT > 0) ? s_araddr[2 +: (A_SHIFT > 0 ? A_SHIFT : 1)] : '0;
                                    rd_is_cyc <= 1'b0;
                                    rd_cnt    <= '0;
                                    rd_state  <= RD_MEM;
                                end
                            end
                            // 2.0: per-descriptor cycle counts, through the same
                            // latency-handled path as ACT
                            R_CYC: begin
                                if (int'(s_araddr[19:2]) >= DESC_DEPTH) begin
                                    s_rdata <= '0; s_rresp <= 2'b11; s_rvalid <= 1'b1;
                                end else if (busy) begin
                                    s_rdata <= '0; s_rresp <= 2'b10; s_rvalid <= 1'b1;
                                end else begin
                                    rd_cyc_q  <= s_araddr[2 +: IDX_W];
                                    rd_is_cyc <= 1'b1;
                                    rd_cnt    <= '0;
                                    rd_state  <= RD_MEM;
                                end
                            end
                            // the descriptor window is write only
                            R_DESC: begin
                                s_rdata <= '0;
                                s_rresp <= (int'(s_araddr[19:2]) >= DESC_DEPTH * 16) ? 2'b11 : 2'b10;
                                s_rvalid <= 1'b1;
                            end
                            // B and P are write only; outside the windows is unmapped
                            R_B, R_P: begin s_rdata <= '0; s_rresp <= 2'b10; s_rvalid <= 1'b1; end
                            default:  begin s_rdata <= '0; s_rresp <= 2'b11; s_rvalid <= 1'b1; end
                        endcase
                    end
                end
                // READ_LATENCY edges with host_rd_en high and a fixed address
                RD_MEM: begin
                    if (int'(rd_cnt) == READ_LATENCY - 1) rd_state <= RD_CAP;
                    else rd_cnt <= rd_cnt + 1'b1;
                end
                // host_rd_data now holds the word; host_rd_en is low, so it stays
                RD_CAP: begin
                    s_rdata  <= rd_is_cyc ? cyc_rd_data
                                          : 32'((A_S*32)'(host_rd_data) >> (rd_slot_q * 32));
                    s_rresp  <= 2'b00;
                    s_rvalid <= 1'b1;
                    rd_state <= RD_IDLE;
                end
                default: rd_state <= RD_IDLE;
            endcase
        end
    end

endmodule
