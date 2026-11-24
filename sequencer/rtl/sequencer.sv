// sequencer: runs a program of layer descriptors on layer_engine without the
// host. contract: sequencer/sequencer_contract.md.
//
//   fetch descriptor idx (16 x 32-bit words) -> END? stop : start the engine
//   -> count edges to done -> cycle RAM[idx] = count -> idx + 1 -> fetch ...
//
// the descriptor RAM is 32 bits wide (one BRAM18 for 32 descriptors) and a
// descriptor is fetched in 16 reads, with rd_en held high and a tag pipeline
// meeting the data, as everywhere else. the decoded fields sit on the engine
// bus continuously; they only matter on the cycle eng_start is high.

module sequencer #(
    parameter int ADDR_WIDTH   = 12,       // act_buffer word address
    parameter int NBUF         = 3,
    parameter int B_ADDR_W     = 13,
    parameter int P_ADDR_W     = 6,
    parameter int DIM_W        = 8,
    parameter int CW_W         = 7,
    parameter int M_BITS       = 8,
    parameter int DESC_DEPTH   = 32,
    parameter int READ_LATENCY = 1,        // of the descriptor and cycle RAMs
    // derived, do not override
    parameter int SEL_WIDTH    = (NBUF > 1) ? $clog2(NBUF) : 1,
    parameter int IDX_W        = (DESC_DEPTH > 1) ? $clog2(DESC_DEPTH) : 1
)(
    input  logic clk,
    input  logic reset,

    // control and status
    input  logic start,
    input  logic [IDX_W-1:0] start_index,
    output logic busy,
    output logic done,
    output logic err,
    output logic [IDX_W-1:0] cur_index,
    output logic [IDX_W:0] layers_done,
    output logic [IDX_W-1:0] err_index,
    output logic [10:0] err_code,

    // descriptor RAM write port (from the CSR)
    input  logic desc_we,
    input  logic [IDX_W+3:0] desc_waddr,
    input  logic [31:0] desc_wdata,

    // cycle RAM read port (to the CSR)
    input  logic cyc_rd_en,
    input  logic [IDX_W-1:0] cyc_rd_addr,
    output logic [31:0] cyc_rd_data,

    // layer_engine command bus
    output logic eng_start,
    output logic [2:0] eng_op,
    output logic [DIM_W-1:0] h_in, w_in, h_out, w_out,
    output logic [CW_W-1:0] cin_words, cout_words,
    output logic [2:0] kh, kw,
    output logic [1:0] stride, pad,
    output logic signed [7:0] zp_in, zp_out, zp_in2,
    output logic relu_en,
    output logic [SEL_WIDTH-1:0] src_sel, dst_sel, src2_sel,
    output logic [ADDR_WIDTH-1:0] src_base, dst_base, src2_base,
    output logic [B_ADDR_W-1:0] b_base,
    output logic [P_ADDR_W-1:0] p_base,
    output logic [M_BITS-1:0] add_ma, add_mb,
    output logic [3:0] add_shift,
    input  logic eng_ready,
    input  logic eng_done,
    input  logic eng_err,
    input  logic [9:0] eng_err_code
);

    initial begin
        if (DESC_DEPTH < 2 || (DESC_DEPTH & (DESC_DEPTH - 1)) != 0)
            $fatal(1, "sequencer: DESC_DEPTH must be a power of two >= 2 (got %0d)", DESC_DEPTH);
        if (ADDR_WIDTH > 16 || B_ADDR_W > 16 || P_ADDR_W > 8 || SEL_WIDTH > 4 || M_BITS > 16)
            $fatal(1, "sequencer: a field is wider than its descriptor slot");
    end

    typedef enum logic [2:0] {S_IDLE, S_FETCH, S_DECODE, S_LAUNCH, S_RUN, S_NEXT} state_t;
    state_t state;
    assign busy = (state != S_IDLE);

    logic [IDX_W-1:0] idx;
    assign cur_index = idx;

    // ------------------------------------------------------------------
    // descriptor RAM and the fetch
    // ------------------------------------------------------------------
    logic [4:0] k_iss;                          // next word to read, 0..16
    logic [31:0] dw [0:15];                     // the descriptor being run
    logic [31:0] desc_rdata;
    logic t_valid;
    logic [3:0] t_k;
    logic [4:0] k_got;                          // words captured

    bram #(.DATA_WIDTH(32), .ADDR_WIDTH(IDX_W + 4), .READ_LATENCY(READ_LATENCY)) u_desc (
        .clk(clk), .we(desc_we), .waddr(desc_waddr), .wdata(desc_wdata),
        .re(state == S_FETCH), .raddr({idx, k_iss[3:0]}), .rdata(desc_rdata)
    );

    // the tag pipeline shifts every cycle, so outside FETCH it fills with
    // invalid tags. clocking it only in FETCH would freeze the last word's
    // valid tag at its output and count it again at the next fetch.
    delay #(.DATA_WIDTH(5), .DEPTH(READ_LATENCY)) u_tag (
        .clk(clk), .reset(reset), .en(1'b1),
        .in_data({(state == S_FETCH) && (k_iss < 5'd16), k_iss[3:0]}),
        .out_data({t_valid, t_k})
    );

    // ------------------------------------------------------------------
    // decode (widths as in the contract's table)
    // ------------------------------------------------------------------
    assign eng_op     = dw[0][2:0];
    assign relu_en    = dw[0][3];
    assign zp_in      = dw[0][15:8];
    assign zp_out     = dw[0][23:16];
    assign zp_in2     = dw[0][31:24];
    assign h_in       = dw[1][7:0];
    assign w_in       = dw[1][15:8];
    assign h_out      = dw[1][23:16];
    assign w_out      = dw[1][31:24];
    assign cin_words  = dw[2][6:0];
    assign cout_words = dw[2][14:8];
    assign kh         = dw[2][18:16];
    assign kw         = dw[2][22:20];
    assign stride     = dw[2][25:24];
    assign pad        = dw[2][29:28];
    assign src_sel    = dw[3][SEL_WIDTH-1:0];
    assign dst_sel    = dw[3][4 +: SEL_WIDTH];
    assign src2_sel   = dw[3][8 +: SEL_WIDTH];
    assign src_base   = dw[3][16 +: ADDR_WIDTH];
    assign dst_base   = dw[4][ADDR_WIDTH-1:0];
    assign src2_base  = dw[4][16 +: ADDR_WIDTH];
    assign b_base     = dw[5][B_ADDR_W-1:0];
    assign p_base     = dw[5][16 +: P_ADDR_W];
    assign add_shift  = dw[5][27:24];
    assign add_ma     = dw[6][M_BITS-1:0];
    assign add_mb     = dw[6][16 +: M_BITS];

    // ------------------------------------------------------------------
    // run
    // ------------------------------------------------------------------
    logic [31:0] cnt;
    logic cyc_we;
    logic [IDX_W-1:0] cyc_waddr;
    logic [31:0] cyc_wdata;

    bram #(.DATA_WIDTH(32), .ADDR_WIDTH(IDX_W), .READ_LATENCY(READ_LATENCY)) u_cyc (
        .clk(clk), .we(cyc_we), .waddr(cyc_waddr), .wdata(cyc_wdata),
        .re(cyc_rd_en), .raddr(cyc_rd_addr), .rdata(cyc_rd_data)
    );

    assign eng_start = (state == S_LAUNCH) && eng_ready;

    always_ff @(posedge clk) begin
        if (reset) begin
            state       <= S_IDLE;
            idx         <= '0;
            k_iss       <= '0;
            k_got       <= '0;
            cnt         <= '0;
            done        <= 1'b0;
            err         <= 1'b0;
            err_code    <= '0;
            err_index   <= '0;
            layers_done <= '0;
            cyc_we      <= 1'b0;
        end else begin
            done   <= 1'b0;
            err    <= 1'b0;
            cyc_we <= 1'b0;

            if (t_valid) dw[t_k] <= desc_rdata;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        idx         <= start_index;
                        layers_done <= '0;
                        err_code    <= '0;
                        k_iss       <= '0;
                        k_got       <= '0;
                        state       <= S_FETCH;
                    end
                end

                // 16 reads, one per cycle, then wait for the last word
                S_FETCH: begin
                    if (k_iss != 5'd16) k_iss <= k_iss + 1'b1;
                    if (t_valid) begin
                        k_got <= k_got + 1'b1;
                        if (k_got == 5'd15) state <= S_DECODE;
                    end
                end

                S_DECODE: begin
                    if (dw[0][2:0] == 3'd0) begin
                        done  <= 1'b1;                 // END
                        state <= S_IDLE;
                    end else begin
                        state <= S_LAUNCH;
                    end
                end

                // the engine is idle here by construction; eng_start is high
                // for exactly the cycle it accepts
                S_LAUNCH: begin
                    if (eng_ready) begin
                        cnt   <= '0;
                        state <= S_RUN;
                    end
                end

                // edges from the accepting edge to done: the CSR CYCLES rule
                S_RUN: begin
                    cnt <= cnt + 1'b1;
                    if (eng_done) begin
                        cyc_we      <= 1'b1;
                        cyc_waddr   <= idx;
                        cyc_wdata   <= cnt;
                        layers_done <= layers_done + 1'b1;
                        state       <= S_NEXT;
                    end else if (eng_err) begin
                        err       <= 1'b1;
                        err_index <= idx;
                        err_code  <= {1'b0, eng_err_code};
                        state     <= S_IDLE;
                    end
                end

                S_NEXT: begin
                    if (int'(idx) == DESC_DEPTH - 1) begin
                        err       <= 1'b1;             // ran off the end: no END
                        err_index <= idx;
                        err_code  <= 11'b100_0000_0000;
                        state     <= S_IDLE;
                    end else begin
                        idx   <= idx + 1'b1;
                        k_iss <= '0;
                        k_got <= '0;
                        state <= S_FETCH;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
