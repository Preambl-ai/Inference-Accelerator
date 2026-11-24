// wdma: weight DMA. fetches a conv layer's B and P from AXI4 memory into the
// on-chip B and P memories (one INCR burst per output group for each, split
// only at the 256-beat and 4 KB limits), and gives the CSR single-beat access
// to the same memory. contract: dma/dma_contract.md.
//
//   IDLE -> CHK -> { GRP -> { SEG -> AR -> R }* (B, then P) }* per group -> IDLE
//   IDLE -> HW -> HB -> IDLE      host write
//   IDLE -> HAR -> HR -> IDLE     host read
//
// one burst outstanding: the next AR goes out after the current RLAST. R is
// always accepted while a burst is open (the B/P write ports take a word
// every cycle), and each beat is written one edge later from a register.
// no multipliers: addresses are running sums and the 4 KB split is a
// subtraction of address bits.

module wdma #(
    parameter int N        = 8,
    parameter int B_ADDR_W = 13,
    parameter int P_ADDR_W = 8,
    parameter int K_WIDTH  = 11,
    parameter int CW_W     = 7,
    parameter int AXI_AW   = 32,
    parameter int AXI_IDW  = 4,
    // derived, do not override
    parameter int P_WIDTH  = 53
)(
    input  logic clk,
    input  logic reset,                 // board reset (the AXI slave's too)
    input  logic abort,                 // soft reset: finish the AXI burst in flight, write nothing more

    // fetch command (latched on f_start & f_ready)
    input  logic f_start,
    input  logic [AXI_AW-1:0] f_b_src,
    input  logic [AXI_AW-1:0] f_p_src,
    input  logic [K_WIDTH-1:0] f_k,
    input  logic [CW_W-1:0] f_groups,
    input  logic [B_ADDR_W-1:0] f_b_dst,
    input  logic [P_ADDR_W-1:0] f_p_dst,
    output logic f_ready,
    output logic busy,
    output logic f_done,
    output logic f_err,
    output logic [3:0] f_err_code,

    // host single beat (CSR DDR window)
    input  logic h_req,
    input  logic h_we,
    input  logic [AXI_AW-1:0] h_addr,
    input  logic [31:0] h_wdata,
    output logic h_ack,
    output logic [1:0] h_resp,
    output logic [31:0] h_rdata,

    // on-chip B and P write ports
    output logic b_we,
    output logic [B_ADDR_W-1:0] b_waddr,
    output logic [N*8-1:0] b_wdata,
    output logic p_we,
    output logic [P_ADDR_W-1:0] p_waddr,
    output logic [P_WIDTH-1:0] p_wdata,

    // AXI4 master
    output logic [AXI_AW-1:0] araddr,
    output logic [7:0]  arlen,
    output logic [2:0]  arsize,
    output logic [1:0]  arburst,
    output logic [AXI_IDW-1:0] arid,
    output logic        arvalid,
    input  logic        arready,
    input  logic [63:0] rdata,
    input  logic [1:0]  rresp,
    input  logic        rlast,
    input  logic [AXI_IDW-1:0] rid,
    input  logic        rvalid,
    output logic        rready,
    output logic [AXI_AW-1:0] awaddr,
    output logic [7:0]  awlen,
    output logic [2:0]  awsize,
    output logic [1:0]  awburst,
    output logic [AXI_IDW-1:0] awid,
    output logic        awvalid,
    input  logic        awready,
    output logic [63:0] wdata,
    output logic [7:0]  wstrb,
    output logic        wlast,
    output logic        wvalid,
    input  logic        wready,
    input  logic [1:0]  bresp,
    input  logic [AXI_IDW-1:0] bid,
    input  logic        bvalid,
    output logic        bready
);

    initial begin
        if (N < 1 || N > 8)
            $fatal(1, "wdma: N must be 1..8 (a B word is one 64-bit beat), got %0d", N);
        if (P_WIDTH > 64)
            $fatal(1, "wdma: P_WIDTH must fit a 64-bit beat");
    end

    localparam logic [3:0] E_DIM = 4'b0001, E_ALIGN = 4'b0010, E_RANGE = 4'b0100,
                           E_RESP = 4'b1000;
    // range-check widths: wide enough for the pointer AND the count, so K
    // never truncates (B_ADDR_W + 2 bits held K = 1024 as 0 when B_ADDR_W = 8)
    localparam int RB_W = ((B_ADDR_W > K_WIDTH) ? B_ADDR_W : K_WIDTH) + 2;
    localparam int RP_W = ((P_ADDR_W > 4) ? P_ADDR_W : 4) + 2;

    typedef enum logic [3:0] {S_IDLE, S_CHK, S_GRP, S_SEG, S_AR, S_R,
                              S_HW, S_HB, S_HAR, S_HR} state_t;
    state_t state;

    // latched command and running pointers
    logic [K_WIDTH-1:0] k_q;
    logic [CW_W-1:0]    groups_q, og;
    logic [AXI_AW-1:0]  b_addr, p_addr;         // next DDR byte address of B and of P
    logic [B_ADDR_W:0]  b_ptr;                  // next on-chip B word (one bit wider: range check)
    logic [P_ADDR_W:0]  p_ptr;
    logic               phase_p;                // 0: B of this group, 1: P
    logic [K_WIDTH-1:0] left;                   // beats of this group's B or P not yet requested
    logic [8:0]         beats;                  // of the burst in flight (1 .. 256)
    logic [8:0]         got;                    // beats of it received
    logic               aborted, resp_bad;

    // host access
    logic               h_we_q, h_hi;
    logic               aw_done, w_done;

    // the burst that fits from `addr`: no more than `left`, 256, or the beats
    // to the next 4 KB line
    logic [AXI_AW-1:0] seg_addr;
    logic [9:0]  to_4k;
    logic [K_WIDTH-1:0] cap;
    logic [K_WIDTH-1:0] seg_beats;
    always_comb begin
        seg_addr = phase_p ? p_addr : b_addr;
        to_4k    = 10'd512 - {1'b0, seg_addr[11:3]};
        cap      = (to_4k > 10'd256) ? K_WIDTH'(256) : K_WIDTH'(to_4k);
        seg_beats = (left < cap) ? left : cap;
    end

    assign f_ready = (state == S_IDLE);
    assign busy    = (state != S_IDLE);
    assign arsize  = 3'd3;
    assign arburst = 2'b01;
    assign arid    = '0;
    assign awsize  = 3'd3;
    assign awburst = 2'b01;
    assign awid    = '0;
    assign awlen   = 8'd0;
    assign wlast   = 1'b1;
    assign rready  = (state == S_R) || (state == S_HR);
    assign bready  = (state == S_HB);

    always_ff @(posedge clk) begin
        if (reset) begin
            state      <= S_IDLE;
            arvalid    <= 1'b0;
            awvalid    <= 1'b0;
            wvalid     <= 1'b0;
            f_done     <= 1'b0;
            f_err      <= 1'b0;
            f_err_code <= '0;
            h_ack      <= 1'b0;
            h_resp     <= '0;
            h_rdata    <= '0;
            b_we       <= 1'b0;
            p_we       <= 1'b0;
            aborted    <= 1'b0;
            resp_bad   <= 1'b0;
        end else begin
            f_done <= 1'b0;
            f_err  <= 1'b0;
            h_ack  <= 1'b0;
            b_we   <= 1'b0;
            p_we   <= 1'b0;
            if (abort && busy) aborted <= 1'b1;

            case (state)
                S_IDLE: begin
                    aborted  <= 1'b0;
                    resp_bad <= 1'b0;
                    if (f_start) begin
                        k_q        <= f_k;
                        groups_q   <= f_groups;
                        b_addr     <= f_b_src;
                        p_addr     <= f_p_src;
                        b_ptr      <= {1'b0, f_b_dst};
                        p_ptr      <= {1'b0, f_p_dst};
                        og         <= '0;
                        f_err_code <= '0;
                        state      <= S_CHK;
                    end else if (h_req) begin
                        h_we_q  <= h_we;
                        h_hi    <= h_addr[2];
                        if (h_we) begin
                            awaddr  <= {h_addr[AXI_AW-1:3], 3'b000};
                            wdata   <= {h_wdata, h_wdata};
                            wstrb   <= h_addr[2] ? 8'hF0 : 8'h0F;
                            awvalid <= 1'b1;
                            wvalid  <= 1'b1;
                            aw_done <= 1'b0;
                            w_done  <= 1'b0;
                            state   <= S_HW;
                        end else begin
                            araddr  <= {h_addr[AXI_AW-1:3], 3'b000};
                            arlen   <= 8'd0;
                            arvalid <= 1'b1;
                            state   <= S_HAR;
                        end
                    end
                end

                S_CHK: begin
                    if (aborted || abort) begin
                        state <= S_IDLE;
                    end else if (groups_q == '0 || k_q == '0 ||
                                 b_addr[2:0] != 3'b0 || p_addr[2:0] != 3'b0) begin
                        f_err_code <= ((groups_q == '0 || k_q == '0) ? E_DIM : 4'b0) |
                                      ((b_addr[2:0] != 3'b0 || p_addr[2:0] != 3'b0) ? E_ALIGN : 4'b0);
                        f_err <= 1'b1;
                        state <= S_IDLE;
                    end else begin
                        state <= S_GRP;
                    end
                end

                // a new output group: both of its destinations must fit
                S_GRP: begin
                    if (aborted || abort) begin
                        state <= S_IDLE;
                    end else if (RB_W'(b_ptr) + RB_W'(k_q) > RB_W'(1) << B_ADDR_W ||
                                 RP_W'(p_ptr) + RP_W'(N) > RP_W'(1) << P_ADDR_W) begin
                        f_err_code <= E_RANGE;
                        f_err      <= 1'b1;
                        state      <= S_IDLE;
                    end else begin
                        phase_p <= 1'b0;
                        left    <= k_q;
                        state   <= S_SEG;
                    end
                end

                S_SEG: begin
                    if (aborted || abort) begin
                        state <= S_IDLE;
                    end else begin
                        araddr  <= seg_addr;
                        arlen   <= 8'(seg_beats - 1'b1);
                        beats   <= 9'(seg_beats);
                        got     <= '0;
                        left    <= left - seg_beats;
                        arvalid <= 1'b1;
                        state   <= S_AR;
                    end
                end

                // ARVALID stays up until the handshake, abort or not
                S_AR: begin
                    if (arready) begin
                        arvalid <= 1'b0;
                        state   <= S_R;
                    end
                end

                S_R: begin
                    if (rvalid) begin
                        got <= got + 1'b1;
                        if (rresp != 2'b00 || rid != '0) resp_bad <= 1'b1;
                        if (!aborted && !abort && !resp_bad && rresp == 2'b00 && rid == '0) begin
                            if (!phase_p) begin
                                b_we    <= 1'b1;
                                b_waddr <= b_ptr[B_ADDR_W-1:0];
                                b_wdata <= rdata[N*8-1:0];
                            end else begin
                                p_we    <= 1'b1;
                                p_waddr <= p_ptr[P_ADDR_W-1:0];
                                p_wdata <= rdata[P_WIDTH-1:0];
                            end
                        end
                        if (!phase_p) begin b_ptr <= b_ptr + 1'b1; b_addr <= b_addr + AXI_AW'(8); end
                        else          begin p_ptr <= p_ptr + 1'b1; p_addr <= p_addr + AXI_AW'(8); end
                        if (rlast) begin
                            if (aborted || abort) begin
                                state <= S_IDLE;
                            end else if (resp_bad || rresp != 2'b00 || rid != '0) begin
                                f_err_code <= E_RESP;
                                f_err      <= 1'b1;
                                state      <= S_IDLE;
                            end else if (left != '0) begin
                                state <= S_SEG;               // the rest of this B or P
                            end else if (!phase_p) begin
                                phase_p <= 1'b1;              // this group's P
                                left    <= K_WIDTH'(N);
                                state   <= S_SEG;
                            end else if (og == groups_q - 1'b1) begin
                                f_done <= 1'b1;
                                state  <= S_IDLE;
                            end else begin
                                og    <= og + 1'b1;
                                state <= S_GRP;
                            end
                        end
                    end
                end

                S_HW: begin
                    if (awvalid && awready) begin awvalid <= 1'b0; aw_done <= 1'b1; end
                    if (wvalid && wready)   begin wvalid  <= 1'b0; w_done  <= 1'b1; end
                    if ((aw_done || (awvalid && awready)) && (w_done || (wvalid && wready)))
                        state <= S_HB;
                end

                S_HB: begin
                    if (bvalid) begin
                        h_resp <= (bid != '0) ? 2'b10 : bresp;
                        h_ack  <= 1'b1;
                        state  <= S_IDLE;
                    end
                end

                S_HAR: begin
                    if (arready) begin
                        arvalid <= 1'b0;
                        state   <= S_HR;
                    end
                end

                S_HR: begin
                    if (rvalid) begin
                        h_rdata <= h_hi ? rdata[63:32] : rdata[31:0];
                        h_resp  <= (rid != '0) ? 2'b10 : rresp;
                        h_ack   <= 1'b1;
                        state   <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    // the slave must end every burst exactly on its last beat
    always @(posedge clk) begin
        if (!reset && rvalid && rready) begin
            if (state == S_R && (rlast != (got == beats - 1'b1)))
                $fatal(1, "wdma: RLAST on beat %0d of a %0d-beat burst", got + 1, beats);
            if (state == S_HR && !rlast)
                $fatal(1, "wdma: single-beat read without RLAST");
        end
    end
`endif

endmodule
