// uart_bridge: UART packets in, AXI4-Lite master out, one response per packet.
//
// packet format, status codes and error rules: top/host_interface_contract.md
// section 1. in short:
//   host:  A5 OP ADDR[4] LEN HCRC[2] [DATA[4*(LEN+1)] DCRC[2]]
//   board: 5A [DATA[4*(LEN+1)] for reads] STATUS RCRC[2]
//   CRC-16/CCITT-FALSE, little-endian fields.
//
// the header is checked before anything reaches the bus. write32 buffers its
// word and writes only after DCRC passes; a burst write writes each word as
// soon as its 4 bytes are in (no on-chip burst buffer) and checks DCRC at the
// end, so a bad burst can only have touched its own declared range.
//
// a byte can arrive while a bus transfer is in progress. the bus side takes a
// handful of cycles and a byte takes 10 bit times, so one holding register
// (byte_q / have_byte) is enough; simulation $fatals if a second byte ever
// lands on top of an unconsumed one.
//
// timeout: TIMEOUT_BITS bit times without a byte in the middle of a packet
// discards it and answers 0x06. after any error answer the bridge drains:
// it drops bytes until the line has been quiet for the same timeout, so the
// tail of a broken packet is never parsed as a new one.
//
// link_err pulses on every HCRC / DCRC / opcode / timeout error; the CSR uses
// it to clear its half-staged wide words.

module uart_bridge #(
    parameter int CLKS_PER_BIT = 100,
    parameter int TIMEOUT_BITS = 1000,
    parameter int ADDR_W       = 24
)(
    input  logic clk,
    input  logic reset,

    // from uart_rx
    input  logic [7:0] rx_data,
    input  logic       rx_valid,
    // to uart_tx
    output logic [7:0] tx_data,
    output logic       tx_valid,
    input  logic       tx_ready,

    output logic       link_err,

    // AXI4-Lite master
    output logic [ADDR_W-1:0] m_awaddr,
    output logic              m_awvalid,
    input  logic              m_awready,
    output logic [31:0]       m_wdata,
    output logic [3:0]        m_wstrb,
    output logic              m_wvalid,
    input  logic              m_wready,
    input  logic [1:0]        m_bresp,
    input  logic              m_bvalid,
    output logic              m_bready,
    output logic [ADDR_W-1:0] m_araddr,
    output logic              m_arvalid,
    input  logic              m_arready,
    input  logic [31:0]       m_rdata,
    input  logic [1:0]        m_rresp,
    input  logic              m_rvalid,
    output logic              m_rready
);

    localparam logic [7:0] SYNC_IN = 8'hA5, SYNC_OUT = 8'h5A;
    localparam logic [7:0] OP_WR = 8'h01, OP_RD = 8'h02, OP_BWR = 8'h03, OP_BRD = 8'h04;
    localparam logic [7:0] ST_OK = 8'h00, ST_SLVERR = 8'h01, ST_DECERR = 8'h02,
                           ST_HCRC = 8'h03, ST_DCRC = 8'h04, ST_BADOP = 8'h05,
                           ST_TIMEOUT = 8'h06;
    localparam int TIMEOUT_CLKS = TIMEOUT_BITS * CLKS_PER_BIT;
    localparam int TO_W = $clog2(TIMEOUT_CLKS + 1);

    // CRC-16/CCITT-FALSE, one byte. the reference is host/protocol.py crc16;
    // the bridge TB checks this against it on every packet.
    function automatic logic [15:0] crc16_byte(input logic [15:0] c, input logic [7:0] b);
        logic [15:0] x;
        x = c ^ {b, 8'h00};
        for (int i = 0; i < 8; i++)
            x = x[15] ? ((x << 1) ^ 16'h1021) : (x << 1);
        return x;
    endfunction

    typedef enum logic [4:0] {
        S_SYNC, S_HDR, S_HCRC, S_DATA, S_DCRC, S_BUS_W, S_BUS_R,
        S_R_SYNC, S_R_DATA, S_R_STATUS, S_R_CRC0, S_R_CRC1, S_DRAIN
    } state_t;
    state_t state;

    // one-byte holding register between uart_rx and the parser
    logic [7:0] byte_q;
    logic have_byte, take;
    logic [TO_W-1:0] idle_cnt;          // clocks since the last received byte
    logic timed_out;

    assign timed_out = (idle_cnt == TO_W'(TIMEOUT_CLKS));

    // packet fields
    logic [7:0]  op;
    logic [31:0] addr;
    logic [7:0]  len;
    logic [2:0]  hdr_idx;                // 0..5 = OP, ADDR0..3, LEN
    logic        crc_idx;                // which CRC byte
    logic [15:0] crc, crc_rx;
    logic [1:0]  byte_idx;               // byte within the current word
    logic [8:0]  word_idx;               // word within the packet
    logic [31:0] word;
    logic [7:0]  status;
    logic        stopped;                // a bus error stopped the burst
    logic        drain_after;            // drain once the answer is out
    logic        is_read;

    logic aw_done, w_done, ar_done;

    // while an answer is going out (and during the bus reads that feed a
    // burst read answer) the host is not supposed to send; anything that
    // arrives then is dropped, and that packet times out on the host side
    logic answering;
    assign answering = (state == S_BUS_R) || (state == S_R_SYNC) || (state == S_R_DATA) ||
                       (state == S_R_STATUS) || (state == S_R_CRC0) || (state == S_R_CRC1);

    // the parser consumes the held byte in the states that read bytes
    assign take = have_byte && (state == S_SYNC || state == S_HDR || state == S_HCRC ||
                                state == S_DATA || state == S_DCRC || state == S_DRAIN);

    // address of the current word, and whether it fits the AXI address
    logic [31:0] cur_addr;
    logic addr_ok;
    assign cur_addr = addr + {word_idx, 2'b00};
    assign addr_ok  = (cur_addr >> ADDR_W) == 0;

    assign m_awaddr = cur_addr[ADDR_W-1:0];
    assign m_araddr = cur_addr[ADDR_W-1:0];
    assign m_wdata  = word;
    assign m_wstrb  = 4'hF;
    assign m_bready = (state == S_BUS_W);
    assign m_rready = (state == S_BUS_R);

    // response bytes come straight from the state
    always_comb begin
        tx_valid = 1'b0;
        tx_data  = 8'h00;
        case (state)
            S_R_SYNC:   begin tx_valid = 1'b1; tx_data = SYNC_OUT; end
            S_R_DATA:   begin tx_valid = 1'b1; tx_data = word[byte_idx*8 +: 8]; end
            S_R_STATUS: begin tx_valid = 1'b1; tx_data = status; end
            S_R_CRC0:   begin tx_valid = 1'b1; tx_data = crc[7:0]; end
            S_R_CRC1:   begin tx_valid = 1'b1; tx_data = crc[15:8]; end
            default: ;
        endcase
    end

    logic tx_go;
    assign tx_go = tx_valid && tx_ready;

    // a response with no data: straight to the status byte after the sync
    task automatic answer(input logic [7:0] st, input logic drain, input logic err);
        status      <= st;
        drain_after <= drain;
        is_read     <= 1'b0;
        link_err    <= err;
        state       <= S_R_SYNC;
    endtask

    always_ff @(posedge clk) begin
        if (reset) begin
            state       <= S_SYNC;
            have_byte   <= 1'b0;
            byte_q      <= '0;
            idle_cnt    <= '0;
            link_err    <= 1'b0;
            m_awvalid   <= 1'b0;
            m_wvalid    <= 1'b0;
            m_arvalid   <= 1'b0;
            aw_done     <= 1'b0;
            w_done      <= 1'b0;
            ar_done     <= 1'b0;
            op          <= '0;
            addr        <= '0;
            len         <= '0;
            hdr_idx     <= '0;
            crc_idx     <= 1'b0;
            crc         <= 16'hFFFF;
            crc_rx      <= '0;
            byte_idx    <= '0;
            word_idx    <= '0;
            word        <= '0;
            status      <= ST_OK;
            stopped     <= 1'b0;
            drain_after <= 1'b0;
            is_read     <= 1'b0;
        end else begin
            link_err <= 1'b0;

            // holding register and the idle timer
            if (rx_valid) begin
                if (!answering) begin
                    byte_q    <= rx_data;
                    have_byte <= 1'b1;
                end
                idle_cnt <= '0;
            end else begin
                if (take) have_byte <= 1'b0;
                if (!timed_out) idle_cnt <= idle_cnt + 1'b1;
            end

            case (state)

                S_SYNC: begin
                    if (take && byte_q == SYNC_IN) begin
                        hdr_idx <= '0;
                        crc     <= 16'hFFFF;
                        state   <= S_HDR;
                    end
                end

                S_HDR: begin
                    if (take) begin
                        crc <= crc16_byte(crc, byte_q);
                        case (hdr_idx)
                            3'd0: op <= byte_q;
                            3'd1: addr[7:0]   <= byte_q;
                            3'd2: addr[15:8]  <= byte_q;
                            3'd3: addr[23:16] <= byte_q;
                            3'd4: addr[31:24] <= byte_q;
                            default: len <= byte_q;
                        endcase
                        if (hdr_idx == 3'd5) begin
                            crc_idx <= 1'b0;
                            state   <= S_HCRC;
                        end else begin
                            hdr_idx <= hdr_idx + 1'b1;
                        end
                    end else if (timed_out) begin
                        answer(ST_TIMEOUT, 1'b0, 1'b1);
                    end
                end

                S_HCRC: begin
                    if (take) begin
                        if (!crc_idx) begin
                            crc_rx[7:0] <= byte_q;
                            crc_idx     <= 1'b1;
                        end else if ({byte_q, crc_rx[7:0]} != crc) begin
                            answer(ST_HCRC, 1'b1, 1'b1);
                        end else if (!(op == OP_WR || op == OP_RD || op == OP_BWR || op == OP_BRD) ||
                                     ((op == OP_WR || op == OP_RD) && len != 0)) begin
                            answer(ST_BADOP, 1'b1, 1'b1);
                        end else begin
                            // header is good: from here address and length are trusted
                            word_idx <= '0;
                            byte_idx <= '0;
                            stopped  <= 1'b0;
                            status   <= ST_OK;
                            crc      <= 16'hFFFF;     // DCRC, or the response CRC for reads
                            if (op == OP_RD || op == OP_BRD) begin
                                is_read     <= 1'b1;
                                drain_after <= 1'b0;
                                state       <= S_R_SYNC;
                            end else begin
                                state <= S_DATA;
                            end
                        end
                    end else if (timed_out) begin
                        answer(ST_TIMEOUT, 1'b0, 1'b1);
                    end
                end

                S_DATA: begin
                    if (take) begin
                        crc <= crc16_byte(crc, byte_q);
                        word[byte_idx*8 +: 8] <= byte_q;
                        byte_idx <= byte_idx + 1'b1;
                        if (byte_idx == 2'd3) begin
                            if (op == OP_BWR && !stopped) begin
                                state <= S_BUS_W;          // write this word now
                            end else if ({1'b0, word_idx} == {1'b0, len}) begin
                                crc_idx <= 1'b0;
                                state   <= S_DCRC;
                            end else begin
                                word_idx <= word_idx + 1'b1;   // stopped burst: just count
                            end
                        end
                    end else if (timed_out) begin
                        answer(ST_TIMEOUT, 1'b0, 1'b1);
                    end
                end

                S_DCRC: begin
                    if (take) begin
                        if (!crc_idx) begin
                            crc_rx[7:0] <= byte_q;
                            crc_idx     <= 1'b1;
                        end else if ({byte_q, crc_rx[7:0]} != crc) begin
                            answer(ST_DCRC, 1'b1, 1'b1);
                        end else if (op == OP_WR) begin
                            state <= S_BUS_W;               // write32: only now
                        end else begin
                            answer(status, 1'b0, 1'b0);     // burst: OK or the bus error
                        end
                    end else if (timed_out) begin
                        answer(ST_TIMEOUT, 1'b0, 1'b1);
                    end
                end

                // one AXI write. AW and W are offered together and may be
                // accepted in either order.
                S_BUS_W: begin
                    if (!addr_ok) begin
                        // above the AXI address range: no bus cycle at all
                        status  <= ST_DECERR;
                        stopped <= 1'b1;
                        if (op == OP_WR) begin
                            answer(ST_DECERR, 1'b0, 1'b0);
                        end else if (word_idx == {1'b0, len}) begin
                            crc_idx <= 1'b0;
                            state   <= S_DCRC;
                        end else begin
                            // a stopped burst still reads (and CRCs) its remaining bytes
                            word_idx <= word_idx + 1'b1;
                            state    <= S_DATA;
                        end
                    end else begin
                        if (!aw_done && !m_awvalid) m_awvalid <= 1'b1;
                        if (!w_done  && !m_wvalid)  m_wvalid  <= 1'b1;
                        if (m_awvalid && m_awready) begin m_awvalid <= 1'b0; aw_done <= 1'b1; end
                        if (m_wvalid  && m_wready)  begin m_wvalid  <= 1'b0; w_done  <= 1'b1; end
                        if (m_bvalid) begin
                            aw_done <= 1'b0;
                            w_done  <= 1'b0;
                            if (m_bresp != 2'b00) begin
                                status  <= (m_bresp == 2'b11) ? ST_DECERR : ST_SLVERR;
                                stopped <= 1'b1;
                            end
                            if (op == OP_WR) begin
                                answer((m_bresp == 2'b00) ? ST_OK :
                                       (m_bresp == 2'b11) ? ST_DECERR : ST_SLVERR, 1'b0, 1'b0);
                            end else if (word_idx == {1'b0, len}) begin
                                crc_idx <= 1'b0;
                                state   <= S_DCRC;
                            end else begin
                                word_idx <= word_idx + 1'b1;
                                state    <= S_DATA;
                            end
                        end
                    end
                end

                // one AXI read, then its 4 bytes go out
                S_BUS_R: begin
                    if (stopped || !addr_ok) begin
                        if (!stopped) begin status <= ST_DECERR; stopped <= 1'b1; end
                        word     <= '0;
                        byte_idx <= '0;
                        state    <= S_R_DATA;
                    end else begin
                        if (!ar_done && !m_arvalid) m_arvalid <= 1'b1;
                        if (m_arvalid && m_arready) begin m_arvalid <= 1'b0; ar_done <= 1'b1; end
                        if (m_rvalid) begin
                            ar_done  <= 1'b0;
                            byte_idx <= '0;
                            if (m_rresp != 2'b00) begin
                                status  <= (m_rresp == 2'b11) ? ST_DECERR : ST_SLVERR;
                                stopped <= 1'b1;
                                word    <= '0;
                            end else begin
                                word <= m_rdata;
                            end
                            state <= S_R_DATA;
                        end
                    end
                end

                S_R_SYNC: begin
                    if (tx_go) begin
                        crc <= 16'hFFFF;
                        if (is_read) state <= S_BUS_R; else state <= S_R_STATUS;
                    end
                end

                S_R_DATA: begin
                    if (tx_go) begin
                        crc      <= crc16_byte(crc, tx_data);
                        byte_idx <= byte_idx + 1'b1;
                        if (byte_idx == 2'd3) begin
                            if (word_idx == {1'b0, len}) begin
                                state <= S_R_STATUS;
                            end else begin
                                word_idx <= word_idx + 1'b1;
                                state    <= S_BUS_R;
                            end
                        end
                    end
                end

                S_R_STATUS: if (tx_go) begin crc <= crc16_byte(crc, status); state <= S_R_CRC0; end
                S_R_CRC0:   if (tx_go) state <= S_R_CRC1;
                S_R_CRC1:   if (tx_go) begin if (drain_after) state <= S_DRAIN; else state <= S_SYNC; end

                // drop everything until the line has been quiet for the timeout
                S_DRAIN: begin
                    if (timed_out && !have_byte && !rx_valid) state <= S_SYNC;
                end

                default: state <= S_SYNC;
            endcase
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (!reset && rx_valid && !answering && have_byte && !take)
            $fatal(1, "uart_bridge: a byte arrived before the previous one was consumed");
    end
`endif

endmodule
