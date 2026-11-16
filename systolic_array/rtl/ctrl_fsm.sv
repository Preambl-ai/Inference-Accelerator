//ctrl_fsm is the tile sequencer for the NxN output stationary array. 

// this block owns the timing, not the data. it omits a slice index 'k_idx' and the 
//feeder supplies the operand bytes MEM_LATENCY cycles later. 
//the tile tags travel thru delay lines of the same depth so that they arrive at the array boundry on the same 
//edge as the operands they tag. 
//
// interface
//   start/k_in/ready : one tile per handshake. a tile is accepted on an edge
//                      where start & ready. ready is low under stall, so a
//                      start pulsed during a stall is not taken; hold start
//                      until ready. k_in is the tile depth K (>= 1). a start
//                      with k_in == 0 is ignored so hardware cannot hang.
//   busy             : high from the edge that accepts a tile until the edge
//                      after done. not gated by stall.
//   op_req/k_idx     : slice request to the feeder, k_idx = 0 .. K-1, one per
//                      enabled cycle. the operands for a slice must reach the
//                      array MEM_LATENCY enabled edges after it is requested,
//                      and the feeder pipeline must hold under stall.
//   arr_*            : drive systolic_array directly. arr_en = ~stall.
//   c_valid/c_row    : c_out of the array holds row c_row. rows come out
//                      N-1 first, then N-2 .. 0, one per enabled cycle.
//   done             : one enabled cycle, the cycle after row 0.
//
// stall convention: stall freezes every register here and in the array.
//   outputs hold their value, so an event (op_req, c_valid, done) only counts
//   on a cycle where stall is low. done and c_valid stretch under stall;
//   consumers qualify them with ~stall.
//
// latency, in enabled edges, from the edge that accepts start:
//   first slice sampled by the array : MEM_LATENCY + 1
//   last slice sampled by the array  : MEM_LATENCY + K
//   row N-1 on c_out, c_valid rises  : MEM_LATENCY + K + 2N (LAST_TO_DRAIN_READY
//                                      after the last slice is sampled)
//   done rises                       : MEM_LATENCY + K + 3N
//   ready rises again                : MEM_LATENCY + K + 3N + 1
// an output that rises after edge e is sampled by its consumer at edge e + 1.
//
// reset is synchronous and wins over stall. it clears the tag delay lines too,
// so a tile cut off by reset leaves nothing behind (the array shares the reset).

module ctrl_fsm #( 
    parameter int N = 8, 
    parameter int K_WIDTH = 16, 
    parameter int MEM_LATENCY = 1,
    parameter int ROW_W = (N > 1) ? $clog2(N) : 1
)( 
    input logic clk, 
    input logic reset,    //synchronous


    input  logic start,
    input  logic [K_WIDTH-1:0] k_in,
    output logic ready,
    output logic busy,
    output logic done,
 
    // operand feeder side
    input  logic stall,
    output logic op_req,
    output logic [K_WIDTH-1:0] k_idx,
 
    // systolic array side
    output logic arr_en,
    output logic arr_first,
    output logic arr_last,
    output logic arr_drain_shift,
 
    // result side
    output logic c_valid,
    output logic [ROW_W-1:0] c_row
    ); 

    // constants 
    localparam int WAIT_CYCLES = 2 * N;
    localparam int WAIT_W =  $clog2(WAIT_CYCLES + 1);
    localparam logic [2:0] S_IDLE   = 3'd0,
                           S_STREAM = 3'd1,
                           S_FLUSH  = 3'd2,
                           S_DRAIN  = 3'd3,
                           S_DONE   = 3'd4;
    logic [2:0] state; 
    logic [K_WIDTH - 1: 0] k_lat;  // tile depth 
    logic raw_first, raw_last;
    logic first_dly, last_dly;

    logic last_seen; //delayed last tag reach the array 
    logic [WAIT_W-1:0] wait_cnt; 

    //tag alignment 
    delay #(.DATA_WIDTH(1), .DEPTH(MEM_LATENCY)) u_first_dly (
        .clk      (clk),
        .reset    (reset),
        .en       (~stall),
        .in_data  (raw_first),
        .out_data (first_dly)
    );
    delay #(.DATA_WIDTH(1), .DEPTH(MEM_LATENCY)) u_last_dly (
        .clk      (clk),
        .reset    (reset),
        .en       (~stall),
        .in_data  (raw_last),
        .out_data (last_dly)
    );

    assign arr_first = first_dly;
    assign arr_last  = last_dly;

    //combinational outputs 
    assign arr_en = ~stall;

    // ready drops under stall so start & ready is the whole handshake
    assign ready = (state == S_IDLE) & ~stall;
    assign busy  = (state != S_IDLE);

    // Sequencer
    always_ff @(posedge clk) begin
        if (reset) begin
            state           <= S_IDLE;
            k_lat           <= '0;
            k_idx           <= '0;
            op_req          <= 1'b0;
            raw_first       <= 1'b0;
            raw_last        <= 1'b0;
            last_seen       <= 1'b0;
            wait_cnt        <= '0;
            arr_drain_shift <= 1'b0;
            c_valid         <= 1'b0;
            c_row           <= '0;
            done            <= 1'b0;
        end else if (!stall) begin
            // Wavefront timer.
            if (state != S_IDLE && last_dly) begin
                last_seen <= 1'b1;
                wait_cnt  <= '0;
            end else if (last_seen) begin
                wait_cnt <= wait_cnt + 1'b1;
            end
 
            done <= 1'b0;   // default: done is a one-cycle pulse
 
            case (state)
 
                S_IDLE: begin
                    // start is ignored while busy, by construction: this arm is
                    // only reachable from S_IDLE.
                    // k_in == 0 would never hit the k_idx == k_lat - 1 compare
                    if (start && (k_in != '0)) begin
                        k_lat     <= k_in;
                        k_idx     <= '0;
                        op_req    <= 1'b1;
                        raw_first <= 1'b1;
                        // K == 1 makes first and last coincide. pe.sv handles
                        // that with active <= (first_in | active) & ~last_in.
                        raw_last  <= (k_in == 1);
                        state     <= S_STREAM;
                    end
                end
 
                S_STREAM: begin
                    raw_first <= 1'b0;
                    // k_idx currently on the bus is the one being consumed this
                    // edge. Comparing it (rather than the post-increment value)
                    // is what makes K == 1 occupy exactly one cycle.
                    if (k_idx == k_lat - 1) begin
                        op_req   <= 1'b0;
                        raw_last <= 1'b0;
                        state    <= S_FLUSH;
                    end else begin
                        k_idx    <= k_idx + 1'b1;
                        raw_last <= ((k_idx + 1'b1) == k_lat - 1);
                    end
                end
 
                S_FLUSH: begin
                    if (last_seen && (wait_cnt == WAIT_CYCLES - 1)) begin
                        state           <= S_DRAIN;
                        c_valid         <= 1'b1;
                        c_row           <= N - 1;       // truncates to ROW_W
                        arr_drain_shift <= (N > 1);
                        last_seen       <= 1'b0;
                        wait_cnt        <= '0;
                    end
                end
 
                S_DRAIN: begin

                    if (c_row == '0) begin
                        c_valid         <= 1'b0;
                        arr_drain_shift <= 1'b0;
                        done            <= 1'b1;
                        state           <= S_DONE;
                    end else begin
                        c_row           <= c_row - 1'b1;
                        arr_drain_shift <= ((c_row - 1'b1) != '0);
                    end
                end
 
                S_DONE: begin
                    state <= S_IDLE;
                end
 
                default: state <= S_IDLE;
 
            endcase
        end
    end
 
    // Simulation-only checks
`ifndef SYNTHESIS
    initial begin
        if (N < 1)
            $fatal(1, "ctrl_fsm: N must be >= 1 (got %0d)", N);
        if (MEM_LATENCY < 0)
            $fatal(1, "ctrl_fsm: MEM_LATENCY must be >= 0 (got %0d)", MEM_LATENCY);
        if (K_WIDTH < 1)
            $fatal(1, "ctrl_fsm: K_WIDTH must be >= 1 (got %0d)", K_WIDTH);
    end
    // +ctrl_fsm_allow_k0 lets the testbench prove the k_in == 0 start is ignored
    always @(posedge clk) begin
        if (!reset && !stall && (state == S_IDLE) && start && (k_in == 0)
            && !$test$plusargs("ctrl_fsm_allow_k0"))
            $fatal(1, "ctrl_fsm: k_in must be >= 1");
    end
`endif
 
endmodule

