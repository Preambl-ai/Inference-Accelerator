// ddr_port: where the MIG goes. top connects wdma's AXI4 master here when
// HAS_DMA = 1.
//
// simulation: this module drives nothing. The cocotb AXI4 slave model
// (dma/tb/axi4_slave.py) drives the slave-side outputs below through this
// instance's hierarchy (top.u_ddr.arready, ...) and reads the master's
// requests from its inputs, so top keeps the Arty's pins and nothing else.
//
// synthesis: replace this module with the MIG 7-series IP generated for the
// Arty A7's DDR3 (AXI4 slave, 64-bit data, ID width AXI_IDW), plus its DDR3
// pins and clocks in top and the XDC. fpga/README.md ("DDR3 through the
// MIG") lists the steps. Until then a HAS_DMA = 1 synthesis stops here on
// purpose rather than build a board with no memory behind wdma.

module ddr_port #(
    parameter int AXI_IDW = 4
)(
    input  logic clk,
    input  logic reset,
    input  logic [31:0] araddr,
    input  logic [7:0]  arlen,
    input  logic [2:0]  arsize,
    input  logic [1:0]  arburst,
    input  logic [AXI_IDW-1:0] arid,
    input  logic        arvalid,
    output logic        arready,
    output logic [63:0] rdata,
    output logic [1:0]  rresp,
    output logic        rlast,
    output logic [AXI_IDW-1:0] rid,
    output logic        rvalid,
    input  logic        rready,
    input  logic [31:0] awaddr,
    input  logic [7:0]  awlen,
    input  logic [2:0]  awsize,
    input  logic [1:0]  awburst,
    input  logic [AXI_IDW-1:0] awid,
    input  logic        awvalid,
    output logic        awready,
    input  logic [63:0] wdata,
    input  logic [7:0]  wstrb,
    input  logic        wlast,
    input  logic        wvalid,
    output logic        wready,
    output logic [1:0]  bresp,
    output logic [AXI_IDW-1:0] bid,
    output logic        bvalid,
    input  logic        bready
);

`ifdef SYNTHESIS
    initial $fatal(1, "ddr_port: HAS_DMA = 1 needs the MIG in place of this stub (fpga/README.md)");
`endif

endmodule
