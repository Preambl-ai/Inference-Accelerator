# act_buffer contract

`memory/rtl/act_buffer.sv` is NBUF activation memories ("banks") built from
`bram.sv`. It has one shared read port, one datapath write port, and a host
write port that takes over the write side between layers. Every timing figure
below was measured on the RTL in Icarus 13 (NBUF 1..4, READ_LATENCY 1 and 2),
not worked out on paper.

## Parameters

| name         | default | legal                 | meaning                                       |
| ------------ | ------- | --------------------- | --------------------------------------------- |
| N            | 8       | >= 1                  | lanes per word                                |
| DATA_WIDTH   | 8       | >= 1                  | bits per lane                                 |
| ADDR_WIDTH   | 8       | >= 1                  | word address width; each bank has 2**ADDR_WIDTH words |
| READ_LATENCY | 1       | 1 or 2                | passed to bram (2 enables the BRAM output register) |
| NBUF         | 3       | >= 1 (see edge case a) | number of banks                              |
| SEL_WIDTH    | derived | do not override       | bank select width                             |

Word width W = N * DATA_WIDTH.

## Ports

| port                          | dir | width      | notes |
| ----------------------------- | --- | ---------- | ----- |
| clk                           | in  | 1          | there is no reset (see edge case c) |
| rd_en                         | in  | 1          | clock enable for the whole read path |
| rd_addr                       | in  | ADDR_WIDTH | |
| rd_sel                        | in  | SEL_WIDTH  | bank to read; must be < NBUF when the read advances |
| rd_data                       | out | W          | registered (never combinational from rd_*) |
| we, waddr, wdata, wr_sel      | in  | 1/AW/W/SW  | datapath write; ignored while host_mode = 1 |
| host_mode                     | in  | 1          | 1 = the host owns the write port and reads are frozen |
| host_we, host_waddr, host_wdata, host_wr_sel | in | 1/AW/W/SW | host write; ignored while host_mode = 0 |

## Packing and signedness

Lane i of a word is `word[i*DATA_WIDTH +: DATA_WIDTH]`, so lane 0 sits in the
LSBs. This matches `im2col_reader` (it reads byte j of `rd_data` as channel
`g*N + j`) and `gemm_tile`'s `a_data` (byte r is pixel r). The buffer only
stores bits and does no arithmetic. By convention every lane holds a two's-complement value of
`DATA_WIDTH` bits (the int8 activations and the `pad_value` zero point). Lanes
are independent, so no carry or sign extension crosses a lane boundary.

## Reads

* A read **advances** on a rising edge when `read_advance = rd_en & !host_mode`.
  The edge samples `rd_addr` and `rd_sel` for the advance.
* `rd_data` shows the word at `{rd_sel, rd_addr}` exactly **READ_LATENCY
  advancing edges** after the edge that sampled it. This was measured as 1 and 2
  advancing edges for READ_LATENCY 1 and 2, and it holds for every NBUF.
* Every bank reads `rd_addr` on each advance. Each bank select travels through
  `sel_pipe`, a READ_LATENCY-deep shift register that has the same enable as the
  bram read registers. The select therefore reaches the output mux on the same
  edge as its data. `rd_sel` may change on every advancing cycle; this was
  measured with a new bank every cycle at READ_LATENCY 2.
* When a read does not advance (`rd_en = 0` or `host_mode = 1`), the following
  all hold: `rd_data`, the word in flight in the READ_LATENCY=2 output register,
  and `sel_pipe`. After a stall the next advance delivers the word that was in
  flight, so a stall neither drops nor repeats a word.
* Only one bank is read per cycle, because the read port is shared.

## Writes and host_mode arbitration

* The write side has a single physical port with a two-way mux on `host_mode`:
  * `host_mode = 0`: `we/waddr/wdata/wr_sel` write and the `host_*` inputs are ignored.
  * `host_mode = 1`: `host_we/host_waddr/host_wdata/host_wr_sel` write and the
    datapath `we/waddr/wdata/wr_sel` inputs are ignored. Reads are frozen as described above.
* A write lands on the rising edge: bank `sel` at `waddr` takes `wdata`.
  Exactly one bank is written. Writes are never gated by `rd_en`.
* `host_mode` acts combinationally on the same edge: the edge that samples
  `host_mode = 1` already takes the host write and freezes the read.
* **System constraint:** the buffer freezes its own read pipeline in host_mode,
  but the reader on the other side of the port does not know that.
  `im2col_reader` keeps `rd_en` high for a whole tile, and its tag delay runs
  freely. If `host_mode` rises mid-tile, the reader's tags and the buffer's data
  fall out of step. The layer controller must therefore raise `host_mode` only
  while no consumer has a read in flight, which means between layers.

## Edge cases (decisions)

### a. NBUF = 1

**Measured:** `SEL_WIDTH = $clog2(1) = 0` does not give zero-width ports. Instead,
`logic [SEL_WIDTH-1:0]` becomes `[-1:0]`, which is a **2-bit** vector. With
NBUF=1 the select ports are 2 bits wide, and the values 1, 2 and 3 all index a
bank that does not exist.

**Decision:** support NBUF = 1 and force the select width to at least 1, using
the same idiom as `im2col_reader`'s `ROW_W`:
`SEL_WIDTH = (NBUF > 1) ? $clog2(NBUF) : 1`. With one bank, sel = 1 becomes the
single out-of-range value and edge case b covers it. NBUF < 1 is an
elaboration-time `$fatal`. Overriding SEL_WIDTH is also a `$fatal`, following
the gemm_tile `P_WIDTH` rule. A wrong override would cut off select bits, or
widen them so they compare against nothing. The only interface change is that
NBUF=1 select ports go from 2 bits to 1. Nothing in the repo instantiates
act_buffer.

### b. Select out of range (sel >= NBUF)

This can only happen when NBUF is not a power of two (including NBUF = 1 after
fix a).

**Measured today:** a write with `wr_sel >= NBUF` matches no bank's
`write_sel == i`, so it is silently dropped and no bank changes. A read with
`rd_sel >= NBUF` returns `X`, because `bank_rdata` is indexed past its end. In
synthesis that index is tool-defined and usually aliases to some real bank.

**Decision:** it is **illegal**. A simulation-only `$fatal` fires on:
* an advancing read with `rd_sel >= NBUF`
* an effective write (after the host_mode mux) with `write_sel >= NBUF`

The ignored side of the mux is not checked, so a junk select on a port that is
not selected stays legal. That keeps it consistent with "idle ports may carry
anything", which the bram TB relies on. A plusarg, `+act_buffer_allow_bad_sel`
(`make ALLOW_BAD_SEL=1`), turns the check off so the testbench can prove the
hardware's behavior. That follows the ctrl_fsm/gemm_tile `allow_k0` pattern.
The defined hardware behavior that the TB checks:
* a bad-select write changes no bank
* a bad-select read gives undefined `rd_data` for its slot only (the TB does not check it)
* the reads before and after it come out correctly, because `sel_pipe` stays aligned

Defining the read as zero was considered and rejected. It would put a
comparator and an AND on the read data path, which is the timing path
READ_LATENCY=2 exists to protect, just to give meaning to an illegal input.

### c. No reset; rd_data before the first read

`sel_pipe` and the bram read registers have no reset. That matches bram: a block
RAM cannot be reset, and these registers do not need one. `rd_data` is
**undefined until READ_LATENCY advancing edges after the first advancing read**,
and after that it is always the word READ_LATENCY advances back. In simulation,
`rd_data` was measured as `X` before the first read. Bank contents are undefined
until written, so reading an unwritten word gives undefined data. The model
represents both as UNKNOWN, and the TB never checks them.

### d. Reads during host_mode

**Measured and confirmed:** while `host_mode = 1`, `rd_data` holds the last value
it showed, whatever `rd_en`, `rd_addr` and `rd_sel` do. A read requested during
host_mode is **dropped, not queued**: it never enters the pipeline. When
`host_mode` falls, the next advancing read continues from where the pipeline
froze. With READ_LATENCY=2, that means the word in flight comes out first.
Host writes made during host_mode are visible to reads issued after it, and
they include writes to the address and bank whose data is currently held on
`rd_data`. The held `rd_data` does not change, because it is a register.

### e. Read and write of the same bank and address on the same edge

This is inherited from `bram.sv`. When the read advances on the edge where the
same bank writes the same address, the read result is **undefined**: it is `X`
in simulation, and 7-series BRAM does not promise old or new data. The **write
always lands**, and the next read of that address returns the new data
(measured). Some combinations are **defined**:
* same address in a different bank (bram only compares inside one bank;
  measured, and the read returns the old contents of its own bank)
* same bank with a different address
* any write while the read does not advance, including every host-mode write,
  because host_mode forces the read enable low

## What the testbench checks

The cocotb TB compares `rd_data` with the golden model every cycle wherever the
model says the value is defined, sampling in the ReadOnly phase. Cases a and b
run in their own sweep configs: NBUF=1, and ALLOW_BAD_SEL=1 on NBUF=3 and
NBUF=1.
