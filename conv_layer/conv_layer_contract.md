# conv_layer contract

`conv_layer/rtl/conv_layer.sv` runs one complete INT8 convolution layer out of
on-chip memory:

    act_buffer bank src_sel -> im2col_reader -> gemm_tile -> write-back -> act_buffer bank dst_sel

It is the first integration block. It holds a layer sequencer, one
`im2col_reader`, one `gemm_tile`, one `act_buffer` and the write-back address
generator. This is the reference version: tiles run strictly one after another
(the next tile starts only after the previous tile's last output word has been
written). Every timing figure in this file is filled in from simulation, not
worked out on paper.

## Facts taken from the existing blocks (checked, not re-derived)

| fact | source |
| ---- | ------ |
| act_buffer word = N channels of one pixel, lane j = channel g*N + j, lane 0 in the LSBs | act_buffer.sv header, im2col_reader.sv header |
| input pixel (y, x), channel group g is at `src_base + (y*w_in + x)*cin_words + g` | im2col_reader.sv header |
| a tile is N consecutive output pixels in raster order from (y0, x0); x wraps at w_out | im2col_reader.sv, `win_order = 0` |
| slice k = ((ky*kw + kx)*cin_words + g)*N + j; K = kh*kw*cin_words*N | im2col_reader.sv header |
| spatial padding: every lane of a padded pixel is `pad_value`; the pixel is never read | im2col_reader.sv, `word[arr_r] <= arr_pad ? {N{padv_q}} : rd_data` |
| channel padding: im2col_reader does **not** pad channels; it reads whatever the host stored in the unused lanes | im2col_reader.sv ("padded up to a multiple of N by the host") |
| the reader walks pixels past the end of the output; their rows must be dropped | im2col_reader.sv header |
| `B` word `b_base + k` = row k of B, byte c = B[k][c] | gemm_tile.sv header |
| `P` word `p_base + c` = `{shift, mult, bias}`, bias in the LSBs | gemm_tile.sv, `p_rdata[0 +: BIAS_WIDTH]` etc. |
| gemm_tile computes `requant(sum_k A[r][k]*B[k][c] + bias[c])`; `zero_point` is the **output** zero point; the array has no input zero point | gemm_tile.sv, requant.sv |
| output beats come out rows N-1 down to 0; `tuser` = row r = pixel r of the tile; byte c = output channel c of the group; `tlast` on row 0 | gemm_tile.sv header |
| `k_in == 0` or `k_in > K_MAX` is ignored by gemm_tile and pulses `err_k` | gemm_tile.sv |

## Parameters

| name         | default | legal | meaning |
| ------------ | ------- | ----- | ------- |
| N            | 8       | >= 1  | lanes per word = tile rows = tile columns |
| DATA_WIDTH   | 8       | 8     | INT8 only; requant and the golden model assume it |
| ADDR_WIDTH   | 12      | >= 1  | act_buffer word address width, each bank is 2**ADDR_WIDTH words |
| READ_LATENCY | 1       | 1, 2  | shared by act_buffer and gemm_tile's B/P memories |
| NBUF         | 2       | >= 2  | act_buffer banks; src and dst must differ, so 2 is the minimum |
| K_MAX        | 1024    | as gemm_tile | |
| B_ADDR_W     | 13      | as gemm_tile | |
| P_ADDR_W     | 6       | `2**P_ADDR_W >= N` | |
| DIM_W        | 8       | as im2col_reader | image height and width fields |
| CW_W         | 7       | as im2col_reader | cin_words and cout_words fields |

The requant widths are gemm_tile's defaults (BIAS 32, M 16 unsigned, SHIFT 5).

## The layer command

Latched on the edge where `start & ready`. Nothing needs to be held after it.

| field | width | meaning |
| ----- | ----- | ------- |
| h_in, w_in | DIM_W | input image size in pixels |
| h_out, w_out | DIM_W | output size; must equal the conv formula (see E_SHAPE) |
| cin_words, cout_words | CW_W | channels / N, rounded up |
| kh, kw | 3 | kernel size, 1..7 (the port cannot carry more than 7) |
| stride | 2 | 1..3 |
| pad | 2 | 0..3, the same on all four sides |
| zp_in | 8 signed | input zero point. It is im2col_reader's `pad_value` |
| zp_out | 8 signed | output zero point. It is gemm_tile's `zero_point` |
| relu_en | 1 | gemm_tile's `relu_en` |
| src_sel, dst_sel | SEL_WIDTH | act_buffer banks read and written |
| src_base, dst_base | ADDR_WIDTH | word address of pixel (0,0), group 0 in each bank |
| b_base | B_ADDR_W | B word of row 0 of output-channel group 0 |
| p_base | P_ADDR_W | P word of output channel 0 |

### B/P base per output-channel group

    b_base(og) = b_base + og*K        K = kh*kw*cin_words*N
    p_base(og) = p_base + og*N

The groups are packed densely, one after another, with no per-group table.
Reasons:
* One latched base and one adder per memory. The sequencer adds K (or N) when
  it moves to the next group, and K is already computed for gemm_tile's `k_in`.
* The layout is the natural "weights in OIHW order, cut into N-column groups"
  that the host packer writes, so the host and RTL cannot disagree about it
  beyond one base address.
* Several layers can stay in B/P at once at different `b_base`/`p_base`. Two
  layers can then run back to back without reloading weights in between; the
  ping-pong test relies on this.
* The fit check is a single compare: `b_base + cout_words*K <= 2**B_ADDR_W`.

### Zero points: who handles zp_in

The array multiplies raw int8 values. Nothing in gemm_tile subtracts the input
zero point, and im2col_reader only uses `zp_in` as the padding value. The
quantized conv is

    acc[c] = sum_k (x[k] - zp_in) * w[k][c] + bias[c]
           = sum_k x[k]*w[k][c]  +  (bias[c] - zp_in * sum_k w[k][c])

so the **host folds the input zero point into the bias** that it stores in P:
`bias_hw[c] = bias[c] - zp_in * sum_k w[k][c]`. This is exact integer
arithmetic. The hardware sum `acc + bias_hw` equals the true sum exactly, so
requant's 32-bit saturation happens at the same values too. Padded pixels hold
`zp_in` and so contribute zero to the quantized conv, which is why `pad_value`
must be `zp_in` and not 0. The golden model computes the plain
`(x - zp_in) * w` form, and its packer does the folding. That way the
reference never sees the trick it is checking.

### Channel padding

* Input lanes past `cin` are stored as `zp_in` by the packer, the same value
  im2col_reader gives padded pixels. Their weights are 0, so their value never
  matters to the sum, and they do not change the bias fold either.
* Output channels past `cout` get weight columns of 0 and P words with
  `bias = 0, mult = 0, shift = 0`, so the output byte is exactly `zp_out`. A
  layer's output therefore pads its unused lanes with `zp_out`. That is the
  next layer's `zp_in` when layers are chained, so the rule above holds again
  with no extra work on the host.

## Loop order: output-channel group outer, pixel tile inner

    for og in 0 .. cout_words-1:          # b_base(og), p_base(og) fixed
        for t in 0 .. ceil(n_pix/N)-1:    # n_pix = h_out*w_out, tile = pixels t*N .. t*N+N-1
            run tile (og, t), write its valid rows, wait for the last write

Because everything is in BRAM here, reuse costs nothing either way. The choice
is about control and about what comes next. With `og` outer, every gemm_tile
command in a run of tiles has the same `b_base`, `p_base`, `zp_out` and
`relu_en`. Only the tile origin (y0, x0) and the output row base change, and
both advance by N pixels per tile with a plain raster stepper that is reset
once per group. With pixel tiles outer, the sequencer would step two bases on
every tile and wrap them on every pixel tile. The DDR3 version decides it:
activations of a layer stay in the on-chip act_buffer (that is what the
ping-pong banks are for), while weights grow with depth and are what will be
streamed from DDR3. With `og` outer, each group's K x N weight block and its N
P words are fetched exactly once per layer, and the next group can be
prefetched into the other half of a double-buffered B while the current group
streams all pixel tiles. With pixel tiles outer, every weight would be
refetched once per pixel tile. The cost of `og` outer is that the input is
re-read `cout_words` times and the output words of one pixel are written in
`cout_words` separate passes. In on-chip memory both are free.

## Write-back and tail masking

Every gemm_tile output beat (`m_axis_tready` is tied to 1; nothing downstream
can be full, because the write port takes one word every cycle) is one
act_buffer word:

    pixel  p = t*N + tuser
    addr     = dst_base + p*cout_words + og,   bank dst_sel,   data = tdata

The write is registered for one cycle (address multiply off the path into the
BRAM) and only happens if `p < n_pix`. Rows of the last tile past the end of
the output are computed on whatever the reader fetched for them, and are then
dropped by that compare. They are never written, so nothing outside
`[dst_base, dst_base + n_pix*cout_words)` in bank `dst_sel` changes. The TB
checks this with a canary fill of the whole destination bank.

## Ownership of the memories

| resource | idle (`busy = 0`) | while `busy = 1` |
| -------- | ----------------- | ---------------- |
| act_buffer write port | host, through `host_mode = 1` and the `host_*` write ports | the write-back (datapath port, `host_mode` must be 0) |
| act_buffer read port | host, through `host_rd_en/addr/sel` (needs `host_mode = 0`, because act_buffer freezes reads in host_mode); `host_rd_data` is valid READ_LATENCY read edges later | im2col_reader; `host_rd_*` are ignored |
| B and P memories | host, `b_we`/`p_we` | gemm_tile reads; the host must not write |

The host loads the input image, B and P before `start`. Nothing touches memory
while `busy`. In simulation, `host_mode`, `host_we`, `b_we` or `p_we` while
busy is a `$fatal` (it is undefined in hardware: `host_mode` would freeze
act_buffer's reads under im2col_reader's free-running tag pipeline, see the act_buffer
contract). `host_mode = 1` at start is rejected (E_HOST below) rather than
undefined, because it is cheap to see.

## Illegal configurations

Every check runs on the latched command right after `start` is accepted, in a
short check phase. On a failure the layer does not run, no memory is read or
written, `err` pulses for one cycle, `err_code` shows every check that failed
(a bitmask), and `ready` returns. `err_code` holds until the next start. On a
pass, the layer runs and `err_code` reads 0.

| bit | name | rejected when | why it is rejected instead of left undefined |
| --- | ---- | ------------- | --------------------------------------------- |
| 0 | E_DIM | any of h_in, w_in, h_out, w_out, cin_words, cout_words, kh, kw, stride is 0 | im2col_reader `$fatal`s on several of these; a zero loop count would underflow the sequencer's counters |
| 1 | E_SHAPE | `h_out != (h_in + 2*pad - kh)/stride + 1` (floor), or `kh > h_in + 2*pad`; the same for w | the reader would walk outside what the layer means; a host packing bug shows up here first |
| 2 | E_K | `K = kh*kw*cin_words*N > K_MAX` | gemm_tile would drop the start (err_k) and the sequencer would wait forever for a tlast |
| 3 | E_BMEM | `b_base + cout_words*K > 2**B_ADDR_W` | the last group's weights would wrap onto group 0 |
| 4 | E_PMEM | `p_base + cout_words*N > 2**P_ADDR_W` | same, for P |
| 5 | E_SRC | `src_base + h_in*w_in*cin_words > 2**ADDR_WIDTH` | the image would wrap inside the bank |
| 6 | E_DST | `dst_base + h_out*w_out*cout_words > 2**ADDR_WIDTH` | the output would wrap and overwrite its own start |
| 7 | E_SEL | `src_sel == dst_sel`, or either `>= NBUF` | in place would overwrite pixels later tiles still read; a bad select is an act_buffer `$fatal` |
| 8 | E_HOST | `host_mode = 1` at start | the reads would freeze (see ownership) |

Not checked, documented instead:
* **kh/kw > 7, stride > 3, pad > 3** cannot be expressed: the fields are 3 and
  2 bits wide.
* **pad >= kh** is legal. It gives output pixels that are entirely padding, and
  their value is `requant(bias)`.
* **win_order = 1** (pooling order) is out of scope. It is not a conv_layer port,
  and the reader's `win_order` is tied to 0. The tail mask and write address
  above assume raster order.
* Writing memory while busy: undefined, simulation `$fatal`.

## start / ready / busy / done

* `ready = !busy`. `start` while busy is ignored.
* `busy` rises on the edge that accepts `start`. It falls on the edge where the
  last output word is written (`done`), or at the end of the check phase if the
  command is rejected (`err`).
* `done` is a one-cycle pulse that rises **on the edge where the last write
  lands**. In the ReadOnly phase after that edge, the whole output is in the
  buffer, and the host may read it (with `host_mode = 0`) or start the next
  layer on the same cycle.
* `err` and `done` never pulse for the same command.

Measured cycle counts (filled in from simulation):

    check phase, start to err/run: TBD
    start to done: TBD, compared in the summary with the gemm_tile/im2col formulas

## Reset

`reset` is synchronous and goes to the sequencer, im2col_reader, gemm_tile
and the write-back register. After a reset in the middle of a layer:
* `ready` is high and `busy`, `done`, `err` are low in the ReadOnly phase after
  the reset edge. A new start is accepted from the next edge.
* **No write lands on or after the reset edge.** The write-back enable is
  gated with `!reset`, because gemm_tile's skid buffer may still be holding a
  beat on that edge.
* Every word of the destination region holds either its old value or its
  correct output. A beat is only written once it is final, so a word is never
  half-written. Nothing outside the region changes, and the source bank, B
  and P are untouched (memories have no reset).
* A clean rerun of the same command after the reset gives the full, correct
  output. The TB checks this.

## Out of scope, and what this contract makes harder later

* **Pooling:** raster order only. A fused pool needs `win_order = 1`
  (N % 4 == 0, even w_out) and a write-back that sends one pooled word per four
  rows. The tail mask and the `p*cout_words` address both assume that one row
  is one output pixel, so they will need a second mode.
* **Residual add:** the write-back is a plain write. A fused residual needs the
  matching word of a third bank at the write address, but act_buffer has a
  single read port and im2col_reader owns it while busy. That needs a second
  read port (or a second buffer), which is the biggest structural change.
* **DDR3:** `og` outer is chosen for it (see loop order). The dense
  `b_base + og*K` layout also maps directly onto one DMA burst per group.
* **UART/CSR host:** the command is a flat set of fields latched on `start`, and
  `err_code`/`done` are plain status bits, so a CSR block maps onto them one
  to one.
* **Tile overlap/prefetch:** gemm_tile already raises `ready` before its last
  rows leave requant, and the write-back uses a different bank from the read.
  The only thing that serializes tiles is the sequencer waiting for the
  last write. The reader's walk past the end of the output would then overlap
  the next tile's reads, which is harmless because those rows are masked at
  write-back.
