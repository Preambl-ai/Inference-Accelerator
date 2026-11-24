# pool_layer contract

`pool/rtl/pool_layer.sv` runs one pooling layer out of on-chip memory: it reads
act_buffer bank `src_sel` and writes bank `dst_sel`, while conv_layer is idle.
It has two modes:

* **MAX2**: 2x2 max pool with stride 2 (LeNet).
* **GAVG**: global average pool (the ResNet head).

It uses no multipliers: every address and size is built from shifts and adds,
and the datapath is compares and adds (0 DSP48). Every timing figure is
measured in simulation.

## Decisions made under rule 9 (open trade-offs, recommendation taken)

### 1. Standalone pool_layer versus pooling fused into conv_layer

| | fused into conv_layer | standalone pool_layer (**chosen**) |
| --- | --- | --- |
| how | im2col `win_order = 1` (2x2 windows, needs N % 4 == 0 and even w_out) and a second write-back mode that keeps one word per 4 rows | its own reader and writer on the act_buffer ports, granted by `layer_engine` while conv is idle |
| extra time | none: the pool rides on the conv write-back | one more pass over the conv output: 4 reads per pooled word (MAX2) |
| what it touches | im2col's verified window order, conv_layer's write-back mask, tail rule and address math, and the conv_layer contract | nothing in conv; conv_layer was only split into conv_core + a wrapper, with its TB unchanged |
| limits | N % 4 == 0, even w_out, MAX only; GAVG still needs a separate pass | any N, odd sizes (floor), both modes |

**Recommendation and choice: standalone.** The fused version saves one pass,
which costs `4 * words_out` cycles. For a LeNet layer that is about 1 % of the
conv in front of it, because the conv costs about K + G*(N+L+2) cycles per
tile of N pixels. In exchange, the fused version would reopen the most
verified path in the repo and still not cover GAVG. Standalone also keeps
pooling usable after any layer, not only after a conv.

### 2. How GAVG rounds

Average of `HW = 2^S` values, by shift:

    out[c] = (sum_p x[p][c] + 2^(S-1)) >>> S        (S = 0: out = x)

This is **round half up** (towards +inf on ties) with an arithmetic shift. It
is the same rounding `requant` uses, so a quantized network sees one rounding
convention everywhere.

The alternative was to feed the sum through the existing requant (bias, mult,
shift, zero point). That would need a per-channel 32x16 multiplier (DSPs we
don't have), and an average needs no rescale anyway:

* **Zero points.** Pooling keeps the input's scale and zero point:
  `mean(x) - zp = mean(x - zp)` exactly, and the same holds for max. The
  output zero point therefore equals the input's, and no correction is needed.
* **Range.** The result is always inside int8: the mean of values in
  [-128, 127] rounds to a value in [-128, 127], since
  `(127*2^S + 2^(S-1)) >> S = 127`. So there is no clamp.
* **Unused lanes** (channel padding holds the input zero point, see the
  conv_layer contract) stay at the zero point in both modes, so the next
  layer's padding rule still holds.

The next layer (the classifier conv) does the rescale in its own requant.

## Parameters

| name | default | legal | meaning |
| ---- | ------- | ----- | ------- |
| N | 8 | >= 1 | lanes per word (channels per act_buffer word) |
| DATA_WIDTH | 8 | 8 | int8 only |
| ADDR_WIDTH | 12 | >= 1 | act_buffer word address width |
| READ_LATENCY | 1 | 1, 2 | act_buffer read latency |
| NBUF | 2 | >= 2 | act_buffer banks; src and dst must differ |
| DIM_W | 8 | as conv_layer | h and w fields |
| CW_W | 7 | as conv_layer | c_words field |

The GAVG accumulator is `ACC_W = DATA_WIDTH + 2*DIM_W` bits signed per lane
(24 by default). `h*w <= 255*255 < 2^16`, so a sum of up to 2^16 int8 values
needs 8 + 16 bits.

## Ports

| port | dir | width | notes |
| ---- | --- | ----- | ----- |
| clk, reset | in | 1 | reset synchronous |
| start | in | 1 | command latched on `start & ready` |
| ready, busy | out | 1 | `ready = !busy` |
| done, err | out | 1 | one-cycle pulses, never both |
| err_code | out | 10 | bitmask, holds until the next start |
| mode | in | 2 | 1 = MAX2, 2 = GAVG, anything else is E_MODE |
| h_in, w_in | in | DIM_W | input image size in pixels |
| c_words | in | CW_W | channel groups (channels / N, rounded up) |
| src_sel, dst_sel | in | SEL_WIDTH | act_buffer banks |
| src_base, dst_base | in | ADDR_WIDTH | word address of pixel (0,0), group 0 |
| host_mode | in | 1 | observed only: E_HOST at start, `$fatal` while running |
| rd_en, rd_addr, rd_sel | out | 1 / AW / SW | act_buffer read port (granted while busy) |
| rd_data | in | N*8 | act_buffer `rd_data`, READ_LATENCY advancing edges after |
| we, waddr, wdata, wr_sel | out | 1 / AW / N*8 / SW | act_buffer datapath write port |

The memory layout is the conv_layer one: pixel (y, x), group g is at
`base + (y*w + x)*c_words + g`, and lane j is channel g*N + j.

## Arithmetic, per lane, signed int8

* **MAX2.** `h_out = h_in >> 1`, `w_out = w_in >> 1` (floor: an odd last
  row or column is dropped, as in the usual framework default).

      out[oy][ox][c] = max(x[2oy][2ox][c], x[2oy][2ox+1][c], x[2oy+1][2ox][c], x[2oy+1][2ox+1][c])

  The output is `h_out x w_out x c_words` words at `dst_base`, in the same
  layout.
* **GAVG.** `h_in*w_in = 2^S`, which must be a power of two. The output is
  one pixel (1 x 1 x c_words words at `dst_base`), with the formula above.

## Order of work (no multipliers)

The check phase forms `w_in*c_words`, `h_in*w_in`, `h_in*w_in*c_words` and the
output size with bit-serial shift-and-add multiplies: 8 steps each, all in
parallel, in two rounds. While running, the addresses come from pointers that
only ever add a constant:

* **MAX2:** loops over oy, ox, g and then the tap, with the tap innermost.
  The tap offsets are 0, c_words, row = w_in*c_words, and row + c_words. A
  new ox adds 2*c_words, and a new oy adds 2*row. Output words come out in
  order, so the write address just counts up from dst_base.
* **GAVG:** loops over g and then pixel p. Each step adds c_words to the
  read pointer, and the output for g goes to dst_base + g.

One read is issued every cycle with `rd_en` held high. Each read carries a tag
{first, last, output address} through a READ_LATENCY-deep delay that meets the
data. On `first` the lane accumulator loads the data. Otherwise it takes the
max (MAX2) or adds the sign-extended data (GAVG). On `last` the result goes
into a write register, and the write lands one edge later.

## Illegal configurations

They are checked after `start` in a fixed check phase (measured below). On
failure, `err` pulses, `err_code` holds every failed check, and nothing is
read or written. Bit positions match conv_layer's where the meaning is the
same, so one decoder serves every engine.

| bit | name | rejected when |
| --- | ---- | ------------- |
| 0 | E_DIM | h_in, w_in or c_words is 0 |
| 1 | E_SHAPE | MAX2: h_in < 2 or w_in < 2 (empty output). GAVG: h_in*w_in is not a power of two. Only checked when E_DIM and E_MODE are clear |
| 5 | E_SRC | src_base + h_in*w_in*c_words > 2^ADDR_WIDTH |
| 6 | E_DST | dst_base + output words > 2^ADDR_WIDTH |
| 7 | E_SEL | src_sel == dst_sel, or either is >= NBUF |
| 8 | E_HOST | host_mode = 1 at start |
| 9 | E_MODE | mode is not 1 or 2 |

Bits 2-4 (conv's K, B and P memory checks) are always 0 here.

## start / done / reset

* `busy` rises on the accepting edge, and falls with `done` or `err`.
* `done` rises on the edge where the last output word is written, so the
  whole output is in the buffer in the ReadOnly phase after it.
* Reset is synchronous and clears the FSM, the tag pipeline and the write
  register. No write lands on or after the reset edge, because the write
  enable is gated with `!reset`. Every destination word holds either its old
  value or its final one, never a partial result. Ready is high after reset.

## Measured (from simulation, N 1/2/4/8 x READ_LATENCY 1/2)

These are edge counts from the edge that accepts `start` (L = READ_LATENCY):

    check phase (err visible)  : DIM_W + CW_W + 1 = 16 edges, for every check
    start to done              : (DIM_W + CW_W + 1) + R + L + 1
        R = reads = 4 * h_out * w_out * c_words   (MAX2)
                  = h_in * w_in * c_words          (GAVG)
    throughput                 : one act_buffer read per cycle; one output
                                 word per 4 cycles (MAX2), per h*w cycles (GAVG)

The formula matches simulation exactly in every TB run, and the TB asserts
it. The `+ L + 1` is the last read's latency plus the write register.

Mutation pass: 12 injected bugs, 11 caught. The survivor is equivalent: making
the GAVG rounding constant unsigned turns `>>>` into a logical shift, but the
zero-filled top bits enter at bit `ACC_W - S >= 24 - 15 = 9`. They never reach
the 8 bits that are kept, so no input can tell the two apart.
