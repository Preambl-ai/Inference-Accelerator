# add_layer contract

`add/rtl/add_layer.sv` is the ResNet residual add. It adds two INT8 tensors
of the same shape, each with its own scale and zero point, element by
element, and writes a third:

    out = sat8( round( (a - za)*ma + (b - zb)*mb ) / 2^shift ) + zo )     (optional ReLU)

Operand a comes from bank `src_sel`, operand b from bank `src2_sel`, and the
output goes to bank `dst_sel`. It uses **no DSPs and no multipliers**: the two
scale multiplies are bit-serial shift-and-add. Every timing figure is measured
in simulation.

## Decisions made under rule 9

### 1. How to read two operands: time-multiplex the one read port, or a dual-read act_buffer

The act_buffer has one read port, shared by every bank (`rd_sel`), and one
datapath write port. An add needs a word from a, a word from b, and one write.

| | time-multiplexed read (**chosen**) | true dual-read act_buffer |
| --- | --- | --- |
| reads per output word | 2, one after the other on the same port | 2 at once |
| best case throughput | 1 word per 2 cycles (read-bound) | 1 word per cycle |
| throughput with this datapath | 1 word per `M_BITS + L + 3` cycles (measured below); the bit-serial rescale, not the port, sets it | the same, because the rescale still sets it |
| BRAM | no change | every bank duplicated, or bram.sv rebuilt as true dual port with the write port shared. Duplicating at N=8, ADDR_WIDTH=12: +8 BRAM36 per bank, so 3 banks become 48, and with B (16) and P the total is ~65 of **50**. It does not fit |
| touches verified blocks | no (act_buffer and bram unchanged) | act_buffer, bram and every engine's read path |

**Chosen: time-multiplex.** The dual-read buffer only pays off if the rest of
the add runs at one word per cycle. That would need 2N multipliers per cycle
(16 DSPs at N=8, which we don't have) or 2N LUT multipliers (about 1.5 k
LUT). It also doesn't fit in BRAM.

Cost in context: a ResNet-20 residual on a 32x32x16 map at N=8 is 2048 words,
which is about 30 k cycles (0.3 ms) here. The convolutions around it take
about 1 M cycles each.

### 2. How to rescale without DSPs

The two inputs of a residual add almost never share a scale, so each must be
multiplied by its ratio to the output scale. The options:

* **DSP multiply:** 2N DSP48s. We have none to spare (rule 6).
* **LUT multiplier (9x8 per lane, two per lane):** about 1.5 k LUT at N=8 and
  1 word per 2 cycles. It still counts as a multiplier in the datapath.
* **Power-of-two ratios only:** shifts, free, but it forces the quantizer to
  use scales a factor of 2 apart. That is too coarse for trained ResNets.
* **Bit-serial shift-and-add (chosen):** `ma` and `mb` are `M_BITS`-bit
  unsigned (8 by default, so the ratio precision is 1/256 after the shift).
  Each cycle, every lane adds `(a - za) << i` if bit i of `ma` is set, and
  the same for b. That is two adders per lane, and `M_BITS` cycles per word.

Chosen: bit-serial. It is adds and shifts only, about 0.3 k LUT, and the
throughput cost is shown above.

## Arithmetic (exact, per lane, signed)

| quantity | range | width |
| -------- | ----- | ----- |
| a, b, za, zb, zo | [-128, 127] | 8 signed |
| ra = a - za, rb = b - zb | [-255, 255] | 9 signed |
| ma, mb | [0, 2^M_BITS - 1] | M_BITS unsigned |
| acc = ra*ma + rb*mb | \|acc\| <= 2*255*255 < 2^17 | ACC_W = M_BITS + 11 = 19 signed |
| shift | [0, 15] | 4 unsigned |
| q = shift ? (acc + 2^(shift-1)) >>> shift : acc | as acc | ACC_W |
| out = clamp(q + zo, relu ? zo : -128, 127) | int8 | 8 signed |

* **Rounding:** round half up with an arithmetic shift, as in requant and
  the GAVG pool. One rounding rule for the whole network.
* **Saturation:** at the int8 rails. ReLU raises the floor to `zo`, as in
  requant.
* **Mapping from real scales.** With real values `Sa*(a - za)`, `Sb*(b - zb)`
  and output scale `So`, the host picks `shift` and
  `ma = round(Sa/So * 2^shift)`, `mb = round(Sb/So * 2^shift)` in range.
  The golden model takes ma, mb and shift as given, so the hardware is
  checked bit-exactly whatever the host chose.
* **Unused lanes** hold each tensor's own zero point (the padding convention
  of every engine). That gives `ra = rb = 0`, so `out = zo`, and the
  convention carries on to the next layer.

## Parameters

| name | default | legal | meaning |
| ---- | ------- | ----- | ------- |
| N, DATA_WIDTH, ADDR_WIDTH, READ_LATENCY, NBUF, DIM_W, CW_W | as layer_engine | | |
| M_BITS | 8 | 1 .. 16 | width of ma and mb, and the cycles of the bit-serial rescale |

## Ports

These are the pool_layer port pattern plus the add fields:
* command: `start`, `h_in`, `w_in`, `c_words`, `src_sel`, `src_base`,
  `src2_sel`, `src2_base`, `dst_sel`, `dst_base`, `za`, `zb`, `zo`, `ma`,
  `mb`, `shift`, `relu_en`, and `host_mode` (E_HOST only)
* status: `ready`, `busy`, `done`, `err`, `err_code[9:0]`
* act_buffer read port (`rd_en`, `rd_addr`, `rd_sel`, `rd_data`) and
  datapath write port (`we`, `waddr`, `wdata`, `wr_sel`), granted by
  layer_engine while busy

## Order of work

Words are numbered `i = 0 .. h*w*c_words - 1`. Word i of each tensor is at
`base + i`: the tensors share a shape and the same packed layout, so an
elementwise op is a linear walk. For each word:
1. Read a[i], then read b[i], on consecutive cycles, with `rd_en` held high.
2. When the two words have arrived (READ_LATENCY later), form ra and rb for
   every lane.
3. Run `M_BITS` shift-and-add steps, one bit of ma and one bit of mb per
   step, LSB first.
4. Round, add zo, clamp, and write `dst_base + i` from a write register.

This reference version serializes the words (the next reads start after the
write is registered). Overlapping word i+1's reads with word i's multiply is
the obvious next step. It saves `L + 2` cycles per word.

## Illegal configurations

| bit | name | rejected when |
| --- | ---- | ------------- |
| 0 | E_DIM | h_in, w_in or c_words is 0 |
| 5 | E_SRC | `src_base + words > 2^ADDR_WIDTH` or `src2_base + words > 2^ADDR_WIDTH` |
| 6 | E_DST | `dst_base + words > 2^ADDR_WIDTH` |
| 7 | E_SEL | `dst_sel == src_sel`, `dst_sel == src2_sel`, or any of the three >= NBUF. `src_sel == src2_sel` is legal: the port reads one operand at a time |
| 8 | E_HOST | host_mode = 1 at start |

In-place (dst = a) is rejected rather than allowed. The serialized walk
would make it safe, but the planned overlapped version reads ahead of the
write, and the rule should not change under the user.

These are checked in a check phase measured below; on failure, `err`
pulses, `err_code` holds the bits, and nothing is read or written.

## start / done / reset

These are as for pool_layer:
* `done` rises on the edge where the last word is written.
* Reset is synchronous, and no write lands on or after the reset edge.
* Destination words are left either old or final.

## Measured (from simulation, N 1/2/4/8 x READ_LATENCY 1/2 x NBUF 2/3)

These are edge counts from the edge that accepts `start` (L = READ_LATENCY):

    check phase (err visible)  : DIM_W + CW_W + 1 = 16 edges, for every check
    start to done              : 16 + W * (M_BITS + L + 3) + 1,   W = h*w*c_words
    per word                   : read a (1) + read b (1) + until b is in (L)
                                 + M_BITS shift-and-add steps + output (1)

The TB asserts this on every command it runs. At M_BITS = 8 and L = 1, a
word costs 12 cycles, so the 32x32x16 ResNet-20 residual (2048 words at
N = 8) takes about 24.6 k cycles, or 0.25 ms at 100 MHz.

The paper estimate in decision 1 first said `2L`. The measurement shows `L`:
b's read is issued one cycle after a's, and a has already arrived by the time
b does.

Mutation pass: 13 injected bugs, all caught. They cover both zero points,
which bank and multiplier each operand uses, the bit position of the
shift-and-add, the last multiplier bit, rounding, both clamps, ReLU, the b
pointer, the two bank checks, the reset gate on the write, and the output
zero point.
