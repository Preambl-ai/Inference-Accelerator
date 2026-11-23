# Host interface contract

How a PC drives `conv_layer` on the Arty A7-35T:

    PC (Python, pyserial) -> USB-UART (FT2232H) -> uart_rx / uart_tx
      -> uart_bridge (packets -> AXI4-Lite master)
      -> csr (AXI4-Lite slave: control, status, command, memory windows)
      -> conv_layer

The same Python packet code (`host/protocol.py`) builds the bytes for the
cocotb UART BFM in simulation and for the real serial port, so the simulated
and real boards see identical traffic. Every cycle figure in this file is
measured in simulation.

## 1. UART link

### Framing and baud rate

8 data bits, no parity, 1 stop bit, LSB first, idle high. One parameter sets
the baud divisor, `CLKS_PER_BIT = CLK_HZ / BAUD`. The simulations run with a
small divisor; the board uses the values below.

**Default: 1,000,000 baud.** With 100 MHz / 100 = 1 Mbaud exactly, and the
FT2232H's 3 MHz baud base / 3 = 1 Mbaud exactly, **neither side has any rate
error**. At 50 MHz (MMCM option, section 5) the divisor is 50 and still exact.
Fallback: 115200 baud (divisor 868, 0.006 % error), selected by one parameter
on each side.

| baud | FPGA divisor (100 MHz) | FPGA error | FT2232H error |
| ---- | ---------------------- | ---------- | ------------- |
| 115200 | 868 | 0.006 % | 0.16 % |
| 921600 | 109 | 0.47 % | 0.16 % |
| 1000000 | 100 | 0 | 0 |
| 3000000 | 33 | 1.0 % | 0 |

A plain 921600 is no faster than 1 Mbaud and has two rate errors that add.
Above 1 Mbaud the FPGA divisor gets coarse. The receiver samples mid-bit, so it tolerates
about ±4 % in total; the TB checks ±2 %.

### Load time for a realistic layer

N = 8, a 16x16x64 input, a 3x3 kernel with pad 1, and 64 output channels:

| item | words | bytes of payload |
| ---- | ----- | ---------------- |
| input image, 256 px x 8 channel words x 8 B | 2048 | 16 384 |
| B, 8 groups x K = 576 rows x 8 B | 4608 | 36 864 |
| P, 64 channels x 8 B (53-bit word in two 32-bit slots) | 64 | 512 |
| output readback, 256 px x 8 words x 8 B | 2048 | 16 384 |

Each burst of 256 32-bit words adds 9 header bytes, 2 data CRC bytes and a
4-byte response, about 1.5 %. There are 10 bits on the wire per byte.

| baud | load (53.8 KB) | readback (16.4 KB) | total |
| ---- | -------------- | ------------------ | ----- |
| 115200 | 4.7 s | 1.4 s | 6.2 s |
| 1000000 | 0.55 s | 0.17 s | 0.72 s |

For comparison, the layer itself takes about 0.36 M cycles (3.6 ms at 100 MHz,
from the conv_layer cycle formula). That is why the link speed matters and
the compute does not, and why reads get a burst opcode too: the readback is as
big as the input image.

### Packets

All multi-byte fields are little endian.

**CRC: CRC-16/CCITT-FALSE.** Poly 0x1021, init 0xFFFF, no input or output
reflection, no final XOR; the check value over the ASCII bytes "123456789" is
0x29B1. It is defined once, in `host/protocol.py` (`crc16`), and every user
of it imports it from there: the host driver, the cocotb BFMs and the models.
The two RTL copies (in the bridge and in the CSR's write-CRC registers) are
checked against that function by their TBs. A CRC-16 rather than an 8-bit
sum: a dropped byte shifts the rest of the packet, and an 8-bit check would
accept a misframed header 1 time in 256. CCITT-FALSE because it is
byte-serial MSB-first with no reflection, which is the cheapest form in
hardware, and there is no external standard to match.

Host to board:

    0xA5  OP  ADDR[4]  LEN  HCRC[2]   [DATA[4*(LEN+1)]  DCRC[2]]

| OP | name | LEN | data phase |
| -- | ---- | --- | ---------- |
| 0x01 | write32 | must be 0 | 4 bytes + DCRC |
| 0x02 | read32 | must be 0 | none |
| 0x03 | burst write | count - 1 (1 .. 256 words) | 4*(LEN+1) bytes + DCRC |
| 0x04 | burst read | count - 1 | none |

* `HCRC` covers OP, ADDR and LEN; `DCRC` covers the data bytes.
* A burst starts at word-aligned `ADDR` and steps by 4 per word.

Board to host, exactly one response per packet:

    0x5A  [DATA[4*(LEN+1)] for reads]  STATUS  RCRC[2]

`RCRC` covers the data and STATUS. STATUS comes **after** the data: a burst
read streams each word out as soon as it has been read, so a bus error at
word k is only known after words 0 .. k-1 have been sent. From word k on the
bridge sends zeros, and STATUS reports the error. The host knows the
response length from its own LEN. There is one outstanding packet at a time:
the host waits for the response (or its own timeout) before it sends the next
packet. Bytes that arrive while a response is being sent are dropped. That
packet then times out and is retried.

| STATUS | meaning | what happened on the bus |
| ------ | ------- | ------------------------ |
| 0x00 | OK | every word was transferred |
| 0x01 | SLVERR | the CSR refused a word (section 4); the burst stops at that word |
| 0x02 | DECERR | an address is not mapped; the burst stops at that word |
| 0x03 | HCRC error | nothing was done |
| 0x04 | DCRC error | write32: nothing written. burst write: see below |
| 0x05 | bad opcode or LEN | nothing was done |
| 0x06 | timeout | a packet stalled midway (a byte was dropped); it was discarded |

The response arrives after the bus transfer has finished, so an OK write is
already in the CSR or memory when the host sees the OK.

### Error detection: a dropped or corrupted byte is never silent

* **Header:** the header is fully received and its HCRC checked before
  anything reaches the bus. A bad header is answered with 0x03, and the bridge
  then drops bytes until the line has been idle for the timeout. Address and
  length are therefore trusted before they are used.
* **write32:** the 4 data bytes and DCRC are buffered, and the AXI write only
  happens after DCRC passes.
* **Burst write:** words are written as they arrive (there is no 1 KB buffer
  on chip), and DCRC is checked at the end. On a mismatch the answer is 0x04.
  Some words in **[ADDR, ADDR + 4*(LEN+1)) only** may hold wrong values;
  nothing outside the range the checked header declared can be touched. The
  host rewrites the whole burst. That is idempotent, so the retry fixes the
  range.
* **A dropped byte:** the packet comes up short, and the bridge waits for the
  missing byte. The inter-byte timeout discards the partial packet and answers
  0x06. It is `TIMEOUT_BITS` bit times (default 1000: 1 ms at 1 Mbaud, 8.7 ms
  at 115200), counted in the hardware as `TIMEOUT_BITS * CLKS_PER_BIT` clocks,
  so it scales with the baud rate on its own. That is far longer than any gap
  inside one packet the host writes in a single call, and far shorter than
  the host's 50 ms response timeout. If the
  drop was in the header, HCRC usually catches it first. A dropped byte in a
  burst write is caught by DCRC or the timeout. Either way the host sees a
  non-OK answer and retries.
* **Responses:** the host checks RCRC and the response length. If either is
  bad, or no response arrives, the host flushes, waits past the timeout and
  retries. Reads and writes are both idempotent, so a retry is always safe.
  The single exception is the start bit in CTRL; see section 4.
* **Every error clears the wide-word staging.** On any HCRC, DCRC,
  bad-opcode or timeout error the bridge pulses `link_err` to the CSR, which
  clears the slot-valid bits of all three windows (section 2). A half-staged
  word from a broken packet can therefore never be completed by a later one.

## 2. CSR map (AXI4-Lite, 32-bit data, 24-bit byte address)

`csr` is an AXI4-Lite slave. Writes may present AW and W in either order or
together; B and R may be held off by the master. Unmapped addresses answer
DECERR. Refused accesses answer SLVERR and change nothing.

### Registers (0x00_0000 .. 0x00_00FF)

| addr | name | access | reset | fields |
| ---- | ---- | ------ | ----- | ------ |
| 0x00 | ID | RO | 0x434E_5631 ("CNV1") | fixed. The host reads this first and stops on a mismatch |
| 0x04 | VERSION | RO | 0x0001_0000 | [31:16] major, [15:0] minor of this map |
| 0x08 | PARAMS0 | RO | from parameters | [7:0] N, [11:8] READ_LATENCY, [19:12] ADDR_WIDTH, [23:20] NBUF, [31:24] B_ADDR_W |
| 0x0C | PARAMS1 | RO | from parameters | [7:0] P_ADDR_W, [23:8] K_MAX |
| 0x10 | CTRL | RW | 0 | [0] start: write 1 pulses `start` for one cycle, reads 0. [1] soft_reset: write 1 pulses conv_layer `reset` for one cycle, reads 0. [2] host_mode: level, read/write. [3] clear_wcrc: write 1 sets all three *_WCRC to 0xFFFF, reads 0 |
| 0x14 | STATUS | RO / W1C | 0x2 | [0] busy, [1] ready, [2] done (sticky), [3] err (sticky), [4] host_mode. Writing 1 to [2] or [3] clears it; start also clears both |
| 0x18 | ERR_CODE | RO | 0 | conv_layer `err_code[8:0]`, holds until the next start |
| 0x1C | CYCLES | RO | 0 | edges from the accepting edge to done (or err), latched there. Counts live while busy |
| 0x20 | SCRATCH | RW | 0 | link test, no effect |
| 0x24 | BUILD_ID | RO | `BUILD_ID` parameter | git short hash of the build, set by build.tcl (0 in simulation) |
| 0x28 | ACT_WCRC | RO | 0xFFFF | running CRC-16 over every byte committed to the act_buffer through the ACT window |
| 0x2C | B_WCRC | RO | 0xFFFF | the same for B |
| 0x30 | P_WCRC | RO | 0xFFFF | the same for P |
| 0x34 | LAUNCH_COUNT | RO | 0 | +1 on every accepted start (including starts that conv_layer later rejects with err). Wraps at 2^32 |
| 0x40 .. 0x88 | command | RW | 0 | one field per register, low bits, in this order: H_IN, W_IN, H_OUT, W_OUT, CIN_WORDS, COUT_WORDS, KH, KW, STRIDE, PAD, ZP_IN, ZP_OUT, RELU_EN, SRC_SEL, DST_SEL, SRC_BASE, DST_BASE, B_BASE, P_BASE. Each is masked to its port width, so a read gives back what conv_layer will see. ZP_* read back sign-extended |

The cycle counter uses the same edge count as the conv_layer TB, so on the
board `CYCLES` must equal the conv_layer cycle formula. The host prints both.

### Memory windows

A memory word wider than 32 bits takes `S = ceil(width/32)` consecutive
32-bit slots, padded to a power of two (`SLOT_SHIFT = log2(S)`). Slot 0 holds
bits [31:0].

| window | base | word index in the address | word width | access |
| ------ | ---- | -------------------------- | ---------- | ------ |
| ACT | 0x10_0000 | `{bank, addr}` (SEL_WIDTH + ADDR_WIDTH bits) | 8N | read and write |
| B | 0x20_0000 | `addr` (B_ADDR_W bits) | 8N | write only (reads are SLVERR) |
| P | 0x30_0000 | `addr` (P_ADDR_W bits) | 53 | write only |

    byte address = base + ((word_index << SLOT_SHIFT) + slot) * 4

* **Writes:** writing a slot below the top one only stages it in a
  per-window register and sets that slot's **valid bit**. Writing the **top
  slot commits** the whole word (the staged lower slots plus the top slot)
  in one memory write, but only if every lower slot's valid bit is set and
  they were all written for this same word index. Otherwise the top-slot
  write answers SLVERR and nothing is written. Every top-slot write, good or
  bad, clears the valid bits. So does a lower-slot write for a different word
  index (it restarts staging with only its own slot), and so does any bridge
  error (`link_err`). The burst write naturally writes slots in ascending
  order. Words of 32 bits or less have one slot, which commits on its own.
* **Write CRCs:** each commit also feeds the ceil(width/8) bytes of the
  committed word, LSB first, into that window's `*_WCRC` (CRC-16/CCITT-FALSE,
  one byte after another, word after word in commit order). That is N bytes
  for ACT and B, and 7 bytes for P (bits 53..55 are 0). The host computes the
  same CRC over what it meant to write and compares before start, which
  catches anything that got past the link checks.
* **A retried burst poisons the write CRC.** A burst write that failed its
  DCRC has already committed some wrong words before the retry rewrote them,
  and the running CRC keeps those bytes. Memory is right after the retry, but
  the CRC can never match. So a load is all or nothing: the host clears the
  CRCs, writes everything, compares, and repeats the whole load until one
  pass is clean (`protocol.load`, at most 3 passes). The top TB injects a
  flipped byte and a dropped byte into a real layer load and checks that this
  path is taken and the output is still exact.
* **ACT reads:** reading any slot reads the whole word from the act_buffer
  through conv_layer's host read port and returns that slot. The CSR holds
  `host_rd_en` high with the same address for READ_LATENCY edges and takes
  `host_rd_data` after that. An act_buffer read advances only while rd_en is
  high, so the pipeline is full of this one address by then. RVALID rises
  only once the data is there. Read latency is absorbed in the AXI handshake
  and the master just waits (measured: section 6).
* **B and P cannot be read back.** gemm_tile has no read port on them, and
  adding one is not needed. A wrong B or P word shows up as a wrong output,
  which the end-to-end compare checks byte for byte.

## 3. Act_buffer readback: why it cannot disturb a running layer

conv_layer already muxes the act_buffer read port on `busy`. While busy, the
reader owns the port and `host_rd_*` are ignored; while idle, the host owns
it. The CSR adds two more locks on top:

* every ACT, B and P window access while `busy` is refused (SLVERR) before
  any memory port is touched, so the host read port is not even driven then;
* ACT reads while `CTRL.host_mode = 1` are refused, because act_buffer
  freezes reads in host mode (act_buffer contract, case d), and so are ACT
  writes while `host_mode = 0`, because act_buffer ignores the host write port
  then.

`host_mode` stays an explicit level that the host sets, not something the CSR
pulses behind the host's back. It is the one ownership flag the act_buffer and
conv_layer contracts already define. `start` with `host_mode = 1` is rejected
by conv_layer itself (E_HOST) and shows up in STATUS.err. A mux that reads
during host mode (act_buffer reads are frozen then anyway) was considered and
rejected: it would change act_buffer's verified contract to save one CTRL write
per layer.

Host sequence per layer:

    CTRL.clear_wcrc
    CTRL.host_mode = 1; burst-write ACT image; CTRL.host_mode = 0
    burst-write B, P
    write all 19 command registers in ONE burst (0x40 .. 0x88 are contiguous)
    read ACT_WCRC, B_WCRC, P_WCRC and compare with the host's own CRCs
    read LAUNCH_COUNT; CTRL.start = 1; poll STATUS until done or err
    read CYCLES, ERR_CODE; burst-read the output region of ACT

Batching the command into one burst matters more than raw baud rate for small
layers. The FTDI chip holds received bytes for up to its latency timer
(16 ms by default) before sending them over USB, so every round trip can cost
16 ms. Nineteen separate writes would cost about 0.3 s; one burst costs one
round trip. Setting the latency timer to 1 ms is the other fix. It is a
driver setting that pyserial cannot change portably (macOS has no sysfs knob),
so the host batches instead. fpga/README.md records this.

## 4. What the host may do while busy

While `STATUS.busy = 1`, the only accepted accesses are:
* register reads (all of 0x00 .. 0x88)
* a write of `CTRL.soft_reset`
* W1C writes to STATUS

Everything else is answered SLVERR and has **no effect**:
* window reads and writes
* writes to the command registers
* `CTRL.start`
* `CTRL.host_mode = 1`

Command registers would be harmless to change while busy (conv_layer latches
them at start), but refusing them catches a host bug early. Because the CSR
refuses these accesses, conv_layer's simulation `$fatal`s (host_mode, B/P
writes while running) **cannot be reached from the UART**.

**Start is not idempotent.** The host reads LAUNCH_COUNT before writing
CTRL.start. If the start packet gets no good response, it reads LAUNCH_COUNT
again: if it moved, the start took (whatever happened to the response); if
not, it is safe to resend. STATUS would be ambiguous here: a very short
layer can already be done, and done is cleared by the next start.

## 5. Clocking and reset

* **One clock domain.** `top` runs on the 100 MHz board clock directly.
  Parameter `CLK_DIV`: 1 = direct (default); 2 = 50 MHz from an MMCME2_BASE,
  synthesis only, used if timing does not close at 100 MHz. `CLK_HZ` follows
  it and sets the UART divisor and the timeout. Every flop, BRAM and DSP in
  the design is on this one clock.
* **Baud divisor:** `CLKS_PER_BIT = CLK_HZ / BAUD`, so CLK_DIV = 2 (50 MHz)
  at 1 Mbaud means a divisor of 50, still exact. The timeout follows
  automatically, because it is in bit times.
* **Reset:** BTN0 (active high, asynchronous to the clock) goes through a
  2-flop synchronizer, then a stretcher that holds the synchronous `reset`
  for 16 cycles. A power-on counter (flops with an initial value, which 7-series
  configuration supports) holds `reset` for its first 16 cycles after
  configuration too. `reset` resets everything: UART, bridge, CSR (all
  registers to the values above) and conv_layer. `CTRL.soft_reset` resets
  only conv_layer plus STATUS.done/err and CYCLES. It **never** resets the UART,
  the bridge, the CSR's bus logic, the command registers, host_mode,
  LAUNCH_COUNT or the write CRCs. The bus is in the middle of answering the
  very write that asked for it, and the host needs LAUNCH_COUNT to survive.
* **UART RX** is the only asynchronous input besides the button. It goes
  through a 2-flop synchronizer (reset value 1 = idle) before the start-bit
  detector.
* **There are no other clock domain crossings.** The FT2232H's TX pin is our
  RX (synchronized); our TX is a flop driving the FT2232H's RX; the LEDs are
  outputs. Nothing else enters or leaves the clock domain.

## 6. Measured figures (from simulation)

    CSR register read:  RVALID 1 edge after the AR handshake edge
    ACT window read:    RVALID READ_LATENCY + 2 edges after the AR handshake
                        (3 at READ_LATENCY 1, 4 at READ_LATENCY 2; csr TB,
                        N 2/4/8). host_rd_en is high for exactly
                        READ_LATENCY edges at one address (checked per read)
    bridge, write32:    the response's start bit begins 2 clocks after the
                        request's last stop bit ends (top TB, CLKS_PER_BIT 8).
                        uart_rx hands over a byte in the middle of its stop
                        bit, so the AXI write fits in that half bit
    CYCLES on the bus:  equals the conv_layer formula for every layer the top
                        TB runs; a rejected command reads 4 (the check phase)

All of these are a few clocks; at 1 Mbaud one byte is 1000 clocks, so the link
time in section 1 is the whole story.

## Out of scope

DDR3/MIG, DMA, pooling, tile overlap/prefetch. The memory-window scheme
(32-bit slots, commit on the top slot) is what a later DMA engine would
replace: it would write whole words into the same memories through the same
CSR-side write ports, and the UART path would stay for control and status.
