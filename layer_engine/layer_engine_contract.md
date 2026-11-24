# layer_engine contract

`layer_engine/rtl/layer_engine.sv` holds the act_buffer and every engine that
computes from it: conv_core (from conv_layer), pool_layer and, from phase 2,
add_layer. It owns the act_buffer's ports and lends them to exactly one owner
at a time:

* the busy engine;
* otherwise, the host (the same host access conv_layer has).

It also holds B and P (inside conv_core), with the same host write ports as
before. It adds no arithmetic and no DSPs.

## Decision under rule 9: one command bus with an op, or one start per engine

| | one start port per engine | one `start` + `op` (**chosen**) |
| --- | --- | --- |
| two engines started together | possible; needs a rule (priority, error, `$fatal`) | impossible: one start goes to one engine |
| command fields | one set per engine, mostly the same (banks, bases, sizes) | one superset; each engine latches the fields it uses |
| descriptor sequencer (phase 3) | decodes the op itself, then drives the matching port set | writes the descriptor fields straight onto the bus |
| adding an engine | new port set, new collision cases | one more op code |

**Chosen: one start with an op.** Ownership is "exactly one" by
construction, and the remaining overlap check (two engines busy at once) is
a pure simulation `$fatal` that no legal input can reach.

## Ops

| op | engine | fields used |
| -- | ------ | ----------- |
| 1 CONV | conv_core | all conv_layer command fields |
| 2 MAX2 | pool_layer (mode 1) | h_in, w_in, cin_words (as c_words), src_sel, dst_sel, src_base, dst_base |
| 3 GAVG | pool_layer (mode 2) | the same |
| other | none | `err` with E_MODE (bit 9), 1 edge after start, nothing touched |

Phase 2 adds op 4, ADD.

## Ports

* **Command:** `start`, `op[2:0]`, and the conv_layer command fields (h_in,
  w_in, h_out, w_out, cin_words, cout_words, kh, kw, stride, pad, zp_in,
  zp_out, relu_en, src_sel, dst_sel, src_base, dst_base, b_base, p_base),
  with conv_layer's widths. All are latched by the chosen engine on
  `start & ready` and need not be held.
* **Status:**
  * `ready = !busy`
  * `busy` = any engine busy (or a bad-op rejection in progress)
  * `done`, `err`: one-cycle pulses, never together
  * `err_code[9:0]`: the finishing engine's code, conv's 9 bits zero-extended;
    it holds until the next start
  * `done_op[2:0]`: the op that just finished, valid with done/err
  * `conv_busy`, `pool_busy`
  * `tile_launch`: conv's per-tile pulse
* **Memory:** the B/P write ports and the act_buffer host ports, exactly as
  on conv_layer. Their rules (idle only, host_mode for writes, host_mode
  low for reads) are unchanged.

## Error codes

These are the same bit positions in every engine, so one decoder serves all
of them:

    0 E_DIM, 1 E_SHAPE, 2 E_K, 3 E_BMEM, 4 E_PMEM, 5 E_SRC, 6 E_DST, 7 E_SEL, 8 E_HOST, 9 E_MODE

## Ownership and the checks

    owner = conv_busy ? CONV : pool_busy ? POOL : HOST

* **Read port** (rd_en, rd_addr, rd_sel): the owner's. When the host owns
  it, `host_rd_*` drive it, as on conv_layer.
* **Datapath write port**: the owner engine's. With no engine busy, the
  enable is held low, so an idle engine can never write.
* **`start` while busy** is ignored, as for every engine.
* **Simulation `$fatal`** if two engines are ever busy in the same cycle, or
  if an engine that does not own the port raises its write enable. Neither
  can be reached by legal input, so they guard the RTL, not the user.

## Timing

The engine adds no register stage: each engine's command, data and status
pass straight through. Every latency is therefore the engine's own:

* conv: the conv_layer formula
* pool: the pool_layer formula
* a bad op: err 1 edge after start

These are measured in the layer_engine TB, which asserts the conv and pool
formulas on every command it runs.

Measured in simulation (N 2/4/8, READ_LATENCY 1/2, NBUF 2/3):
* Every conv count equals the conv_layer formula, every pool count the
  pool_layer formula, including a command started in the cycle right after
  the previous done.
* A bad op shows err exactly 1 edge after start.
* Forty starts of every op aimed at a running conv produce no second engine
  and one done.

Mutation pass: 9 bugs injected, 8 caught.
* The survivor, "pool owns the port before conv", is equivalent: it only
  differs when both engines are busy, which one start cannot cause (and which
  is `$fatal`-guarded).
* The pass also found a TB gap: no test drove `host_we` with `host_mode`
  low. That is now covered (`test_idle_write_port_closed`) and catches "idle
  write port opened to the host path".

## Reset

Reset is synchronous and goes to every engine. The guarantees of the engine
that was running hold: ready afterwards, no write on or after the reset
edge, and destination words either old or final.
