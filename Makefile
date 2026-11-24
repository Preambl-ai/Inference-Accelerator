SIM ?= icarus
TOPLEVEL_LANG ?= verilog

MODULE_NAME ?= delay

DEPTH      ?= 4
DATA_WIDTH ?= 8
ACC_WIDTH  ?= 32
BIAS_WIDTH  ?= 32
SAT_WIDTH   ?= 32
M_WIDTH     ?= 32
SHIFT_WIDTH ?= 6
TAG_WIDTH   ?= 4
K          ?= 8
N          ?= 8
K_WIDTH     ?= 16
MEM_LATENCY ?= 1
ALLOW_K0    ?= 0
ADDR_WIDTH   ?= 10
READ_LATENCY ?= 1
B_ADDR_W     ?= 13
NBUF         ?= 3
ALLOW_BAD_SEL ?= 0
# UART bit period in clocks for the uart, bridge and top sims
CLKS_PER_BIT ?= 16
# bridge inter-byte timeout in bit times (the board uses 1000)
TIMEOUT_BITS ?= 40
# conv_layer needs a source and a destination bank
CONV_NBUF ?= 2
LAYERS ?=
export LAYERS
SEED ?=
export SEED
export K
export N
export ALLOW_K0
export ALLOW_BAD_SEL

# Find RTL by module name inside any block folder (systolic_array/rtl, memory/rtl, ...)
RTL_ALL := $(wildcard $(PWD)/*/rtl/*.sv)
VERILOG_SOURCES += $(filter %/$(MODULE_NAME).sv,$(RTL_ALL))

COCOTB_TOPLEVEL     = $(MODULE_NAME)
COCOTB_TEST_MODULES = test_$(MODULE_NAME)

# Every block's tb/ and model/ folder is importable, so imports stay flat
empty :=
space := $(empty) $(empty)
# host/ holds the protocol module the TBs share with the real host driver
export PYTHONPATH := $(subst $(space),:,$(wildcard $(PWD)/*/tb $(PWD)/*/model) $(PWD)/host):$(PYTHONPATH)

COMPILE_ARGS += -g2012

ifeq ($(MODULE_NAME),delay)
COMPILE_ARGS += -Pdelay.DEPTH=$(DEPTH)
COMPILE_ARGS += -Pdelay.DATA_WIDTH=$(DATA_WIDTH)
SIM_BUILD = sim_build/delay_depth_$(DEPTH)
endif

ifeq ($(MODULE_NAME),pe)
COMPILE_ARGS += -Ppe.DATA_WIDTH=$(DATA_WIDTH)
COMPILE_ARGS += -Ppe.ACC_WIDTH=$(ACC_WIDTH)
SIM_BUILD = sim_build/pe_k_$(K)_acc_$(ACC_WIDTH)
endif

# The array pulls in its children. Icarus does not care about file order.
ifeq ($(MODULE_NAME),systolic_array)
VERILOG_SOURCES += $(PWD)/systolic_array/rtl/pe.sv $(PWD)/systolic_array/rtl/delay.sv
COMPILE_ARGS += -Psystolic_array.N=$(N)
COMPILE_ARGS += -Psystolic_array.DATA_WIDTH=$(DATA_WIDTH)
COMPILE_ARGS += -Psystolic_array.ACC_WIDTH=$(ACC_WIDTH)
SIM_BUILD = sim_build/array_n_$(N)_k_$(K)
endif

ifeq ($(MODULE_NAME),ctrl_fsm)
VERILOG_SOURCES += $(PWD)/systolic_array/rtl/delay.sv
COMPILE_ARGS += -Pctrl_fsm.N=$(N)
COMPILE_ARGS += -Pctrl_fsm.K_WIDTH=$(K_WIDTH)
COMPILE_ARGS += -Pctrl_fsm.MEM_LATENCY=$(MEM_LATENCY)
# the k_in == 0 test needs the simulation-only $$fatal switched off
ifeq ($(ALLOW_K0),1)
PLUSARGS += +ctrl_fsm_allow_k0
endif
SIM_BUILD = sim_build/ctrl_n_$(N)_k_$(K)_kw_$(K_WIDTH)_ml_$(MEM_LATENCY)
endif

ifeq ($(MODULE_NAME),requant)
COMPILE_ARGS += -Prequant.ACC_WIDTH=$(ACC_WIDTH)
COMPILE_ARGS += -Prequant.BIAS_WIDTH=$(BIAS_WIDTH)
COMPILE_ARGS += -Prequant.SAT_WIDTH=$(SAT_WIDTH)
COMPILE_ARGS += -Prequant.M_WIDTH=$(M_WIDTH)
COMPILE_ARGS += -Prequant.SHIFT_WIDTH=$(SHIFT_WIDTH)
COMPILE_ARGS += -Prequant.DATA_WIDTH=$(DATA_WIDTH)
COMPILE_ARGS += -Prequant.TAG_WIDTH=$(TAG_WIDTH)
SIM_BUILD = sim_build/requant_acc_$(ACC_WIDTH)_bias_$(BIAS_WIDTH)_sat_$(SAT_WIDTH)_m_$(M_WIDTH)_sh_$(SHIFT_WIDTH)_dw_$(DATA_WIDTH)_tag_$(TAG_WIDTH)
endif

ifeq ($(MODULE_NAME),bram)
COMPILE_ARGS += -Pbram.DATA_WIDTH=$(DATA_WIDTH)
COMPILE_ARGS += -Pbram.ADDR_WIDTH=$(ADDR_WIDTH)
COMPILE_ARGS += -Pbram.READ_LATENCY=$(READ_LATENCY)
SIM_BUILD = sim_build/bram_dw_$(DATA_WIDTH)_aw_$(ADDR_WIDTH)_rl_$(READ_LATENCY)
endif

ifeq ($(MODULE_NAME),skid_buffer)
COMPILE_ARGS += -Pskid_buffer.DATA_WIDTH=$(DATA_WIDTH)
SIM_BUILD = sim_build/skid_dw_$(DATA_WIDTH)
endif

ifeq ($(MODULE_NAME),gemm_tile)
VERILOG_SOURCES += $(PWD)/systolic_array/rtl/ctrl_fsm.sv $(PWD)/systolic_array/rtl/systolic_array.sv
VERILOG_SOURCES += $(PWD)/systolic_array/rtl/pe.sv $(PWD)/systolic_array/rtl/delay.sv
VERILOG_SOURCES += $(PWD)/systolic_array/rtl/requant.sv
VERILOG_SOURCES += $(PWD)/memory/rtl/bram.sv $(PWD)/axis/rtl/skid_buffer.sv
COMPILE_ARGS += -Pgemm_tile.N=$(N)
COMPILE_ARGS += -Pgemm_tile.DATA_WIDTH=$(DATA_WIDTH)
COMPILE_ARGS += -Pgemm_tile.ACC_WIDTH=$(ACC_WIDTH)
COMPILE_ARGS += -Pgemm_tile.READ_LATENCY=$(READ_LATENCY)
COMPILE_ARGS += -Pgemm_tile.B_ADDR_W=$(B_ADDR_W)
# the bad-K test needs both simulation-only $$fatal checks switched off
ifeq ($(ALLOW_K0),1)
PLUSARGS += +ctrl_fsm_allow_k0 +gemm_tile_allow_bad_k
endif
SIM_BUILD = sim_build/gemm_n_$(N)_k_$(K)_rl_$(READ_LATENCY)_ba_$(B_ADDR_W)
endif

ifeq ($(MODULE_NAME),im2col_reader)
VERILOG_SOURCES += $(PWD)/systolic_array/rtl/delay.sv
COMPILE_ARGS += -Pim2col_reader.N=$(N)
COMPILE_ARGS += -Pim2col_reader.DATA_WIDTH=$(DATA_WIDTH)
COMPILE_ARGS += -Pim2col_reader.ADDR_W=$(ADDR_WIDTH)
COMPILE_ARGS += -Pim2col_reader.READ_LATENCY=$(READ_LATENCY)
SIM_BUILD = sim_build/im2col_n_$(N)_aw_$(ADDR_WIDTH)_rl_$(READ_LATENCY)
endif

# everything a conv layer is built from, below conv_layer / layer_engine
CONV_SRCS  = $(PWD)/conv_layer/rtl/conv_core.sv
CONV_SRCS += $(PWD)/gemm_tile/rtl/gemm_tile.sv $(PWD)/im2col/rtl/im2col_reader.sv
CONV_SRCS += $(PWD)/memory/rtl/act_buffer.sv $(PWD)/memory/rtl/bram.sv
CONV_SRCS += $(PWD)/systolic_array/rtl/ctrl_fsm.sv $(PWD)/systolic_array/rtl/systolic_array.sv
CONV_SRCS += $(PWD)/systolic_array/rtl/pe.sv $(PWD)/systolic_array/rtl/delay.sv
CONV_SRCS += $(PWD)/systolic_array/rtl/requant.sv $(PWD)/axis/rtl/skid_buffer.sv

ifeq ($(MODULE_NAME),conv_layer)
VERILOG_SOURCES += $(CONV_SRCS)
COMPILE_ARGS += -Pconv_layer.N=$(N)
COMPILE_ARGS += -Pconv_layer.ADDR_WIDTH=$(ADDR_WIDTH)
COMPILE_ARGS += -Pconv_layer.READ_LATENCY=$(READ_LATENCY)
COMPILE_ARGS += -Pconv_layer.NBUF=$(CONV_NBUF)
SIM_BUILD = sim_build/conv_n_$(N)_aw_$(ADDR_WIDTH)_rl_$(READ_LATENCY)_nb_$(CONV_NBUF)
endif

ifeq ($(MODULE_NAME),uart_loopback)
VERILOG_SOURCES += $(PWD)/uart/rtl/uart_tx.sv $(PWD)/uart/rtl/uart_rx.sv $(PWD)/uart/tb/uart_loopback.sv
COMPILE_ARGS += -Puart_loopback.CLKS_PER_BIT=$(CLKS_PER_BIT)
SIM_BUILD = sim_build/uart_cpb_$(CLKS_PER_BIT)
endif

ifeq ($(MODULE_NAME),csr)
COMPILE_ARGS += -Pcsr.N=$(N)
COMPILE_ARGS += -Pcsr.ADDR_WIDTH=$(ADDR_WIDTH)
COMPILE_ARGS += -Pcsr.READ_LATENCY=$(READ_LATENCY)
COMPILE_ARGS += -Pcsr.NBUF=$(NBUF)
SIM_BUILD = sim_build/csr_n_$(N)_aw_$(ADDR_WIDTH)_rl_$(READ_LATENCY)_nb_$(NBUF)
endif

ifeq ($(MODULE_NAME),top)
VERILOG_SOURCES += $(PWD)/csr/rtl/csr.sv $(PWD)/uart/rtl/uart_rx.sv $(PWD)/uart/rtl/uart_tx.sv
VERILOG_SOURCES += $(PWD)/uart/rtl/uart_bridge.sv $(PWD)/layer_engine/rtl/layer_engine.sv
VERILOG_SOURCES += $(PWD)/pool/rtl/pool_layer.sv $(CONV_SRCS)
COMPILE_ARGS += -Ptop.N=$(N)
COMPILE_ARGS += -Ptop.READ_LATENCY=$(READ_LATENCY)
COMPILE_ARGS += -Ptop.ADDR_WIDTH=$(ADDR_WIDTH)
# fast sim: a short bit period and timeout, a heartbeat you can see toggle
COMPILE_ARGS += -Ptop.CLKS_PER_BIT=$(CLKS_PER_BIT)
COMPILE_ARGS += -Ptop.TIMEOUT_BITS=$(TIMEOUT_BITS)
COMPILE_ARGS += -Ptop.HB_BIT=6
SIM_BUILD = sim_build/top_n_$(N)_rl_$(READ_LATENCY)_aw_$(ADDR_WIDTH)_cpb_$(CLKS_PER_BIT)
endif

ifeq ($(MODULE_NAME),pool_harness)
VERILOG_SOURCES += $(PWD)/pool/tb/pool_harness.sv $(PWD)/pool/rtl/pool_layer.sv
VERILOG_SOURCES += $(PWD)/memory/rtl/act_buffer.sv $(PWD)/memory/rtl/bram.sv $(PWD)/systolic_array/rtl/delay.sv
COMPILE_ARGS += -Ppool_harness.N=$(N)
COMPILE_ARGS += -Ppool_harness.ADDR_WIDTH=$(ADDR_WIDTH)
COMPILE_ARGS += -Ppool_harness.READ_LATENCY=$(READ_LATENCY)
COMPILE_ARGS += -Ppool_harness.NBUF=$(CONV_NBUF)
SIM_BUILD = sim_build/pool_n_$(N)_aw_$(ADDR_WIDTH)_rl_$(READ_LATENCY)_nb_$(CONV_NBUF)
endif

ifeq ($(MODULE_NAME),layer_engine)
VERILOG_SOURCES += $(PWD)/pool/rtl/pool_layer.sv
VERILOG_SOURCES += $(CONV_SRCS)
COMPILE_ARGS += -Player_engine.N=$(N)
COMPILE_ARGS += -Player_engine.ADDR_WIDTH=$(ADDR_WIDTH)
COMPILE_ARGS += -Player_engine.READ_LATENCY=$(READ_LATENCY)
COMPILE_ARGS += -Player_engine.NBUF=$(CONV_NBUF)
SIM_BUILD = sim_build/engine_n_$(N)_aw_$(ADDR_WIDTH)_rl_$(READ_LATENCY)_nb_$(CONV_NBUF)
endif

ifeq ($(MODULE_NAME),act_buffer)
VERILOG_SOURCES += $(PWD)/memory/rtl/bram.sv
COMPILE_ARGS += -Pact_buffer.N=$(N)
COMPILE_ARGS += -Pact_buffer.DATA_WIDTH=$(DATA_WIDTH)
COMPILE_ARGS += -Pact_buffer.ADDR_WIDTH=$(ADDR_WIDTH)
COMPILE_ARGS += -Pact_buffer.READ_LATENCY=$(READ_LATENCY)
COMPILE_ARGS += -Pact_buffer.NBUF=$(NBUF)
# the bad-select test needs the simulation-only $$fatal switched off
ifeq ($(ALLOW_BAD_SEL),1)
PLUSARGS += +act_buffer_allow_bad_sel
endif
SIM_BUILD = sim_build/act_n_$(N)_dw_$(DATA_WIDTH)_aw_$(ADDR_WIDTH)_rl_$(READ_LATENCY)_nb_$(NBUF)
endif

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: sweep sweep-pe sweep-array sweep-ctrl sweep-requant sweep-bram sweep-skid sweep-gemm sweep-im2col sweep-act sweep-conv sweep-uart sweep-csr sweep-top sweep-pool sweep-engine check-flow sweep-all

sweep:
	@for d in 0 1 2 4 8; do \
	  echo "=== DEPTH=$$d ==="; \
	  $(MAKE) -s MODULE_NAME=delay DEPTH=$$d || exit 1; \
	done
	@echo "=== delay sweep passed ==="

sweep-pe:
	@for k in 1 2 8 64 255; do \
	  echo "=== K=$$k ==="; \
	  $(MAKE) -s MODULE_NAME=pe K=$$k || exit 1; \
	done
	@echo "=== K=16 ACC_WIDTH=18 (deliberate overflow) ==="
	@$(MAKE) -s MODULE_NAME=pe K=16 ACC_WIDTH=18 || exit 1
	@echo "=== pe sweep passed ==="

sweep-array:
	@for n in 2 3 4 8; do \
	  for k in 1 4 16; do \
	    echo "=== N=$$n K=$$k ==="; \
	    $(MAKE) -s MODULE_NAME=systolic_array N=$$n K=$$k || exit 1; \
	  done; \
	done
	@echo "=== array sweep passed ==="

sweep-ctrl:
	@for n in 1 2 4 8; do \
	  for k in 1 2 16 300; do \
	    for ml in 0 1 2; do \
	      echo "=== N=$$n K=$$k MEM_LATENCY=$$ml ==="; \
	      $(MAKE) -s MODULE_NAME=ctrl_fsm N=$$n K=$$k MEM_LATENCY=$$ml || exit 1; \
	    done; \
	  done; \
	done
	@for ml in 0 1 2; do \
	  echo "=== N=4 K=15 K_WIDTH=4 MEM_LATENCY=$$ml (K at the K_WIDTH limit) ==="; \
	  $(MAKE) -s MODULE_NAME=ctrl_fsm N=4 K=15 K_WIDTH=4 MEM_LATENCY=$$ml || exit 1; \
	  echo "=== N=4 K=5 MEM_LATENCY=$$ml ALLOW_K0=1 (k_in == 0 start) ==="; \
	  $(MAKE) -s MODULE_NAME=ctrl_fsm N=4 K=5 MEM_LATENCY=$$ml ALLOW_K0=1 || exit 1; \
	done
	@echo "=== ctrl_fsm sweep passed ==="

# each line is one width set; SHIFT_WIDTH has to shrink with the product width
sweep-requant:
	@echo "=== requant defaults ==="
	@$(MAKE) -s MODULE_NAME=requant || exit 1
	@echo "=== SAT_WIDTH=33 (full sum width, saturation unreachable) ==="
	@$(MAKE) -s MODULE_NAME=requant SAT_WIDTH=33 || exit 1
	@echo "=== SAT_WIDTH=24 SHIFT_WIDTH=5 ==="
	@$(MAKE) -s MODULE_NAME=requant SAT_WIDTH=24 SHIFT_WIDTH=5 || exit 1
	@echo "=== M_WIDTH=16 SHIFT_WIDTH=5 ==="
	@$(MAKE) -s MODULE_NAME=requant M_WIDTH=16 SHIFT_WIDTH=5 || exit 1
	@echo "=== ACC_WIDTH=24 BIAS_WIDTH=16 SAT_WIDTH=20 M_WIDTH=12 SHIFT_WIDTH=5 ==="
	@$(MAKE) -s MODULE_NAME=requant ACC_WIDTH=24 BIAS_WIDTH=16 SAT_WIDTH=20 M_WIDTH=12 SHIFT_WIDTH=5 || exit 1
	@echo "=== DATA_WIDTH=16 TAG_WIDTH=8 ==="
	@$(MAKE) -s MODULE_NAME=requant DATA_WIDTH=16 TAG_WIDTH=8 || exit 1
	@echo "=== TAG_WIDTH=1 ==="
	@$(MAKE) -s MODULE_NAME=requant TAG_WIDTH=1 || exit 1
	@echo "=== requant sweep passed ==="

sweep-bram:
	@for dw in 8 53 64; do \
	  for aw in 2 4 10; do \
	    for rl in 1 2; do \
	      echo "=== DATA_WIDTH=$$dw ADDR_WIDTH=$$aw READ_LATENCY=$$rl ==="; \
	      $(MAKE) -s MODULE_NAME=bram DATA_WIDTH=$$dw ADDR_WIDTH=$$aw READ_LATENCY=$$rl || exit 1; \
	    done; \
	  done; \
	done
	@echo "=== bram sweep passed ==="

sweep-skid:
	@for dw in 1 8 67; do \
	  echo "=== DATA_WIDTH=$$dw ==="; \
	  $(MAKE) -s MODULE_NAME=skid_buffer DATA_WIDTH=$$dw || exit 1; \
	done
	@echo "=== skid_buffer sweep passed ==="

sweep-gemm:
	@for n in 2 4 8; do \
	  for k in 1 16 100; do \
	    for rl in 1 2; do \
	      echo "=== N=$$n K=$$k READ_LATENCY=$$rl ==="; \
	      $(MAKE) -s MODULE_NAME=gemm_tile N=$$n K=$$k READ_LATENCY=$$rl B_ADDR_W=10 || exit 1; \
	    done; \
	  done; \
	done
	@for rl in 1 2; do \
	  echo "=== N=1 K=16 READ_LATENCY=$$rl ==="; \
	  $(MAKE) -s MODULE_NAME=gemm_tile N=1 K=16 READ_LATENCY=$$rl B_ADDR_W=10 || exit 1; \
	  echo "=== N=4 K=5 READ_LATENCY=$$rl ALLOW_K0=1 (bad k_in starts) ==="; \
	  $(MAKE) -s MODULE_NAME=gemm_tile N=4 K=5 READ_LATENCY=$$rl B_ADDR_W=10 ALLOW_K0=1 || exit 1; \
	  echo "=== N=8 K=576 READ_LATENCY=$$rl B_ADDR_W=13 (full-size weight memory, K = K_MAX test) ==="; \
	  $(MAKE) -s MODULE_NAME=gemm_tile N=8 K=576 READ_LATENCY=$$rl B_ADDR_W=13 || exit 1; \
	done
	@echo "=== gemm_tile sweep passed ==="

sweep-im2col:
	@for n in 2 4 8; do \
	  for rl in 1 2; do \
	    echo "=== N=$$n READ_LATENCY=$$rl ==="; \
	    $(MAKE) -s MODULE_NAME=im2col_reader N=$$n READ_LATENCY=$$rl ADDR_WIDTH=12 || exit 1; \
	  done; \
	done
	@for rl in 1 2; do \
	  echo "=== N=8 READ_LATENCY=$$rl ADDR_WIDTH=22 (large images) ==="; \
	  $(MAKE) -s MODULE_NAME=im2col_reader N=8 READ_LATENCY=$$rl ADDR_WIDTH=22 || exit 1; \
	done
	@echo "=== im2col_reader sweep passed ==="

# NBUF=3 leaves select 3 unused and NBUF=1 leaves select 1 unused: those are
# the configs where an out-of-range select exists, so ALLOW_BAD_SEL runs there
sweep-act:
	@for n in 1 4 8; do \
	  for rl in 1 2; do \
	    for nb in 2 3 4; do \
	      echo "=== N=$$n READ_LATENCY=$$rl NBUF=$$nb ==="; \
	      $(MAKE) -s MODULE_NAME=act_buffer N=$$n READ_LATENCY=$$rl NBUF=$$nb ADDR_WIDTH=6 || exit 1; \
	    done; \
	  done; \
	done
	@for rl in 1 2; do \
	  echo "=== N=4 READ_LATENCY=$$rl NBUF=1 (one bank, 1-bit select) ==="; \
	  $(MAKE) -s MODULE_NAME=act_buffer N=4 READ_LATENCY=$$rl NBUF=1 ADDR_WIDTH=6 || exit 1; \
	  echo "=== N=4 READ_LATENCY=$$rl NBUF=1 ALLOW_BAD_SEL=1 (select 1 names no bank) ==="; \
	  $(MAKE) -s MODULE_NAME=act_buffer N=4 READ_LATENCY=$$rl NBUF=1 ADDR_WIDTH=6 ALLOW_BAD_SEL=1 || exit 1; \
	  echo "=== N=4 READ_LATENCY=$$rl NBUF=3 ALLOW_BAD_SEL=1 (select 3 names no bank) ==="; \
	  $(MAKE) -s MODULE_NAME=act_buffer N=4 READ_LATENCY=$$rl NBUF=3 ADDR_WIDTH=6 ALLOW_BAD_SEL=1 || exit 1; \
	  echo "=== N=3 DATA_WIDTH=5 READ_LATENCY=$$rl NBUF=3 ADDR_WIDTH=2 (odd lanes, 4-word banks) ==="; \
	  $(MAKE) -s MODULE_NAME=act_buffer N=3 DATA_WIDTH=5 READ_LATENCY=$$rl NBUF=3 ADDR_WIDTH=2 || exit 1; \
	  echo "=== N=8 READ_LATENCY=$$rl NBUF=3 ADDR_WIDTH=10 (full-depth banks) ==="; \
	  $(MAKE) -s MODULE_NAME=act_buffer N=8 READ_LATENCY=$$rl NBUF=3 ADDR_WIDTH=10 || exit 1; \
	done
	@echo "=== act_buffer sweep passed ==="

# conv_layer runs the whole TB (random layers included) on every config. the
# NBUF=3 run leaves bank select 3 unused, so the E_SEL "select >= NBUF" case runs
sweep-conv:
	@for n in 2 4 8; do \
	  for rl in 1 2; do \
	    echo "=== N=$$n READ_LATENCY=$$rl ==="; \
	    $(MAKE) -s MODULE_NAME=conv_layer N=$$n READ_LATENCY=$$rl ADDR_WIDTH=9 || exit 1; \
	  done; \
	done
	@echo "=== N=4 READ_LATENCY=2 CONV_NBUF=3 (bank select 3 names no bank) ==="
	@$(MAKE) -s MODULE_NAME=conv_layer N=4 READ_LATENCY=2 ADDR_WIDTH=9 CONV_NBUF=3 || exit 1
	@echo "=== conv_layer sweep passed ==="

# UART: a bit period of 4 clocks (the shortest uart_rx accepts), 16, and the
# board's 100 (1 Mbaud at 100 MHz)
sweep-uart:
	@for c in 4 16 100; do \
	  echo "=== uart CLKS_PER_BIT=$$c ==="; \
	  $(MAKE) -s MODULE_NAME=uart_loopback CLKS_PER_BIT=$$c || exit 1; \
	done
	@echo "=== uart sweep passed ==="

# CSR: NBUF=3 leaves bank select 3 unmapped, so the DECERR case runs there
sweep-csr:
	@for n in 2 4 8; do \
	  for rl in 1 2; do \
	    for nb in 2 3; do \
	      echo "=== csr N=$$n READ_LATENCY=$$rl NBUF=$$nb ==="; \
	      $(MAKE) -s MODULE_NAME=csr N=$$n READ_LATENCY=$$rl NBUF=$$nb ADDR_WIDTH=6 || exit 1; \
	    done; \
	  done; \
	done
	@echo "=== csr sweep passed ==="

# the whole board through its UART pins, with the fast bit period
sweep-top:
	@for cfg in "2 1" "4 1" "4 2"; do \
	  set -- $$cfg; \
	  echo "=== top N=$$1 READ_LATENCY=$$2 CLKS_PER_BIT=8 ==="; \
	  $(MAKE) -s MODULE_NAME=top N=$$1 READ_LATENCY=$$2 ADDR_WIDTH=8 CLKS_PER_BIT=8 TIMEOUT_BITS=40 || exit 1; \
	done
	@echo "=== top sweep passed ==="

# pool_layer on the real act_buffer. NBUF=3 leaves bank select 3 unused, so the
# E_SEL "select >= NBUF" case runs there
sweep-pool:
	@for n in 1 2 4 8; do \
	  for rl in 1 2; do \
	    echo "=== pool N=$$n READ_LATENCY=$$rl ==="; \
	    $(MAKE) -s MODULE_NAME=pool_harness N=$$n READ_LATENCY=$$rl ADDR_WIDTH=9 || exit 1; \
	  done; \
	done
	@echo "=== pool N=4 READ_LATENCY=2 CONV_NBUF=3 (bank select 3 names no bank) ==="
	@$(MAKE) -s MODULE_NAME=pool_harness N=4 READ_LATENCY=2 ADDR_WIDTH=9 CONV_NBUF=3 || exit 1
	@echo "=== pool sweep passed ==="

# layer_engine: conv and pool sharing one act_buffer. NBUF=3 runs the E_SEL
# "select >= NBUF" cases and a third bank
sweep-engine:
	@for n in 2 4 8; do \
	  for rl in 1 2; do \
	    echo "=== layer_engine N=$$n READ_LATENCY=$$rl ==="; \
	    $(MAKE) -s MODULE_NAME=layer_engine N=$$n READ_LATENCY=$$rl ADDR_WIDTH=10 || exit 1; \
	  done; \
	done
	@echo "=== layer_engine N=4 READ_LATENCY=2 CONV_NBUF=3 ==="
	@$(MAKE) -s MODULE_NAME=layer_engine N=4 READ_LATENCY=2 ADDR_WIDTH=10 CONV_NBUF=3 || exit 1
	@echo "=== layer_engine sweep passed ==="

# fails if a sweep target is missing from sweep-all, or the Makefile lost its
# trailing newline (how sweep-conv once got skipped without any failure)
check-flow:
	@python3 ci/check_flow.py Makefile

sweep-all: check-flow sweep sweep-pe sweep-array sweep-ctrl sweep-requant sweep-bram sweep-skid sweep-gemm sweep-im2col sweep-act sweep-conv sweep-uart sweep-csr sweep-top sweep-pool sweep-engine
