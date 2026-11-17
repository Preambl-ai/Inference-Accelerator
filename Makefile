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
SEED ?=
export SEED
export K
export N
export ALLOW_K0

# Find RTL by module name inside any block folder (systolic_array/rtl, memory/rtl, ...)
RTL_ALL := $(wildcard $(PWD)/*/rtl/*.sv)
VERILOG_SOURCES += $(filter %/$(MODULE_NAME).sv,$(RTL_ALL))

COCOTB_TOPLEVEL     = $(MODULE_NAME)
COCOTB_TEST_MODULES = test_$(MODULE_NAME)

# Every block's tb/ and model/ folder is importable, so imports stay flat
empty :=
space := $(empty) $(empty)
export PYTHONPATH := $(subst $(space),:,$(wildcard $(PWD)/*/tb $(PWD)/*/model)):$(PYTHONPATH)

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

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: sweep sweep-pe sweep-array sweep-ctrl sweep-requant sweep-bram sweep-skid sweep-gemm sweep-all

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

sweep-all: sweep sweep-pe sweep-array sweep-ctrl sweep-requant sweep-bram sweep-skid sweep-gemm