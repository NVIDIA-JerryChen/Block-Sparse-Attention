# BSA (Block Sparse Attention) — Development Makefile
#
# Usage:
#   make wheel                - Build the unified CuTe DSL wheel
#   make setup                - Reinstall wheel in a provisioned environment
#   make tt                   - Quick correctness test (blk64+128)
#   make tt BLK=64            - Quick test blk64 only
#   make vt                   - Full pytest suite (blk64+128)
#   make vt BLK=64            - Full pytest blk64 only
#   make ttb                  - Quick correctness test for blk64 backward
#   make vtb                  - Full pytest suite for blk64 backward
#   make bb                   - Performance benchmark
#   make bbb                  - Backward benchmark (blk64)
#   make profile              - Single fwd run for ncu
#   make help                 - Show all targets

SHELL := /bin/bash
PYTHON := python
PYTEST := python -m pytest
TEST_FILE := tests/test_flash_fwd.py
BWD_TEST_FILE := tests/test_flash_bwd.py

# BLK: 64, 128, or "64,128" (default: both)
# BLK ?= 64,128
BLK ?= 64
# BLK ?= 128

# Profile feature toggles (0 or 1)
VAR_BN ?= 0
BLKSZ ?= 0

# Env prefix for profile/bm targets
PROF_ENV := BSA_BLK=$(BLK) BSA_VAR_BN=$(VAR_BN) BSA_BLKSZ=$(BLKSZ)
AGENT_SPACE := agent/agent_space
AGENT_PROFILES := agent/agent_profiles
WHEEL_DIR ?= dist
NCU_DIR := $(AGENT_PROFILES)/ncu
SM120_AOT_DIR ?= $(AGENT_SPACE)/sm120_aot
SM120_AOT_ARGS ?=

.PHONY: wheel setup aot-sm120 tt vt ttb vtb bb bbb profile bm bm-cli clean help

wheel:
	mkdir -p $(WHEEL_DIR)
	rm -f $(WHEEL_DIR)/block_sparse_attention-*.whl
	$(PYTHON) -m pip wheel . --no-deps --wheel-dir $(WHEEL_DIR)

setup: wheel
	$(PYTHON) -m pip install --force-reinstall --no-deps \
		$(WHEEL_DIR)/block_sparse_attention-*.whl

aot-sm120:
	$(PYTHON) -m csrc.fwd.sm120_blk64.aot_build \
		--output-dir $(SM120_AOT_DIR) $(SM120_AOT_ARGS)

tt:
	BSA_BLK=$(BLK) $(PYTHON) -u $(TEST_FILE)

vt:
	BSA_BLK=$(BLK) $(PYTEST) $(TEST_FILE) -v -x -s

bb:
	BSA_BLK=$(BLK) $(PYTHON) -u $(TEST_FILE) benchmark

ttb:
	$(PYTHON) -u $(BWD_TEST_FILE)

vtb:
	$(PYTEST) $(BWD_TEST_FILE) -v -x -s

bbb:
	$(PYTHON) -u $(BWD_TEST_FILE) benchmark

profile:
	$(PROF_ENV) $(PYTHON) -u $(TEST_FILE) profile

bm:
	@mkdir -p $(NCU_DIR)
	$(PROF_ENV) ncu --set full --nvtx --nvtx-include "bsa_attn_fwd_kernel/" \
		-f -o $(NCU_DIR)/bsa_fwd.%p \
		$(PYTHON) -u $(TEST_FILE) profile

NCU_METRICS := launch__registers_per_thread,sm__cycles_elapsed.avg,sm__cycles_elapsed.max,sm__cycles_active.avg,sm__cycles_elapsed.avg.per_second,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed.sum,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_xu.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum

bm-cli:
	@echo "=== ncu register/local-spill/smem analysis (BSA) ==="
	$(PROF_ENV) ncu --nvtx --nvtx-include "bsa_attn_fwd_kernel/" \
		--metrics $(NCU_METRICS) \
		--target-processes all \
		$(PYTHON) -u $(TEST_FILE) profile; true
	@echo ""
	@echo "--- Legend ---"
	@echo "SOL:          sm__throughput = compute SOL%, gpu__compute_memory_throughput = memory SOL%"
	@echo "Duration:     gpu__time_duration (kernel wall time)"
	@echo "Cycles:       sm__cycles_elapsed (avg/max across SMs), sm__cycles_active (active only)"
	@echo "Occupancy:    sm__warps_active = achieved occupancy %"
	@echo "Pipe SOL:     pipe_fma / pipe_shared / pipe_tensor (UTCMMA) = per-pipe utilization %"
	@echo "Instructions: sm__inst_executed (total), pipe_lsu/fma/xu (breakdown)"
	@echo "Registers:    launch__registers_per_thread"
	@echo "Local spills: local_op_ld/st sectors = 0 → no spills"
	@echo "Smem banks:   bank_conflicts = 0 → no conflicts"

clean:
	rm -rf /tmp/$${USER}/flash_attention_cute_dsl_cache/
	rm -rf build/ dist/ artifacts/wheels/ $(AGENT_SPACE)/wheels/ *.egg-info
	rm -rf csrc/fwd/sm100_blk64/build/ csrc/fwd/sm100_blk64/dist/
	rm -rf csrc/fwd/sm100_blk64/*.egg-info csrc/fwd/sm100_blk64/*.so
	rm -rf csrc/fwd/sm100_blk64/cpp/build/ csrc/fwd/sm100_blk64/cpp/dist/
	rm -rf csrc/fwd/sm100_blk64/cpp/*.egg-info csrc/fwd/sm100_blk64/cpp/*.so
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "BSA (Block Sparse Attention) — Development Targets"
	@echo ""
	@echo "  make wheel                      Build unified CuTe DSL wheel"
	@echo "  make setup                      Reinstall wheel (dependencies preinstalled)"
	@echo "  make aot-sm120                  Build SM120 CuTe DSL AOT artifacts"
	@echo "  make tt    [BLK=64|128|64,128]  Quick correctness test (default: both)"
	@echo "  make vt    [BLK=64|128|64,128]  Full pytest suite (default: both)"
	@echo "  make ttb                         Quick bwd correctness (blk64)"
	@echo "  make vtb                         Full bwd pytest suite (blk64)"
	@echo "  make bb                          Performance benchmark"
	@echo "  make bbb                         Backward benchmark (blk64)"
	@echo "  make profile                     Single fwd run for ncu"
	@echo "  make bm                          ncu full profile (NVTX-filtered)"
	@echo "  make bm-cli                      ncu register/spill/smem analysis"
	@echo "  make compare                     Compare BSA vs FA4"
	@echo "  make clean                       Clear compile caches and build artifacts"
	@echo ""
	@echo "  Profile toggles: VAR_BN=0|1  BLKSZ=0|1"
	@echo "    e.g. make bm-cli BLK=64 VAR_BN=1 BLKSZ=1"
	@echo "  SM120 AOT: SM120_AOT_DIR=<root> SM120_AOT_ARGS='<builder args>'"
