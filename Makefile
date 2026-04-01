# BSA (Block Sparse Attention) — Development Makefile
#
# Usage:
#   make setup                - Build blk64 C++ extension
#   make tt                   - Quick correctness test (blk64+128)
#   make tt BLK=64            - Quick test blk64 only
#   make vt                   - Full pytest suite (blk64+128)
#   make vt BLK=64            - Full pytest blk64 only
#   make bb                   - Performance benchmark
#   make profile              - Single fwd run for ncu
#   make help                 - Show all targets

SHELL := /bin/bash
PYTHON := python
PYTEST := python -m pytest
TEST_FILE := test_flash_fwd.py
FA_DIR := /home/scratch.cjerry_sw/next-dsa/flash-attention
FA_TEST := flash_attn/cute/test_flash_fwd_sm100.py

# BLK: 64, 128, or "64,128" (default: both)
# BLK ?= 64,128
BLK ?= 64

.PHONY: setup tt vt bb profile bm bm-cli clean compare help

setup:
	@if [ ! -f third_party/cutlass/include/cutlass/cutlass.h ]; then \
		echo "=== Initializing CUTLASS submodule ===" && \
		git submodule update --init --recursive third_party/cutlass; \
	fi
	@if echo "$(BLK)" | grep -q "64"; then \
		echo "=== Building blk64 C++ extension ===" && \
		$(PYTHON) csrc/fwd/sm100_blk64/setup.py build_ext; \
	fi

tt:
	BSA_BLK=$(BLK) $(PYTHON) -u $(TEST_FILE)

vt:
	BSA_BLK=$(BLK) $(PYTEST) $(TEST_FILE) -v -x -s

bb:
	BSA_BLK=$(BLK) $(PYTHON) -u $(TEST_FILE) benchmark

profile:
	BSA_BLK=$(BLK) $(PYTHON) -u $(TEST_FILE) profile

bm:
	BSA_BLK=$(BLK) ncu --set full --nvtx --nvtx-include "bsa_attn_fwd_kernel/" \
		-f -o profile/bsa_fwd.%p \
		$(PYTHON) -u $(TEST_FILE) profile

NCU_METRICS := launch__registers_per_thread,sm__cycles_elapsed.avg,sm__cycles_elapsed.max,sm__cycles_active.avg,sm__cycles_elapsed.avg.per_second,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed.sum,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_xu.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum

bm-cli:
	@echo "=== ncu register/local-spill/smem analysis (BSA) ==="
	ncu --nvtx --nvtx-include "bsa_attn_fwd_kernel/" \
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

compare:
	@echo "================================================================"
	@echo "  BSA benchmark"
	@echo "================================================================"
	$(PYTHON) -u $(TEST_FILE) benchmark
	@echo ""
	@echo "================================================================"
	@echo "  FA4 benchmark (q_stage=1)"
	@echo "================================================================"
	cd $(FA_DIR) && FA_Q_STAGE=1 $(PYTHON) -u $(FA_TEST) benchmark

clean:
	rm -rf /tmp/$$(USER)/flash_attention_cute_dsl_cache/
	rm -rf build/ csrc/fwd/sm100_blk64/build/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "BSA (Block Sparse Attention) — Development Targets"
	@echo ""
	@echo "  make setup [BLK=64]             Build blk64 C++ extension"
	@echo "  make tt    [BLK=64|128|64,128]  Quick correctness test (default: both)"
	@echo "  make vt    [BLK=64|128|64,128]  Full pytest suite (default: both)"
	@echo "  make bb                          Performance benchmark"
	@echo "  make profile                     Single fwd run for ncu"
	@echo "  make bm                          ncu full profile (NVTX-filtered)"
	@echo "  make bm-cli                      ncu register/spill/smem analysis"
	@echo "  make compare                     Compare BSA vs FA4"
	@echo "  make clean                       Clear compile caches + blk64 build"
