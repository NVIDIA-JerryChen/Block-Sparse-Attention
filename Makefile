# BSA (Block Sparse Attention) — Development Makefile
#
# Usage:
#   make tt                   - Quick correctness test
#   make vt                   - Full pytest suite
#   make bb                   - Performance benchmark
#   make profile              - Single fwd run for ncu
#   make compare              - Compare BSA vs FA4 performance
#   make help                 - Show all targets

SHELL := /bin/bash
PYTHON := python
PYTEST := python -m pytest
TEST_FILE := test_flash_fwd.py
FA_DIR := /home/scratch.cjerry_sw/next-dsa/flash-attention
FA_TEST := flash_attn/cute/test_flash_fwd_sm100.py

.PHONY: setup tt vt bb profile bm bm-cli clean compare help

setup:
	pip install -r requirements.txt

tt:
	$(PYTHON) -u $(TEST_FILE)

vt:
	$(PYTEST) $(TEST_FILE) -v -x -s

bb:
	$(PYTHON) -u $(TEST_FILE) benchmark

profile:
	$(PYTHON) -u $(TEST_FILE) profile

bm:
	ncu --set full --nvtx --nvtx-include "bsa_attn_fwd_kernel/" \
		-f -o profile/bsa_fwd.%p \
		$(PYTHON) -u $(TEST_FILE) profile

NCU_METRICS := launch__registers_per_thread,sm__cycles_elapsed.avg,sm__cycles_elapsed.max,sm__inst_executed.sum,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_xu.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum

bm-cli:
	@echo "=== ncu register/local-spill/smem analysis (BSA) ==="
	ncu --nvtx --nvtx-include "bsa_attn_fwd_kernel/" \
		--metrics $(NCU_METRICS) \
		--target-processes all \
		$(PYTHON) -u $(TEST_FILE) profile; true
	@echo ""
	@echo "--- Legend ---"
	@echo "Cycles:       sm__cycles_elapsed (avg/max across SMs)"
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
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "BSA (Block Sparse Attention) — Development Targets"
	@echo ""
	@echo "  make setup          Install dependencies"
	@echo "  make tt             Quick correctness test"
	@echo "  make vt             Full pytest suite"
	@echo "  make bb             Performance benchmark"
	@echo "  make profile        Single fwd run for ncu"
	@echo "  make bm             ncu full profile (NVTX-filtered)"
	@echo "  make bm-cli         ncu register/spill/smem analysis"
	@echo "  make compare        Compare BSA vs FA4 (q_stage=1 jerry local implementation) performance"
	@echo "  make clean          Clear kernel compile cache"
