# CLAUDE.md

## Project Overview

**Supported**: bf16/fp16, MHA/GQA/MQA, hdim 64/96/128/(192,128), persistent scheduling, CLC persistent scheduling, pack_gqa, block-sparse attention
**Not supported**: causal, local, mask_mod, score_mod, split-kv, paged_kv, softcap, varlen

## Build & Install

```bash
pip install -r requirements.txt
```

Dependencies: `nvidia-cutlass-dsl>=4.4.1`, `torch`, `einops`, `quack-kernels>=0.2.10`.

## Running Tests

```bash
# Quick correctness test
make tt

# Full parametric test via pytest
make vt

# Benchmark
make bb

# Profile (for ncu)
make profile
```

## Code Architecture

### Public API (`bsa_attn_interface.py`)
- `bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes, ...)` — SM100 block-sparse forward attention

Tensor layout: `(batch, seqlen, num_heads, head_dim)`, last dim contiguous, 16-byte aligned.

#### Block-Sparse Parameters (mandatory)
- `q2k_block_index`: `(batch, num_heads, num_q_blocks, max_kv_blocks)` int32 — per Q-block list of KV block indices to attend to
- `block_sparse_num`: int (runtime, even, >= 2) — number of KV blocks each Q block attends to. Ignored when `q2k_block_nums` is provided
- `block_sizes`: `(num_kv_blocks,)` int32 — actual token count per KV block (for masking padding positions within a tile)

#### Variable Block-Sparse Parameters (optional)
- `q2k_block_nums`: `(batch, num_heads, num_q_blocks)` int32 — per-Q-block number of KV blocks to attend to (each value >= 1). When provided, `block_sparse_num` is ignored, and for each `(batch, head, m_block)` the first `q2k_block_nums[batch, head, m_block]` entries in `q2k_block_index` are valid. Odd values are handled internally by padding to even with a phantom block (fully masked, zero contribution)

For dense (full) attention, construct `q2k_block_index = [0,1,...,N-1]` for all Q blocks, `block_sparse_num = N`, and `block_sizes = [tile_n]*N` (with last block adjusted for seqlen remainder). See `make_dense_block_sparse_args()` in `test_flash_fwd.py`.

### Forward Kernel (`csrc/fwd/sm100/flash_fwd_sm100.py`)
- `FlashAttentionForwardSm100`: Blackwell forward, qstage=1 only

### Core Abstractions (`csrc/fwd/sm100/`)
- `softmax.py` — `SoftmaxSm100`: online softmax with row_max/row_sum tracking
- `mask.py` — `AttentionMask`: seqlen-only masking (R2P bitmask); `apply_block_size_mask`: per-block variable-size masking
- `block_info.py` — `BlockInfo`: tile dimensions, `get_n_block_idx()` for block-sparse index lookup
- `seqlen_info.py` — `SeqlenInfoQK`: sequence length tracking (non-varlen)
- `pipeline.py` — `PipelineStateSimple`: circular buffer index/phase management
- `tile_scheduler.py` — `SingleTileScheduler`, `StaticPersistentTileScheduler` (with CLC dynamic scheduling support via `SchedulingMode.CLC`)
- `named_barrier.py` — `NamedBarrierFwdSm100`
- `pack_gqa.py` — GQA head packing
- `blackwell_helpers.py` — SM100 UMMA-based GEMM, PTX-optimized paths (2CTA WIP, not yet functional for block-sparse)
- `mma_sm100_desc.py` — Hardware MMA descriptor enums
- `utils.py` — Hash functions, reductions, shift ops, warp helpers
- `fast_math.py` — exp2 polynomial coefficients
- `cute_dsl_utils.py` — Tensor alignment helpers, patched compile

### Utils (`utils/`)
- `cache_utils.py` — JIT compilation cache management
- `fa_logging.py` — `FA_LOG_LEVEL` debug logging
- `testing.py` — `attention_ref`, tolerance helpers
- `bench_utils.py` — flops computation
- `benchmark.py` — `benchmark_forward`

## Key Patterns

- Compile-time constants use `cutlass.Constexpr[type]` for kernel specialization
- `block_sparse_num` is a runtime `Int32` parameter (not compile-time); different values do not require recompilation
- `q2k_block_nums` enables per-Q-block variable KV block counts; uses a separate compile path (`has_variable_block_nums` in compile key). Pipeline phases (`phase_s0`/`phase_s1`) persist across tiles to handle variable `block_iter_count` in persistent scheduling. Odd values are rounded up to even internally; the phantom block uses a clamped index (`max_i` in `get_n_block_idx`) and `block_size=0` mask for zero contribution
- Forward execution: load Q tile → loop over K/V blocks selected by `q2k_block_index` (pipelined) → online softmax with per-block `block_sizes` masking → store O and LSE
- Load order for N KV blocks: K[idx(N-1)], Q, K[idx(N-2)], {V[idx(N-1-i)], K[idx(N-3-i)]}×(N-2), V[idx(1)], V[idx(0)] — where `idx(i) = q2k_block_index[batch, head, m_block, i]`
- 2CTA instructions (hdim=128): WIP, not yet functional for block-sparse. 
- CLC persistent scheduling: enabled via `BSA_CLC=1` env var. Uses NVIDIA's Coordinated Launch Cluster (CLC) for dynamic work-stealing across CTAs. Requires `use_tma_KV=True`, cluster N=1, cluster M∈{1,2}. A dedicated scheduler warp (warp 15 on leader CTA) issues async CLC queries; all warps synchronize via `PipelineClcFetchAsync` mbarriers. Compute warps use `tile_scheduler.consumer_advance()` which encapsulates CLC pipeline wait/release.
