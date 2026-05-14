# CLAUDE.md

## Project Overview

**Supported**: bf16/fp16, MHA/GQA/MQA, hdim 64/96/128/(192,128), persistent scheduling, CLC persistent scheduling, pack_gqa, block-sparse attention, forward + backward (blk64 only)
**Not supported**: causal, local, mask_mod, score_mod, split-kv, paged_kv, softcap, varlen

Forward kernel backends:
- **blk128** (`csrc/fwd/sm100_blk128/`): CuTe DSL / JIT compiled, tile_m=128, tile_n=128
- **blk64** (`csrc/fwd/sm100_blk64/`): C++ AOT / CUTLASS compiled, tile_m=64, tile_n=64 (kDualCols=256, kSparseBlocksPerKV=4), bf16 only

Backward kernel backend:
- **blk64** (`csrc/bwd/sm100_blk64/`): CuTe DSL / JIT compiled, sparse_block_size=64, head_dim=128, MHA only, bf16 only. Pairs with the blk64 forward (same 64×64 sparsity block)

## Build & Install

```bash
pip install -r requirements.txt

# Build blk64 C++ extension (optional, required for blk64 tests)
# Builds a wheel via bdist_wheel and pip installs it
make setup
```

Dependencies: `nvidia-cutlass-dsl>=4.4.1`, `torch`, `einops`, `quack-kernels>=0.2.10`.
blk64 additionally requires CUTLASS headers (git submodule at `third_party/cutlass`).

## Running Tests

```bash
# Quick correctness test (default: blk128)
make tt

# Quick test with BLK selection
make tt BLK=64          # blk64 only
make tt BLK=64,128      # both backends

# Full parametric test via pytest
make vt
make vt BLK=64

# Benchmark
make bb

# Backward (blk64 only — bf16, MHA, hdim=128)
make ttb                # quick correctness
make vtb                # parametric pytest
make bbb                # interface benchmark

# ncu full / register-spill-smem analysis
make bm-cli
```

## Code Architecture

### Public API (`bsa_attn_interface.py`)
- `bsa_attn_fwd(q, k, v, q2k_block_index, max_topk, block_sizes, ...)` — SM100 block-sparse forward attention (blk128 backend)
- `bsa_attn_bwd(dout, q, k, v, out, lse, q2k_block_index, max_topk, block_sizes=None, q2k_block_nums=None, softmax_scale=None, dq=None, dk=None, dv=None, k2q_row_ptr=None, k2q_q_indices=None, k2q_schedule_metadata=None, k2q_schedule_work_counts=None, workspace=None)` — SM100 block-sparse backward attention (blk64 backend, bf16/MHA/hdim=128). q/k/v/out/dout/dq/dk/dv are all in **BHSD** (`(batch, num_heads, seqlen, head_dim)`) layout — differs from `bsa_attn_fwd`'s BSHD. Returns `(dq, dk, dv)`. Reuses the `q2k_block_index` / `max_topk` / `block_sizes` / `q2k_block_nums` from the forward call; the wrapper builds total-packed CSR k2q metadata plus fused qrange-split schedule and allocates the fp32 workspace needed by the kernel. Callers may pass all four prebuilt CSR/schedule tensors to benchmark conversion cost separately.
- `convert_q2k_to_k2q_csr(q2k_block_index, max_topk, num_kv_blocks, q2k_block_nums=None, return_schedule=False)` — helper that inverts the forward's per-Q-block KV-attendee list into total-packed CSR k2q metadata. With `return_schedule=True`, it also returns the fixed qrange-split schedule used by `bsa_attn_bwd`.

Tensor layout: `(batch, seqlen, num_heads, head_dim)`, last dim contiguous, 16-byte aligned.

#### Block-Sparse Parameters (mandatory)
- `q2k_block_index`: `(batch, num_heads, num_q_blocks, max_topk)` int32 — per Q-block list of KV block indices to attend to
- `max_topk`: int (runtime, even, >= 2 for blk128; >= 1 for blk64) — fixed KV-block count per Q block, or q2k storage capacity / maximum topK when `q2k_block_nums` is provided
- `block_sizes`: `(num_kv_blocks,)` int32 — actual token count per KV block (for masking padding positions within a tile)

#### Variable Block-Sparse Parameters (optional)
- `q2k_block_nums`: `(batch, num_heads, num_q_blocks)` int32 — per-Q-block number of KV blocks to attend to (each value >= 0 for blk128, >= 1 for blk64). When provided, for each `(batch, head, m_block)` the first `q2k_block_nums[batch, head, m_block]` entries in the `[... , max_topk]` q2k storage are valid. Odd values are handled internally by padding to even with a phantom block (fully masked, zero contribution)
- `allow_empty_block_nums`: When True (default), `q2k_block_nums` may contain 0 (empty tiles produce O=0, LSE=-inf). When False, all values must be >= 1, enabling compile-time elimination of empty-tile branches for better performance (~2-3%)

For dense (full) attention, construct `q2k_block_index = [0,1,...,N-1]` for all Q blocks, `max_topk = N`, and `block_sizes = [tile_n]*N` (with last block adjusted for seqlen remainder). See `make_dense_block_sparse_args()` in `test_bsa.py`.

### blk128 — CuTe DSL Forward Kernel (`csrc/fwd/sm100_blk128/`)
- `flash_fwd_sm100.py` — `FlashAttentionForwardSm100`: Blackwell forward, qstage=1 only
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

### blk64 — CuTe DSL Backward Kernel (`csrc/bwd/sm100_blk64/`)
- `flash_bwd_sm100.py` — `BlockSparseAttnBackward`: Blackwell backward, `sparse_block_size=64`, MHA + bf16 + `head_dim=128` only. Exposes a single CSR scheduled entry consuming total-packed `k2q_q_indices` plus schedule metadata `[B, H, work, 4]` with fields `(kv_block, q_indices_start, q_count, q_group)`. All Q/K/V/O/dO/dQ/dK/dV tensors are passed in **(batch, num_heads, seqlen, head_dim)** layout (`bsa_attn_bwd` takes BHSD directly — callers holding the forward's BSHD layout must `.transpose(1, 2)` themselves).
- Kernel pipeline: `sum_OdO`, scheduled `bwd`, and `convert`. The main `bwd` loop runs qrange-split work items generated by `csrc/common/build_k2q_csr`, using total-packed CSR storage to avoid dense k2q metadata.
- `sm_100a` target only. MMA tilers are hardcoded at `(*, 64, 128)`, i.e. head_dim must be 128

### blk64 — C++ AOT Forward Kernel (`csrc/fwd/sm100_blk64/`)
- `flash_fwd_kernel_sm100.h` — `FusedAttnKernel`: kernel entry point, templated on `HasVarBlockNums` and `HasBlockSizes`
- `mainloop_fwd_sm100.hpp` — `CollectiveMainloopFwd`: load/mma/softmax mainloop (kRows=64, kDualCols=256, kSparseBlockSize=64, kSparseBlocksPerKV=4)
- `flash_fwd_launch_template.h` — Templated launch function definition
- `flash_fwd_launch_template.cu` — BOOL_SWITCH dispatch + Python binding
- `instantiations/` — Separate TUs per (HasVarBlockNums, HasBlockSizes) variant for register allocation isolation
- `epilogue_fwd_sm100.hpp` — Output epilogue
- `softmax.h` — Softmax implementation
- `pipeline.hpp` — Pipeline management
- `tile_scheduler.hpp` — Tile scheduler
- `bindings.cpp` — PyTorch C++ extension bindings (`bsa_fwd_blk64_ext`), returns `[out, lse]`
- `setup.py` — Build via `torch.utils.cpp_extension.CUDAExtension` + `bdist_wheel` (SM100a target)

### Utils (`utils/`)
- `cache_utils.py` — JIT compilation cache management
- `fa_logging.py` — `FA_LOG_LEVEL` debug logging
- `testing.py` — `attention_ref`, tolerance helpers
- `bench_utils.py` — flops computation

## Key Patterns

- **blk128**: Compile-time constants use `cutlass.Constexpr[type]` for kernel specialization. `max_topk` is a runtime `Int32` parameter (not compile-time); different values do not require recompilation
- **blk64**: Compile-time template parameter `HasVarBlockNums` selects kernel variant. Phantom block padding rounds `max_topk` up to multiples of 8 (`kSparseBlocksPerKV * 2`) for even kv_iters
- `q2k_block_nums` enables per-Q-block variable KV block counts; uses a separate compile path (`has_variable_block_nums` in compile key for blk128, `HasVarBlockNums` template for blk64). Pipeline phases (`phase_s0`/`phase_s1`) persist across tiles to handle variable `block_iter_count` in persistent scheduling. Odd values are rounded up to even internally; the phantom block uses a clamped index (`max_i` in `get_n_block_idx`) and `block_size=0` mask for zero contribution
- Forward execution: load Q tile -> loop over K/V blocks selected by `q2k_block_index` (pipelined) -> online softmax with per-block `block_sizes` masking -> store O and LSE. Both backends output LSE `(batch, num_heads, seqlen_q)` float32; blk128 writes LSE in the correction warp of `flash_fwd_sm100.py`, blk64 writes LSE in the correction warp of `mainloop_fwd_sm100.hpp` after warp-pair stats exchange
- Load order for N KV blocks (blk128): K[idx(N-1)], Q, K[idx(N-2)], {V[idx(N-1-i)], K[idx(N-3-i)]}x(N-2), V[idx(1)], V[idx(0)] -- where `idx(i) = q2k_block_index[batch, head, m_block, i]`
- 2CTA instructions (hdim=128): WIP, not yet functional for block-sparse
- CLC persistent scheduling (blk128 only): enabled via `BSA_CLC=1` env var (default on). Uses NVIDIA's Coordinated Launch Cluster (CLC) for dynamic work-stealing across CTAs. Requires `use_tma_KV=True`, cluster N=1, cluster M in {1,2}. A dedicated scheduler warp (warp 15 on leader CTA) issues async CLC queries; all warps synchronize via `PipelineClcFetchAsync` mbarriers. Compute warps use `tile_scheduler.consumer_advance()` which encapsulates CLC pipeline wait/release
- Test backend selection: `BSA_BLK` env var (or `BLK` make variable) controls which backends to test — "64", "128", or "64,128"

## Debugging Kernel Hangs

**TL;DR:** Set timeout to **30 seconds**. If the kernel produces no output within 30s, it is hung. Do NOT set timeout above 120s — anything beyond that is wasting your time.

See `dev_docs/deadlock.md` for the full debugging guide.
