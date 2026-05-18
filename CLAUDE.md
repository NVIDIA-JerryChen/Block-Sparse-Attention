# CLAUDE.md

## Project Overview

**Supported**: block-sparse attention, forward on SM90/SM100, backward on SM90/SM100 blk64. Forward supports bf16/fp16, MHA/GQA/MQA, hdim 64/96/128/(192,128) depending on backend. Backward supports bf16, MHA, hdim=128.
**Not supported**: causal, local, mask_mod, score_mod, split-kv, paged_kv, softcap, varlen

Forward kernel backends:
- **SM100 blk128** (`csrc/fwd/sm100_blk128/`): CuTe DSL / JIT compiled, tile_m=128, tile_n=128
- **SM90 blk64** (`csrc/fwd/sm90_blk64/`): CuTe DSL / JIT compiled, tile_m=64, tile_n=64
- **SM100 blk64** (`csrc/fwd/sm100_blk64/`): C++ AOT / CUTLASS compiled, tile_m=64, tile_n=64 (kDualCols=256, kSparseBlocksPerKV=4), bf16 only

Backward kernel backends:
- **SM90 blk64** (`csrc/bwd/sm90_blk64/`): CuTe DSL / JIT compiled, sparse_block_size=64, head_dim=128, MHA only, bf16 only
- **SM100 blk64** (`csrc/bwd/sm100_blk64/`): CuTe DSL / JIT compiled, sparse_block_size=64, head_dim=128, MHA only, bf16 only

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
make bbb                # dense-bwd benchmark

# ncu full / register-spill-smem analysis
make bm-cli
```

## Code Architecture

### Public API (`bsa_attn_interface.py`)
- `bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes, ...)` — SM90/SM100 block-sparse forward attention. Repo canonical tensor layout is **BHSD** (`(batch, num_heads, seqlen, head_dim)`). The public wrapper also accepts `layout="bshd"` for compatibility and converts at the wrapper boundary with view-only transposes.
- `bsa_attn_bwd(dout, q, k, v, out, lse, q2k_block_index, block_sparse_num, block_sizes=None, q2k_block_nums=None, softmax_scale=None, dq=None, dk=None, dv=None, layout="bhsd")` — SM90/SM100 block-sparse backward attention (blk64 backend, bf16/MHA/hdim=128). Repo canonical tensor layout is **BHSD**; the public wrapper also accepts `layout="bshd"` for compatibility and converts at the wrapper boundary with view-only transposes. Returns `(dq, dk, dv)` in the same layout as the inputs. Reuses the `q2k_block_index` / `block_sparse_num` / `block_sizes` / `q2k_block_nums` from the forward call; the wrapper builds bucketed k2q CSR internally and allocates the fp32 workspace needed by the kernel
- `convert_q2k_to_k2q(q2k_block_index, block_sparse_num, num_kv_blocks, q2k_block_nums=None)` — legacy helper that inverts the forward's per-Q-block KV-attendee list into a dense per-KV-block Q-attendee list + count tensors. The default backward path builds bucketed k2q CSR instead.

Tensor layout: `(batch, num_heads, seqlen, head_dim)`, last dim contiguous, 16-byte aligned. Layouts with non-contiguous `head_dim`, including physical `BHDS`, are not supported.

#### Block-Sparse Parameters (mandatory)
- `q2k_block_index`: `(batch, num_heads, num_q_blocks, max_kv_blocks)` int32 — per Q-block list of KV block indices to attend to
- `block_sparse_num`: int (runtime, even, >= 2 for blk128; >= 1 for blk64) — number of KV blocks each Q block attends to. Ignored when `q2k_block_nums` is provided
- `block_sizes`: `(num_kv_blocks,)` int32 — actual token count per KV block (for masking padding positions within a tile)

#### Variable Block-Sparse Parameters (optional)
- `q2k_block_nums`: `(batch, num_heads, num_q_blocks)` int32 — per-Q-block number of KV blocks to attend to (each value >= 0 for blk128, >= 1 for blk64). When provided, `block_sparse_num` is ignored, and for each `(batch, head, m_block)` the first `q2k_block_nums[batch, head, m_block]` entries in `q2k_block_index` are valid. Odd values are handled internally by padding to even with a phantom block (fully masked, zero contribution)
- `allow_empty_block_nums`: When True (default), `q2k_block_nums` may contain 0 (empty tiles produce O=0, LSE=-inf). When False, all values must be >= 1, enabling compile-time elimination of empty-tile branches for better performance (~2-3%)

For dense (full) attention, construct `q2k_block_index = [0,1,...,N-1]` for all Q blocks, `block_sparse_num = N`, and `block_sizes = [tile_n]*N` (with last block adjusted for seqlen remainder). See `make_dense_block_sparse_args()` in `tests/test_flash_fwd.py`.

### blk128 — CuTe DSL Forward Kernel (`csrc/fwd/sm100_blk128/`)
- `flash_fwd_sm100.py` — `BlockSparseAttnForwardSm100Blk128`: single-file Blackwell forward, qstage=1 only. The file owns the SM100 forward helpers that were previously split across sibling modules: online softmax, block-size masking, `BlockInfo`, `SeqlenInfoQK`, static/CLC schedulers, pipeline wrappers, named barriers, GQA packing, UMMA descriptor helpers, PTX GEMM helpers, and tensor alignment conversion.

### blk64 — CuTe DSL Forward Kernel (`csrc/fwd/sm90_blk64/`)
- `flash_fwd_sm90.py` — `BlockSparseAttnForwardSm90Blk64`: single-file Hopper forward kernel. The interface owns torch tensor normalization, block-size expansion, compile/cache, and launch; the kernel file owns the CuTe mainloop plus local online softmax and accumulator-layout helpers.

### blk64 — CuTe DSL Backward Kernels (`csrc/bwd/sm90_blk64/`, `csrc/bwd/sm100_blk64/`)
- `csrc/bwd/sm90_blk64/flash_bwd_sm90.py` — `BlockSparseAttnBackwardSm90Blk64`: single-file Hopper bucketed k2q CSR backward, `sparse_block_size=64`, MHA + bf16 + `head_dim=128` only.
- `csrc/bwd/sm100_blk64/flash_bwd_sm100.py` — `BlockSparseAttnBackwardSm100Blk64`: single-file Blackwell bucketed k2q CSR backward, `sparse_block_size=64`, MHA + bf16 + `head_dim=128` only.
- Both expose a single `__call__(problem_shape, dO, O, Q, K, V, LSE, dQ, dK, dV, bucketed_k2q_offsets, bucketed_k2q_indices, variable_block_sizes, workspace, scale_softmax, stream)` entry. `problem_shape` is `(seqlen_q, seqlen_k, head_dim, (num_heads, batch))`. All Q/K/V/O/dO/dQ/dK/dV tensors are passed in **(batch, num_heads, seqlen, head_dim)** layout. The public wrapper builds bucketed k2q CSR from `q2k_block_index` on GPU:
  - `bucketed_k2q_offsets`: `(batch, num_heads, num_q_groups, num_kv_blocks + 1)` int32 — per `(B,H,q_group)` CSR offsets into `bucketed_k2q_indices`
  - `bucketed_k2q_indices`: `(batch, num_heads, edge_count)` int32 — compact attending Q-block list for each bucketed k2q CSR row
  - `variable_block_sizes`: `(batch, num_kv_blocks)` int32 — per-batch, per-KV-block valid token count (the wrapper expands a shared `(num_kv_blocks,)` `block_sizes` to this layout)
  - `workspace`: `(batch, num_heads, Q_pad * (D_pad + 2) * 4 + 2 * K_pad * D_pad * 4)` uint8 zero-initialized scratch space. It stores `sum_OdO`, `scaled_lse`, fp32 `dQ_acc`, fp32 `dK_acc`, and fp32 `dV_acc`
- Kernel pipeline: launches three sub-kernels back-to-back: `sum_OdO` (rowwise `sum(O * dO)` + `scaled_LSE = -log2(e)*LSE`), bucketed k2q CSR `bwd` (one CTA per `(q_group, kv_block, head, batch)` task, accumulating fp32 `dQ/dK/dV` workspace), and `convert` (casts workspace accumulators to bf16 outputs; `dK` is scaled by softmax_scale)
- SM90 targets `sm_90a`; SM100 targets `sm_100a`. MMA tilers are hardcoded at `(*, 64, 128)`, i.e. head_dim must be 128

### blk64 — C++ AOT Forward Kernel (`csrc/fwd/sm100_blk64/`)
- `bsa_fwd_kernel_sm100.h` — `FusedAttnKernel`: kernel entry point, templated on `HasVarBlockNums` and `HasBlockSizes`
- `mainloop_fwd_sm100.hpp` — `CollectiveMainloopFwd`: load/mma/softmax mainloop (kRows=64, kDualCols=256, kSparseBlockSize=64, kSparseBlocksPerKV=4)
- `bsa_fwd_launch_template.h` — Templated launch function definition + BOOL_SWITCH dispatch
- `instantiations/` — Separate TUs per (HasVarBlockNums, HasBlockSizes) variant for register allocation isolation
- `epilogue_fwd_sm100.hpp` — Output epilogue
- `softmax.h` — Softmax implementation
- `pipeline.hpp` — Pipeline management
- `tile_scheduler.hpp` — Tile scheduler
- `bsa_api.cpp` — PyTorch C++ extension bindings (`bsa_fwd_blk64_ext`), returns `[out, lse]`
- `setup.py` — Build via `torch.utils.cpp_extension.CUDAExtension` + `bdist_wheel` (SM100a target)

### Utils (`utils/`)
- `cache_utils.py` — JIT compilation cache management
- `fa_logging.py` — `FA_LOG_LEVEL` debug logging
- `testing.py` — `attention_ref`, tolerance helpers
- `bench_utils.py` — flops computation
- `benchmark.py` — `benchmark_forward`

## Key Patterns

- **blk128**: Compile-time constants use `cutlass.Constexpr[type]` for kernel specialization. `block_sparse_num` is a runtime `Int32` parameter (not compile-time); different values do not require recompilation
- **SM100 blk64 AOT fwd**: Compile-time template parameter `HasVarBlockNums` selects kernel variant. Phantom block padding rounds `block_sparse_num` up to multiples of 8 (`kSparseBlocksPerKV * 2`) for even kv_iters
- `q2k_block_nums` enables per-Q-block variable KV block counts; uses a separate compile path (`has_variable_block_nums` in compile key for blk128, `HasVarBlockNums` template for blk64). Pipeline phases (`phase_s0`/`phase_s1`) persist across tiles to handle variable `block_iter_count` in persistent scheduling. Odd values are rounded up to even internally; the phantom block uses a clamped index (`max_i` in `get_n_block_idx`) and `block_size=0` mask for zero contribution
- Forward execution: load Q tile -> loop over K/V blocks selected by `q2k_block_index` (pipelined) -> online softmax with per-block `block_sizes` masking -> store O and LSE. All forward backends output LSE `(batch, num_heads, seqlen_q)` float32; SM100 blk128 writes LSE in the correction warp of `flash_fwd_sm100.py`, SM90 blk64 writes LSE in `flash_fwd_sm90.py`, and SM100 blk64 AOT writes LSE in `mainloop_fwd_sm100.hpp` after warp-pair stats exchange
- Load order for N KV blocks (blk128): K[idx(N-1)], Q, K[idx(N-2)], {V[idx(N-1-i)], K[idx(N-3-i)]}x(N-2), V[idx(1)], V[idx(0)] -- where `idx(i) = q2k_block_index[batch, head, m_block, i]`
- 2CTA instructions (hdim=128): WIP, not yet functional for block-sparse
- CLC persistent scheduling (blk128 only): controlled by the SM100 forward kernel default. Uses NVIDIA's Coordinated Launch Cluster (CLC) for dynamic work-stealing across CTAs. Requires `use_tma_KV=True`, cluster N=1, cluster M in {1,2}. A dedicated scheduler warp (warp 15 on leader CTA) issues async CLC queries; all warps synchronize via `PipelineClcFetchAsync` mbarriers. Compute warps use `tile_scheduler.consumer_advance()` which encapsulates CLC pipeline wait/release
- Test backend selection: `BSA_BLK` env var (or `BLK` make variable) controls which backends to test — "64", "128", or "64,128"

## Debugging Kernel Hangs

**TL;DR:** Set timeout to **30 seconds**. If the kernel produces no output within 30s, it is hung. Do NOT set timeout above 120s — anything beyond that is wasting your time.

See `dev_docs/deadlock.md` for the full debugging guide.
