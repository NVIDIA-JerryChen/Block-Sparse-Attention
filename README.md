# BSA — Block Sparse Attention

## Features

**Forward backends:**

| | SM100 blk128 (CuTe DSL / JIT) | SM90 blk64 (CuTe DSL / JIT) | SM100 blk64 (C++ AOT / CUTLASS) |
|---|---|---|---|
| Dtype | bf16, fp16 | bf16, fp16 | bf16 only |
| Head dim | 64, 96, 128, (192, 128) | 64, 96, 128 | 128 only |
| Attention | MHA, GQA, MQA | MHA, GQA, MQA | MHA only |
| pack_gqa | Yes | No | No |
| Persistent scheduling | Static + CLC dynamic | Static | CLC dynamic (built-in) |
| Variable block counts (`q2k_block_nums`) | Yes (>= 0) | Yes (>= 1) | Yes (>= 1) |
| KV split (`kv_splits`) | No | Explicit / auto | Explicit / auto |
| LSE output | Yes | Yes | Yes |

**Backward backends:**

| | SM90 blk64 (CuTe DSL / JIT) | SM100 blk64 (CuTe DSL / JIT) |
|---|---|---|
| Dtype | bf16 | bf16 |
| Head dim | 128 | 128 |
| Attention | MHA only | MHA only |
| Sparse task layout | bucketed k2q CSR | bucketed k2q CSR |

**Not supported (current sparse kernels):** causal, local, mask_mod, score_mod, paged_kv, softcap, varlen

Split-KV is supported by the SM90/SM100 blk64 forward paths. The blk128 and
backward paths do not support it.

## Directory Structure

```
BSA/
├── bsa_attn_interface.py         # Public API (fwd, bwd)
├── tests/
│   ├── test_flash_fwd.py         # Forward tests & benchmarks (blk64 + blk128)
│   └── test_flash_bwd.py         # Backward tests & benchmarks (blk64)
├── requirements.txt              # Python dependencies
├── Makefile                      # Build & test automation
│
├── csrc/fwd/
│   ├── sm100_blk128/                 # blk128 — CuTe DSL / JIT compiled
│   │   └── flash_fwd_sm100.py        # Single-file Blackwell forward kernel
│   ├── sm90_blk64/                   # blk64 — SM90 CuTe DSL / JIT compiled
│   │   └── flash_fwd_sm90.py         # Single-file Hopper forward kernel
│   │
│   └── sm100_blk64/                  # blk64 — C++ AOT / CUTLASS compiled
│       ├── bsa_api.cpp                   # PyTorch C++ bindings (returns [out, lse])
│       ├── bsa_fwd_kernel_sm100.h        # Kernel definition
│       ├── bsa_fwd_launch_template.h     # Host launch wrapper
│       ├── mainloop_fwd_sm100.hpp        # Mainloop (load, mma, softmax)
│       ├── epilogue_fwd_sm100.hpp        # Output epilogue
│       ├── softmax.h                     # Softmax
│       ├── pipeline.hpp                  # Pipeline management
│       ├── tile_scheduler.hpp            # Tile scheduler
│       ├── instantiations/               # AOT template instantiations
│       └── setup.py                      # Build script (bdist_wheel + CUDAExtension)
│
├── csrc/bwd/
│   ├── sm90_blk64/                       # blk64 backward — SM90 CuTe DSL / JIT compiled
│   │   └── flash_bwd_sm90.py             # Localized Hopper backward kernel
│   └── sm100_blk64/                      # blk64 backward — SM100 CuTe DSL / JIT compiled
│       └── flash_bwd_sm100.py            # Bucketed k2q CSR backward kernel
│
├── csrc/utils/                            # Shared CuTe DSL device/kernel helpers
│   ├── kernel_utils.py                    # Math, layout, and tensor utilities
│   ├── pipeline.py                        # TMA/UMMA pipeline helpers
│   ├── tile_scheduler.py                  # Shared tile schedulers
│   ├── block_sparse_tile_scheduler.py     # blk64 CLC persistent scheduler
│   ├── softmax.py / pack_gqa.py           # Forward attention helpers
│   └── ...
│
├── utils/
│   ├── cache_utils.py            # JIT compilation cache
│   ├── testing.py                # Reference attention, tolerance helpers
│   ├── benchmark.py              # benchmark_forward
│   ├── bench_utils.py            # FLOPS computation
│   └── fa_logging.py             # Debug logging
│
└── third_party/
    └── cutlass/                  # CUTLASS headers (git submodule)
```

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.5+ with CUDA support
- CUDA 13.0+
- CuTe DSL (`nvidia-cutlass-dsl>=4.4.1`)
- CUTLASS headers (for blk64 C++ AOT build)

### Setup

```bash
# Clone with submodules (CUTLASS headers required for blk64)
git clone --recurse-submodules <repo-url>
# Or if already cloned:
git submodule update --init --recursive

pip install -r requirements.txt

# Build blk64 C++ extension (builds wheel + pip installs it)
make setup
```

### Usage

```python
import torch
from bsa_attn_interface import bsa_attn_fwd, bsa_attn_fwd_blk64, bsa_attn_bwd

q = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)

out, lse = bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes)

# Split long KV lists across partial forward kernels and combine their results.
out, lse = bsa_attn_fwd_blk64(
    q, k, v, q2k_block_index, block_sizes, q2k_block_nums,
    kv_splits="auto", block_sparse_num=block_sparse_num,
)

# BSHD compatibility path. The wrapper canonicalizes to BHSD with view-only transposes.
q_bshd = q.transpose(1, 2)
k_bshd = k.transpose(1, 2)
v_bshd = v.transpose(1, 2)
out_bshd, lse = bsa_attn_fwd(
    q_bshd, k_bshd, v_bshd,
    q2k_block_index, block_sparse_num, block_sizes,
    layout="bshd",
)

# Variable per-Q-block KV block counts
out, lse = bsa_attn_fwd(q, k, v, q2k_block_index, 0, block_sizes,
                         q2k_block_nums=q2k_block_nums)

dout = torch.randn_like(out)

dq, dk, dv = bsa_attn_bwd(
    dout, q, k, v, out, lse,
    q2k_block_index, block_sparse_num, block_sizes,
)

# BSHD compatibility path for backward as well.
dout_bshd = dout.transpose(1, 2)
out_bshd = out.transpose(1, 2)
dq_bshd, dk_bshd, dv_bshd = bsa_attn_bwd(
    dout_bshd, q_bshd, k_bshd, v_bshd, out_bshd, lse,
    q2k_block_index, block_sparse_num, block_sizes,
    layout="bshd",
)

# Optional tuning override for the bucketed k2q CSR backward task layout
dq, dk, dv = bsa_attn_bwd(
    dout, q, k, v, out, lse,
    q2k_block_index, block_sparse_num, block_sizes,
    bucket_size_blocks=512,
)
```

## API Reference

### `bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes, ...)`

Block-sparse forward attention. The repo canonical layout is `BHSD`; the
public API also supports explicit `layout="bshd"` compatibility at the wrapper
boundary.

**Default tensor layout:** `(batch, num_heads, seqlen, head_dim)` (`BHSD`), last dim contiguous, 16-byte aligned.
When `layout="bshd"`, inputs and outputs use `(batch, seqlen, num_heads, head_dim)`; the wrapper uses view-only transposes and does not materialize layout copies.
The last dimension must be `head_dim` with stride 1. Physical `BHDS` / non-contiguous-head-dim layouts are not supported.
`lse` is always `(batch, num_heads, seqlen_q)`.

#### Input Tensors

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| `q` | (batch, num_heads, seqlen_q, head_dim) | bf16/fp16 | Query |
| `k` | (batch, num_heads_kv, seqlen_k, head_dim) | bf16/fp16 | Key |
| `v` | (batch, num_heads_kv, seqlen_k, head_dim_v) | bf16/fp16 | Value |

#### Block-Sparse Parameters (mandatory)

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `q2k_block_index` | (batch, num_heads, num_q_blocks, max_kv_blocks) | int32 | Per Q-block list of KV block indices to attend to |
| `block_sparse_num` | scalar | int | Number of KV blocks per Q block (even, >= 2 for blk128; >= 1 for blk64). Ignored when `q2k_block_nums` is provided |
| `block_sizes` | (num_kv_blocks,) | int32 | Actual token count per KV block (for masking padding positions) |

#### Variable Block-Sparse Parameters (optional)

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `q2k_block_nums` | (batch, num_heads, num_q_blocks) | int32 | Per-Q-block KV block count (>= 0 for blk128, >= 1 for blk64). When provided, `block_sparse_num` is ignored. Odd values handled internally via phantom block padding |
| `allow_empty_block_nums` | scalar | bool | Default True. When False, all `q2k_block_nums` values must be >= 1, enabling compile-time elimination of empty-tile branches (~2-3% faster) |

#### Other Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `softmax_scale` | `1/sqrt(head_dim)` | Softmax scale factor |
| `pack_gqa` | `None` (auto) | Whether to pack GQA heads. Auto-enabled when `qhead_per_kvhead > 1` and `tile_m % qhead_per_kvhead == 0` |
| `return_lse` | `False` | Whether to return log-sum-exp |
| `out` | `None` | Pre-allocated output tensor |
| `lse` | `None` | Pre-allocated LSE tensor |
| `layout` | `"bhsd"` | Input/output layout: `"bhsd"` or `"bshd"` |

#### Dense Attention

For dense (full) attention, construct block-sparse args that cover all KV blocks:

```python
# See make_dense_block_sparse_args() in tests/test_flash_fwd.py
q2k_block_index = [0, 1, ..., N-1]  # for all Q blocks
block_sparse_num = N                  # must be even, >= 2
block_sizes = [tile_n] * N            # last block adjusted for seqlen remainder
```

### `bsa_attn_fwd_blk64(..., kv_splits=1, ...)`

The SM90 and SM100 blk64 forward paths can split each Q block's active KV list:

- `kv_splits=1` uses the legacy single forward kernel without partial-output
  workspace or a combine kernel.
- `kv_splits=2..256` produces FP32 O/LSE partials for each split and combines
  them into the requested output dtype. Partial workspace grows linearly with
  `kv_splits`.
- `kv_splits="auto"` selects 1, 2, 4, or 8 splits at KV block-count thresholds
  256, 450, and 900. On SM90, the 256--449 range stays unsplit for a single Q
  head. With fixed block counts, the policy uses `block_sparse_num`; otherwise
  it uses `q2k_block_index.shape[-1]`. Auto lowers the split count if its
  estimated workspace does not fit; explicit split counts report an error
  instead.

SM90 split-KV supports the same MHA/GQA/MQA and QK/V dimensions (64, 96, or
128) as its single-kernel path. SM100 blk64 retains its existing shape
constraints, and its split path does not use the CLC scheduler; auto split
selection disables CLC for that path.

### `bsa_attn_bwd(dout, q, k, v, out, lse, q2k_block_index, block_sparse_num, block_sizes, ...)`

SM90/SM100 block-sparse backward attention (blk64 backend).

This recomputes the attention probabilities from `q/k/v`, `out`, and `lse`, then returns gradients `(dq, dk, dv)`.

**Default tensor layout:** `(batch, num_heads, seqlen, head_dim)` (`BHSD`), matching the default forward layout. Last dim must be contiguous.
When `layout="bshd"`, `dout/q/k/v/out` and `dq/dk/dv` use `(batch, seqlen, num_heads, head_dim)`; the wrapper uses view-only transposes and does not materialize layout copies.
The last dimension must be `head_dim` with stride 1. Physical `BHDS` / non-contiguous-head-dim layouts are not supported.
`lse` is always `(batch, num_heads, seqlen_q)`.

#### Backward Inputs

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| `dout` | (batch, num_heads, seqlen_q, head_dim) | bf16 | Upstream gradient |
| `q` | (batch, num_heads, seqlen_q, head_dim) | bf16 | Query |
| `k` | (batch, num_heads, seqlen_k, head_dim) | bf16 | Key |
| `v` | (batch, num_heads, seqlen_k, head_dim) | bf16 | Value |
| `out` | (batch, num_heads, seqlen_q, head_dim) | bf16 | Forward output |
| `lse` | (batch, num_heads, seqlen_q) | fp32 | Forward log-sum-exp |

The block-sparse arguments have the same meaning as forward:

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `q2k_block_index` | (batch, num_heads, num_q_blocks, max_kv_blocks) | int32 | Per Q-block list of KV block indices |
| `block_sparse_num` | scalar | int | Fixed KV block count per Q block. Ignored when `q2k_block_nums` is provided |
| `block_sizes` | (num_kv_blocks,) or (batch, num_kv_blocks) | int32 | Actual token count per KV block |
| `q2k_block_nums` | (batch, num_heads, num_q_blocks) | int32 | Optional variable KV block count per Q block |
| `layout` | `"bhsd"` | Input/output layout: `"bhsd"` or `"bshd"` |

#### Backward Outputs

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| `dq` | same as `q` | bf16 | Gradient w.r.t. Q |
| `dk` | same as `k` | bf16 | Gradient w.r.t. K |
| `dv` | same as `v` | bf16 | Gradient w.r.t. V |

#### Backward Limitations

Backward currently supports:

```text
backend: blk64 CuTe DSL
dtype: bf16
head_dim: 128
attention: MHA only, num_heads == num_heads_kv
block size: 64
architecture: SM90/SM100/SM110
```

### `bsa_attn_bwd(..., bucket_size_blocks=None)`

Bucketed k2q CSR backward path for long-sequence sparse attention.

The wrapper builds a GPU-side bucketed k2q CSR task layout from `q2k_block_index` on every call. This is the only blk64 backward implementation kept in the tree for SM90/SM100/SM110.

The main backward kernel runs one task per `(q_group, kv_block)` and uses fp32 workspace accumulation for `dQ/dK/dV` before the final conversion/writeback.

Default bucket sizing is backend-owned: SM90 uses the SM90 blk64 backward default, while SM100/SM110 use the SM100 blk64 backward defaults. Pass `bucket_size_blocks` explicitly to override it for experiments.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bucket_size_blocks` | backend default | Number of Q blocks per bucket. Larger values reduce task count; smaller values improve `dQ_acc` locality |

## Tests & Benchmarks

```bash
make setup                      # Build blk64 C++ extension
make tt                         # Quick correctness test (default: blk128)
make tt BLK=64                  # Quick test blk64 only
make tt BLK=64,128              # Quick test both backends
make vt                         # Full pytest suite
make vt BLK=64                  # Full pytest blk64 only
make bb                         # Performance benchmark
make profile                    # Single fwd run for ncu profiling
make bm                         # ncu full profile (NVTX-filtered)
make bm-cli                     # ncu register/spill/smem analysis
make compare                    # Compare BSA vs FA4 performance
make clean                      # Clear compile caches + blk64 build
make help                       # Show all targets

python test_flash_bwd.py                    # Backward quick correctness tests
python test_flash_bwd.py benchmark          # Backward benchmark
```
