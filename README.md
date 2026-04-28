# BSA — Block Sparse Attention

## Features

**Two kernel backends:**

| | blk128 (CuTe DSL / JIT) | blk64 (C++ AOT / CUTLASS) |
|---|---|---|
| Dtype | bf16, fp16 | bf16 only |
| Head dim | 64, 96, 128, (192, 128) | 128 only |
| Attention | MHA, GQA, MQA | MHA only |
| pack_gqa | Yes | No |
| Persistent scheduling | Static + CLC dynamic | CLC dynamic (built-in) |
| Variable block counts (`q2k_block_nums`) | Yes (>= 0) | Yes (>= 1) |
| LSE output | Yes | Yes |

**Not supported (both backends):** causal, local, mask_mod, score_mod, split-kv, paged_kv, softcap, varlen

## Directory Structure

```
BSA/
├── bsa_attn_interface.py         # Public API (fwd, bwd, qbucket bwd)
├── test_flash_fwd.py             # Forward tests & benchmarks (blk64 + blk128)
├── test_flash_bwd.py             # Backward tests & benchmarks (blk64)
├── requirements.txt              # Python dependencies
├── Makefile                      # Build & test automation
│
├── csrc/fwd/
│   ├── sm100_blk128/                 # blk128 — CuTe DSL / JIT compiled
│   │   ├── flash_fwd_sm100.py        # Main forward kernel
│   │   ├── blackwell_helpers.py      # SM100 UMMA-based GEMM (2CTA WIP)
│   │   ├── softmax.py                # Online softmax
│   │   ├── mask.py                   # Seqlen & block-size masking
│   │   ├── block_info.py             # Tile dims, block-sparse index lookup
│   │   ├── tile_scheduler.py         # Static persistent & CLC scheduling
│   │   ├── pipeline.py               # Circular buffer management
│   │   ├── pack_gqa.py               # GQA head packing
│   │   └── ...                       # utils, named_barrier, mma descriptors
│   │
│   └── sm100_blk64/                  # blk64 — C++ AOT / CUTLASS compiled
│       ├── flash_fwd_kernel_sm100.h      # Kernel definition
│       ├── mainloop_fwd_sm100.hpp        # Mainloop (load, mma, softmax)
│       ├── flash_fwd_launch_template.cu  # Host launch wrapper
│       ├── epilogue_fwd_sm100.hpp        # Output epilogue
│       ├── softmax.h                     # Softmax
│       ├── pipeline.hpp                  # Pipeline management
│       ├── tile_scheduler.hpp            # Tile scheduler
│       ├── bindings.cpp                  # PyTorch C++ bindings (returns [out, lse])
│       └── setup.py                      # Build script (bdist_wheel + CUDAExtension)
│
├── csrc/bwd/
│   └── sm100_blk64/                      # blk64 backward — CuTe DSL / JIT compiled
│       ├── flash_bwd_sm100.py            # KV-major backward kernel
│       └── flash_bwd_sm100_qbucket.py    # Q-range bucketed backward kernel
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
from bsa_attn_interface import bsa_attn_fwd, bsa_attn_bwd, bsa_attn_bwd_qbucket

q = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)

out, lse = bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes)

# Variable per-Q-block KV block counts
out, lse = bsa_attn_fwd(q, k, v, q2k_block_index, 0, block_sizes,
                         q2k_block_nums=q2k_block_nums)

# Backward uses BHSD layout: (batch, heads, seqlen, dim)
q_bhsd = q.transpose(1, 2).contiguous()
k_bhsd = k.transpose(1, 2).contiguous()
v_bhsd = v.transpose(1, 2).contiguous()
out_bhsd = out.transpose(1, 2).contiguous()
dout_bhsd = torch.randn_like(out_bhsd)

dq, dk, dv = bsa_attn_bwd(
    dout_bhsd, q_bhsd, k_bhsd, v_bhsd, out_bhsd, lse,
    q2k_block_index, block_sparse_num, block_sizes,
)

# Long-sequence experimental path: Q-range bucketed backward
dq, dk, dv = bsa_attn_bwd_qbucket(
    dout_bhsd, q_bhsd, k_bhsd, v_bhsd, out_bhsd, lse,
    q2k_block_index, block_sparse_num, block_sizes,
    q_bucket_size_blocks=512,
)
```

## API Reference

### `bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes, ...)`

SM100 block-sparse forward attention (blk128 backend).

**Tensor layout:** `(batch, seqlen, num_heads, head_dim)`, last dim contiguous, 16-byte aligned.

#### Input Tensors

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| `q` | (batch, seqlen_q, num_heads, head_dim) | bf16/fp16 | Query |
| `k` | (batch, seqlen_k, num_heads_kv, head_dim) | bf16/fp16 | Key |
| `v` | (batch, seqlen_k, num_heads_kv, head_dim_v) | bf16/fp16 | Value |

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

#### Dense Attention

For dense (full) attention, construct block-sparse args that cover all KV blocks:

```python
# See make_dense_block_sparse_args() in test_flash_fwd.py
q2k_block_index = [0, 1, ..., N-1]  # for all Q blocks
block_sparse_num = N                  # must be even, >= 2
block_sizes = [tile_n] * N            # last block adjusted for seqlen remainder
```

### `bsa_attn_bwd(dout, q, k, v, out, lse, q2k_block_index, block_sparse_num, block_sizes, ...)`

SM100 block-sparse backward attention (blk64 backend only).

This recomputes the attention probabilities from `q/k/v`, `out`, and `lse`, then returns gradients `(dq, dk, dv)`.

**Tensor layout:** `(batch, num_heads, seqlen, head_dim)` (`BHSD`), last dim contiguous.

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
architecture: SM100/SM110
```

### `bsa_attn_bwd_qbucket(..., q_bucket_size_blocks=512)`

Q-range bucketed backward path for long-sequence sparse attention.

It has the same tensor contract as `bsa_attn_bwd`, but builds a GPU-side Q-range task layout from `q2k_block_index` on every call. The main backward kernel then runs one task per `(q_group, kv_block)` instead of one task per `kv_block`.

This path is intended for long sequences with random-ish topK sparsity where the baseline KV-major backward suffers from poor `dQ_acc` locality. The tradeoff is extra task construction, K/V reloads, and fp32 partial `dK/dV` accumulation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `q_bucket_size_blocks` | `512` | Number of Q blocks per bucket. Larger values reduce task count; smaller values improve `dQ_acc` locality |

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
BSA_BWD_BENCH_IMPL=baseline python test_flash_bwd.py benchmark
BSA_BWD_BENCH_IMPL=qbuck BSA_Q_BUCKET_BLOCKS=512 python test_flash_bwd.py benchmark
```
