# BSA — Block Sparse Attention

## Features

- **BF16 & FP16**
- **MHA, GQA, MQA** (with `pack_gqa`)
- **Head dimensions**: 64, 96, 128, (192, 128)
- **Persistent tile scheduling** (static & CLC dynamic)
- **Block-sparse attention**: per Q-block KV block selection via `q2k_block_index`

## Directory Structure

```
BSA/
├── bsa_attn_interface.py         # Public API (bsa_attn_fwd)
├── test_flash_fwd.py             # Tests & benchmarks
├── requirements.txt              # Python dependencies
├── Makefile                      # Build & test automation
│
├── csrc/                         # Kernel implementation
│   └── fwd/
│       └── sm100/                # SM100 (Blackwell) CuTe DSL forward
│           ├── flash_fwd_sm100.py    # Main forward kernel
│           ├── blackwell_helpers.py  # SM100 UMMA-based GEMM (2CTA WIP)
│           ├── softmax.py            # Online softmax
│           ├── mask.py               # Seqlen & block-size masking
│           ├── block_info.py         # Tile dims, block-sparse index lookup
│           ├── tile_scheduler.py     # Static persistent & CLC scheduling
│           ├── pipeline.py           # Circular buffer management
│           ├── pack_gqa.py           # GQA head packing
│           └── ...                   # utils, named_barrier, mma descriptors
│
└── utils/                        # Non-kernel utilities
    ├── cache_utils.py            # JIT compilation cache
    ├── testing.py                # Reference attention, tolerance helpers
    ├── benchmark.py              # benchmark_forward
    ├── bench_utils.py            # FLOPS computation
    └── fa_logging.py             # Debug logging
```

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.5+ with CUDA support
- CUDA 13.0+
- CuTe DSL (`nvidia-cutlass-dsl>=4.4.1`)

### Setup

```bash
pip install -r requirements.txt
```

### Usage

```python
import torch
from bsa_attn_interface import bsa_attn_fwd

q = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 1024, 8, 128, device="cuda", dtype=torch.bfloat16)

out, lse = bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes)

# Variable per-Q-block KV block counts
out, lse = bsa_attn_fwd(q, k, v, q2k_block_index, 0, block_sizes,
                         q2k_block_nums=q2k_block_nums)
```

## Data Layout

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| `q` | (batch, seqlen_q, num_heads, head_dim) | bf16/fp16 | Query |
| `k` | (batch, seqlen_k, num_heads_kv, head_dim) | bf16/fp16 | Key |
| `v` | (batch, seqlen_k, num_heads_kv, head_dim) | bf16/fp16 | Value |
| `q2k_block_index` | (batch, num_heads, num_q_blocks, max_kv_blocks) | int32 | Per Q-block KV block indices |
| `block_sparse_num` | scalar | int | Number of KV blocks per Q block (even, >= 2). Ignored when `q2k_block_nums` is provided |
| `block_sizes` | (num_kv_blocks,) | int32 | Actual token count per KV block |
| `q2k_block_nums` | (batch, num_heads, num_q_blocks) | int32 | Optional. Per-Q-block KV block count (each value >= 1, odd values supported). When provided, `block_sparse_num` is ignored |

## Tests

```bash
make tt             # Quick correctness test
make vt             # Full pytest suite
make bb             # Performance benchmark
make profile        # Single fwd run for ncu profiling
make compare        # Compare BSA vs FA4 (Jerry local q_stage=1) performance
make help           # Show all targets
```
