# BSA — Block Sparse Attention

## Features

**Forward backends:**

| | SM120 blk64 (CuTe DSL / AOT + JIT) | SM100 blk128 (CuTe DSL / JIT) | SM90 blk64 (CuTe DSL / AOT + JIT) | SM100/SM110 blk64 (CuTe DSL / JIT) |
|---|---|---|---|---|
| Dtype | bf16, fp16 | bf16, fp16 | bf16, fp16 | bf16 only |
| Head dim | 128 only | 64, 96, 128 | 64, 96, 128 | 128 only |
| Attention | MHA, GQA, MQA | MHA, GQA, MQA | MHA, GQA, MQA | MHA only |
| pack_gqa | No | Yes | No | No |
| Persistent scheduling | Static | Static + CLC dynamic | Static | Static + CLC dynamic |
| Variable block counts (`q2k_block_nums`) | Yes (>= 1) | Yes (>= 0) | Yes (>= 0) | Yes (>= 0) |
| KV split (`kv_splits`) | No | No | Explicit / auto | Explicit / auto |
| LSE output | Yes | Yes | Yes | Yes |

**Backward backends:**

| | SM90 blk64 (CuTe DSL / JIT) | SM100/SM110 blk64 (CuTe DSL / JIT) | SM100/SM110 blk128 (CuTe DSL / JIT) |
|---|---|---|---|
| Dtype | bf16 | bf16 | bf16 |
| Head dim | 128 | 128 | 64, 128 |
| Attention | MHA only | MHA only | MHA only |
| Sparse task layout | bucketed k2q CSR | bucketed k2q CSR | bucketed k2q CSR |

**Not supported (current sparse kernels):** causal, local, mask_mod, score_mod, paged_kv, softcap, varlen

Split-KV is supported by the SM90/SM100/SM110 blk64 forward paths. SM120 deliberately
keeps `kv_splits=1`; explicit split counts are rejected so no FP32 partial
workspace is allocated. The blk128 and backward paths do not support split-KV.

## Directory Structure

```
BSA/
├── block_sparse_attention/
│   └── __init__.py                # Public package facade for source checkouts
├── bsa_attn_interface.py          # Public API implementation (fwd, bwd)
├── setup.py                       # Maps root sources into the installed package
├── pyproject.toml                 # Unified wheel metadata
├── tests/
│   ├── test_flash_fwd.py          # Forward tests & benchmarks (blk64 + blk128)
│   ├── test_flash_bwd.py          # Backward tests & benchmarks (blk64 + blk128)
│   ├── test_sm90_aot.py           # SM90 producer/combine AOT validation
│   └── test_sm120_aot.py          # SM120 forward AOT validation
├── requirements.txt               # Python dependencies
├── Makefile                       # Build & test automation
│
├── csrc/fwd/
│   ├── sm100_blk128/                 # blk128 — CuTe DSL / JIT compiled
│   │   └── bsa_fwd_sm100.py          # Single-file Blackwell forward kernel
│   ├── sm90_blk64/                   # blk64 — SM90 CuTe DSL / AOT + JIT
│   │   ├── bsa_fwd_sm90.py           # Hopper forward producer
│   │   ├── aot_build.py              # Producer/combine offline builder
│   │   ├── aot_runtime.py            # Manifest validation and runtime loader
│   │   └── aot_utils.py              # Main/combine variant metadata
│   ├── sm120_blk64/                  # blk64 — SM120 CuTe DSL / AOT + JIT
│   │   ├── bsa_fwd_sm120.py          # SM120 forward kernel
│   │   ├── aot_build.py              # Offline native-ABI artifact builder
│   │   ├── aot_runtime.py            # Manifest validation and runtime loader
│   │   └── aot_utils.py              # Variant and artifact metadata
│   │
│   └── sm100_blk64/                  # blk64 — SM100/SM110 implementation
│       ├── cutedsl/                      # Packaged CuTe DSL / JIT backend
│       │   ├── bsa_fwd_sm100.py          # Forward kernel
│       │   ├── bsa_fwd_helpers.py        # SM100 device helpers
│       │   └── bsa_fwd_combine.py        # Split-KV combine kernel
│       └── cpp/                          # Legacy development sources; excluded from wheel
│
├── csrc/bwd/
│   ├── bsa_bwd_preprocess.py             # Shared backward preprocess kernel
│   ├── bsa_bwd_postprocess.py            # Shared backward postprocess kernel
│   ├── bsa_bwd_prepost.py                # Pre/post compile and launch helpers
│   ├── sm90_blk64/                       # blk64 backward — SM90 CuTe DSL / JIT compiled
│   │   └── bsa_bwd_sm90.py               # Localized Hopper backward kernel
│   ├── sm100_blk64/                      # blk64 backward — SM100 CuTe DSL / JIT compiled
│   │   └── bsa_bwd_sm100.py              # Bucketed k2q CSR backward kernel
│   └── sm100_blk128/                     # blk128 backward — SM100 CuTe DSL / JIT compiled
│       └── bsa_bwd_sm100.py              # Bucketed k2q CSR backward kernel
│
├── csrc/utils/                            # Shared CuTe DSL device/kernel helpers
│   ├── kernel_utils.py                    # Math, layout, and tensor utilities
│   ├── pipeline.py                        # TMA/UMMA pipeline helpers
│   ├── tile_scheduler.py                  # Shared tile schedulers
│   ├── block_sparse_tile_scheduler.py     # blk64 CLC persistent scheduler
│   ├── tcgen05_mma_helpers.py             # Shared SM100 MMA helpers
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
    └── cutlass/                  # Development-only submodule; excluded from wheel
```

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.5+ built for the installed CUDA runtime
- CUDA 13.0+
- CuTe DSL (`nvidia-cutlass-dsl>=4.5.2,<4.6`; Sage FP8 and the SM90/SM120
  dynamic AOT paths are validated with the 4.5.2 ABI)

### Setup

```bash
# Install directly from a source checkout.
pip install .

# Or build one reusable wheel and install it elsewhere.
make wheel
pip install dist/block_sparse_attention-*.whl

# Development shortcut when the runtime dependencies are already installed.
make setup
```

The unified wheel and source distribution expose `bsa_attn_fwd`,
`bsa_attn_bwd`, `bsa_fp8_blk64_fwd`, and `quantize_sage_bhsd` as public APIs.
They contain all supported forward and backward CuTe DSL kernels, the Sage FP8
quantization path, and shared Python utilities. They do not contain the SM100
blk64 C++ implementation, a prebuilt CUDA extension, or the
`third_party/cutlass` submodule. SM100/SM110 blk64 forward dispatches to the
packaged CuTe DSL implementation and JIT-compiles it on first use. `make setup`
uses `--no-deps` for fast development reinstalls; use `pip install .` or install
the wheel directly in a fresh environment to resolve runtime dependencies.
All installed modules live under the `block_sparse_attention` package; the
wheel does not install top-level `csrc`, `utils`, or `bsa_attn_interface`
modules. The source tree remains in its existing layout; the build configuration
maps those files to their package-qualified installation paths.

The wheel includes the SM90 and SM120 AOT builders and runtime loaders, but not
generated `.o`, `.h`, or `.so` artifacts. On those architectures, a compatible
artifact bundle is used when present; otherwise dispatch falls back to JIT unless
the corresponding AOT-only mode is enabled.

API migration note: the former `bsa_attn_fwd_blk64` and
`bsa_attn_fwd_blk64_cutedsl` names are removed. Both unified entry points
default to `sparse_block_size=64`; existing blk128 callers must pass
`sparse_block_size=128` explicitly so their metadata is not reinterpreted.

For an editable development install with tests, use `pip install -e '.[test]'`.

### SM90 CuTe DSL AOT

SM90 blk64 forward can be compiled offline for `sm_90a` and loaded through the
CuTe native ABI. Split-KV bundles contain both the forward producer and the
deduplicated combine kernel, so AOT-only execution does not invoke
`cute.compile()` at either stage. The public attention API is unchanged.

#### Supported configurations

- Target: `sm_90a`
- Dtype: BF16 and FP16
- QK and value head dimensions: 64, 96, or 128 independently
- Attention: MHA, GQA, and MQA
- Block counts: fixed `block_sparse_num` or runtime `q2k_block_nums`
- `block_sizes`: absent or present; input ranks 1/2/3 share one normalized ABI
- Split-KV: exact `kv_splits` values from 1 through 256; the default bundle
  contains 1, 2, 4, and 8
- Minimum DSL version: `nvidia-cutlass-dsl>=4.5.2`

Batch size, absolute head counts, sequence lengths, sparse-index capacity,
active topK, and non-leading tensor strides are runtime dynamic. Dtype, QK/value
dimensions, GQA ratio, `block_sizes` presence, exact split count, and the
single-kernel empty-row specialization select the static forward variant. The
combine variant is selected by dtype, value dimension, and
`ceil(log2(kv_splits))`; one combine artifact can therefore serve multiple main
variants.

#### Build artifacts

Build with the same CPU architecture, CUDA runtime, CUTLASS DSL version, and BSA
source revision as the deployment environment.

```bash
# Default: Dq=Dv=128, bf16/fp16, common GQA ratios, split 1/2/4/8.
make aot-sm90 SM90_AOT_DIR=/shared/bsa-sm90-aot

# Build a deployment-specific subset, including non-default head dimensions.
make aot-sm90 \
  SM90_AOT_DIR=/shared/bsa-sm90-aot \
  SM90_AOT_ARGS='--dtypes bf16 --qk-dims 64,128 --value-dims 128 --gqa-ratios 1,2,4 --kv-splits 1,2,4 --block-sizes both --allow-empty-block-nums both'

# List main and deduplicated combine variants without compiling them.
python -m csrc.fwd.sm90_blk64.aot_build --dry-run
```

The default matrix contains 140 main forward variants and 6 combine variants:

- Main: 2 dtypes × 7 GQA ratios × 2 block-size modes ×
  (`split1` empty-disabled/enabled + `split2/4/8` empty-disabled)
- Combine: 2 dtypes × `logsplit` 1/2/3

This covers every split count selected by `kv_splits="auto"`. Use
`--qk-dims`, `--value-dims`, and `--kv-splits` to add deployment-specific
variants rather than building the full Cartesian product.

The output bundle is self-describing and contains both artifact classes:

```text
/shared/bsa-sm90-aot/
└── x86_64/
    └── sm_90a/
        ├── manifest.json
        ├── bsa_sm90_blk64_bf16_qk128_v128_gqa1_bs0_split1_empty0_dyn.o
        ├── bsa_sm90_blk64_bf16_qk128_v128_gqa1_bs0_split1_empty0_dyn.h
        ├── bsa_sm90_blk64_bf16_qk128_v128_gqa1_bs0_split1_empty0_dyn.so
        ├── bsa_sm90_blk64_bf16_qk128_v128_gqa1_bs0_split4_empty0_dyn.so
        ├── bsa_sm90_blk64_combine_bf16_v128_logsplit2_dyn.so
        └── ...
```

The manifest stores main artifacts under `variants` and combine artifacts under
`combine_variants`. It also records the CPU/GPU target, CUTLASS DSL and CUDA
runtime versions, BSA source fingerprint, filenames, and `.so` SHA256 values.

#### Deploy and load

Set the artifact root before importing or calling BSA:

```bash
export BSA_SM90_AOT_DIR=/shared/bsa-sm90-aot

# Require the complete producer/combine path to be precompiled.
export BSA_SM90_AOT_ONLY=1
```

`BSA_SM90_AOT_DIR` may point to the bundle root shown above or directly to the
directory containing `manifest.json`. If unset, BSA searches
`csrc/fwd/sm90_blk64/aot_artifacts/<cpu-arch>/sm_90a/`.

With `BSA_SM90_AOT_ONLY=1`, a missing main variant or required combine variant
fails before JIT, and `cute.compile()` is never called. Without this flag, the
main and combine stages independently fall back to their existing JIT caches.
Manifest schema, version, source-fingerprint, and checksum failures always
report an error rather than silently falling back.

Current artifacts use the `dynamic_strided_nonbroadcast` layout class. Every
ABI tensor's leading mode must have stride 1, and stride-0 broadcast tensors are
not supported. Materialize tensors produced by `expand()` with `.contiguous()`
before an AOT-only call. Other strides, BHSD/BSHD boundary layouts, and runtime
sizes remain dynamic.

#### Validate a deployment

Build the variants required by the runtime matrix, then run:

```bash
BSA_SM90_AOT_DIR=/shared/bsa-sm90-aot \
BSA_SM90_AOT_ONLY=1 \
BSA_TEST_SM90_AOT_RUNTIME=1 \
python -m pytest tests/test_sm90_aot.py -q
```

The runtime validation monkeypatches `cute.compile()` to fail and includes
single-kernel, split producer/combine, mixed QK/value dimensions, GQA/MQA,
block-size layouts, and empty-row cases.

### SM120 CuTe DSL AOT

SM120 blk64 forward can be compiled offline and loaded through the CuTe native
ABI. This removes `cute.compile()` from the deployment process and avoids its
first-call memory peak. The Python attention API remains unchanged.

#### Supported configurations

- Target: `sm_120f`
- Dtype: BF16 and FP16
- QK/value head dimension: 128
- Attention: MHA, GQA, and MQA
- Block counts: fixed `block_sparse_num` or runtime `q2k_block_nums`
- `block_sizes`: absent, `[N]`, `[B, N]`, or `[B, Hq, N]`
- Split-KV: disabled; SM120 only accepts `kv_splits=1`
- Supported DSL range: `nvidia-cutlass-dsl>=4.5.2,<4.6`

Batch size, absolute head counts, sequence lengths, sparse index capacity,
active topK, and non-leading tensor strides are runtime dynamic. Dtype, D=128,
GQA ratio, fixed/variable block-count mode, and `block_sizes` rank select the
static AOT variant. For example, one `gqa2` artifact can run both `Hq/Hkv=4/2`
and `8/4`, but ratio 6 requires a `gqa6` artifact.

#### Build artifacts

Build the final artifacts with the same CPU architecture, CUDA runtime, CUTLASS
DSL version, and BSA source revision as the deployment environment.

```bash
# Default matrix: bf16/fp16, GQA ratios 1/2/4/8/16/32/64,
# fixed/variable block counts, and block_sizes modes 0/1/2/3.
make aot-sm120 SM120_AOT_DIR=/shared/bsa-sm120-aot

# A smaller deployment-specific matrix is usually preferable.
make aot-sm120 \
  SM120_AOT_DIR=/shared/bsa-sm120-aot \
  SM120_AOT_ARGS='--dtypes bf16,fp16 --gqa-ratios 1,2,6 --block-nums both --block-sizes-modes 0,1,2,3'

# List the default variants without compiling them.
python -m csrc.fwd.sm120_blk64.aot_build --dry-run
```

The default matrix contains 112 variants. A deployment-specific subset is
recommended to reduce build time and package size.

The output bundle is self-describing:

```text
/shared/bsa-sm120-aot/
└── x86_64/
    └── sm_120f/
        ├── manifest.json
        ├── bsa_sm120_blk64_bf16_gqa1_bn0_bs0_dyn.o
        ├── bsa_sm120_blk64_bf16_gqa1_bn0_bs0_dyn.h
        ├── bsa_sm120_blk64_bf16_gqa1_bn0_bs0_dyn.so
        └── ...
```

The manifest records the CPU/GPU target, CUTLASS DSL and CUDA runtime versions,
BSA source fingerprint, variant metadata, filenames, and `.so` SHA256 values.

#### Deploy and load

Set the artifact root before importing or calling BSA:

```bash
export BSA_SM120_AOT_DIR=/shared/bsa-sm120-aot

# Recommended for memory-constrained production deployments.
export BSA_SM120_AOT_ONLY=1
```

`BSA_SM120_AOT_DIR` may point either to the bundle root shown above or directly
to the directory containing `manifest.json`. If it is unset, BSA searches
`csrc/fwd/sm120_blk64/aot_artifacts/<cpu-arch>/sm_120f/`.

With `BSA_SM120_AOT_ONLY=1`, a missing artifact, missing variant, or unsupported
layout fails before JIT and `cute.compile()` is never called. Without this flag,
a missing manifest or variant falls back to the existing JIT cache. Version,
source-fingerprint, and checksum failures always report an error.

#### Call the SM120 kernel

Call `bsa_attn_fwd` with `sparse_block_size=64`. For fixed block counts, omit
`q2k_block_nums`; for variable counts, pass a `[B, Hq, Q_blocks]` int32 tensor.

```python
import torch

from block_sparse_attention import bsa_attn_fwd

# q: [B, Hq, Sq, 128], k/v: [B, Hkv, Sk, 128]
q = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 2, 2048, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)

# [B, Hq, Q_blocks, index_capacity], int32
q2k_block_index = ...
block_sizes = ...
out, lse = bsa_attn_fwd(
    q,
    k,
    v,
    q2k_block_index,
    16,
    block_sizes,
    return_lse=True,
    kv_splits=1,
    sparse_block_size=64,
)
```

Current AOT artifacts use the `dynamic_strided_nonbroadcast` layout class.
The head dimension must have stride 1, and stride-0 tensors produced by
`expand()` must be materialized with `.contiguous()` before an AOT-only call.

#### Validate a deployment

Run the AOT-only matrix after building the variants required by the test:

```bash
BSA_SM120_AOT_DIR=/shared/bsa-sm120-aot \
BSA_SM120_AOT_ONLY=1 \
BSA_TEST_SM120_AOT_RUNTIME=1 \
python -m pytest tests/test_sm120_aot.py -q
```

The validation monkeypatches `cute.compile()` to fail, so every passing launch
must come from a precompiled `.so`. The validated CUDA 13 configuration was
PyTorch CUDA 13.2, CUDA runtime 13.3, CUTLASS DSL 4.5.2, and a 16 GB SM120 GPU.

Common deployment errors:

| Error text | Resolution |
|---|---|
| `requires ... manifest.json` | Correct `BSA_SM120_AOT_DIR` or install the artifact bundle. |
| `missing variant` | Build the required dtype/GQA/block-count/block_sizes combination. |
| `CUTLASS DSL version mismatch` | Use the build-time CUTLASS DSL version or rebuild. |
| `CUDA version mismatch` | Rebuild under the deployment CUDA runtime. |
| `source fingerprint mismatch` | Use artifacts built from the delivered BSA source revision. |
| `checksum mismatch` | Replace the damaged or modified `.so`. |
| `stride-0 broadcast layouts` | Materialize expanded tensors with `.contiguous()`. |
| `does not support split-KV` | Set `kv_splits=1`. |

### Usage

```python
import torch
from block_sparse_attention import (
    bsa_attn_bwd,
    bsa_attn_fwd,
)

q = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
# Assume block_sparse_num, q2k_block_index, q2k_block_nums, and block_sizes
# describe the same sparse_block_size=64 metadata.

out_fixed, lse_fixed = bsa_attn_fwd(
    q, k, v, q2k_block_index, block_sparse_num, block_sizes,
    return_lse=True,
    sparse_block_size=64,
)

# Split long KV lists on SM90/SM100/SM110. SM120 must keep kv_splits=1.
out_split, lse_split = bsa_attn_fwd(
    q, k, v, q2k_block_index, block_sparse_num, block_sizes,
    q2k_block_nums=q2k_block_nums,
    return_lse=True,
    kv_splits="auto",
    sparse_block_size=64,
)

# BSHD compatibility path. The wrapper canonicalizes inputs to its backend layout.
q_bshd = q.transpose(1, 2)
k_bshd = k.transpose(1, 2)
v_bshd = v.transpose(1, 2)
out_bshd, lse_bshd = bsa_attn_fwd(
    q_bshd, k_bshd, v_bshd,
    q2k_block_index, block_sparse_num, block_sizes,
    return_lse=True,
    layout="bshd",
    sparse_block_size=64,
)

# Variable per-Q-block KV block counts
out_variable, lse_variable = bsa_attn_fwd(
    q, k, v, q2k_block_index, 0, block_sizes,
    q2k_block_nums=q2k_block_nums,
    return_lse=True,
    sparse_block_size=64,
)

dout_fixed = torch.randn_like(out_fixed)

dq, dk, dv = bsa_attn_bwd(
    dout_fixed, q, k, v, out_fixed, lse_fixed,
    q2k_block_index, block_sparse_num, block_sizes,
    sparse_block_size=64,
)

# BSHD compatibility path for backward as well.
dout_bshd = torch.randn_like(out_bshd)
dq_bshd, dk_bshd, dv_bshd = bsa_attn_bwd(
    dout_bshd, q_bshd, k_bshd, v_bshd, out_bshd, lse_bshd,
    q2k_block_index, block_sparse_num, block_sizes,
    layout="bshd",
    sparse_block_size=64,
)

# Optional tuning override for the bucketed k2q CSR backward task layout
dq, dk, dv = bsa_attn_bwd(
    dout_fixed, q, k, v, out_fixed, lse_fixed,
    q2k_block_index, block_sparse_num, block_sizes,
    bucket_size_blocks=512,
    sparse_block_size=64,
)
```

## API Reference

### `bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes, ..., sparse_block_size=64)`

Block-sparse forward attention. The repo canonical layout is `BHSD`; the
public API also supports explicit `layout="bshd"` compatibility at the wrapper
boundary. `sparse_block_size` selects the metadata and kernel block size and
must be 64 or 128. It defaults to 64; passing it explicitly is recommended so
the construction of `q2k_block_index` and `block_sizes` is unambiguous.

**Default tensor layout:** `(batch, num_heads, seqlen, head_dim)` (`BHSD`), last dim contiguous, 16-byte aligned.
When `layout="bshd"`, inputs and outputs use `(batch, seqlen, num_heads, head_dim)`.
The wrapper canonicalizes them at the backend boundary; SM100/SM110 blk64 uses
contiguous BHSD compatibility buffers, while other paths use views where their
kernel layout permits.
The last dimension must be `head_dim` with stride 1. Physical `BHDS` / non-contiguous-head-dim layouts are not supported.
The function always returns a two-tuple. By default it returns `(out, None)`.
Pass `return_lse=True` when the caller needs LSE, including before calling
`bsa_attn_bwd`; the returned LSE has shape `(batch, num_heads, seqlen_q)` and
dtype fp32. LSE is also retained when any of `q/k/v` requires gradients or when
an `lse` output buffer is supplied.

#### Input Tensors

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| `q` | (batch, num_heads, seqlen_q, head_dim) | bf16/fp16 | Query |
| `k` | (batch, num_heads_kv, seqlen_k, head_dim) | bf16/fp16 | Key |
| `v` | (batch, num_heads_kv, seqlen_k, head_dim_v) | bf16/fp16 | Value |

#### Block-Sparse Parameters (mandatory)

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `q2k_block_index` | (batch, num_heads, num_q_blocks, max_kv_blocks) | int32 | Per Q-block list of KV block indices, where `num_q_blocks = ceil(seqlen_q / sparse_block_size)` |
| `block_sparse_num` | scalar | int | Number of KV blocks per Q block (even, >= 2 for blk128; >= 1 for blk64). Ignored when `q2k_block_nums` is provided |
| `block_sizes` | (num_kv_blocks,) | int32 | Actual token count per KV block, bounded by `sparse_block_size` (for masking padding positions) |

#### Variable Block-Sparse Parameters (optional)

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `q2k_block_nums` | (batch, num_heads, num_q_blocks) | int32 | Per-Q-block KV block count. When provided, `block_sparse_num` is ignored. Empty rows are supported by SM90 and SM100/SM110; SM120 requires values >= 1 |
| `allow_empty_block_nums` | scalar | bool | Default True. When False, all `q2k_block_nums` values must be >= 1, enabling compile-time elimination of empty-tile branches (~2-3% faster) |

#### Other Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `softmax_scale` | `1/sqrt(head_dim)` | Softmax scale factor |
| `sparse_block_size` | `64` | Sparse Q/KV block size; must be 64 or 128 |
| `pack_gqa` | `None` (auto) | Whether to pack GQA heads; applies to the SM100/SM110 blk128 backend |
| `return_lse` | `False` | Return `(out, lse)` when true; otherwise return `(out, None)` unless gradients require LSE or an `lse` buffer is supplied |
| `out` | `None` | Pre-allocated output tensor |
| `lse` | `None` | Pre-allocated LSE tensor |
| `layout` | `"bhsd"` | Input/output layout: `"bhsd"` or `"bshd"` |
| `use_clc` | `None` (auto) | Scheduler selection for SM100/SM110 blk64 |
| `kv_splits` | `1` | KV-list split count for blk64, or `"auto"`; see below |

#### Dense Attention

For dense (full) attention, construct block-sparse args that cover all KV blocks:

```python
# See make_dense_block_sparse_args() in tests/test_flash_fwd.py
q2k_block_index = [0, 1, ..., N-1]  # for all Q blocks
block_sparse_num = N                  # must be even, >= 2
block_sizes = [tile_n] * N            # last block adjusted for seqlen remainder
```

#### blk64 split-KV

With `sparse_block_size=64`, the SM90 and SM100/SM110 forward paths can split
each Q block's active KV list. SM100/SM110 dispatches to the packaged CuTe DSL
backend; the backend is an implementation detail rather than a separate API.

BHSD is the zero-copy layout. On SM100/SM110, `layout="bshd"` is a compatibility
path that materializes contiguous BHSD inputs before launching the CuTe DSL
kernel, then materializes the BSHD output.

- `kv_splits=1` uses one forward kernel without partial-output
  workspace or a combine kernel.
- `kv_splits=2..256` produces FP32 O/LSE partials for each split and combines
  them into the requested output dtype. Partial workspace grows linearly with
  `kv_splits`.
- `kv_splits="auto"` selects 1, 2, 4, or 8 splits at KV block-count thresholds
  256, 450, and 900. With fixed block counts, the policy uses
  `block_sparse_num`; otherwise it uses `q2k_block_index.shape[-1]`. Auto lowers
  the split count if its estimated workspace does not fit; explicit split
  counts report an error instead.

SM90 split-KV supports the same MHA/GQA/MQA and QK/V dimensions (64, 96, or
128) as its single-kernel path. SM100/SM110 blk64 retains its existing shape
constraints. Its split path supports `use_clc=True` for persistent scheduling;
`use_clc=False` selects one tile per CTA. The default `use_clc=None` keeps CLC
disabled when `kv_splits>1` because the automatic scheduler policy has not been
tuned for split-KV. SM120 does not build or dispatch a split-KV variant:
`kv_splits=1` is the only accepted value, avoiding the split-dependent FP32
O/LSE workspace on memory-constrained devices.
On SM90, a compatible AOT bundle supplies both the split producer and the
deduplicated combine kernel; without matching artifacts, the two stages can
fall back independently to JIT.

### `bsa_attn_bwd(dout, q, k, v, out, lse, q2k_block_index, block_sparse_num, block_sizes, ..., sparse_block_size=64)`

SM90/SM100/SM110 block-sparse backward attention. Pass the same
`sparse_block_size` used to construct the forward metadata; the value must be
64 or 128 and defaults to 64.

This recomputes the attention probabilities from `q/k/v`, `out`, and `lse`, then returns gradients `(dq, dk, dv)`.
Obtain the forward LSE by calling `bsa_attn_fwd(..., return_lse=True)`.

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
| `sparse_block_size` | `64` | Sparse Q/KV block size; must match the forward metadata |

#### Backward Outputs

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| `dq` | same as `q` | bf16 | Gradient w.r.t. Q |
| `dk` | same as `k` | bf16 | Gradient w.r.t. K |
| `dv` | same as `v` | bf16 | Gradient w.r.t. V |

#### Backward Limitations

Backward currently supports BF16 MHA only (`num_heads == num_heads_kv`). SM90
supports blk64 with head dimension 128. SM100/SM110 supports blk64 with head
dimension 128 and blk128 with head dimension 64 or 128. SM120 backward is not
available.

### `bsa_attn_bwd(..., bucket_size_blocks=None)`

Bucketed k2q CSR backward path for long-sequence sparse attention.

The wrapper builds a GPU-side bucketed k2q CSR task layout from
`q2k_block_index` on every call. Both blk64 and blk128 backward dispatch through
this task representation.

The main backward kernel runs one task per `(q_group, kv_block)` and uses fp32 workspace accumulation for `dQ/dK/dV` before the final conversion/writeback.

Default bucket sizing is backend-owned. SM90 uses its blk64 default, while
SM100/SM110 selects the blk64 or blk128 default from `sparse_block_size`. Pass
`bucket_size_blocks` explicitly to override it for experiments.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bucket_size_blocks` | backend default | Number of Q blocks per bucket. Larger values reduce task count; smaller values improve `dQ_acc` locality |

## Tests & Benchmarks

```bash
make wheel                      # Build the unified CuTe DSL wheel
make setup                      # Reinstall wheel after dependencies are provisioned
make aot-sm90                   # Build the default 140 main + 6 combine SM90 matrix
make aot-sm120                  # Build the default SM120 native-ABI AOT matrix
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
make clean                      # Clear compile caches and build artifacts
make help                       # Show all targets

python test_flash_bwd.py                    # Backward quick correctness tests
python test_flash_bwd.py benchmark          # Backward benchmark

# On an SM90 deployment host after building the required AOT variants:
BSA_SM90_AOT_DIR=/shared/bsa-sm90-aot \
BSA_SM90_AOT_ONLY=1 \
BSA_TEST_SM90_AOT_RUNTIME=1 \
python -m pytest tests/test_sm90_aot.py -q

# On an SM120 deployment host after building the required AOT variants:
BSA_SM120_AOT_DIR=/shared/bsa-sm120-aot \
BSA_SM120_AOT_ONLY=1 \
BSA_TEST_SM120_AOT_RUNTIME=1 \
python -m pytest tests/test_sm120_aot.py -q
```
