"""SM120 Sage-compatible INT8/FP8 quantization kernels."""

import triton
import triton.language as tl
from triton.language.extra import libdevice


SAGE_Q_GROUP_SIZE = 32
SAGE_Q_BLOCK_SIZE = 128
SAGE_K_BLOCK_SIZE = 64
SAGE_KV_STATS_CHUNK = 256
SAGE_HEAD_DIM = 128
SAGE_V_SCALE_MAX = 2.25


@triton.jit
def _quantize_sage_q_kernel(
    q_ptr,
    q8_ptr,
    scale_ptr,
    q_stride_b,
    q_stride_h,
    q_stride_s,
    q8_stride_b,
    q8_stride_h,
    q8_stride_s,
    scale_stride_b,
    scale_stride_h,
    seqlen_q,
    batch_size,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    group_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    row_idx = group_idx * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    dim_idx = tl.arange(0, HEAD_DIM)
    valid = row_idx < seqlen_q
    q = tl.load(
        q_ptr
        + batch_idx * q_stride_b
        + head_idx * q_stride_h
        + row_idx[:, None] * q_stride_s
        + dim_idx[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    row_max = tl.max(tl.abs(q), axis=1)
    amax = tl.maximum(tl.max(row_max, axis=0), 1.0e-7)
    scale = amax / 127.0
    q_quant = tl.maximum(
        tl.minimum(libdevice.rint(q / scale), 127.0),
        -127.0,
    )
    tl.store(
        q8_ptr
        + batch_idx * q8_stride_b
        + head_idx * q8_stride_h
        + row_idx[:, None] * q8_stride_s
        + dim_idx[None, :],
        q_quant.to(tl.int8),
        mask=valid[:, None],
    )
    tl.store(
        scale_ptr
        + batch_idx * scale_stride_b
        + head_idx * scale_stride_h
        + group_idx,
        scale,
    )


@triton.jit
def _sage_kv_stats_partial_kernel(
    k_ptr,
    v_ptr,
    k_partial_ptr,
    v_partial_ptr,
    k_stride_b,
    k_stride_h,
    k_stride_s,
    v_stride_b,
    v_stride_h,
    v_stride_s,
    partial_stride_b,
    partial_stride_h,
    partial_stride_c,
    seqlen_k,
    heads,
    batch_size,
    HEAD_DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DIM_TILE: tl.constexpr,
):
    chunk_idx = tl.program_id(0)
    dim_tile_idx = tl.program_id(1)
    batch_head_idx = tl.program_id(2)
    batch_idx = batch_head_idx // heads
    head_idx = batch_head_idx - batch_idx * heads

    row_idx = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    dim_idx = dim_tile_idx * DIM_TILE + tl.arange(0, DIM_TILE)
    valid = row_idx < seqlen_k
    k = tl.load(
        k_ptr
        + batch_idx * k_stride_b
        + head_idx * k_stride_h
        + row_idx[:, None] * k_stride_s
        + dim_idx[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr
        + batch_idx * v_stride_b
        + head_idx * v_stride_h
        + row_idx[:, None] * v_stride_s
        + dim_idx[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    partial_base = (
        batch_idx * partial_stride_b
        + head_idx * partial_stride_h
        + chunk_idx * partial_stride_c
        + dim_idx
    )
    tl.store(partial_base + k_partial_ptr, tl.sum(k, axis=0))
    tl.store(partial_base + v_partial_ptr, tl.max(tl.abs(v), axis=0))


@triton.jit
def _sage_kv_stats_finalize_kernel(
    k_partial_ptr,
    v_partial_ptr,
    k_mean_ptr,
    v_scale_ptr,
    partial_stride_b,
    partial_stride_h,
    partial_stride_c,
    mean_stride_b,
    mean_stride_h,
    scale_stride_b,
    scale_stride_h,
    num_chunks,
    seqlen_k,
    heads,
    batch_size,
    DIM_TILE: tl.constexpr,
    REDUCE_TILE: tl.constexpr,
    SCALE_MAX: tl.constexpr,
):
    dim_tile_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    batch_idx = batch_head_idx // heads
    head_idx = batch_head_idx - batch_idx * heads
    dim_idx = dim_tile_idx * DIM_TILE + tl.arange(0, DIM_TILE)

    sum_acc = tl.zeros((DIM_TILE,), tl.float32)
    max_acc = tl.zeros((DIM_TILE,), tl.float32)
    chunk_base = 0
    while chunk_base < num_chunks:
        chunk_idx = chunk_base + tl.arange(0, REDUCE_TILE)
        mask = chunk_idx < num_chunks
        partial_offset = (
            batch_idx * partial_stride_b
            + head_idx * partial_stride_h
            + chunk_idx[:, None] * partial_stride_c
            + dim_idx[None, :]
        )
        partial_sum = tl.load(
            k_partial_ptr + partial_offset,
            mask=mask[:, None],
            other=0.0,
        )
        partial_max = tl.load(
            v_partial_ptr + partial_offset,
            mask=mask[:, None],
            other=0.0,
        )
        sum_acc += tl.sum(partial_sum, axis=0)
        max_acc = tl.maximum(max_acc, tl.max(partial_max, axis=0))
        chunk_base += REDUCE_TILE

    mean = sum_acc / seqlen_k
    scale = tl.maximum(max_acc, 1.0e-7) / SCALE_MAX
    tl.store(
        k_mean_ptr
        + batch_idx * mean_stride_b
        + head_idx * mean_stride_h
        + dim_idx,
        mean,
    )
    tl.store(
        v_scale_ptr
        + batch_idx * scale_stride_b
        + head_idx * scale_stride_h
        + dim_idx,
        scale,
    )


@triton.jit
def _quantize_sage_kv_kernel(
    k_ptr,
    v_ptr,
    k_mean_ptr,
    v_scale_ptr,
    k8_ptr,
    v8_ptr,
    k_scale_ptr,
    k_stride_b,
    k_stride_h,
    k_stride_s,
    v_stride_b,
    v_stride_h,
    v_stride_s,
    mean_stride_b,
    mean_stride_h,
    v_scale_stride_b,
    v_scale_stride_h,
    k8_stride_b,
    k8_stride_h,
    k8_stride_s,
    v8_stride_b,
    v8_stride_h,
    v8_stride_d,
    k_scale_stride_b,
    k_scale_stride_h,
    seqlen_k,
    batch_size,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    local_row = tl.arange(0, BLOCK_SIZE)
    row_idx = block_idx * BLOCK_SIZE + local_row
    dim_idx = tl.arange(0, HEAD_DIM)
    valid = row_idx < seqlen_k

    mean = tl.load(
        k_mean_ptr
        + batch_idx * mean_stride_b
        + head_idx * mean_stride_h
        + dim_idx
    ).to(tl.float32)
    v_scale = tl.load(
        v_scale_ptr
        + batch_idx * v_scale_stride_b
        + head_idx * v_scale_stride_h
        + dim_idx
    ).to(tl.float32)
    k = tl.load(
        k_ptr
        + batch_idx * k_stride_b
        + head_idx * k_stride_h
        + row_idx[:, None] * k_stride_s
        + dim_idx[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr
        + batch_idx * v_stride_b
        + head_idx * v_stride_h
        + row_idx[:, None] * v_stride_s
        + dim_idx[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    k_centered = tl.where(valid[:, None], k - mean[None, :], 0.0)
    row_max = tl.max(tl.abs(k_centered), axis=1)
    k_amax = tl.maximum(tl.max(row_max, axis=0), 1.0e-7)
    k_scale = k_amax / 127.0
    k_quant = tl.maximum(
        tl.minimum(libdevice.rint(k_centered / k_scale), 127.0),
        -127.0,
    )
    tl.store(
        k8_ptr
        + batch_idx * k8_stride_b
        + head_idx * k8_stride_h
        + row_idx[:, None] * k8_stride_s
        + dim_idx[None, :],
        k_quant.to(tl.int8),
        mask=valid[:, None],
    )
    tl.store(
        k_scale_ptr
        + batch_idx * k_scale_stride_b
        + head_idx * k_scale_stride_h
        + block_idx,
        k_scale,
    )

    row_mod = local_row % 16
    physical_row = (
        block_idx * BLOCK_SIZE
        + (local_row // 16) * 16
        + (row_mod // 8) * 2
        + ((row_mod // 2) % 4) * 4
        + row_mod % 2
    )
    v_quant = tl.where(valid[:, None], v / v_scale[None, :], 0.0)
    tl.store(
        v8_ptr
        + batch_idx * v8_stride_b
        + head_idx * v8_stride_h
        + dim_idx[:, None] * v8_stride_d
        + physical_row[None, :],
        tl.trans(v_quant).to(v8_ptr.dtype.element_ty),
    )


__all__ = [
    "SAGE_HEAD_DIM",
    "SAGE_K_BLOCK_SIZE",
    "SAGE_KV_STATS_CHUNK",
    "SAGE_Q_BLOCK_SIZE",
    "SAGE_Q_GROUP_SIZE",
    "_quantize_sage_kv_kernel",
    "_quantize_sage_q_kernel",
    "_sage_kv_stats_finalize_kernel",
    "_sage_kv_stats_partial_kernel",
]
