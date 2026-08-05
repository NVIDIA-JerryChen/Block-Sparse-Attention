"""Public SM120 Sage INT8/FP8 quantization interface."""

from typing import Optional, Sequence

import torch
import triton

from block_sparse_attention.csrc.fwd.sm120_blk64.bsa_quant_sm120_sage import (
    SAGE_HEAD_DIM,
    SAGE_K_BLOCK_SIZE,
    SAGE_KV_STATS_CHUNK,
    SAGE_Q_BLOCK_SIZE,
    SAGE_Q_GROUP_SIZE,
    SAGE_V_SCALE_MAX,
    _quantize_sage_kv_kernel,
    _quantize_sage_q_kernel,
    _sage_kv_stats_finalize_kernel,
    _sage_kv_stats_partial_kernel,
)
from block_sparse_attention.csrc.fwd.sm120_blk64.quant_aot_runtime import (
    get_sm120_sage_quant_aot,
)


def _require_sm120_bf16_bhsd(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be a rank-4 BHSD tensor")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must use torch.bfloat16")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must use contiguous BHSD storage")
    if tensor.shape[-1] != SAGE_HEAD_DIM:
        raise ValueError(f"{name} must have head dimension 128")
    if tensor.shape[0] < 1 or tensor.shape[1] < 1 or tensor.shape[2] < 1:
        raise ValueError(f"{name} requires positive B, H, and sequence length")
    if torch.cuda.get_device_capability(tensor.device) != (12, 0):
        raise RuntimeError("SM120 Sage quantization requires compute capability 12.0")


def _require_output(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _normalize_out(
    out: Optional[Sequence[torch.Tensor]],
    specs: tuple[tuple[str, tuple[int, ...], torch.dtype], ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if out is None:
        return tuple(
            torch.empty(shape, dtype=dtype, device=device)
            for _, shape, dtype in specs
        )
    if len(out) != len(specs):
        raise ValueError(f"out must contain {len(specs)} tensors")
    result = tuple(out)
    for tensor, (name, shape, dtype) in zip(result, specs):
        _require_output(name, tensor, shape, dtype, device)
    return result


def _align_up(value: int, alignment: int = 256) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def _sage_kv_workspace_layout(k: torch.Tensor) -> tuple[int, int, int]:
    batch, heads, seqlen_k, _ = k.shape
    num_chunks = triton.cdiv(seqlen_k, SAGE_KV_STATS_CHUNK)
    partial_bytes = 2 * batch * heads * num_chunks * SAGE_HEAD_DIM * 4
    mean_offset = _align_up(partial_bytes)
    total_bytes = _align_up(mean_offset + batch * heads * SAGE_HEAD_DIM * 2)
    return int(num_chunks), int(mean_offset), int(total_bytes)


def sage_sm120_kv_quant_workspace_size(k: torch.Tensor) -> int:
    """Return the required byte size for the K/V quantization workspace."""
    if k.ndim != 4 or k.shape[-1] != SAGE_HEAD_DIM:
        raise ValueError("K must have shape [B, H, Sk, 128]")
    if k.shape[0] < 1 or k.shape[1] < 1 or k.shape[2] < 1:
        raise ValueError("K requires positive B, H, and sequence length")
    return _sage_kv_workspace_layout(k)[2]


def quantize_sage_q_sm120(
    q: torch.Tensor,
    *,
    out: Optional[Sequence[torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 BHSD Q to Sage per-warp INT8 on SM120."""
    _require_sm120_bf16_bhsd("Q", q)
    batch, heads, seqlen_q, head_dim = q.shape
    num_groups = triton.cdiv(seqlen_q, SAGE_Q_BLOCK_SIZE) * (
        SAGE_Q_BLOCK_SIZE // SAGE_Q_GROUP_SIZE
    )
    q_int8, q_scale = _normalize_out(
        out,
        (
            ("q_int8", tuple(q.shape), torch.int8),
            ("q_scale", (batch, heads, num_groups), torch.float32),
        ),
        q.device,
    )
    device_capability = torch.cuda.get_device_capability(q.device)
    aot_runtime = get_sm120_sage_quant_aot(
        device_capability[0] * 10 + device_capability[1]
    )
    if aot_runtime is not None:
        aot_runtime.quantize_q(q, q_int8, q_scale)
    else:
        _quantize_sage_q_kernel[(num_groups, heads, batch)](
            q,
            q_int8,
            q_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q_int8.stride(0),
            q_int8.stride(1),
            q_int8.stride(2),
            q_scale.stride(0),
            q_scale.stride(1),
            seqlen_q,
            batch,
            HEAD_DIM=head_dim,
            GROUP_SIZE=SAGE_Q_GROUP_SIZE,
            num_warps=8,
        )
    return q_int8, q_scale


def quantize_sage_kv_sm120(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: Optional[Sequence[torch.Tensor]] = None,
    workspace: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize BF16 BHSD K/V to Sage K64 INT8 and channelwise FP8."""
    _require_sm120_bf16_bhsd("K", k)
    _require_sm120_bf16_bhsd("V", v)
    if v.shape != k.shape:
        raise ValueError("V must have the same shape as K")
    if v.device != k.device:
        raise ValueError("K and V must be on the same CUDA device")

    batch, heads, seqlen_k, head_dim = k.shape
    padded_len = triton.cdiv(seqlen_k, SAGE_K_BLOCK_SIZE) * SAGE_K_BLOCK_SIZE
    num_blocks = padded_len // SAGE_K_BLOCK_SIZE
    specs = (
        ("k_int8", tuple(k.shape), torch.int8),
        ("v_fp8", (batch, heads, head_dim, padded_len), torch.float8_e4m3fn),
        ("k_scale", (batch, heads, num_blocks), torch.float32),
        ("v_scale", (batch, heads, head_dim), torch.float32),
    )
    k_int8, v_fp8, k_scale, v_scale = _normalize_out(out, specs, k.device)

    num_chunks, mean_offset, workspace_bytes = _sage_kv_workspace_layout(k)
    if workspace is None:
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=k.device)
    else:
        if workspace.dtype != torch.uint8 or not workspace.is_cuda:
            raise TypeError("workspace must be a CUDA uint8 tensor")
        if workspace.device != k.device or not workspace.is_contiguous():
            raise ValueError("workspace must be contiguous and on the K device")
        if workspace.numel() < workspace_bytes:
            raise ValueError(
                f"workspace requires at least {workspace_bytes} bytes, "
                f"got {workspace.numel()}"
            )

    partial_numel = batch * heads * num_chunks * head_dim
    partial = workspace[: 2 * partial_numel * 4].view(torch.float32)
    k_partial = partial[:partial_numel].view(batch, heads, num_chunks, head_dim)
    v_partial = partial[partial_numel:].view(batch, heads, num_chunks, head_dim)
    k_mean = workspace[
        mean_offset : mean_offset + batch * heads * head_dim * 2
    ].view(torch.bfloat16).view(batch, heads, head_dim)

    device_capability = torch.cuda.get_device_capability(k.device)
    aot_runtime = get_sm120_sage_quant_aot(
        device_capability[0] * 10 + device_capability[1]
    )
    if aot_runtime is not None:
        aot_runtime.stats_partial(k, v, k_partial, v_partial)
        aot_runtime.stats_finalize(
            k_partial,
            v_partial,
            k_mean,
            v_scale,
            seqlen_k,
        )
        aot_runtime.quantize_kv(
            k,
            v,
            k_mean,
            v_scale,
            k_int8,
            v_fp8,
            k_scale,
        )
    else:
        dim_tile = 32
        _sage_kv_stats_partial_kernel[
            (num_chunks, head_dim // dim_tile, batch * heads)
        ](
            k,
            v,
            k_partial,
            v_partial,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            k_partial.stride(0),
            k_partial.stride(1),
            k_partial.stride(2),
            seqlen_k,
            heads,
            batch,
            HEAD_DIM=head_dim,
            CHUNK_SIZE=SAGE_KV_STATS_CHUNK,
            DIM_TILE=dim_tile,
            num_warps=8,
        )
        _sage_kv_stats_finalize_kernel[(head_dim // dim_tile, batch * heads)](
            k_partial,
            v_partial,
            k_mean,
            v_scale,
            k_partial.stride(0),
            k_partial.stride(1),
            k_partial.stride(2),
            k_mean.stride(0),
            k_mean.stride(1),
            v_scale.stride(0),
            v_scale.stride(1),
            num_chunks,
            seqlen_k,
            heads,
            batch,
            DIM_TILE=dim_tile,
            REDUCE_TILE=16,
            SCALE_MAX=SAGE_V_SCALE_MAX,
            num_warps=4,
        )
        _quantize_sage_kv_kernel[(num_blocks, heads, batch)](
            k,
            v,
            k_mean,
            v_scale,
            k_int8,
            v_fp8,
            k_scale,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            k_mean.stride(0),
            k_mean.stride(1),
            v_scale.stride(0),
            v_scale.stride(1),
            k_int8.stride(0),
            k_int8.stride(1),
            k_int8.stride(2),
            v_fp8.stride(0),
            v_fp8.stride(1),
            v_fp8.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            seqlen_k,
            batch,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=SAGE_K_BLOCK_SIZE,
            num_warps=8,
        )
    return k_int8, v_fp8, k_scale, v_scale


def quantize_sage_qkv_sm120(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: Optional[Sequence[torch.Tensor]] = None,
    workspace: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Quantize BF16 BHSD Q/K/V using the native Sage SM120 contract."""
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
        raise ValueError("Q, K, and V must use matching batch/head dimensions")
    if out is None:
        q_out = None
        kv_out = None
    else:
        if len(out) != 6:
            raise ValueError("out must contain six tensors")
        q_out = (out[0], out[3])
        kv_out = (out[1], out[2], out[4], out[5])

    q_int8, q_scale = quantize_sage_q_sm120(q, out=q_out)
    k_int8, v_fp8, k_scale, v_scale = quantize_sage_kv_sm120(
        k,
        v,
        out=kv_out,
        workspace=workspace,
    )
    return q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale


__all__ = [
    "quantize_sage_kv_sm120",
    "quantize_sage_q_sm120",
    "quantize_sage_qkv_sm120",
    "sage_sm120_kv_quant_workspace_size",
]
