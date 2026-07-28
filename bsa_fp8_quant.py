"""Sage FP8 quantization that keeps tensors in contiguous BHSD storage."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_q_bhsd_kernel(
    q_ptr,
    q8_ptr,
    scale_ptr,
    q_stride_b,
    q_stride_s,
    q_stride_h,
    q8_stride_b,
    q8_stride_s,
    q8_stride_h,
    scale_stride_b,
    scale_stride_h,
    seq_len,
    HEAD_DIM: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    seq_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    seq_idx = seq_block_idx * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    dim_idx = tl.arange(0, HEAD_DIM)
    valid = seq_idx < seq_len
    q = tl.load(
        q_ptr
        + batch_idx * q_stride_b
        + seq_idx[:, None] * q_stride_s
        + head_idx * q_stride_h
        + dim_idx[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    scale = tl.maximum(
        tl.max(tl.abs(q.to(tl.float32)), axis=1),
        1e-3,
    ) / 448.0

    # Match flashinfer-vx's C++ Q recipe exactly: scale is stored as FP32,
    # while reciprocal and multiply are rounded through the input dtype.
    scale_input = scale.to(q_ptr.dtype.element_ty)
    reciprocal_input = (1.0 / scale_input).to(q_ptr.dtype.element_ty)
    q_scaled = (q * reciprocal_input[:, None]).to(q_ptr.dtype.element_ty)
    tl.store(
        q8_ptr
        + batch_idx * q8_stride_b
        + seq_idx[:, None] * q8_stride_s
        + head_idx * q8_stride_h
        + dim_idx[None, :],
        q_scaled.to(q8_ptr.dtype.element_ty),
        mask=valid[:, None],
    )
    tl.store(
        scale_ptr + batch_idx * scale_stride_b + head_idx * scale_stride_h + seq_idx,
        scale,
        mask=valid,
    )


@triton.jit
def _quantize_k_bhsd_kernel(
    k_ptr,
    k8_ptr,
    scale_ptr,
    mean_ptr,
    k_stride_b,
    k_stride_s,
    k_stride_h,
    k8_stride_b,
    k8_stride_s,
    k8_stride_h,
    scale_stride_b,
    scale_stride_h,
    mean_stride_b,
    mean_stride_h,
    seq_len,
    HEAD_DIM: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
):
    seq_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    seq_idx = seq_block_idx * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    dim_idx = tl.arange(0, HEAD_DIM)
    valid = seq_idx < seq_len
    k = tl.load(
        k_ptr
        + batch_idx * k_stride_b
        + seq_idx[:, None] * k_stride_s
        + head_idx * k_stride_h
        + dim_idx[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    k_mean = tl.load(
        mean_ptr
        + batch_idx * mean_stride_b
        + head_idx * mean_stride_h
        + dim_idx[None, :]
    )
    k_centered = k - k_mean
    scale = tl.maximum(tl.max(tl.abs(k_centered)), 1e-3) / 448.0
    tl.store(
        k8_ptr
        + batch_idx * k8_stride_b
        + seq_idx[:, None] * k8_stride_s
        + head_idx * k8_stride_h
        + dim_idx[None, :],
        (k_centered / scale).to(k8_ptr.dtype.element_ty),
        mask=valid[:, None],
    )
    tl.store(
        scale_ptr
        + batch_idx * scale_stride_b
        + head_idx * scale_stride_h
        + seq_block_idx,
        scale,
    )


@torch.compile(mode="max-autotune-no-cudagraphs")
def _k_mean(k_bshd: torch.Tensor) -> torch.Tensor:
    return k_bshd.to(torch.float32).mean(dim=1)


@torch.compile(mode="max-autotune-no-cudagraphs")
def _v_scale(v_bhsd: torch.Tensor) -> torch.Tensor:
    v_max = v_bhsd.abs().float().amax(dim=(0, 2))
    return torch.maximum(v_max, torch.full_like(v_max, 1e-3)) / 448.0


@torch.compile
def _quantize_v(v_bhsd: torch.Tensor, v_scale: torch.Tensor) -> torch.Tensor:
    return (v_bhsd * v_scale.reciprocal().to(v_bhsd.dtype)[None, :, None, :]).to(
        torch.float8_e4m3fn
    )


def quantize_sage_bhsd(
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
):
    """Quantize BF16 BHSD Q/K/V for :func:`bsa_fp8_blk64_fwd`.

    The scale contract matches the flashinfer-vx Sage recipe used by the
    customer benchmark: Q per token, smoothed K per 16 tokens, and V per
    channel. Q/K kernels consume BSHD views but preserve the backing BHSD
    strides, avoiding all six physical BHSD-to-BSHD transpose copies.
    """
    tensors = (q_bhsd, k_bhsd, v_bhsd)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("Q, K, and V must be rank-4 BHSD tensors")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise TypeError("Sage FP8 quantization currently requires BF16 inputs")
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Q, K, and V must be CUDA tensors")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Q, K, and V must use contiguous BHSD storage")
    batch, heads, seqlen_q, head_dim = q_bhsd.shape
    if batch < 1 or heads < 1:
        raise ValueError("Sage FP8 quantization requires positive batch and head counts")
    if head_dim != 128:
        raise ValueError("Sage FP8 quantization requires D=128")
    is_sm120 = torch.cuda.get_device_capability(q_bhsd.device)[0] == 12
    if not is_sm120 and (batch != 1 or heads not in (4, 8)):
        raise ValueError("Sage FP8 v1 requires B=1, H in {4, 8}, and D=128")
    if k_bhsd.shape[:2] != (batch, heads) or k_bhsd.shape[-1] != head_dim:
        raise ValueError("K must match Q batch, heads, and head dimension")
    if v_bhsd.shape != k_bhsd.shape:
        raise ValueError("V must have the same shape as K")
    if not is_sm120 and (seqlen_q % 64 or k_bhsd.shape[2] % 64):
        raise ValueError("Q and K/V sequence lengths must be multiples of 64")

    q_bshd = q_bhsd.transpose(1, 2)
    q_fp8_bshd = torch.empty_like(q_bshd, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty(
        (batch, heads, seqlen_q),
        device=q_bhsd.device,
        dtype=torch.float32,
    )
    q_rows_per_program = 16
    _quantize_q_bhsd_kernel[(triton.cdiv(seqlen_q, q_rows_per_program), heads, batch)](
        q_bshd,
        q_fp8_bshd,
        q_scale,
        q_bshd.stride(0),
        q_bshd.stride(1),
        q_bshd.stride(2),
        q_fp8_bshd.stride(0),
        q_fp8_bshd.stride(1),
        q_fp8_bshd.stride(2),
        q_scale.stride(0),
        q_scale.stride(1),
        seqlen_q,
        HEAD_DIM=head_dim,
        ROWS_PER_PROGRAM=q_rows_per_program,
        num_warps=4,
    )
    q_fp8 = q_fp8_bshd.transpose(1, 2)

    seqlen_k = k_bhsd.shape[2]
    k_bshd = k_bhsd.transpose(1, 2)
    k_fp8_bshd = torch.empty_like(k_bshd, dtype=torch.float8_e4m3fn)
    k_scale = torch.empty(
        (batch, heads, triton.cdiv(seqlen_k, 16)),
        device=k_bhsd.device,
        dtype=torch.float32,
    )
    k_mean = _k_mean(k_bshd)
    _quantize_k_bhsd_kernel[(triton.cdiv(seqlen_k, 16), heads, batch)](
        k_bshd,
        k_fp8_bshd,
        k_scale,
        k_mean,
        k_bshd.stride(0),
        k_bshd.stride(1),
        k_bshd.stride(2),
        k_fp8_bshd.stride(0),
        k_fp8_bshd.stride(1),
        k_fp8_bshd.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        k_mean.stride(0),
        k_mean.stride(1),
        seqlen_k,
        HEAD_DIM=head_dim,
        SEQ_BLOCK=16,
    )
    k_fp8 = k_fp8_bshd.transpose(1, 2)

    v_scale = _v_scale(v_bhsd)
    v_fp8 = _quantize_v(v_bhsd, v_scale)
    if not (q_fp8.is_contiguous() and k_fp8.is_contiguous() and v_fp8.is_contiguous()):
        raise RuntimeError("quantized Q, K, and V must remain BHSD-contiguous")
    return q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale


__all__ = ["quantize_sage_bhsd"]
