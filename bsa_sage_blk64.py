"""Public API for native SM120 Sage INT8/FP8 blk64 BSA."""

import math
from typing import Optional

import torch


if __package__:
    from .bsa_attn_interface import _bsa_attn_fwd_sm120_sage_blk64
else:
    from bsa_attn_interface import _bsa_attn_fwd_sm120_sage_blk64


_SAGE_HEAD_DIM = 128
_SAGE_BLOCK_SIZE = 64


def _require_cuda_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def bsa_sage_blk64_fwd(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_block_index: torch.Tensor,
    topk_num: int,
    softmax_scale: Optional[float] = None,
    *,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run native Sage QK-INT8/PV-FP8 block-sparse attention on SM120.

    Q and K use contiguous BHSD INT8 storage. Q scales cover 32 query rows
    and are padded to four entries per Q128 block; K scales cover one K64
    block. V uses Sage's contiguous ``[B, H, D, round_up(Sk, 64)]`` E4M3
    storage with its 16-token physical permutation, and V scales are per
    ``[B, H, D]`` channel. The returned (or supplied) output is BHSD BF16.
    """
    for name, tensor in (
        ("q_int8", q_int8),
        ("k_int8", k_int8),
        ("v_fp8", v_fp8),
        ("q_scale", q_scale),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
        ("q2k_block_index", q2k_block_index),
    ):
        _require_cuda_tensor(name, tensor, device=q_int8.device)

    capability = torch.cuda.get_device_capability(q_int8.device)
    if capability != (12, 0):
        raise ValueError(
            "bsa_sage_blk64_fwd requires compute capability 12.0, "
            f"got {capability[0]}.{capability[1]}"
        )
    if q_int8.ndim != 4 or k_int8.ndim != 4 or v_fp8.ndim != 4:
        raise ValueError("q_int8, k_int8, and v_fp8 must be rank-4 tensors")
    batch, heads, seqlen_q, head_dim = q_int8.shape
    if batch < 1 or heads < 1 or seqlen_q < 1:
        raise ValueError("batch, heads, and query length must be positive")
    if head_dim != _SAGE_HEAD_DIM:
        raise ValueError("native SM120 Sage attention requires head_dim=128")
    if q_int8.dtype != torch.int8 or k_int8.dtype != torch.int8:
        raise ValueError("q_int8 and k_int8 must use torch.int8")
    if v_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError("v_fp8 must use torch.float8_e4m3fn")
    if not q_int8.is_contiguous() or not k_int8.is_contiguous():
        raise ValueError("q_int8 and k_int8 must be contiguous BHSD tensors")
    if not v_fp8.is_contiguous():
        raise ValueError("v_fp8 must be contiguous Sage HDS storage")

    if k_int8.shape[:2] != (batch, heads) or k_int8.shape[3] != head_dim:
        raise ValueError("k_int8 must have shape [B, H, Sk, 128]")
    seqlen_k = k_int8.shape[2]
    if seqlen_k < 1:
        raise ValueError("key/value length must be positive")
    padded_k = ((seqlen_k + _SAGE_BLOCK_SIZE - 1) // _SAGE_BLOCK_SIZE) * _SAGE_BLOCK_SIZE
    if v_fp8.shape != (batch, heads, head_dim, padded_k):
        raise ValueError(
            "v_fp8 must have shape "
            f"{(batch, heads, head_dim, padded_k)}, got {tuple(v_fp8.shape)}"
        )

    num_q_scales = ((seqlen_q + 127) // 128) * 4
    expected_scales = (
        ("q_scale", q_scale, (batch, heads, num_q_scales)),
        ("k_scale", k_scale, (batch, heads, padded_k // 64)),
        ("v_scale", v_scale, (batch, heads, head_dim)),
    )
    for name, tensor, shape in expected_scales:
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must use torch.float32")
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    num_q_blocks = (seqlen_q + 63) // 64
    if q2k_block_index.dtype != torch.int32:
        raise ValueError("q2k_block_index must use torch.int32")
    if q2k_block_index.ndim != 4 or q2k_block_index.shape[:3] != (
        batch,
        heads,
        num_q_blocks,
    ):
        raise ValueError(
            "q2k_block_index must have shape "
            f"[B, H, ceil(Sq/64), capacity], got {tuple(q2k_block_index.shape)}"
        )
    if q2k_block_index.shape[3] < 1:
        raise ValueError("q2k_block_index capacity must be positive")

    topk_num = int(topk_num)
    if q2k_block_nums is None or q2k_block_nums.numel() == 0:
        if not 1 <= topk_num <= q2k_block_index.shape[3]:
            raise ValueError("topk_num must be within q2k_block_index capacity")
    else:
        _require_cuda_tensor(
            "q2k_block_nums", q2k_block_nums, device=q_int8.device
        )
    if block_sizes is not None and block_sizes.numel() > 0:
        _require_cuda_tensor("block_sizes", block_sizes, device=q_int8.device)

    if out is not None:
        _require_cuda_tensor("out", out, device=q_int8.device)
        if out.dtype != torch.bfloat16:
            raise ValueError("out must use torch.bfloat16")
        if out.shape != q_int8.shape or not out.is_contiguous():
            raise ValueError("out must be contiguous with the same shape as q_int8")

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    softmax_scale = float(softmax_scale)
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")

    with torch.cuda.device(q_int8.device):
        return _bsa_attn_fwd_sm120_sage_blk64(
            q_int8,
            k_int8,
            v_fp8,
            q_scale,
            k_scale,
            v_scale,
            q2k_block_index,
            topk_num,
            softmax_scale,
            block_sizes=block_sizes,
            q2k_block_nums=q2k_block_nums,
            out=out,
        )


__all__ = ["bsa_sage_blk64_fwd"]
