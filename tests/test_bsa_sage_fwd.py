import math

import pytest
import torch


_V_PHYSICAL_TO_LOGICAL = (
    0,
    1,
    8,
    9,
    2,
    3,
    10,
    11,
    4,
    5,
    12,
    13,
    6,
    7,
    14,
    15,
)


def test_bsa_sage_probability_scale_contract() -> None:
    from csrc.fwd.sm120_blk64.bsa_fwd_sm120_sage import (
        SAGE_P_QUANT_SCALE,
        SAGE_P_RESCALE_THRESHOLD,
    )

    assert SAGE_P_QUANT_SCALE == 256.0
    assert SAGE_P_RESCALE_THRESHOLD == math.log2(448.0 / 256.0)


def _require_sm120() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 is required")


def _make_block_index(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_k: int,
    capacity: int,
) -> torch.Tensor:
    num_q_blocks = math.ceil(seqlen_q / 64)
    num_k_blocks = math.ceil(seqlen_k / 64)
    if capacity > num_k_blocks:
        raise ValueError("capacity exceeds the number of KV blocks")
    result = torch.empty(
        (batch, heads, num_q_blocks, capacity),
        device="cuda",
        dtype=torch.int32,
    )
    base = torch.arange(num_k_blocks, device="cuda")
    for batch_idx in range(batch):
        for head_idx in range(heads):
            for q_block_idx in range(num_q_blocks):
                shift = (batch_idx + 3 * head_idx + q_block_idx) % num_k_blocks
                result[batch_idx, head_idx, q_block_idx] = torch.roll(
                    base, shifts=shift
                )[:capacity]
    return result


def _decode_sage_v(v_fp8: torch.Tensor, seqlen_k: int) -> torch.Tensor:
    batch, heads, dim, padded_k = v_fp8.shape
    physical = v_fp8.float().reshape(batch, heads, dim, padded_k // 16, 16)
    physical = physical.permute(0, 1, 3, 4, 2)
    physical_to_logical = torch.tensor(
        _V_PHYSICAL_TO_LOGICAL,
        device=v_fp8.device,
    )
    logical_to_physical = torch.argsort(physical_to_logical)
    logical = physical[:, :, :, logical_to_physical, :]
    return logical.reshape(batch, heads, padded_k, dim)[:, :, :seqlen_k]


def _dequantized_sparse_reference(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_index: torch.Tensor,
    topk_num: int,
    softmax_scale: float,
    *,
    block_nums: torch.Tensor | None = None,
    block_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, heads, seqlen_q, dim = q_int8.shape
    seqlen_k = k_int8.shape[2]
    q = q_int8.float() * q_scale.repeat_interleave(32, dim=2)[
        ..., :seqlen_q, None
    ]
    k = k_int8.float() * k_scale.repeat_interleave(64, dim=2)[
        ..., :seqlen_k, None
    ]
    v = _decode_sage_v(v_fp8, seqlen_k) * v_scale[:, :, None, :]
    out = torch.empty_like(q)

    for batch_idx in range(batch):
        for head_idx in range(heads):
            for q_block_idx in range(math.ceil(seqlen_q / 64)):
                count = (
                    int(block_nums[batch_idx, head_idx, q_block_idx])
                    if block_nums is not None
                    else topk_num
                )
                token_ids = []
                for physical_idx in block_index[
                    batch_idx, head_idx, q_block_idx, :count
                ].tolist():
                    if block_sizes is None:
                        valid = min(64, seqlen_k - physical_idx * 64)
                    elif block_sizes.ndim == 1:
                        valid = int(block_sizes[physical_idx])
                    elif block_sizes.ndim == 2:
                        valid = int(block_sizes[batch_idx, physical_idx])
                    else:
                        valid = int(block_sizes[batch_idx, head_idx, physical_idx])
                    token_ids.extend(
                        range(physical_idx * 64, physical_idx * 64 + valid)
                    )
                token_ids_t = torch.tensor(
                    token_ids,
                    device=q.device,
                    dtype=torch.long,
                )
                q_begin = q_block_idx * 64
                q_end = min(q_begin + 64, seqlen_q)
                q_tile = q[batch_idx, head_idx, q_begin:q_end]
                k_tile = k[batch_idx, head_idx].index_select(0, token_ids_t)
                v_tile = v[batch_idx, head_idx].index_select(0, token_ids_t)
                probability = torch.softmax(
                    q_tile @ k_tile.transpose(0, 1) * softmax_scale,
                    dim=-1,
                )
                out[batch_idx, head_idx, q_begin:q_end] = probability @ v_tile
    return out


def _bf16_sparse_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_index: torch.Tensor,
    topk_num: int,
    softmax_scale: float,
) -> torch.Tensor:
    batch, heads, seqlen_q, _ = q.shape
    result = torch.empty_like(q, dtype=torch.float32)
    for batch_idx in range(batch):
        for head_idx in range(heads):
            for q_block_idx in range(math.ceil(seqlen_q / 64)):
                token_ids = []
                for block_idx in block_index[
                    batch_idx, head_idx, q_block_idx, :topk_num
                ].tolist():
                    begin = block_idx * 64
                    token_ids.extend(range(begin, min(begin + 64, k.shape[2])))
                token_ids_t = torch.tensor(token_ids, device=q.device)
                q_begin = q_block_idx * 64
                q_end = min(q_begin + 64, seqlen_q)
                q_tile = q[batch_idx, head_idx, q_begin:q_end].float()
                k_tile = k[batch_idx, head_idx].index_select(0, token_ids_t).float()
                v_tile = v[batch_idx, head_idx].index_select(0, token_ids_t).float()
                probability = torch.softmax(
                    q_tile @ k_tile.transpose(0, 1) * softmax_scale,
                    dim=-1,
                )
                result[batch_idx, head_idx, q_begin:q_end] = probability @ v_tile
    return result


def test_bsa_sage_blk64_rejects_cpu() -> None:
    from bsa_sage_blk64 import bsa_sage_blk64_fwd

    q = torch.empty((1, 1, 64, 128), dtype=torch.int8)
    k = torch.empty_like(q)
    v = torch.empty((1, 1, 128, 64), dtype=torch.float8_e4m3fn)
    scale = torch.empty((1, 1, 4), dtype=torch.float32)
    v_scale = torch.empty((1, 1, 128), dtype=torch.float32)
    index = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
    with pytest.raises(ValueError, match="CUDA"):
        bsa_sage_blk64_fwd(q, k, v, scale, scale[..., :1], v_scale, index, 1)


@pytest.mark.parametrize(
    "batch,heads,seqlen_q,seqlen_k,topk",
    [
        (1, 1, 64, 64, 1),
        (1, 2, 65, 127, 2),
        (2, 3, 96, 209, 3),
    ],
)
def test_bsa_sage_blk64_prequantized_forward(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_k: int,
    topk: int,
) -> None:
    _require_sm120()
    from bsa_sage_blk64 import bsa_sage_blk64_fwd
    from bsa_sage_quant import quantize_sage_qkv_sm120

    torch.manual_seed(1800 + batch + heads + topk)
    q = torch.randn(
        (batch, heads, seqlen_q, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    k = torch.randn(
        (batch, heads, seqlen_k, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    v = torch.randn_like(k)
    quantized = quantize_sage_qkv_sm120(q, k, v)
    q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale = quantized
    block_index = _make_block_index(batch, heads, seqlen_q, seqlen_k, topk)
    scale = 128**-0.5

    reference = _dequantized_sparse_reference(
        *quantized,
        block_index,
        topk,
        scale,
    )
    out_buffer = torch.empty_like(q)
    out = bsa_sage_blk64_fwd(
        q_int8,
        k_int8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        block_index,
        topk,
        scale,
        out=out_buffer,
    )

    assert out is out_buffer
    assert out.dtype == torch.bfloat16 and out.shape == q.shape
    difference = (out.float() - reference).abs()
    assert difference.max().item() < 0.15
    assert (difference.mean() / reference.abs().mean()).item() < 0.04


@pytest.mark.parametrize("block_sizes_mode", [1, 2, 3])
def test_bsa_sage_blk64_variable_sparse_metadata(block_sizes_mode: int) -> None:
    _require_sm120()
    from bsa_sage_blk64 import bsa_sage_blk64_fwd
    from bsa_sage_quant import quantize_sage_qkv_sm120

    batch, heads, seqlen_q, seqlen_k, capacity = 2, 2, 96, 256, 3
    torch.manual_seed(1900 + block_sizes_mode)
    q = torch.randn(
        (batch, heads, seqlen_q, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    k = torch.randn(
        (batch, heads, seqlen_k, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    v = torch.randn_like(k)
    quantized = quantize_sage_qkv_sm120(q, k, v)
    block_index = _make_block_index(
        batch, heads, seqlen_q, seqlen_k, capacity
    )
    q_block = torch.arange(math.ceil(seqlen_q / 64), device="cuda")
    head = torch.arange(heads, device="cuda").view(1, heads, 1)
    block_nums = (1 + (head + q_block) % capacity).expand(batch, -1, -1)
    block_nums = block_nums.to(torch.int32).contiguous()
    block_id = torch.arange(seqlen_k // 64, device="cuda", dtype=torch.int32)
    if block_sizes_mode == 1:
        block_sizes = 64 - block_id % 7
    elif block_sizes_mode == 2:
        batch_id = torch.arange(batch, device="cuda", dtype=torch.int32)[:, None]
        block_sizes = 64 - (batch_id + block_id[None, :]) % 7
    else:
        batch_id = torch.arange(batch, device="cuda", dtype=torch.int32)[:, None, None]
        head_id = torch.arange(heads, device="cuda", dtype=torch.int32)[None, :, None]
        block_sizes = 64 - (batch_id + head_id + block_id[None, None, :]) % 7
    scale = 128**-0.5

    reference = _dequantized_sparse_reference(
        *quantized,
        block_index,
        capacity,
        scale,
        block_nums=block_nums,
        block_sizes=block_sizes,
    )
    out = bsa_sage_blk64_fwd(
        *quantized,
        block_index,
        capacity,
        scale,
        block_sizes=block_sizes,
        q2k_block_nums=block_nums,
    )
    difference = (out.float() - reference).abs()
    assert difference.max().item() < 0.15
    assert (difference.mean() / reference.abs().mean()).item() < 0.04


def test_bsa_sage_blk64_bf16_end_to_end() -> None:
    _require_sm120()
    from bsa_sage_blk64 import bsa_sage_blk64_fwd
    from bsa_sage_quant import quantize_sage_qkv_sm120

    batch, heads, seqlen_q, seqlen_k, topk = 1, 2, 65, 193, 2
    torch.manual_seed(2001)
    q = torch.randn(
        (batch, heads, seqlen_q, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    k = torch.randn(
        (batch, heads, seqlen_k, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    v = torch.randn_like(k)
    block_index = _make_block_index(batch, heads, seqlen_q, seqlen_k, topk)
    scale = 128**-0.5
    reference = _bf16_sparse_reference(
        q, k, v, block_index, topk, scale
    )

    out = bsa_sage_blk64_fwd(
        *quantize_sage_qkv_sm120(q, k, v),
        block_index,
        topk,
        scale,
    )
    difference = (out.float() - reference).abs()
    assert difference.max().item() < 0.25
    assert (difference.mean() / reference.abs().mean()).item() < 0.05
