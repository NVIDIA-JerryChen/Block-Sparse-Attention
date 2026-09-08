import math

import pytest
import torch


def test_bsa_fp8_probability_scale_contract():
    from csrc.fwd.sm100_blk64.bsa_fwd_sm100 import (
        SAGE_P_QUANT_SCALE,
        SAGE_P_RESCALE_THRESHOLD,
    )

    assert SAGE_P_QUANT_SCALE == 256.0
    assert SAGE_P_RESCALE_THRESHOLD == math.log2(448.0 / 256.0)


@pytest.mark.parametrize(
    "heads,seqlen_q,topk,expected_splits",
    [
        # Short-Q rows retain the split policy tuned for launch parallelism.
        (4, 64, 1, 1),
        (4, 64, 127, 1),
        (4, 64, 128, 16),
        (4, 64, 256, 8),
        (4, 64, 368, 8),
        (4, 64, 450, 16),
        (4, 64, 548, 16),
        (8, 64, 188, 4),
        (8, 64, 548, 16),
        (8, 64, 1089, 16),
        # Long-Q rows already expose enough independent Q tiles.  Avoid the
        # partial-workspace/combine tax until a CTA traverses at least 900
        # sparse KV blocks, then use four splits.
        (4, 119040, 188, 1),
        (8, 119040, 188, 1),
        (4, 234240, 368, 1),
        (8, 234240, 368, 1),
        (4, 349440, 548, 1),
        (8, 349440, 548, 1),
        (4, 464640, 729, 1),
        (8, 464640, 729, 1),
        (4, 579840, 909, 4),
        (8, 579840, 909, 4),
        (4, 695040, 1089, 4),
        (8, 695040, 1089, 4),
        # SLA 0709 customer benchmark rows.
        (4, 116160, 186, 1),
        (8, 116160, 186, 1),
        (4, 109312, 174, 1),
        (8, 109312, 174, 1),
        (4, 216832, 342, 1),
        (8, 216832, 342, 1),
        (4, 695040, 1090, 4),
        (8, 695040, 1090, 4),
    ],
)
def test_bsa_fp8_split_policy(heads, seqlen_q, topk, expected_splits):
    from bsa_attn_interface import _sm100_blk64_auto_fp8_kv_splits

    assert (
        _sm100_blk64_auto_fp8_kv_splits(topk, heads, seqlen_q)
        == expected_splits
    )


def test_bsa_fp8_split_policy_counts_batch_tiles():
    from bsa_attn_interface import _sm100_blk64_auto_fp8_kv_splits

    assert _sm100_blk64_auto_fp8_kv_splits(188, 4, 4096, batch=1) == 16
    assert _sm100_blk64_auto_fp8_kv_splits(188, 4, 4096, batch=2) == 1


@pytest.mark.parametrize(
    "arch,batch,heads,seqlen_q,kv_splits,expected",
    [
        (100, 1, 4, 8192, 1, False),
        (103, 1, 4, 8128, 1, False),
        (103, 1, 4, 8192, 1, True),
        (103, 1, 8, 4096, 1, True),
        (103, 1, 4, 8192, 4, False),
    ],
)
def test_sm103_sage_fp8_ldred_policy(
    arch,
    batch,
    heads,
    seqlen_q,
    kv_splits,
    expected,
):
    from bsa_attn_interface import _sm103_blk64_use_sage_fp8_ldred

    assert (
        _sm103_blk64_use_sage_fp8_ldred(
            arch,
            batch,
            heads,
            seqlen_q,
            kv_splits,
        )
        is expected
    )


def _require_sm100_or_sm110() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("FP8 blk64 BSA requires SM100/SM110")


def _require_sm103() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Sage FP8 ld.red row-max requires SM103")


def _require_sm120() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120 FP8 blk64 BSA requires SM120")


def _sage_quantize_bhsd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Self-contained equivalent of the fixed flashinfer-vx Sage recipe."""
    fp8 = torch.float8_e4m3fn
    q_f32 = q.float()
    q_sfs = q_f32.abs().amax(dim=-1).clamp_min(1e-3) / 448.0
    q_fp8 = (q_f32 / q_sfs.unsqueeze(-1)).to(fp8)

    k_f32 = k.float()
    k_centered = k_f32 - k_f32.mean(dim=2, keepdim=True)
    b, h, sk, d = k.shape
    k_blocks = k_centered.view(b, h, sk // 16, 16, d)
    k_sfs = k_blocks.abs().amax(dim=(-1, -2)).clamp_min(1e-3) / 448.0
    k_fp8 = (
        k_centered
        / k_sfs.repeat_interleave(16, dim=-1).unsqueeze(-1)
    ).to(fp8)

    v_f32 = v.float()
    v_sfs = v_f32.abs().amax(dim=(0, 2)).clamp_min(1e-3) / 448.0
    v_fp8 = (v_f32 / v_sfs.view(1, h, 1, d)).to(fp8)
    return q_fp8, k_fp8, v_fp8, q_sfs.float(), k_sfs.float(), v_sfs.float()


def _dequantized_sparse_reference(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_sfs: torch.Tensor,
    k_sfs: torch.Tensor,
    v_sfs: torch.Tensor,
    block_index: torch.Tensor,
    softmax_scale: float,
    block_nums: torch.Tensor | None = None,
    block_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    b, h, sq, d = q_fp8.shape
    sk = k_fp8.shape[2]
    q = (q_fp8.float() * q_sfs.unsqueeze(-1)).bfloat16()
    k = (
        k_fp8.float()
        * k_sfs.repeat_interleave(16, dim=-1)[..., :sk].unsqueeze(-1)
    ).bfloat16()
    v = (v_fp8.float() * v_sfs.view(1, h, 1, d)).bfloat16()

    out = torch.empty((b, h, sq, d), device=q.device, dtype=torch.float32)
    for batch_idx in range(b):
        for head_idx in range(h):
            for q_block in range(math.ceil(sq / 64)):
                count = (
                    int(block_nums[batch_idx, head_idx, q_block])
                    if block_nums is not None
                    else block_index.shape[-1]
                )
                q_begin = q_block * 64
                q_end = min(q_begin + 64, sq)
                if count == 0:
                    out[batch_idx, head_idx, q_begin:q_end].zero_()
                    continue
                token_ids = []
                for physical_idx in block_index[
                    batch_idx, head_idx, q_block, :count
                ].tolist():
                    if block_sizes is None:
                        block_size = min(64, sk - physical_idx * 64)
                    elif block_sizes.ndim == 1:
                        block_size = int(block_sizes[physical_idx])
                    elif block_sizes.ndim == 2:
                        block_size = int(block_sizes[batch_idx, physical_idx])
                    else:
                        block_size = int(
                            block_sizes[batch_idx, head_idx, physical_idx]
                        )
                    token_ids.extend(
                        range(
                            physical_idx * 64,
                            physical_idx * 64 + block_size,
                        )
                    )
                kv_tokens = torch.tensor(
                    token_ids,
                    dtype=torch.long,
                    device=q.device,
                )
                q_tile = q[batch_idx, head_idx, q_begin:q_end].float()
                k_tile = k[batch_idx, head_idx].index_select(0, kv_tokens).float()
                v_tile = v[batch_idx, head_idx].index_select(0, kv_tokens).float()
                probs = torch.softmax(q_tile @ k_tile.transpose(0, 1) * softmax_scale, dim=-1)
                out[batch_idx, head_idx, q_begin:q_end] = probs @ v_tile
    return out


def _make_block_index(
    heads: int,
    sq: int,
    sk: int,
    topk: int,
    batch: int = 1,
) -> torch.Tensor:
    num_q_blocks = math.ceil(sq / 64)
    num_k_blocks = math.ceil(sk / 64)
    result = torch.empty(
        (batch, heads, num_q_blocks, topk),
        device="cuda",
        dtype=torch.int32,
    )
    base = torch.arange(num_k_blocks, device="cuda")
    for batch_idx in range(batch):
        for head_idx in range(heads):
            for q_block in range(num_q_blocks):
                shift = (batch_idx + 3 * head_idx + q_block) % num_k_blocks
                selected = torch.roll(base, shifts=shift)[:topk].sort().values
                result[batch_idx, head_idx, q_block] = selected.to(torch.int32)
    return result


@pytest.mark.parametrize(
    "heads,sq,sk,topk,flatten_v_scale",
    [
        (4, 128, 640, 1, True),
        (4, 128, 640, 5, False),
        (8, 64, 640, 9, True),
        (4, 64, 75200, 188, True),
        (4, 64, 75200, 1089, True),
    ],
)
def test_bsa_fp8_blk64_forward(heads, sq, sk, topk, flatten_v_scale):
    _require_sm100_or_sm110()
    from bsa_fp8_blk64 import bsa_fp8_blk64_fwd

    torch.manual_seed(1000 + heads + topk)
    q = torch.randn((1, heads, sq, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn((1, heads, sk, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn((1, heads, sk, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    q_fp8, k_fp8, v_fp8, q_sfs, k_sfs, v_sfs = _sage_quantize_bhsd(q, k, v)
    block_index = _make_block_index(heads, sq, sk, topk)
    scale = 1.0 / math.sqrt(128)

    ref = _dequantized_sparse_reference(
        q_fp8, k_fp8, v_fp8, q_sfs, k_sfs, v_sfs, block_index, scale
    )
    out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs.flatten() if flatten_v_scale else v_sfs,
        block_index,
        topk,
        scale,
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == q.shape
    diff = (out.float() - ref).abs()
    assert diff.max().item() < 0.15
    # The accepted Sage FP8 contract allows up to 2.9% relative mean error;
    # the broader 28-case validation currently tops out at about 2.82%.
    assert (diff.mean() / ref.abs().mean()).item() < 0.029


@pytest.mark.parametrize(
    "batch,heads,sq,sk,topk",
    [
        (1, 1, 65, 127, 1),
        (1, 2, 65, 128, 2),
        (1, 5, 96, 209, 3),
        (1, 7, 64, 128, 2),
        (2, 3, 65, 127, 2),
    ],
)
def test_bsa_fp8_blk64_dynamic_shape_sm100(batch, heads, sq, sk, topk):
    _require_sm100_or_sm110()
    from bsa_fp8_blk64 import bsa_fp8_blk64_fwd, quantize_sage_bhsd

    torch.manual_seed(1050 + 100 * batch + 10 * heads + topk)
    q = torch.randn(
        (batch, heads, sq, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    k = torch.randn(
        (batch, heads, sk, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    v = torch.randn_like(k) * 0.5
    q_fp8, k_fp8, v_fp8, q_sfs, k_sfs, v_sfs = quantize_sage_bhsd(
        q, k, v
    )
    block_index = _make_block_index(heads, sq, sk, topk, batch)
    scale = 1.0 / math.sqrt(128)

    ref = _dequantized_sparse_reference(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        scale,
    )
    out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        topk,
        scale,
    )
    cached_out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        topk,
        scale,
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == q.shape
    assert out.is_contiguous()
    torch.testing.assert_close(cached_out, out, rtol=0, atol=0)
    diff = (out.float() - ref).abs()
    assert diff.max().item() < 0.15
    assert (diff.mean() / ref.abs().mean()).item() < 0.029


@pytest.mark.parametrize("heads", [1, 5])
@pytest.mark.parametrize("kv_splits", [1, 4, 8, 16])
def test_bsa_fp8_blk64_dynamic_heads_all_splits_sm100(heads, kv_splits):
    _require_sm100_or_sm110()
    from bsa_attn_interface import _bsa_attn_fwd_sm100_blk64
    from bsa_fp8_quant import quantize_sage_bhsd

    torch.manual_seed(1070 + 10 * heads + kv_splits)
    sq, sk, topk = 64, 2048, 32
    q = torch.randn(
        (1, heads, sq, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    k = torch.randn(
        (1, heads, sk, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    v = torch.randn_like(k) * 0.5
    q_fp8, k_fp8, v_fp8, q_sfs, k_sfs, v_sfs = quantize_sage_bhsd(
        q, k, v
    )
    block_index = _make_block_index(heads, sq, sk, topk)
    scale = 1.0 / math.sqrt(128)
    ref = _dequantized_sparse_reference(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        scale,
    )

    out, _ = _bsa_attn_fwd_sm100_blk64(
        q_fp8,
        k_fp8,
        v_fp8,
        block_index,
        None,
        softmax_scale=scale,
        block_sparse_num=topk,
        use_clc=False,
        kv_splits=kv_splits,
        q_scale=q_sfs,
        k_scale=k_sfs,
        v_scale=v_sfs,
    )

    diff = (out.float() - ref).abs()
    assert diff.max().item() < 0.15
    assert (diff.mean() / ref.abs().mean()).item() < 0.035


@pytest.mark.parametrize(
    "heads,topk",
    [
        (4, 4),
        (4, 128),
    ],
)
def test_bsa_fp8_blk64_block_sizes_sm100(heads, topk):
    _require_sm100_or_sm110()
    from bsa_fp8_blk64 import bsa_fp8_blk64_fwd, quantize_sage_bhsd

    torch.manual_seed(1100 + topk)
    sq = 64
    sk = topk * 64
    q = torch.randn((1, heads, sq, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn((1, heads, sk, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k)
    q_fp8, k_fp8, v_fp8, q_sfs, k_sfs, v_sfs = quantize_sage_bhsd(q, k, v)
    block_index = _make_block_index(heads, sq, sk, topk)
    block_sizes = torch.full((topk,), 64, device="cuda", dtype=torch.int32)
    block_sizes[-1] = 40
    if topk > 4:
        block_sizes[1] = 1
        block_sizes[16] = 17
        block_sizes[63] = 63
    scale = 1.0 / math.sqrt(128)

    ref = _dequantized_sparse_reference(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        scale,
        block_sizes=block_sizes,
    )
    out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        topk,
        scale,
        block_sizes=block_sizes,
    )
    cached_out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        topk,
        scale,
        block_sizes=block_sizes,
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == q.shape
    torch.testing.assert_close(cached_out, out, rtol=0, atol=0)
    diff = (out.float() - ref).abs()
    assert diff.max().item() < 0.15
    assert (diff.mean() / ref.abs().mean()).item() < 0.029


def test_bsa_fp8_blk64_sm103_ldred_with_partial_block():
    _require_sm103()
    from bsa_fp8_blk64 import bsa_fp8_blk64_fwd, quantize_sage_bhsd

    torch.manual_seed(1103)
    heads, sq, sk, topk = 4, 8192, 256, 4
    q = torch.randn((1, heads, sq, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn((1, heads, sk, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k)
    q_fp8, k_fp8, v_fp8, q_sfs, k_sfs, v_sfs = quantize_sage_bhsd(q, k, v)
    block_index = (
        torch.arange(topk, device="cuda", dtype=torch.int32)
        .view(1, 1, 1, topk)
        .expand(1, heads, sq // 64, topk)
        .contiguous()
    )
    block_sizes = torch.tensor(
        [64, 64, 64, 40],
        device="cuda",
        dtype=torch.int32,
    )
    scale = 1.0 / math.sqrt(128)

    q_dequant = q_fp8.float() * q_sfs.unsqueeze(-1)
    k_dequant = (
        k_fp8.float()
        * k_sfs.repeat_interleave(16, dim=-1).unsqueeze(-1)
    )
    v_dequant = v_fp8.float() * v_sfs.view(1, heads, 1, 128)
    valid_tokens = torch.arange(232, device="cuda")
    ref = torch.softmax(
        torch.matmul(
            q_dequant,
            k_dequant.index_select(2, valid_tokens).transpose(-1, -2),
        )
        * scale,
        dim=-1,
    ) @ v_dequant.index_select(2, valid_tokens)

    out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        topk,
        scale,
        block_sizes=block_sizes,
    )
    cached_out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        topk,
        scale,
        block_sizes=block_sizes,
    )

    torch.testing.assert_close(cached_out, out, rtol=0, atol=0)
    diff = (out.float() - ref).abs()
    assert diff.max().item() < 0.15
    assert (diff.mean() / ref.abs().mean()).item() < 0.029


@pytest.mark.parametrize("block_sizes_mode", [0, 1, 2, 3])
def test_bsa_fp8_blk64_variable_metadata_sm100(block_sizes_mode):
    _require_sm100_or_sm110()
    from bsa_fp8_blk64 import bsa_fp8_blk64_fwd, quantize_sage_bhsd

    batch, heads, sq, sk, capacity = 2, 3, 128, 320, 4
    torch.manual_seed(1150 + block_sizes_mode)
    q = torch.randn(
        (batch, heads, sq, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    k = torch.randn(
        (batch, heads, sk, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    v = torch.randn_like(k) * 0.5
    q_fp8, k_fp8, v_fp8, q_sfs, k_sfs, v_sfs = quantize_sage_bhsd(
        q, k, v
    )
    block_index = _make_block_index(heads, sq, sk, capacity, batch)
    num_q_blocks = sq // 64
    num_kv_blocks = sk // 64
    batch_ids = torch.arange(
        batch, dtype=torch.int32, device="cuda"
    ).view(batch, 1, 1)
    head_ids = torch.arange(
        heads, dtype=torch.int32, device="cuda"
    ).view(1, heads, 1)
    q_block_ids = torch.arange(
        num_q_blocks, dtype=torch.int32, device="cuda"
    ).view(1, 1, num_q_blocks)
    block_nums = (batch_ids + head_ids + q_block_ids) % (capacity + 1)

    block_ids = torch.arange(
        num_kv_blocks, dtype=torch.int32, device="cuda"
    )
    if block_sizes_mode == 0:
        block_sizes = None
    elif block_sizes_mode == 1:
        block_sizes = 64 - block_ids % 9
    elif block_sizes_mode == 2:
        block_sizes = 64 - (batch_ids[:, 0] + block_ids.view(1, -1)) % 9
    else:
        block_sizes = 64 - (
            batch_ids + head_ids + block_ids.view(1, 1, -1)
        ) % 9
    scale = 1.0 / math.sqrt(128)

    ref = _dequantized_sparse_reference(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        scale,
        block_nums,
        block_sizes,
    )
    out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        0,
        scale,
        block_sizes=block_sizes,
        q2k_block_nums=block_nums,
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == q.shape
    diff = (out.float() - ref).abs()
    assert diff.max().item() < 0.15
    assert (diff.mean() / ref.abs().mean()).item() < 0.029


def test_bsa_fp8_blk64_variable_metadata_split_sm100():
    _require_sm100_or_sm110()
    from bsa_fp8_blk64 import bsa_fp8_blk64_fwd, quantize_sage_bhsd

    batch, heads, sq, sk, capacity = 1, 2, 128, 8192, 128
    torch.manual_seed(1170)
    q = torch.randn(
        (batch, heads, sq, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    k = torch.randn(
        (batch, heads, sk, 128), device="cuda", dtype=torch.bfloat16
    ) * 0.5
    v = torch.randn_like(k) * 0.5
    q_fp8, k_fp8, v_fp8, q_sfs, k_sfs, v_sfs = quantize_sage_bhsd(
        q, k, v
    )
    block_index = _make_block_index(heads, sq, sk, capacity, batch)
    block_nums = torch.tensor(
        [[[0, 17], [64, 127]]], device="cuda", dtype=torch.int32
    )
    scale = 1.0 / math.sqrt(128)

    ref = _dequantized_sparse_reference(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        scale,
        block_nums,
    )
    out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        0,
        scale,
        q2k_block_nums=block_nums,
    )

    assert torch.count_nonzero(out[0, 0, :64]) == 0
    diff = (out.float() - ref).abs()
    assert diff.max().item() < 0.15
    assert (diff.mean() / ref.abs().mean()).item() < 0.035


@pytest.mark.parametrize(
    "batch,heads,sq,sk,topk,flatten_v_scale",
    [
        (1, 2, 96, 209, 3, True),
        (2, 3, 65, 127, 2, False),
        (1, 4, 128, 640, 1, True),
        (1, 4, 128, 640, 5, False),
        (1, 8, 64, 640, 9, True),
        (1, 4, 64, 75200, 188, True),
        (1, 4, 64, 75200, 1089, True),
    ],
)
def test_bsa_fp8_blk64_forward_sm120(
    batch,
    heads,
    sq,
    sk,
    topk,
    flatten_v_scale,
):
    _require_sm120()
    from bsa_fp8_blk64 import bsa_fp8_blk64_fwd, quantize_sage_bhsd

    torch.manual_seed(1200 + batch + heads + topk)
    q = torch.randn(
        (batch, heads, sq, 128),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.5
    k = torch.randn(
        (batch, heads, sk, 128),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.5
    v = torch.randn(
        (batch, heads, sk, 128),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.5
    q_fp8, k_fp8, v_fp8, q_sfs, k_sfs, v_sfs = quantize_sage_bhsd(q, k, v)
    block_index = _make_block_index(heads, sq, sk, topk, batch)
    scale = 1.0 / math.sqrt(128)

    ref = _dequantized_sparse_reference(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        scale,
    )
    out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs.flatten() if flatten_v_scale else v_sfs,
        block_index,
        topk,
        scale,
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == q.shape
    diff = (out.float() - ref).abs()
    assert diff.max().item() < 0.15
    assert (diff.mean() / ref.abs().mean()).item() < 0.029


@pytest.mark.parametrize(
    "heads,sq,sk,capacity,block_sizes_mode",
    [
        (4, 128, 256, 3, 0),
        (4, 128, 256, 3, 1),
        (4, 128, 320, 4, 2),
        (8, 64, 320, 4, 3),
    ],
)
def test_bsa_fp8_blk64_variable_metadata_sm120(
    heads,
    sq,
    sk,
    capacity,
    block_sizes_mode,
):
    _require_sm120()
    from bsa_fp8_blk64 import bsa_fp8_blk64_fwd, quantize_sage_bhsd

    torch.manual_seed(1210 + heads + capacity + block_sizes_mode)
    q = torch.randn((1, heads, sq, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn((1, heads, sk, 128), device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn_like(k)
    q_fp8, k_fp8, v_fp8, q_sfs, k_sfs, v_sfs = quantize_sage_bhsd(q, k, v)
    block_index = _make_block_index(heads, sq, sk, capacity)
    num_q_blocks = sq // 64
    num_kv_blocks = sk // 64
    head_ids = torch.arange(
        heads,
        dtype=torch.int32,
        device="cuda",
    ).unsqueeze(1)
    q_block_ids = torch.arange(
        num_q_blocks,
        dtype=torch.int32,
        device="cuda",
    ).unsqueeze(0)
    block_nums = (1 + (head_ids + q_block_ids) % capacity).unsqueeze(0)

    block_ids = torch.arange(
        num_kv_blocks,
        dtype=torch.int32,
        device="cuda",
    )
    if block_sizes_mode == 0:
        block_sizes = None
    elif block_sizes_mode == 1:
        block_sizes = 64 - block_ids % 9
    elif block_sizes_mode == 2:
        block_sizes = (64 - block_ids % 9).unsqueeze(0)
    else:
        block_sizes = (64 - (head_ids + block_ids.unsqueeze(0)) % 9).unsqueeze(0)
    scale = 1.0 / math.sqrt(128)

    ref = _dequantized_sparse_reference(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        scale,
        block_nums,
        block_sizes,
    )
    out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_sfs,
        k_sfs,
        v_sfs,
        block_index,
        0,
        scale,
        block_sizes=block_sizes,
        q2k_block_nums=block_nums,
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == q.shape
    diff = (out.float() - ref).abs()
    assert diff.max().item() < 0.15
    assert (diff.mean() / ref.abs().mean()).item() < 0.029
