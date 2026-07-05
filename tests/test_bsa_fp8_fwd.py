import math

import pytest
import torch


def test_bsa_fp8_probability_scale_contract():
    from csrc.fwd.sm100_blk64.cutedsl.bsa_fwd_sm100 import SAGE_P_QUANT_SCALE

    assert SAGE_P_QUANT_SCALE == 448.0


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
        # Long-Q PDF rows already expose enough independent Q tiles and use
        # the same 1/2/4/8 split progression as the BF16 auto policy.
        (4, 119040, 188, 1),
        (8, 119040, 188, 1),
        (4, 234240, 368, 2),
        (8, 234240, 368, 2),
        (4, 349440, 548, 4),
        (8, 349440, 548, 4),
        (4, 464640, 729, 4),
        (8, 464640, 729, 4),
        (4, 579840, 909, 8),
        (8, 579840, 909, 8),
        (4, 695040, 1089, 8),
        (8, 695040, 1089, 8),
    ],
)
def test_bsa_fp8_split_policy(heads, seqlen_q, topk, expected_splits):
    from bsa_attn_interface import _sm100_blk64_auto_fp8_kv_splits

    assert (
        _sm100_blk64_auto_fp8_kv_splits(topk, heads, seqlen_q)
        == expected_splits
    )


def _require_sm100_or_sm110() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("FP8 blk64 BSA requires SM100/SM110")


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
    token_in_block = torch.arange(64, device=q.device)
    for batch_idx in range(b):
        for head_idx in range(h):
            for q_block in range(sq // 64):
                kv_blocks = block_index[batch_idx, head_idx, q_block].long()
                kv_tokens = (kv_blocks[:, None] * 64 + token_in_block).reshape(-1)
                q_tile = q[batch_idx, head_idx, q_block * 64 : (q_block + 1) * 64].float()
                k_tile = k[batch_idx, head_idx].index_select(0, kv_tokens).float()
                v_tile = v[batch_idx, head_idx].index_select(0, kv_tokens).float()
                probs = torch.softmax(q_tile @ k_tile.transpose(0, 1) * softmax_scale, dim=-1)
                out[batch_idx, head_idx, q_block * 64 : (q_block + 1) * 64] = probs @ v_tile
    return out


def _make_block_index(heads: int, sq: int, sk: int, topk: int) -> torch.Tensor:
    num_q_blocks = sq // 64
    num_k_blocks = sk // 64
    result = torch.empty((1, heads, num_q_blocks, topk), device="cuda", dtype=torch.int32)
    base = torch.arange(num_k_blocks, device="cuda")
    for head_idx in range(heads):
        for q_block in range(num_q_blocks):
            shift = (3 * head_idx + q_block) % num_k_blocks
            selected = torch.roll(base, shifts=shift)[:topk].sort().values
            result[0, head_idx, q_block] = selected.to(torch.int32)
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
