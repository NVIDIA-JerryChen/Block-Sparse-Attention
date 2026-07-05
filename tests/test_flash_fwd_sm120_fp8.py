"""SM120 FP8 block-sparse forward attention tests."""

import math

import pytest
import torch

from bsa_attn_interface import bsa_attn_fwd, bsa_attn_fwd_blk64


def _quantize_sage(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Apply the Sage Q-token, K-16-token, and V-channel FP8 recipe."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    q_float = q.float()
    q_descale = q_float.abs().amax(dim=-1).clamp_min(1.0e-3) / fp8_max
    q_fp8 = (q_float / q_descale[..., None]).to(torch.float8_e4m3fn)

    k_centered = k.float() - k.float().mean(dim=2, keepdim=True)
    k_blocks = math.ceil(k.shape[2] / 16)
    k_padded = torch.nn.functional.pad(k_centered, (0, 0, 0, k_blocks * 16 - k.shape[2]))
    k_descale = (
        k_padded.view(k.shape[0], k.shape[1], k_blocks, 16, k.shape[-1])
        .abs()
        .amax(dim=(3, 4))
        .clamp_min(1.0e-3)
        / fp8_max
    )
    k_token_descale = k_descale.repeat_interleave(16, dim=-1)[..., : k.shape[2]]
    k_fp8 = (k_centered / k_token_descale[..., None]).to(torch.float8_e4m3fn)

    v_float = v.float()
    v_descale_hd = v_float.abs().amax(dim=(0, 2)).clamp_min(1.0e-3) / fp8_max
    v_fp8 = (v_float / v_descale_hd[None, :, None, :]).to(torch.float8_e4m3fn)
    v_descale = v_descale_hd.flatten().contiguous()
    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12),
    reason="SM120-only coverage",
)
def test_sm120_blk64_fp8_e4m3_odd_topk_q_tail_block_sizes():
    torch.manual_seed(2026)
    batch, heads, seqlen_q, seqlen_k, head_dim = 1, 2, 96, 256, 128
    block = 64

    q, k, v, q_descale, k_descale, v_descale = _quantize_sage(
        torch.randn(
            batch, heads, seqlen_q, head_dim, device="cuda", dtype=torch.bfloat16
        )
        * 0.5,
        torch.randn(
            batch, heads, seqlen_k, head_dim, device="cuda", dtype=torch.bfloat16
        )
        * 0.5,
        torch.randn(
            batch, heads, seqlen_k, head_dim, device="cuda", dtype=torch.bfloat16
        )
        * 0.5,
    )

    num_q_tiles = math.ceil(seqlen_q / block)
    indices = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    q2k_block_index = indices.view(1, 1, 1, 3).expand(
        batch, heads, num_q_tiles, 3
    ).contiguous()
    empty_block_nums = torch.empty(0, device="cuda", dtype=torch.int32)
    block_sizes = torch.tensor([64, 64, 17, 64], device="cuda", dtype=torch.int32)

    out, lse = bsa_attn_fwd_blk64(
        q,
        k,
        v,
        q2k_block_index,
        block_sizes,
        empty_block_nums,
        block_sparse_num=3,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    generic_out, generic_lse = bsa_attn_fwd(
        q,
        k,
        v,
        q2k_block_index,
        block_sparse_num=3,
        block_sizes=block_sizes,
        return_lse=True,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    columns = list(range(0, 64)) + list(range(128, 145)) + list(range(192, 256))
    q_ref = q.float() * q_descale[..., None]
    k_token_descale = k_descale.repeat_interleave(16, dim=-1)[..., :seqlen_k]
    k_ref = k.float() * k_token_descale[..., None]
    v_ref = v.float() * v_descale.view(heads, head_dim)[None, :, None, :]
    scores = torch.matmul(
        q_ref,
        k_ref[:, :, columns, :].transpose(-1, -2),
    ) * (head_dim ** -0.5)
    ref_out = torch.matmul(
        torch.softmax(scores, dim=-1),
        v_ref[:, :, columns, :],
    ).to(torch.bfloat16)
    ref_lse = torch.logsumexp(scores, dim=-1)

    assert out.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    torch.testing.assert_close(generic_out, out, rtol=0.0, atol=0.0)
    torch.testing.assert_close(generic_lse, lse, rtol=0.0, atol=0.0)
    torch.testing.assert_close(out, ref_out, rtol=5.0e-2, atol=2.0e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=1.0e-3, atol=1.0e-3)
