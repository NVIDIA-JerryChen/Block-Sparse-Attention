"""Regressions for warp convergence in the blk64 output correction path.

Also run this file under compute-sanitizer --tool synccheck: ordinary numerical
checks alone do not detect divergent warp-collective instructions.
"""

import math

import pytest
import torch


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("kv_splits", [1, 2])
@pytest.mark.parametrize("use_clc", [False, True])
@pytest.mark.parametrize("sq", [65, 128])
def test_sm100_blk64_zero_weight_rows(dtype, kv_splits, use_clc, sq):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Requires SM100/SM103")

    from bsa_attn_interface import _bsa_attn_fwd_sm100_blk64
    from bsa_fp8_quant import quantize_sage_bhsd

    torch.manual_seed(20261021)
    batch, heads, sk, dim, capacity = 1, 2, 512, 128, 8
    q = torch.zeros((batch, heads, sq, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((batch, heads, sk, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k) * 0.5

    # Adjacent rows prefer opposite KV block parities. The losing warp-pair's
    # weight underflows to zero for only half of its lanes.
    q[..., 0] = (torch.arange(sq, device="cuda") % 2 * 2 - 1) * 128
    k[..., 0] = (torch.arange(sk, device="cuda") // 64 % 2 * 2 - 1) * 128
    indices = torch.arange(capacity, device="cuda", dtype=torch.int32)
    indices = indices.view(1, 1, 1, capacity).expand(batch, heads, 2, capacity).contiguous()
    counts = torch.tensor([[[8, 0], [3, 7]]], device="cuda", dtype=torch.int32)
    sizes = torch.tensor([64, 40, 17, 64, 1, 63, 64, 32], device="cuda", dtype=torch.int32)

    scales = {}
    if dtype == "fp8":
        q, k, v, qs, ks, vs = quantize_sage_bhsd(q, k, v)
        scales = dict(q_scale=qs, k_scale=ks, v_scale=vs)
        q_ref = q.float() * qs.unsqueeze(-1)
        k_ref = k.float() * ks.repeat_interleave(16, dim=-1).unsqueeze(-1)
        v_ref = v.float() * vs.view(1, heads, 1, dim)
    else:
        q_ref, k_ref, v_ref = q.float(), k.float(), v.float()

    expected = torch.zeros_like(q_ref)
    expected_lse = torch.full((batch, heads, sq), -torch.inf, device="cuda")
    for head in range(heads):
        for qb in range(2):
            count = int(counts[0, head, qb])
            if count == 0:
                continue
            tokens = torch.tensor(
                [block * 64 + row for block in range(count) for row in range(int(sizes[block]))],
                device="cuda",
            )
            rows = slice(qb * 64, min((qb + 1) * 64, sq))
            scores = q_ref[0, head, rows] @ k_ref[0, head, tokens].T / math.sqrt(dim)
            expected[0, head, rows] = scores.softmax(-1) @ v_ref[0, head, tokens]
            expected_lse[0, head, rows] = scores.logsumexp(-1)

    out, lse = _bsa_attn_fwd_sm100_blk64(
        q, k, v, indices, sizes,
        q2k_block_nums=counts,
        block_sparse_num=capacity,
        kv_splits=kv_splits,
        use_clc=use_clc,
        **scales,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), expected, atol=0.004, rtol=0.03)
    torch.testing.assert_close(lse, expected_lse, atol=0.004, rtol=1e-5)
    assert torch.count_nonzero(out[0, 0, 64:]) == 0
    assert torch.isneginf(lse[0, 0, 64:]).all()
