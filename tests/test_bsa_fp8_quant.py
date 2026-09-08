import pytest
import torch


def test_quantize_sage_bhsd_rejects_cpu_inputs():
    from bsa_fp8_quant import quantize_sage_bhsd

    q = torch.empty((1, 4, 64, 128), dtype=torch.bfloat16)
    k = torch.empty((1, 4, 64, 128), dtype=torch.bfloat16)
    v = torch.empty_like(k)
    with pytest.raises(ValueError, match="CUDA"):
        quantize_sage_bhsd(q, k, v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quantize_sage_bhsd_supports_sequence_tails():
    from bsa_fp8_quant import quantize_sage_bhsd

    q = torch.empty((1, 4, 65, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((1, 4, 64, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    actual = quantize_sage_bhsd(q, k, v)
    assert actual[0].shape == q.shape
    assert actual[1].shape == k.shape
    assert actual[2].shape == v.shape


@pytest.mark.parametrize(
    "batch,heads,seqlen_q,seqlen_k",
    [(1, 1, 65, 79), (1, 5, 33, 70), (2, 3, 96, 127)],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quantize_sage_bhsd_dynamic_contract(
    batch,
    heads,
    seqlen_q,
    seqlen_k,
):
    from bsa_fp8_quant import quantize_sage_bhsd

    q = torch.randn(
        (batch, heads, seqlen_q, 128), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        (batch, heads, seqlen_k, 128), device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    actual = quantize_sage_bhsd(q, k, v)

    assert actual[0].shape == q.shape
    assert actual[1].shape == k.shape
    assert actual[2].shape == v.shape
    assert actual[3].shape == (batch, heads, seqlen_q)
    assert actual[4].shape == (batch, heads, (seqlen_k + 15) // 16)
    assert actual[5].shape == (heads, 128)
    assert all(tensor.is_contiguous() for tensor in actual)


@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quantize_sage_bhsd_matches_customer_recipe(heads):
    sage_quant = pytest.importorskip("flashinfer_vx.sage_quant")
    from bsa_fp8_blk64 import bsa_fp8_blk64_fwd
    from bsa_fp8_quant import quantize_sage_bhsd

    torch.manual_seed(20260709)
    q = torch.randn(1, heads, 128, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, heads, 192, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, heads, 192, 128, device="cuda", dtype=torch.bfloat16)

    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    expected = sage_quant.quantize(
        q_bshd,
        k_bshd,
        v_bshd,
        sage_block_size_q=1,
        sage_block_size_k=16,
        sage_block_size_v=1,
        smooth=True,
    )[:6]
    expected = (
        expected[0].transpose(1, 2).contiguous(),
        expected[1].transpose(1, 2).contiguous(),
        expected[2].transpose(1, 2).contiguous(),
        expected[3],
        expected[4],
        expected[5].reshape(heads, 128),
    )
    actual = quantize_sage_bhsd(q, k, v)
    torch.cuda.synchronize()

    assert all(tensor.is_contiguous() for tensor in actual)
    assert torch.equal(actual[0].float(), expected[0].float())
    assert torch.equal(actual[2].float(), expected[2].float())
    assert torch.equal(actual[3], expected[3])
    assert torch.allclose(actual[1].float(), expected[1].float(), atol=4.0, rtol=0.0)
    assert torch.allclose(actual[4], expected[4], atol=1e-6, rtol=0.0)
    assert torch.equal(actual[5], expected[5])

    block_index = (
        torch.arange(3, device="cuda", dtype=torch.int32)
        .view(1, 1, 1, 3)
        .expand(1, heads, 2, 3)
        .contiguous()
    )
    expected_output = bsa_fp8_blk64_fwd(*expected, block_index, 2)
    actual_output = bsa_fp8_blk64_fwd(*actual, block_index, 2)
    output_diff = (actual_output.float() - expected_output.float()).abs()
    assert output_diff.max().item() <= 2e-4
    assert (
        output_diff.mean() / expected_output.float().abs().mean().clamp_min(1e-8)
    ).item() <= 1e-5
