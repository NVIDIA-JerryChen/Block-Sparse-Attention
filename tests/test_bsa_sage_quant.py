import pytest
import torch


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 is required")


def _q_reference(q, q_scale):
    batch, heads, seqlen, dim = q.shape
    scale_for_row = q_scale.repeat_interleave(32, dim=2)[..., :seqlen]
    return torch.round(q.float() / scale_for_row[..., None]).clamp(
        -127, 127
    ).to(torch.int8)


def _v_reference(v, v_scale):
    padded_len = ((v.shape[2] + 63) // 64) * 64
    padded = torch.nn.functional.pad(v.float(), (0, 0, 0, padded_len - v.shape[2]))
    permutation = torch.tensor(
        [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15],
        device=v.device,
    )
    grouped = padded.reshape(*padded.shape[:2], padded_len // 16, 16, 128)
    quantized = (
        grouped[:, :, :, permutation, :] / v_scale[:, :, None, None, :]
    ).to(torch.float8_e4m3fn)
    return quantized.permute(0, 1, 4, 2, 3).reshape(
        v.shape[0], v.shape[1], 128, padded_len
    )


def test_sage_sm120_quant_rejects_cpu():
    from bsa_sage_quant import quantize_sage_q_sm120

    q = torch.empty((1, 1, 64, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        quantize_sage_q_sm120(q)


@pytest.mark.parametrize("seqlen", [1, 32, 65, 128, 129])
def test_sage_sm120_q_quantization(seqlen):
    _require_sm120()
    from bsa_sage_quant import quantize_sage_q_sm120

    torch.manual_seed(900 + seqlen)
    q = torch.randn((2, 3, seqlen, 128), device="cuda", dtype=torch.bfloat16)
    q_int8, q_scale = quantize_sage_q_sm120(q)

    num_groups = ((seqlen + 127) // 128) * 4
    assert q_int8.shape == q.shape
    assert q_int8.dtype == torch.int8
    assert q_scale.shape == (2, 3, num_groups)
    assert q_scale.dtype == torch.float32

    padded = torch.nn.functional.pad(q.float(), (0, 0, 0, num_groups * 32 - seqlen))
    scale_ref = padded.reshape(2, 3, num_groups, 32, 128).abs().amax((-1, -2))
    scale_ref = scale_ref.clamp_min(1.0e-7) / 127.0
    torch.testing.assert_close(q_scale, scale_ref, rtol=0.0, atol=0.0)
    q_ref = _q_reference(q, q_scale)
    q_diff = (q_int8.to(torch.int16) - q_ref.to(torch.int16)).abs()
    assert q_diff.max().item() <= 1
    assert torch.count_nonzero(q_diff).item() / q_diff.numel() < 0.001


@pytest.mark.parametrize("seqlen", [1, 63, 64, 127, 257])
def test_sage_sm120_kv_quantization_and_layout(seqlen):
    _require_sm120()
    from bsa_sage_quant import quantize_sage_kv_sm120

    torch.manual_seed(1000 + seqlen)
    k = torch.randn((1, 2, seqlen, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k_int8, v_fp8, k_scale, v_scale = quantize_sage_kv_sm120(k, v)

    padded_len = ((seqlen + 63) // 64) * 64
    assert k_int8.shape == k.shape
    assert v_fp8.shape == (1, 2, 128, padded_len)
    assert k_scale.shape == (1, 2, padded_len // 64)
    assert v_scale.shape == (1, 2, 128)

    k_mean = k.float().mean(dim=2).to(torch.bfloat16).float()
    centered = k.float() - k_mean[:, :, None, :]
    centered_padded = torch.nn.functional.pad(
        centered, (0, 0, 0, padded_len - seqlen)
    )
    k_scale_ref = centered_padded.reshape(1, 2, -1, 64, 128).abs().amax((-1, -2))
    k_scale_ref = k_scale_ref.clamp_min(1.0e-7) / 127.0
    v_scale_ref = v.float().abs().amax(dim=2).clamp_min(1.0e-7) / 2.25
    torch.testing.assert_close(k_scale, k_scale_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(v_scale, v_scale_ref, rtol=0.0, atol=0.0)

    row_scale = k_scale.repeat_interleave(64, dim=2)[..., :seqlen]
    k_dequant = k_int8.float() * row_scale[..., None]
    assert (k_dequant - centered).abs().max().item() <= k_scale.max().item() * 0.51

    v_ref = _v_reference(v, v_scale)
    torch.testing.assert_close(v_fp8.float(), v_ref.float(), rtol=0.0, atol=0.125)


def test_sage_sm120_quant_zero_and_preallocated_buffers():
    _require_sm120()
    from bsa_sage_quant import (
        quantize_sage_qkv_sm120,
        sage_sm120_kv_quant_workspace_size,
    )

    q = torch.zeros((1, 2, 65, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((1, 2, 127, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    expected = quantize_sage_qkv_sm120(q, k, v)

    out = tuple(torch.empty_like(tensor) for tensor in expected)
    workspace = torch.empty(
        sage_sm120_kv_quant_workspace_size(k),
        device="cuda",
        dtype=torch.uint8,
    )
    actual = quantize_sage_qkv_sm120(q, k, v, out=out, workspace=workspace)
    assert all(actual_tensor is out_tensor for actual_tensor, out_tensor in zip(actual, out))
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor.float(), expected_tensor.float())
        assert torch.isfinite(actual_tensor.float()).all()
    assert torch.count_nonzero(actual[0]).item() == 0
    assert torch.count_nonzero(actual[1]).item() == 0
    assert torch.count_nonzero(actual[2]).item() == 0
