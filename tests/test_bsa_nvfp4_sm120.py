# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native SM120 blk64 BSA with NVFP4 Q/K and Sage FP8 P/V."""

import functools
import importlib

import pytest
import torch


def torch_fork_set_rng(seed=None):
    """Run the test under a forked RNG state, optionally seeded."""

    def decorator_(func):
        @functools.wraps(func)
        def wrapper_(*args, **kwargs):
            with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
                if seed is not None:
                    torch.manual_seed(seed)
                return func(*args, **kwargs)

        return wrapper_

    return decorator_


def block_sparse_mask(q2k_block_index, block_sparse_num, block_sizes, seqlen_q, seqlen_k, block_size, q2k_block_nums=None):
    """Expand BSA block metadata to a token-level additive attention mask."""
    batch, heads, num_q_blocks, _ = q2k_block_index.shape
    mask = torch.full((batch, heads, seqlen_q, seqlen_k), float("-inf"), dtype=torch.float32, device=q2k_block_index.device)
    for b in range(batch):
        for h in range(heads):
            for q_block in range(num_q_blocks):
                count = block_sparse_num if q2k_block_nums is None else int(q2k_block_nums[b, h, q_block])
                q_start = q_block * block_size
                q_end = min(q_start + block_size, seqlen_q)
                for slot in range(count):
                    kv_block = int(q2k_block_index[b, h, q_block, slot])
                    k_start = kv_block * block_size
                    k_end = min(k_start + int(block_sizes[kv_block]), seqlen_k)
                    mask[b, h, q_start:q_end, k_start:k_end] = 0.0
    return mask


def attention_reference(q, k, v, mask, softmax_scale=None):
    """FP32 block-sparse SDPA reference for tensors in BHSD layout."""
    softmax_scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    group_size = q.shape[1] // k.shape[1]
    k_expanded = k.float().repeat_interleave(group_size, dim=1)
    v_expanded = v.float().repeat_interleave(group_size, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float() * softmax_scale, k_expanded) + mask
    probabilities = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    return torch.einsum("bhqk,bhkd->bhqd", probabilities, v_expanded), torch.logsumexp(scores, dim=-1)


def _require_sm120():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native NVFP4 BSA tests require SM120")
    return importlib.import_module("bsa_attn_interface"), importlib.import_module("bsa_nvfp4_quant")


def _outlier_bhsd(shape, std=0.6, channel_gain=8.0, token_gain=6.0, bias=0.0, outlier_channels=4, outlier_token_rate=0.02):
    """BF16 BHSD data with LLM-style outliers: a few large channels per head, a few large tokens, and a per-channel bias.

    The defaults give |x| up to ~100 and softmax rows where ~20% of FP8 P values underflow
    and the largest probability is ~0.5; raise ``channel_gain`` for near one-hot rows.
    """
    batch, heads, seqlen, dim = shape
    x = torch.randn(shape, device="cuda") * std
    channels = torch.rand((heads, dim), device="cuda").argsort(-1)[:, :outlier_channels]
    gain = torch.ones((heads, dim), device="cuda").scatter_(1, channels, channel_gain)
    x = x * gain[None, :, None]
    x = torch.where(torch.rand((batch, heads, seqlen, 1), device="cuda") < outlier_token_rate, x * token_gain, x)
    return (x + bias * torch.randn((1, heads, 1, dim), device="cuda")).to(torch.bfloat16)


def _outlier_v8(shape):
    """FP8 V over most of the E4M3 range, with outlier channels and tokens saturating at +/-448."""
    return _outlier_bhsd(shape, std=4.0, channel_gain=8.0, token_gain=4.0).float().clamp(-448, 448).to(torch.float8_e4m3fn)


def _unique_indices(batch, heads, nq, nk, topk):
    """Per-row distinct K block indices in random order."""
    assert topk <= nk
    return torch.rand((batch, heads, nq, nk), device="cuda").argsort(-1)[..., :topk].to(torch.int32).contiguous()


def _online_nvfp4_reference(q, k, v8, v_scale, indices, counts, sizes, softmax_scale):
    """Model the kernel's blk64 online softmax, including the FP8 P cast, on dequantized Q/K."""
    out = torch.zeros(q.shape, device=q.device)
    lse = torch.full(q.shape[:-1], -torch.inf, device=q.device)
    v = v8.float()
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            for qb in range(indices.shape[2]):
                begin, end = qb * 64, min((qb + 1) * 64, q.shape[2])
                q_tile = q[b, h, begin:end]
                row_max = torch.full((end - begin, 1), -torch.inf, device=q.device)
                row_sum = torch.zeros_like(row_max)
                acc = torch.zeros_like(q_tile)
                for slot in reversed(range(int(counts[b, h, qb]))):
                    kb = int(indices[b, h, qb, slot])
                    valid = min(int(sizes[b, h, kb]), k.shape[2] - kb * 64)
                    if valid <= 0:
                        continue
                    scores = (q_tile @ k[b, h, kb * 64 : kb * 64 + valid].T) * softmax_scale
                    new_max = torch.maximum(row_max, scores.amax(dim=-1, keepdim=True))
                    rescale = torch.exp(row_max - new_max)
                    p = torch.exp(scores - new_max) * 256.0
                    acc = acc * rescale + p.to(torch.float8_e4m3fn).float() @ v[b, h, kb * 64 : kb * 64 + valid]
                    row_sum = row_sum * rescale + p.sum(dim=-1, keepdim=True)
                    row_max = new_max
                out[b, h, begin:end] = torch.where(row_sum > 0, acc / row_sum, 0) * v_scale[h]
                lse[b, h, begin:end] = (row_max + torch.log(row_sum) - torch.log(torch.tensor(256.0, device=q.device))).squeeze(-1)
    return out, lse


def _quantizers(quant):
    """The fused SM120 kernel and the PyTorch reference that defines its bit-level contract."""
    return (quant.quantize_nvfp4_bhsd, quant.quantize_nvfp4_bhsd_reference)


@torch_fork_set_rng(seed=4120)
def test_nvfp4_quantize_round_trip():
    _, quant = _require_sm120()
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    # Exactly representable rows round-trip; midpoints round to the even code.
    x = torch.cat((grid, -grid.flip(0)), dim=0).repeat(8).view(1, 1, 1, 128).to(torch.bfloat16)
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 16, device="cuda").view(1, 1, 1, 128).to(torch.bfloat16)
    for quantize in _quantizers(quant):
        packed, scale = quantize(x)
        assert packed.shape == (1, 1, 1, 64) and scale.shape == (1, 1, 64, 8)
        assert torch.equal(quant.nvfp4_dequantize_reference(packed, scale, 1), x.float())
        assert torch.count_nonzero(scale[:, :, 1:].float()) == 0
        rounded = quant.nvfp4_dequantize_reference(*quantize(ties), 1)
        assert torch.equal(rounded[0, 0, 0, :8], torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0], device="cuda"))


def _check_against_online_reference(batch, heads, seqlen_q, seqlen_k, topk, size_rank, use_counts, channel_gain=8.0, token_gain=6.0):
    interface, quant = _require_sm120()
    dim, softmax_scale = 128, 128**-0.5
    q = _outlier_bhsd((batch, heads, seqlen_q, dim), channel_gain=channel_gain, token_gain=token_gain)
    # A per-channel K bias exercises smooth=True; it shifts each query row's scores by a constant.
    k = _outlier_bhsd((batch, heads, seqlen_k, dim), channel_gain=channel_gain, token_gain=token_gain, bias=2.0)
    v8 = _outlier_v8((batch, heads, seqlen_k, dim))
    v_scale = torch.exp2(torch.empty((heads, dim), device="cuda").uniform_(-4, 2))
    nq, nk = (seqlen_q + 63) // 64, (seqlen_k + 63) // 64
    indices = _unique_indices(batch, heads, nq, nk, topk)
    counts = torch.randint(0, topk + 1, (batch, heads, nq), device="cuda", dtype=torch.int32) if use_counts else None
    sizes = None
    expanded_sizes = torch.full((batch, heads, nk), 64, device="cuda")
    if size_rank:
        sizes = torch.randint(0, 65, {1: (nk,), 2: (batch, nk), 3: (batch, heads, nk)}[size_rank], device="cuda", dtype=torch.int32)
        # block_sizes counts valid tokens, so the tail block cannot exceed what remains.
        sizes[..., -1] = sizes[..., -1].clamp(max=seqlen_k - (nk - 1) * 64)
        expanded_sizes = sizes.expand(batch, heads, -1) if size_rank == 1 else sizes[:, None].expand(-1, heads, -1) if size_rank == 2 else sizes

    q4, q_sf = quant.quantize_nvfp4_bhsd(q)
    k4, k_sf = quant.quantize_nvfp4_bhsd(k, smooth=True)
    actual, lse = interface._bsa_attn_fwd_sm120_nvfp4_blk64(
        q4, k4, v8, q_sf, k_sf, v_scale, indices, topk, softmax_scale, block_sizes=sizes, q2k_block_nums=counts
    )
    torch.cuda.synchronize()

    all_counts = counts if counts is not None else torch.full((batch, heads, nq), topk, device="cuda")
    expected, expected_lse = _online_nvfp4_reference(
        quant.nvfp4_dequantize_reference(q4, q_sf, seqlen_q),
        quant.nvfp4_dequantize_reference(k4, k_sf, seqlen_k),
        v8,
        v_scale,
        indices,
        all_counts,
        expanded_sizes,
        softmax_scale,
    )
    assert actual.dtype == torch.bfloat16 and actual.shape == q.shape
    # Scores are exact up to fp32 accumulation order, so LSE is tight; O
    # differs only where an FP8 P value lands on a rounding boundary.
    assert torch.equal(torch.isneginf(lse), torch.isneginf(expected_lse))
    finite = torch.isfinite(expected_lse)
    torch.testing.assert_close(lse[finite], expected_lse[finite], rtol=1e-5, atol=1e-5)
    assert (actual.float() - expected).abs().max().item() < 2e-2 * max(1.0, expected.abs().max().item())
    if counts is not None:
        empty_rows = counts.repeat_interleave(64, dim=-1)[..., :seqlen_q] == 0
        assert torch.count_nonzero(actual[empty_rows]) == 0


@pytest.mark.parametrize(
    "batch,heads,seqlen_q,seqlen_k,topk,size_rank,use_counts",
    (
        (1, 1, 64, 64, 1, 0, False),
        (2, 3, 147, 531, 3, 0, False),
        (2, 2, 200, 389, 3, 1, True),
        (2, 2, 129, 260, 3, 2, True),
        (1, 3, 64, 1000, 3, 3, True),
        (1, 8, 257, 2176, 24, 3, True),
    ),
)
@torch_fork_set_rng(seed=4121)
def test_sm120_nvfp4_blk64_matches_online_reference(batch, heads, seqlen_q, seqlen_k, topk, size_rank, use_counts):
    _check_against_online_reference(batch, heads, seqlen_q, seqlen_k, topk, size_rank, use_counts)


@pytest.mark.parametrize(("channel_gain", "token_gain"), ((16.0, 4.0), (24.0, 8.0)))
@torch_fork_set_rng(seed=4125)
def test_sm120_nvfp4_blk64_peaked_scores_match_online_reference(channel_gain, token_gain):
    """Near one-hot rows: the running max jumps between K blocks and most FP8 P values underflow to zero."""
    _check_against_online_reference(2, 4, 200, 1500, 16, 3, True, channel_gain=channel_gain, token_gain=token_gain)


@pytest.mark.parametrize(("seqlen_q", "seqlen_k"), ((128, 320), (147, 339)))
@torch_fork_set_rng(seed=4122)
def test_sm120_nvfp4_blk64_end_to_end_accuracy(seqlen_q, seqlen_k):
    """With Sage V quantization, NVFP4 QK stays within a bounded error of BF16 attention."""
    interface, quant = _require_sm120()
    batch, heads, dim, topk = 1, 2, 128, 3
    q = _outlier_bhsd((batch, heads, seqlen_q, dim))
    k = _outlier_bhsd((batch, heads, seqlen_k, dim), bias=2.0)
    v = _outlier_bhsd((batch, heads, seqlen_k, dim), std=1.0)
    nq, nk = (seqlen_q + 63) // 64, (seqlen_k + 63) // 64
    indices = torch.tensor([nk - 1, 0, 2], device="cuda", dtype=torch.int32).expand(batch, heads, nq, topk).contiguous()
    sizes = torch.full((nk,), 64, device="cuda", dtype=torch.int32)
    sizes[-1] = seqlen_k - (nk - 1) * 64
    expected, _ = attention_reference(q, k, v, block_sparse_mask(indices, topk, sizes, seqlen_q, seqlen_k, 64))

    q4, q_sf, k4, k_sf, v8, v_scale = quant.quantize_nvfp4_qk_sage_v_bhsd(q, k, v)
    actual, _ = interface._bsa_attn_fwd_sm120_nvfp4_blk64(q4, k4, v8, q_sf, k_sf, v_scale, indices, topk, dim**-0.5)
    # Baseline: the same NVFP4 Q/K, but BF16 V/P and no kernel.
    mask = block_sparse_mask(indices, topk, sizes, seqlen_q, seqlen_k, 64)
    dequantized = (quant.nvfp4_dequantize_reference(x4, sf, s).to(torch.bfloat16) for x4, sf, s in ((q4, q_sf, seqlen_q), (k4, k_sf, seqlen_k)))
    baseline, _ = attention_reference(*dequantized, v, mask)
    torch.cuda.synchronize()
    assert not actual.isnan().any()
    assert actual.dtype == torch.bfloat16 and actual.shape == q.shape and actual.is_contiguous()
    error = ((actual.float() - expected).abs().mean() / expected.abs().mean()).item()
    baseline_error = ((baseline.float() - expected).abs().mean() / expected.abs().mean()).item()
    # E2M1 has 8 magnitude levels (~10% element error vs ~2% for E4M3), and outliers enlarge the
    # scores so that error becomes a larger softmax error: Q/K quantization alone costs ~0.16-0.19
    # mean relative error (FP8 Sage: ~0.05); the kernel's FP8 V/P adds < 0.01 on top.
    assert baseline_error < 0.25
    assert error <= baseline_error + 0.02


def _random_nvfp4(batch, heads, seqlen, dim=128, low=-6, high=-1, group_gain=3, token_gain=2, outlier_token_rate=0.02):
    """Hand-built packed E2M1 codes and power-of-two E4M3 scales, bypassing the quantizer.

    Scales span 2^low..2^high; one 16-channel group per head is 2^group_gain larger and
    ~2% of tokens another 2^token_gain, so |x| reaches ~100 with outlier channels/tokens.
    """
    packed = torch.randint(0, 256, (batch, heads, seqlen, dim // 2), device="cuda", dtype=torch.uint8)
    groups = dim // 16
    exponent = torch.randint(low, high + 1, (batch, heads, seqlen, groups), device="cuda")
    outlier_group = torch.randint(0, groups, (heads, 1), device="cuda")
    exponent = exponent + group_gain * (torch.arange(groups, device="cuda") == outlier_group)[None, :, None]
    exponent = exponent + token_gain * (torch.rand((batch, heads, seqlen, 1), device="cuda") < outlier_token_rate)
    padded = (seqlen + 63) // 64 * 64
    scale = torch.zeros((batch, heads, padded, groups), device="cuda", dtype=torch.float8_e4m3fn)
    scale[:, :, :seqlen] = torch.exp2(exponent.float()).to(torch.float8_e4m3fn)
    return packed, scale


def _zero_nvfp4(batch, heads, seqlen, dim=128):
    packed = torch.zeros((batch, heads, seqlen, dim // 2), device="cuda", dtype=torch.uint8)
    scale = torch.zeros((batch, heads, (seqlen + 63) // 64 * 64, dim // 16), device="cuda", dtype=torch.float8_e4m3fn)
    return packed, scale


@pytest.mark.parametrize("seqlen", (1, 64, 147))
@torch_fork_set_rng(seed=4123)
def test_nvfp4_quantize_layout_and_padding(seqlen):
    """Scales are amax/6 in E4M3, codes pack low nibble first, and padded scale rows stay zero."""
    _, quant = _require_sm120()
    x = _outlier_bhsd((2, 3, seqlen, 128), std=1.0, channel_gain=16.0, token_gain=4.0, bias=2.0, outlier_token_rate=0.1)
    packed, scale = quant.quantize_nvfp4_bhsd(x)
    padded = (seqlen + 63) // 64 * 64
    assert packed.dtype == torch.uint8 and packed.shape == (2, 3, seqlen, 64) and packed.is_contiguous()
    assert scale.dtype == torch.float8_e4m3fn and scale.shape == (2, 3, padded, 8)
    assert torch.count_nonzero(scale[:, :, seqlen:].float()) == 0
    amax = x.float().view(2, 3, seqlen, 8, 16).abs().amax(dim=-1)
    assert torch.equal(scale[:, :, :seqlen].view(torch.uint8), (amax / 6).to(torch.float8_e4m3fn).view(torch.uint8))
    # The group maximum always maps to code +/-6 up to the E4M3 scale rounding.
    dequant = quant.nvfp4_dequantize_reference(packed, scale, seqlen)
    assert dequant.shape == x.shape
    rel = (dequant - x.float()).abs().amax() / x.float().abs().amax()
    assert rel.item() < 0.2

    # Swapping the two nibbles of every byte swaps adjacent head-dim elements.
    swapped = ((packed & 0xF) << 4) | (packed >> 4)
    expected = dequant.view(2, 3, seqlen, 64, 2).flip(-1).view_as(dequant)
    assert torch.equal(quant.nvfp4_dequantize_reference(swapped, scale, seqlen), expected)


@torch_fork_set_rng(seed=4124)
def test_nvfp4_quantize_smooth_zero_and_saturation():
    _, quant = _require_sm120()
    x = _outlier_bhsd((1, 2, 100, 128), std=1.0, channel_gain=16.0, bias=8.0)
    smooth = quant.quantize_nvfp4_bhsd_reference(x, smooth=True)
    centered = quant.quantize_nvfp4_bhsd_reference(x.float() - x.float().mean(dim=-2, keepdim=True))
    for a, b in zip(smooth, centered):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))

    for quantize in _quantizers(quant):
        # All-zero groups get a zero scale and decode to zero, never NaN; negative
        # values that round to zero do not keep a sign bit.
        tiny = torch.zeros((1, 1, 2, 128), device="cuda", dtype=torch.bfloat16)
        tiny[0, 0, 1, :16] = -1e-3
        tiny[0, 0, 1, 15] = 6.0
        packed, scale = quantize(tiny)
        assert torch.count_nonzero(packed[0, 0, 0]) == 0 and torch.count_nonzero(scale[0, 0, 0].float()) == 0
        assert torch.equal(packed[0, 0, 1, :7], torch.zeros(7, device="cuda", dtype=torch.uint8))
        assert torch.equal(quant.nvfp4_dequantize_reference(packed, scale, 2)[0, 0, 1, :16], torch.tensor([0.0] * 15 + [6.0], device="cuda"))
        assert not quant.nvfp4_dequantize_reference(packed, scale, 2).isnan().any()

        # Scales saturate at the E4M3 maximum; values beyond 6 * 448 clamp to the top code.
        huge = torch.full((1, 1, 1, 128), 1e5, device="cuda", dtype=torch.bfloat16)
        huge[..., 1::2] = -1e5
        packed, scale = quantize(huge)
        assert torch.equal(scale[0, 0, 0].float(), torch.full((8,), 448.0, device="cuda"))
        assert torch.equal(packed, torch.full_like(packed, 0xF7))


@pytest.mark.parametrize(("shape", "message"), (((128, 128), "rank-4"), ((1, 1, 64, 120), "multiple of 16")))
def test_nvfp4_quantize_reference_rejects_bad_shapes(shape, message):
    _, quant = _require_sm120()
    with pytest.raises(ValueError, match=message):
        quant.quantize_nvfp4_bhsd_reference(torch.zeros(shape, device="cuda"))


@pytest.mark.parametrize(
    ("make_input", "error", "message"),
    (
        (lambda: torch.zeros((128, 128), device="cuda", dtype=torch.bfloat16), ValueError, "rank-4"),
        (lambda: torch.zeros((1, 1, 64, 64), device="cuda", dtype=torch.bfloat16), ValueError, "head_dim == 128"),
        (lambda: torch.zeros((1, 1, 64, 128), device="cuda"), TypeError, "BF16"),
        (lambda: torch.zeros((1, 64, 2, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2), ValueError, "contiguous"),
        (lambda: torch.zeros((1, 1, 64, 128), dtype=torch.bfloat16), ValueError, "contiguous CUDA"),
    ),
)
def test_nvfp4_quantize_kernel_rejects_unsupported_inputs(make_input, error, message):
    _, quant = _require_sm120()
    with pytest.raises(error, match=message):
        quant.quantize_nvfp4_bhsd(make_input())


@pytest.mark.parametrize("seqlen", (1, 63, 64, 147, 6045))
@torch_fork_set_rng(seed=4126)
def test_nvfp4_quantize_kernel_matches_reference_bitwise(seqlen):
    """The fused kernel reproduces the reference bytes exactly, including E2M1 midpoint ties."""
    _, quant = _require_sm120()
    x = _outlier_bhsd((2, 3, seqlen, 128), std=1.0, channel_gain=16.0, token_gain=8.0, bias=2.0)
    # Per-group power-of-two gains put many values exactly on E2M1 midpoints after scaling.
    x = (x.float() * torch.exp2(torch.randint(-8, 8, (2, 3, seqlen, 8), device="cuda").float()).repeat_interleave(16, dim=-1)).to(torch.bfloat16)
    actual = quant.quantize_nvfp4_bhsd(x)
    expected = quant.quantize_nvfp4_bhsd_reference(x)
    for a, b in zip(actual, expected):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("seqlen", (1, 147, 6045))
@torch_fork_set_rng(seed=4127)
def test_nvfp4_quantize_kernel_smooth_matches_reference(seqlen):
    """smooth=True sums the mean in a different order, so only boundary codes may move, by one step."""
    _, quant = _require_sm120()
    x = _outlier_bhsd((2, 3, seqlen, 128), std=1.0, channel_gain=16.0, token_gain=8.0, bias=8.0)
    packed, scale = quant.quantize_nvfp4_bhsd(x, smooth=True)
    ref_packed, ref_scale = quant.quantize_nvfp4_bhsd_reference(x, smooth=True)
    scale_mismatch = scale.view(torch.uint8) != ref_scale.view(torch.uint8)
    assert scale_mismatch.float().mean().item() < 1e-4
    # An E4M3 scale can move by one code; compare decoded values to absorb that.
    actual = quant.nvfp4_dequantize_reference(packed, scale, seqlen)
    expected = quant.nvfp4_dequantize_reference(ref_packed, ref_scale, seqlen)
    mismatch = actual != expected
    assert mismatch.float().mean().item() < 1e-4
    group_scale = torch.maximum(scale.float(), ref_scale.float())[:, :, :seqlen].repeat_interleave(16, dim=-1)
    # One E2M1 step is at most 2 * scale (4 -> 6), plus the scale's own one-code wobble.
    assert ((actual - expected).abs() <= 2.25 * group_scale + 1e-30)[mismatch].all()


@torch_fork_set_rng(seed=4128)
def test_nvfp4_quantize_kernel_is_deterministic_and_graph_capturable():
    _, quant = _require_sm120()
    x = _outlier_bhsd((1, 4, 3000, 128), bias=2.0)
    eager = [quant.quantize_nvfp4_bhsd(x, smooth=smooth) for smooth in (False, True)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for smooth in (False, True):
            quant.quantize_nvfp4_bhsd(x, smooth=smooth)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = [quant.quantize_nvfp4_bhsd(x, smooth=smooth) for smooth in (False, True)]
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for eager_pair, captured_pair in zip(eager, captured):
            for a, b in zip(eager_pair, captured_pair):
                assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize(
    "shape_q,shape_k",
    (
        ((1, 1, 1, 128), (1, 1, 1, 128)),
        ((1, 4, 6045, 128), (1, 4, 6045, 128)),
        ((2, 3, 147, 128), (2, 3, 531, 128)),
        ((3, 2, 200, 128), (3, 2, 70, 128)),
        ((2, 2, 64, 128), (2, 2, 40000, 128)),
    ),
)
@torch_fork_set_rng(seed=4131)
def test_nvfp4_qk_sage_v_fused_quantizer(shape_q, shape_k):
    """Fused NVFP4 Q/K + Sage FP8 V: Q matches the reference bitwise, K up to its mean's summation
    order, and V/v_scale are this repo's quantize_sage_bhsd output bit for bit."""
    _, quant = _require_sm120()
    from bsa_fp8_quant import quantize_sage_bhsd

    q = _outlier_bhsd(shape_q, std=1.0, channel_gain=16.0, token_gain=8.0)
    k = _outlier_bhsd(shape_k, std=1.0, channel_gain=16.0, token_gain=8.0, bias=8.0)
    v = _outlier_bhsd(shape_k, std=1.0, channel_gain=16.0)
    # Power-of-two group gains put many scaled values exactly on E2M1 midpoints.
    q = (q.float() * torch.exp2(torch.randint(-8, 8, shape_q[:3] + (8,), device="cuda").float()).repeat_interleave(16, dim=-1)).to(torch.bfloat16)
    q4, q_sf, k4, k_sf, v8, v_scale = quant.quantize_nvfp4_qk_sage_v_bhsd(q, k, v)
    ref_q4, ref_q_sf = quant.quantize_nvfp4_bhsd_reference(q)
    assert torch.equal(q4, ref_q4) and torch.equal(q_sf.view(torch.uint8), ref_q_sf.view(torch.uint8))

    ref_k4, ref_k_sf = quant.quantize_nvfp4_bhsd_reference(k, smooth=True)
    assert k4.shape == ref_k4.shape and k_sf.shape == ref_k_sf.shape
    assert (k_sf.view(torch.uint8) != ref_k_sf.view(torch.uint8)).float().mean().item() < 1e-4
    actual_k = quant.nvfp4_dequantize_reference(k4, k_sf, shape_k[2])
    expected_k = quant.nvfp4_dequantize_reference(ref_k4, ref_k_sf, shape_k[2])
    mismatch = actual_k != expected_k
    assert mismatch.float().mean().item() < 1e-4
    group_scale = torch.maximum(k_sf.float(), ref_k_sf.float())[:, :, : shape_k[2]].repeat_interleave(16, dim=-1)
    assert ((actual_k - expected_k).abs() <= 2.25 * group_scale + 1e-30)[mismatch].all()

    _, _, ref_v8, _, _, ref_v_scale = quantize_sage_bhsd(q, k, v)
    assert torch.equal(v8.view(torch.uint8), ref_v8.view(torch.uint8))
    assert torch.equal(v_scale.view(torch.int32), ref_v_scale.view(torch.int32))


@torch_fork_set_rng(seed=4132)
def test_nvfp4_qk_sage_v_fused_quantizer_matches_cutedsl_quantizer_and_replays():
    """The fused path agrees with the CuTe DSL single-tensor quantizer on Q and is CUDA-graph safe."""
    _, quant = _require_sm120()
    q, k, v = (_outlier_bhsd((2, 4, 3000, 128)) for _ in range(3))
    eager = quant.quantize_nvfp4_qk_sage_v_bhsd(q, k, v)
    cute_q4, cute_q_sf = quant.quantize_nvfp4_bhsd(q)
    assert torch.equal(eager[0], cute_q4) and torch.equal(eager[1].view(torch.uint8), cute_q_sf.view(torch.uint8))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        quant.quantize_nvfp4_qk_sage_v_bhsd(q, k, v)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = quant.quantize_nvfp4_qk_sage_v_bhsd(q, k, v)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        for a, b in zip(captured, eager):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize(
    ("make_inputs", "error", "message"),
    (
        (lambda: (torch.zeros((1, 1, 64, 128), device="cuda"),) * 3, TypeError, "BF16"),
        (lambda: (torch.zeros((1, 1, 64, 64), device="cuda", dtype=torch.bfloat16),) * 3, ValueError, "D=128"),
        (lambda: (torch.zeros((1, 1, 64, 128), device="cuda", dtype=torch.bfloat16), torch.zeros((1, 1, 64, 128), device="cuda", dtype=torch.bfloat16), torch.zeros((1, 1, 32, 128), device="cuda", dtype=torch.bfloat16)), ValueError, "same shape"),
        (lambda: (torch.zeros((1, 64, 2, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2),) * 3, ValueError, "contiguous"),
    ),
)
def test_nvfp4_qk_sage_v_fused_quantizer_rejects_unsupported_inputs(make_inputs, error, message):
    _, quant = _require_sm120()
    with pytest.raises(error, match=message):
        quant.quantize_nvfp4_qk_sage_v_bhsd(*make_inputs())


@pytest.mark.parametrize("size_rank", (1, 2, 3))
@torch_fork_set_rng(seed=6464)
def test_sm120_nvfp4_blk64_scales_counts_and_partial_blocks(size_rank):
    """Hand-built codes/scales with empty, partial and zero-size blocks, independent of the quantizer."""
    interface, quant = _require_sm120()
    batch, heads, sq, sk, dim = 2, 2, 147, 275, 128
    q4, q_sf = _random_nvfp4(batch, heads, sq)
    k4, k_sf = _random_nvfp4(batch, heads, sk)
    v8 = _outlier_v8((batch, heads, sk, dim))
    v_scale = torch.exp2(torch.empty((heads, dim), device="cuda").uniform_(-4, 2))
    # Distinct blocks per row; [4, 0, 2] always hits the 19-token tail and the 37/0-token block.
    indices = torch.tensor([4, 0, 2], device="cuda", dtype=torch.int32).expand(batch, heads, 3, 3).contiguous()
    counts = torch.tensor([3, 0, 1, 2, 2, 3, 0, 1, 3, 3, 2, 0], device="cuda", dtype=torch.int32).view(batch, heads, 3)
    sizes = torch.tensor([64, 64, 37, 64, 19], device="cuda", dtype=torch.int32)
    if size_rank >= 2:
        sizes = sizes.repeat(batch, 1)
        sizes[1, 0] = 45
    if size_rank == 3:
        sizes = sizes[:, None].repeat(1, heads, 1)
        sizes[:, 1, 2] = 0
    expanded_sizes = sizes.expand(batch, heads, -1) if size_rank == 1 else sizes[:, None].expand(-1, heads, -1) if size_rank == 2 else sizes
    softmax_scale = 0.0625
    expected, expected_lse = _online_nvfp4_reference(
        quant.nvfp4_dequantize_reference(q4, q_sf, sq),
        quant.nvfp4_dequantize_reference(k4, k_sf, sk),
        v8,
        v_scale,
        indices,
        counts,
        expanded_sizes,
        softmax_scale,
    )
    actual, lse = interface._bsa_attn_fwd_sm120_nvfp4_blk64(
        q4, k4, v8, q_sf, k_sf, v_scale, indices, 3, softmax_scale, block_sizes=sizes, q2k_block_nums=counts
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), expected, rtol=5e-3, atol=2e-3)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=2e-5)
    empty_rows = counts.repeat_interleave(64, dim=-1)[..., :sq] == 0
    assert torch.count_nonzero(actual[empty_rows]) == 0
    assert torch.isneginf(lse[empty_rows]).all()


@torch_fork_set_rng(seed=6465)
def test_sm120_nvfp4_blk64_full_counts_match_fixed_topk():
    """q2k_block_nums == topk everywhere must be bitwise identical to the fixed-topk launch."""
    interface, _ = _require_sm120()
    batch, heads, sq, sk, topk = 2, 3, 200, 451, 4
    q4, q_sf = _random_nvfp4(batch, heads, sq)
    k4, k_sf = _random_nvfp4(batch, heads, sk)
    v8 = _outlier_v8((batch, heads, sk, 128))
    v_scale = torch.exp2(torch.empty((heads, 128), device="cuda").uniform_(-4, 2))
    nq, nk = (sq + 63) // 64, (sk + 63) // 64
    indices = _unique_indices(batch, heads, nq, nk, topk)
    counts = torch.full((batch, heads, nq), topk, device="cuda", dtype=torch.int32)
    fixed = interface._bsa_attn_fwd_sm120_nvfp4_blk64(q4, k4, v8, q_sf, k_sf, v_scale, indices, topk, 128**-0.5)
    variable = interface._bsa_attn_fwd_sm120_nvfp4_blk64(q4, k4, v8, q_sf, k_sf, v_scale, indices, topk, 128**-0.5, q2k_block_nums=counts)
    torch.cuda.synchronize()
    for a, b in zip(fixed, variable):
        assert torch.equal(a, b)


@torch_fork_set_rng(seed=6466)
def test_sm120_nvfp4_blk64_compile_cache_is_shape_dynamic(monkeypatch):
    """Each metadata mode compiles once; new batch/head/sequence shapes reuse that entry."""
    interface, _ = _require_sm120()
    monkeypatch.setattr(interface._bsa_attn_fwd_sm120_nvfp4_blk64, "compile_cache", {})
    original_compile = interface.cute.compile
    compiled = []

    def record_compile(*args, **kwargs):
        compiled.append(type(args[0]).__name__)
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(interface.cute, "compile", record_compile)

    def launch(batch, heads, sq, sk, mode):
        q4, q_sf = _random_nvfp4(batch, heads, sq)
        k4, k_sf = _random_nvfp4(batch, heads, sk)
        v8 = _outlier_v8((batch, heads, sk, 128))
        v_scale = torch.ones((heads, 128), device="cuda")
        nq, nk = (sq + 63) // 64, (sk + 63) // 64
        indices = torch.zeros((batch, heads, nq, 1), device="cuda", dtype=torch.int32)
        kwargs = {}
        if mode == "counts":
            kwargs["q2k_block_nums"] = torch.ones((batch, heads, nq), device="cuda", dtype=torch.int32)
        elif mode:
            shape = {1: (nk,), 2: (batch, nk), 3: (batch, heads, nk)}[mode]
            kwargs["block_sizes"] = torch.full(shape, 64, device="cuda", dtype=torch.int32)
            kwargs["block_sizes"][..., -1] = sk - (nk - 1) * 64
        interface._bsa_attn_fwd_sm120_nvfp4_blk64(q4, k4, v8, q_sf, k_sf, v_scale, indices, 1, 128**-0.5, **kwargs)

    modes = (None, "counts", 1, 2, 3)
    for mode in modes:
        launch(1, 1, 64, 64, mode)
    assert compiled == ["BlockSparseAttnForwardNvfp4Sm120Blk64"] * len(modes)
    assert len(interface._bsa_attn_fwd_sm120_nvfp4_blk64.compile_cache) == len(modes)
    for mode in modes:
        launch(2, 3, 147, 389, mode)
    torch.cuda.synchronize()
    assert len(compiled) == len(modes)


@torch_fork_set_rng(seed=6467)
def test_sm120_nvfp4_blk64_graph_replay_uses_runtime_metadata(monkeypatch):
    interface, quant = _require_sm120()
    q = _outlier_bhsd((1, 1, 128, 128))
    k = _outlier_bhsd((1, 1, 128, 128), bias=2.0)
    q4, q_sf = quant.quantize_nvfp4_bhsd(q)
    k4, k_sf = quant.quantize_nvfp4_bhsd(k, smooth=True)
    v8 = _outlier_v8(k.shape)
    v_scale = torch.ones((1, 128), device="cuda")
    indices = torch.zeros((1, 1, 2, 1), device="cuda", dtype=torch.int32)

    def run():
        return interface._bsa_attn_fwd_sm120_nvfp4_blk64(q4, k4, v8, q_sf, k_sf, v_scale, indices, 1, 128**-0.5)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    def forbid_compile(*args, **kwargs):
        raise AssertionError("warmed native NVFP4 must reuse its compile cache")

    monkeypatch.setattr(interface.cute, "compile", forbid_compile)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual, _ = run()
    graph.replay()
    first = actual.clone()
    indices.fill_(1)
    graph.replay()
    expected, _ = run()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.equal(actual, first)


@torch_fork_set_rng(seed=6468)
def test_sm120_nvfp4_blk64_producer_replay_with_mixed_counts():
    """Zero Q makes softmax uniform, so O is the mean of the selected V tokens and LSE = log(tokens)."""
    interface, _ = _require_sm120()
    sq, sk, heads, topk = 512, 40 * 64 + 19, 2, 17
    q4, q_sf = _zero_nvfp4(1, heads, sq)
    k4, k_sf = _random_nvfp4(1, heads, sk)
    v8 = _outlier_v8((1, heads, sk, 128))
    vs = torch.exp2(torch.empty((heads, 128), device="cuda").uniform_(-4, 2))
    nq, nk = sq // 64, (sk + 63) // 64
    indices = _unique_indices(1, heads, nq, nk, topk)
    counts = torch.tensor([0, 1, 2, 17, 17, 2, 1, 0, 5, 0, 17, 3, 9, 16, 0, 1], device="cuda", dtype=torch.int32).view(1, heads, nq)

    def run():
        return interface._bsa_attn_fwd_sm120_nvfp4_blk64(q4, k4, v8, q_sf, k_sf, vs, indices, topk, 128**-0.5, q2k_block_nums=counts)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual, lse = run()

    for _ in range(4):
        counts.copy_(counts.flip(-1))
        for _ in range(5):
            graph.replay()
        expected = torch.zeros_like(actual, dtype=torch.float32)
        expected_lse = torch.full_like(lse, -torch.inf)
        for head in range(heads):
            for qb in range(nq):
                count = int(counts[0, head, qb])
                if count:
                    blocks = indices[0, head, qb, :count].long()
                    tokens = (blocks[:, None] * 64 + torch.arange(64, device="cuda")).flatten()
                    tokens = tokens[tokens < sk]
                    expected[0, head, qb * 64 : (qb + 1) * 64] = v8[0, head].float()[tokens].mean(dim=0) * vs[head]
                    expected_lse[0, head, qb * 64 : (qb + 1) * 64] = torch.log(torch.tensor(tokens.numel(), device="cuda", dtype=torch.float32))
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.float(), expected, atol=1e-4, rtol=5e-3)
        torch.testing.assert_close(lse, expected_lse, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "bad",
    ("head_dim", "unpadded_q_scale", "k_scale_dtype", "v_scale_shape", "index_dtype", "counts_shape", "sizes_shape"),
)
def test_sm120_nvfp4_blk64_rejects_malformed_inputs(bad):
    interface, _ = _require_sm120()
    q4, q_sf = _zero_nvfp4(1, 2, 100)
    k4, k_sf = _zero_nvfp4(1, 2, 130)
    v8 = torch.zeros((1, 2, 130, 128), device="cuda").to(torch.float8_e4m3fn)
    v_scale = torch.ones((2, 128), device="cuda")
    indices = torch.zeros((1, 2, 2, 1), device="cuda", dtype=torch.int32)
    kwargs = {}
    if bad == "head_dim":
        q4, k4 = q4[..., :32], k4[..., :32]
    elif bad == "unpadded_q_scale":
        q_sf = q_sf[:, :, :100]
    elif bad == "k_scale_dtype":
        k_sf = k_sf.float()
    elif bad == "v_scale_shape":
        v_scale = v_scale[:1]
    elif bad == "index_dtype":
        indices = indices.long()
    elif bad == "counts_shape":
        kwargs["q2k_block_nums"] = torch.ones((1, 2, 3), device="cuda", dtype=torch.int32)
    elif bad == "sizes_shape":
        kwargs["block_sizes"] = torch.full((2,), 64, device="cuda", dtype=torch.int32)
    with pytest.raises(AssertionError):
        interface._bsa_attn_fwd_sm120_nvfp4_blk64(q4, k4, v8, q_sf, k_sf, v_scale, indices, 1, 128**-0.5, **kwargs)
