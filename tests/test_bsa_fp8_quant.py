import pytest
import torch


@pytest.mark.parametrize(
    "batch,heads,sq,sk",
    [(1, 1, 1, 1), (2, 3, 65, 79), (3, 5, 257, 511),
     (4, 6, 2048, 2051), (1, 4, 129, 16401),
     (1, 3, 37, 127), (1, 3, 129, 128), (1, 3, 33, 129),
     (3, 2, 127, 8193), (2, 3, 9, 32801)],
)
@pytest.mark.parametrize("distribution", ["normal", "scaled", "zero"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quantize_sage_bhsd_numerical_recipe(batch, heads, sq, sk, distribution):
    """Cover both reductions, cross-batch V scales, tails and BF16 rounding."""
    from bsa_fp8_quant import quantize_sage_bhsd

    torch.manual_seed(20261008)
    q = torch.randn(batch, heads, sq, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, heads, sk, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    if distribution == "zero":
        q.zero_()
        k.zero_()
        v.zero_()
    elif distribution == "scaled":
        channels = torch.logspace(-4, 1, 128, device="cuda")[None, None, None, :]
        q = (q.float() * channels).bfloat16()
        k = (k.float() * channels + 2.0).bfloat16()
        batches = torch.arange(1, batch + 1, device="cuda")[:, None, None, None]
        v = (v.float() * channels * batches).bfloat16()

    q8, k8, v8, qs, ks, vs = quantize_sage_bhsd(q, k, v)
    ref_qs = q.float().abs().amax(dim=-1).clamp_min(1e-3) / 448.0
    ref_q8 = (q * ref_qs.bfloat16().reciprocal()[..., None]).to(torch.float8_e4m3fn)
    ref_vs = v.float().abs().amax(dim=(0, 2)).clamp_min(1e-3) / 448.0
    ref_v_scaled = v.float() * ref_vs.reciprocal()[None, :, None, :]
    ref_v8 = ref_v_scaled.to(torch.float8_e4m3fn)
    mean = k.float().mean(dim=2, keepdim=True)
    padded_k = torch.nn.functional.pad(k.float(), (0, 0, 0, (-sk) % 16))
    centered = (padded_k - mean).reshape(batch, heads, -1, 16, 128)
    ref_ks = centered.abs().amax(dim=(-1, -2)).clamp_min(1e-3) / 448.0
    ref_k8 = (centered / ref_ks[..., None, None]).to(torch.float8_e4m3fn)
    ref_k8 = ref_k8.reshape(batch, heads, -1, 128)[:, :, :sk]

    assert torch.equal(q8.float(), ref_q8.float())
    v_different = v8.float() != ref_v8.float()
    if v_different.any():
        # CUDA and Triton FP32 reciprocals can differ by one ULP. Only accept
        # an adjacent FP8 bin at such a midpoint; scales must still match exactly.
        midpoint = ref_v_scaled[v_different]
        lower = torch.nextafter(midpoint, torch.full_like(midpoint, -float("inf")))
        upper = torch.nextafter(midpoint, torch.full_like(midpoint, float("inf")))
        actual = v8.float()[v_different]
        assert ((actual == lower.to(v8.dtype).float())
                | (actual == upper.to(v8.dtype).float())).all()
    assert torch.equal(qs, ref_qs)
    assert torch.equal(vs, ref_vs)
    torch.testing.assert_close(ks, ref_ks, atol=2e-8, rtol=2e-6)
    # FP32 summation order can move exact FP8 midpoint ties by one bin.
    different = (k8.float() != ref_k8.float()).float().mean().item()
    assert different < 1e-4, different
    torch.testing.assert_close(k8.float(), ref_k8.float(), atol=32.0, rtol=0.0)
    assert all(t.is_contiguous() for t in (q8, k8, v8, qs, ks, vs))


def test_quantize_sage_bhsd_rejects_cpu_inputs():
    from bsa_fp8_quant import quantize_sage_bhsd

    q = torch.empty((1, 4, 64, 128), dtype=torch.bfloat16)
    k = torch.empty((1, 4, 64, 128), dtype=torch.bfloat16)
    v = torch.empty_like(k)
    with pytest.raises(ValueError, match="CUDA"):
        quantize_sage_bhsd(q, k, v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch,sk", [(2, 257), (1, 127)])
def test_quantize_sage_bhsd_graph_replay_and_stream(batch, sk):
    from bsa_fp8_quant import quantize_sage_bhsd

    q = torch.randn(batch, 3, 65, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, 3, sk, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        quantize_sage_bhsd(q, k, v)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = quantize_sage_bhsd(q, k, v)
    q.mul_(1.5)
    k[..., :128, :].add_(0.25)
    v[-1].mul_(2.0)
    graph.replay()
    expected = quantize_sage_bhsd(q, k, v)
    for actual, reference in zip(captured, expected):
        torch.testing.assert_close(actual.float(), reference.float(), atol=0, rtol=0)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "batch,heads,sq,sk,offset",
    [(1, 2, 97, 117, 0), (4, 6, 256, 1024, 0),
     (3, 5, 513, 1025, 0), (2, 3, 31, 16403, 0),
     (2, 3, 33, 129, 1)],
)
def test_quantize_sage_native_matches_triton_after_input_changes(batch, heads, sq, sk, offset):
    import bsa_fp8_quant as quant

    def make(seq):
        storage = torch.randn(batch * heads * seq * 128 + offset,
                              device="cuda", dtype=torch.bfloat16)
        return storage[offset:].view(batch, heads, seq, 128)

    q, k, v = make(sq), make(sk), make(sk)
    first = quant.quantize_sage_bhsd(q, k, v)
    key = (q.device, q.shape, k.shape, tuple(t.data_ptr() % 16 == 0 for t in (q, k, v)))
    if quant._native_quant_plans.get(key) is None:
        pytest.skip("Optional native host runtime unavailable")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        q.mul_(1.75)
        k[..., :min(sk, 97), :].add_(0.375)
        v[-1].mul_(2.0)
        actual = quant.quantize_sage_bhsd(q, k, v)
        expected = quant._quantize_sage_bhsd_triton(q, k, v)
        # A second call must allocate independent outputs, not reuse cached data.
        again = quant.quantize_sage_bhsd(q, k, v)
    torch.cuda.current_stream().wait_stream(stream)
    assert isinstance(actual, tuple) and len(actual) == 6
    for a, e, before, repeated in zip(actual, expected, first, again):
        assert torch.equal(a.view(torch.uint8), e.view(torch.uint8))
        assert torch.equal(a.view(torch.uint8), repeated.view(torch.uint8))
        assert a.data_ptr() != before.data_ptr() and a.data_ptr() != repeated.data_ptr()
    assert not torch.equal(actual[5], first[5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quantize_sage_native_validates_warmed_inputs_and_fallback(monkeypatch):
    import bsa_fp8_quant as quant

    q = torch.randn(2, 3, 65, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 3, 129, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    quant.quantize_sage_bhsd(q, k, v)
    actual = quant.quantize_sage_bhsd(q, k, v)
    with pytest.raises(TypeError, match="BF16"):
        quant.quantize_sage_bhsd(q.float(), k, v)
    with pytest.raises(ValueError, match="shape"):
        quant.quantize_sage_bhsd(q, k, v[:, :, :-1].contiguous())
    # Same shape and alignment, but with a non-contiguous view of fresh storage.
    strided = torch.empty(2, 3, 65, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
    with pytest.raises(ValueError, match="contiguous"):
        quant.quantize_sage_bhsd(strided, k, v)
    monkeypatch.setattr(quant, "_native_quant_enabled", False)
    fallback = quant.quantize_sage_bhsd(q, k, v)
    for a, e in zip(actual, fallback):
        assert torch.equal(a.view(torch.uint8), e.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quantize_sage_native_honors_triton_launch_hooks():
    import bsa_fp8_quant as quant

    if quant._triton_runtime is None:
        pytest.skip("Triton runtime knobs unavailable")
    q = torch.randn(2, 3, 65, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 3, 129, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    expected = quant.quantize_sage_bhsd(q, k, v)
    calls = []
    previous = quant._triton_runtime.launch_enter_hook
    quant._triton_runtime.launch_enter_hook = lambda metadata: calls.append(metadata)
    try:
        actual = quant.quantize_sage_bhsd(q, k, v)
    finally:
        quant._triton_runtime.launch_enter_hook = previous
    assert len(calls) == 3
    for a, e in zip(actual, expected):
        assert torch.equal(a.view(torch.uint8), e.view(torch.uint8))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices are required")
def test_quantize_sage_native_uses_input_device_and_current_stream():
    import bsa_fp8_quant as quant

    original_device = torch.cuda.current_device()
    other = 1 if original_device == 0 else 0
    q = torch.randn(2, 3, 65, 128, device=f"cuda:{other}", dtype=torch.bfloat16)
    k = torch.randn(2, 3, 129, 128, device=q.device, dtype=q.dtype)
    v = torch.randn_like(k)
    quant.quantize_sage_bhsd(q, k, v)
    assert torch.cuda.current_device() == original_device
    actual = quant.quantize_sage_bhsd(q, k, v)
    expected = quant._quantize_sage_bhsd_triton(q, k, v)
    for a, e in zip(actual, expected):
        assert a.device == q.device
        assert torch.equal(a.view(torch.uint8), e.view(torch.uint8))
    with pytest.raises(ValueError, match="same CUDA device"):
        quant.quantize_sage_bhsd(q, k, v.to(f"cuda:{original_device}"))


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
