"""Native SM100/SM103 Sage FP8 blk128, against the half-precision matrix."""

import math

import pytest
import torch

from block_sparse_attention import bsa_attn_fwd, bsa_fp8_fwd, quantize_sage_bhsd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Native FP8 blk128 requires SM100/SM103",
)


def make_inputs(batch, heads, sq, sk, *, gain=1.0):
    return tuple(
        (torch.randn(batch, heads, seq, 128, device="cuda") * gain).bfloat16()
        for seq in (sq, sk, sk)
    )


def make_metadata(batch, heads, sq, sk, topk, *, variable=False, sizes_mode=0):
    """Unsorted, distinct sparse indices; inactive slots deliberately invalid."""
    nq, nk = math.ceil(sq / 128), math.ceil(sk / 128)
    index = (
        torch.stack([torch.randperm(nk)[:topk] for _ in range(batch * heads * nq)])
        .reshape(batch, heads, nq, topk)
        .to(device="cuda", dtype=torch.int32)
    )
    nums = None
    if variable:
        nums = (torch.arange(batch * heads * nq) % (topk + 1)).reshape(batch, heads, nq)
        nums = nums.to(device="cuda", dtype=torch.int32)
        inactive = torch.arange(topk, device="cuda") >= nums[..., None]
        index.masked_fill_(inactive, 0x3FFFFFFF)
    sizes = None
    if sizes_mode:
        shape = {1: (nk,), 2: (batch, nk), 3: (batch, heads, nk)}[sizes_mode]
        sizes = torch.randint(1, 129, shape, device="cuda", dtype=torch.int32)
        sizes[..., -1].clamp_(max=sk - (nk - 1) * 128)
    return index, nums, sizes


def sparse_reference(q, k, v, index, topk, nums=None, sizes=None, scale=None):
    """FP32 torch matmul/softmax over exactly the selected valid KV tokens."""
    batch, heads, sq, dim = q.shape
    sk = k.shape[2]
    scale = dim**-0.5 if scale is None else scale
    out = torch.empty_like(q, dtype=torch.float32)
    offsets = torch.arange(128, device=q.device)
    for qb in range(index.shape[2]):
        blocks = index[:, :, qb].long()
        count = topk if nums is None else nums[:, :, qb, None]
        live = torch.arange(blocks.shape[-1], device=q.device) < count
        # Empty and unused slots must never address the underlying tensors.
        blocks = blocks.masked_fill(~live, 0)
        tokens = blocks[..., None] * 128 + offsets
        valid = live[..., None] & (tokens < sk)
        if sizes is not None:
            if sizes.ndim == 1:
                extent = sizes[blocks]
            elif sizes.ndim == 2:
                extent = sizes[torch.arange(batch, device=q.device)[:, None, None], blocks]
            else:
                extent = sizes.gather(2, blocks)
            valid = valid & (offsets < extent[..., None])
        tokens = tokens.clamp(0, sk - 1).flatten(-2)
        gather = tokens[..., None].expand(-1, -1, -1, dim)
        kt = k.gather(2, gather).float()
        vt = v.gather(2, gather).float()
        begin, end = qb * 128, min((qb + 1) * 128, sq)
        scores = (q[:, :, begin:end].float() @ kt.transpose(-1, -2)) * scale
        scores.masked_fill_(~valid.flatten(-2).unsqueeze(-2), -torch.inf)
        probs = torch.softmax(scores, dim=-1).nan_to_num()
        out[:, :, begin:end] = probs @ vt
    return out


def dequantize(quantized):
    q, k, v, qs, ks, vs = quantized
    batch, heads, sq, dim = q.shape
    sk = k.shape[2]
    return (
        (q.float() * qs.reshape(batch, heads, sq, 1)).bfloat16(),
        (
            k.float() * ks.reshape(batch, heads, -1).repeat_interleave(16, dim=-1)[..., :sk, None]
        ).bfloat16(),
        (v.float() * vs.reshape(1, heads, 1, dim)).bfloat16(),
    )


def assert_accuracy(out, ref, *, relative=0.029, absolute=0.15):
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    error = (out.float() - ref.float()).abs()
    rel = (error.mean() / ref.abs().mean().clamp_min(1e-8)).item()
    maximum = error.max().item()
    assert maximum < absolute, f"max absolute error {maximum} >= {absolute}"
    assert rel < relative, f"mean relative error {rel} >= {relative}"
    return rel, maximum


# Same sequence lengths, batch sizes and MHA heads as test_flash_fwd_sm100,
# plus its no_block_sizes cases. FP8 keeps the existing D=128/MHA contract.
_MATRIX = [
    (sq, sk, True)
    for sq, sk in [
        (64, 512),
        (64, 1024),
        (128, 512),
        (128, 1024),
        (256, 512),
        (256, 1024),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]
] + [(128, 512, False), (256, 1024, False), (1024, 1024, False)]


@pytest.mark.parametrize("sq,sk,has_sizes", _MATRIX)
@pytest.mark.parametrize("variable", [False, True])
def test_fp8_blk128_half_precision_matrix(sq, sk, has_sizes, variable):
    torch.manual_seed(1200 + sq + sk + int(variable))
    batch, heads = ((4 if sk <= 2048 else 2), 6) if has_sizes else (2, 4)
    q, k, v = make_inputs(batch, heads, sq, sk)
    topk = min(sk // 128, 8)
    index, nums, sizes = make_metadata(
        batch, heads, sq, sk, topk, variable=variable, sizes_mode=int(has_sizes)
    )
    quantized = quantize_sage_bhsd(q, k, v)
    out = bsa_fp8_fwd(
        *quantized, index, topk, sparse_block_size=128, block_sizes=sizes, q2k_block_nums=nums
    )
    ref = sparse_reference(*dequantize(quantized), index, topk, nums, sizes)
    kernel_rel, kernel_max = assert_accuracy(out, ref)
    half, _ = bsa_attn_fwd(q, k, v, index, topk, sizes, q2k_block_nums=nums, sparse_block_size=128)
    original_ref = sparse_reference(q, k, v, index, topk, nums, sizes)
    torch.testing.assert_close(half.float(), original_ref, atol=0.025, rtol=0.015)
    # This includes Q/K/V quantization, unlike the kernel-only error above.
    total_rel, total_max = assert_accuracy(out, original_ref, relative=0.08, absolute=0.3)
    if nums is not None:
        empty = nums.repeat_interleave(128, dim=-1)[..., :sq] == 0
        assert torch.count_nonzero(out[empty]) == 0
    print(
        f"kernel rel={kernel_rel:.6f} max={kernel_max:.6f}; "
        f"with quantization rel={total_rel:.6f} max={total_max:.6f}"
    )


@pytest.mark.parametrize("sizes_mode", [0, 1, 2, 3])
@pytest.mark.parametrize("variable", [False, True])
def test_fp8_blk128_native_tails_and_metadata(sizes_mode, variable):
    torch.manual_seed(2100 + sizes_mode)
    # Alternate all dynamic extents in one process, including one-token tails,
    # odd topk, topk=1, and batch/head strides that are not multiples of 16.
    for batch, heads, sq, sk, topk in [
        (1, 1, 1, 1, 1),
        (2, 3, 129, 641, 5),
        (1, 5, 257, 259, 3),
        (3, 1, 63, 145, 1),
    ]:
        q, k, v = make_inputs(batch, heads, sq, sk, gain=0.5)
        index, nums, sizes = make_metadata(
            batch, heads, sq, sk, topk, variable=variable, sizes_mode=sizes_mode
        )
        quantized = list(quantize_sage_bhsd(q, k, v))
        # Flattened scales are part of the blk64 API contract too.
        quantized[3:] = [t.flatten() for t in quantized[3:]]
        out = bsa_fp8_fwd(
            *quantized, index, topk, sparse_block_size=128, block_sizes=sizes, q2k_block_nums=nums
        )
        ref = sparse_reference(*dequantize(quantized), index, topk, nums, sizes)
        assert_accuracy(out, ref)


@pytest.mark.parametrize("sizes_mode", [1, 2, 3])
@pytest.mark.parametrize("topk", [1, 2])
def test_fp8_blk128_provided_extent_respects_physical_tail(sizes_mode, topk):
    """Provided extents cannot make padded tokens participate in softmax."""
    q = torch.zeros(1, 1, 64, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 129, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.ones_like(k)
    index = torch.tensor(
        [1] if topk == 1 else [1, 0], device="cuda", dtype=torch.int32
    ).reshape(1, 1, 1, topk)
    # The public API accepts an extent larger than the physical one-token tail.
    sizes = torch.full(
        {1: (2,), 2: (1, 2), 3: (1, 1, 2)}[sizes_mode],
        128,
        device="cuda",
        dtype=torch.int32,
    )
    quantized = quantize_sage_bhsd(q, k, v)
    out = bsa_fp8_fwd(
        *quantized, index, topk, sparse_block_size=128, block_sizes=sizes
    )
    # Constant V has output one regardless of the number of valid KV tokens.
    # Use this analytic oracle instead of requiring exact FP32 reduction sums.
    assert_accuracy(out, torch.ones_like(out, dtype=torch.float32))


@pytest.mark.parametrize("topk", [1, 3, 5, 17, 33])
def test_fp8_blk128_scales_and_online_rescaling(topk):
    torch.manual_seed(3000 + topk)
    batch, heads, sq, sk = 2, 3, 257, 4353
    # Unit FP8 data, nonuniform independent scale factors. The K scales vary
    # within each 128-token block, so block-wide scaling would fail this test.
    q, k, v = [t.to(torch.float8_e4m3fn) for t in make_inputs(batch, heads, sq, sk, gain=0.5)]
    qs = torch.linspace(0.25, 3.0, batch * heads * sq, device="cuda").reshape(batch, heads, sq)
    ks = torch.linspace(0.125, 4.0, batch * heads * math.ceil(sk / 16), device="cuda").reshape(
        batch, heads, -1
    )
    vs = torch.linspace(0.2, 1.5, heads * 128, device="cuda").reshape(heads, 128)
    index, nums, sizes = make_metadata(batch, heads, sq, sk, topk, sizes_mode=3)
    scale = 0.13
    out = bsa_fp8_fwd(
        q, k, v, qs, ks, vs, index, topk, scale, sparse_block_size=128, block_sizes=sizes
    )
    ref = sparse_reference(
        *dequantize((q, k, v, qs, ks, vs)), index, topk, sizes=sizes, scale=scale
    )
    assert_accuracy(out, ref)


def test_fp8_blk128_channel_scales_and_empty_tiles():
    """Check every shuffled channel against an independent constant-V oracle."""
    torch.manual_seed(3811)
    batch, heads, sq, sk, topk = 2, 3, 129, 257, 3
    q = torch.zeros(batch, heads, sq, 128, device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.zeros(batch, heads, sk, 128, device="cuda", dtype=torch.float8_e4m3fn)
    channels = torch.arange(128, device="cuda")
    values = torch.where(channels % 2 == 0, 0.5, -2.0)
    v = values.view(1, 1, 1, 128).expand(batch, heads, sk, 128).contiguous().to(k.dtype)
    qs = torch.ones(batch, heads, sq, device="cuda")
    ks = torch.ones(batch, heads, math.ceil(sk / 16), device="cuda")
    vs = (torch.randperm(heads * 128, device="cuda") % 128 + 1).float().reshape(heads, 128) / 256
    index = torch.arange(topk, device="cuda", dtype=torch.int32)
    index = index.view(1, 1, 1, topk).expand(batch, heads, 2, topk).contiguous()
    nums = (torch.arange(batch * heads * 2, device="cuda") % 4).int().reshape(batch, heads, 2)
    sizes = torch.tensor([128, 128, 1], device="cuda", dtype=torch.int32)
    out = bsa_fp8_fwd(
        q, k, v, qs, ks, vs, index, topk, sparse_block_size=128,
        block_sizes=sizes, q2k_block_nums=nums,
    )
    expected = (values[None, :] * vs).bfloat16()[None, :, None, :]
    expected = expected.expand(batch, heads, sq, 128).clone()
    empty = nums.repeat_interleave(128, dim=-1)[..., :sq] == 0
    expected[empty] = 0
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_fp8_blk128_mixed_row_rescaling():
    """Mix constant and increasing logits across rows, tiles and K streams."""
    torch.manual_seed(3900)
    batch, heads, sq, sk, topk = 1, 2, 80 * 128 + 17, 17 * 128 + 1, 17
    q = torch.zeros(batch, heads, sq, 128, device="cuda")
    q[..., 0] = ((torch.arange(sq, device="cuda") // 32) % 2).float()
    k = torch.zeros(batch, heads, sk, 128, device="cuda")
    k[..., 0] = (torch.arange(sk, device="cuda") // 128).float() * 0.625 - 4.0
    v = torch.randn_like(k) * 0.25
    q, k, v = (x.to(torch.float8_e4m3fn) for x in (q, k, v))
    qs = torch.ones(batch, heads, sq, device="cuda")
    ks = torch.ones(batch, heads, math.ceil(sk / 16), device="cuda")
    vs = torch.ones(heads, 128, device="cuda")
    quantized = (q, k, v, qs, ks, vs)
    # Keep the full GPU workload under racecheck without instrumenting the
    # oracle's many GEMMs, which can exhaust the sanitizer's host memory.
    reference_inputs = tuple(t.cpu() for t in dequantize(quantized))
    nq, nk = math.ceil(sq / 128), math.ceil(sk / 128)
    # The kernel walks these indices in reverse: nonzero query rows see
    # increasing logits, while zero query rows keep the same normalization.
    index = torch.arange(nk - 1, nk - topk - 1, -1, device="cuda", dtype=torch.int32)
    index = index.expand(batch, heads, nq, topk).contiguous()
    sizes = (sk - torch.arange(nk, device="cuda") * 128).clamp(0, 128).int()
    for counts in ((17, 0, 7, 3), (0, 17, 3, 7)):
        nums = torch.tensor(counts, device="cuda", dtype=torch.int32)
        nums = nums.repeat(math.ceil(batch * heads * nq / 4))[: batch * heads * nq]
        nums = nums.reshape(batch, heads, nq)
        selected = index.masked_fill(torch.arange(topk, device="cuda") >= nums[..., None], 0x3FFFFFFF)
        out = bsa_fp8_fwd(
            *quantized, selected, topk, 1.0, sparse_block_size=128,
            block_sizes=sizes, q2k_block_nums=nums,
        )
        ref = sparse_reference(
            *reference_inputs, selected.cpu(), topk, nums.cpu(), sizes.cpu(), scale=1.0
        ).to(out.device)
        assert_accuracy(out, ref)
        for _ in range(3):
            repeat = bsa_fp8_fwd(
                *quantized, selected, topk, 1.0, sparse_block_size=128,
                block_sizes=sizes, q2k_block_nums=nums,
            )
            assert torch.equal(out, repeat)


def test_fp8_block_size_dispatch_and_current_stream():
    from bsa_fp8_blk64 import bsa_fp8_blk64_fwd

    torch.manual_seed(4000)
    q, k, v = make_inputs(1, 2, 256, 512, gain=0.5)
    quantized = quantize_sage_bhsd(q, k, v)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for block in (128, 64, 128, 64):
            index = torch.arange(512 // block, device="cuda", dtype=torch.int32)
            index = index.expand(1, 2, 256 // block, -1).contiguous()
            out = bsa_fp8_fwd(*quantized, index, index.shape[-1], sparse_block_size=block)
            ref = (
                torch.softmax(q.float() @ k.float().transpose(-1, -2) / math.sqrt(128), dim=-1)
                @ v.float()
            )
            assert_accuracy(out, ref, relative=0.08)
            if block == 64:
                old = bsa_fp8_blk64_fwd(*quantized, index, index.shape[-1])
                default = bsa_fp8_fwd(*quantized, index, index.shape[-1])
                torch.testing.assert_close(out, old, atol=0, rtol=0)
                torch.testing.assert_close(default, old, atol=0, rtol=0)
    torch.cuda.current_stream().wait_stream(stream)


def test_fp8_blk128_unused_capacity_and_fully_masked_blocks():
    torch.manual_seed(4500)
    q, k, v = make_inputs(2, 3, 257, 641, gain=0.5)
    quantized = quantize_sage_bhsd(q, k, v)
    index, _, sizes = make_metadata(2, 3, 257, 641, 5, sizes_mode=3)
    # Fixed top-k must ignore capacity beyond its first three entries.
    index[..., 3:] = 0x3FFFFFFF
    sizes.zero_()
    sizes[0, 0, 1] = 128
    out = bsa_fp8_fwd(*quantized, index, 3, sparse_block_size=128, block_sizes=sizes)
    ref = sparse_reference(*dequantize(quantized), index, 3, sizes=sizes)
    assert_accuracy(out, ref)
    assert torch.count_nonzero(out[1]) == 0
    assert torch.count_nonzero(out[0, 1:]) == 0


def test_fp8_blk128_large_kv_batch_stride():
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        pytest.skip("The stride regression needs 12 GiB of free GPU memory")
    # The KV batch stride exceeds signed int32 even for one-byte FP8 storage.
    # Constant values give an exact reference without a second large KV copy.
    batch, heads, sq, sk = 2, 32, 129, 524417
    q = torch.zeros(batch, heads, sq, 128, device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.zeros(batch, heads, sk, 128, device="cuda", dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)
    v[0].fill_(1.0)
    v[1].fill_(2.0)
    assert k.stride(0) > torch.iinfo(torch.int32).max
    qs = torch.ones(batch, heads, sq, device="cuda")
    ks = torch.ones(batch, heads, math.ceil(sk / 16), device="cuda")
    vs = torch.ones(heads, 128, device="cuda")
    index = torch.tensor([0, sk // 128], device="cuda", dtype=torch.int32)
    index = index.expand(batch, heads, math.ceil(sq / 128), 2).contiguous()
    out = bsa_fp8_fwd(q, k, v, qs, ks, vs, index, 2, sparse_block_size=128)
    torch.testing.assert_close(out[0], torch.ones_like(out[0]), atol=0, rtol=0)
    torch.testing.assert_close(out[1], torch.full_like(out[1], 2.0), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_blk128_half_precision_still_supported(dtype):
    torch.manual_seed(5000)
    q, k, v = [t.to(dtype) for t in make_inputs(2, 4, 256, 1024)]
    index, nums, sizes = make_metadata(2, 4, 256, 1024, 6, variable=True, sizes_mode=1)
    out, _ = bsa_attn_fwd(q, k, v, index, 6, sizes, q2k_block_nums=nums, sparse_block_size=128)
    ref = sparse_reference(q, k, v, index, 6, nums, sizes)
    torch.testing.assert_close(out.float(), ref, atol=0.025, rtol=0.015)


@pytest.mark.parametrize("flatten", [False, True])
def test_fp8_blk128_scale_views_and_graph_stream(flatten):
    q, k, v = make_inputs(2, 3, 129, 385)
    q8, k8, v8, qs, ks, vs = quantize_sage_bhsd(q, k, v)
    index, nums, sizes = make_metadata(2, 3, 129, 385, 3, variable=True, sizes_mode=2)
    def attention(scales):
        return bsa_fp8_fwd(q8, k8, v8, *scales, index, 3, sparse_block_size=128,
                           q2k_block_nums=nums, block_sizes=sizes)
    expected = attention((qs, ks, vs))
    views = []
    for scale in (qs, ks, vs):
        storage = torch.empty((*scale.shape, 2), device=scale.device, dtype=scale.dtype)
        storage[..., 0].copy_(scale)
        view = storage[..., 0]
        views.append(view.reshape(-1) if flatten else view)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = attention(views)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = attention(views)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    assert torch.equal(actual, expected)
    assert torch.equal(captured, expected)
