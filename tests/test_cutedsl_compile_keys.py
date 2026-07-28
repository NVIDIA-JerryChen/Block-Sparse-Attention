import math

import pytest
import torch

from block_sparse_attention import bsa_attn_fwd
from block_sparse_attention.bsa_attn_interface import (
    _bsa_fwd_blk64_kv_bucketed_combine_compile_key,
    _bsa_attn_fwd_sm90_blk64,
    _bsa_attn_fwd_sm120_blk64,
    _dynamic_tensors_compile_key,
    _prepare_sm120_sparse_metadata,
    _sm90_bwd_compile_key,
    _sm90_fwd_compile_key,
    _sm120_fp8_fwd_compile_key,
    _sm120_fwd_compile_key,
)
from block_sparse_attention.csrc.bwd.bsa_bwd_prepost import _bwd_preprocess_compile_key
from block_sparse_attention.utils.cache_utils import JITCache


def _make_sm90_bwd_tensors(batch: int, heads: int, seqlen_q: int, seqlen_k: int):
    q_shape = (batch, heads, seqlen_q, 128)
    k_shape = (batch, heads, seqlen_k, 128)
    q_tensors = tuple(torch.empty(q_shape, dtype=torch.bfloat16) for _ in range(4))
    k_tensors = tuple(torch.empty(k_shape, dtype=torch.bfloat16) for _ in range(4))
    lse = torch.empty((batch, heads, seqlen_q), dtype=torch.float32)
    return (*q_tensors[:3], *k_tensors[:2], q_tensors[3], *k_tensors[2:], lse)


def test_sm90_bwd_compile_key_ignores_runtime_shapes():
    stages = (2, 2, 2)
    first = _make_sm90_bwd_tensors(1, 2, 256, 384)
    second = _make_sm90_bwd_tensors(3, 5, 1024, 2048)

    first_key = _sm90_bwd_compile_key(90, torch.bfloat16, 128, False, stages, first)
    second_key = _sm90_bwd_compile_key(90, torch.bfloat16, 128, False, stages, second)

    assert first_key == second_key


def test_sm90_bwd_compile_key_tracks_broadcast_layout():
    stages = (2, 2, 2)
    contiguous = _make_sm90_bwd_tensors(2, 3, 256, 384)
    broadcast_q = torch.empty((2, 1, 256, 128), dtype=torch.bfloat16).expand(
        2, 3, 256, 128
    )
    broadcast = (broadcast_q, *contiguous[1:])

    contiguous_key = _sm90_bwd_compile_key(
        90, torch.bfloat16, 128, False, stages, contiguous
    )
    broadcast_key = _sm90_bwd_compile_key(
        90, torch.bfloat16, 128, False, stages, broadcast
    )

    assert contiguous_key != broadcast_key


def test_dynamic_tensor_compile_key_ignores_runtime_shapes():
    first = torch.empty((1, 2, 64, 128), dtype=torch.bfloat16)
    second = torch.empty((3, 5, 1024, 128), dtype=torch.bfloat16)

    first_key = _dynamic_tensors_compile_key("test", (128,), (first,))
    second_key = _dynamic_tensors_compile_key("test", (128,), (second,))

    assert first_key == second_key


def test_dynamic_tensor_compile_key_tracks_static_type_parts():
    contiguous = torch.empty((2, 3, 64, 128), dtype=torch.bfloat16)
    broadcast = torch.empty((2, 1, 64, 128), dtype=torch.bfloat16).expand(
        2, 3, 64, 128
    )
    float_input = contiguous.float()
    wide_storage = torch.empty(2 * 3 * 64 * 256, dtype=torch.bfloat16)
    padded = torch.as_strided(
        wide_storage,
        contiguous.shape,
        (3 * 64 * 160, 64 * 160, 160, 1),
    )
    non_unit_leading = torch.as_strided(
        wide_storage,
        contiguous.shape,
        (3 * 64 * 256, 64 * 256, 256, 2),
    )

    contiguous_key = _dynamic_tensors_compile_key("test", (128,), (contiguous,))
    broadcast_key = _dynamic_tensors_compile_key("test", (128,), (broadcast,))
    float_key = _dynamic_tensors_compile_key("test", (128,), (float_input,))
    padded_key = _dynamic_tensors_compile_key("test", (128,), (padded,))
    non_unit_key = _dynamic_tensors_compile_key(
        "test", (128,), (non_unit_leading,)
    )

    assert contiguous_key == padded_key
    assert contiguous_key != broadcast_key
    assert contiguous_key != float_key
    assert contiguous_key != non_unit_key


def _make_sm90_fwd_tensors(
    batch: int,
    q_heads: int,
    kv_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    capacity: int,
    qk_dim: int = 128,
    value_dim: int = 128,
    kv_splits: int = 1,
):
    q = torch.empty((batch, q_heads, seqlen_q, qk_dim), dtype=torch.bfloat16)
    k = torch.empty((batch, kv_heads, seqlen_k, qk_dim), dtype=torch.bfloat16)
    v = torch.empty(
        (batch, kv_heads, seqlen_k, value_dim), dtype=torch.bfloat16
    )
    output_dtype = torch.float32 if kv_splits > 1 else torch.bfloat16
    out = torch.empty(
        (batch, kv_splits * q_heads, seqlen_q, value_dim), dtype=output_dtype
    )
    lse = torch.empty(
        (batch, kv_splits * q_heads, seqlen_q), dtype=torch.float32
    )
    num_q_blocks = seqlen_q // 64
    num_kv_blocks = math.ceil(seqlen_k / 64)
    q2k = torch.empty(
        (batch, q_heads, num_q_blocks, capacity), dtype=torch.int32
    )
    q2k_nums = torch.empty((batch, q_heads, num_q_blocks), dtype=torch.int32)
    block_sizes = torch.empty(
        (batch, q_heads, num_kv_blocks), dtype=torch.int32
    )
    split_offsets = (
        torch.empty(
            (batch, q_heads, num_q_blocks, kv_splits + 1), dtype=torch.int32
        ).permute(3, 2, 1, 0)
        if kv_splits > 1
        else q2k_nums.permute(2, 1, 0)
    )
    return (
        q.permute(2, 3, 1, 0),
        k.permute(2, 3, 1, 0),
        v.permute(3, 2, 1, 0),
        out.permute(2, 3, 1, 0),
        lse.permute(2, 1, 0),
        q2k.permute(3, 2, 1, 0),
        q2k_nums.permute(2, 1, 0),
        block_sizes.permute(2, 1, 0),
        split_offsets,
    )


def test_sm90_fwd_compile_key_ignores_runtime_shapes():
    first = _make_sm90_fwd_tensors(1, 4, 2, 64, 128, 2, kv_splits=2)
    second = _make_sm90_fwd_tensors(3, 8, 4, 256, 513, 11, kv_splits=2)

    first_key = _sm90_fwd_compile_key(
        90, torch.bfloat16, 128, 128, 2, True, 2, False, first
    )
    second_key = _sm90_fwd_compile_key(
        90, torch.bfloat16, 128, 128, 2, True, 2, False, second
    )
    assert first_key == second_key


def test_sm90_fwd_compile_key_tracks_static_features():
    tensors = _make_sm90_fwd_tensors(1, 4, 2, 64, 128, 2)
    base_args = (90, torch.bfloat16, 128, 128, 2, True, 1, False)
    base = _sm90_fwd_compile_key(*base_args, tensors)
    alternatives = (
        (91, torch.bfloat16, 128, 128, 2, True, 1, False),
        (90, torch.float16, 128, 128, 2, True, 1, False),
        (90, torch.bfloat16, 64, 128, 2, True, 1, False),
        (90, torch.bfloat16, 128, 96, 2, True, 1, False),
        (90, torch.bfloat16, 128, 128, 4, True, 1, False),
        (90, torch.bfloat16, 128, 128, 2, False, 1, False),
        (90, torch.bfloat16, 128, 128, 2, True, 2, False),
        (90, torch.bfloat16, 128, 128, 2, True, 1, True),
    )
    assert all(
        base != _sm90_fwd_compile_key(*alternative, tensors)
        for alternative in alternatives
    )


def _make_sm120_fwd_tensors(
    batch: int,
    q_heads: int,
    kv_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    capacity: int,
):
    q = torch.empty((batch, q_heads, seqlen_q, 128), dtype=torch.bfloat16)
    k = torch.empty((batch, kv_heads, seqlen_k, 128), dtype=torch.bfloat16)
    v = torch.empty_like(k)
    out = torch.empty_like(q)
    lse = torch.empty((batch, q_heads, seqlen_q), dtype=torch.float32)
    num_q_blocks = math.ceil(seqlen_q / 64)
    num_kv_blocks = math.ceil(seqlen_k / 64)
    q2k = torch.empty(
        (batch, q_heads, num_q_blocks, capacity), dtype=torch.int32
    )
    q2k_nums = torch.empty((batch, q_heads, num_q_blocks), dtype=torch.int32)
    block_sizes = torch.empty((num_kv_blocks,), dtype=torch.int32)
    return (
        q.permute(2, 3, 1, 0),
        k.permute(2, 3, 1, 0),
        v.permute(3, 2, 1, 0),
        out.permute(2, 3, 1, 0),
        lse.permute(2, 1, 0),
        q2k.permute(3, 2, 1, 0),
        q2k_nums.permute(2, 1, 0),
        block_sizes,
    )


def test_sm120_fwd_compile_key_ignores_runtime_shapes():
    first = _make_sm120_fwd_tensors(1, 4, 2, 64, 128, 2)
    second = _make_sm120_fwd_tensors(3, 8, 4, 257, 513, 11)

    first_key = _sm120_fwd_compile_key(
        120, torch.bfloat16, 128, 128, 2, True, True, 1, first
    )
    second_key = _sm120_fwd_compile_key(
        120, torch.bfloat16, 128, 128, 2, True, True, 1, second
    )

    assert first_key == second_key


def test_sm120_fwd_compile_key_tracks_static_features():
    tensors = _make_sm120_fwd_tensors(1, 4, 2, 64, 128, 2)
    base = _sm120_fwd_compile_key(
        120, torch.bfloat16, 128, 128, 2, True, True, 1, tensors
    )

    assert base != _sm120_fwd_compile_key(
        120, torch.bfloat16, 128, 128, 1, True, True, 1, tensors
    )
    assert base != _sm120_fwd_compile_key(
        120, torch.bfloat16, 128, 128, 2, False, True, 1, tensors
    )
    assert base != _sm120_fwd_compile_key(
        120, torch.bfloat16, 128, 128, 2, True, True, 2, tensors
    )


def _make_sm120_fp8_fwd_tensors(
    heads: int,
    seqlen_q: int,
    seqlen_k: int,
    capacity: int,
):
    q = torch.empty((1, heads, seqlen_q, 128), dtype=torch.float8_e4m3fn)
    k = torch.empty((1, heads, seqlen_k, 128), dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)
    out = torch.empty((1, heads, seqlen_q, 128), dtype=torch.bfloat16)
    lse = torch.empty((1, heads, seqlen_q), dtype=torch.float32)
    q_scale = torch.empty((1, heads, seqlen_q), dtype=torch.float32)
    k_scale = torch.empty((1, heads, seqlen_k // 16), dtype=torch.float32)
    v_scale = torch.empty((heads, 128), dtype=torch.float32)
    q2k = torch.empty(
        (1, heads, seqlen_q // 64, capacity),
        dtype=torch.int32,
    )
    q2k_nums = torch.empty(
        (1, heads, seqlen_q // 64),
        dtype=torch.int32,
    )
    block_sizes = torch.empty(
        (1, heads, seqlen_k // 64),
        dtype=torch.int32,
    )
    return (
        q.permute(2, 3, 1, 0),
        k.permute(2, 3, 1, 0),
        v.permute(3, 2, 1, 0),
        out.permute(2, 3, 1, 0),
        lse.permute(2, 1, 0),
        q_scale.permute(2, 1, 0),
        k_scale.permute(2, 1, 0),
        v_scale.permute(1, 0),
        q2k.permute(3, 2, 1, 0),
        q2k_nums.permute(2, 1, 0),
        block_sizes.permute(2, 1, 0),
    )


def test_sm120_fp8_fwd_compile_key_ignores_runtime_shapes():
    first = _make_sm120_fp8_fwd_tensors(4, 64, 128, 2)
    second = _make_sm120_fp8_fwd_tensors(8, 256, 512, 7)

    assert _sm120_fp8_fwd_compile_key(
        120,
        128,
        True,
        True,
        3,
        first,
    ) == _sm120_fp8_fwd_compile_key(
        120,
        128,
        True,
        True,
        3,
        second,
    )


def test_sm120_fp8_fwd_compile_key_tracks_sparse_metadata_modes():
    tensors = _make_sm120_fp8_fwd_tensors(4, 64, 128, 2)
    base = _sm120_fp8_fwd_compile_key(
        120,
        128,
        False,
        False,
        0,
        tensors,
    )
    alternatives = (
        (121, 128, False, False, 0),
        (120, 64, False, False, 0),
        (120, 128, True, False, 0),
        (120, 128, False, True, 1),
        (120, 128, False, True, 2),
        (120, 128, False, True, 3),
    )

    assert all(
        base != _sm120_fp8_fwd_compile_key(*alternative, tensors)
        for alternative in alternatives
    )


@pytest.mark.parametrize(
    ("block_sizes_shape", "expected_mode", "expected_shape"),
    (
        (None, 0, (2, 3, 2)),
        ((5,), 1, (5,)),
        ((2, 5), 2, (5, 2)),
        ((2, 3, 5), 3, (5, 3, 2)),
    ),
)
def test_prepare_sm120_sparse_metadata_layouts(
    block_sizes_shape,
    expected_mode,
    expected_shape,
):
    batch, heads, num_q_blocks, num_kv_blocks, capacity = 2, 3, 2, 5, 4
    q2k = torch.empty(
        (batch, heads, num_q_blocks, capacity * 2),
        dtype=torch.int32,
    )[..., ::2]
    block_nums = torch.empty(
        (batch, heads, num_q_blocks, 2),
        dtype=torch.int32,
    )[..., 0]
    block_sizes = (
        None
        if block_sizes_shape is None
        else torch.empty(block_sizes_shape, dtype=torch.int32)
    )

    (
        has_block_nums,
        has_block_sizes,
        block_sizes_mode,
        q2k_t,
        q2k_nums_t,
        block_sizes_t,
    ) = _prepare_sm120_sparse_metadata(
        q2k,
        block_nums,
        block_sizes,
        0,
        batch_size=batch,
        num_heads=heads,
        num_q_blocks=num_q_blocks,
        num_kv_blocks=num_kv_blocks,
        device=q2k.device,
    )

    assert has_block_nums
    assert has_block_sizes == (block_sizes_shape is not None)
    assert block_sizes_mode == expected_mode
    assert q2k_t.shape == (capacity, num_q_blocks, heads, batch)
    assert q2k_nums_t.shape == (num_q_blocks, heads, batch)
    assert block_sizes_t.shape == expected_shape
    assert q2k_t.stride(0) == 1
    assert q2k_nums_t.stride(0) == 1
    assert block_sizes_t.stride(0) == 1
    if block_sizes_shape is None:
        assert block_sizes_t is q2k_nums_t


def test_prepare_sm120_sparse_metadata_uses_uniform_sentinels():
    q2k = torch.empty((1, 4, 2, 3), dtype=torch.int32)

    (
        has_block_nums,
        has_block_sizes,
        block_sizes_mode,
        q2k_t,
        q2k_nums_t,
        block_sizes_t,
    ) = _prepare_sm120_sparse_metadata(
        q2k,
        None,
        None,
        2,
        batch_size=1,
        num_heads=4,
        num_q_blocks=2,
        num_kv_blocks=5,
        device=q2k.device,
    )

    assert not has_block_nums
    assert not has_block_sizes
    assert block_sizes_mode == 0
    assert q2k_nums_t is q2k_t
    assert block_sizes_t is q2k_t


def test_arch_specific_helper_keys_do_not_cross_devices():
    combine_args = (torch.bfloat16, 128, 16, 64, 2, 128, 4)
    assert _bsa_fwd_blk64_kv_bucketed_combine_compile_key(
        90, *combine_args
    ) != _bsa_fwd_blk64_kv_bucketed_combine_compile_key(100, *combine_args)

    preprocess_args = (
        torch.bfloat16,
        128,
        128,
        64,
        False,
        False,
        False,
        True,
        True,
    )
    assert _bwd_preprocess_compile_key(
        90, *preprocess_args
    ) != _bwd_preprocess_compile_key(100, *preprocess_args)


def _run_sm90_fwd_case(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_k: int,
    capacity: int,
):
    head_dim = 128
    block_sparse_num = 2
    q = torch.randn(
        (batch, heads, seqlen_q, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        (batch, heads, seqlen_k, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.randn_like(k)
    num_q_blocks = seqlen_q // 64
    num_kv_blocks = seqlen_k // 64
    q2k_block_index = torch.full(
        (batch, heads, num_q_blocks, capacity),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    q2k_block_index[..., 0] = 0
    q2k_block_index[..., 1] = 1
    block_sizes = torch.full(
        (num_kv_blocks,),
        64,
        dtype=torch.int32,
        device="cuda",
    )

    out, lse = _bsa_attn_fwd_sm90_blk64(
        q,
        k,
        v,
        q2k_block_index,
        block_sparse_num,
        block_sizes=block_sizes,
    )

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.einsum(
        "bhqd,bhkd->bhqk",
        q.float(),
        k[:, :, : 2 * 64].float(),
    ) * scale
    ref_lse = torch.logsumexp(scores, dim=-1)
    ref_out = torch.einsum(
        "bhqk,bhkd->bhqd",
        torch.softmax(scores, dim=-1),
        v[:, :, : 2 * 64].float(),
    )
    torch.testing.assert_close(out.float(), ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=0.0, atol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 is required",
)
def test_sm90_fwd_reuses_compiled_kernel_across_runtime_shapes(monkeypatch):
    torch.manual_seed(0)
    compile_cache = JITCache()
    monkeypatch.setattr(bsa_attn_fwd, "compile_cache", compile_cache)

    _run_sm90_fwd_case(1, 2, 64, 128, 2)
    assert len(compile_cache.cache) == 1

    _run_sm90_fwd_case(2, 4, 192, 256, 6)
    assert len(compile_cache.cache) == 1


def _run_sm120_fwd_case(
    batch: int,
    q_heads: int,
    kv_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    capacity: int,
):
    head_dim = 128
    block_sparse_num = 2
    q = torch.randn(
        (batch, q_heads, seqlen_q, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        (batch, kv_heads, seqlen_k, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.randn_like(k)
    num_q_blocks = math.ceil(seqlen_q / 64)
    q2k_block_index = torch.full(
        (batch, q_heads, num_q_blocks, capacity),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    q2k_block_index[..., 0] = 0
    q2k_block_index[..., 1] = 1

    out, lse = _bsa_attn_fwd_sm120_blk64(
        q,
        k,
        v,
        q2k_block_index,
        block_sparse_num,
        block_sizes=None,
    )

    gqa_ratio = q_heads // kv_heads
    k_expanded = k.float().repeat_interleave(gqa_ratio, dim=1)
    v_expanded = v.float().repeat_interleave(gqa_ratio, dim=1)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k_expanded) * scale
    ref_lse = torch.logsumexp(scores, dim=-1)
    ref_out = torch.einsum(
        "bhqk,bhkd->bhqd",
        torch.softmax(scores, dim=-1),
        v_expanded,
    )
    torch.testing.assert_close(out.float(), ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=0.0, atol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="SM120 is required",
)
def test_sm120_fwd_reuses_compiled_kernel_across_runtime_shapes(monkeypatch):
    torch.manual_seed(0)
    compile_cache = JITCache()
    monkeypatch.setattr(bsa_attn_fwd, "compile_cache", compile_cache)

    _run_sm120_fwd_case(1, 4, 2, 64, 128, 2)
    assert len(compile_cache.cache) == 1

    _run_sm120_fwd_case(2, 8, 4, 130, 100, 7)
    assert len(compile_cache.cache) == 1
