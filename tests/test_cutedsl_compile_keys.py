import math

import pytest
import torch

from bsa_attn_interface import (
    _bsa_fwd_blk64_kv_bucketed_combine_compile_key,
    _bsa_attn_fwd_sm90_blk64,
    _dynamic_tensors_compile_key,
    _sm90_bwd_compile_key,
    bsa_attn_fwd,
)
from csrc.common.fa_cute.interface import _bwd_preprocess_compile_key
from utils.cache_utils import JITCache


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
