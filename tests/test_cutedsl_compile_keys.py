import torch

from bsa_attn_interface import (
    _dynamic_tensors_compile_key,
    _sm90_bwd_compile_key,
)


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
    non_unit_leading = torch.as_strided(
        wide_storage,
        contiguous.shape,
        (3 * 64 * 256, 64 * 256, 256, 2),
    )

    contiguous_key = _dynamic_tensors_compile_key("test", (128,), (contiguous,))
    broadcast_key = _dynamic_tensors_compile_key("test", (128,), (broadcast,))
    float_key = _dynamic_tensors_compile_key("test", (128,), (float_input,))
    non_unit_key = _dynamic_tensors_compile_key(
        "test", (128,), (non_unit_leading,)
    )

    assert contiguous_key != broadcast_key
    assert contiguous_key != float_key
    assert contiguous_key != non_unit_key
