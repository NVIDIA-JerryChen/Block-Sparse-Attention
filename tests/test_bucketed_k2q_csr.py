import pytest
import torch

from block_sparse_attention.bsa_attn_interface import _build_bucketed_k2q_csr
from block_sparse_attention.csrc.bwd.bucketed_k2q_csr import (
    _bucketed_k2q_csr_compile_key,
    build_bucketed_k2q_csr_cutedsl,
)
from block_sparse_attention.utils.cache_utils import JITCache


def _reference_bucketed_k2q(
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    num_kv_blocks: int,
    bucket_size_blocks: int,
    q2k_block_nums: torch.Tensor | None,
):
    q2k_cpu = q2k_block_index.cpu()
    nums_cpu = q2k_block_nums.cpu() if q2k_block_nums is not None else None
    batch_size, num_heads, num_q_blocks, max_kv_blocks = q2k_cpu.shape
    num_q_groups = (num_q_blocks + bucket_size_blocks - 1) // bucket_size_blocks
    offsets = torch.empty(
        (batch_size, num_heads, num_q_groups, num_kv_blocks + 1),
        dtype=torch.int32,
    )
    rows = {}

    for batch_idx in range(batch_size):
        for head_idx in range(num_heads):
            base = 0
            for q_group_idx in range(num_q_groups):
                group_rows = [[] for _ in range(num_kv_blocks)]
                q_begin = q_group_idx * bucket_size_blocks
                q_end = min(num_q_blocks, q_begin + bucket_size_blocks)
                for q_block_idx in range(q_begin, q_end):
                    num_active_blocks = (
                        int(nums_cpu[batch_idx, head_idx, q_block_idx])
                        if nums_cpu is not None
                        else block_sparse_num
                    )
                    num_active_blocks = min(max(0, num_active_blocks), max_kv_blocks)
                    for block_idx in range(num_active_blocks):
                        kv_block_idx = int(
                            q2k_cpu[
                                batch_idx,
                                head_idx,
                                q_block_idx,
                                block_idx,
                            ]
                        )
                        if 0 <= kv_block_idx < num_kv_blocks:
                            group_rows[kv_block_idx].append(q_block_idx)

                running = base
                for kv_block_idx, expected_row in enumerate(group_rows):
                    offsets[
                        batch_idx,
                        head_idx,
                        q_group_idx,
                        kv_block_idx,
                    ] = running
                    rows[(batch_idx, head_idx, q_group_idx, kv_block_idx)] = (
                        expected_row
                    )
                    running += len(expected_row)
                offsets[
                    batch_idx,
                    head_idx,
                    q_group_idx,
                    num_kv_blocks,
                ] = running
                base = running
    return offsets, rows


def _assert_csr_matches_reference(
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    num_kv_blocks: int,
    bucket_size_blocks: int,
    q2k_block_nums: torch.Tensor | None = None,
):
    offsets, indices, num_q_groups, max_rows = _build_bucketed_k2q_csr(
        q2k_block_index,
        block_sparse_num,
        num_kv_blocks,
        bucket_size_blocks=bucket_size_blocks,
        q2k_block_nums=q2k_block_nums,
    )
    torch.cuda.synchronize()
    expected_offsets, expected_rows = _reference_bucketed_k2q(
        q2k_block_index,
        block_sparse_num,
        num_kv_blocks,
        bucket_size_blocks,
        q2k_block_nums,
    )

    assert offsets.dtype == torch.int32
    assert indices.dtype == torch.int32
    assert offsets.is_cuda and indices.is_cuda
    assert num_q_groups == expected_offsets.shape[2]
    assert max_rows == num_kv_blocks
    torch.testing.assert_close(offsets.cpu(), expected_offsets, rtol=0, atol=0)

    offsets_cpu = offsets.cpu()
    indices_cpu = indices.cpu()
    batch_size, num_heads, _, _ = offsets.shape
    for batch_idx in range(batch_size):
        for head_idx in range(num_heads):
            for q_group_idx in range(num_q_groups):
                for kv_block_idx in range(num_kv_blocks):
                    row_begin = int(
                        offsets_cpu[
                            batch_idx,
                            head_idx,
                            q_group_idx,
                            kv_block_idx,
                        ]
                    )
                    row_end = int(
                        offsets_cpu[
                            batch_idx,
                            head_idx,
                            q_group_idx,
                            kv_block_idx + 1,
                        ]
                    )
                    actual_row = indices_cpu[
                        batch_idx,
                        head_idx,
                        row_begin:row_end,
                    ].tolist()
                    expected_row = expected_rows[
                        (batch_idx, head_idx, q_group_idx, kv_block_idx)
                    ]
                    assert sorted(actual_row) == sorted(expected_row)

    num_q_blocks = q2k_block_index.shape[2]
    expected_capacity = max(
        1,
        num_q_blocks
        * (
            q2k_block_index.shape[3] if q2k_block_nums is not None else block_sparse_num
        ),
    )
    assert indices.shape == (batch_size, num_heads, expected_capacity)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bucketed_k2q_csr_fixed_multigroup():
    batch_size, num_heads, num_q_blocks = 2, 3, 5
    num_kv_blocks, storage_blocks, block_sparse_num = 7, 5, 3
    values = torch.arange(
        batch_size * num_heads * num_q_blocks * storage_blocks,
        dtype=torch.int32,
    ).reshape(batch_size, num_heads, num_q_blocks, storage_blocks)
    values[..., :block_sparse_num] %= num_kv_blocks
    values[..., block_sparse_num:] = -1
    q2k_block_index = values.cuda()
    _assert_csr_matches_reference(
        q2k_block_index,
        block_sparse_num,
        num_kv_blocks,
        bucket_size_blocks=2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bucketed_k2q_csr_variable_multigroup():
    q2k_block_index = torch.tensor(
        [
            [
                [
                    [0, 0, 3, -1, 9, 2],
                    [1, 4, 4, 2, 0, 6],
                    [2, 5, 1, 1, 3, 0],
                    [6, 2, 3, 4, 5, 1],
                    [0, 3, 6, 6, 2, 1],
                ],
                [
                    [6, 5, 4, 3, 2, 1],
                    [0, 8, 1, 2, 3, 4],
                    [3, 3, 3, 3, 3, 3],
                    [1, 2, 3, 4, 5, 6],
                    [5, 0, 2, 4, 6, 1],
                ],
            ]
        ],
        dtype=torch.int32,
        device="cuda",
    )
    q2k_block_nums = torch.tensor(
        [[[5, -1, 9, 3, 1], [6, 2, 4, 5, 3]]],
        dtype=torch.int32,
        device="cuda",
    )
    _assert_csr_matches_reference(
        q2k_block_index,
        block_sparse_num=0,
        num_kv_blocks=7,
        bucket_size_blocks=2,
        q2k_block_nums=q2k_block_nums,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bucketed_k2q_csr_fixed_topk_over_1024():
    num_q_blocks, num_kv_blocks, topk = 5, 1103, 1025
    q2k_block_index = torch.arange(
        num_q_blocks * topk,
        dtype=torch.int32,
        device="cuda",
    ).reshape(1, 1, num_q_blocks, topk)
    q2k_block_index %= num_kv_blocks
    _assert_csr_matches_reference(
        q2k_block_index,
        block_sparse_num=topk,
        num_kv_blocks=num_kv_blocks,
        bucket_size_blocks=2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bucketed_k2q_csr_variable_empty_edge_width():
    q2k_block_index = torch.empty(
        (1, 1, 2, 0),
        dtype=torch.int32,
        device="cuda",
    )
    q2k_block_nums = torch.zeros(
        (1, 1, 2),
        dtype=torch.int32,
        device="cuda",
    )
    _assert_csr_matches_reference(
        q2k_block_index,
        block_sparse_num=0,
        num_kv_blocks=3,
        bucket_size_blocks=1,
        q2k_block_nums=q2k_block_nums,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bucketed_k2q_csr_reuses_compile_across_runtime_shapes(monkeypatch):
    block_sparse_num = 2
    bucket_size_blocks = 3
    device_capability = torch.cuda.get_device_capability()
    compile_key = _bucketed_k2q_csr_compile_key(
        device_capability,
        block_sparse_num,
        bucket_size_blocks,
        False,
        4,
    )
    compile_cache = JITCache()
    monkeypatch.setattr(
        build_bucketed_k2q_csr_cutedsl,
        "compile_cache",
        compile_cache,
    )

    first = torch.tensor(
        [[[[0, 1, -1, -1], [1, 2, -1, -1]]]],
        dtype=torch.int32,
        device="cuda",
    )
    _assert_csr_matches_reference(
        first,
        block_sparse_num,
        num_kv_blocks=3,
        bucket_size_blocks=bucket_size_blocks,
    )
    compiled = compile_cache[compile_key]
    assert len(compile_cache.cache) == 1

    second = torch.arange(
        2 * 3 * 5 * 6,
        dtype=torch.int32,
        device="cuda",
    ).reshape(2, 3, 5, 6)
    second[..., :block_sparse_num] %= 7
    second[..., block_sparse_num:] = -1
    _assert_csr_matches_reference(
        second,
        block_sparse_num,
        num_kv_blocks=7,
        bucket_size_blocks=bucket_size_blocks,
    )

    assert len(compile_cache.cache) == 1
    assert compile_cache[compile_key] is compiled
