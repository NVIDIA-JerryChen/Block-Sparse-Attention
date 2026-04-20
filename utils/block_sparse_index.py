"""Triton helpers to invert q→kv block-sparse indices into kv→q form.

The two-kernel split (scatter-to-map, then pack set bits into a dense
index tensor) is much faster than a torch ``scatter``/``sort`` pipeline
because:

  * the scatter kernel has no Python loop for variable per-(b, h, q) counts
  * the pack kernel does an O(num_q_blocks) linear scan instead of an
    O(num_q_blocks · log num_q_blocks) sort
  * both kernels fuse the two passes into single-kernel grid launches
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_q2k_to_kv_major_map_kernel(
    map_ptr,
    index_ptr,
    num_ptr,
    map_bs_s, map_h_s, map_kv_s, map_q_s,
    idx_bs_s, idx_h_s, idx_q_s, idx_k_s,
    num_bs_s, num_h_s, num_q_s,
    num_kv_blocks,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    q = tl.program_id(2)

    idx_base = index_ptr + b * idx_bs_s + h * idx_h_s + q * idx_q_s
    map_base = map_ptr + b * map_bs_s + h * map_h_s + q * map_q_s
    n = tl.load(num_ptr + b * num_bs_s + h * num_h_s + q * num_q_s)

    for i in tl.range(n):
        kv = tl.load(idx_base + i * idx_k_s)
        if (kv >= 0) & (kv < num_kv_blocks):
            tl.store(map_base + kv * map_kv_s, 1)


@triton.jit
def _map_to_k2q_index_kernel(
    map_ptr,
    index_ptr,
    num_ptr,
    map_bs_s, map_h_s, map_kv_s, map_q_s,
    idx_bs_s, idx_h_s, idx_kv_s, idx_q_s,
    num_bs_s, num_h_s, num_kv_s,
    num_q_blocks,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    kv = tl.program_id(2)

    map_base = map_ptr + b * map_bs_s + h * map_h_s + kv * map_kv_s
    idx_base = index_ptr + b * idx_bs_s + h * idx_h_s + kv * idx_kv_s

    cnt = 0
    for i in tl.range(num_q_blocks):
        v = tl.load(map_base + i * map_q_s)
        if v != 0:
            tl.store(idx_base + cnt * idx_q_s, i)
            cnt += 1

    tl.store(num_ptr + b * num_bs_s + h * num_h_s + kv * num_kv_s, cnt)


def convert_q2k_to_k2q_triton(
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    num_kv_blocks: int,
    q2k_block_nums: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton implementation of :func:`bsa_attn_interface.convert_q2k_to_k2q`.

    See that function for the argument contract. This version runs two Triton
    kernels (scatter to a boolean map, then pack into dense indices + counts).
    """
    assert q2k_block_index.is_cuda, "triton path requires CUDA tensors"
    assert q2k_block_index.dtype == torch.int32

    q2k_block_index = q2k_block_index.contiguous()
    bs, h, num_q_blocks, _ = q2k_block_index.shape
    device = q2k_block_index.device

    if q2k_block_nums is None:
        nums = torch.full(
            (bs, h, num_q_blocks), block_sparse_num,
            dtype=torch.int32, device=device,
        )
    else:
        assert q2k_block_nums.dtype == torch.int32
        assert q2k_block_nums.shape == (bs, h, num_q_blocks)
        nums = q2k_block_nums.contiguous()

    # kv-major boolean map: map[b, h, kv, q] = True iff q attends kv.
    # uint8 (not bool) for straightforward triton store semantics.
    block_map = torch.zeros(
        (bs, h, num_kv_blocks, num_q_blocks), dtype=torch.uint8, device=device,
    )
    k2q_index = torch.zeros(
        (bs, h, num_kv_blocks, num_q_blocks), dtype=torch.int32, device=device,
    )
    k2q_num = torch.empty(
        (bs, h, num_kv_blocks), dtype=torch.int32, device=device,
    )

    grid_scatter = (bs, h, num_q_blocks)
    _scatter_q2k_to_kv_major_map_kernel[grid_scatter](
        block_map, q2k_block_index, nums,
        block_map.stride(0), block_map.stride(1), block_map.stride(2), block_map.stride(3),
        q2k_block_index.stride(0), q2k_block_index.stride(1),
        q2k_block_index.stride(2), q2k_block_index.stride(3),
        nums.stride(0), nums.stride(1), nums.stride(2),
        num_kv_blocks,
    )

    grid_pack = (bs, h, num_kv_blocks)
    _map_to_k2q_index_kernel[grid_pack](
        block_map, k2q_index, k2q_num,
        block_map.stride(0), block_map.stride(1), block_map.stride(2), block_map.stride(3),
        k2q_index.stride(0), k2q_index.stride(1), k2q_index.stride(2), k2q_index.stride(3),
        k2q_num.stride(0), k2q_num.stride(1), k2q_num.stride(2),
        num_q_blocks,
    )
    return k2q_index, k2q_num
