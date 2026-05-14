"""Lazy-loaded CUDA extension for BSA q2k -> k2q CSR conversion."""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS_DIR, "build_k2q_csr.cu")
_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        _EXT = load(
            name="bsa_build_k2q_csr_ext",
            sources=[_SRC],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-arch=sm_100",
                "--ptxas-options=-v",
                "--expt-relaxed-constexpr",
            ],
            verbose=False,
        )
    return _EXT


def run_build_k2q_csr(
    q2k: torch.Tensor,
    q2k_nums: torch.Tensor,
    bh_offsets: torch.Tensor,
    row_ptr: torch.Tensor,
    q_indices: torch.Tensor,
    max_topk: int,
    num_kv_blocks: int,
    total_edges: int,
    has_variable_nums: bool,
) -> None:
    """Fill ``row_ptr`` and ``q_indices`` in place.

    Args:
      q2k: int32 CUDA tensor with shape [B, H, Q_blocks, max_topk].
      q2k_nums: int32 CUDA tensor with shape [B, H, Q_blocks] when
        ``has_variable_nums`` is true; otherwise any CUDA int32 tensor.
      bh_offsets: int32 CUDA tensor with shape [B * H + 1] when
        ``has_variable_nums`` is true; otherwise an empty CUDA int32 tensor.
      row_ptr: int32 CUDA tensor with shape [B, H, num_kv_blocks + 1].
      q_indices: int32 CUDA tensor with shape [total_edges].
      max_topk: fixed topK, or q2k storage capacity / maximum topK when
        ``has_variable_nums`` is true.
      num_kv_blocks: number of blk64 KV blocks.
      total_edges: packed q_indices element count.
      has_variable_nums: whether ``q2k_nums`` provides per-row valid counts.
    """
    _load_ext().run_build_k2q_csr(
        q2k,
        q2k_nums,
        bh_offsets,
        row_ptr,
        q_indices,
        torch.empty((0,), dtype=torch.int32, device=q2k.device),
        torch.empty((0,), dtype=torch.int32, device=q2k.device),
        1,
        0,
        False,
        0,
        0,
        int(max_topk),
        int(num_kv_blocks),
        int(total_edges),
        bool(has_variable_nums),
    )


def run_build_k2q_csr_with_schedule(
    q2k: torch.Tensor,
    q2k_nums: torch.Tensor,
    bh_offsets: torch.Tensor,
    row_ptr: torch.Tensor,
    q_indices: torch.Tensor,
    schedule_metadata: torch.Tensor,
    schedule_work_counts: torch.Tensor,
    target_q_per_cta: int,
    schedule_capacity_per_bh: int,
    qrange_num_q_groups: int,
    qrange_q_bucket_size_blocks: int,
    max_topk: int,
    num_kv_blocks: int,
    total_edges: int,
    has_variable_nums: bool,
) -> None:
    """Fill CSR metadata and per-(B,H) flat schedule metadata in place."""
    _load_ext().run_build_k2q_csr(
        q2k,
        q2k_nums,
        bh_offsets,
        row_ptr,
        q_indices,
        schedule_metadata,
        schedule_work_counts,
        int(target_q_per_cta),
        int(schedule_capacity_per_bh),
        True,
        int(qrange_num_q_groups),
        int(qrange_q_bucket_size_blocks),
        int(max_topk),
        int(num_kv_blocks),
        int(total_edges),
        bool(has_variable_nums),
    )


__all__ = ["run_build_k2q_csr", "run_build_k2q_csr_with_schedule"]
