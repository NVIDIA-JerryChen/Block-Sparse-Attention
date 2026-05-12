"""CSR helpers for BSA block-sparse backward metadata."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


_INT32_MAX = torch.iinfo(torch.int32).max
_SCHEDULE_MODES = {"row_chunk": 1, "qrange": 2}


def _check_total_edges_int32(total_edges: int) -> int:
    total_edges = int(total_edges)
    if total_edges < 0:
        raise ValueError(f"total CSR edges must be non-negative, got {total_edges}")
    if total_edges > _INT32_MAX:
        raise OverflowError(
            f"CSR total_edges={total_edges} exceeds int32 row_ptr capacity; "
            "int64 CSR offsets are not implemented yet"
        )
    return total_edges


def _compute_bh_offsets_and_total_edges(
    B: int,
    H: int,
    num_q_blocks: int,
    block_sparse_num: int,
    device: torch.device,
    q2k_block_nums: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, int]:
    if q2k_block_nums is None:
        edges_per_bh = int(num_q_blocks) * int(block_sparse_num)
        total_edges = _check_total_edges_int32(int(B) * int(H) * edges_per_bh)
        return torch.empty((0,), dtype=torch.int32, device=device), total_edges

    bh_edges64 = q2k_block_nums.to(torch.int64).sum(dim=2).reshape(-1)
    bh_offsets64 = torch.empty((int(B) * int(H) + 1,), dtype=torch.int64, device=device)
    bh_offsets64[0] = 0
    bh_offsets64[1:] = torch.cumsum(bh_edges64, dim=0)
    # Variable sizing intentionally performs one D2H scalar sync for the exact
    # allocation size. Avoid reintroducing per-BH max/host sizing here.
    total_edges = _check_total_edges_int32(int(bh_offsets64[-1].item()))
    return bh_offsets64.to(torch.int32).contiguous(), total_edges


def _schedule_capacity_per_bh(
    num_kv_blocks: int,
    num_q_blocks: int,
    max_kv: int,
    block_sparse_num: int,
    q2k_block_nums: Optional[torch.Tensor],
    target_q_per_cta: int,
) -> int:
    if target_q_per_cta <= 0:
        raise ValueError(f"target_q_per_cta must be positive, got {target_q_per_cta}")
    if q2k_block_nums is None:
        edge_upper = int(num_q_blocks) * int(block_sparse_num)
    else:
        # Keep variable-count schedule sizing free of an additional D2H max sync.
        edge_upper = int(num_q_blocks) * int(max_kv)
    return int(num_kv_blocks) + (edge_upper + int(target_q_per_cta) - 1) // int(target_q_per_cta)


def _default_qrange_bucket_size_blocks(num_q_blocks: int) -> int:
    # Customer random-topK + sink benchmarks are most stable with a 1024-block
    # q-range. Keep this runtime-tunable through BSA_BWD_CSR_QRANGE_BLOCKS.
    return 1024


def convert_q2k_to_k2q_csr(
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    num_kv_blocks: int,
    q2k_block_nums: Optional[torch.Tensor] = None,
    return_schedule: bool = False,
    schedule_target_q_blocks: int = 256,
    schedule_mode: str = "qrange",
    schedule_q_bucket_size_blocks: Optional[int] = None,
) -> Tuple[torch.Tensor, ...]:
    """Build CSR k2q metadata from BSA q2k block indices.

    Args:
        q2k_block_index: int32 CUDA tensor [B, H, Q_blocks, max_kv].
        block_sparse_num: runtime fixed topK when ``q2k_block_nums`` is None.
        num_kv_blocks: total blk64 KV block count.
        q2k_block_nums: optional int32 CUDA tensor [B, H, Q_blocks].
        return_schedule: when true, emit per-(B,H) bwd schedule chunks.
        schedule_target_q_blocks: target Q block count per scheduled CTA.
        schedule_mode: "qrange" for qbucket-like q-range buckets, or
            "row_chunk" for v1 row slices.
        schedule_q_bucket_size_blocks: q-range width in Q blocks for qrange mode.

    Returns:
        k2q_row_ptr: int32 CUDA tensor [B, H, num_kv_blocks + 1].
        k2q_q_indices: int32 CUDA tensor [total_edges].
        If ``return_schedule=True``, also returns schedule metadata
        [B, H, work_capacity, 4] with fields
        (kv_block, q_indices_start, q_count, reserved/q_group), plus work counts [B, H].
    """
    if not q2k_block_index.is_cuda:
        raise ValueError("convert_q2k_to_k2q_csr requires CUDA q2k_block_index")
    if q2k_block_index.dtype != torch.int32:
        raise TypeError(f"q2k_block_index must be int32, got {q2k_block_index.dtype}")
    if q2k_block_index.ndim != 4:
        raise ValueError(f"q2k_block_index must be rank-4, got {q2k_block_index.shape}")
    if num_kv_blocks < 0:
        raise ValueError(f"num_kv_blocks must be non-negative, got {num_kv_blocks}")

    q2k_block_index = q2k_block_index.contiguous()
    B, H, num_q_blocks, max_kv = q2k_block_index.shape
    has_variable_nums = q2k_block_nums is not None

    if has_variable_nums:
        assert q2k_block_nums is not None
        if q2k_block_nums.dtype != torch.int32:
            raise TypeError(f"q2k_block_nums must be int32, got {q2k_block_nums.dtype}")
        if q2k_block_nums.shape != (B, H, num_q_blocks):
            raise ValueError(
                f"q2k_block_nums shape {tuple(q2k_block_nums.shape)} does not match "
                f"{(B, H, num_q_blocks)}"
            )
        if q2k_block_nums.device != q2k_block_index.device:
            raise ValueError("q2k_block_nums must share q2k_block_index device")
        q2k_nums_arg = q2k_block_nums.contiguous()
    else:
        if block_sparse_num < 0 or block_sparse_num > max_kv:
            raise ValueError(
                f"block_sparse_num={block_sparse_num} must be in [0, {max_kv}]"
            )
        q2k_nums_arg = torch.empty((1,), dtype=torch.int32, device=q2k_block_index.device)

    bh_offsets, total_edges = _compute_bh_offsets_and_total_edges(
        B,
        H,
        num_q_blocks,
        block_sparse_num,
        q2k_block_index.device,
        q2k_nums_arg if has_variable_nums else None,
    )
    row_ptr = torch.empty(
        (B, H, int(num_kv_blocks) + 1),
        dtype=torch.int32,
        device=q2k_block_index.device,
    )
    q_indices = torch.empty(
        (total_edges,),
        dtype=torch.int32,
        device=q2k_block_index.device,
    )

    if return_schedule:
        if schedule_mode not in _SCHEDULE_MODES:
            raise ValueError(
                f"schedule_mode must be one of {sorted(_SCHEDULE_MODES)}, got {schedule_mode!r}"
            )
        qrange_num_q_groups = 0
        qrange_q_bucket_size_blocks = 0
        if schedule_mode == "qrange":
            if schedule_q_bucket_size_blocks is None:
                schedule_q_bucket_size_blocks = _default_qrange_bucket_size_blocks(num_q_blocks)
            if schedule_q_bucket_size_blocks <= 0:
                raise ValueError(
                    f"schedule_q_bucket_size_blocks must be positive, got {schedule_q_bucket_size_blocks}"
                )
            qrange_q_bucket_size_blocks = int(schedule_q_bucket_size_blocks)
            qrange_num_q_groups = (
                int(num_q_blocks) + qrange_q_bucket_size_blocks - 1
            ) // qrange_q_bucket_size_blocks
            schedule_capacity = int(num_kv_blocks) * qrange_num_q_groups
        else:
            schedule_capacity = _schedule_capacity_per_bh(
                num_kv_blocks,
                num_q_blocks,
                max_kv,
                block_sparse_num,
                q2k_nums_arg if has_variable_nums else None,
                schedule_target_q_blocks,
            )
        schedule_metadata = torch.empty(
            (B, H, schedule_capacity, 4),
            dtype=torch.int32,
            device=q2k_block_index.device,
        )
        schedule_work_counts = torch.empty(
            (B, H),
            dtype=torch.int32,
            device=q2k_block_index.device,
        )

        from csrc.bwd.sm100_blk64.build_k2q_csr import run_build_k2q_csr_with_schedule

        run_build_k2q_csr_with_schedule(
            q2k_block_index,
            q2k_nums_arg,
            bh_offsets,
            row_ptr,
            q_indices,
            schedule_metadata,
            schedule_work_counts,
            schedule_target_q_blocks,
            schedule_capacity,
            _SCHEDULE_MODES[schedule_mode],
            qrange_num_q_groups,
            qrange_q_bucket_size_blocks,
            block_sparse_num,
            num_kv_blocks,
            total_edges,
            has_variable_nums,
        )
        return row_ptr, q_indices, schedule_metadata, schedule_work_counts

    from csrc.bwd.sm100_blk64.build_k2q_csr import run_build_k2q_csr

    run_build_k2q_csr(
        q2k_block_index,
        q2k_nums_arg,
        bh_offsets,
        row_ptr,
        q_indices,
        block_sparse_num,
        num_kv_blocks,
        total_edges,
        has_variable_nums,
    )
    return row_ptr, q_indices


def materialize_k2q_csr_to_dense(
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    num_q_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize CSR metadata to the existing dense k2q test format.

    This helper is intentionally simple and test-oriented.
    """
    if k2q_row_ptr.ndim != 3 or k2q_q_indices.ndim != 1:
        raise ValueError("expected row_ptr [B,H,Nkv+1] and q_indices [total_edges]")
    B, H, nrows_plus_1 = k2q_row_ptr.shape
    num_kv_blocks = nrows_plus_1 - 1
    k2q_num = k2q_row_ptr[..., 1:] - k2q_row_ptr[..., :-1]
    dense = torch.zeros(
        (B, H, num_kv_blocks, num_q_blocks),
        dtype=torch.int32,
        device=k2q_q_indices.device,
    )
    for b in range(B):
        for h in range(H):
            for kv in range(num_kv_blocks):
                start = int(k2q_row_ptr[b, h, kv].item())
                end = int(k2q_row_ptr[b, h, kv + 1].item())
                if end > start:
                    dense[b, h, kv, : end - start] = k2q_q_indices[start:end]
    return dense, k2q_num.contiguous()


__all__ = ["convert_q2k_to_k2q_csr", "materialize_k2q_csr_to_dense"]
