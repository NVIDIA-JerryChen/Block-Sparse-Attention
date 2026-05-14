"""CSR metadata builder for BSA block-sparse backward."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


_INT32_MAX = torch.iinfo(torch.int32).max
DEFAULT_SCHEDULE_TARGET_Q_BLOCKS = 128
DEFAULT_QRANGE_Q_BUCKET_SIZE_BLOCKS = 1024
SCHEDULE_POLICY_ENV = "BSA_CSR_SCHEDULE_POLICY"


def _ceil_div(lhs: int, rhs: int) -> int:
    return (int(lhs) + int(rhs) - 1) // int(rhs)


def _env_int(name: str) -> Optional[int]:
    import os

    value = os.environ.get(name)
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _env_str(name: str) -> Optional[str]:
    import os

    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value.strip().lower()


def _select_schedule_policy(
    B: int,
    H: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    max_topk: int,
) -> Tuple[int, int]:
    """Select qrange-split schedule parameters.

    Customer sink-heavy patterns need different tradeoffs at different
    parallelism levels: single-head cases prefer finer 128-Q split chunks,
    while medium/large multi-head cases benefit from fewer CTAs and use 256-Q
    chunks. Medium videos use qrange=1152 in the customer full-benchmark
    steady state. Large single-head inputs use qrange=1088 to reduce per-range
    overhead without losing enough parallelism to hurt occupancy.
    """
    policy = _env_str(SCHEDULE_POLICY_ENV)
    target_override = _env_int("BSA_CSR_SCHEDULE_TARGET_Q_BLOCKS")
    qrange_override = _env_int("BSA_CSR_SCHEDULE_QRANGE_BLOCKS")
    q_blocks = int(num_q_blocks)
    parallel_heads = int(B) * int(H)

    if policy in {"qbuck", "qbucket", "qbucket_compatible"}:
        if qrange_override is not None:
            qrange_blocks = qrange_override
        elif q_blocks < 2048 or q_blocks >= 8192:
            qrange_blocks = 1088
        else:
            qrange_blocks = 1152
        if target_override is not None:
            return qrange_blocks, target_override
        # qbucket emits one task per (qrange, kv row). A target no smaller than
        # the qrange disables the CSR schedule's hotspot split for A/B testing.
        return qrange_blocks, qrange_blocks

    if policy not in {None, "adaptive", "csr_adaptive_qrange_v1"}:
        raise ValueError(
            f"unsupported {SCHEDULE_POLICY_ENV}={policy!r}; "
            "expected adaptive or qbucket"
        )

    if target_override is not None:
        target_q_blocks = target_override
    elif parallel_heads >= 4 and q_blocks >= 3000:
        target_q_blocks = 256
    else:
        target_q_blocks = DEFAULT_SCHEDULE_TARGET_Q_BLOCKS

    if qrange_override is not None:
        return qrange_override, target_q_blocks

    if 3000 <= q_blocks < 4096:
        qrange_blocks = 1152
    elif q_blocks >= 8192 and parallel_heads == 1:
        qrange_blocks = 1088
    else:
        qrange_blocks = DEFAULT_QRANGE_Q_BUCKET_SIZE_BLOCKS

    return qrange_blocks, target_q_blocks


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
    max_topk: int,
    device: torch.device,
    q2k_block_nums: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, int]:
    if q2k_block_nums is None:
        edges_per_bh = int(num_q_blocks) * int(max_topk)
        total_edges = _check_total_edges_int32(int(B) * int(H) * edges_per_bh)
        return torch.empty((0,), dtype=torch.int32, device=device), total_edges

    bh_edges64 = q2k_block_nums.to(torch.int64).sum(dim=2).reshape(-1)
    bh_offsets64 = torch.empty((int(B) * int(H) + 1,), dtype=torch.int64, device=device)
    bh_offsets64[0] = 0
    bh_offsets64[1:] = torch.cumsum(bh_edges64, dim=0)
    # Variable sizing intentionally performs one D2H scalar sync for the exact
    # packed allocation size. Avoid reintroducing per-BH max host sizing here.
    total_edges = _check_total_edges_int32(int(bh_offsets64[-1].item()))
    return bh_offsets64.to(torch.int32).contiguous(), total_edges


def _schedule_capacity_per_bh(
    num_kv_blocks: int,
    num_q_blocks: int,
    q2k_capacity: int,
    max_topk: int,
    q2k_block_nums: Optional[torch.Tensor],
    qrange_q_bucket_size_blocks: int,
    target_q_blocks: int,
) -> Tuple[int, int]:
    qrange_num_q_groups = _ceil_div(num_q_blocks, qrange_q_bucket_size_blocks)
    qrange_base_capacity = int(num_kv_blocks) * qrange_num_q_groups
    if q2k_block_nums is None:
        edge_upper = int(num_q_blocks) * int(max_topk)
    else:
        # Keep variable-count schedule sizing free of an additional D2H max sync.
        edge_upper = int(num_q_blocks) * int(q2k_capacity)
    split_extra_capacity = _ceil_div(edge_upper, target_q_blocks)
    if int(target_q_blocks) >= int(qrange_q_bucket_size_blocks):
        split_extra_capacity = 0
    return qrange_base_capacity + split_extra_capacity, qrange_num_q_groups


def convert_q2k_to_k2q_csr(
    q2k_block_index: torch.Tensor,
    max_topk: int,
    num_kv_blocks: int,
    q2k_block_nums: Optional[torch.Tensor] = None,
    return_schedule: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Build packed k2q CSR metadata from q2k block indices.

    Args:
        q2k_block_index: int32 CUDA tensor [B, H, Q_blocks, max_topk].
        max_topk: fixed topK, or q2k storage capacity / maximum topK when
            ``q2k_block_nums`` is provided.
        num_kv_blocks: total blk64 KV block count.
        q2k_block_nums: optional int32 CUDA tensor [B, H, Q_blocks].
        return_schedule: when true, also emit the qrange-split bwd schedule.

    Returns:
        ``k2q_row_ptr`` as int32 [B, H, num_kv_blocks + 1] and packed
        ``k2q_q_indices`` as int32 [total_edges]. With ``return_schedule=True``,
        also returns schedule metadata [B, H, work_capacity, 4] with fields
        (kv_block, q_indices_start, q_count, q_group) plus work counts [B, H].
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
    B, H, num_q_blocks, q2k_capacity = q2k_block_index.shape
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
        if max_topk < 0 or max_topk > q2k_capacity:
            raise ValueError(
                f"max_topk={max_topk} must be in [0, {q2k_capacity}]"
            )
        q2k_nums_arg = torch.empty((1,), dtype=torch.int32, device=q2k_block_index.device)

    bh_offsets, total_edges = _compute_bh_offsets_and_total_edges(
        B,
        H,
        num_q_blocks,
        max_topk,
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
        qrange_q_bucket_size_blocks, target_q_blocks = _select_schedule_policy(
            B,
            H,
            num_q_blocks,
            num_kv_blocks,
            max_topk,
        )
        schedule_capacity, qrange_num_q_groups = _schedule_capacity_per_bh(
            num_kv_blocks,
            num_q_blocks,
            q2k_capacity,
            max_topk,
            q2k_nums_arg if has_variable_nums else None,
            qrange_q_bucket_size_blocks,
            target_q_blocks,
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

        from csrc.common.build_k2q_csr import run_build_k2q_csr_with_schedule

        run_build_k2q_csr_with_schedule(
            q2k_block_index,
            q2k_nums_arg,
            bh_offsets,
            row_ptr,
            q_indices,
            schedule_metadata,
            schedule_work_counts,
            target_q_blocks,
            schedule_capacity,
            qrange_num_q_groups,
            qrange_q_bucket_size_blocks,
            max_topk,
            num_kv_blocks,
            total_edges,
            has_variable_nums,
        )
        max_work = max(1, int(schedule_work_counts.max().item()))
        if max_work < schedule_capacity:
            schedule_metadata = schedule_metadata[:, :, :max_work, :].contiguous()
        return row_ptr, q_indices, schedule_metadata, schedule_work_counts

    from csrc.common.build_k2q_csr import run_build_k2q_csr

    run_build_k2q_csr(
        q2k_block_index,
        q2k_nums_arg,
        bh_offsets,
        row_ptr,
        q_indices,
        max_topk,
        num_kv_blocks,
        total_edges,
        has_variable_nums,
    )
    return row_ptr, q_indices


__all__ = ["convert_q2k_to_k2q_csr"]
