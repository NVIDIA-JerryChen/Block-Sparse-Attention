"""
Block-sparse runtime utilities for CUTE DSL kernels.

This module contains runtime execution functions for block-sparse attention kernels.
These utilities are used by CUTE DSL kernels to produce and consume block-sparse loads.
"""

from typing import Optional
from functools import partial
import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from quack import copy_utils

# Import data structures from block_sparsity
from csrc.bwd.sm90_blk64.block_sparsity import BlockSparseTensors
from csrc.bwd.sm90_blk64.named_barrier import NamedBarrierBwd


@cute.jit
def get_total_q_block_count_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """Count total tile iterations for given n_block (KV tile) in backward."""
    q_block_cnt, _, full_block_cnt, _ = blocksparse_tensors
    total = q_block_cnt[batch_idx, head_idx, n_block]
    if const_expr(full_block_cnt is not None):
        total = total + full_block_cnt[batch_idx, head_idx, n_block]
    return total * subtile_factor


@cute.jit
def get_block_sparse_iteration_info_bwd(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """Extract block-sparse iteration info for backward pass.

    Returns (curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, total_count).
    """
    q_cnt, q_idx, full_cnt, full_idx = blocksparse_tensors
    curr_q_cnt = q_cnt[batch_idx, head_idx, n_block]
    curr_q_idx = q_idx[batch_idx, head_idx, n_block, None]

    if const_expr(full_cnt is not None):
        curr_full_cnt = full_cnt[batch_idx, head_idx, n_block]
        curr_full_idx = full_idx[batch_idx, head_idx, n_block, None]
    else:
        curr_full_cnt = Int32(0)
        curr_full_idx = None

    sparse_block_count = curr_q_cnt
    if const_expr(full_cnt is not None):
        sparse_block_count = sparse_block_count + curr_full_cnt
    total_count = sparse_block_count * subtile_factor

    return curr_q_cnt, curr_q_idx, curr_full_cnt, curr_full_idx, total_count


@cute.jit
def get_m_block_from_iter_bwd(
    iter_idx,
    curr_q_cnt,
    curr_q_idx: cute.Tensor,
    curr_full_cnt,
    curr_full_idx: Optional[cute.Tensor],
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
):
    """Derive m_block index and is_full_block flag from iteration index.

    Returns (m_block, is_full_block):
        - m_block: The actual Q-tile block index
        - is_full_block: True if this is a full block (no mask_mod needed)
    """
    sparse_iter_idx = iter_idx // subtile_factor
    subtile_offset = iter_idx % subtile_factor

    sparse_m_block = Int32(0)
    is_full_block = False
    if const_expr(curr_full_idx is not None):
        if sparse_iter_idx < curr_q_cnt:
            sparse_m_block = curr_q_idx[sparse_iter_idx]
        else:
            sparse_m_block = curr_full_idx[sparse_iter_idx - curr_q_cnt]
            is_full_block = True
    else:
        sparse_m_block = curr_q_idx[sparse_iter_idx]

    return sparse_m_block * subtile_factor + subtile_offset, is_full_block


@cute.jit
def _load_q_do_block_sm90(
    m_block,
    producer_state_Q,
    producer_state_dO,
    pipeline_Q,
    pipeline_dO,
    load_K,
    load_V,
    load_Q,
    load_dO,
    load_LSE,
    load_dPsum,
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    Q_stage_eq_dO_stage: cutlass.Constexpr,
    load_kv: bool,
):
    """Load one Q/dO block, optionally loading K/V on first iteration."""
    if load_kv:
        pipeline_Q.producer_acquire(producer_state_Q, extra_tx_count=tma_copy_bytes_K)
        load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
    else:
        pipeline_Q.producer_acquire(producer_state_Q)
    load_Q(m_block, producer_state=producer_state_Q)
    load_LSE(m_block, producer_state=producer_state_Q)

    producer_state_dO_cur = (
        producer_state_dO if const_expr(not Q_stage_eq_dO_stage) else producer_state_Q
    )
    if load_kv:
        pipeline_dO.producer_acquire(producer_state_dO_cur, extra_tx_count=tma_copy_bytes_V)
        load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
    else:
        pipeline_dO.producer_acquire(producer_state_dO_cur)
    load_dO(m_block, producer_state=producer_state_dO_cur)
    load_dPsum(m_block, producer_state=producer_state_dO_cur)

    producer_state_Q.advance()
    producer_state_dO.advance()
    return producer_state_Q, producer_state_dO


@cute.jit
def produce_block_sparse_q_loads_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    producer_state_Q,
    producer_state_dO,
    pipeline_Q,
    pipeline_dO,
    load_K,
    load_V,
    load_Q,
    load_dO,
    load_LSE,
    load_dPsum,
    tma_copy_bytes_K,
    tma_copy_bytes_V,
    Q_stage_eq_dO_stage: cutlass.Constexpr,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
):
    """SM90 backward block sparse loading with separate partial/full loops.

    K/V are loaded with the first valid block. Iterates partial blocks first,
    then full blocks, matching consumer order.

    Returns updated (producer_state_Q, producer_state_dO).
    """
    q_cnt, q_idx, full_cnt, full_idx = blocksparse_tensors
    curr_q_cnt = q_cnt[batch_idx, head_idx, n_block]
    curr_q_idx = q_idx[batch_idx, head_idx, n_block, None]

    if const_expr(full_cnt is not None):
        curr_full_cnt = full_cnt[batch_idx, head_idx, n_block]
        curr_full_idx = full_idx[batch_idx, head_idx, n_block, None]
    else:
        curr_full_cnt = Int32(0)
        curr_full_idx = None

    kv_loaded = False

    for iter_idx in cutlass.range(curr_q_cnt * subtile_factor, unroll=1):
        sparse_idx = iter_idx // subtile_factor
        subtile_offset = iter_idx % subtile_factor
        m_block = curr_q_idx[sparse_idx] * subtile_factor + subtile_offset

        if m_block < m_block_max:
            producer_state_Q, producer_state_dO = _load_q_do_block_sm90(
                m_block,
                producer_state_Q,
                producer_state_dO,
                pipeline_Q,
                pipeline_dO,
                load_K,
                load_V,
                load_Q,
                load_dO,
                load_LSE,
                load_dPsum,
                tma_copy_bytes_K,
                tma_copy_bytes_V,
                Q_stage_eq_dO_stage,
                load_kv=not kv_loaded,
            )
            kv_loaded = True

    if const_expr(full_cnt is not None):
        for iter_idx in cutlass.range(curr_full_cnt * subtile_factor, unroll=1):
            sparse_idx = iter_idx // subtile_factor
            subtile_offset = iter_idx % subtile_factor
            m_block = curr_full_idx[sparse_idx] * subtile_factor + subtile_offset

            if m_block < m_block_max:
                producer_state_Q, producer_state_dO = _load_q_do_block_sm90(
                    m_block,
                    producer_state_Q,
                    producer_state_dO,
                    pipeline_Q,
                    pipeline_dO,
                    load_K,
                    load_V,
                    load_Q,
                    load_dO,
                    load_LSE,
                    load_dPsum,
                    tma_copy_bytes_K,
                    tma_copy_bytes_V,
                    Q_stage_eq_dO_stage,
                    load_kv=not kv_loaded,
                )
                kv_loaded = True

    return producer_state_Q, producer_state_dO


@cute.jit
def consume_block_sparse_mma_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    consumer_state_Q,
    consumer_state_dO,
    mma_one_m_block_fn,
    mask,
    mask_mod,
    is_causal: cutlass.Constexpr,
    is_local: cutlass.Constexpr,
    thr_mma_SdP,
    score_mod_fn=None,
    score_mod_bwd_fn=None,
    subtile_factor: cutlass.Constexpr = 1,
    m_block_max: int = 0,
    aux_tensors=None,
    fastdiv_mods=(None, None),
    block_size_k=None,
):
    """SM90 backward block sparse MMA consumption with separate partial/full loops.

    Partial blocks are processed first (with mask_mod applied), then full blocks
    (without mask_mod). This ensures mask_mod is only applied where needed.

    Returns updated (consumer_state_Q, consumer_state_dO).
    """
    q_cnt, q_idx, full_cnt, full_idx = blocksparse_tensors
    curr_q_cnt = q_cnt[batch_idx, head_idx, n_block]
    curr_q_idx = q_idx[batch_idx, head_idx, n_block, None]

    if const_expr(full_cnt is not None):
        curr_full_cnt = full_cnt[batch_idx, head_idx, n_block]
        curr_full_idx = full_idx[batch_idx, head_idx, n_block, None]
    else:
        curr_full_cnt = Int32(0)
        curr_full_idx = None

    dKV_accumulate = False

    mask_fn_partial = partial(
        mask.apply_mask,
        batch_idx=batch_idx,
        head_idx=head_idx,
        n_block=n_block,
        thr_mma=thr_mma_SdP,
        mask_seqlen=True,
        mask_causal=is_causal,
        mask_local=is_local,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        fastdiv_mods=fastdiv_mods,
        block_size_k=block_size_k,
    )

    mask_fn_full = partial(
        mask.apply_mask,
        batch_idx=batch_idx,
        head_idx=head_idx,
        n_block=n_block,
        thr_mma=thr_mma_SdP,
        mask_seqlen=True,
        mask_causal=is_causal,
        mask_local=is_local,
        aux_tensors=aux_tensors,
        fastdiv_mods=fastdiv_mods,
        block_size_k=block_size_k,
    )

    for iter_idx in cutlass.range(curr_q_cnt * subtile_factor, unroll=1):
        sparse_idx = iter_idx // subtile_factor
        subtile_offset = iter_idx % subtile_factor
        m_block = curr_q_idx[sparse_idx] * subtile_factor + subtile_offset

        if m_block < m_block_max:
            consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                m_block,
                consumer_state_Q,
                consumer_state_dO,
                mask_fn=mask_fn_partial,
                score_mod_fn=score_mod_fn,
                score_mod_bwd_fn=score_mod_bwd_fn,
                dKV_accumulate=dKV_accumulate,
            )
            dKV_accumulate = True

    if const_expr(full_cnt is not None):
        for iter_idx in cutlass.range(curr_full_cnt * subtile_factor, unroll=1):
            sparse_idx = iter_idx // subtile_factor
            subtile_offset = iter_idx % subtile_factor
            m_block = curr_full_idx[sparse_idx] * subtile_factor + subtile_offset

            if m_block < m_block_max:
                consumer_state_Q, consumer_state_dO = mma_one_m_block_fn(
                    m_block,
                    consumer_state_Q,
                    consumer_state_dO,
                    mask_fn=mask_fn_full,
                    score_mod_fn=score_mod_fn,
                    score_mod_bwd_fn=score_mod_bwd_fn,
                    dKV_accumulate=dKV_accumulate,
                )
                dKV_accumulate = True

    return consumer_state_Q, consumer_state_dO


@cute.jit
def _store_one_dQaccum_sm90(
    m_block,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
):
    """Store dQaccum for a single m_block."""
    for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
        cute.arch.cp_async_bulk_wait_group(num_dQ_warp_groups - 1 - warp_group_idx, read=True)
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
            number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )
    for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
        cute.arch.barrier(
            barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
            number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )
        with cute.arch.elect_one():
            copy_utils.cpasync_reduce_bulk_add_f32(
                sdQaccum[None, warp_group_idx].iterator,
                gdQaccum[(None, warp_group_idx), m_block].iterator,
                tma_copy_bytes_dQ,
            )
        cute.arch.cp_async_bulk_commit_group()


@cute.jit
def dQaccum_store_block_sparse_bwd_sm90(
    blocksparse_tensors: BlockSparseTensors,
    batch_idx,
    head_idx,
    n_block,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    subtile_factor: cutlass.Constexpr,
    m_block_max: int,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
):
    """SM90 backward block sparse dQaccum store with separate partial/full loops.

    Iterates partial blocks first, then full blocks, matching producer/consumer order.
    """
    q_cnt, q_idx, full_cnt, full_idx = blocksparse_tensors
    curr_q_cnt = q_cnt[batch_idx, head_idx, n_block]
    curr_q_idx = q_idx[batch_idx, head_idx, n_block, None]

    if const_expr(full_cnt is not None):
        curr_full_cnt = full_cnt[batch_idx, head_idx, n_block]
        curr_full_idx = full_idx[batch_idx, head_idx, n_block, None]
    else:
        curr_full_cnt = Int32(0)
        curr_full_idx = None

    for iter_idx in cutlass.range(curr_q_cnt * subtile_factor, unroll=1):
        sparse_idx = iter_idx // subtile_factor
        subtile_offset = iter_idx % subtile_factor
        m_block = curr_q_idx[sparse_idx] * subtile_factor + subtile_offset

        if m_block < m_block_max:
            _store_one_dQaccum_sm90(
                m_block,
                sdQaccum,
                gdQaccum,
                num_dQ_warp_groups,
                num_threads_per_warp_group,
                tma_copy_bytes_dQ,
            )

    if const_expr(full_cnt is not None):
        for iter_idx in cutlass.range(curr_full_cnt * subtile_factor, unroll=1):
            sparse_idx = iter_idx // subtile_factor
            subtile_offset = iter_idx % subtile_factor
            m_block = curr_full_idx[sparse_idx] * subtile_factor + subtile_offset

            if m_block < m_block_max:
                _store_one_dQaccum_sm90(
                    m_block,
                    sdQaccum,
                    gdQaccum,
                    num_dQ_warp_groups,
                    num_threads_per_warp_group,
                    tma_copy_bytes_dQ,
                )
