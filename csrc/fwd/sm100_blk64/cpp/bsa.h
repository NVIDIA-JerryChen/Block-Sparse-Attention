/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 ******************************************************************************/
// bsa_fwd_params — BSA forward attention parameters.
// Naming aligned with FlashAttention Hopper (flash.h / Flash_fwd_params).
// Pure POD struct — no CUTLASS/torch dependencies.

#pragma once

#include <cstdint>

struct bsa_fwd_params {
    using index_t = int64_t;

    // The QKV and O matrices (BHSD layout: batch, num_heads, seqlen, head_dim).
    void const* __restrict__ q_ptr;
    void const* __restrict__ k_ptr;
    void const* __restrict__ v_ptr;
    void*       __restrict__ o_ptr;

    // The stride between axes of the Q, K, V and O matrices (in elements).
    // row_stride is along the seqlen axis; head_stride is along the num_heads axis.
    index_t q_batch_stride, q_row_stride, q_head_stride;
    index_t k_batch_stride, k_row_stride, k_head_stride;
    index_t v_batch_stride, v_row_stride, v_head_stride;
    index_t o_batch_stride, o_row_stride, o_head_stride, o_split_stride;

    // The pointer to the softmax log-sum-exp.
    void* __restrict__ softmax_lse_ptr;
    index_t lse_batch_stride, lse_row_stride, lse_head_stride, lse_split_stride;

    // Block-sparse indices.
    int const* __restrict__ block_indices_ptr;
    int const* __restrict__ block_sizes_ptr;
    int const* __restrict__ q2k_block_nums_ptr;
    int const* __restrict__ split_offsets_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, d;
    int h, h_k;
    int seqlen_k_rounded;    // used by K/V TMA descriptors to size the sparse-block axis

    // Tile counts.
    int num_m_blocks;       // ceil(seqlen_q / kRows)

    // Sparse config.
    int block_indices_stride;
    int uniform_block_sparse_num;  // per-tile raw block count when HasVarBlockNums=false
    int kv_splits;                 // 1 for the legacy fwd path.

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;
};
