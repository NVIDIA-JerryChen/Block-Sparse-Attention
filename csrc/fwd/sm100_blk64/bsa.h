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

    // The QKV and O matrices (BSHD layout).
    void const* __restrict__ q_ptr;
    void const* __restrict__ k_ptr;
    void const* __restrict__ v_ptr;
    void*       __restrict__ o_ptr;

    // The stride between rows of the Q, K, V and O matrices (in elements).
    index_t q_batch_stride, q_row_stride, q_head_stride;
    index_t k_batch_stride, k_row_stride, k_head_stride;
    index_t v_batch_stride, v_row_stride, v_head_stride;
    index_t o_batch_stride, o_row_stride, o_head_stride;

    // The pointer to the softmax log-sum-exp.
    void* __restrict__ softmax_lse_ptr;

    // Block-sparse indices.
    int const* __restrict__ block_indices_ptr;
    int const* __restrict__ block_sizes_ptr;
    int const* __restrict__ q2k_block_nums_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, d;
    int h, h_k;
    int seqlen_q_rounded, seqlen_k_rounded;

    // Tile counts.
    int num_m_blocks;       // seqlen_q_rounded / kRows
    int num_kv_iters;       // padded blocks / kSparseBlocksPerKV

    // Sparse config.
    int block_indices_stride;
    int raw_block_sparse_num;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;
};
