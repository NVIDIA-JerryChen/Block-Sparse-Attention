/******************************************************************************
  * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
  ******************************************************************************/
// KV-bucketed forward helper kernels.

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace flash {

namespace {

__global__ void build_kv_split_offsets_kernel(
        int const* __restrict__ q2k_block_nums,
        int* __restrict__ split_offsets,
        int total_tiles,
        int kv_splits,
        int uniform_block_sparse_num) {
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= total_tiles) {
        return;
    }

    int valid_kv = q2k_block_nums != nullptr
        ? q2k_block_nums[tile]
        : uniform_block_sparse_num;
    if (valid_kv < 0) {
        valid_kv = 0;
    }

    int base = tile * (kv_splits + 1);
    for (int split = 0; split <= kv_splits; ++split) {
        split_offsets[base + split] = (valid_kv * split + kv_splits - 1) / kv_splits;
    }
}

} // namespace

void run_bsa_fwd_kv_split_schedule(
        int const* q2k_block_nums,
        int* split_offsets,
        int total_tiles,
        int kv_splits,
        int uniform_block_sparse_num,
        cudaStream_t stream) {
    if (total_tiles <= 0) {
        return;
    }
    constexpr int kThreads = 256;
    int blocks = (total_tiles + kThreads - 1) / kThreads;
    build_kv_split_offsets_kernel<<<blocks, kThreads, 0, stream>>>(
            q2k_block_nums, split_offsets, total_tiles, kv_splits, uniform_block_sparse_num);
    C10_CUDA_CHECK(cudaGetLastError());
}

} // namespace flash
