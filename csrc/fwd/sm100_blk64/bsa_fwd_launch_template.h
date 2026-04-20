/******************************************************************************
  * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
  ******************************************************************************/
// Host launch for BSA fused attention forward kernel (blk=64, C++ AOT)
// FA Hopper pattern: run_bsa_fwd(bsa_fwd_params, stream) builds TMA + launches kernel
#pragma once
#define CUDA_CTA_RECONFIG_ACTIVATED 1

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "cutlass/cluster_launch.hpp"

#include "bsa.h"
#include "bsa_fwd_kernel_sm100.h"

namespace flash {

template<int kHeadDim, bool HasBlockSizes, bool HasVarBlockNums, bool UseClc>
void run_bsa_fwd(bsa_fwd_params const& p, cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    using namespace cute;

    using Kernel = FusedAttnKernel<HasBlockSizes, HasVarBlockNums, UseClc>;
    using ML = typename Kernel::CollectiveMainloop;
    using EL = typename Kernel::CollectiveEpilogue;

    // Build TMA descriptors from bsa_fwd_params
    auto tma_q = ML::make_tma_load_Q(p);
    auto tma_k = ML::make_tma_load_K(p);
    auto tma_v = ML::make_tma_load_V(p);
    auto tma_o = EL::make_tma_store_O(p);

    typename Kernel::Params kernel_params{
        p,
        tma_q, tma_k, tma_v, tma_o,
        ML::make_shape_Q(p), ML::make_shape_K(p), ML::make_shape_V(p),
        EL::make_shape_O(p),
    };

    if constexpr (UseClc) {
        int device_id = 0;
        C10_CUDA_CHECK(cudaGetDevice(&device_id));
        int sm_count = 0;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(
                &sm_count, cudaDevAttrMultiProcessorCount, device_id));
        kernel_params.sm_count = sm_count;
    }

    dim3 dim_grid = Kernel::get_grid_shape(kernel_params);
    dim3 dim_block = Kernel::get_block_shape();
    int smem_bytes = Kernel::SharedStorageSize;

    auto* kernel_ptr = &fused_attn_device<Kernel>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    if constexpr (UseClc) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    }

    nvtxRangePushA("bsa_attn_fwd_kernel");
    if constexpr (UseClc) {
        dim3 dim_cluster = Kernel::get_cluster_shape();
        void* args[] = {&kernel_params};
        cutlass::ClusterLauncher::launch(
                dim_grid, dim_cluster, dim_block,
                static_cast<size_t>(smem_bytes), stream,
                reinterpret_cast<void const*>(kernel_ptr), args);
    } else {
        fused_attn_device<Kernel><<<dim_grid, dim_block, smem_bytes, stream>>>(kernel_params);
    }
    nvtxRangePop();
    C10_CUDA_CHECK(cudaGetLastError());
#else
    TORCH_CHECK(false, "requires SM100");
#endif
}

} // namespace flash
