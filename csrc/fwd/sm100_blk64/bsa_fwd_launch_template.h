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

#include "bsa.h"
#include "bsa_fwd_kernel_sm100.h"

namespace flash {

template<int kHeadDim>
void run_bsa_fwd(bsa_fwd_params const& p, cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    using namespace cute;

    using Kernel = FusedAttnKernel;
    using ML = Kernel::CollectiveMainloop;
    using EL = Kernel::CollectiveEpilogue;

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

    dim3 dim_grid = Kernel::get_grid_shape(kernel_params);
    dim3 dim_block = Kernel::get_block_shape();
    int smem_bytes = Kernel::SharedStorageSize;

    auto* kernel_ptr = &fused_attn_device<Kernel>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

    nvtxRangePushA("bsa_attn_fwd_kernel");
    fused_attn_device<Kernel><<<dim_grid, dim_block, smem_bytes, stream>>>(kernel_params);
    nvtxRangePop();
    C10_CUDA_CHECK(cudaGetLastError());
#else
    TORCH_CHECK(false, "requires SM100");
#endif
}

} // namespace flash
