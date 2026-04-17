/******************************************************************************
  * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
  ******************************************************************************/
// C++ API for BSA fused attention forward kernel (blk=64)
// FA Hopper pattern: set_params_fprop() + run_bsa_fwd()

#include <cmath>
#include <tuple>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "bsa.h"
#include "static_switch.h"

namespace flash {

// Defined in bsa_fwd_launch_template.h, instantiated in instantiations/bsa_fwd_hdim128_bf16_hbs{0,1}_hvbn{0,1}_sm100.cu
template<int kHeadDim, bool HasBlockSizes, bool HasVarBlockNums>
void run_bsa_fwd(bsa_fwd_params const&, cudaStream_t);

// FA Hopper pattern: populate bsa_fwd_params from torch tensors.
void set_params_fprop(bsa_fwd_params &params,
                      int b, int seqlen_q, int seqlen_k, int h, int h_k, int d,
                      const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
                      torch::Tensor &out, torch::Tensor &lse,
                      float scale_softmax,
                      const torch::Tensor &block_indices, int block_indices_stride,
                      const torch::Tensor &block_sizes,
                      int const* q2k_block_nums_ptr,
                      int uniform_block_sparse_num, int num_m_blocks,
                      int seqlen_q_rounded, int seqlen_k_rounded) {
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = lse.data_ptr();

    params.q_batch_stride = q.stride(0); params.q_row_stride = q.stride(1); params.q_head_stride = q.stride(2);
    params.k_batch_stride = k.stride(0); params.k_row_stride = k.stride(1); params.k_head_stride = k.stride(2);
    params.v_batch_stride = v.stride(0); params.v_row_stride = v.stride(1); params.v_head_stride = v.stride(2);
    params.o_batch_stride = out.stride(0); params.o_row_stride = out.stride(1); params.o_head_stride = out.stride(2);

    params.block_indices_ptr = block_indices.data_ptr<int>();
    params.block_sizes_ptr = block_sizes.defined() && block_sizes.numel() > 0
                             ? block_sizes.data_ptr<int>() : nullptr;
    params.q2k_block_nums_ptr = q2k_block_nums_ptr;

    params.b = b; params.seqlen_q = seqlen_q; params.seqlen_k = seqlen_k; params.d = d;
    params.h = h; params.h_k = h_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.num_m_blocks = num_m_blocks;
    params.block_indices_stride = block_indices_stride;
    params.uniform_block_sparse_num = uniform_block_sparse_num;
    params.scale_softmax = scale_softmax;
    params.scale_softmax_log2 = float(scale_softmax * M_LOG2E);
}

// Entry point: extract dims, populate params, allocate output, launch.
std::tuple<torch::Tensor, torch::Tensor> bsa_fused_fwd_blk64_impl(
        torch::Tensor q, torch::Tensor k, torch::Tensor v,
        torch::Tensor q2k_block_index, int64_t block_sparse_num,
        torch::Tensor block_sizes, double softmax_scale,
        torch::Tensor q2k_block_nums)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");
    TORCH_CHECK(q.size(3) == 128, "requires D=128");

    const bool has_var_block_nums = q2k_block_nums.defined() && q2k_block_nums.numel() > 0;

    const int b = q.size(0), seqlen_q = q.size(1), h = q.size(2), d = q.size(3);
    const int seqlen_k = k.size(1);
    const int h_k = k.size(2);

    constexpr int kRows = 64, kSparseBlockSize = 64;
    constexpr int kOutputCols = 128;
    const int seqlen_q_rounded = ((seqlen_q + kRows - 1) / kRows) * kRows;
    const int seqlen_k_rounded = ((seqlen_k + kSparseBlockSize - 1) / kSparseBlockSize) * kSparseBlockSize;
    const int num_m_blocks = seqlen_q_rounded / kRows;

    // ======== Q/K passed directly in BSHD (5D TMA handles strided access) ========
    // Python interface ensures seqlen_q/seqlen_k are multiples of kRows/kSparseBlockSize.
    torch::Tensor q_contig = q.contiguous();
    torch::Tensor k_contig = k.contiguous();

    // V: sub-tile transpose (swap token↔dim within each 64×64 block).
    // Required because PV dual GEMM reduces over K direction (= tokens),
    // and K-major SMEM layout makes dim 1 contiguous.
    constexpr int kDimHalf = 64;    // kDualK / 2
    constexpr int kDimHalves = 2;
    int total_sparse_blocks = seqlen_k_rounded / kSparseBlockSize;
    torch::Tensor v_contig = v.view({b, total_sparse_blocks, kSparseBlockSize, h_k, kDimHalves, kDimHalf})
                              .permute({0, 1, 5, 3, 4, 2})
                              .reshape({b, seqlen_k_rounded, h_k, d})
                              .contiguous();

    // ======== Output (BSHD, torch::empty) ========
    auto out = torch::empty({b, seqlen_q_rounded, h, kOutputCols}, q.options());
    auto lse = torch::empty({b, h, seqlen_q_rounded},
                            torch::dtype(torch::kFloat32).device(q.device()));

    // ======== Block indices ========
    auto bi_flat = q2k_block_index.reshape({b * h, num_m_blocks, -1}).contiguous();
    int block_indices_stride = static_cast<int>(bi_flat.size(2));

    // q2k_block_nums: only reshape/materialize when user supplied a non-empty tensor.
    // Otherwise pass nullptr and let the kernel read the uniform scalar via
    // params.uniform_block_sparse_num (HasVarBlockNums=false compile-time branch).
    // bn_flat must outlive the kernel launch, so keep it in scope here.
    torch::Tensor bn_flat;
    int const* q2k_block_nums_ptr = nullptr;
    if (has_var_block_nums) {
        bn_flat = q2k_block_nums.reshape({b * h * num_m_blocks}).contiguous();
        q2k_block_nums_ptr = bn_flat.data_ptr<int>();
    }

    // ======== Populate params (FA Hopper set_params_fprop pattern) ========
    bsa_fwd_params params{};
    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, h, h_k, d,
                     q_contig, k_contig, v_contig,
                     out, lse,
                     softmax_scale,
                     bi_flat, block_indices_stride,
                     block_sizes, q2k_block_nums_ptr,
                     static_cast<int>(block_sparse_num), num_m_blocks,
                     seqlen_q_rounded, seqlen_k_rounded);

    auto stream = c10::cuda::getCurrentCUDAStream(q.device().index()).stream();
    const bool has_block_sizes = (params.block_sizes_ptr != nullptr);
    BOOL_SWITCH(has_block_sizes, HAS_BLOCK_SIZES, [&] {
        BOOL_SWITCH(has_var_block_nums, HAS_VAR_BLOCK_NUMS, [&] {
            run_bsa_fwd<128, HAS_BLOCK_SIZES, HAS_VAR_BLOCK_NUMS>(params, stream);
        });
    });

    // ======== Return BSHD output directly (Python handles slicing) ========
    return std::make_tuple(out, lse);
}

} // namespace flash

// ---- FA Hopper pattern: TORCH_LIBRARY registration ----
// Importing the .so triggers these static initializers.
extern "C" {
PyObject* PyInit_bsa_fwd_blk64_ext(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT, "bsa_fwd_blk64_ext", NULL, -1, NULL,
    };
    return PyModule_Create(&module_def);
}
}

TORCH_LIBRARY(bsa_blk64, m) {
    m.def("fwd(Tensor q, Tensor k, Tensor v, Tensor q2k_block_index, "
          "int block_sparse_num, Tensor block_sizes, float scale, "
          "Tensor q2k_block_nums) -> (Tensor, Tensor)");
}
// Note: schema uses "int" (maps to int64_t) and "float" (maps to double) in C++.

TORCH_LIBRARY_IMPL(bsa_blk64, CUDA, m) {
    m.impl("fwd", &flash::bsa_fused_fwd_blk64_impl);
}
