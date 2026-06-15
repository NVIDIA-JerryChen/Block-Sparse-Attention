/******************************************************************************
  * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
  ******************************************************************************/
// C++ API for BSA fused attention forward kernel (blk=64)
// FA Hopper pattern: set_params_fprop() + run_bsa_fwd()
//
// Layout convention: BHSD = (batch, num_heads, seqlen, head_dim) with head_dim
// stride-1 (innermost). The C++ API must not introduce host-side transposes:
// Q/K/V are consumed in the caller-provided natural BHSD layout.

#include <cmath>
#include <limits>
#include <tuple>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "bsa.h"
#include "static_switch.h"

namespace flash {

namespace {

void check_bhsd_view(torch::Tensor const& t, char const* name) {
    TORCH_CHECK(t.stride(3) == 1, name, " must have head_dim stride 1");
    TORCH_CHECK(t.stride(0) > 0 && t.stride(1) > 0 && t.stride(2) > 0,
                name, " must have positive B/H/S strides");
    TORCH_CHECK(t.stride(1) <= std::numeric_limits<int>::max()
                && t.stride(2) <= std::numeric_limits<int>::max(),
                name, " head/seqlen strides must fit int32");
}

} // namespace

// Defined in bsa_fwd_launch_template.h, instantiated in instantiations/bsa_fwd_hdim128_bf16_hbs{0,1}_hvbn{0,1}_{,clc_}sm100.cu
template<int kHeadDim, bool HasBlockSizes, bool HasVarBlockNums,
         bool UseClc, bool HasKvSplits = true>
void run_bsa_fwd(bsa_fwd_params const&, cudaStream_t);

void run_bsa_fwd_kv_split_schedule(
        int const* q2k_block_nums,
        int* split_offsets,
        int total_tiles,
        int kv_splits,
        int uniform_block_sparse_num,
        cudaStream_t stream);

// FA Hopper pattern: populate bsa_fwd_params from torch tensors.
// Logical BHSD: stride(1) is the head stride, stride(2) is the seqlen stride.
void set_params_fprop(bsa_fwd_params &params,
                      int b, int seqlen_q, int seqlen_k, int h, int h_k, int d,
                      const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
                      torch::Tensor &out, torch::Tensor &lse,
                      float scale_softmax,
                      const torch::Tensor &block_indices, int block_indices_stride,
                      const torch::Tensor &block_sizes,
                      int const* q2k_block_nums_ptr,
                      int uniform_block_sparse_num, int num_m_blocks,
                      int seqlen_k_rounded,
                      int const* split_offsets_ptr = nullptr,
                      int kv_splits = 1) {
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = lse.data_ptr();

    // Logical BHSD: axis 0=batch, 1=head, 2=seq, 3=dim (stride 1).
    params.q_batch_stride = q.stride(0);   params.q_head_stride = q.stride(1);   params.q_row_stride = q.stride(2);
    params.k_batch_stride = k.stride(0);   params.k_head_stride = k.stride(1);   params.k_row_stride = k.stride(2);
    params.v_batch_stride = v.stride(0);   params.v_head_stride = v.stride(1);   params.v_row_stride = v.stride(2);
    if (kv_splits > 1) {
        // KV-bucketed partial layout: (B, Split * H, S, D). Keeping rows
        // contiguous for each folded split/head improves the combine load path
        // while preserving a 4D TMA store descriptor.
        params.o_batch_stride = out.stride(0);
        params.o_head_stride = out.stride(1);
        params.o_row_stride = out.stride(2);
        params.o_split_stride = h * out.stride(1);

        // KV-bucketed partial LSE layout: (B, S, Split * H).
        params.lse_batch_stride = lse.stride(0);
        params.lse_row_stride = lse.stride(1);
        params.lse_head_stride = lse.stride(2);
        params.lse_split_stride = h * lse.stride(2);
    } else {
        // Final output layout: (B, H, S, D), final LSE layout: (B, H, S).
        params.o_batch_stride = out.stride(0);
        params.o_head_stride = out.stride(1);
        params.o_row_stride = out.stride(2);
        params.o_split_stride = 0;

        params.lse_batch_stride = lse.stride(0);
        params.lse_head_stride = lse.stride(1);
        params.lse_row_stride = lse.stride(2);
        params.lse_split_stride = 0;
    }

    params.block_indices_ptr = block_indices.data_ptr<int>();
    params.block_sizes_ptr = block_sizes.defined() && block_sizes.numel() > 0
                             ? block_sizes.data_ptr<int>() : nullptr;
    params.q2k_block_nums_ptr = q2k_block_nums_ptr;
    params.split_offsets_ptr = split_offsets_ptr;

    params.b = b; params.seqlen_q = seqlen_q; params.seqlen_k = seqlen_k; params.d = d;
    params.h = h; params.h_k = h_k;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.num_m_blocks = num_m_blocks;
    params.block_indices_stride = block_indices_stride;
    params.uniform_block_sparse_num = uniform_block_sparse_num;
    params.kv_splits = kv_splits;
    params.scale_softmax = scale_softmax;
    params.scale_softmax_log2 = float(scale_softmax * M_LOG2E);
}

// Entry point: logical BHSD, zero-copy for Q/K/V. The tensors may be
// non-contiguous views as long as head_dim is stride-1 and B/H/S strides are
// representable by the TMA descriptors.
std::tuple<torch::Tensor, torch::Tensor> bsa_fused_fwd_blk64_impl(
        torch::Tensor q, torch::Tensor k, torch::Tensor v,
        torch::Tensor q2k_block_index, int64_t block_sparse_num,
        torch::Tensor block_sizes, double softmax_scale,
        torch::Tensor q2k_block_nums,
        bool use_clc)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");
    TORCH_CHECK(q.size(3) == 128, "requires D=128");
    TORCH_CHECK(k.size(3) == 128 && v.size(3) == 128, "k/v require D=128");
    check_bhsd_view(q, "q");
    check_bhsd_view(k, "k");
    check_bhsd_view(v, "v");

    // BHSD: (batch, num_heads, seqlen, head_dim)
    const int b = q.size(0);
    const int h = q.size(1);
    const int seqlen_q = q.size(2);
    const int d = q.size(3);
    const int h_k = k.size(1);
    const int seqlen_k = k.size(2);

    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
                "q/k/v batch size mismatch");
    TORCH_CHECK(h == k.size(1) && h == v.size(1),
                "blk64 requires MHA: q/k/v must share num_heads (size(1))");
    TORCH_CHECK(seqlen_k == v.size(2), "k/v seqlen mismatch");

    const bool has_var_block_nums = q2k_block_nums.defined() && q2k_block_nums.numel() > 0;

    constexpr int kRows = 64, kSparseBlockSize = 64;
    constexpr int kOutputCols = 128;
    const int seqlen_k_rounded = ((seqlen_k + kSparseBlockSize - 1) / kSparseBlockSize) * kSparseBlockSize;
    const int num_m_blocks = (seqlen_q + kRows - 1) / kRows;

    // ======== Output (logical BHSD, matching Q layout) ========
    // out: actual seqlen_q. O TMA descriptor is 4D with seq as a single mode
    //   (globalDim[seq] = seqlen_q), so the last partial tile's OOB rows are
    //   silently dropped by TMA store — no host pad / allocation rounding needed.
    // lse: actual seqlen_q; kernel has a row bounds-check around the thread-level
    //   store (matches blk128 pattern).
    auto out = torch::empty_strided(
        {b, h, seqlen_q, kOutputCols},
        {q.stride(0), q.stride(1), q.stride(2), q.stride(3)},
        q.options());
    auto lse = torch::empty({b, h, seqlen_q},
                            torch::dtype(torch::kFloat32).device(q.device()));

    // ======== Block indices ========
    // Expect (B, H, num_m_blocks, max_kv) int32 contiguous. Kernel indexes linearly as
    //   idx = ((b*H + h) * num_m_blocks + m_block) * max_kv + sub
    // which matches the contiguous flat offset, so no reshape/copy needed.
    TORCH_CHECK(q2k_block_index.dim() == 4,
                "q2k_block_index must be 4D (B, H, num_m_blocks, max_kv)");
    TORCH_CHECK(q2k_block_index.size(0) == b && q2k_block_index.size(1) == h
                && q2k_block_index.size(2) == num_m_blocks,
                "q2k_block_index shape must be (B, H, num_m_blocks, max_kv)");
    TORCH_CHECK(q2k_block_index.is_contiguous(), "q2k_block_index must be contiguous");
    TORCH_CHECK(q2k_block_index.scalar_type() == torch::kInt32,
                "q2k_block_index must be int32");
    int block_indices_stride = static_cast<int>(q2k_block_index.size(3));

    // ======== Block sizes (optional) ========
    // When defined and non-empty, expect (num_kv_blocks,) int32 contiguous; kernel
    // indexes ptr_block_sizes[sparse_block_idx] directly. When empty/undefined,
    // HasBlockSizes=false compile-time branch is used and the kernel assumes
    // every sparse block is full (= kSparseBlockSize tokens).
    const bool has_block_sizes = block_sizes.defined() && block_sizes.numel() > 0;
    if (has_block_sizes) {
        const int num_kv_blocks = seqlen_k_rounded / kSparseBlockSize;
        TORCH_CHECK(block_sizes.dim() == 1,
                    "block_sizes must be 1D (num_kv_blocks,)");
        TORCH_CHECK(block_sizes.size(0) == num_kv_blocks,
                    "block_sizes size must equal num_kv_blocks = "
                    "ceil(seqlen_k / kSparseBlockSize)");
        TORCH_CHECK(block_sizes.is_contiguous(), "block_sizes must be contiguous");
        TORCH_CHECK(block_sizes.scalar_type() == torch::kInt32,
                    "block_sizes must be int32");
    }

    // q2k_block_nums: optional — when empty, kernel uses uniform_block_sparse_num scalar
    // (HasVarBlockNums=false compile-time branch). When present, expect (B, H, num_m_blocks).
    int const* q2k_block_nums_ptr = nullptr;
    if (has_var_block_nums) {
        TORCH_CHECK(q2k_block_nums.dim() == 3,
                    "q2k_block_nums must be 3D (B, H, num_m_blocks)");
        TORCH_CHECK(q2k_block_nums.size(0) == b && q2k_block_nums.size(1) == h
                    && q2k_block_nums.size(2) == num_m_blocks,
                    "q2k_block_nums shape must be (B, H, num_m_blocks)");
        TORCH_CHECK(q2k_block_nums.is_contiguous(), "q2k_block_nums must be contiguous");
        TORCH_CHECK(q2k_block_nums.scalar_type() == torch::kInt32,
                    "q2k_block_nums must be int32");
        q2k_block_nums_ptr = q2k_block_nums.data_ptr<int>();
    }

    // ======== Populate params (FA Hopper set_params_fprop pattern) ========
    bsa_fwd_params params{};
    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, h, h_k, d,
                     q, k, v,
                     out, lse,
                     softmax_scale,
                     q2k_block_index, block_indices_stride,
                     block_sizes, q2k_block_nums_ptr,
                     static_cast<int>(block_sparse_num), num_m_blocks,
                     seqlen_k_rounded);

    auto stream = c10::cuda::getCurrentCUDAStream(q.device().index()).stream();
    BOOL_SWITCH(has_block_sizes, HAS_BLOCK_SIZES, [&] {
        BOOL_SWITCH(has_var_block_nums, HAS_VAR_BLOCK_NUMS, [&] {
            BOOL_SWITCH(use_clc, USE_CLC, [&] {
                run_bsa_fwd<128, HAS_BLOCK_SIZES, HAS_VAR_BLOCK_NUMS, USE_CLC, false>(params, stream);
            });
        });
    });

    // ======== Return logical BHSD output directly ========
    return std::make_tuple(out, lse);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bsa_fused_fwd_blk64_kv_bucketed_impl(
        torch::Tensor q, torch::Tensor k, torch::Tensor v,
        torch::Tensor q2k_block_index, int64_t block_sparse_num,
        torch::Tensor block_sizes, double softmax_scale,
        torch::Tensor q2k_block_nums,
        int64_t kv_splits,
        bool use_clc)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");
    TORCH_CHECK(q.size(3) == 128, "requires D=128");
    TORCH_CHECK(k.size(3) == 128 && v.size(3) == 128, "k/v require D=128");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16
                && k.scalar_type() == torch::kBFloat16
                && v.scalar_type() == torch::kBFloat16,
                "SM100 blk64 kv-bucketed fwd requires bf16 q/k/v");
    TORCH_CHECK(kv_splits >= 1 && kv_splits <= 256,
                "kv_splits must be in [1, 256]");
    TORCH_CHECK(!use_clc,
                "SM100 blk64 kv-bucketed fwd does not support use_clc=true yet");
    check_bhsd_view(q, "q");
    check_bhsd_view(k, "k");
    check_bhsd_view(v, "v");

    const int b = q.size(0);
    const int h = q.size(1);
    const int seqlen_q = q.size(2);
    const int d = q.size(3);
    const int h_k = k.size(1);
    const int seqlen_k = k.size(2);
    const int kv_splits_i = static_cast<int>(kv_splits);

    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
                "q/k/v batch size mismatch");
    TORCH_CHECK(h == k.size(1) && h == v.size(1),
                "blk64 requires MHA: q/k/v must share num_heads (size(1))");
    TORCH_CHECK(seqlen_k == v.size(2), "k/v seqlen mismatch");

    const bool has_var_block_nums = q2k_block_nums.defined() && q2k_block_nums.numel() > 0;

    constexpr int kRows = 64, kSparseBlockSize = 64;
    constexpr int kOutputCols = 128;
    const int seqlen_k_rounded = ((seqlen_k + kSparseBlockSize - 1) / kSparseBlockSize) * kSparseBlockSize;
    const int num_m_blocks = (seqlen_q + kRows - 1) / kRows;

    TORCH_CHECK(q2k_block_index.dim() == 4,
                "q2k_block_index must be 4D (B, H, num_m_blocks, max_kv)");
    TORCH_CHECK(q2k_block_index.size(0) == b && q2k_block_index.size(1) == h
                && q2k_block_index.size(2) == num_m_blocks,
                "q2k_block_index shape must be (B, H, num_m_blocks, max_kv)");
    TORCH_CHECK(q2k_block_index.is_contiguous(), "q2k_block_index must be contiguous");
    TORCH_CHECK(q2k_block_index.scalar_type() == torch::kInt32,
                "q2k_block_index must be int32");
    int block_indices_stride = static_cast<int>(q2k_block_index.size(3));

    const bool has_block_sizes = block_sizes.defined() && block_sizes.numel() > 0;
    if (has_block_sizes) {
        const int num_kv_blocks = seqlen_k_rounded / kSparseBlockSize;
        TORCH_CHECK(block_sizes.dim() == 1,
                    "block_sizes must be 1D (num_kv_blocks,)");
        TORCH_CHECK(block_sizes.size(0) == num_kv_blocks,
                    "block_sizes size must equal num_kv_blocks = ceil(seqlen_k / kSparseBlockSize)");
        TORCH_CHECK(block_sizes.is_contiguous(), "block_sizes must be contiguous");
        TORCH_CHECK(block_sizes.scalar_type() == torch::kInt32,
                    "block_sizes must be int32");
    }

    int const* q2k_block_nums_ptr = nullptr;
    if (has_var_block_nums) {
        TORCH_CHECK(q2k_block_nums.dim() == 3,
                    "q2k_block_nums must be 3D (B, H, num_m_blocks)");
        TORCH_CHECK(q2k_block_nums.size(0) == b && q2k_block_nums.size(1) == h
                    && q2k_block_nums.size(2) == num_m_blocks,
                    "q2k_block_nums shape must be (B, H, num_m_blocks)");
        TORCH_CHECK(q2k_block_nums.is_contiguous(), "q2k_block_nums must be contiguous");
        TORCH_CHECK(q2k_block_nums.scalar_type() == torch::kInt32,
                    "q2k_block_nums must be int32");
        q2k_block_nums_ptr = q2k_block_nums.data_ptr<int>();
    }

    auto split_offsets = torch::empty(
        {b, h, num_m_blocks, kv_splits_i + 1},
        q2k_block_index.options());
    auto o_partial = torch::empty(
        {b, kv_splits_i * h, seqlen_q, kOutputCols},
        q.options());
    auto lse_partial = torch::empty_strided(
        {b, seqlen_q, kv_splits_i * h},
        {seqlen_q * kv_splits_i * h, 1, seqlen_q},
        torch::dtype(torch::kFloat32).device(q.device()));

    auto stream = c10::cuda::getCurrentCUDAStream(q.device().index()).stream();
    run_bsa_fwd_kv_split_schedule(
            q2k_block_nums_ptr,
            split_offsets.data_ptr<int>(),
            b * h * num_m_blocks,
            kv_splits_i,
            static_cast<int>(block_sparse_num),
            stream);

    bsa_fwd_params params{};
    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, h, h_k, d,
                     q, k, v,
                     o_partial, lse_partial,
                     softmax_scale,
                     q2k_block_index, block_indices_stride,
                     block_sizes, q2k_block_nums_ptr,
                     static_cast<int>(block_sparse_num), num_m_blocks,
                     seqlen_k_rounded,
                     split_offsets.data_ptr<int>(), kv_splits_i);

    BOOL_SWITCH(has_block_sizes, HAS_BLOCK_SIZES, [&] {
        BOOL_SWITCH(has_var_block_nums, HAS_VAR_BLOCK_NUMS, [&] {
            run_bsa_fwd<128, HAS_BLOCK_SIZES, HAS_VAR_BLOCK_NUMS, false>(params, stream);
        });
    });

    return std::make_tuple(o_partial, lse_partial, split_offsets);
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
          "Tensor q2k_block_nums, bool use_clc) -> (Tensor, Tensor)");
    m.def("fwd_kv_bucketed(Tensor q, Tensor k, Tensor v, Tensor q2k_block_index, "
          "int block_sparse_num, Tensor block_sizes, float scale, "
          "Tensor q2k_block_nums, int kv_splits, bool use_clc) -> (Tensor, Tensor, Tensor)");
}
// Note: schema uses "int" (maps to int64_t) and "float" (maps to double) in C++.
// Tensors must be logical BHSD: (batch, num_heads, seqlen, head_dim), with
// head_dim stride 1. B/H/S strides may be dynamic.

TORCH_LIBRARY_IMPL(bsa_blk64, CUDA, m) {
    m.impl("fwd", &flash::bsa_fused_fwd_blk64_impl);
    m.impl("fwd_kv_bucketed", &flash::bsa_fused_fwd_blk64_kv_bucketed_impl);
}
