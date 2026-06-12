/******************************************************************************
  * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
  ******************************************************************************/
// CollectiveMainloopFwd — load, mma, softmax for fused attention (Step 2: BSA-aligned)
#pragma once

#include <math_constants.h>
#include <type_traits>

#include "cutlass/arch/barrier.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/algorithm/cooperative_copy.hpp"
#include "cute/tensor.hpp"

#include "bsa.h"
#include "pipeline.hpp"
#include "utils.h"
#include "softmax.h"

namespace flash {

namespace cute = ::cute;


template <int kHeadDim_>
struct CollectiveMainloopFwd {
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim == 128,
                  "sm100 blk64 fwd only supports head_dim = 128 "
                  "(kHeadDim template kept for future D=64 redesign)");
    // ---- Element types ----
    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementAccumulator = float;

    // ---- Tile sizes ----
    static constexpr int kRows = 64;
    static constexpr int kQkN = 256;
    static constexpr int kQkK = kHeadDim;
    static constexpr int kOutputCols = kHeadDim;
    static constexpr int kDualCols = 256;
    static constexpr int kDualK = kHeadDim;

    // ---- Sparse block constants ----
    static constexpr int kSparseBlockSize = 64;   // tokens per sparse block
    static constexpr int kSparseBlocksPerKV = kDualCols / kSparseBlockSize;  // 4
    // 128B swizzle atom contig dim = 64 bf16 elems.
    //   D=128 → split dim into 2 halves of 64 (kDimHalves=2)
    //   D=64  → single 64-wide half (kDimHalves=1)
    static constexpr int kDimHalves = kHeadDim / 64;
    static constexpr int kDimHalf = kDualK / kDimHalves;   // always 64
    static constexpr int kVDimParts = kHeadDim / 64;
    static constexpr int kVDimPart = kDualK / kVDimParts;  // always 64
    static_assert(kDualK % kVDimParts == 0, "V dim split must divide kDualK");
    static_assert(kVDimParts == 1 || kVDimParts == 2,
                  "V dim split must be 1-way or 2-way for the validated MN-major sublayout mapping");
    // SMEM offsets for K sub-tile (per dim-half, per sparse sub-block in stage).
    static constexpr int kKSubStride = kSparseBlockSize * kDimHalf;
    static constexpr int kKHalfStride = kDualCols * kDimHalf;
    // ---- MMA types ----
    // QK GEMM: ss mode (Q and K both in SMEM)
    using QkTiledMma = decltype(cute::make_tiled_mma(cute::SM100_MMA_F16BF16_WS_SS_NOELECT<
            ElementA, ElementB, ElementAccumulator,
            kRows, kDualCols,
            cute::UMMA::Major::K, cute::UMMA::Major::K>{}));

    // PV GEMM: ts mode (P in TMEM, V in SMEM) — dual pattern.
    // V is MN-major (MN axis = dim, the non-reduction axis for PV). MN-major SMEM
    // has dim contig, matching natural V (dim contig at stride 1, BHSD with
    // head_dim innermost) — no host V transpose required.
    using PvTiledMma = decltype(cute::make_tiled_mma(
            cute::SM100_MMA_F16BF16_WS_TS_NOELECT<
                    ElementA, ElementB, ElementAccumulator, kRows, kDualCols,
                    cute::UMMA::Major::K, cute::UMMA::Major::MN>{}));

    using ALogicalShape = cute::Shape<cute::Int<kRows>, cute::Int<kDualK>>;

    // ---- SMEM layouts ----
    using SmemLayoutQ = decltype(flash::make_umma_k_major_layout<kRows, kQkK, 128, ElementA>());
    using SmemLayoutB = decltype(flash::make_umma_k_major_layout<kOutputCols, kDualCols, 128, ElementB>());
    // K uses K-major for QK GEMM (K of MMA = dim, dim contig in SMEM matches natural V's dim-contig layout).
    using SmemLayoutBDual = decltype(flash::make_umma_k_major_layout<kDualCols, kDualK, 128, ElementB>());
    using SmemLayoutVStandaloneSubTile = decltype(cute::coalesce(cute::tile_to_shape(
            cute::UMMA::Layout_MN_SW128_Atom<ElementB>{},
            cute::Shape<cute::Int<kVDimPart>, cute::Int<kSparseBlockSize>>{},
            cute::Step<cute::_1, cute::_2>{}), cute::Shape<cute::_1, cute::_1>{}));
    // V uses MN-major for PV GEMM (MN of MMA = dim, dim contig matches natural V —
    // no host transpose needed). Same cosize as SmemLayoutBDual, shares smem_kv stage buffer.
    using SmemLayoutVDual = decltype(cute::coalesce(cute::tile_to_shape(
            SmemLayoutVStandaloneSubTile{},
            cute::Shape<cute::Int<kDualCols>, cute::Int<kDualK>>{},
            cute::Step<cute::_1, cute::_2>{}), cute::Shape<cute::_1, cute::_1>{}));

    // ---- KV pipeline ----
    static constexpr int kKVStages = kPipelineKVStages;
    static constexpr int kKVElemsPerStage = cute::cosize_v<SmemLayoutBDual>;
    static_assert(kKVElemsPerStage == cute::cosize_v<SmemLayoutB>);
    static_assert(kKVElemsPerStage == cute::cosize_v<SmemLayoutVDual>,
                  "V MN-major layout must share stage cosize with K-major BDual");
    static constexpr int kKVTotalElems = kKVElemsPerStage * kKVStages;

    // ---- TMEM constants (2 S stages + 2 O stages) ----
    // S0/S1 are P (post-softmax) — 128 cols/stage independent of head_dim,
    //   matches kDualCols / 2 (each stage covers half the KV col span).
    // O0/O1 are PV accumulators — head_dim cols/stage. We keep stage spacing
    //   at 128 cols (matching the D=128 layout) so D=64 leaves 64 unused cols
    //   per stage; this avoids reshuffling the TMEM map for both heads.
    static constexpr uint32_t kTmemS0 = 0;
    static constexpr uint32_t kTmemS1 = 128;
    static constexpr uint32_t kTmemO0 = 256;
    static constexpr uint32_t kTmemO1 = 384;

    static constexpr int kCSpan = 128;          // P col span per stage (D-indep)
    static constexpr int kOSpan = kOutputCols;  // O col span per stage = head_dim
    static constexpr int kPackedCols = 64;

    // ---- Warp counts ----
    static constexpr int kSoftmaxWarps = 4;  // per WG
    static constexpr int kCorrWarps = 4;
    // SmStatsNotify NamedBarrier: per-warp, 32 softmax + 32 correction = 64 threads
    // BSA pattern: arrive_w_index(stage*4+warp_idx), 8 barriers total
    static constexpr int kSmStatsNotifyThreads = 64;

    // ---- Byte constants ----
    static constexpr int kQBytes = kRows * kQkK * sizeof(ElementA);
    static constexpr int kKVBytes = kDualCols * kDualK * sizeof(ElementB);
    static constexpr int kSubTileBytes = kSparseBlockSize * kDimHalf * sizeof(ElementB);  // 8KB per TMA

    // ---- Compact (64,64) K-major SMEM layout for K sparse TMA ----
    using SmemLayoutSubTile = decltype(cute::coalesce(cute::tile_to_shape(
            cute::UMMA::Layout_K_SW128_Atom<ElementB>{},
            cute::Shape<cute::Int<kSparseBlockSize>, cute::Int<kDimHalf>>{},
            cute::Step<cute::_1, cute::_2>{}), cute::Shape<cute::_1, cute::_1>{}));
    // ---- MN-major SMEM sub-tile for V per-sparse-block TMA ----
    // TMA-friendly mode builds the full V layout by tiling this standalone
    // sub-tile, so each V copy atom covers the full 64x64 box instead of
    // degenerating into many 64x1 TMA atoms.
    using SmemLayoutVSubTile = SmemLayoutVStandaloneSubTile;
    using SmemLayoutVTma = SmemLayoutVSubTile;

    // ---- TMA types ----
    // Interface is BHSD (batch, num_heads, seqlen, head_dim). All seqlen/head/batch
    // strides are runtime (from bsa_fwd_params); head_dim is always stride-1.
    //
    // Q 5D: (kRows, kQkK, H, num_m_blocks, B). mode 3 pairs with mode 0 to stride over
    //   seqlen blocks: effective seq extent = kRows * num_m_blocks.
    using ShapeQ5 = cute::Shape<cute::Int<kRows>, cute::Int<kQkK>, int, int, int>;
    using StrideQ5 = cute::Stride<int, cute::_1, int, int, int64_t>;
    // K 6D: (64_tok, 64_dim_half, 2_halves, H_k, sparse_blocks, B). K-major sub-tile
    //   (mode 1 dim_half contig, stride 1). Mode 0 tok strided by k_row_stride.
    using ShapeKV6 = cute::Shape<cute::Int<kSparseBlockSize>, cute::Int<kDimHalf>,
                                 cute::Int<kDimHalves>, int, int, int>;
    using StrideKV6 = cute::Stride<int, cute::_1,
                                   cute::Int<kDimHalf>, int, int, int64_t>;
    // V 6D (natural, no host transpose): (dim_part, 64_tok, dim_parts, H_k,
    //   sparse_blocks, B). Splitting dim into two 64-wide TMA boxes is faster
    //   than a single 128x64 box for the full MN-major PV layout.
    using ShapeV6 = cute::Shape<cute::Int<kVDimPart>, cute::Int<kSparseBlockSize>,
                                cute::Int<kVDimParts>, int, int, int>;
    using StrideV6 = cute::Stride<cute::_1, int,
                                  cute::Int<kVDimPart>, int, int, int64_t>;
    using ShapeV = ShapeV6;
    using StrideV = StrideV6;

    using TMA_Q = decltype(cute::make_tma_copy(cute::SM90_TMA_LOAD{},
            cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementA const*>(nullptr)),
                                                cute::make_layout(ShapeQ5{}, StrideQ5{})),
            SmemLayoutQ{}));
    // K: K-major sub-tile (64,64), 6D indexing.
    using TMA_K = decltype(cute::make_tma_copy(cute::SM90_TMA_LOAD{},
            cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementB const*>(nullptr)),
                                                cute::make_layout(ShapeKV6{}, StrideKV6{})),
            SmemLayoutSubTile{}));
    // V: per-sparse-block (128_dim, 64_tok) sub-tile into MN-major SMEM.
    // SmemLayoutVSubTile is a composition of the full SmemLayoutVDual so its strides
    // match full's — avoiding the atom offset mismatch for mode 1.
    using TMA_V = decltype(cute::make_tma_copy(cute::SM90_TMA_LOAD{},
            cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementB const*>(nullptr)),
                              cute::make_layout(ShapeV{}, StrideV{})),
            SmemLayoutVTma{}));

    // ---- TensorStorage ----
    struct TensorStorage {
        alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutQ>> smem_q;
        alignas(128) cute::ArrayEngine<ElementB, kKVTotalElems> smem_kv;
        alignas(16) float smem_max[256];      // 128 per softmax WG
        alignas(16) float smem_sum[256];      // 128 per softmax WG
        alignas(16) float corr_scale[2][128]; // per-stage rescale factor
    };

    // ---- Static TMA construction methods (called from run_bsa_fwd) ----
    // These read from bsa_fwd_params to build TMA descriptors and shapes.

    static TMA_Q make_tma_load_Q(bsa_fwd_params const& p) {
        using namespace cute;
        // 5D TMA: (kRows, kQkK, H, num_m_blocks, B) — strides are runtime.
        auto shape_q  = make_shape(Int<kRows>{}, Int<kQkK>{}, p.h, p.num_m_blocks, p.b);
        auto stride_q = make_stride(int(p.q_row_stride), _1{}, int(p.q_head_stride),
                                    kRows * int(p.q_row_stride), p.q_batch_stride);
        return make_tma_copy(SM90_TMA_LOAD{},
                make_tensor(make_gmem_ptr(static_cast<ElementA const*>(p.q_ptr)),
                            make_layout(shape_q, stride_q)),
                SmemLayoutQ{});
    }

    static ShapeQ5 make_shape_Q(bsa_fwd_params const& p) {
        using namespace cute;
        return make_shape(Int<kRows>{}, Int<kQkK>{}, p.h, p.num_m_blocks, p.b);
    }

    static TMA_K make_tma_load_K(bsa_fwd_params const& p) {
        using namespace cute;
        int const total_sparse_blocks = p.seqlen_k_rounded / kSparseBlockSize;
        int stride_S = int(p.k_row_stride);
        int stride_H = int(p.k_head_stride);
        // 6D: (64_tok, 64_dim_half, 2_halves, H_k, sparse_blocks, B)
        auto shape_k  = make_shape(Int<kSparseBlockSize>{}, Int<kDimHalf>{},
                                   Int<kDimHalves>{}, p.h_k,
                                   total_sparse_blocks, p.b);
        auto stride_k = make_stride(stride_S, _1{},
                                    Int<kDimHalf>{}, stride_H,
                                    kSparseBlockSize * stride_S, p.k_batch_stride);
        return make_tma_copy(SM90_TMA_LOAD{},
                make_tensor(make_gmem_ptr(static_cast<ElementB const*>(p.k_ptr)),
                            make_layout(shape_k, stride_k)),
                SmemLayoutSubTile{});
    }

    static ShapeKV6 make_shape_K(bsa_fwd_params const& p) {
        using namespace cute;
        int const total_sparse_blocks = p.seqlen_k_rounded / kSparseBlockSize;
        return make_shape(Int<kSparseBlockSize>{}, Int<kDimHalf>{},
                          Int<kDimHalves>{}, p.h_k,
                          total_sparse_blocks, p.b);
    }

    // V loads natural V (BHSD): dim contig (stride 1), per-sparse-sub-block box
    // (dim_part, 64_tok). Writes into MN-major SmemLayoutVSubTile.
    static TMA_V make_tma_load_V(bsa_fwd_params const& p) {
        using namespace cute;
        int const total_k_blocks = (p.seqlen_k_rounded + kSparseBlockSize - 1) / kSparseBlockSize;
        int stride_S = int(p.v_row_stride);   // BHSD with stride-1 dim: = D
        int stride_H = int(p.v_head_stride);
        auto shape_v  = make_shape(Int<kVDimPart>{}, Int<kSparseBlockSize>{},
                                   Int<kVDimParts>{}, p.h_k,
                                   total_k_blocks, p.b);
        auto stride_v = make_stride(_1{}, stride_S,
                                    Int<kVDimPart>{}, stride_H,
                                    kSparseBlockSize * stride_S, p.v_batch_stride);
        return make_tma_copy(SM90_TMA_LOAD{},
                make_tensor(make_gmem_ptr(static_cast<ElementB const*>(p.v_ptr)),
                            make_layout(shape_v, stride_v)),
                SmemLayoutVTma{});
    }

    static ShapeV make_shape_V(bsa_fwd_params const& p) {
        using namespace cute;
        int const total_k_blocks = (p.seqlen_k_rounded + kSparseBlockSize - 1) / kSparseBlockSize;
        return make_shape(Int<kVDimPart>{}, Int<kSparseBlockSize>{},
                          Int<kVDimParts>{}, p.h_k,
                          total_k_blocks, p.b);
    }

    // Helper: compute per-tile num_kv_blocks (pipeline iterations) and raw block count.
    // Phantom block padding: round up raw count to multiple of kSparseBlocksPerKV*2=8,
    // then divide by kSparseBlocksPerKV to get even kv_iters.
    // raw_count: actual sparse blocks (for index clamping and phantom detection).
    // HasVarBlockNums=true  → per-tile raw count from q2k_block_nums_ptr[tile_flat]
    // HasVarBlockNums=false → uniform raw count from fwd.uniform_block_sparse_num (kernel-param scalar)
    template<bool HasVarBlockNums, bool HasKvSplits = false>
    CUTLASS_DEVICE static int get_tile_num_kv_blocks(
            bsa_fwd_params const& fwd, int batch, int head, int row_tile, int split) {
        int raw_count = get_tile_raw_block_count<HasVarBlockNums, HasKvSplits>(
                fwd, batch, head, row_tile, split);
        return get_tile_num_kv_blocks_from_raw_count(raw_count);
    }

    CUTLASS_DEVICE static int get_tile_num_kv_blocks_from_raw_count(int raw_count) {
        if (raw_count <= 0) return 0;  // empty tile
        // Round up to multiple of 8 (kSparseBlocksPerKV * 2), then /4 → even kv_iters
        constexpr int kAlign = kSparseBlocksPerKV * 2;  // 8
        int padded = (raw_count + kAlign - 1) & ~(kAlign - 1);
        return padded / kSparseBlocksPerKV;
    }

    // Get the raw (unpadded) block count for a tile — used for index clamping and phantom detection.
    template<bool HasVarBlockNums, bool HasKvSplits = false>
    CUTLASS_DEVICE static int get_tile_raw_block_count(
            bsa_fwd_params const& fwd, int batch, int head, int row_tile, int split) {
        if constexpr (HasKvSplits) {
            int64_t tile_flat = (int64_t(batch) * fwd.h + head) * fwd.num_m_blocks + row_tile;
            int64_t base = tile_flat * int64_t(fwd.kv_splits + 1) + split;
            return fwd.split_offsets_ptr[base + 1] - fwd.split_offsets_ptr[base];
        }
        if constexpr (HasVarBlockNums) {
            int tile_flat = (batch * fwd.h + head) * fwd.num_m_blocks + row_tile;
            return fwd.q2k_block_nums_ptr[tile_flat];
        } else {
            return fwd.uniform_block_sparse_num;
        }
    }

    template<bool HasKvSplits = false>
    CUTLASS_DEVICE static int get_tile_split_offset(
            bsa_fwd_params const& fwd, int batch, int head, int row_tile, int split) {
        if constexpr (!HasKvSplits) {
            return 0;
        }
        int64_t tile_flat = (int64_t(batch) * fwd.h + head) * fwd.num_m_blocks + row_tile;
        return fwd.split_offsets_ptr[tile_flat * int64_t(fwd.kv_splits + 1) + split];
    }

    // ===========================================================================
    // Load warp (BSA q_stage=1 reverse order)
    // KV order: K[N-1], Q, K[N-2], {V[N-1-i], K[N-3-i]}x(N-2), V[1], V[0]
    // N >= 2 and even (guaranteed by host padding)
    // ===========================================================================

    struct LoadState {
        // Producer start state: phase=1 (example 77 make_producer_start_state)
        PipelineKVState kv_state = cutlass::make_producer_start_state<PipelineKV>();
    };

    // Prefetch TMA descriptors (called once from kernel, not per-tile).
    template<typename KernelParams>
    CUTLASS_DEVICE static void prefetch_tma_descriptors(KernelParams const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    }

    // Resolve sparse block index: indirect via block_indices or direct (dense)
    // Phantom block clamping: if logical_idx >= raw_block_count,
    // clamp to last valid index (phantom loads same data, masked in softmax).
    CUTLASS_DEVICE static int get_sparse_idx(
            int kv_block_idx, int sub, int raw_block_count,
            int const* tile_block_indices) {
        int logical_idx = kv_block_idx * kSparseBlocksPerKV + sub;
        int clamped = (logical_idx < raw_block_count) ? logical_idx
                    : max(raw_block_count - 1, 0);
        return (tile_block_indices != nullptr)
            ? tile_block_indices[clamped]
            : clamped;
    }

    CUTLASS_DEVICE static int get_interleaved_k_slot(int sub) {
        return (sub == 0) ? 0 : (sub == 1) ? 2 : (sub == 2) ? 1 : 3;
    }

    CUTLASS_DEVICE static int get_v_smem_offset(int stage_base, int sub, int h) {
        using namespace cute;
        // Linearize (sub, h) into a 2D tile coordinate inside SmemLayoutVDual
        // (kDualCols × head_dim). The M-axis fits kKVMTiles = kDualCols / kVDimPart = 4
        // tiles regardless of D; the N-axis fits kVDimParts tiles (2 for D=128, 1 for D=64).
        // For D=128 this reproduces the original ((sub&1)*kVDimParts + h, sub>>1) mapping
        // since `sub * kVDimParts + h` is identical when projected modulo 4 / divided by 4.
        constexpr int kKVMTiles = kDualCols / kVDimPart;
        int lin_idx = sub * kVDimParts + h;
        int tile_m = (lin_idx % kKVMTiles) * kVDimPart;
        int tile_n = (lin_idx / kKVMTiles) * kSparseBlockSize;
        return stage_base + int(SmemLayoutVDual{}(make_coord(tile_m, tile_n)));
    }

    CUTLASS_DEVICE static constexpr cute::TMA::CacheHintSm90 tma_kv_cache_hint() {
        return cute::TMA::CacheHintSm90::EVICT_NORMAL;
    }

    template<typename TmaLoad, typename ThrTma, typename GTile, typename STile>
    CUTLASS_DEVICE static void issue_one_v_tma(
            TmaLoad const& tma_load,
            uint64_t& tma_mbar,
            ThrTma& thr_tma,
            GTile const& g_tile,
            STile& s_tile) {
        using namespace cute;
        cute::copy(tma_load.with(tma_mbar, 0, tma_kv_cache_hint()),
                thr_tma.partition_S(g_tile),
                thr_tma.partition_D(s_tile));
    }

    // Load K: 8 TMAs per KV block (4 sparse blocks × 2 dim halves)
    // Each TMA loads (64, 64) bf16 = 8KB. 8 × 8KB = 64KB per stage.
    // SMEM offset for sub-block i, dim-half h: K: i*kKSubStride+h*kKHalfStride
    template<typename ThrTmaK, typename GKFull, typename MainloopStorage,
             typename TmaBarrier>
    CUTLASS_DEVICE static void issue_K_tmas(
            int kv_block_idx, int stage_base,
            TmaBarrier* tma_bar, auto const& params,
            ThrTmaK& thr_tma_k, GKFull& gK_full,
            MainloopStorage& ml,
            int head, int batch, int raw_block_count,
            int const* tile_block_indices) {
        using namespace cute;
        auto& tma_mbar = reinterpret_cast<uint64_t&>(*tma_bar);
        CUTLASS_PRAGMA_UNROLL
        for (int sub = 0; sub < kSparseBlocksPerKV; ++sub) {
            int sparse_idx = get_sparse_idx(kv_block_idx, sub, raw_block_count, tile_block_indices);
            int slot = get_interleaved_k_slot(sub);
            CUTLASS_PRAGMA_UNROLL
            for (int h = 0; h < kDimHalves; ++h) {
                int smem_offset = stage_base + slot * kKSubStride + h * kKHalfStride;
                auto sK_sub = make_tensor(make_smem_ptr(
                        ml.smem_kv.begin() + smem_offset), SmemLayoutSubTile{});
                cute::copy(params.tma_load_K.with(tma_mbar, 0, tma_kv_cache_hint()),
                        thr_tma_k.partition_S(gK_full(_, _, h, head, sparse_idx, batch)),
                        thr_tma_k.partition_D(sK_sub));
            }
        }
    }

    template<typename ThrTmaK, typename GKFull, typename MainloopStorage>
    CUTLASS_DEVICE static PipelineKVState load_K(
            int kv_block_idx, PipelineKVState kv_st,
            PipelineKV& pipeline_kv, auto const& params,
            ThrTmaK& thr_tma_k, GKFull& gK_full,
            MainloopStorage& ml,
            int head, int batch, int raw_block_count,
            int const* tile_block_indices) {
        using namespace cute;
        // K interleave map: sub-block -> SMEM slot. Keeps warp-col distribution balanced:
        // {0->0, 1->2, 2->1, 3->3}.
        pipeline_kv.producer_acquire(kv_st);
        auto* tma_bar = pipeline_kv.producer_get_barrier(kv_st);
        int stage_base = kv_st.index() * kKVElemsPerStage;
        issue_K_tmas(
                kv_block_idx, stage_base, tma_bar, params,
                thr_tma_k, gK_full, ml,
                head, batch, raw_block_count, tile_block_indices);
        ++kv_st;
        return kv_st;
    }

    template<typename ThrTmaV, typename GVFull, typename MainloopStorage,
             typename TmaBarrier>
    CUTLASS_DEVICE static void issue_V_tmas(
            int kv_block_idx, int stage_base,
            TmaBarrier* tma_bar, auto const& params,
            ThrTmaV& thr_tma_v, GVFull& gV_full,
            MainloopStorage& ml,
            int head, int batch, int raw_block_count,
            int const* tile_block_indices) {
        using namespace cute;
        CUTLASS_PRAGMA_UNROLL
        for (int sub = 0; sub < kSparseBlocksPerKV; ++sub) {
            int sparse_idx = get_sparse_idx(kv_block_idx, sub, raw_block_count, tile_block_indices);
            CUTLASS_PRAGMA_UNROLL
            for (int h = 0; h < kVDimParts; ++h) {
                int smem_offset = get_v_smem_offset(stage_base, sub, h);
                auto sV_sub = make_tensor(make_smem_ptr(
                        ml.smem_kv.begin() + smem_offset), SmemLayoutVTma{});
                auto& tma_mbar = reinterpret_cast<uint64_t&>(*tma_bar);
                issue_one_v_tma(params.tma_load_V, tma_mbar, thr_tma_v,
                        gV_full(_, _, h, head, sparse_idx, batch), sV_sub);
            }
        }
    }

    // Load V: 4*kVDimParts TMAs per KV block.
    // In TMA-friendly mode, SmemLayoutVDual is a blocked product of standalone
    // sub-tiles, so explicit offset + SmemLayoutVTma is both PV-compatible and
    // lets CuTe issue one full-box TMA atom per V sub-tile.
    // Sub-tile coords in full:
    //   full (256,128): x = kVDimParts*(sub_i % 2)+dim_part, y = sub_i/2.
    template<typename ThrTmaV, typename GVFull, typename MainloopStorage>
    CUTLASS_DEVICE static PipelineKVState load_V(
            int kv_block_idx, PipelineKVState kv_st,
            PipelineKV& pipeline_kv, auto const& params,
            ThrTmaV& thr_tma_v, GVFull& gV_full,
            MainloopStorage& ml,
            int head, int batch, int raw_block_count,
            int const* tile_block_indices) {
        using namespace cute;
        pipeline_kv.producer_acquire(kv_st);
        auto* tma_bar = pipeline_kv.producer_get_barrier(kv_st);
        int stage_base = kv_st.index() * kKVElemsPerStage;
        issue_V_tmas(
                kv_block_idx, stage_base, tma_bar, params,
                thr_tma_v, gV_full, ml,
                head, batch, raw_block_count, tile_block_indices);
        ++kv_st;
        return kv_st;
    }

    template<bool HasKvSplits, typename KernelParams, typename SharedStorage>
    CUTLASS_DEVICE LoadState load(
            KernelParams const& params, PipelineKV& pipeline_kv,
            SharedStorage& shared_storage,
            int head, int row_tile, int batch, int split, int num_row_tiles, int num_kv_blocks,
            int raw_block_count,  // actual sparse block count (for phantom clamping)
            LoadState state)
    {
        using namespace cute;
        auto& ml = shared_storage.tensors.mainloop;

        if (cute::elect_one_sync()) {
            auto thr_tma_q = params.tma_load_Q.get_slice(Int<0>{});
            auto thr_tma_k = params.tma_load_K.get_slice(Int<0>{});
            auto thr_tma_v = params.tma_load_V.get_slice(Int<0>{});

            Tensor gQ_full = params.tma_load_Q.get_tma_tensor(params.shape_Q);
            Tensor gK_full = params.tma_load_K.get_tma_tensor(params.shape_K);
            Tensor gV_full = params.tma_load_V.get_tma_tensor(params.shape_V);

            // Block indices for this (batch, head, row_tile)
            int const* tile_block_indices = nullptr;
            if (params.fwd.block_indices_ptr != nullptr) {
                int64_t tile_idx_flat = (int64_t(batch) * params.fwd.h + head) * num_row_tiles + row_tile;
                int split_offset = get_tile_split_offset<HasKvSplits>(
                        params.fwd, batch, head, row_tile, split);
                tile_block_indices = params.fwd.block_indices_ptr
                        + tile_idx_flat * params.fwd.block_indices_stride + split_offset;
            }

            // Load Q (one-shot TMA into persistent smem_q)
            {
                // 5D indexing: (_, _, head, row_tile, batch)
                Tensor gQ_tile = gQ_full(_, _, head, row_tile, batch);
                auto sQ = make_tensor(make_smem_ptr(ml.smem_q.begin()), SmemLayoutQ{});
                cute::set_barrier_transaction_bytes(shared_storage.pipelines.bar_q_ready, kQBytes);
                cute::copy(params.tma_load_Q.with(reinterpret_cast<uint64_t&>(shared_storage.pipelines.bar_q_ready)),
                                      thr_tma_q.partition_S(gQ_tile), thr_tma_q.partition_D(sQ));
            }

            auto kv_state = state.kv_state;

            // BSA reverse order: K[N-1], K[N-2], {V[N-1-i], K[N-3-i]}x(N-2), V[1], V[0]
            kv_state = load_K(
                    num_kv_blocks - 1, kv_state,
                    pipeline_kv, params, thr_tma_k, gK_full, ml,
                    head, batch, raw_block_count, tile_block_indices);
            kv_state = load_K(
                    num_kv_blocks - 2, kv_state,
                    pipeline_kv, params, thr_tma_k, gK_full, ml,
                    head, batch, raw_block_count, tile_block_indices);

            CUTE_NO_UNROLL
            for (int i = 0; i < num_kv_blocks - 2; ++i) {
                kv_state = load_V(num_kv_blocks - 1 - i, kv_state,
                        pipeline_kv, params, thr_tma_v, gV_full, ml,
                        head, batch, raw_block_count, tile_block_indices);
                kv_state = load_K(
                        num_kv_blocks - 3 - i, kv_state,
                        pipeline_kv, params, thr_tma_k, gK_full, ml,
                        head, batch, raw_block_count, tile_block_indices);
            }

            kv_state = load_V(1, kv_state,
                    pipeline_kv, params, thr_tma_v, gV_full, ml,
                    head, batch, raw_block_count, tile_block_indices);
            kv_state = load_V(0, kv_state,
                    pipeline_kv, params, thr_tma_v, gV_full, ml,
                    head, batch, raw_block_count, tile_block_indices);

            state.kv_state = kv_state;
        }
        return state;
    }

    // ===========================================================================
    // MMA warp (BSA q_stage=1: alternating stage 0/1, PV before QK in main loop)
    // ===========================================================================

    struct MmaState {
        PipelineKVState kv_state;  // consumer starts at phase=0 (default)
        PipeState spo_state;       // alternating stages 0→1→0→1 for producer_acquire
        PipeState pls_state_0;     // p_lastsplit stage 0 phase tracking
        PipeState pls_state_1 = PipeState(1, 0, 0);  // p_lastsplit stage 1
        int q_phase = 0;
    };

    // Split PV GEMM: issue first half tiles, wait p_lastsplit, issue remaining half.
    // Earlier SPO release (1/4 instead of 1/2 or 3/4) gives MMA earlier P access,
    // reducing pipeline wait stalls at the cost of a longer p_lastsplit wait.
    // Granularity is the 4 P fragments; fractions that round to 0 would deadlock.
    static constexpr int kSplitNumer = 1;
    static constexpr int kSplitDenom = 4;
    static_assert(kSplitDenom > 0 && kSplitNumer >= 0 && kSplitNumer <= kSplitDenom,
                  "Invalid PV split fraction");

    template <typename TiledMma, typename TensorA, typename TensorB, typename TensorFragC>
    CUTE_DEVICE static void utcmma_ts_split(
            TiledMma& tiled_mma, TensorA tA_frag, TensorB sB, TensorFragC tC_frag,
            bool clear_accum, uint32_t p_lastsplit_addr, int p_lastsplit_phase)
    {
        using namespace cute;
        tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
        auto thr_mma = tiled_mma.get_slice(_0{});
        auto sB_frag = thr_mma.partition_fragment_B(sB);
        constexpr int kTotal = decltype(size<2>(tA_frag))::value;
        constexpr int kSplitK = kTotal * kSplitNumer / kSplitDenom;
        static_assert(kSplitK > 0 && kSplitK < kTotal,
                      "PV split must map to a non-empty first and second split");

        CUTE_UNROLL
        for (int k = 0; k < kSplitK; ++k) {
            cute::gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }

        // Wait for last split of P
        wait_barrier_addr(p_lastsplit_addr, p_lastsplit_phase);

        CUTE_UNROLL
        for (int k = kSplitK; k < kTotal; ++k) {
            cute::gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
        }
    }

    template<typename SharedStorage>
    CUTLASS_DEVICE MmaState mma(
            PipelineKV& pipeline_kv, PipelineSPO& pipeline_s_p_o, PipelineOAcc& pipeline_o_acc,
            PipelinePLastSplit& pipeline_p_lastsplit,
            SharedStorage& shared_storage,
            uint32_t tmem_base, int num_kv_blocks,
            MmaState state)
    {
        using namespace cute;
        auto& ml = shared_storage.tensors.mainloop;

        const uint32_t tmem_s[2] = {tmem_base + kTmemS0, tmem_base + kTmemS1};
        const uint32_t tmem_o[2] = {tmem_base + kTmemO0, tmem_base + kTmemO1};

        QkTiledMma qk_mma;
        PvTiledMma pv_mma;
        Tensor tC_qk = partition_fragment_C(qk_mma, Shape<Int<kRows>, Int<kDualCols>>{});
        Tensor tC_pv = partition_fragment_C(pv_mma, Shape<Int<kRows>, Int<kDualCols>>{});
        Tensor tP = pv_mma.get_slice(_0{}).make_fragment_A(
                partition_shape_A(pv_mma, ALogicalShape{}));

        auto sQ = make_tensor(make_smem_ptr(ml.smem_q.begin()), SmemLayoutQ{});

        if (elect_one_sync()) {
            auto kv_state = state.kv_state;
            auto spo_state = state.spo_state;
            auto pls_state_0 = state.pls_state_0;
            auto pls_state_1 = state.pls_state_1;
            bool o_acc_s0 = false, o_acc_s1 = false;

            // Wait for Q TMA
            int q_phase = state.q_phase;
            flash::wait_barrier_addr(
                    cute::cast_smem_ptr_to_uint(&shared_storage.pipelines.bar_q_ready),
                    q_phase);
            q_phase ^= 1;
            flash::tcgen05_commit();

            // ---- Prologue: S0 = Q@K[N-1], S1 = Q@K[N-2] (N>=2 guaranteed) ----
            // mma_qk stage=0
            {
                constexpr int stage = 0;
                pipeline_kv.consumer_wait(kv_state);
                flash::tcgen05_commit();
                tC_qk.data() = tmem_s[stage];
                auto sK = make_tensor(make_smem_ptr(
                        ml.smem_kv.begin() + kv_state.index() * kKVElemsPerStage),
                        SmemLayoutBDual{});
                flash::utcmma_ss(qk_mma, sQ, sK, tC_qk, true);
                flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                        shared_storage.pipelines.spo.full_barrier_[stage]));
                flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                        shared_storage.pipelines.kv.empty_barrier_[kv_state.index()]));
                ++kv_state;
            }
            // mma_qk stage=1
            {
                constexpr int stage = 1;
                pipeline_kv.consumer_wait(kv_state);
                flash::tcgen05_commit();
                tC_qk.data() = tmem_s[stage];
                auto sK = make_tensor(make_smem_ptr(
                        ml.smem_kv.begin() + kv_state.index() * kKVElemsPerStage),
                        SmemLayoutBDual{});
                flash::utcmma_ss(qk_mma, sQ, sK, tC_qk, true);
                flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                        shared_storage.pipelines.spo.full_barrier_[stage]));
                flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                        shared_storage.pipelines.kv.empty_barrier_[kv_state.index()]));
                ++kv_state;
            }

            // ---- Main loop: pairs of {PV[stage] + QK[stage]} alternating stage 0,1 ----
            int pair_count = (num_kv_blocks - 2) / 2;
            CUTE_NO_UNROLL
            for (int i = 0; i < pair_count; ++i) {
                // ---- stage 0: PV then QK ----
                pipeline_s_p_o.producer_acquire(spo_state);
                // mma_pv stage=0
                {
                    constexpr int stage = 0;
                    pipeline_kv.consumer_wait(kv_state);
                    flash::tcgen05_commit();
                    tC_pv.data() = tmem_o[stage];
                    tP.data() = tmem_s[stage];
                    auto sV = make_tensor(make_smem_ptr(
                            ml.smem_kv.begin() + kv_state.index() * kKVElemsPerStage),
                            SmemLayoutVDual{});
                    uint32_t pls_addr = cute::cast_smem_ptr_to_uint(
                            &shared_storage.pipelines.p_lastsplit.full_barrier_[stage]);
                    utcmma_ts_split(pv_mma, tP, sV, tC_pv, !o_acc_s0,
                            pls_addr, pls_state_0.phase());
                    ++pls_state_0; ++pls_state_0;
                    flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                        shared_storage.pipelines.kv.empty_barrier_[kv_state.index()]));
                    ++kv_state;
                }
                // mma_qk stage=0
                {
                    constexpr int stage = 0;
                    pipeline_kv.consumer_wait(kv_state);
                    flash::tcgen05_commit();
                    tC_qk.data() = tmem_s[stage];
                    auto sK = make_tensor(make_smem_ptr(
                            ml.smem_kv.begin() + kv_state.index() * kKVElemsPerStage),
                            SmemLayoutBDual{});
                    flash::utcmma_ss(qk_mma, sQ, sK, tC_qk, true);
                    flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                        shared_storage.pipelines.spo.full_barrier_[stage]));
                    flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                        shared_storage.pipelines.kv.empty_barrier_[kv_state.index()]));
                    ++kv_state;
                }
                ++spo_state;
                o_acc_s0 = true;

                // ---- stage 1: PV then QK ----
                pipeline_s_p_o.producer_acquire(spo_state);
                // mma_pv stage=1
                {
                    constexpr int stage = 1;
                    pipeline_kv.consumer_wait(kv_state);
                    flash::tcgen05_commit();
                    tC_pv.data() = tmem_o[stage];
                    tP.data() = tmem_s[stage];
                    auto sV = make_tensor(make_smem_ptr(
                            ml.smem_kv.begin() + kv_state.index() * kKVElemsPerStage),
                            SmemLayoutVDual{});
                    uint32_t pls_addr = cute::cast_smem_ptr_to_uint(
                            &shared_storage.pipelines.p_lastsplit.full_barrier_[stage]);
                    utcmma_ts_split(pv_mma, tP, sV, tC_pv, !o_acc_s1,
                            pls_addr, pls_state_1.phase());
                    ++pls_state_1; ++pls_state_1;
                    flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                        shared_storage.pipelines.kv.empty_barrier_[kv_state.index()]));
                    ++kv_state;
                }
                // mma_qk stage=1
                {
                    constexpr int stage = 1;
                    pipeline_kv.consumer_wait(kv_state);
                    flash::tcgen05_commit();
                    tC_qk.data() = tmem_s[stage];
                    auto sK = make_tensor(make_smem_ptr(
                            ml.smem_kv.begin() + kv_state.index() * kKVElemsPerStage),
                            SmemLayoutBDual{});
                    flash::utcmma_ss(qk_mma, sQ, sK, tC_qk, true);
                    flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                        shared_storage.pipelines.spo.full_barrier_[stage]));
                    flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                        shared_storage.pipelines.kv.empty_barrier_[kv_state.index()]));
                    ++kv_state;
                }
                ++spo_state;
                o_acc_s1 = true;
            }

            // ---- Epilogue: 2 final PV GEMMs, signal O_acc ----
            // mma_pv stage=0 (epilogue)
            pipeline_s_p_o.producer_acquire(spo_state);
            {
                constexpr int stage = 0;
                pipeline_kv.consumer_wait(kv_state);
                flash::tcgen05_commit();
                tC_pv.data() = tmem_o[stage];
                tP.data() = tmem_s[stage];
                auto sV = make_tensor(make_smem_ptr(
                        ml.smem_kv.begin() + kv_state.index() * kKVElemsPerStage),
                        SmemLayoutVDual{});
                uint32_t pls_addr = cute::cast_smem_ptr_to_uint(
                        &shared_storage.pipelines.p_lastsplit.full_barrier_[stage]);
                utcmma_ts_split(pv_mma, tP, sV, tC_pv, !o_acc_s0,
                        pls_addr, pls_state_0.phase());
                ++pls_state_0; ++pls_state_0;
                flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                        shared_storage.pipelines.kv.empty_barrier_[kv_state.index()]));
                ++kv_state;
            }
            // UMMA arrive on OAcc full barrier (final O0 ready)
            flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                    shared_storage.pipelines.o_acc.full_barrier_[0]));
            ++spo_state;

            // mma_pv stage=1 (epilogue)
            pipeline_s_p_o.producer_acquire(spo_state);
            {
                constexpr int stage = 1;
                pipeline_kv.consumer_wait(kv_state);
                flash::tcgen05_commit();
                tC_pv.data() = tmem_o[stage];
                tP.data() = tmem_s[stage];
                auto sV = make_tensor(make_smem_ptr(
                        ml.smem_kv.begin() + kv_state.index() * kKVElemsPerStage),
                        SmemLayoutVDual{});
                uint32_t pls_addr = cute::cast_smem_ptr_to_uint(
                        &shared_storage.pipelines.p_lastsplit.full_barrier_[stage]);
                utcmma_ts_split(pv_mma, tP, sV, tC_pv, !o_acc_s1,
                        pls_addr, pls_state_1.phase());
                ++pls_state_1; ++pls_state_1;
                flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                        shared_storage.pipelines.kv.empty_barrier_[kv_state.index()]));
                ++kv_state;
            }
            // UMMA arrive on OAcc full barrier (final O1 ready)
            flash::umma_arrive(reinterpret_cast<cute::uint64_t&>(
                    shared_storage.pipelines.o_acc.full_barrier_[1]));
            ++spo_state;

            state.kv_state = kv_state;
            state.spo_state = spo_state;
            state.pls_state_0 = pls_state_0;
            state.pls_state_1 = pls_state_1;
            state.q_phase = q_phase;
        }
        return state;
    }

    static constexpr int kFrgTile = 32;
    static constexpr int kFrgCount = kCSpan / kFrgTile;

    // ===========================================================================
    // Softmax (parameterized by stage: 0 or 1)
    // Each softmax WG processes only its own stage (WG0 -> stage 0, WG1 -> stage 1).
    // Two-pass T2R: pass 1 for fmax only, pass 2 for scale+exp2+sum+bf16+R2T.
    // ===========================================================================

    struct SoftmaxState {
        PipeState spo_state;        // consumer_wait on SPO (stage from template param)
        PipeState sm_stats_state;   // producer_acquire on SmStats (stage from template param)
    };

    // R2P bitmask: keep positions < limit within 32-element chunk s.
    // Matches blk128 mask.py: r2p_bitmask_below(limit, s).
    CUTLASS_DEVICE static uint32_t r2p_bitmask_below(int limit, int s) {
        uint32_t shift = static_cast<uint32_t>(max(0, (s + 1) * 32 - limit));
        uint32_t result;
        asm("shr.b32 %0, 0xFFFFFFFF, %1;" : "=r"(result) : "r"(shift));
        return result;
    }

    // Apply R2P block-size mask to a sub-block at offset.
    // Matches blk128 mask.py: apply_block_size_mask + mask_r2p_lambda.
    template<typename Tensor>
    CUTLASS_DEVICE static void apply_block_size_mask(Tensor& tSrS, int block_size, int offset) {
        if (block_size < kSparseBlockSize) {
            constexpr int kChunks = kSparseBlockSize / 32;
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < kChunks; ++s) {
                uint32_t mask = r2p_bitmask_below(block_size, s);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 32; ++i) {
                    float val = tSrS(offset + s * 32 + i);
                    asm("{\n\t"
                        "  .reg .pred p;\n\t"
                        "  .reg .u32 tmp;\n\t"
                        "  and.b32 tmp, %1, %2;\n\t"
                        "  setp.eq.u32 p, tmp, 0;\n\t"
                        "  @p mov.f32 %0, 0fFF800000;\n\t"
                        "}" : "+f"(val) : "r"(mask), "r"(1u << i));
                    tSrS(offset + s * 32 + i) = val;
                }
            }
        }
    }

    // Single softmax iteration — matches blk128 softmax_step exactly:
    //   1. T2R load + mask
    //   2. update_row_max
    //   3. scale_subtract_rowmax  (separate FMA pass over all 128 elements)
    //   4. apply_exp2_convert     (per-fragment: exp2 in-place → bf16 convert → TMEM store)
    //   5. update_row_sum
    template<int Stage, bool IsFirst, typename SharedStorage, typename NamedBarriers>
    CUTLASS_DEVICE void softmax_step(
            int sm_idx, int sm_stats_bar,
            PipelineSPO& pipeline_s_p_o,
            PipelineSmStats& pipeline_sm_stats,
            PipelinePLastSplit& pipeline_p_lastsplit,
            SharedStorage& shared_storage,
            uint32_t tmem_s_cur, float sm_scale,
            float& row_max, float& row_sum,
            PipeState& spo_state, PipeState& sm_stats_state,
            int block_size_lo = kSparseBlockSize,
            int block_size_hi = kSparseBlockSize)
    {
        using namespace cute;
        using cutlass::arch::NamedBarrier;
        auto& ml = shared_storage.tensors.mainloop;

        // 1. Wait for S ready in TMEM
        pipeline_s_p_o.consumer_wait(spo_state);

        // 2. T2R load
        auto tSrS_t2r = cute::make_tensor<float>(cute::Shape<cute::Int<kCSpan>>{});
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < kFrgCount; ++c) {
            flash::tmem_load<kFrgTile>(tmem_s_cur + c * kFrgTile, &tSrS_t2r(c * kFrgTile));
        }

        // 2b. Block-size mask (R2P bitmask pattern, matches blk128 mask.py).
        // block_size == kSparseBlockSize makes the mask a no-op (see apply_block_size_mask);
        // block_size == 0 masks the whole sub-block (used for phantom blocks).
        apply_block_size_mask(tSrS_t2r, block_size_lo, 0);
        apply_block_size_mask(tSrS_t2r, block_size_hi, kSparseBlockSize);

        // 3. update_row_max
        float acc_scale, m_new_safe;
        update_row_max<IsFirst>(tSrS_t2r, sm_scale, row_max, acc_scale, m_new_safe);
        if constexpr (!IsFirst) {
            ml.corr_scale[Stage][sm_idx] = acc_scale;
        }

        // 4. Notify correction
        NamedBarrier::arrive(kSmStatsNotifyThreads, sm_stats_bar);

        // 5. scale_subtract_rowmax (separate FMA pass, matches blk128)
        {
            float neg_m_scaled = -m_new_safe * sm_scale;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCSpan; i += 2) {
                ffma2(tSrS_t2r(i), tSrS_t2r(i + 1), sm_scale, neg_m_scaled);
            }
        }

        // 6. apply_exp2_convert (per-fragment: exp2 → bf16 → TMEM store, matches blk128)
        {
            // B200 (sm_100a) mixes MUFU ex2.approx with FMA-pipe polynomial
            // emulation to spread exp2 work across pipes. B300 (sm_103a) has
            // enough MUFU throughput that pure ex2.approx wins, so skip the mix.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1030
            constexpr bool kUseExp2Emu = false;
#else
            constexpr bool kUseExp2Emu = true;
#endif
            constexpr int kEmuFreq = 12;
            constexpr int kEmuRes = 4;
            constexpr int kEmuStartFrg = 0;
            constexpr int kPairsPerFrg = kFrgTile / 2;

            auto tSrS_frg = cute::make_tensor(tSrS_t2r.data(),
                    cute::make_shape(cute::Int<kFrgTile>{}, cute::Int<kFrgCount>{}));

            CUTLASS_PRAGMA_UNROLL
            for (int frag = 0; frag < kFrgCount; ++frag) {
                // exp2 in-place (with partial FMA emulation on B200)
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kFrgTile; i += 2) {
                    if (!kUseExp2Emu || frag < kEmuStartFrg ||
                            frag >= kFrgCount - 1 ||
                            i % kEmuFreq < kEmuFreq - kEmuRes) {
                        tSrS_frg(i, frag) = exp2f(tSrS_frg(i, frag));
                        tSrS_frg(i + 1, frag) = exp2f(tSrS_frg(i + 1, frag));
                    } else {
                        exp2_emu2(tSrS_frg(i, frag), tSrS_frg(i + 1, frag));
                    }
                }

                // Convert to bf16 + TMEM store
                uint32_t p_frag[kPairsPerFrg];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < kPairsPerFrg; ++i) {
                    nv_bfloat162 pair = __floats2bfloat162_rn(
                            tSrS_frg(2 * i, frag), tSrS_frg(2 * i + 1, frag));
                    p_frag[i] = reinterpret_cast<uint32_t const&>(pair);
                }
                [&]<size_t... Is>(cute::index_sequence<Is...>) {
                    SM100_TMEM_STORE_32dp32b16x::copy(
                            p_frag[Is]..., tmem_s_cur + frag * kPairsPerFrg);
                }(cute::make_index_sequence<kPairsPerFrg>{});

                // split_P_arrive
                constexpr int kSplitPFragments = kFrgCount * kSplitNumer / kSplitDenom;
                static_assert(kSplitPFragments > 0 && kSplitPFragments < kFrgCount,
                              "PV split must release after a non-empty prefix of P fragments");
                if (frag + 1 == kSplitPFragments) {
                    cutlass::arch::fence_view_async_tmem_store();
                    pipeline_s_p_o.consumer_release(spo_state);
                    ++spo_state; ++spo_state;
                }
            }
        }

        // 7. All P written — signal p_lastsplit
        cutlass::arch::fence_view_async_tmem_store();
        __syncwarp();
        if (elect_one_sync()) {
            pipeline_p_lastsplit.producer_commit(PipeState(Stage, 0, 0));
        }

        // ---- 8. update_row_sum (blk128: softmax.update_row_sum) ----
        update_row_sum<IsFirst>(tSrS_t2r, acc_scale, row_sum);

        // ---- 9. Acquire the stats slot for the next iteration. Do this after
        // local row_sum work so any back-pressure is less exposed on the critical path.
        pipeline_sm_stats.producer_acquire(sm_stats_state);
        // Advance sm_stats_state by 2: stay on same Stage, flip phase
        ++sm_stats_state; ++sm_stats_state;
    }

    // Softmax outer loop (BSA: softmax_loop).
    // Stage is compile-time: enables constant folding for TMEM addresses, barrier IDs.
    template<int Stage, bool HasBlockSizes,
             typename SharedStorage, typename NamedBarriers>
    CUTLASS_DEVICE SoftmaxState softmax(
            PipelineSPO& pipeline_s_p_o,
            PipelineSmStats& pipeline_sm_stats,
            PipelinePLastSplit& pipeline_p_lastsplit,
            SharedStorage& shared_storage,
            uint32_t tmem_base, float sm_scale, int num_kv_blocks,
            SoftmaxState state,
            int const* tile_block_indices = nullptr,
            int const* ptr_block_sizes = nullptr,
            int raw_block_count = 0)
    {
        using namespace cute;
        using cutlass::arch::NamedBarrier;
        auto& ml = shared_storage.tensors.mainloop;

        constexpr uint32_t kTmemS = (Stage == 0) ? kTmemS0 : kTmemS1;
        const uint32_t tmem_s_cur = tmem_base + kTmemS;
        const int sm_idx = threadIdx.x - Stage * kSoftmaxWarps * 32;
        const int warp_in_wg = sm_idx / 32;
        const int sm_stats_bar = NamedBarriers::SmStatsNotify + Stage * kSoftmaxWarps + warp_in_wg;

        float row_max = -CUDART_INF_F;
        float row_sum = 0.0f;
        PipeState spo_state = state.spo_state;
        PipeState sm_stats_state = state.sm_stats_state;
        int wg_count = num_kv_blocks / 2;

        // Block-size lookup: slot → sub mapping (inverse of kInterleavedSlot)
        // Warp-col 0 (warps 0,1): TMEM cols 0-127 → subs 0,2
        // Warp-col 1 (warps 2,3): TMEM cols 128-255 → subs 1,3
        // sub_lo = warp_col, sub_hi = warp_col + 2 (avoids array indexing → no LDL)
        const int warp_col = warp_in_wg / 2;

        // HasBlockSizes=true:  look up per-block size via ptr_block_sizes; phantom → 0.
        // HasBlockSizes=false: per-block size uniformly kSparseBlockSize; phantom → 0
        //                     (mask still required so padded-out tail blocks contribute 0).
        auto get_block_sizes = [&] (int k, int& bs_lo, int& bs_hi) {
            int kv_block = num_kv_blocks - 1 - (2 * k + Stage);
            int logical_lo = kv_block * kSparseBlocksPerKV + warp_col;
            int logical_hi = kv_block * kSparseBlocksPerKV + warp_col + 2;
            if constexpr (HasBlockSizes) {
                int clamped_lo = (logical_lo < raw_block_count) ? logical_lo : max(raw_block_count - 1, 0);
                int clamped_hi = (logical_hi < raw_block_count) ? logical_hi : max(raw_block_count - 1, 0);
                int bi_lo = tile_block_indices[clamped_lo];
                int bi_hi = tile_block_indices[clamped_hi];
                bs_lo = (logical_lo < raw_block_count) ? ptr_block_sizes[bi_lo] : 0;
                bs_hi = (logical_hi < raw_block_count) ? ptr_block_sizes[bi_hi] : 0;
            } else {
                bs_lo = (logical_lo < raw_block_count) ? kSparseBlockSize : 0;
                bs_hi = (logical_hi < raw_block_count) ? kSparseBlockSize : 0;
            }
        };

        // BSA: acquire before loop
        pipeline_sm_stats.producer_acquire(sm_stats_state);
        // Advance sm_stats_state by 2: stay on same Stage, flip phase
        ++sm_stats_state; ++sm_stats_state;

        // BSA: 1st block peeled (IsFirst=true), remaining blocks in loop (IsFirst=false)
        {
            int bs_lo, bs_hi;
            get_block_sizes(0, bs_lo, bs_hi);
            softmax_step<Stage, /*IsFirst=*/true, SharedStorage, NamedBarriers>(
                sm_idx, sm_stats_bar,
                pipeline_s_p_o, pipeline_sm_stats, pipeline_p_lastsplit, shared_storage,
                tmem_s_cur, sm_scale, row_max, row_sum, spo_state, sm_stats_state,
                bs_lo, bs_hi);
        }

        CUTE_NO_UNROLL
        for (int k = 1; k < wg_count; ++k) {
            int bs_lo, bs_hi;
            get_block_sizes(k, bs_lo, bs_hi);
            softmax_step<Stage, /*IsFirst=*/false, SharedStorage, NamedBarriers>(
                sm_idx, sm_stats_bar,
                pipeline_s_p_o, pipeline_sm_stats, pipeline_p_lastsplit, shared_storage,
                tmem_s_cur, sm_scale, row_max, row_sum, spo_state, sm_stats_state,
                bs_lo, bs_hi);
        }

        // Write final stats for correction combine
        ml.smem_sum[Stage * 128 + sm_idx] = row_sum;
        ml.smem_max[Stage * 128 + sm_idx] = row_max;
        __threadfence_block();
        NamedBarrier::arrive(kSmStatsNotifyThreads, sm_stats_bar);
        state.spo_state = spo_state;
        state.sm_stats_state = sm_stats_state;
        return state;
    }

    // ===========================================================================
    // Correction: indexed rescale + final combine (Correction warps 8-11)
    // Moved from epilogue_fwd.hpp — accesses both ml.* and el.* via SharedStorage union.
    // ===========================================================================

    // ---- SMEM layout for O (used by correction_combine for sO writes) ----
    using SmemLayoutO = decltype(cute::coalesce(cute::tile_to_shape(
            cute::UMMA::Layout_K_SW128_Atom<ElementA>{},
            cute::Shape<cute::Int<kRows>, cute::Int<kOutputCols>>{},
            cute::Step<cute::_1, cute::_2>{}), cute::Shape<cute::_1, cute::_1>{}));

    // ---- Chunk size for TMEM loads in correction/combine (O path) ----
    // Uses kOSpan (= head_dim) so D=64 loads 2 chunks vs D=128's 4 chunks.
    static constexpr int kChunk = 32;
    static constexpr int kNumChunks = kOSpan / kChunk;

    struct CorrState {
        PipeState o_acc_state_0;                   // {0, 0, 0} default — OAcc consumer_wait stage 0
        PipeState o_acc_state_1 = PipeState(1, 0, 0); // OAcc consumer_wait stage 1
        PipeState o_epi_state = PipeState(0, 1, 0); // producer start: phase=1 (no prefill, matches blk128)
    };

    CUTLASS_DEVICE static void correction_rescale(
            uint32_t tmem_o, float rescale, int corr_idx)
    {
        using namespace cute;
        unsigned should_rescale = __ballot_sync(0xFFFFFFFF, rescale < 1.0f);
        if (should_rescale) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < kNumChunks; ++c) {
                auto tOrO = cute::make_tensor<float>(cute::Shape<cute::Int<kChunk>>{});
                flash::tmem_load<kChunk>(tmem_o + c * kChunk, &tOrO(0));
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < kChunk; j += 2)
                    fmul2(tOrO(j), tOrO(j + 1), rescale);
                [&]<size_t... Is>(cute::index_sequence<Is...>) {
                    SM100_TMEM_STORE_32dp32b32x::copy(
                            reinterpret_cast<uint32_t const&>(tOrO(Is))...,
                            tmem_o + c * kChunk);
                }(cute::make_index_sequence<kChunk>{});
            }
        }
    }

    template<typename SmemTensorO, typename EpiStorage, typename NamedBarriers>
    CUTLASS_DEVICE static void correction_combine(
            uint32_t tmem_o0, uint32_t tmem_o1,
            float my_scale0, float my_scale1,
            int corr_warp, int lane_idx,
            uint32_t reduce_mbar_addr, int reduce_mbar_phase,
            EpiStorage& el, SmemTensorO& sO)
    {
        using namespace cute;

        // Pass 1: each warp computes weighted O and writes ALL chunks to OWN slot
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < kNumChunks; ++c) {
            auto tOrO0 = make_tensor<float>(Shape<Int<kChunk>>{});
            auto tOrO1 = make_tensor<float>(Shape<Int<kChunk>>{});
            flash::tmem_load<kChunk>(tmem_o0 + c * kChunk, &tOrO0(0));
            flash::tmem_load<kChunk>(tmem_o1 + c * kChunk, &tOrO1(0));

            auto tOrO_combined = make_tensor<float>(Shape<Int<kChunk>>{});
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kChunk; j += 2) {
                float scaled_o0 = tOrO0(j), scaled_o1 = tOrO0(j + 1);
                fmul2(scaled_o0, scaled_o1, my_scale0);
                float partner_o0 = tOrO1(j), partner_o1 = tOrO1(j + 1);
                fmul2(partner_o0, partner_o1, my_scale1);
                fadd2(scaled_o0, scaled_o1, partner_o0, partner_o1);
                tOrO_combined(j) = scaled_o0;
                tOrO_combined(j + 1) = scaled_o1;
            }

            // Write to OWN exchange slot
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kChunk / 4; ++i) {
                flash::smem_store_float4(
                        &el.o_exchange[corr_warp][c * 32 * kChunk + i * 32 * 4 + lane_idx * 4],
                        *reinterpret_cast<float4*>(&tOrO_combined(i * 4)));
            }
        }

        // Single barrier: all warps' exchange writes visible.
        // mbarrier instead of NamedBarrier — avoids the Reduce_02/Reduce_13
        // hw-id collision with SmStatsNotify; see reduce_mbar comment in
        // bsa_fwd_kernel_sm100.h.
        flash::mbar_arrive_and_wait(reduce_mbar_addr, reduce_mbar_phase);

        // Pass 2: warps 0,1 read own + partner exchange data → add → bf16 → sO
        {
            const int out_row = (corr_warp & 1) * 32 + lane_idx;
            const int partner_warp = corr_warp ^ 2;

            if (corr_warp < 2 && out_row < kRows) {
                CUTLASS_PRAGMA_UNROLL
                for (int c = 0; c < kNumChunks; ++c) {
                    auto tOrO_final = make_tensor<float>(Shape<Int<kChunk>>{});
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < kChunk / 4; ++i) {
                        const int off = c * 32 * kChunk + i * 32 * 4 + lane_idx * 4;
                        float4 own = flash::smem_load_float4(&el.o_exchange[corr_warp][off]);
                        float4 partner = flash::smem_load_float4(&el.o_exchange[partner_warp][off]);
                        float2* dst = reinterpret_cast<float2*>(&tOrO_final(i * 4));
                        float2 const* a = reinterpret_cast<float2 const*>(&own);
                        float2 const* b = reinterpret_cast<float2 const*>(&partner);
                        dst[0] = flash::float2_add(a[0], b[0]);
                        dst[1] = flash::float2_add(a[1], b[1]);
                    }

                    // Convert fp32 → bf16 and write to sO via STS.128
                    const int out_col_base = c * kChunk;
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < kChunk; j += 8) {
                        nv_bfloat162 p0 = __floats2bfloat162_rn(tOrO_final(j + 0), tOrO_final(j + 1));
                        nv_bfloat162 p1 = __floats2bfloat162_rn(tOrO_final(j + 2), tOrO_final(j + 3));
                        nv_bfloat162 p2 = __floats2bfloat162_rn(tOrO_final(j + 4), tOrO_final(j + 5));
                        nv_bfloat162 p3 = __floats2bfloat162_rn(tOrO_final(j + 6), tOrO_final(j + 7));
                        uint32_t addr = cute::cast_smem_ptr_to_uint(&sO(out_row, out_col_base + j));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n"
                            : : "r"(addr),
                                "r"(reinterpret_cast<uint32_t const&>(p0)),
                                "r"(reinterpret_cast<uint32_t const&>(p1)),
                                "r"(reinterpret_cast<uint32_t const&>(p2)),
                                "r"(reinterpret_cast<uint32_t const&>(p3)));
                    }
                }
            }
        }
    }

    // ptr_LSE_tile: base pointer for this tile's LSE output. The caller
    // pre-computes the batch/head/split/m_block offset and passes row stride.
    // lse_valid_rows: number of rows to write (= seqlen_q - m_block*kRows, clamped to [0, kRows]).
    //   Writes are guarded so tensor can be allocated at actual seqlen_q (not rounded).
    template<bool HasKvSplits, typename SharedStorage, typename NamedBarriers>
    CUTLASS_DEVICE CorrState correction(
            float sm_scale_log2,
            PipelineSPO& pipeline_s_p_o, PipelineSmStats& pipeline_sm_stats,
            PipelineOAcc& pipeline_o_acc,
            PipelineOEpi& pipeline_o_epi,
            SharedStorage& shared_storage,
            uint32_t tmem_base, int num_kv_blocks,
            CorrState corr_state,
            float* ptr_LSE_tile = nullptr, int lse_valid_rows = 0,
            int64_t lse_row_stride = 1)
    {
        using namespace cute;
        using cutlass::arch::NamedBarrier;
        auto& ml = shared_storage.tensors.mainloop;
        auto& el = shared_storage.tensors.epilogue;

        const uint32_t tmem_o0 = tmem_base + kTmemO0;
        const uint32_t tmem_o1 = tmem_base + kTmemO1;
        const int warp_idx = threadIdx.x / 32;
        const int lane_idx = threadIdx.x % 32;
        const int corr_warp = warp_idx - 8;       // 0..3
        const int corr_idx = threadIdx.x - 8 * 32; // 0..127

        // Static PipeState helpers for stage-indexed release (phase not needed for arrive)
        PipeState const st0(0, 0, 0);
        PipeState const st1(1, 0, 0);

        // ---- (a) Skip first pair: no rescale needed (BSA pattern) ----
        pipeline_s_p_o.consumer_release(st0);
        pipeline_s_p_o.consumer_release(st1);
        NamedBarrier::arrive_and_wait(kSmStatsNotifyThreads, NamedBarriers::SmStatsNotify + 0 * kCorrWarps + corr_warp);
        pipeline_sm_stats.consumer_release(st0);
        NamedBarrier::arrive_and_wait(kSmStatsNotifyThreads, NamedBarriers::SmStatsNotify + 1 * kCorrWarps + corr_warp);

        // ---- (b) Paired rescale loop (BSA: seqlen_corr_loop_steps) ----
        int pair_count = (num_kv_blocks - 2) / 2;
        CUTE_NO_UNROLL
        for (int i = 0; i < pair_count; ++i) {
            // Stage 0: rescale O0
            {
                NamedBarrier::arrive_and_wait(kSmStatsNotifyThreads, NamedBarriers::SmStatsNotify + 0 * kCorrWarps + corr_warp);
                correction_rescale(tmem_o0, ml.corr_scale[0][corr_idx], corr_idx);
                pipeline_s_p_o.consumer_release(st0);
                pipeline_sm_stats.consumer_release(st1);
            }
            // Stage 1: rescale O1
            {
                NamedBarrier::arrive_and_wait(kSmStatsNotifyThreads, NamedBarriers::SmStatsNotify + 1 * kCorrWarps + corr_warp);
                correction_rescale(tmem_o1, ml.corr_scale[1][corr_idx], corr_idx);
                pipeline_s_p_o.consumer_release(st1);
                pipeline_sm_stats.consumer_release(st0);
            }
        }

        // BSA: post-loop release for stage 1
        pipeline_sm_stats.consumer_release(st1);

        // ---- (c) Read final stats (BSA: sm_stats_barrier for both stages) ----
        NamedBarrier::arrive_and_wait(kSmStatsNotifyThreads, NamedBarriers::SmStatsNotify + 0 * kCorrWarps + corr_warp);
        float row_sum0 = ml.smem_sum[0 * 128 + corr_idx];
        float row_max0 = ml.smem_max[0 * 128 + corr_idx];
        pipeline_sm_stats.consumer_release(st0);

        NamedBarrier::arrive_and_wait(kSmStatsNotifyThreads, NamedBarriers::SmStatsNotify + 1 * kCorrWarps + corr_warp);
        float row_sum1 = ml.smem_sum[1 * 128 + corr_idx];
        float row_max1 = ml.smem_max[1 * 128 + corr_idx];
        pipeline_sm_stats.consumer_release(st1);

        // ---- (d) Compute cross-stage combine scales ----
        float rm0 = (row_sum0 > 0.0f) ? row_max0 : -CUDART_INF_F;
        float rm1 = (row_sum1 > 0.0f) ? row_max1 : -CUDART_INF_F;
        float max_combined = fmaxf(rm0, rm1);
        float max_safe = (max_combined > -CUDART_INF_F) ? max_combined : 0.0f;
        float scale0 = (row_sum0 > 0.0f) ? exp2f((rm0 - max_safe) * sm_scale_log2) : 0.0f;
        float scale1 = (row_sum1 > 0.0f) ? exp2f((rm1 - max_safe) * sm_scale_log2) : 0.0f;
        float my_sum = row_sum0 * scale0 + row_sum1 * scale1;
        float my_max = max_safe;

        // ---- (e) Wait for final O from MMA ----
        pipeline_o_acc.consumer_wait(corr_state.o_acc_state_0);
        flash::tcgen05_commit();
        pipeline_o_acc.consumer_wait(corr_state.o_acc_state_1);
        flash::tcgen05_commit();
        // Advance OAcc states by 2 (stay on same stage, flip phase)
        ++corr_state.o_acc_state_0; ++corr_state.o_acc_state_0;
        ++corr_state.o_acc_state_1; ++corr_state.o_acc_state_1;

        // ---- (f) Warp-pair stats exchange + combine weight ----
        auto sO = make_tensor(make_smem_ptr(el.sO.begin()), SmemLayoutO{});
        // mbarrier slot 0 ↔ warp pair (0,2), slot 1 ↔ warp pair (1,3).
        // Replaces the Reduce_02/Reduce_13 NamedBarriers — see
        // SharedStorage::reduce_mbar comment for the hw-id collision
        // those NamedBarriers had with SmStatsNotify.
        const int reduce_mbar_idx = corr_warp & 1;
        const uint32_t reduce_mbar_addr = cute::cast_smem_ptr_to_uint(
                &shared_storage.reduce_mbar[reduce_mbar_idx]);

        float my_weight;
        {
            el.o_staging[corr_warp ^ 2][lane_idx * 2 + 0] = my_sum;
            el.o_staging[corr_warp ^ 2][lane_idx * 2 + 1] = my_max;
            flash::mbar_arrive_and_wait(reduce_mbar_addr, /*phase=*/0);

            float partner_sum = el.o_staging[corr_warp][lane_idx * 2 + 0];
            float partner_max = el.o_staging[corr_warp][lane_idx * 2 + 1];

            float max_total = fmaxf(my_max, partner_max);
            float max_total_safe = (max_total > -CUDART_INF_F) ? max_total : 0.0f;
            float my_rescale = (my_sum > 0.0f) ? exp2f((my_max - max_total_safe) * sm_scale_log2) : 0.0f;
            float partner_rescale = (partner_sum > 0.0f) ? exp2f((partner_max - max_total_safe) * sm_scale_log2) : 0.0f;
            float sum_total = my_sum * my_rescale + partner_sum * partner_rescale;
            float inv_sum_total = (sum_total > 0.0f) ? __frcp_rn(sum_total) : 0.0f;  // BSA: rcp_approx
            my_weight = my_rescale * inv_sum_total;

            // ---- Write LSE to global memory (one warp per warp-pair) ----
            // Bounded by lse_valid_rows so LSE tensor can be sized at actual seqlen_q
            // (not rounded). Matches blk128's pattern in flash_fwd_sm100.py:1942.
            if (ptr_LSE_tile != nullptr && corr_warp < 2) {
                int out_row = (corr_warp & 1) * 32 + lane_idx;
                if (out_row < kRows && out_row < lse_valid_rows) {
                    float lse;
                    if (sum_total > 0.0f) {
                        lse = (max_total_safe * sm_scale_log2 + log2f(sum_total))
                              * 0.6931471805599453f;  // LN2
                    } else {
                        lse = -CUDART_INF_F;
                    }
                    if constexpr (HasKvSplits) {
                        ptr_LSE_tile[int64_t(out_row) * lse_row_stride] = lse;
                    } else {
                        ptr_LSE_tile[out_row] = lse;
                    }
                }
            }
        }

        // ---- (g) Wait for sO slot free, then combine (blk128: producer_acquire before combine) ----
        pipeline_o_epi.producer_acquire(corr_state.o_epi_state);
        float my_scale0 = scale0 * my_weight;
        float my_scale1 = scale1 * my_weight;
        correction_combine<decltype(sO), decltype(el), NamedBarriers>(
                tmem_o0, tmem_o1, my_scale0, my_scale1,
                corr_warp, lane_idx,
                reduce_mbar_addr, /*reduce_mbar_phase=*/1,
                el, sO);

        // ---- (h) Fence + signal epilogue warp ----
        cutlass::arch::fence_view_async_shared();
        pipeline_o_epi.producer_commit(corr_state.o_epi_state);
        // Advance OEpi state by 2: stay on stage 0, flip phase
        ++corr_state.o_epi_state; ++corr_state.o_epi_state;
        return corr_state;
    }
};

} // namespace flash
