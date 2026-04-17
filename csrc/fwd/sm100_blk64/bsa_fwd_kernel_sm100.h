/******************************************************************************
  * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
  ******************************************************************************/
// FusedAttnFwdSm100 — kernel class with SharedStorage, operator(), warp dispatch
// 4 WG (512 threads) — 2 Softmax WGs + Correction WG + MMA/Load/Epi WG
#pragma once

#ifndef CUDA_CTA_RECONFIG_ACTIVATED
#define CUDA_CTA_RECONFIG_ACTIVATED 1
#endif

#include <math_constants.h>

#include "cutlass/arch/barrier.h"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/tensor.hpp"

#include "cutlass/arch/reg_reconfig.h"
#include "utils.h"

#include "bsa.h"
#include "pipeline.hpp"
#include "tile_scheduler.hpp"
#include "mainloop_fwd_sm100.hpp"
#include "epilogue_fwd_sm100.hpp"

namespace flash {

namespace cute = ::cute;

#define CUTLASS_ARCH_MMA_SM100_SUPPORTED 1

template<uint32_t N>
__device__ __forceinline__ void warpgroup_reg_set() {
    if constexpr (N < 128) {
        cutlass::arch::warpgroup_reg_dealloc<N>();
    } else {
        cutlass::arch::warpgroup_reg_alloc<N>();
    }
}

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template<typename CollectiveMainloop_, typename CollectiveEpilogue_,
         typename TileScheduler_, bool HasBlockSizes>
struct FusedAttnFwdSm100 {
    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;
    using TileScheduler = TileScheduler_;

    using ElementA = typename CollectiveMainloop::ElementA;
    using ElementB = typename CollectiveMainloop::ElementB;
    using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;

    static constexpr int kRows = CollectiveMainloop::kRows;
    static constexpr int kQkK = CollectiveMainloop::kQkK;
    static constexpr int kOutputCols = CollectiveMainloop::kOutputCols;
    static constexpr int kDualCols = CollectiveMainloop::kDualCols;
    static constexpr int kQkN = CollectiveMainloop::kQkN;

    static constexpr int kThreads = 512;
    static constexpr int kMmaWarp = 12;
    static constexpr int kEpiWarp = 13;
    static constexpr int kLoadWarp = 14;

    static constexpr int kSoftmaxWarps = CollectiveMainloop::kSoftmaxWarps;
    static constexpr int kCorrWarps = CollectiveMainloop::kCorrWarps;

    static constexpr int kRegsSoftmax = 184;
    static constexpr int kRegsCorrection = 88;
    static constexpr int kRegsOther = 56;

    enum NamedBarriers : int {
        SmStatsNotify = 0,
        Reduce_02 = 4,
        Reduce_13 = 5,
    };

    // ---- SharedStorage ----
    struct SharedStorage {
        struct TensorStorage {
            union {
                typename CollectiveMainloop::TensorStorage mainloop;
                typename CollectiveEpilogue::TensorStorage epilogue;
            };
        } tensors;

        struct PipelineStorage {
            PipelineKV::SharedStorage kv;
            PipelineSPO::SharedStorage spo;
            PipelineOAcc::SharedStorage o_acc;
            PipelineSmStats::SharedStorage sm_stats;
            PipelineOEpi::SharedStorage o_epi;
            PipelinePLastSplit::SharedStorage p_lastsplit;
            alignas(16) cute::uint64_t bar_q_ready;
        } pipelines;

        alignas(16) cute::uint32_t tmem_base_ptr;
        alignas(4) int tmem_ready;
        // mbarrier replacements for Reduce_02 / Reduce_13 NamedBarriers.
        // On SM103a (B300, CUDA 13.2) ptxas always emits BAR.SYNC as
        // BAR.SYNC.DEFER_BLOCKING; the scoreboard slot that variant registers
        // is not reliably released, so a downstream scoreboard-dependent
        // instruction (a constant-bank LDCU in correction rescale) can hang
        // forever. mbarrier has no "defer" counterpart.
        alignas(16) cute::uint64_t reduce_mbar[2];
    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);
    static_assert(SharedStorageSize <= 228 * 1024, "SharedStorage exceeds SM100 228KB SMEM limit");

    // ---- Flat Params (FA Hopper pattern: bsa_fwd_params + TMA descriptors) ----
    struct Params {
        bsa_fwd_params fwd;

        typename CollectiveMainloop::TMA_Q tma_load_Q;
        typename CollectiveMainloop::TMA_KV tma_load_K;
        typename CollectiveMainloop::TMA_KV tma_load_V;
        typename CollectiveEpilogue::TMA_O tma_store_O;

        typename CollectiveMainloop::ShapeQ5 shape_Q;
        typename CollectiveMainloop::ShapeKV5 shape_K;
        typename CollectiveMainloop::ShapeKV5 shape_V;
        typename CollectiveEpilogue::ShapeO5 shape_O;
    };

    static dim3 get_grid_shape(Params const& params) {
        return TileScheduler::get_grid_shape(params.fwd.num_m_blocks, params.fwd.h, params.fwd.b);
    }

    static dim3 get_block_shape() { return dim3(kThreads, 1, 1); }

    // ---- operator(): pipeline init, warp dispatch (single tile) ----
    CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
        using namespace cute;
        using cutlass::arch::NamedBarrier;

        auto& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        const int warp_idx = threadIdx.x / 32;
        const int lane_idx = threadIdx.x % 32;
        const int global_num_kv_blocks = params.fwd.num_kv_iters;

        TileScheduler tile_sched;
        auto work = tile_sched.get_initial_work();

        // ======== Phase 1: Pipeline & barrier init ========
        int lane_predicate = cute::elect_one_sync();
        PipelineKV::Params pipeline_kv_params;
        pipeline_kv_params.transaction_bytes = CollectiveMainloop::kKVBytes;
        pipeline_kv_params.role = (warp_idx == kLoadWarp)
            ? PipelineKV::ThreadCategory::Producer
            : (warp_idx == kMmaWarp)
                ? PipelineKV::ThreadCategory::Consumer
                : PipelineKV::ThreadCategory::NonParticipant;
        pipeline_kv_params.is_leader = lane_predicate && (warp_idx == kLoadWarp);
        pipeline_kv_params.num_consumers = 1;

        PipelineKV pipeline_kv(shared_storage.pipelines.kv, pipeline_kv_params,
                               cute::Shape<cute::_1, cute::_1, cute::_1>{},
                               cute::false_type{}, cute::false_type{});

        PipelineSPO::Params spo_params;
        spo_params.producer_arv_count = 1;
        spo_params.consumer_arv_count = 256;
        spo_params.role = PipelineSPO::ThreadCategory::ProducerConsumer;

        PipelineOAcc::Params oacc_params;
        oacc_params.producer_arv_count = 1;
        oacc_params.consumer_arv_count = 1;
        oacc_params.role = PipelineOAcc::ThreadCategory::ProducerConsumer;

        PipelineSmStats::Params smstats_params;
        smstats_params.producer_arv_count = 128;
        smstats_params.consumer_arv_count = 128;
        smstats_params.role = PipelineSmStats::ThreadCategory::ProducerConsumer;

        PipelineOEpi::Params oepi_params;
        oepi_params.producer_arv_count = 128;
        oepi_params.consumer_arv_count = 1;
        oepi_params.role = PipelineOEpi::ThreadCategory::ProducerConsumer;

        PipelinePLastSplit::Params pls_params;
        pls_params.producer_arv_count = 4;
        pls_params.consumer_arv_count = 1;
        pls_params.role = PipelinePLastSplit::ThreadCategory::ProducerConsumer;

        PipelineSPO     pipeline_s_p_o(shared_storage.pipelines.spo, spo_params, cute::false_type{});
        PipelineOAcc    pipeline_o_acc(shared_storage.pipelines.o_acc, oacc_params, cute::false_type{});
        PipelineSmStats pipeline_sm_stats(shared_storage.pipelines.sm_stats, smstats_params, cute::false_type{});
        PipelineOEpi    pipeline_o_epi(shared_storage.pipelines.o_epi, oepi_params, cute::false_type{});
        PipelinePLastSplit pipeline_p_lastsplit(shared_storage.pipelines.p_lastsplit, pls_params, cute::false_type{});

        if (warp_idx == 0) {
            PipelineKV::init_barriers(shared_storage.pipelines.kv, pipeline_kv_params,
                                     cute::Shape<cute::_1, cute::_1, cute::_1>{});
            pipeline_s_p_o.init_barriers(shared_storage.pipelines.spo, spo_params);
            pipeline_o_acc.init_barriers(shared_storage.pipelines.o_acc, oacc_params);
            pipeline_sm_stats.init_barriers(shared_storage.pipelines.sm_stats, smstats_params);
            pipeline_o_epi.init_barriers(shared_storage.pipelines.o_epi, oepi_params);
            pipeline_p_lastsplit.init_barriers(shared_storage.pipelines.p_lastsplit, pls_params);
            if (lane_predicate) {
                cute::initialize_barrier(shared_storage.pipelines.bar_q_ready, 1);
                cute::initialize_barrier(shared_storage.reduce_mbar[0], 64);
                cute::initialize_barrier(shared_storage.reduce_mbar[1], 64);
                shared_storage.tmem_ready = 0;
            }
        }
        fence_barrier_init();
        __syncthreads();
        pipeline_kv.init_masks(ClusterShape1x1x1{});

        // ======== Phase 2: Prefetch TMA descriptors ========
        if (warp_idx == kLoadWarp && elect_one_sync()) {
            CollectiveMainloop::prefetch_tma_descriptors(params);
        }
        if (warp_idx == kEpiWarp && elect_one_sync()) {
            CollectiveEpilogue::prefetch_tma_descriptors(params);
        }

        // ======== Phase 3: WG3 reg set ========
        if (warp_idx >= 12) {
            warpgroup_reg_set<kRegsOther>();
        }

        // ======== Phase 4: Warp dispatch (single tile) ========
        using TmemAllocator = TMEM::Allocator1Sm;
        TmemAllocator tmem_alloc{};
        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;

        if (warp_idx >= 15) {
            // Warp 15: Idle
        }
        else if (warp_idx == kMmaWarp) {
            tmem_alloc.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                                                    &shared_storage.tmem_base_ptr);
            tmem_alloc.release_allocation_lock();
            __threadfence_block();
            *reinterpret_cast<volatile int*>(&shared_storage.tmem_ready) = 1;

            const uint32_t tmem_base = shared_storage.tmem_base_ptr;
            typename CollectiveMainloop::MmaState mma_state;
            int tile_nkv = CollectiveMainloop::get_tile_num_kv_blocks(
                    params.fwd, work.batch, work.head, work.m_block, global_num_kv_blocks);
            mma_state = mainloop.mma(pipeline_kv, pipeline_s_p_o, pipeline_o_acc, pipeline_p_lastsplit,
                                      shared_storage, tmem_base, tile_nkv, mma_state);

            tmem_alloc.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
        }
        else if (warp_idx == kEpiWarp) {
            while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {}
            __threadfence_block();
            typename CollectiveEpilogue::EpiState epi_state;
            epi_state = epilogue.tma_store(
                    params, pipeline_o_epi, shared_storage,
                    work.head, work.m_block, work.batch, params.fwd.num_m_blocks, epi_state);
        }
        else if (warp_idx == kLoadWarp) {
            while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {}
            __threadfence_block();

            typename CollectiveMainloop::LoadState load_state;
            int tile_nkv = CollectiveMainloop::get_tile_num_kv_blocks(
                    params.fwd, work.batch, work.head, work.m_block, global_num_kv_blocks);
            int raw_bc = CollectiveMainloop::get_tile_raw_block_count(
                    params.fwd, work.batch, work.head, work.m_block);
            load_state = mainloop.load(params, pipeline_kv, shared_storage,
                                        work.head, work.m_block, work.batch, params.fwd.num_m_blocks, tile_nkv,
                                        raw_bc, load_state);
        }
        else if (warp_idx >= 8) {
            cutlass::arch::warpgroup_reg_dealloc<kRegsCorrection>();
            while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {}
            __threadfence_block();
            const uint32_t tmem_base = shared_storage.tmem_base_ptr;
            typename CollectiveMainloop::CorrState corr_state;

            int tile_nkv = CollectiveMainloop::get_tile_num_kv_blocks(
                    params.fwd, work.batch, work.head, work.m_block, global_num_kv_blocks);
            int lse_tile_offset = (work.batch * params.fwd.h + work.head)
                                  * params.fwd.num_m_blocks + work.m_block;
            corr_state = mainloop.template correction<SharedStorage, NamedBarriers>(
                    params.fwd.scale_softmax_log2,
                    pipeline_s_p_o, pipeline_sm_stats, pipeline_o_acc, pipeline_o_epi,
                    shared_storage,
                    tmem_base, tile_nkv, corr_state,
                    static_cast<float*>(params.fwd.softmax_lse_ptr), lse_tile_offset);
            pipeline_o_epi.producer_acquire(corr_state.o_epi_state);
        }
        else if (warp_idx >= 4) {
            warpgroup_reg_set<kRegsSoftmax>();
            while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {}
            __threadfence_block();

            const uint32_t tmem_base = shared_storage.tmem_base_ptr;
            typename CollectiveMainloop::SoftmaxState softmax1_state;
            softmax1_state.spo_state = PipeState(1, 0, 0);
            softmax1_state.sm_stats_state = PipeState(1, 1, 0);

            int tile_nkv = CollectiveMainloop::get_tile_num_kv_blocks(
                    params.fwd, work.batch, work.head, work.m_block, global_num_kv_blocks);
            int const* tile_bi = nullptr;
            if (params.fwd.block_indices_ptr != nullptr) {
                int tile_idx = (work.batch * params.fwd.h + work.head) * params.fwd.num_m_blocks + work.m_block;
                tile_bi = params.fwd.block_indices_ptr
                        + tile_idx * params.fwd.block_indices_stride;
            }
            softmax1_state = mainloop.template softmax</*Stage=*/1, HasBlockSizes, SharedStorage, NamedBarriers>(
                    pipeline_s_p_o, pipeline_sm_stats, pipeline_p_lastsplit,
                    shared_storage, tmem_base, params.fwd.scale_softmax_log2, tile_nkv, softmax1_state,
                    tile_bi, params.fwd.block_sizes_ptr,
                    CollectiveMainloop::get_tile_raw_block_count(
                        params.fwd, work.batch, work.head, work.m_block));
        }
        else {
            warpgroup_reg_set<kRegsSoftmax>();
            while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {}
            __threadfence_block();

            const uint32_t tmem_base = shared_storage.tmem_base_ptr;
            typename CollectiveMainloop::SoftmaxState softmax0_state;
            softmax0_state.sm_stats_state = PipeState(0, 1, 0);

            int tile_nkv = CollectiveMainloop::get_tile_num_kv_blocks(
                    params.fwd, work.batch, work.head, work.m_block, global_num_kv_blocks);
            int const* tile_bi = nullptr;
            if (params.fwd.block_indices_ptr != nullptr) {
                int tile_idx = (work.batch * params.fwd.h + work.head) * params.fwd.num_m_blocks + work.m_block;
                tile_bi = params.fwd.block_indices_ptr
                        + tile_idx * params.fwd.block_indices_stride;
            }
            softmax0_state = mainloop.template softmax</*Stage=*/0, HasBlockSizes, SharedStorage, NamedBarriers>(
                    pipeline_s_p_o, pipeline_sm_stats, pipeline_p_lastsplit,
                    shared_storage, tmem_base, params.fwd.scale_softmax_log2, tile_nkv, softmax0_state,
                    tile_bi, params.fwd.block_sizes_ptr,
                    CollectiveMainloop::get_tile_raw_block_count(
                        params.fwd, work.batch, work.head, work.m_block));
        }
    }
};

template <class Kernel>
__global__ static void __launch_bounds__(Kernel::kThreads, 1)
fused_attn_device(
        __grid_constant__ const typename Kernel::Params params)
{
    extern __shared__ char shared_memory[];
    Kernel kernel;
    kernel(params, shared_memory);
}

template<bool HasBlockSizes>
using FusedAttnKernel = FusedAttnFwdSm100<CollectiveMainloopFwd, CollectiveEpilogueFwd,
                                          SingleTileScheduler, HasBlockSizes>;

#endif
} // namespace flash
