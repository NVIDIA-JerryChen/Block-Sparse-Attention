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
         typename TileScheduler_, bool HasBlockSizes, bool HasVarBlockNums,
         bool UseClc = false>
struct FusedAttnFwdSm100 {
    static constexpr bool kUseClc = UseClc;
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

    // CLC persistent scheduler (used when UseClc=true). When UseClc=false the
    // SharedStorage members below still occupy ~96 B of SMEM but are unused;
    // this is cheaper than template-specializing SharedStorage.
    static constexpr int kClcStages = 3;
    using ClcClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using ClcSched = ClcPersistentTileScheduler<ClcClusterShape, kClcStages>;

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
            // CLC pipeline + response buffer (always allocated; only used when
            // UseClc=true). 2 × 8 B mbarriers × kClcStages + 16 B × kClcStages.
            alignas(16) typename ClcSched::Pipeline::SharedStorage clc_pipe;
            alignas(16) typename ClcSched::CLCResponse clc_response[kClcStages];
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
        typename CollectiveMainloop::ShapeKV6 shape_K;
        typename CollectiveMainloop::ShapeKV6 shape_V;
        typename CollectiveEpilogue::ShapeO5 shape_O;

        // CLC-only runtime fields (unused when UseClc=false; kept always to
        // avoid specializing Params on a template parameter).
        int sm_count = 0;
    };

    static dim3 get_grid_shape(Params const& params) {
        if constexpr (kUseClc) {
            return ClcSched::get_grid_shape(params.fwd.num_m_blocks, params.fwd.h, params.fwd.b);
        } else {
            return TileScheduler::get_grid_shape(params.fwd.num_m_blocks, params.fwd.h, params.fwd.b);
        }
    }

    static dim3 get_cluster_shape() {
        if constexpr (kUseClc) {
            return ClcSched::get_cluster_dim();
        } else {
            return dim3(1, 1, 1);
        }
    }

    static dim3 get_block_shape() { return dim3(kThreads, 1, 1); }

    // Per-tile helper to compute tile_bi pointer from (batch, head, m_block).
    CUTLASS_DEVICE static int const* compute_tile_bi(
            Params const& params, int batch, int head, int m_block) {
        if (params.fwd.block_indices_ptr == nullptr) return nullptr;
        int tile_idx = (batch * params.fwd.h + head) * params.fwd.num_m_blocks + m_block;
        return params.fwd.block_indices_ptr + tile_idx * params.fwd.block_indices_stride;
    }

    CUTLASS_DEVICE static int compute_lse_tile_offset(
            Params const& params, int batch, int head, int m_block) {
        return (batch * params.fwd.h + head) * params.fwd.num_m_blocks + m_block;
    }

    // ---- operator(): pipeline init, warp dispatch ----
    CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
        using namespace cute;
        using cutlass::arch::NamedBarrier;

        auto& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        const int warp_idx = threadIdx.x / 32;

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

        // CLC pipeline (always constructed; barrier init is gated inside the
        // ctor on warp_idx == initializing_warp, so the cost when UseClc=false
        // is just a handful of dead mbarrier inits at kernel start). The
        // pipeline handle is used only inside the UseClc=true branch below.
        //
        // NOTE: initializing_warp = 15 (not 0). The other blk64 pipelines all
        // use warp 0 for their init_barriers; putting CLC on a different warp
        // lets the init run in parallel and — more importantly — avoids piling
        // multiple cluster-scope `fence_barrier_init` calls onto a single warp
        // (which empirically correlated with cluster_arrive_relaxed launch
        // failures on B200).
        typename ClcSched::Pipeline::Params clc_params;
        clc_params.transaction_bytes = 16;   // CLCResponse is a fixed 16B payload
        clc_params.producer_arv_count = 1;   // elect_one_sync: single arrive per query
        clc_params.consumer_arv_count =
                kThreads * static_cast<uint32_t>(cute::size<0>(typename ClcSched::ClusterShape{}))
                         * static_cast<uint32_t>(cute::size<1>(typename ClcSched::ClusterShape{}))
                         * static_cast<uint32_t>(cute::size<2>(typename ClcSched::ClusterShape{}));
        clc_params.producer_blockid = 0;     // leader CTA rank within cluster
        clc_params.initializing_warp = 15;   // disjoint from warp 0 used by other pipelines
        // Role = ProducerConsumer so both pipeline_check_is_producer and
        // pipeline_check_is_consumer pass (they're active in non-NDEBUG builds).
        // For 1x1 cluster every CTA is its own leader, so every CTA's warp 15
        // acts as producer + consumer, and all other warps act as consumers
        // (the role guards only gate debug asserts, not runtime behavior).
        clc_params.role = ClcSched::Pipeline::ThreadCategory::ProducerConsumer;
        typename ClcSched::Pipeline clc_pipeline(
                shared_storage.pipelines.clc_pipe, clc_params,
                typename ClcSched::ClusterShape{});

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
        // Post-init sync. CLC mbarriers are cluster-scope; the CUTLASS test
        // pattern uses cluster_arrive_relaxed + cluster_wait after the pipeline
        // ctor. Non-CLC path keeps cta-scope __syncthreads.
        if constexpr (kUseClc) {
            cute::cluster_arrive_relaxed();
            cute::cluster_wait();
        } else {
            __syncthreads();
        }
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

        // ======== Phase 4: Warp dispatch ========
        if constexpr (kUseClc) {
            run_clc(params, shared_storage, clc_pipeline,
                    pipeline_kv, pipeline_s_p_o, pipeline_o_acc,
                    pipeline_sm_stats, pipeline_o_epi, pipeline_p_lastsplit,
                    warp_idx);
        } else {
            run_single(params, shared_storage,
                       pipeline_kv, pipeline_s_p_o, pipeline_o_acc,
                       pipeline_sm_stats, pipeline_o_epi, pipeline_p_lastsplit,
                       warp_idx);
        }
    }

    // ------------------------------------------------------------------
    // run_single — one tile per CTA (fallback, unchanged from pre-CLC path)
    // ------------------------------------------------------------------
    CUTLASS_DEVICE void run_single(
            Params const& params, SharedStorage& shared_storage,
            PipelineKV& pipeline_kv, PipelineSPO& pipeline_s_p_o,
            PipelineOAcc& pipeline_o_acc, PipelineSmStats& pipeline_sm_stats,
            PipelineOEpi& pipeline_o_epi, PipelinePLastSplit& pipeline_p_lastsplit,
            int warp_idx)
    {
        using namespace cute;
        using cutlass::arch::NamedBarrier;

        TileScheduler tile_sched;
        auto work = tile_sched.get_initial_work();

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
            int tile_nkv = CollectiveMainloop::template get_tile_num_kv_blocks<HasVarBlockNums>(
                    params.fwd, work.batch, work.head, work.m_block);
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
            int tile_nkv = CollectiveMainloop::template get_tile_num_kv_blocks<HasVarBlockNums>(
                    params.fwd, work.batch, work.head, work.m_block);
            int raw_bc = CollectiveMainloop::template get_tile_raw_block_count<HasVarBlockNums>(
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

            int tile_nkv = CollectiveMainloop::template get_tile_num_kv_blocks<HasVarBlockNums>(
                    params.fwd, work.batch, work.head, work.m_block);
            int lse_tile_offset = compute_lse_tile_offset(params, work.batch, work.head, work.m_block);
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

            int tile_nkv = CollectiveMainloop::template get_tile_num_kv_blocks<HasVarBlockNums>(
                    params.fwd, work.batch, work.head, work.m_block);
            int const* tile_bi = compute_tile_bi(params, work.batch, work.head, work.m_block);
            softmax1_state = mainloop.template softmax</*Stage=*/1, HasBlockSizes, SharedStorage, NamedBarriers>(
                    pipeline_s_p_o, pipeline_sm_stats, pipeline_p_lastsplit,
                    shared_storage, tmem_base, params.fwd.scale_softmax_log2, tile_nkv, softmax1_state,
                    tile_bi, params.fwd.block_sizes_ptr,
                    CollectiveMainloop::template get_tile_raw_block_count<HasVarBlockNums>(
                        params.fwd, work.batch, work.head, work.m_block));
        }
        else {
            warpgroup_reg_set<kRegsSoftmax>();
            while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {}
            __threadfence_block();

            const uint32_t tmem_base = shared_storage.tmem_base_ptr;
            typename CollectiveMainloop::SoftmaxState softmax0_state;
            softmax0_state.sm_stats_state = PipeState(0, 1, 0);

            int tile_nkv = CollectiveMainloop::template get_tile_num_kv_blocks<HasVarBlockNums>(
                    params.fwd, work.batch, work.head, work.m_block);
            int const* tile_bi = compute_tile_bi(params, work.batch, work.head, work.m_block);
            softmax0_state = mainloop.template softmax</*Stage=*/0, HasBlockSizes, SharedStorage, NamedBarriers>(
                    pipeline_s_p_o, pipeline_sm_stats, pipeline_p_lastsplit,
                    shared_storage, tmem_base, params.fwd.scale_softmax_log2, tile_nkv, softmax0_state,
                    tile_bi, params.fwd.block_sizes_ptr,
                    CollectiveMainloop::template get_tile_raw_block_count<HasVarBlockNums>(
                        params.fwd, work.batch, work.head, work.m_block));
        }
    }

    // ------------------------------------------------------------------
    // run_clc — CLC persistent dispatch. Warp 15 is the scheduler (producer
    // + consumer on every CTA, since ClusterShape=1x1x1 makes every CTA its
    // own leader). All other warps wrap their per-tile body in
    // `while (work.is_valid_tile)` and pull the next tile at the end via
    // ClcSched::consumer_advance.
    //
    // Loop-carried state (per warp) is declared once outside the while loop
    // so pipeline phases persist across tiles. tmem allocation and the
    // `tmem_ready` spin are also hoisted to run once per CTA.
    // ------------------------------------------------------------------
    CUTLASS_DEVICE void run_clc(
            Params const& params, SharedStorage& shared_storage,
            typename ClcSched::Pipeline& clc_pipeline,
            PipelineKV& pipeline_kv, PipelineSPO& pipeline_s_p_o,
            PipelineOAcc& pipeline_o_acc, PipelineSmStats& pipeline_sm_stats,
            PipelineOEpi& pipeline_o_epi, PipelinePLastSplit& pipeline_p_lastsplit,
            int warp_idx)
    {
        using namespace cute;
        using cutlass::arch::NamedBarrier;

        using TmemAllocator = TMEM::Allocator1Sm;
        TmemAllocator tmem_alloc{};
        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;

        auto* clc_response_ptr = shared_storage.pipelines.clc_response;
        typename ClcSched::PipelineState clc_consumer_state{};

        if (warp_idx >= 15) {
            // Scheduler warp: producer + consumer (1x1 cluster => every CTA leader).
            typename ClcSched::PipelineState clc_producer_state =
                    cutlass::make_producer_start_state<typename ClcSched::Pipeline>();
            auto work = ClcSched::get_initial_work();
            while (work.is_valid_tile) {
                ClcSched::advance_producer(clc_pipeline, clc_producer_state, clc_response_ptr);
                work = ClcSched::consumer_advance(clc_pipeline, clc_consumer_state, clc_response_ptr);
            }
            clc_pipeline.producer_tail(clc_producer_state);
        }
        else if (warp_idx == kMmaWarp) {
            tmem_alloc.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                                                    &shared_storage.tmem_base_ptr);
            tmem_alloc.release_allocation_lock();
            __threadfence_block();
            *reinterpret_cast<volatile int*>(&shared_storage.tmem_ready) = 1;

            const uint32_t tmem_base = shared_storage.tmem_base_ptr;
            typename CollectiveMainloop::MmaState mma_state;

            auto work = ClcSched::get_initial_work();
            while (work.is_valid_tile) {
                int tile_nkv = CollectiveMainloop::template get_tile_num_kv_blocks<HasVarBlockNums>(
                        params.fwd, work.batch, work.head, work.m_block);
                mma_state = mainloop.mma(
                        pipeline_kv, pipeline_s_p_o, pipeline_o_acc, pipeline_p_lastsplit,
                        shared_storage, tmem_base, tile_nkv, mma_state);
                work = ClcSched::consumer_advance(clc_pipeline, clc_consumer_state, clc_response_ptr);
            }

            tmem_alloc.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
        }
        else if (warp_idx == kEpiWarp) {
            while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {}
            __threadfence_block();
            typename CollectiveEpilogue::EpiState epi_state;

            auto work = ClcSched::get_initial_work();
            while (work.is_valid_tile) {
                epi_state = epilogue.tma_store(
                        params, pipeline_o_epi, shared_storage,
                        work.head, work.m_block, work.batch, params.fwd.num_m_blocks, epi_state);
                work = ClcSched::consumer_advance(clc_pipeline, clc_consumer_state, clc_response_ptr);
            }
        }
        else if (warp_idx == kLoadWarp) {
            while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {}
            __threadfence_block();

            typename CollectiveMainloop::LoadState load_state;
            auto work = ClcSched::get_initial_work();
            while (work.is_valid_tile) {
                int tile_nkv = CollectiveMainloop::template get_tile_num_kv_blocks<HasVarBlockNums>(
                        params.fwd, work.batch, work.head, work.m_block);
                int raw_bc = CollectiveMainloop::template get_tile_raw_block_count<HasVarBlockNums>(
                        params.fwd, work.batch, work.head, work.m_block);
                load_state = mainloop.load(params, pipeline_kv, shared_storage,
                        work.head, work.m_block, work.batch, params.fwd.num_m_blocks, tile_nkv,
                        raw_bc, load_state);
                work = ClcSched::consumer_advance(clc_pipeline, clc_consumer_state, clc_response_ptr);
            }
        }
        else if (warp_idx >= 8) {
            cutlass::arch::warpgroup_reg_dealloc<kRegsCorrection>();
            while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {}
            __threadfence_block();
            const uint32_t tmem_base = shared_storage.tmem_base_ptr;
            typename CollectiveMainloop::CorrState corr_state;

            auto work = ClcSched::get_initial_work();
            while (work.is_valid_tile) {
                int tile_nkv = CollectiveMainloop::template get_tile_num_kv_blocks<HasVarBlockNums>(
                        params.fwd, work.batch, work.head, work.m_block);
                int lse_tile_offset = compute_lse_tile_offset(params, work.batch, work.head, work.m_block);
                corr_state = mainloop.template correction<SharedStorage, NamedBarriers>(
                        params.fwd.scale_softmax_log2,
                        pipeline_s_p_o, pipeline_sm_stats, pipeline_o_acc, pipeline_o_epi,
                        shared_storage,
                        tmem_base, tile_nkv, corr_state,
                        static_cast<float*>(params.fwd.softmax_lse_ptr), lse_tile_offset);
                work = ClcSched::consumer_advance(clc_pipeline, clc_consumer_state, clc_response_ptr);
            }
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

            auto work = ClcSched::get_initial_work();
            while (work.is_valid_tile) {
                int tile_nkv = CollectiveMainloop::template get_tile_num_kv_blocks<HasVarBlockNums>(
                        params.fwd, work.batch, work.head, work.m_block);
                int const* tile_bi = compute_tile_bi(params, work.batch, work.head, work.m_block);
                softmax1_state = mainloop.template softmax</*Stage=*/1, HasBlockSizes, SharedStorage, NamedBarriers>(
                        pipeline_s_p_o, pipeline_sm_stats, pipeline_p_lastsplit,
                        shared_storage, tmem_base, params.fwd.scale_softmax_log2, tile_nkv, softmax1_state,
                        tile_bi, params.fwd.block_sizes_ptr,
                        CollectiveMainloop::template get_tile_raw_block_count<HasVarBlockNums>(
                            params.fwd, work.batch, work.head, work.m_block));
                work = ClcSched::consumer_advance(clc_pipeline, clc_consumer_state, clc_response_ptr);
            }
        }
        else {
            warpgroup_reg_set<kRegsSoftmax>();
            while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {}
            __threadfence_block();

            const uint32_t tmem_base = shared_storage.tmem_base_ptr;
            typename CollectiveMainloop::SoftmaxState softmax0_state;
            softmax0_state.sm_stats_state = PipeState(0, 1, 0);

            auto work = ClcSched::get_initial_work();
            while (work.is_valid_tile) {
                int tile_nkv = CollectiveMainloop::template get_tile_num_kv_blocks<HasVarBlockNums>(
                        params.fwd, work.batch, work.head, work.m_block);
                int const* tile_bi = compute_tile_bi(params, work.batch, work.head, work.m_block);
                softmax0_state = mainloop.template softmax</*Stage=*/0, HasBlockSizes, SharedStorage, NamedBarriers>(
                        pipeline_s_p_o, pipeline_sm_stats, pipeline_p_lastsplit,
                        shared_storage, tmem_base, params.fwd.scale_softmax_log2, tile_nkv, softmax0_state,
                        tile_bi, params.fwd.block_sizes_ptr,
                        CollectiveMainloop::template get_tile_raw_block_count<HasVarBlockNums>(
                            params.fwd, work.batch, work.head, work.m_block));
                work = ClcSched::consumer_advance(clc_pipeline, clc_consumer_state, clc_response_ptr);
            }
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

template<bool HasBlockSizes, bool HasVarBlockNums, bool UseClc = false>
using FusedAttnKernel = FusedAttnFwdSm100<CollectiveMainloopFwd, CollectiveEpilogueFwd,
                                          SingleTileScheduler, HasBlockSizes, HasVarBlockNums,
                                          UseClc>;

#endif
} // namespace flash
