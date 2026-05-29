/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 ******************************************************************************/
// Tile schedulers for blk64 forward attention.
//
//   SingleTileScheduler         — each CTA = 1 tile (current default; FA Hopper
//                                  pattern).
//
//   ClcPersistentTileScheduler  — Blackwell SM100 Cluster Launch Control (CLC)
//                                  based dynamic persistent scheduler. Wraps
//                                  cutlass::gemm::kernel::detail::
//                                  PersistentTileSchedulerSm100 and exposes the
//                                  type aliases + host-side helpers blk64's
//                                  kernel needs.
//
// PHASE A (this file): types + host helpers only. Device-side persistent loop
// wiring (warp 15 producer/consumer, while(work.is_valid_tile), CLC pipeline
// init) lives in bsa_fwd_kernel_sm100.h and will be added in a follow-up.
// See design/docs/blk64_clc_scheduler.md for the design.

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"

namespace flash {

struct SingleTileScheduler {

    struct WorkTileInfo {
        int m_block;
        int head;
        int batch;
        bool is_valid_tile;
    };

    CUTLASS_DEVICE
    WorkTileInfo get_initial_work() const {
        return {static_cast<int>(blockIdx.x),
                static_cast<int>(blockIdx.y),
                static_cast<int>(blockIdx.z), true};
    }

    CUTLASS_DEVICE
    WorkTileInfo get_next_work(WorkTileInfo const&) const {
        return {0, 0, 0, false};
    }

    static dim3 get_grid_shape(int num_m_blocks, int num_heads, int batch) {
        return dim3(num_m_blocks, num_heads, batch);
    }
};

// ============================================================================
// ClcPersistentTileScheduler
//
// Wraps cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100 for the
// blk64 kernel. The launch grid matches the full tile space
// (num_m_blocks, num_heads, batch) so every blockIdx is a valid initial tile;
// CLC then hands out additional tiles to CTAs that finish early, providing
// work-stealing for variable-block-num sparse tiles.
//
// Coordinate mapping (blk64 convention): WorkTileInfo (M_idx, N_idx, L_idx)
// maps to (m_block, head, batch).
// ============================================================================

template <class ClusterShape_, uint32_t Stages_>
struct ClcPersistentTileScheduler {
    using ClusterShape = ClusterShape_;
    static constexpr uint32_t Stages = Stages_;

    using Underlying = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
            ClusterShape, Stages>;
    using Pipeline      = typename Underlying::Pipeline;       // PipelineCLCFetchAsync
    using CLCResponse   = typename Underlying::CLCResponse;    // 16B opaque
    using WorkTileInfo  = typename Underlying::WorkTileInfo;
    using Params        = typename Underlying::Params;
    using PipelineState = cutlass::PipelineState<Stages>;

    // blk64-flavored work info (m_block, head, batch).
    struct WorkInfo {
        int m_block;
        int head;
        int batch;
        bool is_valid_tile;
    };

    CUTLASS_DEVICE static WorkInfo to_work_info(WorkTileInfo const& w) {
        return {w.M_idx, w.N_idx, w.L_idx, w.is_valid_tile};
    }

    // Initial tile = this CTA's blockIdx, marked valid. Subsequent tiles come
    // from CLC queries via advance_producer / consumer_advance below.
    CUTLASS_DEVICE static WorkInfo get_initial_work() {
        return {static_cast<int>(blockIdx.x),
                static_cast<int>(blockIdx.y),
                static_cast<int>(blockIdx.z), true};
    }

    // Producer side (runs on scheduler warp). One thread per cluster issues
    // the CLC query; all threads of the producer warp must call producer_acquire
    // so the full_barrier transaction bytes are reserved.
    CUTLASS_DEVICE static void advance_producer(
            Pipeline& pipe, PipelineState& p_state, CLCResponse* clc_response_ptr) {
        pipe.producer_acquire(p_state);
        uint32_t mbar = pipe.producer_get_barrier(p_state);
        if (cute::elect_one_sync()) {
            Underlying::issue_clc_query(p_state, mbar, clc_response_ptr);
        }
        ++p_state;
    }

    // Consumer side (called by every warp at tile boundary): wait for the next
    // CLC response, extract the work tile, release the stage, advance state.
    //
    // The leading __syncthreads forces all 16 warps to converge on the same
    // tile boundary before issuing consumer_wait. Without it, fast warps
    // (scheduler, load) can race ahead of slower compute warps and read a
    // stale CLC response from the previous tile — the race only shows up when
    // CTAs process >1 tile (grid > SM count).
    CUTLASS_DEVICE static WorkInfo consumer_advance(
            Pipeline& pipe, PipelineState& c_state, CLCResponse* clc_response_ptr) {
        __syncthreads();
        pipe.consumer_wait(c_state);
        uint32_t smem_addr = cute::cast_smem_ptr_to_uint(&clc_response_ptr[c_state.index()]);
        WorkTileInfo w = Underlying::work_tile_info_from_clc_response(smem_addr);
        pipe.consumer_release(c_state);
        ++c_state;
        return to_work_info(w);
    }

    // ------- Host-side helpers -------
    //
    // Grid shape mirrors the tile space exactly so every blockIdx maps to a
    // valid initial tile. CLC hands out the remainder dynamically.
    static dim3 get_grid_shape(int num_m_blocks, int num_heads, int batch) {
        int cluster_m = static_cast<int>(cute::size<0>(ClusterShape{}));
        int grid_m = ((num_m_blocks + cluster_m - 1) / cluster_m) * cluster_m;
        return dim3(static_cast<unsigned>(grid_m),
                    static_cast<unsigned>(num_heads),
                    static_cast<unsigned>(batch));
    }

    // Cluster dim to pass to cutlass::ClusterLauncher::launch.
    static dim3 get_cluster_dim() {
        return dim3(static_cast<unsigned>(cute::size<0>(ClusterShape{})),
                    static_cast<unsigned>(cute::size<1>(ClusterShape{})),
                    static_cast<unsigned>(cute::size<2>(ClusterShape{})));
    }
};

} // namespace flash
