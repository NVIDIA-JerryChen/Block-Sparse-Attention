/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"
#include "pipeline.hpp"
#include "utils.h"

namespace flash {

// CLC persistent tile scheduler — symmetric wait-then-release pattern (matches blk128).
//
// ALL threads (kThreads=512) participate in consumer_release, including the scheduler
// warp. This avoids the asymmetric barrier pattern that causes CLC response delivery
// stalls on SM100/SM103.
//
// consumer_arv_count = kThreads (512): scheduler warp also does consumer_release.
// Scheduler warp: producer (advance_to_next_work) + consumer (fetch_next_work).
// Worker warps:   consumer only (consumer_advance → fetch_next_work).
struct CLCTileScheduler {

    struct Params {
        int num_row_tiles;
        int num_kv_blocks;
        int num_heads;
        int num_batch;
    };

    PipelineCLC& pipeline_clc;
    CLCResponse* clc_response;
    Params params;
    PipelineCLCState prod_state;
    PipelineCLCState cons_state;

    CUTLASS_DEVICE
    CLCTileScheduler(PipelineCLC& clc, CLCResponse* resp, Params const& p)
        : pipeline_clc(clc)
        , clc_response(resp)
        , params(p)
        , prod_state(cutlass::make_producer_start_state<PipelineCLC>())  // {0, 1, 0}
        , cons_state()  // {0, 0, 0}
    {}

    CUTLASS_DEVICE
    static WorkTileInfo decode(int linear_idx, int num_row_tiles, int num_heads) {
        int tiles_per_batch = num_row_tiles * num_heads;
        int batch = linear_idx / tiles_per_batch;
        int rem = linear_idx % tiles_per_batch;
        int head = rem / num_row_tiles;
        int row_tile = rem % num_row_tiles;
        return {row_tile, head, batch, true};
    }

    CUTLASS_DEVICE
    WorkTileInfo initial_work_tile_info() {
        return decode(static_cast<int>(blockIdx.x), params.num_row_tiles, params.num_heads);
    }

    // ---- Sched warp producer: acquire(empty) → [reinit] → issue CLC query ----
    // Called by ALL 32 threads of the scheduler warp.
    template <typename ReinitFn>
    CUTLASS_DEVICE void advance_to_next_work(ReinitFn&& reinit_fn) {
        pipeline_clc.producer_acquire(prod_state);
        // Safe reinit window: all threads have released (done with previous tile).
        reinit_fn();
        uint32_t mbar = pipeline_clc.producer_get_barrier(prod_state);
        if (cute::elect_one_sync()) {
            uint32_t resp_addr = smem_ptr_to_uint(&clc_response[prod_state.index()]);
            issue_clc_query(resp_addr, mbar);
        }
        ++prod_state;
    }

    CUTLASS_DEVICE void advance_to_next_work() {
        pipeline_clc.producer_acquire(prod_state);
        uint32_t mbar = pipeline_clc.producer_get_barrier(prod_state);
        if (cute::elect_one_sync()) {
            uint32_t resp_addr = smem_ptr_to_uint(&clc_response[prod_state.index()]);
            issue_clc_query(resp_addr, mbar);
        }
        ++prod_state;
    }

    // ---- All warps consumer: wait(full) → decode → release(empty) ----
    // Matches blk128 pattern: wait + read + release for a single CLC result.
    // ALL threads (including scheduler) call this.
    CUTLASS_DEVICE WorkTileInfo fetch_next_work() {
        pipeline_clc.consumer_wait(cons_state);
        uint32_t resp_addr = smem_ptr_to_uint(&clc_response[cons_state.index()]);
        auto resp = decode_clc_response(resp_addr);
        pipeline_clc.consumer_release(cons_state);
        ++cons_state;
        if (!resp.is_valid) {
            return {0, 0, 0, false};
        }
        return decode(resp.row_tile, params.num_row_tiles, params.num_heads);
    }

    // ---- Worker consumer: pure CLC pipeline operation (matches blk128) ----
    // Only does CLC wait/release. tcgen05_commit is caller's responsibility
    // (only MMA/Load warps that operate TMEM need it).
    CUTLASS_DEVICE WorkTileInfo consumer_advance() {
        return fetch_next_work();
    }

    // ---- Producer tail: drain pipeline before exit ----
    CUTLASS_DEVICE void producer_tail() {
        pipeline_clc.producer_tail(prod_state);
    }
};

} // namespace flash
