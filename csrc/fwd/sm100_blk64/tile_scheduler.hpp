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

// CLC persistent tile scheduler with batch support.
// Grid = (num_row_tiles * num_heads * batch, 1, 1) for CLC 1D flattening.
// CLC hardware returns (bx, by, bz) = (row_tile, head, batch) for 3D grid.
struct CLCTileScheduler {

    struct Params {
        int num_row_tiles;
        int num_kv_blocks;
        int num_heads;
        int num_batch;
    };

    PipelineCLC& pipeline_clc;
    Params params;
    uint32_t prod_phase = 0;
    uint32_t cons_phase = 0;

    CUTLASS_DEVICE
    CLCTileScheduler(PipelineCLC& clc, Params const& p)
        : pipeline_clc(clc), params(p) {}

    // Decode flattened 1D index → (row_tile, head, batch).
    // Layout: batch-major, then head, then row_tile (matches blockIdx.x flattening).
    CUTLASS_DEVICE
    static WorkTileInfo decode(int linear_idx, int num_row_tiles, int num_heads) {
        int tiles_per_batch = num_row_tiles * num_heads;
        int batch = linear_idx / tiles_per_batch;
        int rem = linear_idx % tiles_per_batch;
        int head = rem / num_row_tiles;
        int row_tile = rem % num_row_tiles;
        return {row_tile, head, batch, true};
    }

    // Get initial work tile from blockIdx (1D flattened, all warps).
    CUTLASS_DEVICE
    WorkTileInfo initial_work_tile_info() {
        return decode(static_cast<int>(blockIdx.x), params.num_row_tiles, params.num_heads);
    }

    // ---- Sched warp producer: acquire(empty) → [reinit] → issue CLC query ----
    template <typename ReinitFn>
    CUTLASS_DEVICE void advance_to_next_work(ReinitFn&& reinit_fn) {
        pipeline_clc.producer_acquire(prod_phase);
        prod_phase ^= 1;
        reinit_fn();
        pipeline_clc.producer_expect_tx();
        issue_clc_query(pipeline_clc.get_response_addr(),
                        pipeline_clc.producer_get_barrier());
    }

    CUTLASS_DEVICE void advance_to_next_work() {
        pipeline_clc.producer_acquire(prod_phase);
        prod_phase ^= 1;
        pipeline_clc.producer_expect_tx();
        issue_clc_query(pipeline_clc.get_response_addr(),
                        pipeline_clc.producer_get_barrier());
    }

    // ---- Sched warp consumer: wait(full) → decode ----
    CUTLASS_DEVICE WorkTileInfo fetch_next_work() {
        pipeline_clc.consumer_wait(cons_phase);
        cons_phase ^= 1;
        auto resp = decode_clc_response(pipeline_clc.get_response_addr());
        if (!resp.is_valid) {
            return {0, 0, 0, false};
        }
        // CLC with 1D grid: bx=linear_idx, by=0, bz=0. Decode bx to 3D.
        return decode(resp.row_tile, params.num_row_tiles, params.num_heads);
    }

    // ---- Worker consumer: fence → release(empty) → wait(full) → decode ----
    CUTLASS_DEVICE WorkTileInfo consumer_advance() {
        flash::tcgen05_fence_before_sync();
        cutlass::arch::fence_view_async_tmem_store();
        cutlass::arch::fence_view_async_shared();
        pipeline_clc.consumer_release();
        pipeline_clc.consumer_wait(cons_phase);
        cons_phase ^= 1;
        auto resp = decode_clc_response(pipeline_clc.get_response_addr());
        if (!resp.is_valid) {
            return {0, 0, 0, false};
        }
        flash::tcgen05_commit();
        return decode(resp.row_tile, params.num_row_tiles, params.num_heads);
    }
};

} // namespace flash
