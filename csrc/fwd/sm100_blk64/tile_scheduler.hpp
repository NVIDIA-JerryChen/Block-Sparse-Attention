/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 ******************************************************************************/
// SingleTileScheduler — each CTA processes exactly one tile from the 3D grid.
// Matches FA Hopper SingleTileScheduler pattern.

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"

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

} // namespace flash
