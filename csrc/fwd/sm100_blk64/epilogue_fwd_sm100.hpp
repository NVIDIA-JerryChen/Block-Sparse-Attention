/******************************************************************************
  * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
  ******************************************************************************/
// CollectiveEpilogueFwd — TMA store for output O.
// Correction logic is in CollectiveMainloopFwd (mainloop_fwd_sm100.hpp).
// This file retains TensorStorage (o_exchange, o_staging, sO) and TMA store.
#pragma once

#include "cutlass/arch/barrier.h"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cute/tensor.hpp"

#include "bsa.h"
#include "pipeline.hpp"
#include "utils.h"

namespace flash {

namespace cute = ::cute;


struct CollectiveEpilogueFwd {
    // ---- Element types ----
    using ElementA = cutlass::bfloat16_t;

    // ---- Tile sizes ----
    static constexpr int kRows = 64;
    static constexpr int kOutputCols = 128;

    // ---- SMEM layout for O ----
    using SmemLayoutO = decltype(cute::coalesce(cute::tile_to_shape(
            cute::UMMA::Layout_K_SW128_Atom<ElementA>{},
            cute::Shape<cute::Int<kRows>, cute::Int<kOutputCols>>{},
            cute::Step<cute::_1, cute::_2>{}), cute::Shape<cute::_1, cute::_1>{}));
    static constexpr int kOBytes = kRows * kOutputCols * sizeof(ElementA);

    // ---- TMA type aliases (5D BSHD) ----
    using ShapeO5 = cute::Shape<cute::Int<kRows>, cute::Int<kOutputCols>, int, int, int>;
    using StrideO5 = cute::Stride<int, cute::_1, int, int, int64_t>;

    using TMA_O = decltype(cute::make_tma_copy(cute::SM90_TMA_STORE{},
            cute::make_tensor(cute::make_gmem_ptr(static_cast<ElementA*>(nullptr)),
                                                cute::make_layout(ShapeO5{}, StrideO5{})),
            SmemLayoutO{}));

    // ---- TensorStorage ----
    static constexpr int kCSpan = 128;
    static constexpr int kExchangePerWarp = kCSpan * 32;
    struct TensorStorage {
        alignas(16) float o_exchange[4][kExchangePerWarp];
        alignas(16) float o_staging[4][64];
        alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutO>> sO;
    };

    // ---- Static TMA construction (called from run_bsa_fwd) ----
    static TMA_O make_tma_store_O(bsa_fwd_params const& p) {
        using namespace cute;
        // 5D TMA store to BSHD: (kRows, kOutputCols, H, num_m_blocks, B)
        auto shape_o  = make_shape(Int<kRows>{}, Int<kOutputCols>{}, p.h, p.num_m_blocks, p.b);
        auto stride_o = make_stride(int(p.o_row_stride), _1{}, int(p.o_head_stride),
                                    kRows * int(p.o_row_stride), p.o_batch_stride);
        return make_tma_copy(SM90_TMA_STORE{},
                make_tensor(make_gmem_ptr(static_cast<ElementA*>(p.o_ptr)),
                            make_layout(shape_o, stride_o)),
                SmemLayoutO{});
    }

    static ShapeO5 make_shape_O(bsa_fwd_params const& p) {
        using namespace cute;
        return make_shape(Int<kRows>{}, Int<kOutputCols>{}, p.h, p.num_m_blocks, p.b);
    }

    // ===========================================================================
    // TMA store sO -> GMEM (Epilogue warp 13)
    // ===========================================================================

    struct EpiState {
        int o_epi_phase = 0;
    };

    template<typename KernelParams>
    CUTLASS_DEVICE static void prefetch_tma_descriptors(KernelParams const& params) {
        cute::prefetch_tma_descriptor(params.tma_store_O.get_tma_descriptor());
    }

    template<typename KernelParams, typename SharedStorage>
    CUTLASS_DEVICE EpiState tma_store(
            KernelParams const& params, PipelineOEpi& pipeline_o_epi,
            SharedStorage& shared_storage,
            int head, int row_tile, int batch, int num_row_tiles,
            EpiState state)
    {
        using namespace cute;
        auto& el = shared_storage.tensors.epilogue;

        if (elect_one_sync()) {
            PipeState epi_wait_state(0, state.o_epi_phase, 0);
            pipeline_o_epi.consumer_wait(epi_wait_state);
            state.o_epi_phase ^= 1;

            auto thr_tma_o = params.tma_store_O.get_slice(Int<0>{});
            auto sO = make_tensor(make_smem_ptr(el.sO.begin()), SmemLayoutO{});
            Tensor gO_full = params.tma_store_O.get_tma_tensor(params.shape_O);
            // 5D indexing: (_, _, head, row_tile, batch)
            Tensor gO_tile = gO_full(_, _, head, row_tile, batch);
            cute::copy(params.tma_store_O, thr_tma_o.partition_S(sO), thr_tma_o.partition_D(gO_tile));
            tma_store_arrive();
            tma_store_wait<0>();

            pipeline_o_epi.consumer_release(PipeState(0, 0, 0));
        }
        return state;
    }
};

} // namespace flash
