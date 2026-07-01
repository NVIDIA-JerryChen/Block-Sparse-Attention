/******************************************************************************
  * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
  ******************************************************************************/
// Pipeline abstractions for fused attention kernel (SM100)
//
// Pipeline overview:
//   PipelineKV         — K/V multi-stage buffer (Load <-> MMA), TMA-based
//   PipelineSPO        — S/P/O TMEM coordination (MMA UMMA -> Softmax+Correction)
//   PipelineOAcc       — Final O accumulator ready (MMA UMMA -> Correction)
//   PipelineSmStats    — Softmax stats back-pressure (Softmax <-> Correction)
//   PipelineOEpi       — sO SMEM staging (Correction -> Epilogue TMA store)
//   PipelinePLastSplit — Last-split P ready (Softmax -> MMA)
//
// SM100 types from cutlass/pipeline/sm100_pipeline.hpp:
//   PipelineTmaUmmaAsync  — TMA producer, UMMA consumer (KV)
//   PipelineUmmaAsync     — UMMA producer, SW consumer (SPO, OAcc)
//   PipelineAsync          — SW producer, SW consumer (SmStats, OEpi, PLastSplit)
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/pipeline/sm100_pipeline.hpp"

namespace flash {

static constexpr uint32_t kMBarTicks = 1;
static constexpr int kPipelineKVStages = 3;

// ============================================================================
// Raw PTX barrier helpers (kept: used by MMA for p_lastsplit wait and Q-ready)
// ============================================================================

__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n");
}

__device__ __forceinline__ void wait_barrier_addr(uint32_t addr, int phase) {
        asm volatile(
    "{\n"
    ".reg .pred P1;\n"
    "WAIT_BAR_%=:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
    "@P1 bra.uni DONE_BAR_%=;\n"
    "bra.uni WAIT_BAR_%=;\n"
    "DONE_BAR_%=:\n"
    "}\n"
        : : "r"(addr), "r"(phase), "r"(kMBarTicks) : "memory");
}

// Atomic arrive + try_wait on a SMEM mbarrier. Drop-in replacement for
// NamedBarrier::arrive_and_wait when the latter would hit a hw-id
// collision (in blk64: Reduce_02/Reduce_13 vs SmStatsNotify — see the
// reduce_mbar comment in bsa_fwd_kernel_sm100.h).
// `phase` is the parity to wait for; caller toggles it between calls.
__device__ __forceinline__ void mbar_arrive_and_wait(uint32_t addr, int phase) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(addr) : "memory");
    asm volatile(
    "{\n"
    ".reg .pred P1;\n"
    "WAIT_MBAR_%=:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
    "@P1 bra.uni DONE_MBAR_%=;\n"
    "bra.uni WAIT_MBAR_%=;\n"
    "DONE_MBAR_%=:\n"
    "}\n"
        : : "r"(addr), "r"(phase), "r"(kMBarTicks) : "memory");
}

// ============================================================================
// PipelineKV: TMA producer, UMMA consumer (3-stage, 1CTA cluster)
// ============================================================================

using PipelineKV = cutlass::PipelineTmaUmmaAsync<
    /*Stages=*/kPipelineKVStages,
    /*ClusterShape=*/cute::Shape<cute::_1, cute::_1, cute::_1>,
    /*AtomThrShape_MNK=*/cute::Shape<cute::_1, cute::_1, cute::_1>>;
using PipelineKVState = cutlass::PipelineState<kPipelineKVStages>;

// ============================================================================
// Intra-CTA pipelines (SM100 types)
//
//   PipelineSPO:        UMMA producer (MMA), SW consumer (Softmax+Correction)
//   PipelineOAcc:       UMMA producer (MMA), SW consumer (Correction)
//   PipelineSmStats:    SW producer (Softmax), SW consumer (Correction)
//   PipelineOEpi:       SW producer (Correction), SW consumer (Epilogue)
//   PipelinePLastSplit: SW producer (Softmax), SW consumer (MMA via PTX wait)
// ============================================================================

static constexpr int kPipeStages = 2;
using ClusterShape1x1x1 = cute::Shape<cute::_1, cute::_1, cute::_1>;

// UMMA-produced pipelines: MMA warp uses flash::umma_arrive() for barrier arrive
// (not PipelineUmmaAsync — its internal elect_one_sync nests with MMA's outer elect)
using PipelineSPO  = cutlass::PipelineAsync<kPipeStages>;
using PipelineOAcc = cutlass::PipelineAsync<kPipeStages>;

// Software-only pipelines
using PipelineSmStats    = cutlass::PipelineAsync<kPipeStages>;
using PipelineOEpi       = cutlass::PipelineAsync<kPipeStages>;
using PipelinePLastSplit = cutlass::PipelineAsync<kPipeStages>;

using PipeState = cutlass::PipelineState<kPipeStages>;

} // namespace flash
