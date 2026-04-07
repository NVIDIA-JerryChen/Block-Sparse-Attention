/******************************************************************************
  * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
  ******************************************************************************/
// Pipeline abstractions for fused attention kernel
//
// Pipeline overview (BSA naming):
//   PipelineKV       — K/V multi-stage buffer (Load <-> MMA), TMA-based
//   PipelineSPO      — S/P/O TMEM coordination (MMA -> Softmax+Correction)
//   PipelineOAcc     — Final O accumulator ready (MMA -> Correction)
//   PipelineSmStats  — Softmax stats back-pressure (Softmax <-> Correction)
//   PipelineOEpi     — sO SMEM staging (Correction -> Epilogue TMA store)
//   PipelinePLastSplit — Last-split P ready (Softmax -> MMA)
//   PipelineCLC      — Cluster Launch Control (CLC) scheduler pipeline
//
// All intra-CTA pipelines (SPO, OAcc, SmStats, OEpi, PLastSplit) use
// CUTLASS PipelineAsync<2>. UMMA hardware arrives are done externally.
#pragma once

#ifndef CUTLASS_ARCH_CLC_ENABLED
#define CUTLASS_ARCH_CLC_ENABLED
#endif

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/pipeline/sm100_pipeline.hpp"

namespace flash {

static constexpr uint32_t kMBarTicks = 1;

// ============================================================================
// Barrier helpers (retained: used by MMA inline PTX for p_lastsplit wait
// and Q-ready barrier)
// ============================================================================

__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n");
}

// Address-based overloads (for pre-computed UR-promoted addresses).
__device__ __forceinline__ void mbarrier_arrive_addr(uint32_t addr) {
        asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" : : "r"(addr) : "memory");
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

__device__ __forceinline__ void wait_barrier_acquire_addr(uint32_t addr, int phase) {
        asm volatile(
    "{\n"
    ".reg .pred P1;\n"
    "WAIT_ACQ_%=:\n"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n"
    "@P1 bra.uni DONE_ACQ_%=;\n"
    "bra.uni WAIT_ACQ_%=;\n"
    "DONE_ACQ_%=:\n"
    "}\n"
        : : "r"(addr), "r"(phase), "r"(kMBarTicks) : "memory");
}

// Broadcast a barrier SMEM address via __shfl_sync for UR promotion.
__device__ __forceinline__ uint32_t barrier_smem_addr(cute::uint64_t& bar) {
        return __shfl_sync(0xFFFFFFFF, cute::cast_smem_ptr_to_uint(&bar), 0);
}

// ============================================================================
// PipelineKV: K/V multi-stage buffer (TMA-based, 3-stage)
//
//   Producer: Load warp (1 thread) — TMA loads K/V into alternating slots
//   Consumer: MMA warp (1 thread)  — reads K/V for QK/PV MMA
// ============================================================================

// PipelineKV: use CUTLASS PipelineTmaUmmaAsync directly (example 77 pattern).
// 1CTA cluster, 1x1x1 atom shape.
using PipelineKV = cutlass::PipelineTmaUmmaAsync<
    /*Stages=*/3,
    /*ClusterShape=*/cute::Shape<cute::_1, cute::_1, cute::_1>,
    /*AtomThrShape_MNK=*/cute::Shape<cute::_1, cute::_1, cute::_1>>;
using PipelineKVState = cutlass::PipelineState<3>;

// ============================================================================
// Intra-CTA pipelines: CUTLASS PipelineAsync<2>
//
// All 5 intra-CTA pipelines use PipelineAsync<2> as their base type
// (blk128 PipelineUmmaAsync/PipelineAsyncUmma also wrap PipelineAsync).
// UMMA hardware arrives are done externally via flash::umma_arrive().
//
//   PipelineSPO:       producer=MMA(1 UMMA), consumer=Softmax+Correction(256)
//   PipelineOAcc:      producer=MMA(1 UMMA), consumer=dummy(1)
//   PipelineSmStats:   producer=Softmax(128), consumer=Correction(128)
//   PipelineOEpi:      producer=Correction(128), consumer=Epilogue(1)
//   PipelinePLastSplit: producer=Softmax(4 warps), consumer=dummy(1)
// ============================================================================

static constexpr int kPipeStages = 2;

using PipelineSPO        = cutlass::PipelineAsync<kPipeStages>;
using PipelineOAcc       = cutlass::PipelineAsync<kPipeStages>;
using PipelineSmStats    = cutlass::PipelineAsync<kPipeStages>;
using PipelineOEpi       = cutlass::PipelineAsync<kPipeStages>;
using PipelinePLastSplit = cutlass::PipelineAsync<kPipeStages>;

using PipeState = cutlass::PipelineState<kPipeStages>;

// CTA-local barrier operations for single-CTA kernels.
// PipelineAsync::consumer_release uses mbarrier.arrive.shared::cluster (via mapa).
// For single-CTA kernels this is functionally identical to shared::cta, but we
// provide explicit CTA-scope wrappers to avoid the mapa instruction overhead.
template<typename SharedStorage>
__device__ __forceinline__ void consumer_release_cta(
        SharedStorage& storage, PipeState state) {
    uint32_t addr = cute::cast_smem_ptr_to_uint(&storage.empty_barrier_[state.index()]);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(addr) : "memory");
}

template<typename SharedStorage>
__device__ __forceinline__ void producer_commit_cta(
        SharedStorage& storage, PipeState state) {
    uint32_t addr = cute::cast_smem_ptr_to_uint(&storage.full_barrier_[state.index()]);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" : : "r"(addr) : "memory");
}

// ============================================================================
// CLC (Cluster Launch Control) infrastructure
// ============================================================================

// CLC response buffer: 128-bit (4x uint32) to hold the clusterlaunchcontrol
// query_cancel response from hardware.
struct CLCResponse { uint32_t data[4]; };

// Decoded work tile info from a CLC response.
struct WorkTileInfo {
    int row_tile;
    int head;
    int batch;
    bool is_valid;
};

// Inline helper: cast any smem pointer to uint32 address (uses same idiom as
// cute::cast_smem_ptr_to_uint but without the cute dependency at callsite).
__device__ __forceinline__ uint32_t smem_ptr_to_uint(void const* ptr) {
    uint32_t r;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
      : "=r"(r) : "l"(ptr));
    return r;
}

// decode_clc_response: read a CLCResponse from smem_addr and decode it.
// Returns WorkTileInfo with (row_tile, head, batch, is_valid).
// row_tile <- ctaidx.x, head <- ctaidx.y, batch <- ctaidx.z.
__device__ __forceinline__ WorkTileInfo decode_clc_response(uint32_t smem_addr) {
    WorkTileInfo info{0, 0, 0, false};
#if defined(CUTLASS_ARCH_CLC_ENABLED)
    uint32_t bx = 0, by = 0, bz = 0, valid_u32 = 0;
    asm volatile(
    "{\n"
    "  .reg .pred p;\n"
    "  .reg .b128 r;\n"
    "  ld.shared.b128 r, [%4];\n"
    "  clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r;\n"
    "  selp.u32 %3, 1, 0, p;\n"
    "  @p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0,%1,%2,_}, r;\n"
    "}\n"
    : "=r"(bx), "=r"(by), "=r"(bz), "=r"(valid_u32)
    : "r"(smem_addr)
    : "memory");
    asm volatile("fence.proxy.async;\n" ::: "memory");
    info.row_tile = static_cast<int>(bx);
    info.head     = static_cast<int>(by);
    info.batch    = static_cast<int>(bz);
    info.is_valid = (valid_u32 == 1);
#endif
    return info;
}

// issue_clc_query: issue a clusterlaunchcontrol.try_cancel async query.
// result_addr   — smem address of CLCResponse (128-bit aligned)
// mbarrier_addr — smem address of the transaction mbarrier (full barrier)
//                 which will be signaled when the response arrives (16 bytes tx).
__device__ __forceinline__ void issue_clc_query(uint32_t result_addr,
                                                                                                uint32_t mbarrier_addr) {
#if defined(CUTLASS_ARCH_CLC_ENABLED)
    asm volatile(
    "clusterlaunchcontrol.try_cancel.async.shared::cta"
    ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n"
    : : "r"(result_addr), "r"(mbarrier_addr));
#endif
}

// ============================================================================
// PipelineCLC: CLC scheduler pipeline — CUTLASS PipelineCLCFetchAsync<1>
//
//   Producer: Sched warp (warp 15 / kSchedWarp) — issues CLC queries.
//   Consumer: Worker warps (kWorkerThreads = 480) — read decoded tile info.
//
//   CUTLASS manages full/empty barriers internally.
//   CLCResponse buffer is stored separately in kernel SharedStorage.
// ============================================================================

static constexpr int kCLCStages = 1;
using PipelineCLC = cutlass::PipelineCLCFetchAsync<kCLCStages>;
using PipelineCLCState = cutlass::PipelineState<kCLCStages>;

} // namespace flash
