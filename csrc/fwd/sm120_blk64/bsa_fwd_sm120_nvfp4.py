# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 block-sparse attention forward with NVFP4 QK and Sage FP8 PV.

Q and K are NVFP4 (E2M1 values, one E4M3 scale per 16 head-dim elements) and
feed the SM120 block-scaled warp MMA, which applies the scales inside the
instruction. P and V keep the Sage FP8 path of ``bsa_fwd_sm120_fp8``.
"""

import operator

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu.warp.mma import Field as WarpField
from cutlass.utils import LayoutEnum, SmemAllocator
import cuda.bindings.driver as cuda
from block_sparse_attention.csrc.utils import kernel_utils
from block_sparse_attention.csrc.utils import layout_utils
from block_sparse_attention.csrc.utils.batched_static_scheduler import (
    BatchedStaticSchedulerMixin,
)
from block_sparse_attention.csrc.fwd.sm120_blk64.bsa_fwd_sm120_fp8 import (
    _gemm_rs_fp8,
    _make_acc_into_fp8_op,
    _mask_fp8,
    _rescale_o_for_next_acc_fp8,
    _rescale_o_with_sage_v_scale_fp8,
)

SM120_FWD_BLOCK_SIZE = 64
NVFP4_SF_VEC_SIZE = 16
# Each 64-row tile's scales are stored as one contiguous 512-byte row-major
# (64, head_dim / 16) block, which TMA moves as a (32, 16) byte box.
NVFP4_SF_TMA_COLS = 16


# =============================================================================
# Public kernel class
# =============================================================================
class BlockSparseAttnForwardNvfp4Sm120Blk64(BatchedStaticSchedulerMixin):
    def __init__(
        self,
        gqa_ratio: int = 1,
        head_dim: int = 128,
        value_dim: int = 128,
        blocksparse_blocksize_q: int = 64,
        blocksparse_blocksize_k: int = 64,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        has_block_sizes: bool = True,
        has_block_nums: bool = True,
        block_sizes_mode: int = 0,
    ):
        self.acc_dtype = acc_dtype
        assert self.acc_dtype is cutlass.Float32
        self.softmax_p_scale_log2 = 8.0

        self.tile_size = 64

        assert blocksparse_blocksize_q == 64, "Only block_size_m=64 is supported in this kernel."
        assert blocksparse_blocksize_k in [64], "block_size_n should be one of [64]"
        self.num_threads = 128
        self.num_mma_warps = 4
        # V keeps one stage (its transpose load addresses stage 0).
        self.kv_stage = 1
        self.k_stage = 1
        self.q_stage = 1
        # Three CTAs per SM: caps registers at 168 per thread.
        self.min_blocks_per_mp = 3

        assert gqa_ratio >= 1
        assert head_dim == 128, "SM120 NVFP4 blk64 fwd currently requires QK dim 128"
        assert value_dim == 128, "SM120 NVFP4 blk64 fwd currently requires value dim 128"
        self.gqa_ratio = gqa_ratio
        self.qk_dim = head_dim
        self.value_dim = value_dim
        self.tile_shape_qk = (self.tile_size, self.tile_size, self.qk_dim)
        self.tile_shape_pv = (self.tile_size, self.value_dim, self.tile_size)
        self.sf_vec_size = NVFP4_SF_VEC_SIZE
        self.num_sf_per_row = self.qk_dim // self.sf_vec_size
        self.sf_tile_bytes = self.tile_size * self.num_sf_per_row
        assert self.sf_tile_bytes % NVFP4_SF_TMA_COLS == 0
        self.sf_tma_tile = (NVFP4_SF_TMA_COLS, self.sf_tile_bytes // NVFP4_SF_TMA_COLS)

        self.scheduler = None

        self.use_tma_o = True
        self.has_block_sizes = has_block_sizes
        self.has_block_nums = has_block_nums
        self.block_sizes_mode = block_sizes_mode

    def check_dim(self, tensor: cute.Tensor | list[cute.Tensor], mode: int):
        if isinstance(tensor, list):
            for t in tensor:
                self.check_dim(t, mode)
            return
        assert tensor.shape[mode] == 128, f"dim must be 128 in mode {mode}."
        assert tensor.stride[mode] == 1, f"dim must be contiguous in mode {mode}."

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mQSF: cute.Tensor,
        mKSF: cute.Tensor,
        mVScale: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        tma_atom_QSF: cute.CopyAtom,
        tma_atom_KSF: cute.CopyAtom,
        blocksparse_indices_q2k: cute.Tensor,
        blocksparse_num_blocks_q2k: cute.Tensor,
        block_sparse_num: cutlass.Int32,
        blocksparse_varblk: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        mma_atom_qk: cute.MmaAtom,
        tiled_mma_pv: cute.TiledMma,
        Q_smem_layout: cute.ComposedLayout,
        K_smem_layout: cute.ComposedLayout,
        V_smem_layout: cute.ComposedLayout,
        Vt_smem_layout: cute.ComposedLayout,
        O_smem_layout: cute.ComposedLayout,
        SF_smem_layout: cute.Layout,
        KSF_smem_layout: cute.Layout,
        scale_softmax_log2e: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        work_desc = self.get_work_desc()
        seqlen = mK.shape[0]
        num_compute_tiles = cute.ceil_div(seqlen, self.tile_size)

        shared_storage = SmemAllocator().allocate(self.shared_storage_t)

        if warp_idx == 0 and lane_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_K)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_V)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_QSF)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_KSF)

            if cutlass.const_expr(self.use_tma_o):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_O)

        cg = pipeline.CooperativeGroup(pipeline.Agent.Thread)

        # Q/K data and their scale tiles share one TMA barrier per stage.
        sf_tile_bytes = cute.size_in_bytes(self.SF_dtype, cute.select(SF_smem_layout, mode=[0, 1]))
        Q_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=cg,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads // 32),
            tx_count=cute.size_in_bytes(self.Q_dtype, cute.select(Q_smem_layout, mode=[0, 1])) + sf_tile_bytes,
            barrier_storage=shared_storage.Q_barrier.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        Q_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        Q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        K_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.k_stage,
            producer_group=cg,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads // 32),
            tx_count=cute.size_in_bytes(self.K_dtype, cute.select(K_smem_layout, mode=[0, 1])) + sf_tile_bytes,
            barrier_storage=shared_storage.K_barrier.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        V_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=cg,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_threads // 32),
            tx_count=cute.size_in_bytes(self.V_dtype, cute.select(V_smem_layout, mode=[0, 1])),
            barrier_storage=shared_storage.V_barrier.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        K_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.k_stage)
        K_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.k_stage)
        V_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.kv_stage)
        V_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.kv_stage)

        # partition tensors
        # Q, K and V tiles are dead once the KV loop ends, so the BF16 output
        # tile reuses their bytes; this keeps three CTAs resident per SM.
        smem_base = shared_storage.QKVO_smem.data_ptr()
        sQ = cute.make_tensor(
            cute.recast_ptr(smem_base, Q_smem_layout.inner, dtype=self.Q_dtype),
            Q_smem_layout.outer,
        )
        sK = cute.make_tensor(
            cute.recast_ptr(smem_base + self.K_smem_offset, K_smem_layout.inner, dtype=self.K_dtype),
            K_smem_layout.outer,
        )
        sV = cute.make_tensor(
            cute.recast_ptr(smem_base + self.V_smem_offset, V_smem_layout.inner, dtype=self.V_dtype),
            V_smem_layout.outer,
        )
        sQSF = shared_storage.QSF_smem.get_tensor(SF_smem_layout)
        sKSF = shared_storage.KSF_smem.get_tensor(KSF_smem_layout)
        # MMA view of the same bytes: row-major (tile, head_dim) with one
        # scale broadcast over each 16-element head-dim group.
        sf_mma_layout = cute.make_layout(
            (self.tile_size, (self.sf_vec_size, self.num_sf_per_row), 1),
            stride=(self.num_sf_per_row, (0, 1), self.sf_tile_bytes),
        )
        sQSF_mma = cute.make_tensor(sQSF.iterator, sf_mma_layout)
        sKSF_mma = cute.make_tensor(sKSF.iterator, sf_mma_layout)
        # This K-major view only defines the FP8 B-fragment layout. The direct
        # transpose load below sources data from the N-major sV.
        sVt = cute.make_tensor(
            cute.recast_ptr(
                smem_base,
                Vt_smem_layout.inner,
                dtype=self.V_dtype,
            ),
            Vt_smem_layout.outer,
        )

        mO_slice = mO[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mLSE_slice = mLSE[None, work_desc.qo_head_idx, work_desc.batch_idx]
        mQ_slice = mQ[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mK_slice = mK[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        mV_slice = mV[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        mQSF_slice = mQSF[None, None, work_desc.qo_head_idx, work_desc.batch_idx]
        mKSF_slice = mKSF[None, None, work_desc.kv_head_idx, work_desc.batch_idx]
        gVScale = mVScale[None, work_desc.kv_head_idx]
        gO = cute.local_tile(mO_slice, (self.tile_shape_pv[0], self.tile_shape_pv[1]), coord=(work_desc.qo_tile_idx, 0))
        gQ = cute.local_tile(mQ_slice, (self.tile_shape_qk[0], self.tile_shape_qk[2]), coord=(work_desc.qo_tile_idx, 0))
        gK = cute.local_tile(mK_slice, (self.tile_shape_qk[1], self.tile_shape_qk[2]), coord=(None, 0))
        gV = cute.local_tile(mV_slice, (self.tile_shape_pv[1], self.tile_shape_pv[2]), coord=(0, None))
        gQSF = cute.local_tile(mQSF_slice, self.sf_tma_tile, coord=(0, work_desc.qo_tile_idx))
        gKSF = cute.local_tile(mKSF_slice, self.sf_tma_tile, coord=(0, None))

        gIndices = blocksparse_indices_q2k[None, work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx]
        if cutlass.const_expr(self.has_block_nums):
            num_n_tiles = blocksparse_num_blocks_q2k[work_desc.qo_tile_idx, work_desc.qo_head_idx, work_desc.batch_idx]
        else:
            num_n_tiles = block_sparse_num
        if cutlass.const_expr(self.has_block_sizes):
            if cutlass.const_expr(self.block_sizes_mode == 1):
                gBSZ = blocksparse_varblk
            elif cutlass.const_expr(self.block_sizes_mode == 2):
                gBSZ = blocksparse_varblk[None, work_desc.batch_idx]
            else:
                gBSZ = blocksparse_varblk[None, work_desc.qo_head_idx, work_desc.batch_idx]

        cta_coord_layout = (0, cute.make_layout(1))  # CTA coord layout for TMA multicasting, effectively no multicast
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            *cta_coord_layout,
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            *cta_coord_layout,
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            *cta_coord_layout,
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )
        tQSFsQSF, tQSFgQSF = cute.nvgpu.cpasync.tma_partition(
            tma_atom_QSF,
            *cta_coord_layout,
            cute.group_modes(sQSF, 0, 2),
            cute.group_modes(gQSF, 0, 2),
        )
        tKSFsKSF, tKSFgKSF = cute.nvgpu.cpasync.tma_partition(
            tma_atom_KSF,
            *cta_coord_layout,
            cute.group_modes(sKSF, 0, 2),
            cute.group_modes(gKSF, 0, 2),
        )

        cS = cute.make_identity_tensor(self.tile_shape_qk[:2])

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        tSsQ = thr_mma_qk.partition_A(sQ)
        tSsK = thr_mma_qk.partition_B(sK)
        tSrQ = tiled_mma_qk.make_fragment_A(tSsQ[None, None, None, 0])
        tSrK = tiled_mma_qk.make_fragment_B(tSsK[None, None, None, 0])
        tSrS = cute.make_rmem_tensor(thr_mma_qk.partition_shape_C((self.tile_shape_qk[0], self.tile_shape_qk[1])), self.acc_dtype)
        tScS = thr_mma_qk.partition_C(cS)
        tSsQSF, tSrQSF = _partition_sfa(sQSF_mma[None, None, 0], tiled_mma_qk, tidx)
        tSsKSF, tSrKSF = _partition_sfb(sKSF_mma[None, None, 0], tiled_mma_qk, tidx)

        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tOsV = thr_mma_pv.partition_B(sVt)
        tOrV = tiled_mma_pv.make_fragment_B(tOsV[None, None, None, 0])
        tOrO = cute.make_rmem_tensor(thr_mma_pv.partition_shape_C((self.tile_shape_pv[0], self.tile_shape_pv[1])), self.acc_dtype)
        cO = cute.make_identity_tensor(self.tile_shape_pv[:2])
        tOcO = thr_mma_pv.partition_C(cO)

        atom_copy_ldmatrix_Q = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False,
                num_matrices=4,
            ),
            self.Q_dtype,
        )
        atom_copy_ldmatrix_K = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False,
                num_matrices=4,
            ),
            self.K_dtype,
        )
        atom_copy_ldmatrix_V = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix16x16x8bOp(
                transpose=True,
                num_matrices=2,
            ),
            self.V_dtype,
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(atom_copy_ldmatrix_Q, tiled_mma_qk)
        smem_tiled_copy_K = cute.make_tiled_copy_B(atom_copy_ldmatrix_K, tiled_mma_qk)

        thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        tSsQ_copy = thr_copy_Q.partition_S(sQ)
        tSrQ_copy = thr_copy_Q.retile(tSrQ)
        tSsK_copy = thr_copy_K.partition_S(sK)
        tSrK_copy = thr_copy_K.retile(tSrK)

        max_m_layout = cute.make_layout(cute.size(layout_utils.reshape_acc_to_mn(tOrO).layout, mode=[0]))
        max_m = cute.make_rmem_tensor_like(max_m_layout, cutlass.Float32)
        sum_m = cute.make_rmem_tensor_like(max_m, cutlass.Float32)

        tOrO.store(cute.full_like(tOrO, 0.0, self.acc_dtype))
        max_m.store(cute.full_like(max_m, float("-inf"), cutlass.Float32))
        sum_m.store(cute.full_like(sum_m, 0.0, cutlass.Float32))

        if warp_idx == 0:
            Q_pipeline.producer_acquire(Q_producer_state)
            cute.copy(
                tma_atom_Q,
                tQgQ,
                tQsQ[None, 0],
                tma_bar_ptr=Q_pipeline.producer_get_barrier(Q_producer_state),
            )
            cute.copy(
                tma_atom_QSF,
                tQSFgQSF,
                tQSFsQSF[None, 0],
                tma_bar_ptr=Q_pipeline.producer_get_barrier(Q_producer_state),
            )
            Q_pipeline.producer_commit(Q_producer_state)
            Q_producer_state.advance()

            for preload_count in cutlass.range_constexpr(self.k_stage):
                if preload_count < num_n_tiles:
                    physical_idx = gIndices[num_n_tiles - 1 - preload_count]
                    K_pipeline.producer_acquire(K_producer_state)
                    cute.copy(
                        tma_atom_K,
                        tKgK[None, physical_idx],
                        tKsK[None, K_producer_state.index],
                        tma_bar_ptr=K_pipeline.producer_get_barrier(K_producer_state),
                    )
                    cute.copy(
                        tma_atom_KSF,
                        tKSFgKSF[None, physical_idx],
                        tKSFsKSF[None, K_producer_state.index],
                        tma_bar_ptr=K_pipeline.producer_get_barrier(K_producer_state),
                    )
                    K_pipeline.producer_commit(K_producer_state)
                    K_producer_state.advance()
                    if cutlass.const_expr(preload_count == 0):
                        V_pipeline.producer_acquire(V_producer_state)
                        cute.copy(
                            tma_atom_V,
                            tVgV[None, physical_idx],
                            tVsV[None, V_producer_state.index],
                            tma_bar_ptr=V_pipeline.producer_get_barrier(V_producer_state),
                        )
                        V_pipeline.producer_commit(V_producer_state)
                        V_producer_state.advance()

        cute.arch.sync_threads()

        # Q and its scales stay in registers for the whole KV loop.
        Q_wait_status = Q_pipeline.consumer_try_wait(Q_consumer_state)
        Q_pipeline.consumer_wait(Q_consumer_state, Q_wait_status)
        tQsQ_p = tSsQ_copy[None, None, None, 0]
        for k_block_idx in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
            tQsQ_k = tQsQ_p[None, None, k_block_idx]
            tQsQ_k = cute.make_tensor(
                tQsQ_k.iterator.align(16),
                tQsQ_k.layout,
            )
            cute.copy(
                smem_tiled_copy_Q,
                tQsQ_k,
                tSrQ_copy[None, None, k_block_idx],
            )
        _load_sf_fragment(tSsQSF, tSrQSF)
        Q_pipeline.consumer_release(Q_consumer_state)
        Q_consumer_state.advance()

        for load_count in cutlass.range(0, num_n_tiles, 1, unroll=1):
            n_tile_ind = num_n_tiles - 1 - load_count
            n_tile_idx = gIndices[n_tile_ind]
            if cutlass.const_expr(self.has_block_sizes):
                varblk = gBSZ[n_tile_idx]
            else:
                varblk = cutlass.Int32(self.tile_size)
                if n_tile_idx == num_compute_tiles - 1:
                    varblk = seqlen - n_tile_idx * self.tile_size

            K_wait_status = K_pipeline.consumer_try_wait(K_consumer_state)
            K_pipeline.consumer_wait(K_consumer_state, K_wait_status)
            k_stage = K_consumer_state.index
            for k_block_idx in cutlass.range_constexpr(cute.size(tSrK, mode=[2])):
                tKsK_k = tSsK_copy[None, None, k_block_idx, k_stage]
                tKsK_k = cute.make_tensor(
                    tKsK_k.iterator.align(16),
                    tKsK_k.layout,
                )
                cute.copy(
                    smem_tiled_copy_K,
                    tKsK_k,
                    tSrK_copy[None, None, k_block_idx],
                )
            _load_sf_fragment(
                cute.make_tensor(tSsKSF.iterator + k_stage * self.sf_tile_bytes, tSsKSF.layout),
                tSrKSF,
            )
            K_pipeline.consumer_release(K_consumer_state)
            K_consumer_state.advance()

            if warp_idx == 0 and load_count + self.k_stage < num_n_tiles:
                preload_physical_k = gIndices[n_tile_ind - self.k_stage]
                K_pipeline.producer_acquire(K_producer_state)
                cute.copy(
                    tma_atom_K,
                    tKgK[None, preload_physical_k],
                    tKsK[None, K_producer_state.index],
                    tma_bar_ptr=K_pipeline.producer_get_barrier(K_producer_state),
                )
                cute.copy(
                    tma_atom_KSF,
                    tKSFgKSF[None, preload_physical_k],
                    tKSFsKSF[None, K_producer_state.index],
                    tma_bar_ptr=K_pipeline.producer_get_barrier(K_producer_state),
                )
                K_pipeline.producer_commit(K_producer_state)
                K_producer_state.advance()

            _gemm_nvfp4_qk(mma_atom_qk, tSrS, tSrQ, tSrK, tSrQSF, tSrKSF)

            if varblk < self.tile_size:
                _mask_fp8(tiled_mma_qk, tSrS, tScS, varblk)
            row_scale = _online_softmax_nvfp4(
                tSrS,
                max_m,
                sum_m,
                scale_softmax_log2e,
                self.softmax_p_scale_log2,
            )

            # Compute P @ V.
            _rescale_o_for_next_acc_fp8(tiled_mma_pv, tOrO, row_scale)
            tOrP = _make_acc_into_fp8_op(
                tSrS,
                tiled_mma_pv.tv_layout_A,
                self.V_dtype,
            )

            V_wait_status = V_pipeline.consumer_try_wait(V_consumer_state)
            V_pipeline.consumer_wait(V_consumer_state, V_wait_status)

            # Load each 16x32 N-major V subtile with the native 8-bit
            # transpose path; the LDSM.T result directly forms the FP8 B
            # fragment of the SM120 warp MMA.
            destination_layout = cute.make_layout(
                ((4, (2, 2)),),
                stride=((1, (16, 4)),),
            )
            for k_block in cutlass.range_constexpr(2):
                for n_block in cutlass.range_constexpr(8):
                    source_ptr = sV.iterator + cute.crd2idx(
                        (n_block * 16, k_block * 32 + lane_idx, 0),
                        sV.layout,
                    )
                    source = cute.make_tensor(
                        source_ptr.align(16),
                        cute.make_layout(16),
                    )
                    block_idx = k_block * 8 + n_block
                    destination_base = (block_idx % 8) * 8 + (block_idx // 8) * 2
                    destination = cute.make_tensor(
                        tOrV.iterator + destination_base * 4,
                        destination_layout,
                    )
                    cute.copy(
                        atom_copy_ldmatrix_V,
                        source,
                        destination,
                    )

            _gemm_rs_fp8(
                tiled_mma_pv,
                tOrO,
                tOrP,
                tOrV,
                None,
                None,
                B_in_regs=True,
            )
            V_pipeline.consumer_release(V_consumer_state)
            V_consumer_state.advance()

            if warp_idx == 0 and load_count + 1 < num_n_tiles:
                V_pipeline.producer_acquire(V_producer_state)
                cute.copy(
                    tma_atom_V,
                    tVgV[None, gIndices[n_tile_ind - 1]],
                    tVsV[None, V_producer_state.index],
                    tma_bar_ptr=V_pipeline.producer_get_barrier(V_producer_state),
                )
                V_pipeline.producer_commit(V_producer_state)
                V_producer_state.advance()

        final_ratio, lse = _finalize_softmax_nvfp4(
            max_m,
            sum_m,
            scale_softmax_log2e,
            self.softmax_p_scale_log2,
        )
        _rescale_o_with_sage_v_scale_fp8(
            tOrO,
            tOcO,
            final_ratio,
            gVScale,
        )
        tScS_mn = layout_utils.reshape_acc_to_mn(tScS)
        for m in cutlass.range_constexpr(cute.size(lse)):
            row_idx = work_desc.qo_tile_idx * self.tile_size + tScS_mn[m, 0][0]
            if tScS_mn[m, 0][1] == 0:
                if row_idx < mQ.shape[0]:
                    mLSE_slice[row_idx] = lse[m]

        tOrO_cvt = cute.make_rmem_tensor_like(tOrO, self.O_dtype)
        tOrO_cvt.store(tOrO.load().to(self.O_dtype))

        # sO aliases sQ/sK/sV: wait until every warp has finished its last
        # V load before overwriting them.
        cute.arch.sync_threads()
        sO = cute.make_tensor(
            cute.recast_ptr(smem_base, O_smem_layout.inner, dtype=self.O_dtype),
            O_smem_layout.outer,
        )
        tiled_copy_o_r2s = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(self.O_layout.is_m_major_c(), 4),
                self.O_dtype,
            ),
            tiled_mma_pv,
        )
        tOrO_cv = tiled_copy_o_r2s.retile(tOrO_cvt)
        tOsO = tiled_copy_o_r2s.get_slice(tidx).partition_D(sO)
        cute.copy(tiled_copy_o_r2s, tOrO_cv, tOsO)

        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        # S2G with explicit TMA store wait.
        tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
            tma_atom_O,
            *cta_coord_layout,
            cute.group_modes(sO, 0, 2),
            cute.group_modes(gO, 0, 2),
        )
        if warp_idx == 0:
            cute.copy(tma_atom_O, tOsO, tOgO)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (head_dim / 2, seqlen, nheads, batch) packed E2M1 bytes
        mK: cute.Tensor,  # (head_dim / 2, seqlen, nheads, batch) packed E2M1 bytes
        mV: cute.Tensor,  # (value_dim, seqlen, nheads, batch) FP8
        mO: cute.Tensor,  # (value_dim, seqlen, nheads, batch)
        mLSE: cute.Tensor,  # (seqlen, nheads, batch)
        mQSF: cute.Tensor,  # (16, padded_seqlen_q * head_dim / 256, nheads_q, batch) E4M3
        mKSF: cute.Tensor,  # (16, padded_seqlen_k * head_dim / 256, nheads_kv, batch) E4M3
        mVScale: cute.Tensor,  # (value_dim, nheads_kv)
        blocksparse_indices_q2k: cute.Tensor,  # (k, q, nheads, batch)
        blocksparse_num_blocks_q2k: cute.Tensor,  # (q, nheads, batch)
        block_sparse_num: cutlass.Int32,
        blocksparse_varblk: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        # Reinterpret packed bytes as E2M1 and restore compile-time head
        # dimensions while keeping runtime tensor modes dynamic.
        mQ = cute.recast_tensor(mQ, cutlass.Float4E2M1FN)
        mK = cute.recast_tensor(mK, cutlass.Float4E2M1FN)
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (mQ.shape[0], self.qk_dim, mQ.shape[2], mQ.shape[3]),
                stride=mQ.stride,
            ),
        )
        mK = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (mK.shape[0], self.qk_dim, mK.shape[2], mK.shape[3]),
                stride=mK.stride,
            ),
        )
        mV = cute.make_tensor(
            mV.iterator,
            cute.make_layout(
                (self.value_dim, mV.shape[1], mV.shape[2], mV.shape[3]),
                stride=mV.stride,
            ),
        )
        mO = cute.make_tensor(
            mO.iterator,
            cute.make_layout(
                (mO.shape[0], self.value_dim, mO.shape[2], mO.shape[3]),
                stride=mO.stride,
            ),
        )
        mQSF = cute.make_tensor(
            mQSF.iterator,
            cute.make_layout(
                (NVFP4_SF_TMA_COLS, mQSF.shape[1], mQSF.shape[2], mQSF.shape[3]),
                stride=mQSF.stride,
            ),
        )
        mKSF = cute.make_tensor(
            mKSF.iterator,
            cute.make_layout(
                (NVFP4_SF_TMA_COLS, mKSF.shape[1], mKSF.shape[2], mKSF.shape[3]),
                stride=mKSF.stride,
            ),
        )
        mVScale = cute.make_tensor(
            mVScale.iterator,
            cute.make_layout(
                (self.value_dim, mVScale.shape[1]),
                stride=mVScale.stride,
            ),
        )
        self.check_dim([mQ, mK, mO], 1)
        self.check_dim(mV, 0)

        Q_layout = LayoutEnum.from_tensor(mQ)
        K_layout = LayoutEnum.from_tensor(mK)
        V_layout = LayoutEnum.from_tensor(mV)
        O_layout = LayoutEnum.from_tensor(mO)

        self.Q_dtype = mQ.element_type
        self.K_dtype = mK.element_type
        self.V_dtype = mV.element_type
        self.O_dtype = mO.element_type
        self.SF_dtype = mQSF.element_type
        assert self.Q_dtype == self.K_dtype == cutlass.Float4E2M1FN
        assert self.V_dtype == cutlass.Float8E4M3FN
        assert self.SF_dtype == mKSF.element_type == cutlass.Float8E4M3FN
        assert self.O_dtype is cutlass.BFloat16
        self.Q_layout = Q_layout
        self.K_layout = K_layout
        self.V_layout = V_layout
        self.O_layout = O_layout

        # Both MMAs tile four warps along M and pair two n8 atoms along N, so
        # the QK accumulator has exactly the FP8 kernel's C layout and the
        # FP8 P conversion applies unchanged.
        atom_layout_mnk = (4, 1, 1)
        mma_op_qk = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.Q_dtype,
            self.acc_dtype,
            self.SF_dtype,
        )
        qk_inst_mnk = mma_op_qk.shape_mnk
        tiled_mma_qk = cute.make_tiled_mma(
            mma_op_qk,
            cute.make_layout(atom_layout_mnk),
            permutation_mnk=(
                atom_layout_mnk[0] * qk_inst_mnk[0],
                atom_layout_mnk[1] * qk_inst_mnk[1] * 2,
                atom_layout_mnk[2] * qk_inst_mnk[2],
            ),
        )
        mma_atom_qk = cute.make_mma_atom(mma_op_qk)
        pv_inst_mnk = (16, 8, 32)
        mma_op_pv = cute.nvgpu.warp.MmaFP8Op(
            self.V_dtype,
            self.acc_dtype,
            pv_inst_mnk,
        )
        tiled_mma_pv = cute.make_tiled_mma(
            mma_op_pv,
            cute.make_layout(atom_layout_mnk),
            permutation_mnk=(
                atom_layout_mnk[0] * pv_inst_mnk[0],
                atom_layout_mnk[1] * pv_inst_mnk[1] * 2,
                atom_layout_mnk[2] * pv_inst_mnk[2],
            ),
        )

        self.Q_smem_layout = sm90_utils.make_smem_layout_a(
            Q_layout,
            self.tile_shape_qk,
            self.Q_dtype,
            self.q_stage,
        )
        self.K_smem_layout = sm90_utils.make_smem_layout_b(
            K_layout,
            self.tile_shape_qk,
            self.K_dtype,
            self.k_stage,
        )
        self.V_smem_layout = sm90_utils.make_smem_layout_b(
            V_layout,
            self.tile_shape_pv,
            self.V_dtype,
            self.kv_stage,
        )
        vt_smem_layout_atom = cute.nvgpu.tcgen05.make_smem_layout_atom(
            cute.nvgpu.tcgen05.SmemLayoutAtomKind.K_SW64,
            self.V_dtype,
        )
        self.Vt_smem_layout = cute.tile_to_shape(
            vt_smem_layout_atom,
            (*self.tile_shape_pv[1:], self.kv_stage),
            order=(0, 1, 2),
        )
        O_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.O_dtype,
            O_layout,
            self.tile_shape_pv[:2],
            1,
        )
        self.O_smem_layout = cute.select(O_smem_layout_staged, mode=[0, 1])
        self.SF_smem_layout = cute.make_layout(
            (*self.sf_tma_tile, self.q_stage),
            stride=(1, NVFP4_SF_TMA_COLS, self.sf_tile_bytes),
        )
        self.KSF_smem_layout = cute.make_layout(
            (*self.sf_tma_tile, self.k_stage),
            stride=(1, NVFP4_SF_TMA_COLS, self.sf_tile_bytes),
        )

        def _aligned_bytes(dtype, layout):
            return (cute.size_in_bytes(dtype, layout) + 1023) // 1024 * 1024

        self.K_smem_offset = _aligned_bytes(self.Q_dtype, self.Q_smem_layout)
        self.V_smem_offset = self.K_smem_offset + _aligned_bytes(self.K_dtype, self.K_smem_layout)
        qkvo_bytes = max(
            self.V_smem_offset + _aligned_bytes(self.V_dtype, self.V_smem_layout),
            _aligned_bytes(self.O_dtype, self.O_smem_layout),
        )

        @cute.struct
        class SharedStorage:
            Q_barrier: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            K_barrier: cute.struct.MemRange[cutlass.Int64, self.k_stage * 2]
            V_barrier: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]

            QKVO_smem: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, qkvo_bytes], 1024]
            QSF_smem: cute.struct.Align[cute.struct.MemRange[self.SF_dtype, cute.cosize(self.SF_smem_layout)], 128]
            KSF_smem: cute.struct.Align[cute.struct.MemRange[self.SF_dtype, cute.cosize(self.KSF_smem_layout)], 128]

        self.shared_storage_t = SharedStorage

        tma_copy_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mQ,
            self.Q_smem_layout,
            (self.tile_shape_qk[0], self.tile_shape_qk[2]),
            num_multicast=1,
        )
        tma_atom_K, tma_tensor_K = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mK,
            self.K_smem_layout,
            (self.tile_shape_qk[1], self.tile_shape_qk[2]),
            num_multicast=1,
        )
        tma_atom_V, tma_tensor_V = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mV,
            self.V_smem_layout,
            (self.tile_shape_pv[1], self.tile_shape_pv[2]),
            num_multicast=1,
        )
        sf_tile_layout = cute.select(self.SF_smem_layout, mode=[0, 1])
        tma_atom_QSF, tma_tensor_QSF = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mQSF,
            sf_tile_layout,
            self.sf_tma_tile,
            num_multicast=1,
        )
        tma_atom_KSF, tma_tensor_KSF = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mKSF,
            sf_tile_layout,
            self.sf_tma_tile,
            num_multicast=1,
        )

        tma_copy_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_O, tma_tensor_O = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_copy_op,
            mO,
            self.O_smem_layout,
            (self.tile_shape_pv[0], self.tile_shape_pv[1]),
            num_multicast=1,
        )

        log2_e = 1.44269504088896340736

        grid_config = self.get_grid_config(mQ.shape[0], mQ.shape[2], mQ.shape[3])
        block_config = (self.num_threads, 1, 1)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_O,
            mLSE,
            tma_tensor_QSF,
            tma_tensor_KSF,
            mVScale,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            tma_atom_QSF,
            tma_atom_KSF,
            blocksparse_indices_q2k,
            blocksparse_num_blocks_q2k,
            block_sparse_num,
            blocksparse_varblk,
            tiled_mma_qk,
            mma_atom_qk,
            tiled_mma_pv,
            self.Q_smem_layout,
            self.K_smem_layout,
            self.V_smem_layout,
            self.Vt_smem_layout,
            self.O_smem_layout,
            self.SF_smem_layout,
            self.KSF_smem_layout,
            softmax_scale * log2_e,
        ).launch(
            grid=grid_config,
            block=block_config,
            smem=self.shared_storage_t.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )


# =============================================================================
# Local CuTe helpers
# =============================================================================
#
# The SM120 block-scaled warp MMA reads its scale factors from registers; the
# partitioning below follows CUTLASS's SM120 BlockScaled SFA/SFB thread-value
# layouts (cute/atom/mma_traits_sm120.hpp).


def _thrfrg_sfa(sfa_layout, tiled_mma: cute.TiledMma):
    atom_shape_mnk = tiled_mma.shape_mnk
    atom_sfa_layout = cute.make_layout(shape=((2, 2, 8), 64), stride=((8, 0, 1), 16))
    permutation_mnk = tiled_mma.permutation_mnk
    thr_layout_vmnk = tiled_mma.thr_layout_vmnk
    t_tensor = cute.logical_divide(sfa_layout, (permutation_mnk[0], permutation_mnk[2]))
    a_tensor = cute.zipped_divide(
        t_tensor,
        (cute.make_layout(atom_shape_mnk[0]), cute.make_layout(atom_shape_mnk[2])),
    )
    tv_tensor = cute.composition(a_tensor, (atom_sfa_layout, None))
    thr_tile = (
        None,
        (
            cute.make_layout(cute.size(thr_layout_vmnk[1])),
            cute.make_layout(cute.size(thr_layout_vmnk[3])),
        ),
    )
    return cute.zipped_divide(tv_tensor, thr_tile)


def _thrfrg_sfb(sfb_layout, tiled_mma: cute.TiledMma):
    atom_shape_mnk = tiled_mma.shape_mnk
    atom_sfb_layout = cute.make_layout(shape=((4, 8), 64), stride=((0, 1), 8))
    permutation_mnk = tiled_mma.permutation_mnk
    thr_layout_vmnk = tiled_mma.thr_layout_vmnk
    t_tensor = cute.logical_divide(sfb_layout, (permutation_mnk[1], permutation_mnk[2]))
    a_tensor = cute.zipped_divide(
        t_tensor,
        (cute.make_layout(atom_shape_mnk[1]), cute.make_layout(atom_shape_mnk[2])),
    )
    tv_tensor = cute.composition(a_tensor, (atom_sfb_layout, None))
    thr_tile = (
        None,
        (
            cute.make_layout(cute.size(thr_layout_vmnk[2])),
            cute.make_layout(cute.size(thr_layout_vmnk[3])),
        ),
    )
    return cute.zipped_divide(tv_tensor, thr_tile)


@cute.jit
def _partition_sfa(sfa_tensor: cute.Tensor, tiled_mma: cute.TiledMma, tidx):
    """Return this thread's (V, M, K) SFA view of ``sfa_tensor`` and a matching register fragment."""
    thr_tensor = cute.make_tensor(sfa_tensor.iterator, _thrfrg_sfa(sfa_tensor.layout, tiled_mma))
    thr_vmnk = tiled_mma.thr_layout_vmnk.get_flat_coord(tidx)
    part = thr_tensor[(thr_vmnk[0], (thr_vmnk[1], thr_vmnk[3])), (None, None)]
    part = cute.group_modes(cute.flatten(part), 0, 2)
    return part, cute.make_fragment_like(part)


@cute.jit
def _partition_sfb(sfb_tensor: cute.Tensor, tiled_mma: cute.TiledMma, tidx):
    """Return this thread's (V, N, K) SFB view of ``sfb_tensor`` and a matching register fragment."""
    thr_tensor = cute.make_tensor(sfb_tensor.iterator, _thrfrg_sfb(sfb_tensor.layout, tiled_mma))
    thr_vmnk = tiled_mma.thr_layout_vmnk.get_flat_coord(tidx)
    part = thr_tensor[(thr_vmnk[0], (thr_vmnk[2], thr_vmnk[3])), (None, None)]
    part = cute.group_modes(cute.flatten(part), 0, 2)
    # A layout-valued N permutation splits N into sub-modes; regroup them so
    # the view is (V, N, K) like the A side.
    if cutlass.const_expr(cute.rank(part) > 3):
        part = cute.group_modes(part, 1, cute.rank(part) - 1)
    return part, cute.make_fragment_like(part)


@cute.jit
def _load_sf_fragment(src: cute.Tensor, dst: cute.Tensor) -> None:
    """Copy the distinct scale bytes of a thread's SF view into registers."""
    src_vals = cute.filter_zeros(src)
    dst_vals = cute.filter_zeros(dst)
    for i in cutlass.range_constexpr(cute.size(dst_vals)):
        dst_vals[i] = src_vals[i]


@cute.jit
def _gemm_nvfp4_qk(
    mma_atom: cute.MmaAtom,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCrSFA: cute.Tensor,
    tCrSFB: cute.Tensor,
) -> None:
    """S = Q K^T with per-16-element scales applied by the MMA."""
    # ``set`` rebinds the atom's SSA value; use a local atom so a call inside
    # a dynamic branch does not leak that value out of the branch.
    mma_atom = cute.make_mma_atom(mma_atom.op)
    acc.fill(0.0)
    for k_block_idx in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
        for m_idx in cutlass.range_constexpr(cute.size(tCrA, mode=[1])):
            for n_idx in cutlass.range_constexpr(cute.size(tCrB, mode=[1])):
                mma_atom.set(WarpField.SFA, tCrSFA[None, m_idx, k_block_idx].iterator)
                mma_atom.set(WarpField.SFB, tCrSFB[None, n_idx, k_block_idx].iterator)
                cute.gemm(
                    mma_atom,
                    acc[None, m_idx, n_idx],
                    tCrA[None, m_idx, k_block_idx],
                    tCrB[None, n_idx, k_block_idx],
                    acc[None, m_idx, n_idx],
                )


@cute.jit
def _online_softmax_nvfp4(
    tSrS: cute.Tensor,
    row_max: cute.Tensor,
    row_sum: cute.Tensor,
    softmax_scale_log2e: cutlass.Float32,
    exp_scale_log2: cutlass.Float32,
) -> cute.Tensor:
    tSrS_mn = layout_utils.reshape_acc_to_mn(tSrS)
    row_scale = cute.make_rmem_tensor_like(row_max, cutlass.Float32)

    for m in cutlass.range(cute.size(row_max), unroll_full=True):
        row_max_local = row_max[m]
        for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
            row_max_local = cute.arch.fmax(row_max_local, tSrS_mn[m, n])
        row_max_cur = cute.arch.warp_reduction_max(
            row_max_local,
            threads_in_group=4,
        )

        row_max_prev = row_max[m]
        row_max[m] = row_max_cur
        row_max_safe = cutlass.Float32(0.0) if row_max_cur == -cutlass.Float32.inf else row_max_cur
        # Shift the exponent by log2(P scale) before FP8 conversion. Keeping
        # row_sum in the same scale removes a vector multiply from every tile.
        row_max_scaled = row_max_safe * softmax_scale_log2e - exp_scale_log2
        for n in cutlass.range_constexpr(cute.size(tSrS_mn, mode=[1])):
            tSrS_mn[m, n] = cute.math.exp2(
                tSrS_mn[m, n] * softmax_scale_log2e - row_max_scaled,
                fastmath=True,
            )
        acc_S_row_exp = tSrS_mn[m, None].load()
        row_scale[m] = cute.math.exp2(
            (row_max_prev - row_max_safe) * softmax_scale_log2e,
            fastmath=True,
        )
        row_sum[m] = kernel_utils.fadd_reduce(
            acc_S_row_exp,
            init_val=row_sum[m] * row_scale[m],
            arch=80,
        )
        tSrS_mn[m, None].store(acc_S_row_exp)

    return row_scale


@cute.jit
def _finalize_softmax_nvfp4(
    row_max: cute.Tensor,
    row_sum: cute.Tensor,
    softmax_scale_log2e: cutlass.Float32,
    exp_scale_log2: cutlass.Float32,
) -> cute.Tensor:
    row_sum.store(kernel_utils.warp_reduce(row_sum.load(), operator.add, width=4))
    final_ratio = cute.make_rmem_tensor_like(row_sum, cutlass.Float32)
    lse = cute.make_rmem_tensor_like(row_sum, cutlass.Float32)

    for m in cutlass.range(cute.size(row_sum), unroll_full=True):
        final_sum = row_sum[m]
        is_zero_or_nan = final_sum == 0.0 or final_sum != final_sum
        final_ratio[m] = cute.arch.rcp_approx(final_sum if not is_zero_or_nan else 1.0)

        ln2 = 0.693147180559945309417
        lse[m] = (
            -cutlass.Float32.inf if is_zero_or_nan else (row_max[m] * softmax_scale_log2e + cute.math.log2(final_sum, fastmath=True) - exp_scale_log2) * ln2
        )

    return final_ratio, lse
