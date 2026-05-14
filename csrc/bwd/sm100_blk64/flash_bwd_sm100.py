import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32, Int64, BFloat16, Boolean
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from quack import copy_utils
from csrc.common.mask import apply_block_size_mask

from typing import Tuple, Type

import math


@dsl_user_op
def cpasync_reduce_bulk_add_f32(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    store_bytes: int | Int32,
    *,
    loc=None,
    ip=None,
):
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value()],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class BlockSparseAttnBackward:
    def __init__(
        self,
        sparse_block_size: int,
        has_block_sizes: bool = True,
        full_kv_blocks: bool = False,
        use_dkv_tma_reduce: bool = True,
        use_dkv_stage_layout: bool = True,
    ):
        self.sparse_block_size = sparse_block_size
        self.has_block_sizes = has_block_sizes
        self.full_kv_blocks = full_kv_blocks
        self.use_dkv_tma_reduce = use_dkv_tma_reduce
        self.use_dkv_stage_layout = use_dkv_stage_layout

        self.QK_mma_tiler = (128,64,128)
        self.fake_QK_mma_tiler = (64,64,128)
        self.dOP_mma_tiler = (128,64,128)
        self.dOV_mma_tiler = (128,64,128)
        self.fake_dOV_mma_tiler = (64,64,128)
        self.dSK_mma_tiler = (128,128,64)
        self.fake_dSK_mma_tiler = (64,128,64)
        self.QdS_mma_tiler = (128,64,128)
        # Use the FlashAttention/CUTLASS dQAccm TMA-reduce stage width.  This
        # keeps the TMEM load fragment profile compatible with the vectorized
        # R2S path and lets TMA reduce write one 64x32 half-tile at a time.
        self.dQ_reduce_ncol = 32
        self.dKV_tmem_load_ncol = 16
        self.dKV_reduce_ncol = 32

        self.element_dtype = BFloat16
        self.acc_dtype = Float32

        # =================== Sum OdO ================================
        self.sum_OdO_max_threads_per_block = 128
        self.sum_OdO_block_q = 16
        self.sum_OdO_num_threads_d = 8
        self.sum_OdO_num_threads_q = (
            self.sum_OdO_max_threads_per_block // self.sum_OdO_num_threads_d
        )
        self.sum_OdO_elem_per_load = 2

        self.reduce_warp_id = (0, 1, 2, 3)
        self.compute_warp_id = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.empty_warp_id = 14

        self.num_reduce_warps = 4
        self.num_compute_warps = 8

        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * (
            self.num_reduce_warps + self.num_compute_warps + 4
        )

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * (self.num_compute_warps + 1 + self.num_reduce_warps),
        )
        self.compute_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.num_compute_warps * self.threads_per_warp,
        )
        self.epilogue_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.num_compute_warps * self.threads_per_warp,
        )
        self.reduce_sync_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=self.num_reduce_warps * self.threads_per_warp,
        )

        self.tmem_dK_offset = 0
        self.tmem_dV_offset = self.tmem_dK_offset + self.QdS_mma_tiler[1] # 64
        self.tmem_dQ_offset = self.tmem_dV_offset + self.dOP_mma_tiler[1] # 64 + 64 = 128 
        self.tmem_dP_offset = self.tmem_dQ_offset # 128
        self.tmem_S_offset = self.tmem_dP_offset + self.dSK_mma_tiler[1] # 128 + 128 = 256

        self.num_regs_reduce = 152
        self.num_regs_compute = 128
        self.num_regs_mma = 96
        self.num_regs_empty = 96
        self.num_regs_load = 96

        self.buffer_align_bytes = 1024
    
    def _setup_attributes(self):
        self.load_mma_Q_stage = 2
        self.load_mma_dO_stage = 1
        self.load_compute_LSE_stage = 1
        self.load_compute_sum_OdO_stage = 1
        self.mma_compute_S_stage = 1
        self.mma_compute_dP_stage = 1
        self.mma_reduce_dQ_stage = 1
        self.compute_mma_P_stage = 1
        self.compute_mma_dS_stage = 1
        self.mma_compute_dKdV_stage = 2
        self.reduce_tma_store_stage = 64 // self.dQ_reduce_ncol
    
    @staticmethod
    def _get_workspace_size(
        q: int, k: int, d: int, h: int, b: int, acc_dtype: Type[cutlass.Numeric]
    ):
        sparse_block_size = 64
        d = (d + 7) // 8 * 8  # round up to 8
        q_scalar = (q + 7) // 8 * 8  # round up to 8
        q_acc = (q + sparse_block_size - 1) // sparse_block_size * sparse_block_size
        k_acc = (k + sparse_block_size - 1) // sparse_block_size * sparse_block_size
        acc_bytes = acc_dtype.width // 8
        # Workspace holds, contiguously:
        #   - sum_OdO:    B * H * Q float32 values
        #   - scaled_lse: B * H * Q float32 values
        #   - dQ_acc:     B * H * round_up(Q, 64) * D float32 values
        #   - dK_acc:     B * H * round_up(K, 64) * D float32 values
        #   - dV_acc:     B * H * round_up(K, 64) * D float32 values
        # Accumulator tensors use stage-contiguous layouts and are remapped by
        # postprocess kernels before writing final dQ/dK/dV.
        # Total bytes = B * H * (2 * Q_scalar + Q_acc * D + 2 * K_acc * D) * acc_bytes.
        # Return a multi-dim uint8 shape so that no individual shape dim or
        # contiguous row-major stride exceeds int32 (the MLIR/CUTE layout
        # attribute type), even when the total byte count does.
        return (
            b,
            h,
            (2 * q_scalar + q_acc * d + 2 * k_acc * d) * acc_bytes,
        )
    
    def get_workspace_tensor(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        workspace: cute.Tensor,
        acc_dtype: Type[cutlass.Numeric],
    ) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
        Q, K, D, HB = (
            problem_shape[0],
            problem_shape[1],
            problem_shape[2],
            problem_shape[3],
        )
        H, B = cute.size(problem_shape[3][0]), cute.size(problem_shape[3][1])
        D = cute.round_up(D, 8)
        Q_scalar = cute.round_up(Q, 8)
        Q_acc = cute.ceil_div(Q, self.sparse_block_size) * self.sparse_block_size
        K_acc = cute.ceil_div(K, self.sparse_block_size) * self.sparse_block_size
        Q_blocks = cute.ceil_div(Q, self.sparse_block_size)
        K_blocks = cute.ceil_div(K, self.sparse_block_size)

        acc_bytes = acc_dtype.width // 8
        # Byte offsets used to split workspace into three sub-tensors. These
        # are small enough to fit Int32 (< 2^31), so Int32 arithmetic is fine
        # for the pointer stepping below.
        sum_OdO_bytes = cute.assume(B * H * Q_scalar * acc_bytes, divby=acc_bytes)
        scaled_lse_bytes = cute.assume(B * H * Q_scalar * acc_bytes, divby=acc_bytes)

        sum_OdO_iter = workspace.iterator
        scaled_lse_iter = sum_OdO_iter + sum_OdO_bytes
        dQ_acc_iter = scaled_lse_iter + scaled_lse_bytes
        dK_acc_iter = dQ_acc_iter + cute.assume(B * H * Q_acc * D * acc_bytes, divby=acc_bytes)
        dV_acc_iter = dK_acc_iter + cute.assume(B * H * K_acc * D * acc_bytes, divby=acc_bytes)

        sum_OdO_iter = cute.recast_ptr(sum_OdO_iter, dtype=self.acc_dtype)
        scaled_lse_iter = cute.recast_ptr(scaled_lse_iter, dtype=self.acc_dtype)
        dQ_acc_iter = cute.recast_ptr(dQ_acc_iter, dtype=self.acc_dtype)
        dK_acc_iter = cute.recast_ptr(dK_acc_iter, dtype=self.acc_dtype)
        dV_acc_iter = cute.recast_ptr(dV_acc_iter, dtype=self.acc_dtype)

        # Cast problem dims to Int64 so layout strides promote Int32 indices
        # to Int64 when computing flat element offsets. Without this, the
        # last-element offset in dQ_acc (B*H*Q*D) can exceed 2^31 for large
        # Q (e.g. num_blocks=10860 with default B=4, H=8), silently wrapping
        # to a negative value and causing illegal memory accesses in the
        # convert kernel that iterates over dQ_acc.
        Q_scalar_i64 = Int64(Q_scalar)
        Q_blocks_i64 = Int64(Q_blocks)
        K_blocks_i64 = Int64(K_blocks)
        H_i64 = Int64(H)
        q_acc_block_elems = Int64(self.sparse_block_size * D)
        k_acc_block_elems = Int64(self.sparse_block_size * D)
        dKV_stage_elems = Int64(self.sparse_block_size * self.dKV_reduce_ncol)

        sum_OdO = cute.make_tensor(
            sum_OdO_iter,
            cute.make_layout(
                (Q_scalar, (H, B)),
                stride=(1, (Q_scalar_i64, Q_scalar_i64 * H_i64)),
            ),
        )
        scaled_lse = cute.make_tensor(
            scaled_lse_iter,
            cute.make_layout(
                (Q_scalar, (H, B)),
                stride=(1, (Q_scalar_i64, Q_scalar_i64 * H_i64)),
            ),
        )
        dQ_acc = cute.make_tensor(
            dQ_acc_iter,
            cute.make_layout(
                (Q_acc, D, (H, B)),
                stride=(D, 1, (q_acc_block_elems * Q_blocks_i64, q_acc_block_elems * Q_blocks_i64 * H_i64)),
            ),
        )
        if self.use_dkv_stage_layout:
            dK_acc = cute.make_tensor(
                dK_acc_iter,
                cute.make_layout(
                    (
                        self.sparse_block_size * self.dKV_reduce_ncol,
                        D // self.dKV_reduce_ncol,
                        K_blocks,
                        (H, B),
                    ),
                    stride=(
                        1,
                        dKV_stage_elems,
                        k_acc_block_elems,
                        (k_acc_block_elems * K_blocks_i64, k_acc_block_elems * K_blocks_i64 * H_i64),
                    ),
                ),
            )
            dV_acc = cute.make_tensor(
                dV_acc_iter,
                cute.make_layout(
                    (
                        self.sparse_block_size * self.dKV_reduce_ncol,
                        D // self.dKV_reduce_ncol,
                        K_blocks,
                        (H, B),
                    ),
                    stride=(
                        1,
                        dKV_stage_elems,
                        k_acc_block_elems,
                        (k_acc_block_elems * K_blocks_i64, k_acc_block_elems * K_blocks_i64 * H_i64),
                    ),
                ),
            )
        else:
            dK_acc = cute.make_tensor(
                dK_acc_iter,
                cute.make_layout(
                    (D, K_acc, (H, B)),
                    stride=(1, D, (k_acc_block_elems * K_blocks_i64, k_acc_block_elems * K_blocks_i64 * H_i64)),
                ),
            )
            dV_acc = cute.make_tensor(
                dV_acc_iter,
                cute.make_layout(
                    (D, K_acc, (H, B)),
                    stride=(1, D, (k_acc_block_elems * K_blocks_i64, k_acc_block_elems * K_blocks_i64 * H_i64)),
                ),
            )

        return sum_OdO, scaled_lse, dQ_acc, dQ_acc, dK_acc, dV_acc
    
    @staticmethod
    def _compute_sum_OdO_grid(
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        block_q: int,
    ) -> Tuple[int, int, int]:
        grid = (
            cute.ceil_div(cute.size(problem_shape[0]), block_q),
            cute.size(problem_shape[3][0]),  # H
            cute.size(problem_shape[3][1]),  # B
        )
        return grid

    def _compute_bwd_grid(self, problem_shape, task_offsets: cute.Tensor):
        H, B = problem_shape[3][0], problem_shape[3][1]
        work_capacity_per_bh = cute.size(task_offsets.shape[0])
        return (work_capacity_per_bh, H, B)

    @cute.jit
    def __call__(
        self,
        # [s_q, s_k, d, (h, b)]
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        dO: cute.Tensor,
        O: cute.Tensor,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        LSE: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        task_offsets: cute.Tensor,
        task_q_indices: cute.Tensor,
        variable_block_sizes: cute.Tensor,
        workspace: cute.Tensor,
        scale_softmax: Float32,
        stream: cuda.CUstream,
    ):
        q_seq_max, k_seq_max, d, hb = problem_shape
        h, b = hb
        # (b, h, s, d) -> (s, d, (h, b))
        Q = cute.make_tensor(
            Q.iterator,
            cute.group_modes(
                cute.select(Q.layout, mode=[2, 3, 1, 0]),
                2, 4
            )
        )
        # (b, h, s, d) -> (s, d, (h, b))
        K = cute.make_tensor(
            K.iterator,
            cute.group_modes(
                cute.select(K.layout, mode=[2, 3, 1, 0]),
                2, 4
            )
        )
        # (b, h, s, d) -> (s, d, (h, b))
        V = cute.make_tensor(
            V.iterator,
            cute.group_modes(
                cute.select(V.layout, mode=[2, 3, 1, 0]),
                2, 4
            )
        )
        O = cute.make_tensor(O.iterator, Q.layout)

        dQ = cute.make_tensor(dQ.iterator, Q.layout)
        dK = cute.make_tensor(
            dK.iterator,
            cute.group_modes(
                cute.select(dK.layout, mode=[3, 2, 1, 0]),
                2, 4
            )
        )
        # (b, h, s, d) -> (d, s, (h, b))
        dV = cute.make_tensor(
            dV.iterator,
            cute.group_modes(
                cute.select(dV.layout, mode=[3, 2, 1, 0]),
                2, 4
            )
        )
        dO = cute.make_tensor(dO.iterator, O.layout)

        # (b, h, s) -> (s, (h, b))
        LSE = cute.make_tensor(
            LSE.iterator,
            cute.group_modes(
                cute.select(LSE.layout, mode=[2, 1, 0]),
                1, 3
            )
        )

        # CSR schedule: [b, h, work, field] -> (work, field, (h, b)).
        # field layout: (kv_block, q_indices_start, q_count, q_group).
        task_offsets = cute.make_tensor(
            task_offsets.iterator,
            cute.group_modes(
                cute.select(task_offsets.layout, mode=[2, 3, 1, 0]),
                2, 4
            )
        )
        # task_q_indices is total-packed CSR: [total_edges].
        self.Q_major_mode = utils.LayoutEnum.from_tensor(Q).mma_major_mode()
        self.dQ_major_mode = utils.LayoutEnum.from_tensor(dQ).mma_major_mode()
        self.K_major_mode = utils.LayoutEnum.from_tensor(K).mma_major_mode()
        self.dK_major_mode = utils.LayoutEnum.from_tensor(dK).mma_major_mode()
        self.V_major_mode = utils.LayoutEnum.from_tensor(V).mma_major_mode()
        self.dV_major_mode = utils.LayoutEnum.from_tensor(dV).mma_major_mode()
        self.O_major_mode = utils.LayoutEnum.from_tensor(O).mma_major_mode()
        self.dO_major_mode = utils.LayoutEnum.from_tensor(dO).mma_major_mode()

        if cutlass.const_expr(self.Q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.dQ_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dq is not supported")
        if cutlass.const_expr(self.K_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.dK_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of dk is not supported")
        if cutlass.const_expr(self.V_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of v is not supported")
        if cutlass.const_expr(self.dV_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of dv is not supported")
        
        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE

        # Compute S
        QK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.QK_mma_tiler[:2]
        )
        fake_QK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.fake_QK_mma_tiler[:2]
        )
        # Compute dP
        dOV_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.dOV_mma_tiler[:2]
        )
        fake_dOV_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.fake_dOV_mma_tiler[:2]
        )
        # Compute dV
        dOP_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.dOP_mma_tiler[:2]
        )
        # Compute dK
        QdS_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.QdS_mma_tiler[:2]
        )
        # Compute dQ
        dSK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.dSK_mma_tiler[:2]
        )
        self.cluster_layout_vmnk = (
            cute.make_layout(((1), (1, 1, 1)), stride=((0), (0, 0, 0)))
        )

        Q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            QK_tiled_mma,
            self.QK_mma_tiler,
            self.element_dtype,
            self.load_mma_Q_stage
        )
        fake_Q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            fake_QK_tiled_mma,
            self.fake_QK_mma_tiler,
            self.element_dtype,
            1,
        )
        K_smem_layout_staged = sm100_utils.make_smem_layout_b(
            QK_tiled_mma,
            self.QK_mma_tiler,
            self.element_dtype,
            1
        )
        dO_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dOV_tiled_mma,
            self.dOV_mma_tiler,
            self.element_dtype,
            self.load_mma_dO_stage,
        )
        fake_dO_smem_layout_staged = sm100_utils.make_smem_layout_a(
            fake_dOV_tiled_mma,
            self.fake_dOV_mma_tiler,
            self.element_dtype,
            1,
        )
        V_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dOV_tiled_mma,
            self.dOV_mma_tiler,
            self.element_dtype,
            1,
        )
        dS_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dSK_tiled_mma,
            self.dSK_mma_tiler,
            self.element_dtype,
            self.compute_mma_dS_stage
        )
        KT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dSK_tiled_mma,
            self.dSK_mma_tiler,
            self.element_dtype,
            1,
        )
        QT_smem_layout_staged = sm100_utils.make_smem_layout_a(
            QdS_tiled_mma,
            self.QdS_mma_tiler,
            self.element_dtype,
            self.load_mma_Q_stage,
        )
        dST_smem_layout_staged = sm100_utils.make_smem_layout_b(
            QdS_tiled_mma,
            self.QdS_mma_tiler,
            self.element_dtype,
            self.compute_mma_dS_stage,
        )
        dOT_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dOP_tiled_mma,
            self.dOP_mma_tiler,
            self.element_dtype,
            self.load_mma_dO_stage,
        )
        P_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dOP_tiled_mma,
            self.dOP_mma_tiler,
            self.element_dtype,
            self.compute_mma_P_stage,
        )

        LSE_smem_layout = cute.make_layout(
            (self.QK_mma_tiler[0], self.load_compute_LSE_stage)
        )
        sum_OdO_smem_layout = cute.make_layout(
            (self.QK_mma_tiler[0], self.load_compute_sum_OdO_stage)
        )
        dQ_smem_layout_atom = sm100_utils.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(
                tcgen05.OperandMajorMode.K,
                self.acc_dtype,
                (self.dSK_mma_tiler[0], self.dQ_reduce_ncol),
            ),
            self.acc_dtype,
        )
        dQ_accum_smem_layout = cute.tile_to_shape(
            dQ_smem_layout_atom,
            (self.dSK_mma_tiler[0], self.dQ_reduce_ncol, self.reduce_tma_store_stage),
            order=(1, 0, 2),
        )
        fake_dQ_smem_layout_atom = sm100_utils.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(
                tcgen05.OperandMajorMode.K,
                self.acc_dtype,
                (self.sparse_block_size, self.dQ_reduce_ncol),
            ),
            self.acc_dtype,
        )
        fake_dQ_smem_layout_staged = cute.tile_to_shape(
            fake_dQ_smem_layout_atom,
            (self.sparse_block_size, self.dQ_reduce_ncol, self.reduce_tma_store_stage),
            order=(1, 0, 2),
        )
        dQ_tma_smem_layout = cute.select(fake_dQ_smem_layout_staged, mode=[0, 1])
        dKV_accum_smem_elements = (
            self.QdS_mma_tiler[1] * self.QdS_mma_tiler[0]
        )
        dKV_accum_smem_layout = cute.make_layout(
            (dKV_accum_smem_elements,),
            stride=(1,),
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_reduce_op = cpasync.CopyReduceBulkTensorTileS2GOp()

        Q_smem_layout = cute.select(fake_Q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            Q,
            Q_smem_layout,
            self.fake_QK_mma_tiler,
            fake_QK_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        K_smem_layout = cute.select(K_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            K,
            K_smem_layout,
            self.QK_mma_tiler,
            QK_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        V_smem_layout = cute.select(V_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            V,
            V_smem_layout,
            self.dOV_mma_tiler,
            dOV_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        dO_smem_layout = cute.select(fake_dO_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            dO,
            dO_smem_layout,
            self.fake_dOV_mma_tiler,
            fake_dOV_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        self.tma_copy_Q_bytes = cute.size_in_bytes(self.element_dtype, Q_smem_layout)
        self.tma_copy_K_bytes = cute.size_in_bytes(self.element_dtype, K_smem_layout)
        self.tma_copy_V_bytes = cute.size_in_bytes(self.element_dtype, V_smem_layout)
        self.tma_copy_dO_bytes = cute.size_in_bytes(self.element_dtype, dO_smem_layout)

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_mma_Q_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_mma_Q_stage * 2
            ]
            load_mma_dO_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_mma_dO_stage * 2
            ]
            load_compute_lse_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_compute_LSE_stage * 2
            ]
            load_compute_sum_OdO_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_compute_sum_OdO_stage * 2
            ]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_S_stage * 2
            ]
            mma_compute_dP_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_dP_stage * 2
            ]
            mma_reduce_dQ_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_reduce_dQ_stage * 2
            ]
            compute_mma_P_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.compute_mma_P_stage * 2
            ]
            compute_mma_dS_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.compute_mma_dS_stage * 2
            ]
            mma_compute_dKdV_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_dKdV_stage * 2
            ]
            tmem_holding_buf: cutlass.Int32
            # Smem tensors
            sK: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(K_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(V_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(Q_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(P_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(dO_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[
                    self.element_dtype, cute.cosize(dS_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sdQ: cute.struct.Align[
                cute.struct.MemRange[
                    self.acc_dtype, cute.cosize(dQ_accum_smem_layout)
                ],
                self.buffer_align_bytes,
            ]
            sdKV: cute.struct.Align[
                cute.struct.MemRange[
                    self.acc_dtype, cute.cosize(dKV_accum_smem_layout)
                ],
                self.buffer_align_bytes,
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(LSE_smem_layout)],
                self.buffer_align_bytes,
            ]
            sSum_OdO: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(sum_OdO_smem_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        sum_OdO, scaled_LSE, dQ_acc, dQ_acc_tma, dK_acc, dV_acc = self.get_workspace_tensor(
            problem_shape, workspace, self.acc_dtype
        )
        tma_atom_dQ_acc, tma_tensor_dQ_acc = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_reduce_op,
            dQ_acc_tma,
            dQ_tma_smem_layout,
            (self.sparse_block_size, self.dQ_reduce_ncol),
            self.cluster_layout_vmnk.shape,
        )

        # =============================== Sum OdO ===============================
        sum_OdO_scale = Float32(-1.0)
        LSE_scale = Float32(-math.log2(math.e))

        sum_OdO_grid = self._compute_sum_OdO_grid(problem_shape, self.sum_OdO_block_q)

        self.sum_OdO(
            O,
            dO,
            sum_OdO,
            LSE,
            scaled_LSE,
            sum_OdO_scale,
            LSE_scale,
            problem_shape,
        ).launch(
            grid=sum_OdO_grid,
            block=[self.sum_OdO_num_threads_d, self.sum_OdO_num_threads_q, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

        bwd_grid = self._compute_bwd_grid(problem_shape, task_offsets)
        self.bwd(
            QK_tiled_mma,
            fake_QK_tiled_mma,
            dOV_tiled_mma,
            fake_dOV_tiled_mma,
            dOP_tiled_mma,
            QdS_tiled_mma,
            dSK_tiled_mma,
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_V,
            tma_tensor_V,
            tma_atom_dO,
            tma_tensor_dO,
            tma_atom_dQ_acc,
            tma_tensor_dQ_acc,
            dK_acc,
            dV_acc,
            scaled_LSE,
            scale_softmax,
            sum_OdO,
            task_offsets,
            task_q_indices,
            problem_shape,
            variable_block_sizes,
            Q_smem_layout_staged,
            K_smem_layout_staged,
            V_smem_layout_staged,
            dO_smem_layout_staged,
            dS_smem_layout_staged,
            KT_smem_layout_staged,
            QT_smem_layout_staged,
            dST_smem_layout_staged,
            dOT_smem_layout_staged,
            dQ_accum_smem_layout,
            P_smem_layout_staged,
            dKV_accum_smem_layout,
            LSE_smem_layout,
            sum_OdO_smem_layout,
        ).launch(
            grid=bwd_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        self.postprocess_block_seq = 8
        self.num_threads_D_postprocess = 16
        self.num_threads_seq_postprocess = 128 // self.num_threads_D_postprocess
        self.postprocess_elem_per_load = 4

        postprocess_dq_grid = [
            cute.ceil_div(problem_shape[0], self.sparse_block_size),
            cute.size(problem_shape[3][0]),
            cute.size(problem_shape[3][1]),
        ]
        postprocess_dkv_grid = [
            (problem_shape[1] + self.postprocess_block_seq - 1)
            // self.postprocess_block_seq,
            cute.size(problem_shape[3][0]),
            cute.size(problem_shape[3][1]),
        ]
        postprocess_block = [
            self.num_threads_D_postprocess,
            self.num_threads_seq_postprocess,
            1,
        ]

        self.postprocess_dq(
            dQ_acc,
            dQ,
            problem_shape[0],
            problem_shape[2],
            scale_softmax,
        ).launch(
            grid=postprocess_dq_grid,
            block=[128, 1, 1],
            cluster=[1, 1, 1],
            smem=0,
            stream=stream,
        )

        if cutlass.const_expr(self.use_dkv_stage_layout):
            self.postprocess_dkv(
                dK_acc,
                dV_acc,
                dK,
                dV,
                problem_shape[1],
                problem_shape[2],
                scale_softmax,
            ).launch(
                grid=postprocess_dkv_grid,
                block=postprocess_block,
                cluster=[1, 1, 1],
                smem=0,
                stream=stream,
            )
        else:
            self.postprocess_dkv_linear(
                dK_acc,
                dV_acc,
                dK,
                dV,
                problem_shape[1],
                problem_shape[2],
                scale_softmax,
            ).launch(
                grid=postprocess_dkv_grid,
                block=postprocess_block,
                cluster=[1, 1, 1],
                smem=0,
                stream=stream,
            )


    @cute.kernel
    def sum_OdO(
        self,
        O: cute.Tensor,
        dO: cute.Tensor,
        sum_OdO: cute.Tensor,
        lse: cute.Tensor,
        scaled_lse: cute.Tensor,
        sum_OdO_scale: Float32,
        lse_scale: Float32,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, tidy, tidz = cute.arch.thread_idx()

        seqlen_q = problem_shape[0]

        for idx_q_t in cutlass.range(
            tidy, self.sum_OdO_block_q, self.sum_OdO_num_threads_q, unroll_full=True
        ):
            idx_q = idx_q_t + self.sum_OdO_block_q * bidx
            if idx_q < seqlen_q:
                O_bhq = O[idx_q, None, (bidy, bidz)]
                O_bhq = cute.logical_divide(
                    O_bhq, cute.make_layout(self.sum_OdO_elem_per_load)
                )
                dO_bhq = dO[idx_q, None, (bidy, bidz)]
                dO_bhq = cute.logical_divide(
                    dO_bhq, cute.make_layout(self.sum_OdO_elem_per_load)
                )

                idx_d_start = tidx
                idx_d_step = self.sum_OdO_num_threads_d
                acc = 0.0
                for idx_d in cutlass.range(
                    idx_d_start, O.shape[1] // self.sum_OdO_elem_per_load, idx_d_step
                ):
                    O_frag = O_bhq[None, idx_d].load()
                    dO_frag = dO_bhq[None, idx_d].load()
                    prod_frag = O_frag * dO_frag
                    prod_frag = prod_frag.to(self.acc_dtype)
                    acc += prod_frag.reduce(
                        cute.ReductionOp.ADD, 0.0, reduction_profile=0
                    )

                acc = cute.arch.warp_reduction_sum(
                    acc, threads_in_group=self.sum_OdO_num_threads_d
                )

                if tidx == 0:
                    lse_bhq = lse[idx_q, (bidy, bidz)]
                    sum_OdO[idx_q, (bidy, bidz)] = sum_OdO_scale * acc
                    scaled_lse[idx_q, (bidy, bidz)] = lse_scale * lse_bhq

    @cute.kernel
    def bwd(
        self,
        QK_tiled_mma: cute.TiledMma,
        fake_QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        fake_dOV_tiled_mma: cute.TiledMma,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        Q_in: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        K_in: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        V_in: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        dO_in: cute.Tensor,
        tma_atom_dQ_acc: cute.CopyAtom,
        dQ_acc_tma: cute.Tensor,
        dK_acc: cute.Tensor,
        dV_acc: cute.Tensor,
        LSE: cute.Tensor,
        scale_softmax: Float32,
        sum_OdO: cute.Tensor,
        task_offsets: cute.Tensor,
        task_q_indices: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        variable_block_sizes: cute.Tensor,
        Q_smem_layout_staged: cute.ComposedLayout,
        K_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        dS_smem_layout_staged: cute.ComposedLayout,
        KT_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        dST_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        dQ_accum_smem_layout: cute.ComposedLayout,
        P_smem_layout_staged: cute.ComposedLayout,
        dKV_accum_smem_layout: cute.Layout,
        LSE_smem_layout: cute.Layout,
        sum_OdO_smem_layout: cute.Layout,
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        seqlen_q, seqlen_k, head_dim, HB = problem_shape
        num_heads, batch_size = HB

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_dO)
        
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_mma_Q_pipeline = self.make_and_init_load_mma_Q_pipeline(
            storage.load_mma_Q_mbar_ptr.data_ptr()
        )
        load_mma_dO_pipeline = self.make_and_init_load_mma_dO_pipeline(
            storage.load_mma_dO_mbar_ptr.data_ptr()
        )
        load_compute_LSE_pipeline = self.make_and_init_load_compute_LSE_pipeline(
            storage.load_compute_lse_mbar_ptr.data_ptr()
        )
        load_compute_sum_OdO_pipeline = (
            self.make_and_init_load_compute_sum_OdO_pipeline(
                storage.load_compute_sum_OdO_mbar_ptr.data_ptr()
            )
        )
        mma_compute_S_pipeline = self.make_and_init_mma_compute_S_pipeline(
            storage.mma_compute_S_mbar_ptr.data_ptr()
        )
        mma_compute_dP_pipeline = self.make_and_init_mma_compute_dP_pipeline(
            storage.mma_compute_dP_mbar_ptr.data_ptr()
        )
        mma_reduce_dQ_pipeline = self.make_and_init_mma_reduce_dQ_pipeline(
            storage.mma_reduce_dQ_mbar_ptr.data_ptr()
        )
        compute_mma_P_pipeline = self.make_and_init_compute_mma_P_pipeline(
            storage.compute_mma_P_mbar_ptr.data_ptr()
        )
        compute_mma_dS_pipeline = self.make_and_init_compute_mma_dS_pipeline(
            storage.compute_mma_dS_mbar_ptr.data_ptr()
        )
        mma_compute_dKdV_pipeline = self.make_and_init_mma_compute_dKdV_pipeline(
            storage.mma_compute_dKdV_mbar_ptr.data_ptr()
        )
        reduce_tma_store_pipeline = self.make_and_init_reduce_tma_store_pipeline()

        self.cta_sync_barrier.arrive_and_wait()

        sQ = storage.sQ.get_tensor(
            Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner
        )
        sV = storage.sV.get_tensor(
            V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner
        )
        sP = storage.sP.get_tensor(
            P_smem_layout_staged.outer, swizzle=P_smem_layout_staged.inner
        )
        sdO = storage.sdO.get_tensor(
            dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner
        )
        sdS = storage.sdS.get_tensor(
            dS_smem_layout_staged.outer, swizzle=dS_smem_layout_staged.inner
        )
        sdQ = storage.sdQ.get_tensor(
            dQ_accum_smem_layout.outer,
            swizzle=dQ_accum_smem_layout.inner,
        )
        sdKV = storage.sdKV.get_tensor(dKV_accum_smem_layout)
        sLSE = storage.sLSE.get_tensor(LSE_smem_layout)
        sSum_OdO = storage.sSum_OdO.get_tensor(sum_OdO_smem_layout)

        tmem_holding_buf = storage.tmem_holding_buf
        tmem = utils.TmemAllocator(
            tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
        )

        sQT_ptr = cute.recast_ptr(sQ.iterator, QT_smem_layout_staged.inner)
        sQT = cute.make_tensor(sQT_ptr, QT_smem_layout_staged.outer)
        sKT_ptr = cute.recast_ptr(sK.iterator, KT_smem_layout_staged.inner)
        sKT = cute.make_tensor(sKT_ptr, KT_smem_layout_staged.outer)
        sdST_ptr = cute.recast_ptr(sdS.iterator, dST_smem_layout_staged.inner)
        sdST = cute.make_tensor(sdST_ptr, dST_smem_layout_staged.outer)
        sdOT_ptr = cute.recast_ptr(sdO.iterator, dOT_smem_layout_staged.inner)
        sdOT = cute.make_tensor(sdOT_ptr, dOT_smem_layout_staged.outer)

        # (MMA, MMA_M, MMA_K, STAGE)
        tSrQ = QK_tiled_mma.make_fragment_A(sQ)
        # (MMA, MMA_N, MMA_K, STAGE)
        tSrK = QK_tiled_mma.make_fragment_B(sK)

        tdPrdO = dOV_tiled_mma.make_fragment_A(sdO)
        tdPrV = dOV_tiled_mma.make_fragment_B(sV)

        tdKTrQT = QdS_tiled_mma.make_fragment_A(sQT)
        tdKTrdST = QdS_tiled_mma.make_fragment_B(sdST)

        tdVTrdOT = dOP_tiled_mma.make_fragment_A(sdOT)
        tdVTrP = dOP_tiled_mma.make_fragment_B(sP)

        tdQrdS = dSK_tiled_mma.make_fragment_A(sdS)
        tdQrKT = dSK_tiled_mma.make_fragment_B(sKT)

        kv_block_idx = task_offsets[bidx, Int32(0), (bidy, bidz)]
        task_start = task_offsets[bidx, Int32(1), (bidy, bidz)]
        iter_count = task_offsets[bidx, Int32(2), (bidy, bidz)]
        iter_index = Int32(0)
        load_iter_count = iter_count
        mma_iter_count = cute.ceil_div(iter_count, 2)
        compute_iter_count = mma_iter_count
        reduce_iter_count = iter_count
        
        task_has_work = iter_count > 0
        if cutlass.const_expr(not self.full_kv_blocks):
            task_has_work = task_has_work and kv_block_idx * self.QK_mma_tiler[1] < seqlen_k

        if task_has_work:
            if warp_idx == self.load_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
                self.load(
                    Q_in,
                    K_in,
                    V_in,
                    dO_in,
                    LSE,
                    sum_OdO,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sSum_OdO,
                    task_q_indices,
                    task_start,
                    kv_block_idx,
                    fake_QK_tiled_mma,
                    fake_dOV_tiled_mma,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    problem_shape,
                    load_iter_count,
                    iter_index,
                    (load_mma_Q_pipeline, load_compute_LSE_pipeline, load_mma_dO_pipeline, load_compute_sum_OdO_pipeline)
                )
            elif warp_idx == self.mma_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_mma)
                
                tmem.allocate(self.tmem_alloc_cols)
                # Barrier before retrieve tensor memory ptr from shared memory
                tmem.wait_for_alloc()
                # Retrieve tmem ptr
                tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

                tStS_shape = QK_tiled_mma.partition_shape_C(
                    cute.select(self.QK_mma_tiler, mode=[0, 1])
                )
                tStS = QK_tiled_mma.make_fragment_C(tStS_shape)
                tStS = cute.make_tensor(tmem_ptr_base + self.tmem_S_offset, tStS.layout)

                tdPtdP_shape = dOV_tiled_mma.partition_shape_C(
                    cute.select(self.dOV_mma_tiler, mode=[0, 1])
                )
                tdPtdP = dOV_tiled_mma.make_fragment_C(tdPtdP_shape)
                tdPtdP = cute.make_tensor(tmem_ptr_base + self.tmem_dP_offset, tdPtdP.layout)

                tdQtdQ_shape = dSK_tiled_mma.partition_shape_C(
                    cute.select(self.dSK_mma_tiler, mode=[0, 1])
                )
                tdQtdQ = dSK_tiled_mma.make_fragment_C(tdQtdQ_shape)
                tdQtdQ = cute.make_tensor(tmem_ptr_base + self.tmem_dQ_offset, tdQtdQ.layout)

                tdKTtdKT_shape = QdS_tiled_mma.partition_shape_C(
                    cute.select(self.QdS_mma_tiler, mode=[0, 1])
                )
                tdKTtdKT = QdS_tiled_mma.make_fragment_C(tdKTtdKT_shape)
                tdKTtdKT = cute.make_tensor(tmem_ptr_base + self.tmem_dK_offset, tdKTtdKT.layout)

                tdVTtdVT_shape = dOP_tiled_mma.partition_shape_C(
                    cute.select(self.dOP_mma_tiler, mode=[0, 1])
                )
                tdVTtdVT = dOP_tiled_mma.make_fragment_C(tdVTtdVT_shape)
                tdVTtdVT = cute.make_tensor(tmem_ptr_base + self.tmem_dV_offset, tdVTtdVT.layout)

                self.mma(
                    QK_tiled_mma,
                    dOV_tiled_mma,
                    dOP_tiled_mma,
                    QdS_tiled_mma,
                    dSK_tiled_mma,
                    tStS,
                    tSrQ,
                    tSrK,
                    tdPtdP,
                    tdPrdO,
                    tdPrV,
                    tdVTtdVT,
                    tdVTrdOT,
                    tdVTrP,
                    tdQtdQ,
                    tdQrdS,
                    tdQrKT,
                    tdKTtdKT,
                    tdKTrQT,
                    tdKTrdST,
                    mma_iter_count,
                    (load_mma_Q_pipeline, mma_compute_S_pipeline, load_mma_dO_pipeline, mma_compute_dP_pipeline, mma_reduce_dQ_pipeline, compute_mma_P_pipeline, compute_mma_dS_pipeline, mma_compute_dKdV_pipeline)
                )
            elif warp_idx in self.compute_warp_id:
                cute.arch.warpgroup_reg_alloc(self.num_regs_compute)
                tmem.wait_for_alloc()
                # Retrieve tmem ptr
                tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

                tStS_shape = QK_tiled_mma.partition_shape_C(
                    cute.select(self.QK_mma_tiler, mode=[0, 1])
                )
                tStS = QK_tiled_mma.make_fragment_C(tStS_shape)
                tStS = cute.make_tensor(tmem_ptr_base + self.tmem_S_offset, tStS.layout)

                tdPtdP_shape = dOV_tiled_mma.partition_shape_C(
                    cute.select(self.dOV_mma_tiler, mode=[0, 1])
                )
                tdPtdP = dOV_tiled_mma.make_fragment_C(tdPtdP_shape)
                tdPtdP = cute.make_tensor(tmem_ptr_base + self.tmem_dP_offset, tdPtdP.layout)

                tdKTtdKT_shape = QdS_tiled_mma.partition_shape_C(
                    cute.select(self.QdS_mma_tiler, mode=[0, 1])
                )
                tdKTtdKT = QdS_tiled_mma.make_fragment_C(tdKTtdKT_shape)
                tdKTtdKT = cute.make_tensor(tmem_ptr_base + self.tmem_dK_offset, tdKTtdKT.layout)

                tdVTtdVT_shape = dOP_tiled_mma.partition_shape_C(
                    cute.select(self.dOP_mma_tiler, mode=[0, 1])
                )
                tdVTtdVT = dOP_tiled_mma.make_fragment_C(tdVTtdVT_shape)
                tdVTtdVT = cute.make_tensor(tmem_ptr_base + self.tmem_dV_offset, tdVTtdVT.layout)
                self.compute(
                    tStS,
                    tdPtdP,
                    tdVTrP,
                    sLSE,
                    sdS,
                    sP,
                    sSum_OdO,
                    dK_acc,
                    dV_acc,
                    sdKV,
                    tdKTtdKT,
                    tdVTtdVT,
                    kv_block_idx,
                    variable_block_sizes,
                    dOP_tiled_mma,
                    QdS_tiled_mma,
                    problem_shape,
                    compute_iter_count,
                    scale_softmax,
                    (mma_compute_S_pipeline, compute_mma_P_pipeline, load_compute_LSE_pipeline, load_compute_sum_OdO_pipeline, mma_compute_dP_pipeline, compute_mma_dS_pipeline, mma_compute_dKdV_pipeline)
                )

                self.epilogue_sync_barrier.arrive_and_wait()
                if warp_idx % self.num_compute_warps == 0:
                    tmem_ptr = cute.arch.retrieve_tmem_ptr(
                        Float32,
                        alignment=16,
                        ptr_to_buffer_holding_addr=tmem_holding_buf,
                    )
                    cute.arch.dealloc_tmem(tmem_ptr, self.tmem_alloc_cols)
            elif warp_idx in self.reduce_warp_id:
                cute.arch.warpgroup_reg_alloc(self.num_regs_reduce)

                tmem.wait_for_alloc()
                # Retrieve tmem ptr
                tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

                tdQtdQ_shape = dSK_tiled_mma.partition_shape_C(
                    cute.select(self.dSK_mma_tiler, mode=[0, 1])
                )
                tdQtdQ = dSK_tiled_mma.make_fragment_C(tdQtdQ_shape)
                tdQtdQ = cute.make_tensor(tmem_ptr_base + self.tmem_dQ_offset, tdQtdQ.layout)

                self.reduce(
                    problem_shape,
                    tdQtdQ,
                    dSK_tiled_mma,
                    task_q_indices,
                    task_start,
                    tma_atom_dQ_acc,
                    dQ_acc_tma,
                    sdQ,
                    reduce_iter_count,
                    (mma_reduce_dQ_pipeline, reduce_tma_store_pipeline),
                )
            else:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)
    
    @cute.kernel
    def postprocess_dq(
        self,
        dQ_acc: cute.Tensor,
        dQ: cute.Tensor,
        q_count: Int32,
        d_dim: Int32,
        scale_softmax: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_block_idx, h_idx, b_idx = cute.arch.block_idx()

        seq_thread = tidx // self.num_threads_D_postprocess
        dim_thread = tidx - seq_thread * self.num_threads_D_postprocess

        for q_local in cutlass.range(
            seq_thread,
            self.sparse_block_size,
            self.num_threads_seq_postprocess,
            unroll_full=True,
        ):
            q_idx = q_block_idx * self.sparse_block_size + q_local
            if q_idx < q_count:
                dQ_acc_row = dQ_acc[q_idx, None, (h_idx, b_idx)]
                dQ_acc_row = cute.logical_divide(
                    dQ_acc_row, cute.make_layout(self.postprocess_elem_per_load)
                )
                dQ_row = dQ[q_idx, None, (h_idx, b_idx)]
                dQ_row = cute.logical_divide(
                    dQ_row, cute.make_layout(self.postprocess_elem_per_load)
                )
                for d_vec in cutlass.range(
                    dim_thread,
                    d_dim // self.postprocess_elem_per_load,
                    self.num_threads_D_postprocess,
                ):
                    dQ_acc_frg = dQ_acc_row[None, d_vec].load()
                    dQ_row[None, d_vec].store(
                        (scale_softmax * dQ_acc_frg).to(self.element_dtype)
                    )

    @cute.kernel
    def postprocess_dkv(
        self,
        dK_acc: cute.Tensor,
        dV_acc: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        k_count: Int32,
        d_dim: Int32,
        scale_softmax: Float32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        seq_tile_idx, h_idx, b_idx = cute.arch.block_idx()

        for idx_s_t in cutlass.range(
            tidy,
            self.postprocess_block_seq,
            self.num_threads_seq_postprocess,
            unroll_full=True,
        ):
            idx_s = idx_s_t + self.postprocess_block_seq * seq_tile_idx
            if idx_s < k_count:
                kv_block = idx_s // self.sparse_block_size
                kv_local = idx_s - kv_block * self.sparse_block_size
                dK_bhs = dK[None, idx_s, (h_idx, b_idx)]
                dK_bhs = cute.logical_divide(
                    dK_bhs, cute.make_layout(self.postprocess_elem_per_load)
                )
                dV_bhs = dV[None, idx_s, (h_idx, b_idx)]
                dV_bhs = cute.logical_divide(
                    dV_bhs, cute.make_layout(self.postprocess_elem_per_load)
                )

                thr_start = tidx
                thr_step = self.num_threads_D_postprocess
                for idx_d in cutlass.range(
                    thr_start,
                    d_dim // self.postprocess_elem_per_load,
                    thr_step,
                ):
                    d_start = idx_d * self.postprocess_elem_per_load
                    d_stage = d_start // self.dKV_reduce_ncol
                    d_stage_vec = (
                        d_start - d_stage * self.dKV_reduce_ncol
                    ) // self.postprocess_elem_per_load
                    acc_vec_idx = (
                        kv_local * self.dKV_reduce_ncol // self.postprocess_elem_per_load
                        + d_stage_vec
                    )
                    dK_acc_stage = dK_acc[None, d_stage, kv_block, (h_idx, b_idx)]
                    dK_acc_stage = cute.logical_divide(
                        dK_acc_stage, cute.make_layout(self.postprocess_elem_per_load)
                    )
                    dV_acc_stage = dV_acc[None, d_stage, kv_block, (h_idx, b_idx)]
                    dV_acc_stage = cute.logical_divide(
                        dV_acc_stage, cute.make_layout(self.postprocess_elem_per_load)
                    )
                    dK_acc_frg = dK_acc_stage[None, acc_vec_idx].load()
                    dV_acc_frg = dV_acc_stage[None, acc_vec_idx].load()
                    dK_bhs[None, idx_d].store(
                        (scale_softmax * dK_acc_frg).to(self.element_dtype)
                    )
                    dV_bhs[None, idx_d].store(dV_acc_frg.to(self.element_dtype))

    @cute.kernel
    def postprocess_dkv_linear(
        self,
        dK_acc: cute.Tensor,
        dV_acc: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        k_count: Int32,
        d_dim: Int32,
        scale_softmax: Float32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        seq_tile_idx, h_idx, b_idx = cute.arch.block_idx()

        for idx_s_t in cutlass.range(
            tidy,
            self.postprocess_block_seq,
            self.num_threads_seq_postprocess,
            unroll_full=True,
        ):
            idx_s = idx_s_t + self.postprocess_block_seq * seq_tile_idx
            if idx_s < k_count:
                dK_acc_bhs = dK_acc[None, idx_s, (h_idx, b_idx)]
                dK_acc_bhs = cute.logical_divide(
                    dK_acc_bhs, cute.make_layout(self.postprocess_elem_per_load)
                )
                dV_acc_bhs = dV_acc[None, idx_s, (h_idx, b_idx)]
                dV_acc_bhs = cute.logical_divide(
                    dV_acc_bhs, cute.make_layout(self.postprocess_elem_per_load)
                )
                dK_bhs = dK[None, idx_s, (h_idx, b_idx)]
                dK_bhs = cute.logical_divide(
                    dK_bhs, cute.make_layout(self.postprocess_elem_per_load)
                )
                dV_bhs = dV[None, idx_s, (h_idx, b_idx)]
                dV_bhs = cute.logical_divide(
                    dV_bhs, cute.make_layout(self.postprocess_elem_per_load)
                )

                for idx_d in cutlass.range(
                    tidx,
                    d_dim // self.postprocess_elem_per_load,
                    self.num_threads_D_postprocess,
                ):
                    dK_acc_frg = dK_acc_bhs[None, idx_d].load()
                    dV_acc_frg = dV_acc_bhs[None, idx_d].load()
                    dK_bhs[None, idx_d].store(
                        (scale_softmax * dK_acc_frg).to(self.element_dtype)
                    )
                    dV_bhs[None, idx_d].store(dV_acc_frg.to(self.element_dtype))

    @cute.jit
    def _load_task_q_block(
        self,
        task_q_indices: cute.Tensor,
        task_start: Int32,
        iter_index: Int32,
        hb_coord: tuple,
    ) -> Int32:
        return task_q_indices[task_start + iter_index]

    @cute.jit
    def load(
        self,
        Q_in: cute.Tensor,
        K_in: cute.Tensor,
        V_in: cute.Tensor,
        dO_in: cute.Tensor,
        LSE: cute.Tensor,
        sum_OdO: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        task_q_indices: cute.Tensor,
        task_start: Int32,
        kv_block_idx: Int32,
        fake_QK_tiled_mma: cute.TiledMma,
        fake_dOV_tiled_mma: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        iter_count: Int32,
        iter_index: Int32,
        pipeline_args: tuple,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, blk_coord_h, blk_coord_b = cute.arch.block_idx()
        seqlen_q, seqlen_k, head_dim, HB = problem_shape
        num_heads, batch_size = HB
        (
            load_mma_Q_pipeline,
            load_compute_LSE_pipeline,
            load_mma_dO_pipeline,
            load_compute_sum_OdO_pipeline,
        ) = pipeline_args

        total_iter_count = iter_count

        # (bM, bK, RestM, RestK, (H, B))
        gQ = cute.local_tile(
            Q_in, cute.select(self.fake_QK_mma_tiler, mode=[0, 2]), (None, None, None)
        )
        # (bN, bK, RestN, RestK, (H, B))
        gK = cute.local_tile(
            K_in, cute.select(self.QK_mma_tiler, mode=[1, 2]), (None, None, None)
        )
        # (bM, bK, RestM, RestK, (H, B))
        gdO = cute.local_tile(
            dO_in, cute.select(self.fake_dOV_mma_tiler, mode=[0, 2]), (None, None, None)
        )
        # (bN, bK, RestN, RestK, (H, B))
        gV = cute.local_tile(
            V_in, cute.select(self.dOV_mma_tiler, mode=[1, 2]), (None, None, None)
        )

        QK_thr_mma = fake_QK_tiled_mma.get_slice(0)
        dOV_thr_mma = fake_dOV_tiled_mma.get_slice(0)

        tSgQ = QK_thr_mma.partition_A(gQ)
        tSgK = QK_thr_mma.partition_B(gK)
        tdPgdO = dOV_thr_mma.partition_A(gdO)
        tdPgV = dOV_thr_mma.partition_B(gV)

        load_mma_Q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.load_mma_Q_stage
        )
        load_compute_LSE_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.load_compute_LSE_stage
        )
        load_mma_dO_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.load_mma_dO_stage
        )
        load_compute_sum_OdO_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.load_compute_sum_OdO_stage
        )

        sQ = cute.make_tensor(
            sQ.iterator,
            cute.make_layout(((64,16),2,(4,2),2), stride=((64,1),4096,(16,8192),16384))
        )
        sQ_0 = sQ[None, 0, None, load_mma_Q_producer_state.index]
        sQ_1 = sQ[None, 1, None, load_mma_Q_producer_state.index]
        
        sdO = cute.make_tensor(
            sdO.iterator,
            cute.make_layout(((64,16),2,(4,2),2), stride=((64,1),4096,(16,8192),16384))
        )
        sdO_0 = sdO[None, 0, None, load_mma_dO_producer_state.index]
        sdO_1 = sdO[None, 1, None, load_mma_dO_producer_state.index]
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))
        tQsQ_0, tQgQ_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ_0, 0, 2),
            cute.group_modes(tSgQ, 0, 3)
        )
        tQsQ_1, _ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ_1, 0, 2),
            cute.group_modes(tSgQ, 0, 3)
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))
        tKsK, tKgK_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSgK, 0, 3)
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, (H, B))
        tdOsdO_0, tdOgdO_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dO,
            0,
            cute.make_layout(1),
            cute.group_modes(sdO_0, 0, 2),
            cute.group_modes(tdPgdO, 0, 3)
        )
        tdOsdO_1, _ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dO,
            0,
            cute.make_layout(1),
            cute.group_modes(sdO_1, 0, 2),
            cute.group_modes(tdPgdO, 0, 3)
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, (H, B))
        tVsV, tVgV_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tdPgV, 0, 3)
        )

        q_block_idx_0 = self._load_task_q_block(
            task_q_indices, task_start, iter_index, (blk_coord_h, blk_coord_b)
        )
        iter_index += 1
        q_block_idx_1 = seqlen_q // self.sparse_block_size # out of box, tma can fill zeros automatically
        if iter_index < total_iter_count:
            q_block_idx_1 = self._load_task_q_block(
                task_q_indices, task_start, iter_index, (blk_coord_h, blk_coord_b)
            )
        q_block_0_full = (q_block_idx_0 + 1) * self.sparse_block_size <= seqlen_q
        
        load_mma_Q_pipeline.producer_acquire(load_mma_Q_producer_state)
        tma_barrier = load_mma_Q_pipeline.producer_get_barrier(
            load_mma_Q_producer_state
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_Q_bytes * 2)
        
        # Load K
        cute.copy(
            tma_atom_K,
            tKgK_mkl[(None, kv_block_idx, 0, (blk_coord_h, blk_coord_b))],
            tKsK[None, 0],
            tma_bar_ptr=tma_barrier,
        )

        # Load Q0
        cute.copy(
            tma_atom_Q,
            tQgQ_mkl[(None, q_block_idx_0, 0, (blk_coord_h, blk_coord_b))],
            tQsQ_0,
            tma_bar_ptr=tma_barrier,
        )
        
        # Load Q1
        cute.copy(
            tma_atom_Q,
            tQgQ_mkl[(None, q_block_idx_1, 0, (blk_coord_h, blk_coord_b))],
            tQsQ_1,
            tma_bar_ptr=tma_barrier,
        )

        load_mma_Q_producer_state.advance()

        load_compute_LSE_pipeline.producer_acquire(load_compute_LSE_producer_state)

        thread_idx = tidx % self.threads_per_warp
        async_copy_num_elts = self.sparse_block_size // self.threads_per_warp
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            self.acc_dtype,
            num_bits_per_copy=self.acc_dtype.width,
        )

        # Load LSE
        # 32 threads load 64 values, each thread loads 2 values
        sLSE_for_copy = cute.flat_divide(sLSE, (1,))
        LSE_for_copy = cute.flat_divide(LSE, (1,))
        for i in cutlass.range_constexpr(async_copy_num_elts):
            copy_offset = i * self.threads_per_warp + thread_idx
            LSE_idx = q_block_idx_0 * self.sparse_block_size + copy_offset
            if q_block_0_full:
                cute.copy(
                    atom_async_copy,
                    LSE_for_copy[None, LSE_idx, (blk_coord_h, blk_coord_b)],
                    sLSE_for_copy[
                        None,
                        copy_offset,
                        load_compute_LSE_producer_state.index,
                    ],
                )
            elif cute.elem_less(LSE_idx, seqlen_q):
                cute.copy(
                    atom_async_copy,
                    LSE_for_copy[None, LSE_idx, (blk_coord_h, blk_coord_b)],
                    sLSE_for_copy[
                        None,
                        copy_offset,
                        load_compute_LSE_producer_state.index,
                    ],
                )
            else:
                sLSE_for_copy[
                    None,
                    copy_offset,
                    load_compute_LSE_producer_state.index,
                ].fill(0.0)

        for i in cutlass.range_constexpr(async_copy_num_elts):
            copy_offset = i * self.threads_per_warp + thread_idx
            LSE_idx = q_block_idx_1 * self.sparse_block_size + copy_offset
            if cute.elem_less(LSE_idx, seqlen_q):
                cute.copy(
                    atom_async_copy,
                    LSE_for_copy[None, LSE_idx, (blk_coord_h, blk_coord_b)],
                    sLSE_for_copy[
                        None,
                        self.sparse_block_size + copy_offset,
                        load_compute_LSE_producer_state.index,
                    ],
                )
            else:
                sLSE_for_copy[
                    None,
                    self.sparse_block_size + copy_offset,
                    load_compute_LSE_producer_state.index,
                ].fill(0.0)
        
        load_compute_LSE_pipeline.producer_commit(load_compute_LSE_producer_state)
        load_compute_LSE_producer_state.advance()

        load_mma_dO_pipeline.producer_acquire(load_mma_dO_producer_state)
        tma_barrier = load_mma_dO_pipeline.producer_get_barrier(
            load_mma_dO_producer_state
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_dO_bytes * 2)
        
        # Load dO0
        cute.copy(
            tma_atom_dO,
            tdOgdO_mkl[(None, q_block_idx_0, 0, (blk_coord_h, blk_coord_b))],
            tdOsdO_0,
            tma_bar_ptr=tma_barrier,
        )
        # Load dO1
        cute.copy(
            tma_atom_dO,
            tdOgdO_mkl[(None, q_block_idx_1, 0, (blk_coord_h, blk_coord_b))],
            tdOsdO_1,
            tma_bar_ptr=tma_barrier,
        )

        # Load V
        cute.copy(
            tma_atom_V,
            tVgV_mkl[(None, kv_block_idx, 0, (blk_coord_h, blk_coord_b))],
            tVsV[None, 0],
            tma_bar_ptr=tma_barrier,
        )

        load_mma_dO_producer_state.advance()

        load_compute_sum_OdO_pipeline.producer_acquire(
            load_compute_sum_OdO_producer_state
        )

        sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))
        sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1,))
        for i in cutlass.range_constexpr(async_copy_num_elts):
            copy_offset = i * self.threads_per_warp + thread_idx
            sum_OdO_idx = q_block_idx_0 * self.sparse_block_size + copy_offset
            if q_block_0_full:
                cute.copy(
                    atom_async_copy,
                    sum_OdO_for_copy[None, sum_OdO_idx, (blk_coord_h, blk_coord_b)],
                    sSum_OdO_for_copy[
                        None,
                        copy_offset,
                        load_compute_sum_OdO_producer_state.index,
                    ],
                )
            elif cute.elem_less(sum_OdO_idx, seqlen_q):
                cute.copy(
                    atom_async_copy,
                    sum_OdO_for_copy[None, sum_OdO_idx, (blk_coord_h, blk_coord_b)],
                    sSum_OdO_for_copy[
                        None,
                        copy_offset,
                        load_compute_sum_OdO_producer_state.index,
                    ],
                )
            else:
                sSum_OdO_for_copy[
                    None,
                    copy_offset,
                    load_compute_sum_OdO_producer_state.index,
                ].fill(0.0)
        for i in cutlass.range_constexpr(async_copy_num_elts):
            copy_offset = i * self.threads_per_warp + thread_idx
            sum_OdO_idx = q_block_idx_1 * self.sparse_block_size + copy_offset
            if cute.elem_less(sum_OdO_idx, seqlen_q):
                cute.copy(
                    atom_async_copy,
                    sum_OdO_for_copy[None, sum_OdO_idx, (blk_coord_h, blk_coord_b)],
                    sSum_OdO_for_copy[
                        None,
                        self.sparse_block_size + copy_offset,
                        load_compute_sum_OdO_producer_state.index,
                    ],
                )
            else:
                sSum_OdO_for_copy[
                    None,
                    self.sparse_block_size + copy_offset,
                    load_compute_sum_OdO_producer_state.index,
                ].fill(0.0)
        
        load_compute_sum_OdO_pipeline.producer_commit(
            load_compute_sum_OdO_producer_state
        )
        load_compute_sum_OdO_producer_state.advance()
        
        iter_count -= 2
        iter_index += 1

        while iter_count > 0:

            sQ = cute.make_tensor(
                sQ.iterator,
                cute.make_layout(((64,16),2,(4,2),2), stride=((64,1),4096,(16,8192),16384))
            )
            sQ_0 = sQ[None, 0, None, load_mma_Q_producer_state.index]
            sQ_1 = sQ[None, 1, None, load_mma_Q_producer_state.index]

            sdO = cute.make_tensor(
                sdO.iterator,
                cute.make_layout(((64,16),2,(4,2),2), stride=((64,1),4096,(16,8192),16384)) 
            )
            sdO_0 = sdO[None, 0, None, load_mma_dO_producer_state.index]
            sdO_1 = sdO[None, 1, None, load_mma_dO_producer_state.index]

            # ((atom_v, rest_v), STAGE)
            # ((atom_v, rest_v), RestM, RestK, (H, B))
            tQsQ_0, _ = cute.nvgpu.cpasync.tma_partition(
                tma_atom_Q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ_0, 0, 2),
                cute.group_modes(tSgQ, 0, 3)
            )
            tQsQ_1, _ = cute.nvgpu.cpasync.tma_partition(
                tma_atom_Q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ_1, 0, 2),
                cute.group_modes(tSgQ, 0, 3)
            )
            # ((atom_v, rest_v), STAGE)
            # ((atom_v, rest_v), RestM, RestK, (H, B))
            tdOsdO_0, _ = cute.nvgpu.cpasync.tma_partition(
                tma_atom_dO,
                0,
                cute.make_layout(1),
                cute.group_modes(sdO_0, 0, 2),
                cute.group_modes(tdPgdO, 0, 3)
            )
            tdOsdO_1, _ = cute.nvgpu.cpasync.tma_partition(
                tma_atom_dO,
                0,
                cute.make_layout(1),
                cute.group_modes(sdO_1, 0, 2),
                cute.group_modes(tdPgdO, 0, 3)
            )

            load_mma_Q_pipeline.producer_acquire(load_mma_Q_producer_state)
            tma_barrier = load_mma_Q_pipeline.producer_get_barrier(
                load_mma_Q_producer_state
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_Q_bytes)

            q_block_idx_0 = self._load_task_q_block(
                task_q_indices, task_start, iter_index, (blk_coord_h, blk_coord_b)
            )
            iter_index += 1
            q_block_idx_1 = seqlen_q // self.sparse_block_size # out of box, tma can fill zeros automatically
            if iter_index < total_iter_count:
                q_block_idx_1 = self._load_task_q_block(
                    task_q_indices, task_start, iter_index, (blk_coord_h, blk_coord_b)
                )
            q_block_0_full = (q_block_idx_0 + 1) * self.sparse_block_size <= seqlen_q
            
            # Load Q0
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[(None, q_block_idx_0, 0, (blk_coord_h, blk_coord_b))],
                tQsQ_0,
                tma_bar_ptr=tma_barrier,
            )

            # Load Q1
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[(None, q_block_idx_1, 0, (blk_coord_h, blk_coord_b))],
                tQsQ_1,
                tma_bar_ptr=tma_barrier,
            )

            load_mma_Q_producer_state.advance()

            load_compute_LSE_pipeline.producer_acquire(load_compute_LSE_producer_state)

            # Load LSE
            # 32 threads load 64 values, each thread loads 2 values
            sLSE_for_copy = cute.flat_divide(sLSE, (1,))
            LSE_for_copy = cute.flat_divide(LSE, (1,))
            for i in cutlass.range_constexpr(async_copy_num_elts):
                copy_offset = i * self.threads_per_warp + thread_idx
                LSE_idx = q_block_idx_0 * self.sparse_block_size + copy_offset
                if q_block_0_full:
                    cute.copy(
                        atom_async_copy,
                        LSE_for_copy[None, LSE_idx, (blk_coord_h, blk_coord_b)],
                        sLSE_for_copy[
                            None,
                            copy_offset,
                            load_compute_LSE_producer_state.index,
                        ],
                    )
                elif cute.elem_less(LSE_idx, seqlen_q):
                    cute.copy(
                        atom_async_copy,
                        LSE_for_copy[None, LSE_idx, (blk_coord_h, blk_coord_b)],
                        sLSE_for_copy[
                            None,
                            copy_offset,
                            load_compute_LSE_producer_state.index,
                        ],
                    )
                else:
                    sLSE_for_copy[
                        None,
                        copy_offset,
                        load_compute_LSE_producer_state.index,
                    ].fill(0.0)

            for i in cutlass.range_constexpr(async_copy_num_elts):
                copy_offset = i * self.threads_per_warp + thread_idx
                LSE_idx = q_block_idx_1 * self.sparse_block_size + copy_offset
                if cute.elem_less(LSE_idx, seqlen_q):
                    cute.copy(
                        atom_async_copy,
                        LSE_for_copy[None, LSE_idx, (blk_coord_h, blk_coord_b)],
                        sLSE_for_copy[
                            None,
                            self.sparse_block_size + copy_offset,
                            load_compute_LSE_producer_state.index,
                        ],
                    )
                else:
                    sLSE_for_copy[
                        None,
                        self.sparse_block_size + copy_offset,
                        load_compute_LSE_producer_state.index,
                    ].fill(0.0)
            
            load_compute_LSE_pipeline.producer_commit(load_compute_LSE_producer_state)
            load_compute_LSE_producer_state.advance()

            load_mma_dO_pipeline.producer_acquire(load_mma_dO_producer_state)
            tma_barrier = load_mma_dO_pipeline.producer_get_barrier(
                load_mma_dO_producer_state
            )
            with cute.arch.elect_one():
                cute.arch.mbarrier_expect_tx(tma_barrier, self.tma_copy_dO_bytes)

            # Load dO0
            cute.copy(
                tma_atom_dO,
                tdOgdO_mkl[(None, q_block_idx_0, 0, (blk_coord_h, blk_coord_b))],
                tdOsdO_0,
                tma_bar_ptr=tma_barrier,
            )
            # Load dO1
            cute.copy(
                tma_atom_dO,
                tdOgdO_mkl[(None, q_block_idx_1, 0, (blk_coord_h, blk_coord_b))],
                tdOsdO_1,
                tma_bar_ptr=tma_barrier,
            )

            load_mma_dO_producer_state.advance()

            load_compute_sum_OdO_pipeline.producer_acquire(
                load_compute_sum_OdO_producer_state
            )

            sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))
            sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1,))
            for i in cutlass.range_constexpr(async_copy_num_elts):
                copy_offset = i * self.threads_per_warp + thread_idx
                sum_OdO_idx = q_block_idx_0 * self.sparse_block_size + copy_offset
                if q_block_0_full:
                    cute.copy(
                        atom_async_copy,
                        sum_OdO_for_copy[None, sum_OdO_idx, (blk_coord_h, blk_coord_b)],
                        sSum_OdO_for_copy[
                            None,
                            copy_offset,
                            load_compute_sum_OdO_producer_state.index,
                        ],
                    )
                elif cute.elem_less(sum_OdO_idx, seqlen_q):
                    cute.copy(
                        atom_async_copy,
                        sum_OdO_for_copy[None, sum_OdO_idx, (blk_coord_h, blk_coord_b)],
                        sSum_OdO_for_copy[
                            None,
                            copy_offset,
                            load_compute_sum_OdO_producer_state.index,
                        ],
                    )
                else:
                    sSum_OdO_for_copy[
                        None,
                        copy_offset,
                        load_compute_sum_OdO_producer_state.index,
                    ].fill(0.0)
            for i in cutlass.range_constexpr(async_copy_num_elts):
                copy_offset = i * self.threads_per_warp + thread_idx
                sum_OdO_idx = q_block_idx_1 * self.sparse_block_size + copy_offset
                if cute.elem_less(sum_OdO_idx, seqlen_q):
                    cute.copy(
                        atom_async_copy,
                        sum_OdO_for_copy[None, sum_OdO_idx, (blk_coord_h, blk_coord_b)],
                        sSum_OdO_for_copy[
                            None,
                            self.sparse_block_size + copy_offset,
                            load_compute_sum_OdO_producer_state.index,
                        ],
                    )
                else:
                    sSum_OdO_for_copy[
                        None,
                        self.sparse_block_size + copy_offset,
                        load_compute_sum_OdO_producer_state.index,
                    ].fill(0.0)
            
            load_compute_sum_OdO_pipeline.producer_commit(
                load_compute_sum_OdO_producer_state
            )
            load_compute_sum_OdO_producer_state.advance()

            iter_count -= 2
            iter_index += 1

    @cute.jit
    def mma(
        self,
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        dSK_tiled_mma: cute.TiledMma,
        tStS: cute.Tensor,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdPrdO: cute.Tensor,
        tdPrV: cute.Tensor,
        tdVTtdVT: cute.Tensor,
        tdVTrdOT: cute.Tensor,
        tdVTrP: cute.Tensor,
        tdQtdQ: cute.Tensor,
        tdQrdS: cute.Tensor,
        tdQrKT: cute.Tensor,
        tdKTtdKT: cute.Tensor,
        tdKTrQT: cute.Tensor,
        tdKTrdST: cute.Tensor,
        iter_count: Int32,
        pipeline_args: tuple,
    ):
        (
            load_mma_Q_pipeline,
            mma_compute_S_pipeline,
            load_mma_dO_pipeline,
            mma_compute_dP_pipeline,
            mma_reduce_dQ_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dKdV_pipeline,
        ) = pipeline_args
        
        load_mma_Q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_mma_Q_stage
        )
        load_mma_Q_release_state = load_mma_Q_consumer_state.clone()
        mma_compute_S_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_compute_S_stage
        )
        compute_mma_dS_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage
        )
        mma_compute_dP_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_compute_dP_stage
        )
        mma_reduce_dQ_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_reduce_dQ_stage
        )
        load_mma_dO_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_mma_dO_stage
        )
        compute_mma_P_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.compute_mma_P_stage
        )
        mma_compute_dKdV_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_compute_dKdV_stage
        )

        load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
        mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)

        # S = Q * K
        QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll_full=True):
            cute.gemm(
                QK_tiled_mma,
                tStS,
                tSrQ[None, None, k_block, load_mma_Q_consumer_state.index],
                tSrK[None, None, k_block, 0],
                tStS,
            )
            QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        
        load_mma_Q_consumer_state.advance()
        mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
        mma_compute_S_producer_state.advance()

        load_mma_dO_pipeline.consumer_wait(load_mma_dO_consumer_state)

        mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)
        mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)

        # dP = dO * V
        dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(0, cute.size(tdPrdO, mode=[2]), unroll_full=True):
            cute.gemm(
                dOV_tiled_mma,
                tdPtdP,
                tdPrdO[None, None, k_block, load_mma_dO_consumer_state.index],
                tdPrV[None, None, k_block, 0],
                tdPtdP
            )
            dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        
        mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
        mma_compute_dP_producer_state.advance()

        compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)

        # dV = dO * P
        dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(0, cute.size(tdVTrdOT, mode=[2]), unroll_full=True):
            cute.gemm(
                dOP_tiled_mma,
                tdVTtdVT,
                tdVTrdOT[None, None, k_block, load_mma_dO_consumer_state.index],
                tdVTrP[None, None, k_block, 0],
                tdVTtdVT,
            )
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        
        compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
        compute_mma_P_consumer_state.advance()

        load_mma_dO_pipeline.consumer_release(load_mma_dO_consumer_state)
        load_mma_dO_consumer_state.advance()

        iter_count -= 1

        while iter_count > 0:
            load_mma_Q_pipeline.consumer_wait(load_mma_Q_consumer_state)
            mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)

            # S = Q * K
            QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll_full=True):
                cute.gemm(
                    QK_tiled_mma,
                    tStS,
                    tSrQ[None, None, k_block, load_mma_Q_consumer_state.index],
                    tSrK[None, None, k_block, 0],
                    tStS,
                )
                QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            load_mma_Q_consumer_state.advance()
            mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
            mma_compute_S_producer_state.advance()

            compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)

            mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)

            # dQ = dS * K
            dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(
                0, cute.size(tdQrdS, mode=[2]), unroll_full=True
            ):
                cute.gemm(
                    dSK_tiled_mma,
                    tdQtdQ,
                    tdQrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQrKT[None, None, k_block, 0],
                    tdQtdQ
                )
                dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            
            mma_reduce_dQ_pipeline.producer_commit(mma_reduce_dQ_producer_state)
            mma_reduce_dQ_producer_state.advance()

            # dK = Q * dS
            for k_block in cutlass.range(
                0, cute.size(tdKTrQT, mode=[2]), unroll_full=True
            ):
                cute.gemm(
                    QdS_tiled_mma,
                    tdKTtdKT,
                    tdKTrQT[None, None, k_block, load_mma_Q_release_state.index],
                    tdKTrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdKTtdKT,
                )
                QdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            
            load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
            load_mma_Q_release_state.advance()

            compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
            compute_mma_dS_consumer_state.advance()

            mma_reduce_dQ_pipeline.producer_acquire(mma_reduce_dQ_producer_state)
            load_mma_dO_pipeline.consumer_wait(load_mma_dO_consumer_state)

            # dP = dO * V
            dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(
                0, cute.size(tdPrdO, mode=[2]), unroll_full=True
            ):
                cute.gemm(
                    dOV_tiled_mma,
                    tdPtdP,
                    tdPrdO[None, None, k_block, load_mma_dO_consumer_state.index],
                    tdPrV[None, None, k_block, 0],
                    tdPtdP,
                )
                dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            
            mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
            mma_compute_dP_producer_state.advance()

            compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state) 

            # dV = dO * P
            for k_block in cutlass.range(
                0, cute.size(tdVTrdOT, mode=[2]), unroll_full=True
            ):
                cute.gemm(
                    dOP_tiled_mma,
                    tdVTtdVT,
                    tdVTrdOT[None, None, k_block, load_mma_dO_consumer_state.index],
                    tdVTrP[None, None, k_block, 0],
                    tdVTtdVT,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            
            compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
            compute_mma_P_consumer_state.advance()

            load_mma_dO_pipeline.consumer_release(load_mma_dO_consumer_state)
            load_mma_dO_consumer_state.advance()

            iter_count -= 1
        
        mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_producer_state)
        mma_compute_dKdV_pipeline.producer_commit(mma_compute_dKdV_producer_state)
        mma_compute_dKdV_producer_state.advance()

        mma_compute_dKdV_pipeline.producer_acquire(mma_compute_dKdV_producer_state)

        compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)

        # dK = Q * dS
        for k_block in cutlass.range(
            0, cute.size(tdKTrQT, mode=[2]), unroll_full=True
        ):
            cute.gemm(
                QdS_tiled_mma,
                tdKTtdKT,
                tdKTrQT[None, None, k_block, load_mma_Q_release_state.index],
                tdKTrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                tdKTtdKT,
            )
            QdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        
        mma_compute_dKdV_pipeline.producer_commit(mma_compute_dKdV_producer_state)
        mma_compute_dKdV_producer_state.advance()

        # dQ = dS * K
        dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_block in cutlass.range(
            0, cute.size(tdQrdS, mode=[2]), unroll_full=True
        ):
            cute.gemm(
                dSK_tiled_mma,
                tdQtdQ,
                tdQrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                tdQrKT[None, None, k_block, 0],
                tdQtdQ
            )
            dSK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        
        mma_reduce_dQ_pipeline.producer_commit(mma_reduce_dQ_producer_state)
        mma_reduce_dQ_producer_state.advance()

        load_mma_Q_pipeline.consumer_release(load_mma_Q_release_state)
        load_mma_Q_release_state.advance()

        compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
        compute_mma_dS_consumer_state.advance()


    @cute.jit
    def compute(
        self,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVTrP: cute.Tensor,
        sLSE: cute.Tensor,
        sdS: cute.Tensor,
        sP: cute.Tensor,
        sSum_OdO: cute.Tensor,
        dK_acc: cute.Tensor,
        dV_acc: cute.Tensor,
        sdKV: cute.Tensor,
        tdKTtdKT: cute.Tensor,
        tdVTtdVT: cute.Tensor,
        kv_block_idx: Int32,
        variable_block_sizes: cute.Tensor,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        iter_count: Int32,
        scale_softmax: Float32,
        pipeline_args: tuple,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, blk_coord_h, blk_coord_b = cute.arch.block_idx()
        seqlen_q, seqlen_k, head_dim, HB = problem_shape
        num_heads, batch_size = HB
        (
            mma_compute_S_pipeline,
            compute_mma_P_pipeline,
            load_compute_LSE_pipeline,
            load_compute_sum_OdO_pipeline,
            mma_compute_dP_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dKdV_pipeline,
        ) = pipeline_args

        mma_compute_S_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_compute_S_stage
        )
        compute_mma_P_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.compute_mma_P_stage
        )
        load_compute_LSE_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_compute_LSE_stage
        )
        load_compute_sum_OdO_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_compute_sum_OdO_stage
        )
        mma_compute_dP_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_compute_dP_stage
        )
        compute_mma_dS_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.compute_mma_dS_stage
        )
        mma_compute_dKdV_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_compute_dKdV_stage
        )

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)),
            self.acc_dtype,
        )

        # (128, 64)
        tStS = tStS[(None, None), 0, 0]
        # (128, 64)
        tdPtdP = tdPtdP[(None, None), 0, 0]

        cS = cute.make_identity_tensor(cute.select(self.QK_mma_tiler, mode=[0, 1]))
        cdP = cute.make_identity_tensor(cute.select(self.dOV_mma_tiler, mode=[0, 1]))

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        tTR_cS_p = thr_t2r.partition_D(cS)
        tTR_cS = self.split_wg(tTR_cS_p, num_warp_groups, wg_idx)
        tTR_rS = cute.make_rmem_tensor(tTR_cS.shape, self.acc_dtype)

        tTR_tS = thr_t2r.partition_S(tStS)
        tTR_tS = self.split_wg(tTR_tS, num_warp_groups, wg_idx)

        tTR_cdP_p = thr_t2r.partition_D(cdP)
        tTR_cdP = self.split_wg(tTR_cdP_p, num_warp_groups, wg_idx)
        tTR_rdP = cute.make_rmem_tensor(tTR_cdP.shape, self.acc_dtype)

        tTR_tdP = thr_t2r.partition_S(tdPtdP)
        tTR_tdP = self.split_wg(tTR_tdP, num_warp_groups, wg_idx)

        block_size_k = Int32(self.sparse_block_size)
        if cutlass.const_expr(self.has_block_sizes):
            block_size_k = variable_block_sizes[blk_coord_b, kv_block_idx]

        while iter_count > 0:
            # Wait for S and P
            mma_compute_S_pipeline.consumer_wait(mma_compute_S_consumer_state)
            compute_mma_P_pipeline.producer_acquire(compute_mma_P_producer_state)
            # Wait for LSE
            load_compute_LSE_pipeline.consumer_wait(load_compute_LSE_consumer_state)

            # Compute P = softmax(S, LSE)
            cute.copy(tiled_t2r, tTR_tS, tTR_rS)

            if cutlass.const_expr(self.has_block_sizes):
                apply_block_size_mask(
                    tTR_rS,
                    block_size=block_size_k,
                    n_block_size=self.sparse_block_size,
                )
            
            log2_e = Float32(math.log2(math.e))
            softmax_scale_log2_e = scale_softmax * log2_e

            for i in cutlass.range(0, cute.size(tTR_rS), 2, unroll_full=True):
                lse = (
                    sLSE[
                        cute.get(tTR_cS[i], mode=[0]),
                        load_compute_LSE_consumer_state.index,
                    ],
                    sLSE[
                        cute.get(tTR_cS[i + 1], mode=[0]),
                        load_compute_LSE_consumer_state.index,
                    ],
                )
                tTR_rS[i], tTR_rS[i + 1] = cute.arch.fma_packed_f32x2(
                    (tTR_rS[i], tTR_rS[i + 1]),
                    (softmax_scale_log2_e, softmax_scale_log2_e),
                    lse,
                )
                tTR_rS[i] = cute.math.exp2(tTR_rS[i], fastmath=True)
                tTR_rS[i + 1] = cute.math.exp2(tTR_rS[i + 1], fastmath=True)
            
            # convert fp32 P to bf16 P which will be used in the dOP
            tRS_rP = self.quantize(tTR_rS, 4)
            
            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()
            cute.arch.fence_view_async_tmem_load()

            # store to smem P
            sP_slice = sP[None, None, None, compute_mma_P_producer_state.index]
            thread_layout = cute.make_ordered_layout((128, 64), (1, 0))
            sP_slice_tmp = cute.composition(sP_slice, thread_layout)
            sP_slice_p = cute.composition(
                sP_slice_tmp[dp_idx, None], cute.make_layout(tTR_cS_p.shape)
            )
            sP_slice = self.split_wg(sP_slice_p, num_warp_groups, wg_idx)
            cute.autovec_copy(tRS_rP, sP_slice)

            # Fence for shared memory
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )

            # Notify for P
            compute_mma_P_pipeline.producer_commit(compute_mma_P_producer_state)
            compute_mma_P_producer_state.advance()

            # Release S
            mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
            mma_compute_S_consumer_state.advance()

            # Release LSE
            load_compute_LSE_pipeline.consumer_release(load_compute_LSE_consumer_state)
            load_compute_LSE_consumer_state.advance()

            # Wait for OdO
            load_compute_sum_OdO_pipeline.consumer_wait(
                load_compute_sum_OdO_consumer_state
            )
            # Wait for dP
            mma_compute_dP_pipeline.consumer_wait(mma_compute_dP_consumer_state)

            # Wait for dS
            compute_mma_dS_pipeline.producer_acquire(compute_mma_dS_producer_state)

            # Compute dS = dsoftmax(P, dP, sum_OdO)
            cute.copy(tiled_t2r, tTR_tdP, tTR_rdP)

            for i in cutlass.range(0, cute.size(tTR_rdP), 2, unroll_full=True):
                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.add_packed_f32x2(
                    (tTR_rdP[i], tTR_rdP[i + 1]),
                    (
                        sSum_OdO[
                            cute.get(tTR_cdP[i], mode=[0]),
                            load_compute_sum_OdO_consumer_state.index,
                        ],
                        sSum_OdO[
                            cute.get(tTR_cdP[i + 1], mode=[0]),
                            load_compute_sum_OdO_consumer_state.index,
                        ],
                    ),
                )
                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.mul_packed_f32x2(
                    (tTR_rdP[i], tTR_rdP[i + 1]), (tTR_rS[i], tTR_rS[i + 1])
                )
            
            # convert fp32 dS to bf16 dS which will be used in the computation of dK and dQ
            tTR_rdS = self.quantize(tTR_rdP, 4)

            # Release dP
            cute.arch.fence_view_async_tmem_load()
            mma_compute_dP_pipeline.consumer_release(mma_compute_dP_consumer_state)
            mma_compute_dP_consumer_state.advance()

            sdS_slice = sdS[None, None, None, compute_mma_dS_producer_state.index]

            thread_layout = cute.make_ordered_layout((128, 64), (0, 1))
            sdS_slice_tmp = cute.composition(sdS_slice, thread_layout)
            sdS_slice_p = cute.composition(
                sdS_slice_tmp[dp_idx, None], cute.make_layout(tTR_cdP_p.shape)
            )
            sdS_slice = self.split_wg(sdS_slice_p, num_warp_groups, wg_idx)

            cute.autovec_copy(tTR_rdS, sdS_slice)

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
            compute_mma_dS_producer_state.advance()

            # Release OdO
            load_compute_sum_OdO_pipeline.consumer_release(load_compute_sum_OdO_consumer_state)
            load_compute_sum_OdO_consumer_state.advance()

            iter_count -= 1
        
        self.epilogue(
            problem_shape,
            dK_acc,
            dV_acc,
            sdKV,
            tdKTtdKT,
            tdVTtdVT,
            dOP_tiled_mma,
            QdS_tiled_mma,
            kv_block_idx,
            (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state),
        )

    @cute.jit
    def reduce(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        tdQtdQ: cute.Tensor,
        dSK_tiled_mma: cute.TiledMma,
        task_q_indices: cute.Tensor,
        task_start: Int32,
        tma_atom_dQ_acc: cute.CopyAtom,
        mdQ_acc_tma: cute.Tensor,
        sdQ: cute.Tensor,
        iter_count: Int32,
        pipeline_args: tuple,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        Q, K, D, HB = problem_shape
        H, B = HB
        _, blk_coord_h, blk_coord_b = cute.arch.block_idx()
        mma_reduce_dQ_pipeline, reduce_tma_store_pipeline = pipeline_args
        total_iter_count = iter_count

        mma_reduce_dQ_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_reduce_dQ_stage
        )
        reduce_tma_store_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.reduce_tma_store_stage
        )

        thread_idx = tidx % (self.num_reduce_warps * self.threads_per_warp)
        tdQtdQ = tdQtdQ[(None, None), 0, 0]

        load_op = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol)),
            self.acc_dtype,
        )
        tiled_t2r = tcgen05.make_tmem_copy(load_op, tdQtdQ)
        thr_t2r = tiled_t2r.get_slice(thread_idx)
        tTR_tdQ = thr_t2r.partition_S(tdQtdQ)

        cdQ = cute.make_identity_tensor((self.dSK_mma_tiler[0], self.dSK_mma_tiler[1]))
        tTR_cdQ = thr_t2r.partition_D(cdQ)
        tdQsdQ_t2r = thr_t2r.partition_D(sdQ)
        gdQ = cute.local_tile(
            mdQ_acc_tma,
            (self.sparse_block_size, self.dQ_reduce_ncol),
            (None, None, None),
        )
        sdQ_tma = cute.make_tensor(
            sdQ.iterator,
            cute.make_layout(
                (
                    (8, 8),
                    2,
                    (self.dQ_reduce_ncol, 1),
                    self.reduce_tma_store_stage,
                ),
                stride=(
                    (self.dQ_reduce_ncol, self.dQ_reduce_ncol * 8),
                    self.dQ_reduce_ncol * self.sparse_block_size,
                    (1, 0),
                    self.dSK_mma_tiler[0] * self.dQ_reduce_ncol,
                ),
            ),
        )

        iter_index = Int32(0)

        while iter_count > 0:
            
            mma_reduce_dQ_pipeline.consumer_wait(mma_reduce_dQ_consumer_state)

            q_block_idx_0 = self._load_task_q_block(
                task_q_indices, task_start, iter_index, (blk_coord_h, blk_coord_b)
            )
            iter_index += 1
            has_q1 = iter_index < total_iter_count
            q_block_idx_1 = cute.ceil_div(Q, self.sparse_block_size)
            if has_q1:
                q_block_idx_1 = self._load_task_q_block(
                    task_q_indices, task_start, iter_index, (blk_coord_h, blk_coord_b)
                )

            tTR_rdQ = cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype)
            cute.copy(thr_t2r, tTR_tdQ, tTR_rdQ)
            cute.arch.fence_view_async_tmem_load()
            mma_reduce_dQ_pipeline.consumer_release(mma_reduce_dQ_consumer_state)
            mma_reduce_dQ_consumer_state.advance()

            for i in cutlass.range_constexpr(cute.size(tTR_cdQ, mode=[2])):
                if warp_idx == 0:
                    reduce_tma_store_pipeline.producer_acquire()
                self.reduce_sync_barrier.arrive_and_wait()

                smem_idx = reduce_tma_store_producer_state.index
                cute.autovec_copy(
                    tTR_rdQ[None, None, i],
                    tdQsdQ_t2r[None, None, 0, smem_idx],
                )

                cute.arch.fence_proxy("async.shared", space="cta")
                self.reduce_sync_barrier.arrive_and_wait()

                if warp_idx == 0:
                    sdQ_0 = sdQ_tma[None, 0, None, smem_idx]
                    tdQsdQ_0, tdQgdQ = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_dQ_acc,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sdQ_0, 0, 2),
                        cute.group_modes(gdQ, 0, 2),
                    )
                    cute.copy(
                        tma_atom_dQ_acc,
                        tdQsdQ_0,
                        tdQgdQ[
                            None,
                            q_block_idx_0,
                            i,
                            (blk_coord_h, blk_coord_b),
                        ],
                    )
                    if has_q1:
                        sdQ_1 = sdQ_tma[None, 1, None, smem_idx]
                        tdQsdQ_1, tdQgdQ = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_dQ_acc,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sdQ_1, 0, 2),
                            cute.group_modes(gdQ, 0, 2),
                        )
                        cute.copy(
                            tma_atom_dQ_acc,
                            tdQsdQ_1,
                            tdQgdQ[
                                None,
                                q_block_idx_1,
                                i,
                                (blk_coord_h, blk_coord_b),
                            ],
                        )
                    reduce_tma_store_pipeline.producer_commit()

                self.reduce_sync_barrier.arrive_and_wait()

                reduce_tma_store_producer_state.advance()

            iter_count -= 2
            iter_index += 1
        
        reduce_tma_store_pipeline.producer_tail()

    @cute.jit
    def store_add_fp32_dkv_stage_atomic(
        self,
        gmem: cute.Tensor,
        tdKVtdKV: cute.Tensor,
        kv_block_idx: Int32,
        blk_coord_h: Int32,
        blk_coord_b: Int32,
        wg_idx: Int32,
        dp_idx: Int32,
    ):
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dKV_tmem_load_ncol)),
            self.acc_dtype,
        )
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV)
        thr_t2r = tiled_t2r.get_slice(dp_idx)
        num_warp_groups = self.num_compute_warps // 4

        tdKVtdKV_t2r = thr_t2r.partition_S(tdKVtdKV)
        tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r, num_warp_groups, wg_idx)

        cdKV = cute.make_identity_tensor((self.QdS_mma_tiler[0], self.QdS_mma_tiler[1]))
        tdKVcdKV_t2r = thr_t2r.partition_D(cdKV)
        tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r, num_warp_groups, wg_idx)

        regs = cute.make_rmem_tensor(tdKVcdKV_t2r.shape, self.acc_dtype)
        cute.copy(thr_t2r, tdKVtdKV_t2r, regs)
        cute.arch.fence_view_async_tmem_load()

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.acc_dtype,
        )
        copy_op = cute.make_cotiled_copy(
            copy_atom,
            cute.make_layout((1, 128 // self.acc_dtype.width)),
            regs.layout,
        )
        thr_copy = copy_op.get_slice(0)
        tCr = thr_copy.partition_S(regs)
        tPc = thr_copy.partition_D(tdKVcdKV_t2r)

        for v in cutlass.range_constexpr(tPc.shape[0][1]):
            for m in cutlass.range_constexpr(tPc.shape[1]):
                for n in cutlass.range_constexpr(tPc.shape[2]):
                    for k in cutlass.range_constexpr(tPc.shape[3]):
                        coord = ((0, v), m, n, k)
                        logical_coord = tPc[coord]
                        d_col = logical_coord[0]
                        kv_local = logical_coord[1]
                        d_stage = d_col // self.dKV_reduce_ncol
                        d_in_stage = d_col - d_stage * self.dKV_reduce_ncol
                        stage_row = kv_local * Int32(self.dKV_reduce_ncol) + d_in_stage
                        ptr = gmem.iterator + cute.crd2idx(
                            (
                                stage_row,
                                d_stage,
                                kv_block_idx,
                                (blk_coord_h, blk_coord_b),
                            ),
                            gmem.layout,
                        )
                        cute.arch.atomic_add(
                            ptr.llvm_ptr,
                            tCr[coord],
                            sem="relaxed",
                            scope="gpu",
                        )

    @cute.jit
    def store_add_fp32_dkv_linear_atomic(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        gmem: cute.Tensor,
        tdKVtdKV: cute.Tensor,
        kv_block_idx: Int32,
        wg_idx: Int32,
        dp_idx: Int32,
    ):
        _, K, D, _ = problem_shape
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dKV_tmem_load_ncol)),
            self.acc_dtype,
        )
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        gdKV = cute.local_tile(
            gmem, cute.select(self.QdS_mma_tiler, mode=[0, 1]), (None, None, None)
        )
        _, blk_coord_h, blk_coord_b = cute.arch.block_idx()
        gdKV = gdKV[None, None, 0, kv_block_idx, (blk_coord_h, blk_coord_b)]

        cdKV = cute.domain_offset(
            (0, kv_block_idx * self.QdS_mma_tiler[1]),
            cute.make_identity_tensor((self.QdS_mma_tiler[0], self.QdS_mma_tiler[1])),
        )
        num_warp_groups = self.num_compute_warps // 4

        tTR_cdKV = thr_t2r.partition_D(cdKV)
        tTR_cdKV = self.split_wg(tTR_cdKV, num_warp_groups, wg_idx)
        tTR_gdKV = thr_t2r.partition_D(gdKV)
        tTR_gdKV = self.split_wg(tTR_gdKV, num_warp_groups, wg_idx)
        tTR_rdKV = cute.make_rmem_tensor(tTR_cdKV.shape, self.acc_dtype)
        tTR_tdKV = thr_t2r.partition_S(tdKVtdKV)
        tTR_tdKV = self.split_wg(tTR_tdKV, num_warp_groups, wg_idx)

        cute.copy(tiled_t2r, tTR_tdKV, tTR_rdKV)
        if cutlass.const_expr(self.full_kv_blocks):
            self.store_add_fp32_full(tTR_gdKV, tTR_rdKV, tTR_cdKV)
        else:
            self.store_add_fp32(tTR_gdKV, tTR_rdKV, tTR_cdKV, (D, K))
        cute.arch.fence_view_async_tmem_load()

    @cute.jit
    def store_add_fp32_bulk(
        self,
        gmem: cute.Tensor,
        tdKVtdKV: cute.Tensor,
        sdKV: cute.Tensor,
        kv_block_idx: Int32,
        blk_coord_h: Int32,
        blk_coord_b: Int32,
        wg_idx: Int32,
        dp_idx: Int32,
    ):
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dKV_tmem_load_ncol)),
            self.acc_dtype,
        )
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV)
        thr_t2r = tiled_t2r.get_slice(dp_idx)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        leader_warp = warp_idx % 4 == 0
        num_warp_groups = self.num_compute_warps // 4
        num_epi_stages = self.QdS_mma_tiler[0] // self.dKV_reduce_ncol
        rows_per_wg = self.QdS_mma_tiler[1] // num_warp_groups
        row_groups_per_wg = 2
        rows_per_group = rows_per_wg // row_groups_per_wg
        sdKV_wg = cute.make_tensor(
            sdKV.iterator + wg_idx * Int32(rows_per_wg * self.QdS_mma_tiler[0]),
            cute.make_layout(
                (rows_per_wg, self.QdS_mma_tiler[0]),
                stride=(self.QdS_mma_tiler[0], 1),
            ),
        )

        tdKVtdKV_t2r = thr_t2r.partition_S(tdKVtdKV)
        tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r, num_warp_groups, wg_idx)

        cdKV = cute.make_identity_tensor((self.QdS_mma_tiler[0], self.QdS_mma_tiler[1]))
        tdKVcdKV_t2r = thr_t2r.partition_D(cdKV)
        tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r, num_warp_groups, wg_idx)

        regs = cute.make_rmem_tensor(tdKVcdKV_t2r.shape, self.acc_dtype)
        cute.copy(thr_t2r, tdKVtdKV_t2r, regs)
        cute.arch.fence_view_async_tmem_load()

        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.acc_dtype,
        )
        copy_op = cute.make_cotiled_copy(
            copy_atom,
            cute.make_layout((1, 128 // self.acc_dtype.width)),
            regs.layout,
        )
        thr_copy = copy_op.get_slice(0)
        tCr = thr_copy.partition_S(regs)
        tPc = thr_copy.partition_D(tdKVcdKV_t2r)

        for v in cutlass.range_constexpr(tPc.shape[0][1]):
            for m in cutlass.range_constexpr(tPc.shape[1]):
                for n in cutlass.range_constexpr(tPc.shape[2]):
                    for k in cutlass.range_constexpr(tPc.shape[3]):
                        coord = ((0, v), m, n, k)
                        logical_coord = tPc[coord]
                        col = logical_coord[0]
                        local_row = Int32(v + m * tPc.shape[0][1] + k * rows_per_group)
                        sdKV_wg[local_row, col] = tCr[coord]

        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=6 + wg_idx, number_of_threads=128)

        if leader_warp:
            for epi_stage in cutlass.range_constexpr(num_epi_stages):
                local_col_stage_start = Int32(epi_stage * self.dKV_reduce_ncol)
                with cute.arch.elect_one():
                    for row_group in cutlass.range_constexpr(row_groups_per_wg):
                        for row_in_group in cutlass.range_constexpr(rows_per_group):
                            local_row = row_group * rows_per_group + row_in_group
                            row = (
                                wg_idx * Int32(rows_per_group)
                                + Int32(row_in_group)
                                + Int32(row_group * rows_per_group * num_warp_groups)
                            )
                            cpasync_reduce_bulk_add_f32(
                                sdKV_wg.iterator
                                + Int32(local_row * self.QdS_mma_tiler[0])
                                + local_col_stage_start,
                                gmem.iterator
                                + cute.crd2idx(
                                    (
                                        row * Int32(self.dKV_reduce_ncol),
                                        epi_stage,
                                        kv_block_idx,
                                        (blk_coord_h, blk_coord_b),
                                    ),
                                    gmem.layout,
                                ),
                                self.dKV_reduce_ncol * Float32.width // 8,
                            )
                cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=6 + wg_idx, number_of_threads=128)

    @cute.jit
    def epilogue(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        dK_acc: cute.Tensor,
        dV_acc: cute.Tensor,
        sdKV: cute.Tensor,
        tdKTtdKT: cute.Tensor,
        tdVTtdVT: cute.Tensor,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        kv_block_idx: Int32,
        # (mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state)
        pipeline_args: tuple,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        Q, K, D, HB = problem_shape
        H, B = HB
        _, blk_coord_h, blk_coord_b = cute.arch.block_idx()
        mma_compute_dKdV_pipeline, mma_compute_dKdV_consumer_state = pipeline_args

        tdKTtdKT = tdKTtdKT[(None, None), 0, 0]
        tdVTtdVT = tdVTtdVT[(None, None), 0, 0]

        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        mma_compute_dKdV_pipeline.consumer_wait(mma_compute_dKdV_consumer_state)

        if cutlass.const_expr(self.use_dkv_tma_reduce):
            self.store_add_fp32_bulk(
                dV_acc,
                tdVTtdVT,
                sdKV,
                kv_block_idx,
                blk_coord_h,
                blk_coord_b,
                wg_idx,
                dp_idx,
            )
        elif cutlass.const_expr(self.use_dkv_stage_layout):
            self.store_add_fp32_dkv_stage_atomic(
                dV_acc,
                tdVTtdVT,
                kv_block_idx,
                blk_coord_h,
                blk_coord_b,
                wg_idx,
                dp_idx,
            )
        else:
            self.store_add_fp32_dkv_linear_atomic(
                problem_shape,
                dV_acc,
                tdVTtdVT,
                kv_block_idx,
                wg_idx,
                dp_idx,
            )

        mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        mma_compute_dKdV_consumer_state.advance()

        mma_compute_dKdV_pipeline.consumer_wait(mma_compute_dKdV_consumer_state)

        if cutlass.const_expr(self.use_dkv_tma_reduce):
            self.store_add_fp32_bulk(
                dK_acc,
                tdKTtdKT,
                sdKV,
                kv_block_idx,
                blk_coord_h,
                blk_coord_b,
                wg_idx,
                dp_idx,
            )
        elif cutlass.const_expr(self.use_dkv_stage_layout):
            self.store_add_fp32_dkv_stage_atomic(
                dK_acc,
                tdKTtdKT,
                kv_block_idx,
                blk_coord_h,
                blk_coord_b,
                wg_idx,
                dp_idx,
            )
        else:
            self.store_add_fp32_dkv_linear_atomic(
                problem_shape,
                dK_acc,
                tdKTtdKT,
                kv_block_idx,
                wg_idx,
                dp_idx,
            )
        mma_compute_dKdV_pipeline.consumer_release(mma_compute_dKdV_consumer_state)
        mma_compute_dKdV_consumer_state.advance()

    @cute.jit
    def store(
        self,
        gmem: cute.Tensor,
        regs: cute.Tensor,
        coord: cute.Tensor,
        tensor_shape: cute.Shape,
    ):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.element_dtype,
            # num_bits_per_copy=128,
        )
        copy_op = cute.make_cotiled_copy(
            copy_atom,
            cute.make_layout((1, 128 // self.element_dtype.width)),
            regs.layout,
        )
        thr_copy = copy_op.get_slice(0)

        tCg = thr_copy.partition_D(gmem)
        tCr = thr_copy.partition_S(self.quantize(regs, 4))
        tPc = thr_copy.partition_D(coord)

        preds_shape = (tPc.shape[0][1], tPc.shape[1], tPc.shape[2], tPc.shape[3])
        preds = cute.make_rmem_tensor(preds_shape, Boolean)
        for v in cutlass.range_constexpr(preds.shape[0]):
            for m in cutlass.range_constexpr(preds.shape[1]):
                for n in cutlass.range_constexpr(preds.shape[2]):
                    for k in cutlass.range_constexpr(preds.shape[3]):
                        lhs = tPc[(0, v), m, n, k]
                        val = cute.elem_less(lhs, tensor_shape)
                        preds[v, m, n, k] = val

        cute.copy(copy_atom, tCr, tCg, pred=preds)

    @cute.jit
    def store_add_fp32(
        self,
        gmem: cute.Tensor,
        regs: cute.Tensor,
        coord: cute.Tensor,
        tensor_shape: cute.Shape,
    ):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.acc_dtype,
        )
        copy_op = cute.make_cotiled_copy(
            copy_atom,
            cute.make_layout((1, 128 // self.acc_dtype.width)),
            regs.layout,
        )
        thr_copy = copy_op.get_slice(0)

        tCg = thr_copy.partition_D(gmem)
        tCr = thr_copy.partition_S(regs)
        tPc = thr_copy.partition_D(coord)

        preds_shape = (tPc.shape[0][1], tPc.shape[1], tPc.shape[2], tPc.shape[3])
        preds = cute.make_rmem_tensor(preds_shape, Boolean)
        for v in cutlass.range_constexpr(preds.shape[0]):
            for m in cutlass.range_constexpr(preds.shape[1]):
                for n in cutlass.range_constexpr(preds.shape[2]):
                    for k in cutlass.range_constexpr(preds.shape[3]):
                        lhs = tPc[(0, v), m, n, k]
                        preds[v, m, n, k] = cute.elem_less(lhs, tensor_shape)

        for v in cutlass.range_constexpr(preds.shape[0]):
            for m in cutlass.range_constexpr(preds.shape[1]):
                for n in cutlass.range_constexpr(preds.shape[2]):
                    for k in cutlass.range_constexpr(preds.shape[3]):
                        coord = ((0, v), m, n, k)
                        if preds[v, m, n, k]:
                            ptr = tCg.iterator + cute.crd2idx(coord, tCg.layout)
                            cute.arch.atomic_add(
                                ptr.llvm_ptr,
                                tCr[coord],
                                sem="relaxed",
                                scope="gpu",
                            )

    @cute.jit
    def store_add_fp32_full(
        self,
        gmem: cute.Tensor,
        regs: cute.Tensor,
        coord: cute.Tensor,
    ):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.acc_dtype,
        )
        copy_op = cute.make_cotiled_copy(
            copy_atom,
            cute.make_layout((1, 128 // self.acc_dtype.width)),
            regs.layout,
        )
        thr_copy = copy_op.get_slice(0)

        tCg = thr_copy.partition_D(gmem)
        tCr = thr_copy.partition_S(regs)
        tPc = thr_copy.partition_D(coord)

        for v in cutlass.range_constexpr(tPc.shape[0][1]):
            for m in cutlass.range_constexpr(tPc.shape[1]):
                for n in cutlass.range_constexpr(tPc.shape[2]):
                    for k in cutlass.range_constexpr(tPc.shape[3]):
                        coord = ((0, v), m, n, k)
                        ptr = tCg.iterator + cute.crd2idx(coord, tCg.layout)
                        cute.arch.atomic_add(
                            ptr.llvm_ptr,
                            tCr[coord],
                            sem="relaxed",
                            scope="gpu",
                        )

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        num_warp_groups: Int32,
        wg_idx: Int32,
    ) -> cute.Tensor:
        ret = None
        if cutlass.const_expr(cute.rank(t.layout) == 3):
            p = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        t.shape[1],
                        (num_warp_groups, cute.size(t, mode=[2]) // num_warp_groups),
                    )
                ),
            )
            ret = p[None, None, (wg_idx, None)]
        else:
            p = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        t.shape[1],
                        t.shape[2],
                        (num_warp_groups, cute.size(t, mode=[3]) // num_warp_groups),
                    )
                ),
            )
            ret = p[None, None, None, (wg_idx, None)]
        return ret

    @cute.jit
    def split_wg_tmem(
        self,
        t: cute.Tensor,
        wg_idx: Int32,
        num_warp_groups: cutlass.Constexpr[int],
    ) -> cute.Tensor:
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if cutlass.const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for t in split_wg_tmem"
            t = cute.logical_divide(
                t, (reduced_shape[0], reduced_shape[1] // num_warp_groups)
            )
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for t in split_wg_tmem"
            if cutlass.const_expr(rank == 3):
                t = cute.logical_divide(
                    t,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2] // num_warp_groups,
                    ),
                )
                coord = (None, None, (None, wg_idx)) + (None,) * (rank - 3)
            else:
                t = cute.logical_divide(
                    t,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2],
                        reduced_shape[3] // num_warp_groups,
                    ),
                )
                coord = (
                    None,
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 4)
        return t[coord]


    @cute.jit
    def quantize(
        self,
        input: cute.Tensor,
        frg_cnt: Int32,
    ) -> cute.Tensor:
        output = cute.make_rmem_tensor(input.shape, self.element_dtype)
        frg_tile = cute.size(input) // frg_cnt
        t_frg = cute.logical_divide(input, cute.make_layout(frg_cnt))
        output_frg = cute.make_tensor(output.iterator, t_frg.layout)
        for i in cutlass.range(frg_tile, unroll_full=True):
            frg_vec = t_frg[None, i].load()
            output_frg[None, i].store(frg_vec.to(self.element_dtype))
        return output

    def make_and_init_load_mma_Q_pipeline(self, load_mma_Q_mbar_ptr):
        load_mma_Q_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_Q_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_Q_mbar_ptr,
            num_stages=self.load_mma_Q_stage,
            producer_group=load_mma_Q_producer_group,
            consumer_group=load_mma_Q_consumer_group,
            tx_count=self.tma_copy_Q_bytes,
        )

    def make_and_init_load_mma_dO_pipeline(self, load_mma_dO_mbar_ptr):
        load_mma_dO_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_dO_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_dO_mbar_ptr,
            num_stages=self.load_mma_dO_stage,
            producer_group=load_mma_dO_producer_group,
            consumer_group=load_mma_dO_consumer_group,
            tx_count=self.tma_copy_dO_bytes,
        )

    def make_and_init_load_compute_LSE_pipeline(self, load_compute_lse_mbar_ptr):
        load_compute_lse_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
        )
        load_compute_lse_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_compute_lse_mbar_ptr,
            num_stages=self.load_compute_LSE_stage,
            producer_group=load_compute_lse_producer_group,
            consumer_group=load_compute_lse_consumer_group,
        )

    def make_and_init_load_compute_sum_OdO_pipeline(
        self, load_compute_sum_OdO_mbar_ptr
    ):
        load_compute_sum_OdO_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
        )
        load_compute_sum_OdO_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_compute_sum_OdO_mbar_ptr,
            num_stages=self.load_compute_sum_OdO_stage,
            producer_group=load_compute_sum_OdO_producer_group,
            consumer_group=load_compute_sum_OdO_consumer_group,
        )

    def make_and_init_mma_compute_S_pipeline(self, mma_compute_S_mbar_ptr):
        mma_compute_S_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_S_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_S_mbar_ptr,
            num_stages=self.mma_compute_S_stage,
            producer_group=mma_compute_S_producer_group,
            consumer_group=mma_compute_S_consumer_group,
        )

    def make_and_init_mma_compute_dP_pipeline(self, mma_compute_dP_mbar_ptr):
        mma_compute_dP_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dP_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dP_mbar_ptr,
            num_stages=self.mma_compute_dP_stage,
            producer_group=mma_compute_dP_producer_group,
            consumer_group=mma_compute_dP_consumer_group,
        )

    def make_and_init_mma_reduce_dQ_pipeline(self, mma_reduce_dQ_mbar_ptr):
        mma_reduce_dQ_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_reduce_dQ_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_reduce_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_reduce_dQ_mbar_ptr,
            num_stages=self.mma_reduce_dQ_stage,
            producer_group=mma_reduce_dQ_producer_group,
            consumer_group=mma_reduce_dQ_consumer_group,
        )

    def make_and_init_compute_mma_P_pipeline(self, compute_mma_P_mbar_ptr):
        compute_mma_P_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        compute_mma_P_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_P_mbar_ptr,
            num_stages=self.compute_mma_P_stage,
            producer_group=compute_mma_P_producer_group,
            consumer_group=compute_mma_P_consumer_group,
        )

    def make_and_init_compute_mma_dS_pipeline(self, compute_mma_dS_mbar_ptr):
        compute_mma_dS_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        compute_mma_dS_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )

        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_dS_mbar_ptr,
            num_stages=self.compute_mma_dS_stage,
            producer_group=compute_mma_dS_producer_group,
            consumer_group=compute_mma_dS_consumer_group,
        )

    def make_and_init_mma_compute_dKdV_pipeline(self, mma_compute_dKdV_mbar_ptr):
        mma_compute_dKdV_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        mma_compute_dKdV_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dKdV_mbar_ptr,
            num_stages=self.mma_compute_dKdV_stage,
            producer_group=mma_compute_dKdV_producer_group,
            consumer_group=mma_compute_dKdV_consumer_group,
        )

    def make_and_init_reduce_tma_store_pipeline(self):
        reduce_tma_store_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_reduce_warps * self.threads_per_warp,
        )
        return pipeline.PipelineTmaStore.create(
            num_stages=self.reduce_tma_store_stage,
            producer_group=reduce_tma_store_producer_group,
        )
