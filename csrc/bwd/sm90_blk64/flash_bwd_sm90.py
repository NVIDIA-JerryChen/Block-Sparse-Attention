import math
import os
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack

from csrc.bwd.sm90_blk64.block_sparsity import (
    BlockSparseTensorsTorch,
    normalize_block_sparse_config_bwd,
    to_cute_block_sparse_tensors,
)
from csrc.bwd.sm90_blk64.flash_bwd_sm90_helpers import (
    _bwd_postprocess_convert,
    _bwd_preprocess,
)
from csrc.bwd.sm90_blk64.flash_bwd_sm90_mainloop import FlashAttentionBackwardSm90

from utils.cache_utils import get_jit_cache
from utils.testing import is_fake_mode


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _convert_to_cute_tensor(
    t: torch.Tensor, assumed_align: int = 16, enable_tvm_ffi: bool = True
) -> cute.Tensor:
    return (
        from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
        .mark_layout_dynamic(leading_dim=t.ndim - 1)
    )


class BlockSparseAttnBackwardSm90:
    """Hopper backward baseline for blk64 BSA development.

    The launch/preprocess/postprocess path stays close to the FA SM90 style, but
    all runtime imports and kernel modules are local to this repo. Dense and
    block-sparse paths share the same local 64x64 mainloop.
    """

    tile_m = 64
    tile_n = 64
    arch = 90

    @staticmethod
    def _get_workspace_size(
        q: int, d: int, h: int, b: int, acc_dtype=cutlass.Float32
    ) -> Tuple[int, int, int]:
        head_dim_rounded = (d + 31) // 32 * 32
        q_rounded = (
            (q + BlockSparseAttnBackwardSm90.tile_m - 1) // BlockSparseAttnBackwardSm90.tile_m
        )
        q_rounded *= BlockSparseAttnBackwardSm90.tile_m
        return (b, h, q_rounded * head_dim_rounded)

    @staticmethod
    def _get_stats_size(q: int, h: int, b: int) -> Tuple[int, int, int]:
        q_rounded = (
            (q + BlockSparseAttnBackwardSm90.tile_m - 1) // BlockSparseAttnBackwardSm90.tile_m
        )
        q_rounded *= BlockSparseAttnBackwardSm90.tile_m
        return (b, h, q_rounded)

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        num_threads: int = 384,
    ):
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        self.dtype = dtype
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.num_threads = num_threads
        self.num_mma_wg = num_threads // 128 - 1
        assert self.num_mma_wg == 2, "SM90 dense blk64 baseline assumes WG0 + 2 MMA WGs"

    def compile_mainloop(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mBlockSizes: Optional[cute.Tensor],
        stream: cuda.CUstream,
        blocksparse_tensors=None,
        skip_score_mask: bool = False,
        sdp_swap_ab: bool = False,
        kv_in_regs: bool = False,
        dkv_rs_wg1: bool = False,
        dkv_rs_split: bool = False,
        dqaccum_stage: int = 1,
        pds_stage: int = 1,
        qdo_stage: int = 2,
    ):
        wg_specialized_pipeline = blocksparse_tensors is None
        assert not kv_in_regs or wg_specialized_pipeline, "KV-in-regs is dense-only"
        assert not kv_in_regs or sdp_swap_ab, "KV-in-regs requires SdP_swapAB"
        assert not dkv_rs_wg1 or wg_specialized_pipeline, "WG1 dKV-RS is dense-only"
        assert not dkv_rs_wg1 or sdp_swap_ab, "WG1 dKV-RS requires SdP_swapAB"
        assert not dkv_rs_split or wg_specialized_pipeline, "Split dKV-RS is dense-only"
        assert not dkv_rs_split or sdp_swap_ab, "Split dKV-RS requires SdP_swapAB"
        assert not (dkv_rs_wg1 and dkv_rs_split), "dKV-RS experiments are mutually exclusive"
        assert pds_stage in [1, 2], "PdS stage must be 1 or 2"
        assert qdo_stage in [1, 2, 3], "Q/dO stage must be 1, 2, or 3"
        assert pds_stage == 1 or pds_stage == qdo_stage, "PdS stage must be 1 or match Q/dO stage"
        bwd = FlashAttentionBackwardSm90(
            self.dtype,
            self.head_dim,
            self.head_dim_v,
            qhead_per_kvhead=1,
            is_causal=False,
            is_local=False,
            deterministic=False,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            Q_stage=qdo_stage,
            dO_stage=qdo_stage,
            PdS_stage=pds_stage,
            SdP_swapAB=sdp_swap_ab,
            dKV_swapAB=not wg_specialized_pipeline,
            dQ_swapAB=False,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=2,
            AtomLayoutMdQ=1,
            num_threads=self.num_threads,
            V_in_regs=kv_in_regs,
            dQ_single_wg=wg_specialized_pipeline,
            wg_specialized_pipeline=wg_specialized_pipeline,
            skip_score_mask=skip_score_mask,
            dKV_rs_wg1=dkv_rs_wg1,
            dKV_rs_split=dkv_rs_split,
            dQaccum_stage=dqaccum_stage,
        )
        compile_options = "--enable-tvm-ffi"
        dump_dir = os.environ.get("BSA_SM90_BWD_DUMP_DIR")
        if dump_dir:
            compile_options += f" --dump-dir={dump_dir} --keep-cubin --keep-ptx"
        return cute.compile(
            bwd,
            mQ,
            mK,
            mV,
            mdO,
            mLSElog2,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            softmax_scale,
            mBlockSizes,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            blocksparse_tensors,
            stream,
            options=compile_options,
        )


def bsa_attn_bwd_sm90(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    block_sizes: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SM90 blk64 backward baseline.

    Public BSA tensors are BHSD. The local CuTe kernels consume BSHD, so this
    wrapper transposes at the boundary and returns BHSD gradients.
    """
    q, k, v, out, dout = [_maybe_contiguous(t) for t in (q, k, v, out, dout)]
    lse = _maybe_contiguous(lse)

    assert q.dtype in [torch.float16, torch.bfloat16]
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
    assert lse.dtype == torch.float32
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4

    batch_size, num_heads, seqlen_q, head_dim = q.shape
    num_heads_kv, seqlen_k, head_dim_k = k.shape[1], k.shape[2], k.shape[3]
    assert head_dim == 128 and head_dim_k == 128, "SM90 blk64 dense bwd requires D=128"
    assert num_heads == num_heads_kv, "SM90 blk64 dense bwd does not support GQA/MQA yet"
    assert v.shape == k.shape
    assert out.shape == q.shape and dout.shape == q.shape
    assert lse.shape == (batch_size, num_heads, seqlen_q)
    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v, out, dout, lse))
        if block_sizes is not None:
            assert block_sizes.is_cuda and block_sizes.dtype == torch.int32

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if block_sizes is not None:
        if block_sizes.ndim == 1:
            assert block_sizes.shape == ((seqlen_k + 63) // 64,)
            block_sizes = block_sizes.unsqueeze(0).expand(batch_size, -1).contiguous()
        else:
            assert block_sizes.shape == (batch_size, (seqlen_k + 63) // 64)
            block_sizes = block_sizes.contiguous()
        if (
            block_sparse_tensors is None
            and seqlen_k % BlockSparseAttnBackwardSm90.tile_n == 0
            and not is_fake_mode()
            and torch.all(block_sizes == BlockSparseAttnBackwardSm90.tile_n).item()
        ):
            block_sizes = None

    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    out_bshd = out.transpose(1, 2).contiguous()
    dout_bshd = dout.transpose(1, 2).contiguous()

    dq_bshd = torch.empty_like(q_bshd)
    dk_bshd = torch.empty_like(k_bshd)
    dv_bshd = torch.empty_like(v_bshd)

    dtype = torch2cute_dtype_map[q.dtype]
    kernel = BlockSparseAttnBackwardSm90(dtype, head_dim, head_dim)

    normalized_block_sparse_tensors = None
    block_sparse_broadcast_pattern = None
    if block_sparse_tensors is not None:
        normalized_block_sparse_tensors, block_sparse_broadcast_pattern = (
            normalize_block_sparse_config_bwd(
                block_sparse_tensors,
                batch_size=batch_size,
                num_head=num_heads,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                block_size=(kernel.tile_m, kernel.tile_n),
                subtile_factor=1,
            )
        )

    stats_shape = kernel._get_stats_size(seqlen_q, num_heads, batch_size)
    dpsum = torch.empty(stats_shape, dtype=torch.float32, device=q.device)
    lse_log2 = torch.empty(stats_shape, dtype=torch.float32, device=q.device)
    dq_accum = torch.empty(
        kernel._get_workspace_size(seqlen_q, head_dim, num_heads, batch_size),
        dtype=torch.float32,
        device=q.device,
    )

    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )
    sdp_swap_ab = os.environ.get("BSA_SM90_BWD_SDP_SWAPAB", "0") == "1"
    kv_in_regs = (
        normalized_block_sparse_tensors is None
        and os.environ.get("BSA_SM90_BWD_KV_REGS", "1") != "0"
    )
    dkv_rs_wg1 = (
        normalized_block_sparse_tensors is None
        and os.environ.get("BSA_SM90_BWD_DKV_RS_WG1", "0") == "1"
    )
    dkv_rs_split_env = os.environ.get("BSA_SM90_BWD_DKV_RS_SPLIT", "1")
    dkv_rs_split = (
        normalized_block_sparse_tensors is None
        and not dkv_rs_wg1
        and dkv_rs_split_env != "0"
    )
    assert not (dkv_rs_wg1 and dkv_rs_split), "dKV-RS experiments are mutually exclusive"
    if dkv_rs_wg1:
        sdp_swap_ab = True
    if dkv_rs_split:
        sdp_swap_ab = True
    if kv_in_regs:
        sdp_swap_ab = True
    dqaccum_stage = int(os.environ.get("BSA_SM90_BWD_DQACCUM_STAGE", "1"))
    assert dqaccum_stage in [1, 2, 3], "BSA_SM90_BWD_DQACCUM_STAGE must be 1, 2, or 3"
    pds_stage = int(os.environ.get("BSA_SM90_BWD_PDS_STAGE", "1"))
    assert pds_stage in [1, 2], "BSA_SM90_BWD_PDS_STAGE must be 1 or 2"
    qdo_stage_default = "3" if normalized_block_sparse_tensors is None else "2"
    qdo_stage = int(os.environ.get("BSA_SM90_BWD_QDO_STAGE", qdo_stage_default))
    assert qdo_stage in [1, 2, 3], "BSA_SM90_BWD_QDO_STAGE must be 1, 2, or 3"
    if pds_stage > qdo_stage:
        pds_stage = qdo_stage
    if pds_stage != 1 and pds_stage != qdo_stage:
        pds_stage = 1

    main_key = (
        q.dtype,
        head_dim,
        seqlen_q <= kernel.tile_m,
        seqlen_k <= kernel.tile_n,
        normalized_block_sparse_tensors is not None,
        block_sparse_broadcast_pattern,
        block_sizes is not None,
        block_sizes is None and block_sparse_tensors is None and seqlen_k % kernel.tile_n == 0,
        sdp_swap_ab,
        kv_in_regs,
        dkv_rs_wg1,
        dkv_rs_split,
        dqaccum_stage,
        pds_stage,
        qdo_stage,
    )

    _bwd_preprocess(
        out_bshd,
        dout_bshd,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        None,
        None,
        None,
        dtype,
        head_dim,
        head_dim,
        kernel.tile_m,
    )
    if main_key not in bsa_attn_bwd_sm90.mainloop_cache:
        sparse_tensors_compile = (
            to_cute_block_sparse_tensors(normalized_block_sparse_tensors)
            if normalized_block_sparse_tensors is not None
            else None
        )
        bsa_attn_bwd_sm90.mainloop_cache[main_key] = kernel.compile_mainloop(
            _convert_to_cute_tensor(q_bshd),
            _convert_to_cute_tensor(k_bshd),
            _convert_to_cute_tensor(v_bshd),
            _convert_to_cute_tensor(dout_bshd),
            _convert_to_cute_tensor(lse_log2),
            _convert_to_cute_tensor(dpsum),
            _convert_to_cute_tensor(dq_accum),
            _convert_to_cute_tensor(dk_bshd),
            _convert_to_cute_tensor(dv_bshd),
            Float32(softmax_scale),
            _convert_to_cute_tensor(block_sizes, assumed_align=4) if block_sizes is not None else None,
            current_stream,
            blocksparse_tensors=sparse_tensors_compile,
            skip_score_mask=(
                block_sizes is None
                and normalized_block_sparse_tensors is None
                and seqlen_k % kernel.tile_n == 0
            ),
            sdp_swap_ab=sdp_swap_ab,
            kv_in_regs=kv_in_regs,
            dkv_rs_wg1=dkv_rs_wg1,
            dkv_rs_split=dkv_rs_split,
            dqaccum_stage=dqaccum_stage,
            pds_stage=pds_stage,
            qdo_stage=qdo_stage,
        )

    if not is_fake_mode():
        sparse_tensors_runtime = (
            normalized_block_sparse_tensors[:4]
            if normalized_block_sparse_tensors is not None
            else None
        )
        bsa_attn_bwd_sm90.mainloop_cache[main_key](
            q_bshd,
            k_bshd,
            v_bshd,
            dout_bshd,
            lse_log2,
            dpsum,
            dq_accum,
            dk_bshd,
            dv_bshd,
            softmax_scale,
            block_sizes,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            sparse_tensors_runtime,
            current_stream,
        )
    _bwd_postprocess_convert(
        dq_accum,
        dq_bshd,
        softmax_scale,
        None,
        None,
        kernel.arch,
        dtype,
        head_dim,
        kernel.tile_m,
        kernel.num_mma_wg * 128,
        1,
        False,
    )

    dq_result = dq_bshd.transpose(1, 2).contiguous()
    dk_result = dk_bshd.transpose(1, 2).contiguous()
    dv_result = dv_bshd.transpose(1, 2).contiguous()
    if dq is not None:
        dq.copy_(dq_result)
        dq_result = dq
    if dk is not None:
        dk.copy_(dk_result)
        dk_result = dk
    if dv is not None:
        dv.copy_(dv_result)
        dv_result = dv
    return dq_result, dk_result, dv_result


bsa_attn_bwd_sm90.mainloop_cache = get_jit_cache("bsa_bwd_sm90")
