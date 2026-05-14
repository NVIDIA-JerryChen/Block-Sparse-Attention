# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# BSA: Extracted SM100 Forward Kernel (qstage=1, non-causal, non-mask, non-varlen)

import os
import math
from functools import lru_cache
from typing import Optional, Tuple

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack
from utils.cache_utils import get_jit_cache
from utils.testing import is_fake_mode

from csrc.fwd.sm100_blk128 import utils
from utils import fa_logging
from csrc.fwd.sm100_blk128.cute_dsl_utils import to_cute_tensor
from csrc.fwd.sm100_blk128.flash_fwd_sm100 import FlashAttentionForwardSm100
from csrc.bwd.sm100_blk64.flash_bwd_sm100 import (
    BlockSparseAttnBackward,
)

try:
    import bsa_fwd_blk64_ext  # triggers TORCH_LIBRARY registration of bsa_blk64.fwd
except ImportError:
    pass  # blk64 wheel not built — bsa_attn_fwd_blk64 will fail at call time

BSA_BWD_SPARSE_BLOCK_SIZE = 64
BSA_BWD_HEAD_DIM = 128
BSA_BWD_CSR_SCHEDULE_POLICY = "csr_adaptive_qrange_v1"
BSA_BWD_DKV_EPILOGUE_ENV = "BSA_BWD_DKV_EPILOGUE"

_bsa_clc_enabled: bool = os.environ.get("BSA_CLC", "1") == "1"

def _get_use_clc_scheduler() -> bool:
    return _bsa_clc_enabled


def _get_dkv_epilogue_mode() -> Tuple[bool, bool, str]:
    value = os.environ.get(BSA_BWD_DKV_EPILOGUE_ENV, "tma").strip().lower()
    if value in {"tma", "tma_reduce", "1", "true"}:
        return True, True, "tma"
    if value in {"atomic", "atomic_add", "atomic_stage", "0", "false"}:
        return False, True, "atomic_stage"
    if value in {"atomic_linear", "linear_atomic", "qbucket_atomic"}:
        return False, False, "atomic_linear"
    raise ValueError(
        f"unsupported {BSA_BWD_DKV_EPILOGUE_ENV}={value!r}; "
        "expected tma, atomic_stage, or atomic_linear"
    )


def _parse_arch_str(arch_str):
    """Parse arch string (e.g. 'sm_80', 'sm_90a', '80', '100') to int."""
    import re
    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch():
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def bsa_attn_fwd_blk64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sizes: torch.Tensor,
    q2k_block_nums: torch.Tensor,
    softmax_scale: Optional[float] = None,
    layout: str = "bhsd",
    use_clc: bool = False,
):
    """BSA forward attention (blk64 backend, bf16 only, D=128).

    The blk64 kernel consumes BHSD tensors directly. The default layout is
    therefore "bhsd" for the customer zero-copy path. BSHD callers are still
    supported by an explicit layout="bshd" conversion at the wrapper boundary.

    Args:
        q, k, v: (B, H, S, D) if layout="bhsd" or (B, S, H, D) if layout="bshd"
        q2k_block_index: (B, H, Q_tiles, max_topk) int32
        block_sizes: (num_kv_blocks,) int32
        q2k_block_nums: (B, H, Q_tiles) int32
        softmax_scale: default 1/sqrt(D)
        layout: "bhsd" (default, zero-copy) or "bshd" (converted to BHSD).
        use_clc: enable the SM100 CLC persistent scheduler path. Default False
            uses the SingleTileScheduler fallback (one tile per CTA).
    """
    assert q.dtype == torch.bfloat16, "blk64 requires bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4

    if layout == "bshd":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    else:
        assert layout == "bhsd", f"layout must be 'bhsd' or 'bshd', got {layout!r}"

    assert q.size(3) == 128, "blk64 requires D=128"
    assert q.stride(3) == 1 and k.stride(3) == 1 and v.stride(3) == 1, \
        "head_dim (axis 3) must be stride-1 / innermost"

    seqlen_q = q.size(2)

    if softmax_scale is None:
        softmax_scale = q.size(3) ** -0.5

    # No F.pad: seqlen_q rounding is handled by the kernel's row bounds checks
    # and output TMA descriptor; seqlen_k masking is controlled by block_sizes.
    max_topk = q2k_block_index.shape[-1]
    out, lse = torch.ops.bsa_blk64.fwd(
        q, k, v, q2k_block_index, max_topk, block_sizes, softmax_scale, q2k_block_nums, use_clc)

    assert out.size(2) == seqlen_q and lse.size(2) == seqlen_q
    if layout == "bshd":
        out = out.transpose(1, 2).contiguous()
    return out, lse


def bsa_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    max_topk: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    allow_empty_block_nums: bool = True,
    softmax_scale: Optional[float] = None,
    pack_gqa: Optional[bool] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for BSA block-sparse attention (SM100 only, qstage=1, non-causal, non-varlen).

    Args:
        q: Query tensor (batch, seqlen_q, num_heads, head_dim)
        k: Key tensor (batch, seqlen_k, num_heads_kv, head_dim)
        v: Value tensor (batch, seqlen_k, num_heads_kv, head_dim_v)
        q2k_block_index: Block sparse index tensor (batch, num_heads, num_q_blocks, max_topk), int32.
            Fixed path: all max_topk entries are valid. Variable path: q2k_block_nums gives the
            valid prefix length per row, and max_topk is the storage capacity / maximum topK.
        max_topk: Fixed KV-block count per Q block, or q2k_block_index capacity when
            q2k_block_nums is provided. Fixed blk128 path requires even max_topk >= 2.
        block_sizes: Actual token count per KV block (num_kv_blocks,), int32. Used for masking padding positions.
            When None, block_size masking is skipped (assumes all blocks are full).
        q2k_block_nums: Per-(batch, head, q_block) number of KV blocks to attend to,
            (batch, num_heads, num_q_blocks) int32, each value >= 0.
            When None, uses fixed max_topk for all Q blocks.
        allow_empty_block_nums: When True (default), q2k_block_nums may contain 0 (empty tiles
            produce O=0, LSE=-inf). When False, all values must be >= 1, enabling compile-time
            elimination of empty-tile branches for better performance (~2-3%).
        softmax_scale: Softmax scale (default: 1/sqrt(head_dim))
        pack_gqa: Whether to pack GQA heads
        return_lse: Whether to return log-sum-exp
        out: Pre-allocated output tensor
        lse: Pre-allocated LSE tensor
    """
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    batch_size, seqlen_q, num_head, head_dim = q.shape
    seqlen_k = k.shape[1]
    num_head_kv = k.shape[2]
    head_dim_v = v.shape[-1]

    assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
    assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"

    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v)), "inputs must be on CUDA device"

    arch = _get_device_arch()
    assert arch // 10 in [10, 11], "BSA only supports SM100/SM110"
    assert num_head % num_head_kv == 0

    # Block-sparse parameter validation
    assert q2k_block_index.dtype == torch.int32, "q2k_block_index must be int32"
    has_block_sizes = block_sizes is not None
    if has_block_sizes:
        assert block_sizes.dtype == torch.int32, "block_sizes must be int32"
    if q2k_block_nums is not None:
        q2k_block_nums = maybe_contiguous(q2k_block_nums)
        assert q2k_block_nums.dtype == torch.int32, "q2k_block_nums must be int32"
        assert q2k_block_nums.ndim == 3, (
            f"q2k_block_nums must be 3D (batch, num_heads, num_q_blocks), got {q2k_block_nums.ndim}D"
        )
        assert 0 <= max_topk <= q2k_block_index.shape[-1], (
            f"max_topk={max_topk} must be in [0, {q2k_block_index.shape[-1]}]"
        )
    else:
        assert max_topk >= 2 and max_topk % 2 == 0, (
            f"max_topk={max_topk} must be even and >= 2"
        )
        assert q2k_block_index.shape[-1] >= max_topk, (
            f"q2k_block_index last dim ({q2k_block_index.shape[-1]}) must be >= max_topk ({max_topk})"
        )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    qhead_per_kvhead = num_head // num_head_kv

    out_torch_dtype = q.dtype
    device = q.device
    tile_n = 128

    seqlen_q_packgqa = seqlen_q * qhead_per_kvhead

    # Block-sparse attention: 2CTA disabled (m=64+2CTA correction rewrite in progress)
    use_2cta_instrs = False
    tile_m = 128

    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    if pack_gqa and (tile_m % qhead_per_kvhead != 0):
        pack_gqa = False

    lse_shape = (batch_size, num_head, seqlen_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(
            batch_size, seqlen_q, num_head, head_dim_v, dtype=out_torch_dtype, device=device
        )

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )

    dtype = torch2cute_dtype_map[q.dtype]

    use_clc_scheduler = _get_use_clc_scheduler()

    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )

    has_variable_block_nums = q2k_block_nums is not None

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        lse is None,
        tile_m,
        tile_n,
        pack_gqa,
        arch,
        use_2cta_instrs,
        use_clc_scheduler,
        fa_logging.get_fa_log_level(),
        has_variable_block_nums,
        allow_empty_block_nums and has_variable_block_nums,
        has_block_sizes,
    )

    if compile_key not in bsa_attn_fwd.compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out)
        ]
        lse_tensor = to_cute_tensor(lse, assumed_align=4) if lse is not None else None
        block_index_tensor = to_cute_tensor(q2k_block_index)
        block_sizes_tensor = to_cute_tensor(block_sizes) if has_block_sizes else None
        block_nums_tensor = to_cute_tensor(q2k_block_nums) if has_variable_block_nums else None

        fa_fwd = FlashAttentionForwardSm100(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            pack_gqa=pack_gqa,
            m_block_size=tile_m,
            n_block_size=tile_n,
            is_persistent=True,
            use_2cta_instrs=use_2cta_instrs,
            use_clc_scheduler=use_clc_scheduler,
            allow_empty_block_nums=allow_empty_block_nums and has_variable_block_nums,
            has_block_sizes=has_block_sizes,
        )

        bsa_attn_fwd.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            block_index_tensor,
            block_sizes_tensor,
            max_topk,
            block_nums_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        with torch.cuda.nvtx.range("bsa_attn_fwd_kernel"):
            bsa_attn_fwd.compile_cache[compile_key](
                q.detach(),
                k.detach(),
                v.detach(),
                out.detach(),
                lse,
                softmax_scale,
                q2k_block_index.detach(),
                block_sizes.detach() if has_block_sizes else None,
                max_topk,
                q2k_block_nums.detach() if has_variable_block_nums else None,
                current_stream,
            )

    return out, lse


bsa_attn_fwd.compile_cache = get_jit_cache("bsa_fwd")



def convert_q2k_to_k2q_csr(
    q2k_block_index: torch.Tensor,
    max_topk: int,
    num_kv_blocks: int,
    q2k_block_nums: Optional[torch.Tensor] = None,
    return_schedule: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Build total-packed CSR k2q metadata plus adaptive qrange schedule."""
    from csrc.common.block_sparse_csr import convert_q2k_to_k2q_csr as _build_csr

    return _build_csr(
        q2k_block_index,
        max_topk,
        num_kv_blocks,
        q2k_block_nums=q2k_block_nums,
        return_schedule=return_schedule,
    )


def bsa_attn_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q2k_block_index: torch.Tensor,
    max_topk: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    k2q_row_ptr: Optional[torch.Tensor] = None,
    k2q_q_indices: Optional[torch.Tensor] = None,
    k2q_schedule_metadata: Optional[torch.Tensor] = None,
    k2q_schedule_work_counts: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for BSA block-sparse attention using CSR scheduled metadata.

    If CSR schedule metadata is not provided, it is built from q2k in the
    production format: total-packed CSR plus adaptive qrange-split schedule.
    """
    q, k, v, out, dout = [maybe_contiguous(t) for t in (q, k, v, out, dout)]
    lse = maybe_contiguous(lse)

    assert q.dtype == torch.bfloat16, "bwd only supports bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
    assert lse.dtype == torch.float32
    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v, out, dout, lse))

    batch_size, num_heads, seqlen_q, head_dim = q.shape
    num_heads_kv, seqlen_k = k.shape[1], k.shape[2]
    assert head_dim == BSA_BWD_HEAD_DIM, (
        f"bwd only supports head_dim={BSA_BWD_HEAD_DIM}, got {head_dim}"
    )
    assert num_heads == num_heads_kv, "bwd does not support GQA/MQA"
    assert k.shape == v.shape == (batch_size, num_heads, seqlen_k, head_dim)
    assert out.shape == (batch_size, num_heads, seqlen_q, head_dim)
    assert dout.shape == out.shape
    assert lse.shape == (batch_size, num_heads, seqlen_q)

    arch = _get_device_arch()
    assert arch // 10 in [10, 11], "BSA bwd only supports SM100/SM110"

    sparse_block_size = BSA_BWD_SPARSE_BLOCK_SIZE
    num_q_blocks = (seqlen_q + sparse_block_size - 1) // sparse_block_size
    num_kv_blocks = (seqlen_k + sparse_block_size - 1) // sparse_block_size

    assert q2k_block_index.dtype == torch.int32
    assert q2k_block_index.shape[:3] == (batch_size, num_heads, num_q_blocks), (
        f"q2k_block_index has shape {tuple(q2k_block_index.shape)}, expected "
        f"(b={batch_size}, h={num_heads}, num_q_blocks={num_q_blocks}, max_topk)"
    )
    if q2k_block_nums is not None:
        q2k_block_nums = maybe_contiguous(q2k_block_nums)
        assert q2k_block_nums.dtype == torch.int32
        assert q2k_block_nums.shape == (batch_size, num_heads, num_q_blocks)
        assert 0 <= max_topk <= q2k_block_index.shape[-1], (
            f"max_topk={max_topk} must be in [0, {q2k_block_index.shape[-1]}]"
        )

    prebuilt = (
        k2q_row_ptr,
        k2q_q_indices,
        k2q_schedule_metadata,
        k2q_schedule_work_counts,
    )
    if any(t is not None for t in prebuilt) and not all(t is not None for t in prebuilt):
        raise ValueError(
            "k2q_row_ptr, k2q_q_indices, k2q_schedule_metadata, and "
            "k2q_schedule_work_counts must be provided together"
        )
    if k2q_row_ptr is None:
        (
            k2q_row_ptr,
            k2q_q_indices,
            k2q_schedule_metadata,
            k2q_schedule_work_counts,
        ) = convert_q2k_to_k2q_csr(
            q2k_block_index,
            max_topk,
            num_kv_blocks,
            q2k_block_nums=q2k_block_nums,
            return_schedule=True,
        )

    assert k2q_row_ptr is not None
    assert k2q_q_indices is not None
    assert k2q_schedule_metadata is not None
    assert k2q_schedule_work_counts is not None
    assert k2q_row_ptr.dtype == torch.int32
    assert k2q_row_ptr.shape == (batch_size, num_heads, num_kv_blocks + 1)
    assert k2q_row_ptr.device == q.device
    assert k2q_q_indices.dtype == torch.int32 and k2q_q_indices.ndim == 1
    assert k2q_q_indices.device == q.device
    assert k2q_schedule_metadata.dtype == torch.int32
    assert k2q_schedule_metadata.shape[:2] == (batch_size, num_heads)
    assert k2q_schedule_metadata.shape[3] == 4
    assert k2q_schedule_metadata.device == q.device
    assert k2q_schedule_work_counts.dtype == torch.int32
    assert k2q_schedule_work_counts.shape == (batch_size, num_heads)
    assert k2q_schedule_work_counts.device == q.device

    has_block_sizes = block_sizes is not None
    if not has_block_sizes:
        variable_block_sizes = torch.empty((1, 1), dtype=torch.int32, device=q.device)
    else:
        assert block_sizes.dtype == torch.int32
        if block_sizes.ndim == 1:
            assert block_sizes.shape == (num_kv_blocks,)
            variable_block_sizes = block_sizes.unsqueeze(0).expand(batch_size, -1).contiguous()
        else:
            assert block_sizes.shape == (batch_size, num_kv_blocks)
            variable_block_sizes = block_sizes.contiguous()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)

    workspace_shape = BlockSparseAttnBackward._get_workspace_size(
        q=seqlen_q,
        k=seqlen_k,
        d=head_dim,
        h=num_heads,
        b=batch_size,
        acc_dtype=Float32,
    )
    if workspace is None:
        workspace = torch.zeros(workspace_shape, dtype=torch.uint8, device=q.device)
    else:
        assert workspace.dtype == torch.uint8
        assert workspace.device == q.device
        assert tuple(workspace.shape) == tuple(workspace_shape)
        workspace.zero_()

    problem_shape = (seqlen_q, seqlen_k, head_dim, (num_heads, batch_size))
    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )

    full_kv_blocks = seqlen_k % sparse_block_size == 0 and head_dim == BSA_BWD_HEAD_DIM
    use_dkv_tma_reduce, use_dkv_stage_layout, dkv_epilogue_mode = _get_dkv_epilogue_mode()
    compile_key = (
        q.dtype,
        head_dim,
        num_heads,
        sparse_block_size,
        arch,
        has_block_sizes,
        full_kv_blocks,
        dkv_epilogue_mode,
        BSA_BWD_CSR_SCHEDULE_POLICY,
        fa_logging.get_fa_log_level(),
    )

    def convert_to_cute_tensor(t: torch.Tensor, enable_tvm_ffi: bool = True) -> cute.Tensor:
        return (
            from_dlpack(t.detach(), assumed_align=16, enable_tvm_ffi=enable_tvm_ffi)
            .mark_layout_dynamic()
            .mark_compact_shape_dynamic(
                mode=3, stride_order=t.dim_order(), divisibility=128
            )
        )

    if compile_key not in bsa_attn_bwd.compile_cache:
        dO_t = convert_to_cute_tensor(dout)
        O_t = convert_to_cute_tensor(out)
        Q_t = convert_to_cute_tensor(q)
        K_t = convert_to_cute_tensor(k)
        V_t = convert_to_cute_tensor(v)
        dQ_t = convert_to_cute_tensor(dq)
        dK_t = convert_to_cute_tensor(dk)
        dV_t = convert_to_cute_tensor(dv)
        LSE_t = to_cute_tensor(lse, leading_dim=2)
        schedule_t = to_cute_tensor(k2q_schedule_metadata, leading_dim=3)
        q_indices_t = to_cute_tensor(k2q_q_indices, leading_dim=0)
        var_bs_t = to_cute_tensor(variable_block_sizes, leading_dim=1)
        ws_t = to_cute_tensor(workspace, fully_dynamic=True)

        bwd_kernel = BlockSparseAttnBackward(
            sparse_block_size=sparse_block_size,
            has_block_sizes=has_block_sizes,
            full_kv_blocks=full_kv_blocks,
            use_dkv_tma_reduce=use_dkv_tma_reduce,
            use_dkv_stage_layout=use_dkv_stage_layout,
        )

        bsa_attn_bwd.compile_cache[compile_key] = cute.compile(
            bwd_kernel,
            problem_shape,
            dO_t,
            O_t,
            Q_t,
            K_t,
            V_t,
            LSE_t,
            dQ_t,
            dK_t,
            dV_t,
            schedule_t,
            q_indices_t,
            var_bs_t,
            ws_t,
            softmax_scale,
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        with torch.cuda.nvtx.range("bsa_attn_bwd_kernel"):
            bsa_attn_bwd.compile_cache[compile_key](
                problem_shape,
                dout,
                out,
                q,
                k,
                v,
                lse,
                dq,
                dk,
                dv,
                k2q_schedule_metadata,
                k2q_q_indices,
                variable_block_sizes,
                workspace,
                softmax_scale,
                current_stream,
            )

    return dq, dk, dv


bsa_attn_bwd.compile_cache = get_jit_cache("bsa_bwd_csr_scheduled")
