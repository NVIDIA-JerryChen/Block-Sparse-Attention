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
from cutlass import Int32, Float32
from cutlass.cute.runtime import from_dlpack
import triton
import triton.language as tl
from utils.cache_utils import get_jit_cache
from utils.testing import is_fake_mode

from csrc.fwd.sm100_blk128 import utils
from utils import fa_logging
from csrc.fwd.sm100_blk128.cute_dsl_utils import to_cute_tensor
from csrc.fwd.sm100_blk128.flash_fwd_sm100 import FlashAttentionForwardSm100
from csrc.bwd.sm100_blk64.flash_bwd_sm100 import BlockSparseAttnBackward
from csrc.bwd.sm100_blk64.flash_bwd_sm100_qbucket import (
    BlockSparseAttnBackwardQRangeBucketed,
)

BSA_BWD_SPARSE_BLOCK_SIZE = 64
BSA_BWD_HEAD_DIM = 128

_bsa_clc_enabled: bool = os.environ.get("BSA_CLC", "1") == "1"

def _get_use_clc_scheduler() -> bool:
    return _bsa_clc_enabled


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
    layout: str = "bshd",
    use_clc: bool = False,
):
    """BSA forward attention (blk64 backend, bf16 only, D=128).

    The underlying C++ kernel expects BHSD tensors (zero-copy path).
    When layout="bshd" the user's BSHD inputs are permuted to BHSD here;
    when layout="bhsd" the inputs are forwarded without any data movement.

    Args:
        q, k, v: (B, S, H, D) if layout="bshd" or (B, H, S, D) if layout="bhsd"
        q2k_block_index: (B, H, Q_tiles, max_kv) int32
        block_sizes: (num_kv_blocks,) int32
        q2k_block_nums: (B, H, Q_tiles) int32
        softmax_scale: default 1/sqrt(D)
        layout: "bshd" or "bhsd"
        use_clc: enable the SM100 CLC persistent scheduler path. Default False
            uses the SingleTileScheduler fallback (one tile per CTA).
    """
    assert q.dtype == torch.bfloat16, "blk64 requires bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4

    # Normalize to BHSD for the kernel. BSHD inputs are permuted to BHSD here.
    if layout == "bshd":
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

    # q/k/v are now BHSD: (B, H, S, D).
    assert q.size(3) == 128, "blk64 requires D=128"
    seqlen_q = q.size(2)
    seqlen_k = k.size(2)

    if softmax_scale is None:
        softmax_scale = q.size(3) ** -0.5

    # Pad seqlen (dim 2) to multiples of 64 if needed. For 4D BHSD tensors the
    # pad tuple counts from the last dim: (0,0, 0,pad) -> pad only dim 2 (seq).
    if seqlen_q % 64 != 0:
        pad_q = 64 - seqlen_q % 64
        q = torch.nn.functional.pad(q, (0, 0, 0, pad_q))
    if seqlen_k % 64 != 0:
        pad_k = 64 - seqlen_k % 64
        k = torch.nn.functional.pad(k, (0, 0, 0, pad_k))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad_k))

    import bsa_fwd_blk64_ext  # triggers TORCH_LIBRARY registration
    out, lse = torch.ops.bsa_blk64.fwd(
        q, k, v, q2k_block_index, 0, block_sizes, softmax_scale, q2k_block_nums, use_clc)

    # Kernel returns out as BHSD (B, H, S_q_rounded, D). Trim seqlen (dim 2).
    if out.size(2) != seqlen_q:
        out = out[:, :, :seqlen_q]
    if lse.size(2) != seqlen_q:
        lse = lse[:, :, :seqlen_q]

    if layout == "bshd":
        out = out.permute(0, 2, 1, 3).contiguous()
    return out, lse


def bsa_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
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
        q2k_block_index: Block sparse index tensor (batch, num_heads, num_q_blocks, max_kv_blocks), int32.
            For each (batch, head, q_block), the first block_sparse_num entries are the KV block indices to attend to.
        block_sparse_num: Number of KV blocks each Q block attends to. Must be even and >= 2.
            Ignored when q2k_block_nums is provided.
        block_sizes: Actual token count per KV block (num_kv_blocks,), int32. Used for masking padding positions.
            When None, block_size masking is skipped (assumes all blocks are full).
        q2k_block_nums: Per-(batch, head, q_block) number of KV blocks to attend to,
            (batch, num_heads, num_q_blocks) int32, each value >= 0.
            When None, uses fixed block_sparse_num for all Q blocks.
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
    else:
        assert block_sparse_num >= 2 and block_sparse_num % 2 == 0, (
            f"block_sparse_num={block_sparse_num} must be even and >= 2"
        )
        assert q2k_block_index.shape[-1] >= block_sparse_num, (
            f"q2k_block_index last dim ({q2k_block_index.shape[-1]}) must be >= block_sparse_num ({block_sparse_num})"
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
            block_sparse_num,
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
                block_sparse_num,
                q2k_block_nums.detach() if has_variable_block_nums else None,
                current_stream,
            )

    return out, lse


bsa_attn_fwd.compile_cache = get_jit_cache("bsa_fwd")


def convert_q2k_to_k2q(
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    num_kv_blocks: int,
    q2k_block_nums: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert ``q2k_block_index`` into the ``k2q`` layout expected by the bwd kernel.

    Args:
        q2k_block_index: ``(batch, num_heads, num_q_blocks, max_kv_blocks)`` int32.
            For each (batch, head, q_block), the attended KV block indices.
        block_sparse_num: Number of valid entries per Q block (used only when
            ``q2k_block_nums`` is None).
        num_kv_blocks: Total number of KV blocks.
        q2k_block_nums: Optional ``(batch, num_heads, num_q_blocks)`` int32 holding
            per-Q-block valid counts (overrides ``block_sparse_num`` when set).

    Returns:
        k2q_block_index: ``(batch, num_heads, num_kv_blocks, num_q_blocks)`` int32.
            For each (batch, head, kv_block), the attending Q block indices,
            padded with zeros.
        k2q_block_nums: ``(batch, num_heads, num_kv_blocks)`` int32 holding the
            number of attending Q blocks per KV block.
    """
    from utils.block_sparse_index import convert_q2k_to_k2q_triton
    return convert_q2k_to_k2q_triton(
        q2k_block_index, block_sparse_num, num_kv_blocks,
        q2k_block_nums=q2k_block_nums,
    )


def bsa_attn_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for BSA block-sparse attention (SM100 blk64 only).

    Paired with ``bsa_attn_fwd`` (or the blk64 C++ fwd), this recomputes
    dQ, dK, dV from stored ``out``/``lse`` and the upstream ``dout`` gradient.

    Args:
        dout: Upstream gradient w.r.t. ``out`` (batch, num_heads, seqlen_q, head_dim), bf16.
        q, k, v: Forward inputs (batch, num_heads, seqlen, head_dim), bf16.
            Note this is the ``BHSD`` layout, not the ``BSHD`` layout that
            ``bsa_attn_fwd`` takes — callers that hold BSHD tensors should
            ``.transpose(1, 2)`` before calling this function.
        out: Forward output (same shape/dtype as ``q``).
        lse: Forward log-sum-exp (batch, num_heads, seqlen_q), float32.
        q2k_block_index: Same tensor used for the forward
            (batch, num_heads, num_q_blocks, max_kv_blocks), int32.
        block_sparse_num: Same as forward.
        block_sizes: Same as forward (optional, shape ``(num_kv_blocks,)`` int32).
            When None, all KV blocks are treated as full ``sparse_block_size``.
        q2k_block_nums: Optional per-Q-block variable block count (same semantics
            as forward).
        softmax_scale: Softmax scale (default: 1/sqrt(head_dim)).
        dq, dk, dv: Optional pre-allocated output buffers matching the shapes of
            q/k/v. When None, fresh zero-initialized tensors are allocated.

    Returns:
        (dq, dk, dv): Gradients w.r.t. q, k, v in the same ``BHSD`` layout as
        the forward inputs.

    Notes:
        * Only ``head_dim == 128``, bf16, MHA (num_heads == num_heads_kv) is
          supported. No GQA/MQA, no causal/local, no varlen.
        * Block size is fixed at 64 (``sparse_block_size``) and must match the
          ``blk_m == blk_n == 64`` used for the forward.
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
        f"(b={batch_size}, h={num_heads}, num_q_blocks={num_q_blocks}, max_kv_blocks)"
    )
    if q2k_block_nums is not None:
        assert q2k_block_nums.dtype == torch.int32
        assert q2k_block_nums.shape == (batch_size, num_heads, num_q_blocks)

    k2q_block_index, k2q_block_nums = convert_q2k_to_k2q(
        q2k_block_index, block_sparse_num, num_kv_blocks, q2k_block_nums=q2k_block_nums,
    )

    if block_sizes is None:
        variable_block_sizes = torch.full(
            (batch_size, num_kv_blocks),
            sparse_block_size,
            dtype=torch.int32,
            device=q.device,
        )
    else:
        assert block_sizes.dtype == torch.int32
        if block_sizes.ndim == 1:
            assert block_sizes.shape == (num_kv_blocks,)
            variable_block_sizes = (
                block_sizes.unsqueeze(0).expand(batch_size, -1).contiguous()
            )
        else:
            assert block_sizes.shape == (batch_size, num_kv_blocks)
            variable_block_sizes = block_sizes.contiguous()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if dq is None:
        dq = torch.zeros_like(q)
    else:
        dq.zero_()
    if dk is None:
        dk = torch.zeros_like(k)
    else:
        dk.zero_()
    if dv is None:
        dv = torch.zeros_like(v)
    else:
        dv.zero_()

    workspace_shape = BlockSparseAttnBackward._get_workspace_size(
        q=seqlen_q, d=head_dim, h=num_heads, b=batch_size,
        acc_dtype=Float32,
    )
    workspace = torch.zeros(workspace_shape, dtype=torch.uint8, device=q.device)

    problem_shape = (seqlen_q, seqlen_k, head_dim, (num_heads, batch_size))

    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )

    compile_key = (
        q.dtype,
        head_dim,
        num_heads,
        sparse_block_size,
        arch,
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
        k2q_idx_t = to_cute_tensor(k2q_block_index, leading_dim=3)
        k2q_num_t = to_cute_tensor(k2q_block_nums, leading_dim=2)
        var_bs_t = to_cute_tensor(variable_block_sizes, leading_dim=1)
        ws_t = to_cute_tensor(workspace, fully_dynamic=True)

        bwd_kernel = BlockSparseAttnBackward(sparse_block_size=sparse_block_size)

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
            k2q_idx_t,
            k2q_num_t,
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
                k2q_block_index,
                k2q_block_nums,
                variable_block_sizes,
                workspace,
                softmax_scale,
                current_stream,
            )

    return dq, dk, dv


bsa_attn_bwd.compile_cache = get_jit_cache("bsa_bwd")


@triton.jit
def _qb_count_edges_kernel(
    counts,
    q2k_index,
    q2k_nums,
    idx_b_s: tl.constexpr,
    idx_h_s: tl.constexpr,
    idx_q_s: tl.constexpr,
    idx_k_s: tl.constexpr,
    nums_b_s: tl.constexpr,
    nums_h_s: tl.constexpr,
    nums_q_s: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_blocks: tl.constexpr,
    num_q_groups: tl.constexpr,
    q_bucket_size_blocks: tl.constexpr,
    max_k: tl.constexpr,
    block_sparse_num: tl.constexpr,
    has_variable_nums: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    q = tl.program_id(2)

    n = block_sparse_num
    if has_variable_nums:
        n = tl.load(q2k_nums + b * nums_b_s + h * nums_h_s + q * nums_q_s)

    q_group = q // q_bucket_size_blocks
    q2k_base = q2k_index + b * idx_b_s + h * idx_h_s + q * idx_q_s
    count_base = ((b * num_heads + h) * num_q_groups + q_group) * num_kv_blocks

    for i in tl.range(0, max_k):
        if i < n:
            kv = tl.load(q2k_base + i * idx_k_s)
            if (kv >= 0) & (kv < num_kv_blocks):
                tl.atomic_add(counts + count_base + kv, 1, sem="relaxed")


@triton.jit
def _qb_local_offsets_kernel(
    counts,
    local_offsets,
    group_totals,
    num_heads: tl.constexpr,
    num_kv_blocks: tl.constexpr,
    num_q_groups: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    g = tl.program_id(2)

    count_base = ((b * num_heads + h) * num_q_groups + g) * num_kv_blocks
    offset_base = ((b * num_heads + h) * num_q_groups + g) * (num_kv_blocks + 1)

    running = tl.full((), 0, tl.int32)
    for kv in tl.range(0, num_kv_blocks):
        tl.store(local_offsets + offset_base + kv, running)
        running += tl.load(counts + count_base + kv)
    tl.store(local_offsets + offset_base + num_kv_blocks, running)
    tl.store(group_totals + (b * num_heads + h) * num_q_groups + g, running)


@triton.jit
def _qb_finalize_offsets_kernel(
    local_offsets,
    group_totals,
    task_offsets,
    task_kv,
    num_heads: tl.constexpr,
    num_kv_blocks: tl.constexpr,
    num_q_groups: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    g = tl.program_id(2)

    bh_group_base = (b * num_heads + h) * num_q_groups
    base = tl.full((), 0, tl.int32)
    for prev_g in tl.range(0, num_q_groups):
        if prev_g < g:
            base += tl.load(group_totals + bh_group_base + prev_g)

    local_base = (bh_group_base + g) * (num_kv_blocks + 1)
    task_offset_base = (bh_group_base + g) * (num_kv_blocks + 1)
    task_kv_base = (bh_group_base + g) * num_kv_blocks

    for kv in tl.range(0, num_kv_blocks):
        tl.store(task_offsets + task_offset_base + kv, base + tl.load(local_offsets + local_base + kv))
        tl.store(task_kv + task_kv_base + kv, kv)
    tl.store(
        task_offsets + task_offset_base + num_kv_blocks,
        base + tl.load(local_offsets + local_base + num_kv_blocks),
    )


@triton.jit
def _qb_scatter_q_indices_kernel(
    cursors,
    task_offsets,
    task_q_indices,
    q2k_index,
    q2k_nums,
    idx_b_s: tl.constexpr,
    idx_h_s: tl.constexpr,
    idx_q_s: tl.constexpr,
    idx_k_s: tl.constexpr,
    nums_b_s: tl.constexpr,
    nums_h_s: tl.constexpr,
    nums_q_s: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_blocks: tl.constexpr,
    num_q_groups: tl.constexpr,
    q_bucket_size_blocks: tl.constexpr,
    max_edges_per_bh: tl.constexpr,
    max_k: tl.constexpr,
    block_sparse_num: tl.constexpr,
    has_variable_nums: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    q = tl.program_id(2)

    n = block_sparse_num
    if has_variable_nums:
        n = tl.load(q2k_nums + b * nums_b_s + h * nums_h_s + q * nums_q_s)

    q_group = q // q_bucket_size_blocks
    q2k_base = q2k_index + b * idx_b_s + h * idx_h_s + q * idx_q_s
    cursor_base = ((b * num_heads + h) * num_q_groups + q_group) * num_kv_blocks
    offset_base = ((b * num_heads + h) * num_q_groups + q_group) * (num_kv_blocks + 1)
    q_indices_base = (b * num_heads + h) * max_edges_per_bh

    for i in tl.range(0, max_k):
        if i < n:
            kv = tl.load(q2k_base + i * idx_k_s)
            if (kv >= 0) & (kv < num_kv_blocks):
                pos = tl.atomic_add(cursors + cursor_base + kv, 1, sem="relaxed")
                task_offset = tl.load(task_offsets + offset_base + kv)
                tl.store(task_q_indices + q_indices_base + task_offset + pos, q)


def build_q_range_bucketed_tasks(
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    num_kv_blocks: int,
    *,
    q_bucket_size_blocks: int = 512,
    q2k_block_nums: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build per-(B,H) q-range CSR tasks on GPU for the q-bucket bwd path."""
    assert q2k_block_index.dtype == torch.int32
    assert q2k_block_index.is_cuda
    B, H, num_q_blocks, max_kv = q2k_block_index.shape
    G = q_bucket_size_blocks
    num_q_groups = (num_q_blocks + G - 1) // G

    q2k_block_index = q2k_block_index.contiguous()
    if q2k_block_nums is None:
        nums = q2k_block_index
        max_edges = num_q_blocks * int(block_sparse_num)
        max_k = int(block_sparse_num)
        has_variable_nums = False
    else:
        assert q2k_block_nums.dtype == torch.int32
        assert q2k_block_nums.shape == (B, H, num_q_blocks)
        nums = q2k_block_nums.contiguous()
        max_edges = num_q_blocks * max_kv
        max_k = max_kv
        has_variable_nums = True

    max_edges = max(1, max_edges)
    max_tasks_per_group = num_kv_blocks

    counts = torch.zeros(
        (B, H, num_q_groups, num_kv_blocks),
        dtype=torch.int32,
        device=q2k_block_index.device,
    )
    local_offsets = torch.empty(
        (B, H, num_q_groups, num_kv_blocks + 1),
        dtype=torch.int32,
        device=q2k_block_index.device,
    )
    group_totals = torch.empty(
        (B, H, num_q_groups),
        dtype=torch.int32,
        device=q2k_block_index.device,
    )
    task_offsets = torch.empty_like(local_offsets)
    task_kv = torch.empty_like(counts)
    task_q_indices = torch.empty(
        (B, H, max_edges), dtype=torch.int32, device=q2k_block_index.device
    )

    grid_q = (B, H, num_q_blocks)
    _qb_count_edges_kernel[grid_q](
        counts,
        q2k_block_index,
        nums,
        q2k_block_index.stride(0),
        q2k_block_index.stride(1),
        q2k_block_index.stride(2),
        q2k_block_index.stride(3),
        nums.stride(0) if has_variable_nums else 0,
        nums.stride(1) if has_variable_nums else 0,
        nums.stride(2) if has_variable_nums else 0,
        H,
        num_kv_blocks,
        num_q_groups,
        G,
        max_k,
        int(block_sparse_num),
        has_variable_nums,
    )

    grid_group = (B, H, num_q_groups)
    _qb_local_offsets_kernel[grid_group](
        counts,
        local_offsets,
        group_totals,
        H,
        num_kv_blocks,
        num_q_groups,
    )
    _qb_finalize_offsets_kernel[grid_group](
        local_offsets,
        group_totals,
        task_offsets,
        task_kv,
        H,
        num_kv_blocks,
        num_q_groups,
    )

    cursors = torch.zeros_like(counts)
    _qb_scatter_q_indices_kernel[grid_q](
        cursors,
        task_offsets,
        task_q_indices,
        q2k_block_index,
        nums,
        q2k_block_index.stride(0),
        q2k_block_index.stride(1),
        q2k_block_index.stride(2),
        q2k_block_index.stride(3),
        nums.stride(0) if has_variable_nums else 0,
        nums.stride(1) if has_variable_nums else 0,
        nums.stride(2) if has_variable_nums else 0,
        H,
        num_kv_blocks,
        num_q_groups,
        G,
        max_edges,
        max_k,
        int(block_sparse_num),
        has_variable_nums,
    )

    return task_kv, task_offsets, task_q_indices, num_q_groups, max_tasks_per_group


def bsa_attn_bwd_qbucket(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    q_bucket_size_blocks: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q-range bucketed backward pass for BSA block-sparse attention.

    This has the same tensor contract as :func:`bsa_attn_bwd`, but builds a
    compact q-range task layout and runs the bucketed SM100 blk64 backward
    kernel. Bucket task construction is performed on GPU with Triton on every
    call, so this path is suitable when the sparse pattern changes each
    backward.
    """
    q, k, v, out, dout = [maybe_contiguous(t) for t in (q, k, v, out, dout)]
    lse = maybe_contiguous(lse)

    assert q.dtype == torch.bfloat16, "q-bucket bwd only supports bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
    assert lse.dtype == torch.float32
    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v, out, dout, lse))

    batch_size, num_heads, seqlen_q, head_dim = q.shape
    num_heads_kv, seqlen_k = k.shape[1], k.shape[2]
    assert head_dim == BSA_BWD_HEAD_DIM
    assert num_heads == num_heads_kv, "q-bucket bwd does not support GQA/MQA"
    assert k.shape == v.shape == (batch_size, num_heads, seqlen_k, head_dim)
    assert out.shape == (batch_size, num_heads, seqlen_q, head_dim)
    assert dout.shape == out.shape
    assert lse.shape == (batch_size, num_heads, seqlen_q)

    arch = _get_device_arch()
    assert arch // 10 in [10, 11], "BSA q-bucket bwd only supports SM100/SM110"

    sparse_block_size = BSA_BWD_SPARSE_BLOCK_SIZE
    num_q_blocks = (seqlen_q + sparse_block_size - 1) // sparse_block_size
    num_kv_blocks = (seqlen_k + sparse_block_size - 1) // sparse_block_size

    assert q2k_block_index.dtype == torch.int32
    assert q2k_block_index.shape[:3] == (batch_size, num_heads, num_q_blocks)
    if q2k_block_nums is not None:
        q2k_block_nums = maybe_contiguous(q2k_block_nums)
        assert q2k_block_nums.dtype == torch.int32
        assert q2k_block_nums.shape == (batch_size, num_heads, num_q_blocks)

    task_kv, task_offsets, task_q_indices, _num_q_groups, _max_tasks_per_group = (
        build_q_range_bucketed_tasks(
            q2k_block_index,
            block_sparse_num,
            num_kv_blocks,
            q_bucket_size_blocks=q_bucket_size_blocks,
            q2k_block_nums=q2k_block_nums,
        )
    )

    if block_sizes is None:
        variable_block_sizes = torch.full(
            (batch_size, num_kv_blocks),
            sparse_block_size,
            dtype=torch.int32,
            device=q.device,
        )
    else:
        assert block_sizes.dtype == torch.int32
        if block_sizes.ndim == 1:
            assert block_sizes.shape == (num_kv_blocks,)
            variable_block_sizes = (
                block_sizes.unsqueeze(0).expand(batch_size, -1).contiguous()
            )
        else:
            assert block_sizes.shape == (batch_size, num_kv_blocks)
            variable_block_sizes = block_sizes.contiguous()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if dq is None:
        dq = torch.zeros_like(q)
    else:
        dq.zero_()
    if dk is None:
        dk = torch.zeros_like(k)
    else:
        dk.zero_()
    if dv is None:
        dv = torch.zeros_like(v)
    else:
        dv.zero_()

    workspace_shape = BlockSparseAttnBackwardQRangeBucketed._get_workspace_size(
        q=seqlen_q,
        k=seqlen_k,
        d=head_dim,
        h=num_heads,
        b=batch_size,
        acc_dtype=Float32,
    )
    workspace = torch.zeros(workspace_shape, dtype=torch.uint8, device=q.device)

    problem_shape = (seqlen_q, seqlen_k, head_dim, (num_heads, batch_size))
    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )

    compile_key = (
        q.dtype,
        head_dim,
        num_heads,
        sparse_block_size,
        arch,
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

    if compile_key not in bsa_attn_bwd_qbucket.compile_cache:
        dO_t = convert_to_cute_tensor(dout)
        O_t = convert_to_cute_tensor(out)
        Q_t = convert_to_cute_tensor(q)
        K_t = convert_to_cute_tensor(k)
        V_t = convert_to_cute_tensor(v)
        dQ_t = convert_to_cute_tensor(dq)
        dK_t = convert_to_cute_tensor(dk)
        dV_t = convert_to_cute_tensor(dv)
        LSE_t = to_cute_tensor(lse, leading_dim=2)
        task_kv_t = to_cute_tensor(task_kv, leading_dim=3)
        task_offsets_t = to_cute_tensor(task_offsets, leading_dim=3)
        task_q_indices_t = to_cute_tensor(task_q_indices, leading_dim=2)
        var_bs_t = to_cute_tensor(variable_block_sizes, leading_dim=1)
        ws_t = to_cute_tensor(workspace, fully_dynamic=True)

        bwd_kernel = BlockSparseAttnBackwardQRangeBucketed(
            sparse_block_size=sparse_block_size,
        )

        bsa_attn_bwd_qbucket.compile_cache[compile_key] = cute.compile(
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
            task_kv_t,
            task_offsets_t,
            task_q_indices_t,
            var_bs_t,
            ws_t,
            softmax_scale,
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        with torch.cuda.nvtx.range("bsa_attn_bwd_qbucket_kernel"):
            bsa_attn_bwd_qbucket.compile_cache[compile_key](
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
                task_kv,
                task_offsets,
                task_q_indices,
                variable_block_sizes,
                workspace,
                softmax_scale,
                current_stream,
            )

    return dq, dk, dv


bsa_attn_bwd_qbucket.compile_cache = get_jit_cache("bsa_bwd_qbucket")
