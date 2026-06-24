# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# BSA attention interface for SM90/SM100 block-sparse kernels.

import math
import time
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

from utils import fa_logging
from csrc.fwd.sm100_blk128.flash_fwd_sm100 import (
    BlockSparseAttnForwardSm100Blk128,
)
from csrc.fwd.sm90_blk64.flash_fwd_sm90 import (
    BlockSparseAttnForwardSm90Blk64,
    SM90_FWD_BLOCK_SIZE,
)
from csrc.fwd.sm120_blk64.flash_fwd_sm120 import (
    BlockSparseAttnForwardSm120Blk64,
    SM120_FWD_BLOCK_SIZE,
)
try:
    from csrc.fwd.sm100_blk64.bsa_fwd_combine import FlashAttentionForwardCombine
except ImportError:
    FlashAttentionForwardCombine = None
from csrc.fwd.sm100_blk64.flash_fwd_sm100 import FlashAttentionForwardSm100Blk64
from csrc.bwd.sm100_blk64.flash_bwd_sm100 import (
    BlockSparseAttnBackwardSm100Blk64,
    SM100_BWD_HEAD_DIM,
    SM100_BLK64_BWD_SPARSE_BLOCK_SIZE,
    sm100_bwd_auto_bucketed_k2q_size_blocks,
    sm100_bwd_default_bucketed_k2q_size_blocks,
)
from csrc.bwd.sm100_blk128.flash_bwd_sm100 import (
    SM100_BWD_HEAD_DIM as SM100_BLK128_BWD_HEAD_DIM,
    SM100_BLK128_BWD_SPARSE_BLOCK_SIZE,
    bsa_sm100_blk128_bwd_bucketed_k2q_csr,
    sm100_blk128_bwd_default_bucketed_k2q_size_blocks,
)
from csrc.bwd.sm90_blk64.flash_bwd_sm90 import (
    BlockSparseAttnBackwardSm90Blk64,
    SM90_BWD_HEAD_DIM,
    SM90_BWD_SPARSE_BLOCK_SIZE,
    sm90_bwd_auto_bucketed_k2q_size_blocks,
    sm90_bwd_default_bucketed_k2q_size_blocks,
)

try:
    import bsa_fwd_blk64_ext  # triggers TORCH_LIBRARY registration of bsa_blk64.fwd
except ImportError:
    pass  # blk64 wheel not built — bsa_attn_fwd_blk64 will fail at call time

@lru_cache(maxsize=None)
def _get_device_arch():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _to_cute_tensor(
    t: torch.Tensor,
    assumed_align: int = 16,
    leading_dim: int = -1,
    fully_dynamic: bool = False,
    enable_tvm_ffi: bool = True,
) -> cute.Tensor:
    tensor = from_dlpack(
        t.detach(),
        assumed_align=assumed_align,
        enable_tvm_ffi=enable_tvm_ffi,
    )
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


def _to_cute_tensor_dynamic_compact_shape(
    t: torch.Tensor,
    mode: int | tuple[int, ...],
    assumed_align: int = 16,
    leading_dim: int = -1,
    divisibility: int = 1,
    stride_order: tuple[int, ...] | None = None,
) -> cute.Tensor:
    tensor = _to_cute_tensor(t, assumed_align=assumed_align, leading_dim=leading_dim)
    if isinstance(mode, int):
        mode = (mode,)
    stride_order = t.dim_order() if stride_order is None else stride_order
    for mode_i in mode:
        tensor = tensor.mark_compact_shape_dynamic(
            mode=mode_i,
            stride_order=stride_order,
            divisibility=divisibility,
        )
    return tensor


def _to_sm90_bwd_cute_tensor(
    t: torch.Tensor, assumed_align: int = 16, enable_tvm_ffi: bool = True
) -> cute.Tensor:
    return _to_cute_tensor(
        t,
        assumed_align=assumed_align,
        enable_tvm_ffi=enable_tvm_ffi,
    )


def _workaround_cutlass_hash_import_bug():
    """Avoid optional generated dialect imports that are broken in this environment."""
    import importlib
    import sys
    import types

    for suffix in ("arith", "dialect_proxy", "gpu", "lru_cache_ir", "op"):
        canonical = f"cutlass._mlir_helpers.{suffix}"
        alias = f"cutlass.base_dsl._mlir_helpers.{suffix}"
        try:
            sys.modules.setdefault(alias, importlib.import_module(canonical))
        except Exception:
            pass

    for name in (
        "cutlass._mlir.dialects._iket_ops_gen",
        "cutlass._mlir.dialects._bitfield_ops_gen",
        "cutlass._mlir.dialects._pyir_ops_gen",
        "cutlass._mlir.dialects._ub_ops_gen",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

_SM100_BLK64_INT32_MAX = torch.iinfo(torch.int32).max


def _sm100_blk64_require_int32(name: str, value: int) -> int:
    value = int(value)
    if value < 0 or value > _SM100_BLK64_INT32_MAX:
        raise ValueError(
            f"SM100 blk64 {name}={value} must fit in int32 "
            f"(<= {_SM100_BLK64_INT32_MAX})"
        )
    return value


def _sm100_blk64_round_up_to_block(name: str, value: int, block: int = 64) -> int:
    value = _sm100_blk64_require_int32(name, value)
    rounded = ((value + block - 1) // block) * block
    return _sm100_blk64_require_int32(f"{name}_rounded", rounded)


def _validate_sm100_blk64_int32_bounds(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor],
    q2k_block_nums: Optional[torch.Tensor],
) -> None:
    """Guard values that the SM100 blk64 C++/CUDA path stores or casts as int32."""
    batch, num_heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[2]
    num_m_blocks = (seqlen_q + 63) // 64

    for tensor_name, tensor in (("q", q), ("k", k), ("v", v)):
        _sm100_blk64_require_int32(f"{tensor_name}.shape[0]", tensor.shape[0])
        _sm100_blk64_require_int32(f"{tensor_name}.shape[1]", tensor.shape[1])
        _sm100_blk64_require_int32(f"{tensor_name}.shape[2]", tensor.shape[2])
        _sm100_blk64_require_int32(f"{tensor_name}.shape[3]", tensor.shape[3])
        _sm100_blk64_require_int32(f"{tensor_name}.stride(1)", tensor.stride(1))
        _sm100_blk64_require_int32(f"{tensor_name}.stride(2)", tensor.stride(2))
        _sm100_blk64_require_int32(f"{tensor_name}.stride(3)", tensor.stride(3))

    _sm100_blk64_round_up_to_block("seqlen_k", seqlen_k)
    _sm100_blk64_require_int32("num_m_blocks", num_m_blocks)
    _sm100_blk64_require_int32("block_indices_stride", q2k_block_index.shape[-1])
    _sm100_blk64_require_int32("total_q_tiles", batch * num_heads * num_m_blocks)
    _sm100_blk64_require_int32("block_sparse_num", block_sparse_num)

    if block_sizes is not None and block_sizes.numel() > 0:
        _sm100_blk64_require_int32("block_sizes.numel", block_sizes.numel())
    if q2k_block_nums is not None and q2k_block_nums.numel() > 0:
        _sm100_blk64_require_int32("q2k_block_nums.numel", q2k_block_nums.numel())


def _tensor_compile_key(t: torch.Tensor):
    return (tuple(t.shape), tuple(t.stride()), t.dtype)


def _tensor_layout_compile_key(t: torch.Tensor):
    return (tuple(t.dim_order()), tuple(s == 0 for s in t.stride()))


def _ceil_div_int(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _ceil_log2_int(x: int) -> int:
    x = int(x)
    assert x >= 1
    return (x - 1).bit_length()


def _bsa_fwd_blk64_kv_bucketed_combine_compile_key(
    dtype,
    head_dim: int,
    combine_tile_m: int,
    combine_k_block_size: int,
    log_max_splits: int,
    combine_num_threads: int,
    combine_stages: int,
):
    return (
        dtype,
        cutlass.Float32,
        int(head_dim),
        int(combine_tile_m),
        int(combine_k_block_size),
        int(log_max_splits),
        int(combine_num_threads),
        int(combine_stages),
        "bshd_nonvarlen_seqlen_dynamic",
    )


def _to_cute_tensor_with_dynamic_modes(
    t: torch.Tensor,
    dynamic_modes: int | tuple[int, ...],
    assumed_align: int = 16,
    leading_dim: int = -1,
    divisibility: int = 1,
) -> cute.Tensor:
    return _to_cute_tensor_dynamic_compact_shape(
        t,
        mode=dynamic_modes,
        assumed_align=assumed_align,
        leading_dim=leading_dim,
        divisibility=divisibility,
    )


def _sm100_blk64_auto_kv_splits(
    q: torch.Tensor,
    q2k_block_index: torch.Tensor,
    fixed_block_sparse_num: int,
    max_kv_splits: int = 16,
) -> int:
    """Choose KV splits for the SM100 blk64 KV-bucketed target cases."""
    if is_fake_mode() or not q.is_cuda:
        return 1

    kv_blocks = int(fixed_block_sparse_num)
    if kv_blocks <= 0:
        kv_blocks = int(q2k_block_index.shape[-1])
    if kv_blocks <= 1:
        return 1

    if kv_blocks >= 900:
        splits = 8
    elif kv_blocks >= 450:
        splits = 4
    elif kv_blocks >= 256:
        splits = 2
    else:
        splits = 1
    return max(1, min(int(splits), int(max_kv_splits), kv_blocks))


def _infer_sm100_bwd_sparse_block_size(
    *,
    batch_size: int,
    num_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    q2k_block_index: torch.Tensor,
    block_sizes: Optional[torch.Tensor],
) -> int:
    candidates = []
    for block_size in (SM100_BLK64_BWD_SPARSE_BLOCK_SIZE, SM100_BLK128_BWD_SPARSE_BLOCK_SIZE):
        num_q_blocks = _ceil_div_int(seqlen_q, block_size)
        num_kv_blocks = _ceil_div_int(seqlen_k, block_size)
        if q2k_block_index.shape[:3] != (batch_size, num_heads, num_q_blocks):
            continue
        if block_sizes is not None and block_sizes.numel() > 0:
            if block_sizes.ndim == 1 and block_sizes.shape != (num_kv_blocks,):
                continue
            if block_sizes.ndim == 2 and block_sizes.shape != (batch_size, num_kv_blocks):
                continue
        candidates.append(block_size)

    assert candidates, (
        "Could not infer SM100 bwd sparse block size from q2k/block_sizes shapes"
    )
    # Preserve the legacy blk64 path for tiny ambiguous shapes.
    return (
        SM100_BLK64_BWD_SPARSE_BLOCK_SIZE
        if SM100_BLK64_BWD_SPARSE_BLOCK_SIZE in candidates
        else candidates[0]
    )


def _empty_bwd_workspace_with_zeroed_accum(
    *,
    batch_size: int,
    num_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    round_q_to: int,
    round_k_to: int,
    round_d_to: int,
    zero_dq_accum: bool,
    device: torch.device,
) -> torch.Tensor:
    q_rounded = ((seqlen_q + round_q_to - 1) // round_q_to) * round_q_to
    k_rounded = ((seqlen_k + round_k_to - 1) // round_k_to) * round_k_to
    d_rounded = ((head_dim + round_d_to - 1) // round_d_to) * round_d_to
    elems_per_bh = 2 * q_rounded + q_rounded * d_rounded + 2 * k_rounded * d_rounded
    workspace = torch.empty(
        (
            batch_size,
            num_heads,
            elems_per_bh,
        ),
        dtype=torch.float32,
        device=device,
    )
    accum_offset = 2 * q_rounded
    if not zero_dq_accum:
        accum_offset += q_rounded * d_rounded
    flat_accum_offset = batch_size * num_heads * accum_offset
    workspace.reshape(-1)[flat_accum_offset:].zero_()
    return workspace


def _bsa_attn_fwd_sm90_blk64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch the SM90 blk64 sparse forward kernel on BHSD tensors."""
    assert q.dtype in (torch.float16, torch.bfloat16), "SM90 blk64 fwd supports fp16/bf16"
    assert q.dtype == k.dtype == v.dtype
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4

    batch, num_q_heads, seqlen_q, head_dim = q.shape
    batch_k, num_kv_heads, seqlen_k, head_dim_k = k.shape
    assert batch_k == batch and head_dim_k == head_dim
    assert v.shape[:3] == (batch, num_kv_heads, seqlen_k)
    assert head_dim in (64, 96, 128), "SM90 blk64 fwd supports QK dim 64, 96, or 128"
    assert v.shape[-1] in (64, 96, 128), "SM90 blk64 fwd supports value dim 64, 96, or 128"
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    assert seqlen_q % SM90_FWD_BLOCK_SIZE == 0, (
        "SM90 blk64 fwd requires seqlen_q to be a multiple of 64"
    )
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"

    gqa_ratio = num_q_heads // num_kv_heads
    num_q_blocks = seqlen_q // SM90_FWD_BLOCK_SIZE
    num_kv_blocks = (seqlen_k + SM90_FWD_BLOCK_SIZE - 1) // SM90_FWD_BLOCK_SIZE
    assert q2k_block_index.dtype == torch.int32
    assert q2k_block_index.shape[:3] == (batch, num_q_heads, num_q_blocks)

    if q2k_block_nums is None:
        assert block_sparse_num >= 1
        q2k_block_nums = torch.full(
            (batch, num_q_heads, num_q_blocks),
            block_sparse_num,
            dtype=torch.int32,
            device=q.device,
        )
    else:
        assert q2k_block_nums.dtype == torch.int32
        assert q2k_block_nums.shape == (batch, num_q_heads, num_q_blocks)
        q2k_block_nums = q2k_block_nums.contiguous()

    has_block_sizes = block_sizes is not None
    if has_block_sizes:
        assert block_sizes.dtype == torch.int32
        if block_sizes.ndim == 1:
            assert block_sizes.shape == (num_kv_blocks,)
            block_sizes_bh = (
                block_sizes[None, None, :].expand(batch, num_q_heads, -1).contiguous()
            )
        elif block_sizes.ndim == 2:
            assert block_sizes.shape == (batch, num_kv_blocks)
            block_sizes_bh = block_sizes[:, None, :].expand(-1, num_q_heads, -1).contiguous()
        else:
            assert block_sizes.shape == (batch, num_q_heads, num_kv_blocks)
            block_sizes_bh = block_sizes.contiguous()

    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    if out is None:
        out = torch.empty(
            (batch, num_q_heads, seqlen_q, v.shape[-1]), dtype=q.dtype, device=q.device
        )
    else:
        assert out.shape == (batch, num_q_heads, seqlen_q, v.shape[-1])
        assert out.dtype == q.dtype and out.is_cuda

    lse = torch.empty((batch, num_q_heads, seqlen_q), dtype=torch.float32, device=q.device)

    q2k_block_index = q2k_block_index.contiguous()

    q_t = q.permute(2, 3, 1, 0)
    k_t = k.permute(2, 3, 1, 0)
    v_t = v.permute(3, 2, 1, 0)
    out_t = out.permute(2, 3, 1, 0)
    lse_t = lse.permute(2, 1, 0)
    q2k_t = q2k_block_index.permute(3, 2, 1, 0)
    q2k_nums_t = q2k_block_nums.permute(2, 1, 0)
    if has_block_sizes:
        block_sizes_t = block_sizes_bh.permute(2, 1, 0)

    q_cute = from_dlpack(q_t.detach(), assumed_align=128)
    k_cute = from_dlpack(k_t.detach(), assumed_align=128)
    v_cute = from_dlpack(v_t.detach(), assumed_align=128)
    out_cute = from_dlpack(out_t.detach(), assumed_align=128)
    lse_cute = from_dlpack(lse_t.detach(), assumed_align=4)
    q2k_cute = from_dlpack(q2k_t.detach())
    q2k_nums_cute = from_dlpack(q2k_nums_t.detach())
    block_sizes_cute = (
        from_dlpack(block_sizes_t.detach()) if has_block_sizes else q2k_nums_cute
    )

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fwd_kernel = BlockSparseAttnForwardSm90Blk64(
        gqa_ratio=gqa_ratio,
        head_dim=head_dim,
        value_dim=v.shape[-1],
        blocksparse_blocksize_q=SM90_FWD_BLOCK_SIZE,
        blocksparse_blocksize_k=SM90_FWD_BLOCK_SIZE,
        dtype=torch2cute_dtype_map[q.dtype],
        acc_dtype=cutlass.Float32,
        has_block_sizes=has_block_sizes,
    )

    compile_key = (
        "sm90_blk64",
        q.dtype,
        head_dim,
        v.shape[-1],
        gqa_ratio,
        SM90_FWD_BLOCK_SIZE,
        _tensor_compile_key(q_t),
        _tensor_compile_key(k_t),
        _tensor_compile_key(v_t),
        _tensor_compile_key(out_t),
        _tensor_compile_key(lse_t),
        _tensor_compile_key(q2k_t),
        _tensor_compile_key(q2k_nums_t),
        has_block_sizes,
        _tensor_compile_key(block_sizes_t) if has_block_sizes else None,
    )
    args = (
        q_cute,
        k_cute,
        v_cute,
        out_cute,
        lse_cute,
        q2k_cute,
        q2k_nums_cute,
        block_sizes_cute,
        softmax_scale,
        current_stream,
    )
    if compile_key not in bsa_attn_fwd.compile_cache:
        t0 = time.time()
        bsa_attn_fwd.compile_cache[compile_key] = cute.compile(fwd_kernel, *args)
        print(f"Compiled sm90 blk64 fwd in {time.time() - t0:.1f}s")

    if not is_fake_mode():
        with torch.cuda.nvtx.range("bsa_attn_fwd_sm90_blk64_kernel"):
            bsa_attn_fwd.compile_cache[compile_key](*args)

    return out, lse


def _bsa_attn_fwd_sm120_blk64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch the SM120 blk64 sparse forward kernel on BHSD tensors."""
    assert q.dtype in (torch.float16, torch.bfloat16), "SM120 blk64 fwd supports fp16/bf16"
    assert q.dtype == k.dtype == v.dtype
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4

    batch, num_q_heads, seqlen_q, head_dim = q.shape
    batch_k, num_kv_heads, seqlen_k, head_dim_k = k.shape
    assert batch_k == batch and head_dim_k == head_dim
    assert v.shape[:3] == (batch, num_kv_heads, seqlen_k)
    assert head_dim == 128, "SM120 blk64 fwd currently requires QK dim 128"
    assert v.shape[-1] == 128, "SM120 blk64 fwd currently requires value dim 128"
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"

    gqa_ratio = num_q_heads // num_kv_heads
    num_q_blocks = _ceil_div_int(seqlen_q, SM120_FWD_BLOCK_SIZE)
    num_kv_blocks = _ceil_div_int(seqlen_k, SM120_FWD_BLOCK_SIZE)
    assert q2k_block_index.dtype == torch.int32
    assert q2k_block_index.shape[:3] == (batch, num_q_heads, num_q_blocks)

    has_block_nums = q2k_block_nums is not None and q2k_block_nums.numel() > 0
    if not has_block_nums:
        assert block_sparse_num >= 1
        assert block_sparse_num <= q2k_block_index.shape[-1]
    else:
        assert q2k_block_nums.dtype == torch.int32
        assert q2k_block_nums.shape == (batch, num_q_heads, num_q_blocks)
        q2k_block_nums = q2k_block_nums.contiguous()

    has_block_sizes = block_sizes is not None and block_sizes.numel() > 0
    block_sizes_mode = 0
    if has_block_sizes:
        assert block_sizes.dtype == torch.int32
        if block_sizes.ndim == 1:
            assert block_sizes.shape == (num_kv_blocks,)
            block_sizes_t = block_sizes.contiguous()
            block_sizes_mode = 1
        elif block_sizes.ndim == 2:
            assert block_sizes.shape == (batch, num_kv_blocks)
            block_sizes_t = block_sizes.contiguous().permute(1, 0)
            block_sizes_mode = 2
        else:
            assert block_sizes.shape == (batch, num_q_heads, num_kv_blocks)
            block_sizes_t = block_sizes.contiguous().permute(2, 1, 0)
            block_sizes_mode = 3

    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    if out is None:
        out = torch.empty(
            (batch, num_q_heads, seqlen_q, v.shape[-1]), dtype=q.dtype, device=q.device
        )
    else:
        assert out.shape == (batch, num_q_heads, seqlen_q, v.shape[-1])
        assert out.dtype == q.dtype and out.is_cuda

    lse = torch.empty((batch, num_q_heads, seqlen_q), dtype=torch.float32, device=q.device)

    q2k_block_index = q2k_block_index.contiguous()

    q_t = q.permute(2, 3, 1, 0)
    k_t = k.permute(2, 3, 1, 0)
    v_t = v.permute(3, 2, 1, 0)
    out_t = out.permute(2, 3, 1, 0)
    lse_t = lse.permute(2, 1, 0)
    q2k_t = q2k_block_index.permute(3, 2, 1, 0)
    q2k_nums_t = q2k_block_nums.permute(2, 1, 0) if has_block_nums else q2k_t

    q_cute = from_dlpack(q_t.detach(), assumed_align=128)
    k_cute = from_dlpack(k_t.detach(), assumed_align=128)
    v_cute = from_dlpack(v_t.detach(), assumed_align=128)
    out_cute = from_dlpack(out_t.detach(), assumed_align=128)
    lse_cute = from_dlpack(lse_t.detach(), assumed_align=4)
    q2k_cute = from_dlpack(q2k_t.detach())
    q2k_nums_cute = from_dlpack(q2k_nums_t.detach())
    block_sizes_cute = (
        from_dlpack(block_sizes_t.detach()) if has_block_sizes else q2k_nums_cute
    )

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fwd_kernel = BlockSparseAttnForwardSm120Blk64(
        gqa_ratio=gqa_ratio,
        head_dim=head_dim,
        value_dim=v.shape[-1],
        blocksparse_blocksize_q=SM120_FWD_BLOCK_SIZE,
        blocksparse_blocksize_k=SM120_FWD_BLOCK_SIZE,
        dtype=torch2cute_dtype_map[q.dtype],
        acc_dtype=cutlass.Float32,
        has_block_sizes=has_block_sizes,
        has_block_nums=has_block_nums,
        block_sizes_mode=block_sizes_mode,
    )

    compile_key = (
        "sm120_blk64",
        q.dtype,
        head_dim,
        v.shape[-1],
        gqa_ratio,
        SM120_FWD_BLOCK_SIZE,
        _tensor_compile_key(q_t),
        _tensor_compile_key(k_t),
        _tensor_compile_key(v_t),
        _tensor_compile_key(out_t),
        _tensor_compile_key(lse_t),
        _tensor_compile_key(q2k_t),
        _tensor_compile_key(q2k_nums_t) if has_block_nums else None,
        has_block_nums,
        has_block_sizes,
        block_sizes_mode,
        _tensor_compile_key(block_sizes_t) if has_block_sizes else None,
    )
    args = (
        q_cute,
        k_cute,
        v_cute,
        out_cute,
        lse_cute,
        q2k_cute,
        q2k_nums_cute,
        block_sparse_num,
        block_sizes_cute,
        softmax_scale,
        current_stream,
    )
    if compile_key not in bsa_attn_fwd.compile_cache:
        t0 = time.time()
        bsa_attn_fwd.compile_cache[compile_key] = cute.compile(fwd_kernel, *args)
        print(f"Compiled sm120 blk64 fwd in {time.time() - t0:.1f}s")

    if not is_fake_mode():
        with torch.cuda.nvtx.range("bsa_attn_fwd_sm120_blk64_kernel"):
            bsa_attn_fwd.compile_cache[compile_key](*args)

    return out, lse


def _bsa_attn_fwd_blk64_kv_bucketed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    fixed_block_sparse_num: int,
    block_sizes: torch.Tensor,
    q2k_block_nums: torch.Tensor,
    softmax_scale: float,
    kv_splits: int,
    use_clc: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch KV-bucketed SM100 blk64 fwd and combine partial outputs."""
    if FlashAttentionForwardCombine is None:
        raise ImportError(
            "FlashAttentionForwardCombine is unavailable. Ensure local fa_cute "
            "helpers are importable."
        )

    kv_splits = int(kv_splits)
    assert 1 <= kv_splits <= 256, "kv_splits must be in [1, 256]"

    o_partial_phys, lse_partial_phys, split_offsets = torch.ops.bsa_blk64.fwd_kv_bucketed(
        q,
        k,
        v,
        q2k_block_index,
        fixed_block_sparse_num,
        block_sizes,
        softmax_scale,
        q2k_block_nums,
        kv_splits,
        use_clc,
    )

    batch, num_heads, seqlen_q, head_dim = q.shape
    if o_partial_phys.dtype != torch.float32:
        raise TypeError("KV-bucketed blk64 fwd requires fp32 O partial")

    split_heads = kv_splits * num_heads
    o_partial = o_partial_phys.as_strided(
        (kv_splits, batch, seqlen_q, num_heads, head_dim),
        (
            num_heads * seqlen_q * head_dim,
            seqlen_q * split_heads * head_dim,
            head_dim,
            seqlen_q * head_dim,
            1,
        ),
    )
    lse_partial = lse_partial_phys.as_strided(
        (kv_splits, batch, seqlen_q, num_heads),
        (
            num_heads * seqlen_q,
            seqlen_q * split_heads,
            1,
            seqlen_q,
        ),
    )
    out_bshd = torch.empty(
        (batch, seqlen_q, num_heads, head_dim),
        dtype=q.dtype,
        device=q.device,
    )
    lse_bsh = torch.empty(
        (batch, seqlen_q, num_heads),
        dtype=torch.float32,
        device=q.device,
    )
    dtype = torch2cute_dtype_map[q.dtype]
    log_max_splits = _ceil_log2_int(kv_splits)
    # Baseline combine geometry; a single configuration is easier to maintain.
    combine_tile_m = 16
    combine_k_block_size = 64
    combine_num_threads = 128
    combine_stages = 4

    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )
    compile_key = _bsa_fwd_blk64_kv_bucketed_combine_compile_key(
        dtype,
        head_dim,
        combine_tile_m,
        combine_k_block_size,
        log_max_splits,
        combine_num_threads,
        combine_stages,
    )
    if compile_key not in _bsa_attn_fwd_blk64_kv_bucketed.compile_cache:
        combine_kernel = FlashAttentionForwardCombine(
            dtype=dtype,
            head_dim=head_dim,
            tile_m=combine_tile_m,
            k_block_size=combine_k_block_size,
            log_max_splits=log_max_splits,
            num_threads=combine_num_threads,
            stages=combine_stages,
        )
        args = (
            _to_cute_tensor_dynamic_compact_shape(
                o_partial,
                mode=(0, 1, 2, 3),
                stride_order=(1, 0, 3, 2, 4),
            ),
            _to_cute_tensor_dynamic_compact_shape(
                lse_partial,
                mode=(0, 1, 2, 3),
                assumed_align=4,
                leading_dim=2,
                stride_order=(1, 0, 3, 2),
            ),
            _to_cute_tensor_dynamic_compact_shape(
                out_bshd,
                mode=(0, 1, 2),
                stride_order=(0, 1, 2, 3),
            ),
            _to_cute_tensor_dynamic_compact_shape(
                lse_bsh,
                mode=(0, 1, 2),
                assumed_align=4,
                stride_order=(0, 1, 2),
            ),
            None,
            None,
            None,
            None,
            None,
            current_stream,
        )
        t0 = time.time()
        _bsa_attn_fwd_blk64_kv_bucketed.compile_cache[compile_key] = cute.compile(
            combine_kernel,
            *args,
            options="--enable-tvm-ffi",
        )
        print(f"Compiled kv-bucketed fwd combine in {time.time() - t0:.1f}s")

    if not is_fake_mode():
        _bsa_attn_fwd_blk64_kv_bucketed.compile_cache[compile_key](
            o_partial,
            lse_partial,
            out_bshd,
            lse_bsh,
            None,
            None,
            None,
            None,
            None,
            current_stream,
        )

    # Keep split_offsets alive through the combine launch on the same stream.
    _ = split_offsets
    # Combine writes its native BSHD/BSH layout. Return BHSD/BHS views without D2D.
    out = out_bshd.transpose(1, 2)
    lse = lse_bsh.transpose(1, 2)
    return out, lse


_bsa_attn_fwd_blk64_kv_bucketed.compile_cache = get_jit_cache(
    "bsa_fwd_blk64_kv_bucket_combine"
)


def choose_blk64_use_clc(
    q: torch.Tensor,
    block_sparse_num: int,
    q2k_block_nums: Optional[torch.Tensor] = None,
    layout: str = "bhsd",
) -> bool:
    """Select the measured-fastest blk64 scheduler for the common wrapper path.

    The low-level ``torch.ops.bsa_blk64.fwd(..., use_clc)`` API treats
    ``use_clc`` as an explicit scheduler request. This helper is for interface
    defaults: callers can pass ``use_clc=True`` or ``False`` to force a path, or
    leave it as ``None`` to use this shape-based policy.
    """
    if q2k_block_nums is not None and q2k_block_nums.numel() > 0:
        return True

    if layout == "bshd":
        batch, seqlen_q, h, _ = q.shape
    else:
        assert layout == "bhsd", f"layout must be 'bhsd' or 'bshd', got {layout!r}"
        batch, h, seqlen_q, _ = q.shape

    num_m_blocks = (seqlen_q + 63) // 64
    large_long_topk = num_m_blocks >= 8192 and block_sparse_num >= 512
    if large_long_topk:
        return True

    if h == 1:
        return False

    total_tiles = batch * h * num_m_blocks
    enough_tiles = num_m_blocks >= 128 and total_tiles >= 512
    light_tile = block_sparse_num <= (64 if h == 2 else 128)
    return enough_tiles and light_tile


def choose_blk64_cutedsl_use_clc(
    q: torch.Tensor,
    block_sparse_num: int,
    q2k_block_nums: Optional[torch.Tensor] = None,
    layout: str = "bhsd",
) -> bool:
    """Select the measured-fastest scheduler for the blk64 CuTeDSL backend."""
    if q2k_block_nums is not None and q2k_block_nums.numel() > 0:
        return True

    if layout == "bshd":
        batch, seqlen_q, h, _ = q.shape
    else:
        assert layout == "bhsd", f"layout must be 'bhsd' or 'bshd', got {layout!r}"
        batch, h, seqlen_q, _ = q.shape

    num_m_blocks = (seqlen_q + 63) // 64
    total_tiles = batch * h * num_m_blocks
    return num_m_blocks >= 128 and total_tiles >= 512


def bsa_attn_fwd_blk64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sizes: torch.Tensor,
    q2k_block_nums: torch.Tensor,
    softmax_scale: Optional[float] = None,
    layout: str = "bhsd",
    use_clc: Optional[bool] = None,
    kv_splits: int | str = 1,
    block_sparse_num: Optional[int] = None,
):
    """BSA forward attention (blk64 backend).

    SM90 supports fp16/bf16, MHA/GQA/MQA, and QK/V dimensions 64, 96, or 128.
    SM120 uses the CuTe DSL blk64 fwd path. The SM100/SM110 blk64 AOT path
    remains constrained by the built extension.
    The blk64 kernel consumes BHSD tensors directly. The default layout is
    therefore "bhsd" for the customer zero-copy path. BSHD callers are still
    supported by an explicit layout="bshd" conversion at the wrapper boundary.

    Args:
        q, k, v: (B, H, S, D) if layout="bhsd" or (B, S, H, D) if layout="bshd"
        q2k_block_index: (B, H, Q_tiles, max_kv) int32
        block_sizes: (num_kv_blocks,) int32
        q2k_block_nums: (B, H, Q_tiles) int32
        softmax_scale: default 1/sqrt(D)
        layout: "bhsd" (default, zero-copy) or "bshd" (converted to BHSD).
        use_clc: True forces the SM100 CLC persistent scheduler path, False
            forces the SingleTileScheduler path, and None (default) uses the
            interface's shape-based auto policy.
        kv_splits: number of KV buckets per Q block on SM100. kv_splits=1 keeps
            the legacy single-kernel fwd path; kv_splits>1 uses pre-schedule,
            partial attention, and combine. Pass "auto" to choose splits from
            the estimated K/V working set and device L2 cache size.
        block_sparse_num: fixed number of valid KV blocks per Q block when
            q2k_block_nums is empty. Defaults to q2k_block_index.shape[-1].
    """
    assert q.dtype == torch.bfloat16, "blk64 requires bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4

    arch = _get_device_arch()
    auto_kv_splits = isinstance(kv_splits, str)
    if auto_kv_splits:
        assert kv_splits == "auto", "kv_splits string value must be 'auto'"
        kv_splits_i = 1
    else:
        kv_splits_i = int(kv_splits)
        assert kv_splits_i >= 1, "kv_splits must be >= 1"
    out_bshd = None
    if layout == "bshd":
        if arch // 10 == 9:
            out_bshd = torch.empty(
                q.shape[0],
                q.shape[1],
                q.shape[2],
                v.shape[-1],
                dtype=q.dtype,
                device=q.device,
            )
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
    else:
        assert layout == "bhsd", f"layout must be 'bhsd' or 'bshd', got {layout!r}"

    if arch // 10 == 9:
        assert q.size(3) in (64, 96, 128), "SM90 blk64 fwd supports QK dim 64, 96, or 128"
        assert v.size(3) in (64, 96, 128), "SM90 blk64 fwd supports value dim 64, 96, or 128"
        assert q.size(1) % k.size(1) == 0, "num_q_heads must be divisible by num_kv_heads"
        assert (
            q.stride(3) == 1 and k.stride(3) == 1 and v.stride(3) == 1
        ), "SM90 blk64 fwd requires head_dim (axis 3) to be stride-1 / innermost"
    else:
        assert q.size(3) == 128, "blk64 requires D=128"
        assert (
            q.stride(3) == 1 and k.stride(3) == 1 and v.stride(3) == 1
        ), "SM100/SM120 blk64 fwd requires head_dim (axis 3) to be stride-1 / innermost"

    seqlen_q = q.size(2)

    if softmax_scale is None:
        softmax_scale = q.size(3) ** -0.5

    if arch // 10 == 9:
        assert kv_splits_i == 1, "kv_splits is only supported by the SM100 blk64 path"
        block_sizes_sm90 = None if block_sizes is None or block_sizes.numel() == 0 else block_sizes
        block_nums_sm90 = (
            None if q2k_block_nums is None or q2k_block_nums.numel() == 0 else q2k_block_nums
        )
        fixed_block_sparse_num = (
            q2k_block_index.shape[-1] if block_sparse_num is None else int(block_sparse_num)
        )
        assert fixed_block_sparse_num <= q2k_block_index.shape[-1], (
            "block_sparse_num must be <= q2k_block_index.shape[-1]"
        )
        fixed_block_sparse_num = fixed_block_sparse_num if block_nums_sm90 is None else 0
        out, lse = _bsa_attn_fwd_sm90_blk64(
            q,
            k,
            v,
            q2k_block_index,
            fixed_block_sparse_num,
            block_sizes=block_sizes_sm90,
            q2k_block_nums=block_nums_sm90,
            softmax_scale=softmax_scale,
            out=out_bshd.transpose(1, 2) if out_bshd is not None else None,
        )
        return (out_bshd, lse) if out_bshd is not None else (out, lse)

    if arch // 10 == 12:
        assert kv_splits_i == 1, "kv_splits is only supported by the SM100 blk64 path"
        block_sizes_sm120 = None if block_sizes is None or block_sizes.numel() == 0 else block_sizes
        block_nums_sm120 = (
            None if q2k_block_nums is None or q2k_block_nums.numel() == 0 else q2k_block_nums
        )
        fixed_block_sparse_num = (
            q2k_block_index.shape[-1] if block_sparse_num is None else int(block_sparse_num)
        )
        assert fixed_block_sparse_num <= q2k_block_index.shape[-1], (
            "block_sparse_num must be <= q2k_block_index.shape[-1]"
        )
        fixed_block_sparse_num = fixed_block_sparse_num if block_nums_sm120 is None else 0
        out, lse = _bsa_attn_fwd_sm120_blk64(
            q,
            k,
            v,
            q2k_block_index,
            fixed_block_sparse_num,
            block_sizes=block_sizes_sm120,
            q2k_block_nums=block_nums_sm120,
            softmax_scale=softmax_scale,
        )
        if layout == "bshd":
            out = out.transpose(1, 2)
        return out, lse

    # No F.pad: seqlen_q rounding is handled by the kernel's row bounds checks
    # and output TMA descriptor; seqlen_k masking is controlled by block_sizes.
    fixed_block_sparse_num_arg = (
        q2k_block_index.shape[-1] if block_sparse_num is None else int(block_sparse_num)
    )
    assert fixed_block_sparse_num_arg <= q2k_block_index.shape[-1], (
        "block_sparse_num must be <= q2k_block_index.shape[-1]"
    )
    fixed_block_sparse_num = (
        fixed_block_sparse_num_arg
        if q2k_block_nums is None or q2k_block_nums.numel() == 0
        else 0
    )
    if use_clc is None:
        block_nums_for_policy = (
            q2k_block_nums
            if q2k_block_nums is not None and q2k_block_nums.numel() > 0
            else None
        )
        policy_block_sparse_num = (
            fixed_block_sparse_num
            if fixed_block_sparse_num > 0
            else q2k_block_index.shape[-1]
        )
        use_clc = choose_blk64_use_clc(
            q, policy_block_sparse_num, block_nums_for_policy, layout="bhsd"
        )
    if auto_kv_splits:
        kv_splits_i = _sm100_blk64_auto_kv_splits(
            q, q2k_block_index, fixed_block_sparse_num
        )
    _validate_sm100_blk64_int32_bounds(
        q, k, v, q2k_block_index, fixed_block_sparse_num, block_sizes, q2k_block_nums
    )
    if kv_splits_i > 1:
        kv_bucket_use_clc = False if auto_kv_splits else use_clc
        out, lse = _bsa_attn_fwd_blk64_kv_bucketed(
            q,
            k,
            v,
            q2k_block_index,
            fixed_block_sparse_num,
            block_sizes,
            q2k_block_nums,
            softmax_scale,
            kv_splits_i,
            kv_bucket_use_clc,
        )
        assert out.size(2) == seqlen_q and lse.size(2) == seqlen_q
        if layout == "bshd":
            out = out.transpose(1, 2)
        return out, lse

    single_tile_use_clc = False if auto_kv_splits else use_clc
    out, lse = torch.ops.bsa_blk64.fwd(
        q,
        k,
        v,
        q2k_block_index,
        fixed_block_sparse_num,
        block_sizes,
        softmax_scale,
        q2k_block_nums,
        single_tile_use_clc,
    )

    assert out.size(2) == seqlen_q and lse.size(2) == seqlen_q
    if layout == "bshd":
        out = out.transpose(1, 2)
    return out, lse


def bsa_attn_fwd_blk64_cutedsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sizes: torch.Tensor,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    layout: str = "bhsd",
    block_sparse_num: int = 0,
    use_clc: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BSA forward attention through an independent blk64 CuTeDSL kernel class."""
    assert q.dtype == torch.bfloat16, "blk64 CuTeDSL requires bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4

    if layout == "bhsd":
        q_bhsd, k_bhsd, v_bhsd = [maybe_contiguous(t) for t in (q, k, v)]
    else:
        assert layout == "bshd", f"layout must be 'bhsd' or 'bshd', got {layout!r}"
        q_bhsd = q.transpose(1, 2).contiguous()
        k_bhsd = k.transpose(1, 2).contiguous()
        v_bhsd = v.transpose(1, 2).contiguous()

    batch_size, num_head, seqlen_q, head_dim = q_bhsd.shape
    seqlen_k = k_bhsd.shape[2]
    num_head_kv = k_bhsd.shape[1]
    head_dim_v = v_bhsd.shape[-1]

    assert head_dim == 128 and head_dim_v == 128, "blk64 CuTeDSL requires D=DV=128"
    assert num_head == num_head_kv, "blk64 CuTeDSL currently supports MHA only"
    assert k_bhsd.shape == (batch_size, num_head_kv, seqlen_k, head_dim)
    assert v_bhsd.shape == (batch_size, num_head_kv, seqlen_k, head_dim_v)
    assert q2k_block_index.dtype == torch.int32
    assert block_sizes.dtype == torch.int32
    num_q_blocks = (seqlen_q + 63) // 64
    has_variable_block_nums = q2k_block_nums is not None and q2k_block_nums.numel() > 0
    if has_variable_block_nums:
        q2k_block_nums = maybe_contiguous(q2k_block_nums)
        assert q2k_block_nums.dtype == torch.int32
        assert q2k_block_nums.shape == (
            batch_size,
            num_head,
            num_q_blocks,
        ), (
            "q2k_block_nums must be shaped "
            f"(B, H, ceil(S_q/64)); got {tuple(q2k_block_nums.shape)}"
        )
        uniform_block_sparse_num = 0
    else:
        assert block_sparse_num > 0, (
            "block_sparse_num must be provided when q2k_block_nums is None or empty"
        )
        assert q2k_block_index.shape[-1] >= block_sparse_num, (
            f"q2k_block_index last dim ({q2k_block_index.shape[-1]}) must be "
            f">= block_sparse_num ({block_sparse_num})"
        )
        uniform_block_sparse_num = int(block_sparse_num)
        q2k_block_nums = None

    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    out_bhsd = torch.empty(
        (batch_size, num_head, seqlen_q, head_dim_v),
        dtype=q_bhsd.dtype,
        device=q_bhsd.device,
    )
    lse = torch.empty((batch_size, num_head, seqlen_q), dtype=torch.float32, device=q.device)

    dtype = torch2cute_dtype_map[q_bhsd.dtype]
    arch = _get_device_arch()
    has_block_sizes = True
    allow_empty_block_nums = has_variable_block_nums
    sparse_block_size = 64
    qhead_per_kvhead = 1
    tile_m = 64
    tile_n = 256
    use_2cta_instrs = False
    if use_clc is None:
        use_clc_scheduler = choose_blk64_cutedsl_use_clc(
            q_bhsd,
            uniform_block_sparse_num,
            q2k_block_nums if has_variable_block_nums else None,
            layout="bhsd",
        )
    else:
        use_clc_scheduler = bool(use_clc)
    is_persistent = use_clc_scheduler
    pack_gqa = False
    input_layout = "bhsd_native"

    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        pack_gqa,
        tile_m,
        tile_n,
        sparse_block_size,
        arch,
        fa_logging.get_fa_log_level(),
        has_variable_block_nums,
        allow_empty_block_nums,
        has_block_sizes,
        is_persistent,
        use_clc_scheduler,
        input_layout,
    )

    if compile_key not in bsa_attn_fwd_blk64_cutedsl.compile_cache:
        _workaround_cutlass_hash_import_bug()
        q_tensor, k_tensor, v_tensor, o_tensor = [
            _to_cute_tensor(t) for t in (q_bhsd, k_bhsd, v_bhsd, out_bhsd)
        ]
        lse_tensor = _to_cute_tensor(lse, assumed_align=4)
        block_index_tensor = _to_cute_tensor(q2k_block_index)
        block_sizes_tensor = _to_cute_tensor(block_sizes)
        block_nums_tensor = (
            _to_cute_tensor(q2k_block_nums) if has_variable_block_nums else None
        )

        fa_fwd = FlashAttentionForwardSm100Blk64(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            pack_gqa=pack_gqa,
            m_block_size=tile_m,
            n_block_size=tile_n,
            sparse_block_size=sparse_block_size,
            is_persistent=is_persistent,
            use_2cta_instrs=use_2cta_instrs,
            use_clc_scheduler=use_clc_scheduler,
            allow_empty_block_nums=allow_empty_block_nums,
            has_block_sizes=has_block_sizes,
        )

        t0 = time.time()
        bsa_attn_fwd_blk64_cutedsl.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            block_index_tensor,
            block_sizes_tensor,
            uniform_block_sparse_num,
            block_nums_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )
        print(f"Compiled blk64 CuTeDSL fwd in {time.time() - t0:.1f}s")

    if not is_fake_mode():
        with torch.cuda.nvtx.range("bsa_attn_fwd_blk64_cutedsl_kernel"):
            bsa_attn_fwd_blk64_cutedsl.compile_cache[compile_key](
                q_bhsd.detach(),
                k_bhsd.detach(),
                v_bhsd.detach(),
                out_bhsd.detach(),
                lse,
                softmax_scale,
                q2k_block_index.detach(),
                block_sizes.detach(),
                uniform_block_sparse_num,
                q2k_block_nums.detach() if has_variable_block_nums else None,
                current_stream,
            )

    out = out_bhsd if layout == "bhsd" else out_bhsd.transpose(1, 2).contiguous()
    return out, lse


bsa_attn_fwd_blk64_cutedsl.compile_cache = get_jit_cache("bsa_fwd_blk64_cutedsl")


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
    layout: str = "bhsd",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for BSA block-sparse attention (SM90/SM100, non-causal, non-varlen).

    Args:
        q: Query tensor. Shape is (batch, num_heads, seqlen_q, head_dim)
            when layout="bhsd" (default), or (batch, seqlen_q, num_heads,
            head_dim) when layout="bshd".
        k: Key tensor. Shape follows q layout, using num_heads_kv and seqlen_k.
        v: Value tensor. Shape follows q layout, using num_heads_kv, seqlen_k,
            and head_dim_v.
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
        layout: "bhsd" (default) or "bshd". Output follows the same layout as input.
    """
    assert layout in ("bhsd", "bshd"), f"layout must be 'bhsd' or 'bshd', got {layout!r}"
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    if layout == "bhsd":
        batch_size, num_head, seqlen_q, head_dim = q.shape
        batch_k, num_head_kv, seqlen_k, head_dim_k = k.shape
        batch_v, num_head_kv_v, seqlen_k_v, head_dim_v = v.shape
    else:
        batch_size, seqlen_q, num_head, head_dim = q.shape
        batch_k, seqlen_k, num_head_kv, head_dim_k = k.shape
        batch_v, seqlen_k_v, num_head_kv_v, head_dim_v = v.shape

    assert batch_k == batch_size and batch_v == batch_size
    assert seqlen_k_v == seqlen_k and num_head_kv_v == num_head_kv
    assert head_dim_k == head_dim
    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"

    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v)), "inputs must be on CUDA device"

    arch = _get_device_arch()
    assert arch // 10 in [9, 10, 11, 12], "BSA only supports SM90/SM100/SM110/SM120"
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
        if arch // 10 == 12:
            assert block_sparse_num >= 1, (
                f"block_sparse_num={block_sparse_num} must be >= 1 on SM120"
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

    if arch // 10 == 9:
        assert q.dtype in (torch.float16, torch.bfloat16), "SM90 blk64 fwd supports fp16/bf16"
        assert head_dim in (64, 96, 128), "SM90 blk64 fwd supports QK dim 64, 96, or 128"
        assert head_dim_v in (64, 96, 128), "SM90 blk64 fwd supports value dim 64, 96, or 128"
        assert num_head % num_head_kv == 0, "num_q_heads must be divisible by num_kv_heads"
        assert seqlen_q % SM90_FWD_BLOCK_SIZE == 0, (
            "SM90 blk64 fwd requires seqlen_q to be a multiple of 64"
        )
        num_q_blocks_sm90 = seqlen_q // SM90_FWD_BLOCK_SIZE
        assert q2k_block_index.shape[:3] == (batch_size, num_head, num_q_blocks_sm90), (
            f"SM90 blk64 fwd expects q2k_block_index shape prefix "
            f"{(batch_size, num_head, num_q_blocks_sm90)}, got {tuple(q2k_block_index.shape[:3])}"
        )
        if out is None:
            out_shape = (
                (batch_size, num_head, seqlen_q, head_dim_v)
                if layout == "bhsd"
                else (batch_size, seqlen_q, num_head, head_dim_v)
            )
            out = torch.empty(out_shape, dtype=q.dtype, device=q.device)
        else:
            expected_out_shape = (
                (batch_size, num_head, seqlen_q, head_dim_v)
                if layout == "bhsd"
                else (batch_size, seqlen_q, num_head, head_dim_v)
            )
            assert out.shape == expected_out_shape
            assert out.dtype == q.dtype and out.is_cuda
            assert out.stride(-1) == 1, "SM90 blk64 fwd requires output head_dim to be contiguous"
        if layout == "bhsd":
            q_bhsd, k_bhsd, v_bhsd = q, k, v
        else:
            q_bhsd, k_bhsd, v_bhsd = (
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            )
        out_bhsd = out if layout == "bhsd" else out.transpose(1, 2)
        _out_bhsd, lse_sm90 = _bsa_attn_fwd_sm90_blk64(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            q2k_block_index,
            block_sparse_num,
            block_sizes=block_sizes,
            q2k_block_nums=q2k_block_nums,
            softmax_scale=softmax_scale,
            out=out_bhsd,
        )
        if lse is not None:
            lse.copy_(lse_sm90)
        else:
            lse = lse_sm90
        return out, lse

    if arch // 10 == 12:
        assert q.dtype in (torch.float16, torch.bfloat16), "SM120 blk64 fwd supports fp16/bf16"
        assert head_dim == 128, "SM120 blk64 fwd currently requires QK dim 128"
        assert head_dim_v == 128, "SM120 blk64 fwd currently requires value dim 128"
        assert num_head % num_head_kv == 0, "num_q_heads must be divisible by num_kv_heads"
        num_q_blocks_sm120 = _ceil_div_int(seqlen_q, SM120_FWD_BLOCK_SIZE)
        assert q2k_block_index.shape[:3] == (batch_size, num_head, num_q_blocks_sm120), (
            f"SM120 blk64 fwd expects q2k_block_index shape prefix "
            f"{(batch_size, num_head, num_q_blocks_sm120)}, got {tuple(q2k_block_index.shape[:3])}"
        )
        if out is None:
            out_shape = (
                (batch_size, num_head, seqlen_q, head_dim_v)
                if layout == "bhsd"
                else (batch_size, seqlen_q, num_head, head_dim_v)
            )
            out = torch.empty(out_shape, dtype=q.dtype, device=q.device)
        else:
            expected_out_shape = (
                (batch_size, num_head, seqlen_q, head_dim_v)
                if layout == "bhsd"
                else (batch_size, seqlen_q, num_head, head_dim_v)
            )
            assert out.shape == expected_out_shape
            assert out.dtype == q.dtype and out.is_cuda
            assert out.stride(-1) == 1, "SM120 blk64 fwd requires output head_dim to be contiguous"
        if layout == "bhsd":
            q_bhsd, k_bhsd, v_bhsd = q, k, v
        else:
            q_bhsd, k_bhsd, v_bhsd = (
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            )
        out_bhsd = out if layout == "bhsd" else out.transpose(1, 2)
        block_sizes_sm120 = None if block_sizes is None or block_sizes.numel() == 0 else block_sizes
        block_nums_sm120 = (
            None if q2k_block_nums is None or q2k_block_nums.numel() == 0 else q2k_block_nums
        )
        fixed_block_sparse_num = block_sparse_num if block_nums_sm120 is None else 0
        _out_bhsd, lse_sm120 = _bsa_attn_fwd_sm120_blk64(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            q2k_block_index,
            fixed_block_sparse_num,
            block_sizes=block_sizes_sm120,
            q2k_block_nums=block_nums_sm120,
            softmax_scale=softmax_scale,
            out=out_bhsd,
        )
        if lse is not None:
            lse.copy_(lse_sm120)
        else:
            lse = lse_sm120
        return out, lse

    qhead_per_kvhead = num_head // num_head_kv

    out_torch_dtype = q.dtype
    device = q.device
    lse_shape = (batch_size, num_head, seqlen_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out_shape = (
            (batch_size, num_head, seqlen_q, head_dim_v)
            if layout == "bhsd"
            else (batch_size, seqlen_q, num_head, head_dim_v)
        )
        out = torch.empty(out_shape, dtype=out_torch_dtype, device=device)
    else:
        expected_out_shape = (
            (batch_size, num_head, seqlen_q, head_dim_v)
            if layout == "bhsd"
            else (batch_size, seqlen_q, num_head, head_dim_v)
        )
        assert out.shape == expected_out_shape

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )

    dtype = torch2cute_dtype_map[q.dtype]

    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )

    has_variable_block_nums = q2k_block_nums is not None
    if layout == "bhsd":
        q_kernel, k_kernel, v_kernel, out_kernel = q, k, v, out
    else:
        q_kernel, k_kernel, v_kernel, out_kernel = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            out.transpose(1, 2),
        )
    bsa_fwd_kernel = BlockSparseAttnForwardSm100Blk128(
        head_dim,
        head_dim_v,
        qhead_per_kvhead=qhead_per_kvhead,
        pack_gqa=pack_gqa,
        allow_empty_block_nums=allow_empty_block_nums and has_variable_block_nums,
        has_block_sizes=has_block_sizes,
    )

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        lse is None,
        bsa_fwd_kernel.m_block_size,
        bsa_fwd_kernel.n_block_size,
        bsa_fwd_kernel.pack_gqa,
        arch,
        bsa_fwd_kernel.use_2cta_instrs,
        bsa_fwd_kernel.use_clc_scheduler,
        bsa_fwd_kernel.is_persistent,
        fa_logging.get_fa_log_level(),
        has_variable_block_nums,
        allow_empty_block_nums and has_variable_block_nums,
        has_block_sizes,
        "bhsd_kernel_boundary",
    )

    if compile_key not in bsa_attn_fwd.compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [
            _to_cute_tensor_with_dynamic_modes(t, dynamic_modes=(0, 1, 2))
            for t in (q_kernel, k_kernel, v_kernel, out_kernel)
        ]
        lse_tensor = (
            _to_cute_tensor_with_dynamic_modes(
                lse,
                dynamic_modes=(0, 1, 2),
                assumed_align=4,
            )
            if lse is not None
            else None
        )
        block_index_tensor = _to_cute_tensor_with_dynamic_modes(
            q2k_block_index,
            dynamic_modes=(0, 1, 2, 3),
        )
        block_sizes_tensor = (
            _to_cute_tensor_with_dynamic_modes(block_sizes, dynamic_modes=0)
            if has_block_sizes
            else None
        )
        block_nums_tensor = (
            _to_cute_tensor_with_dynamic_modes(q2k_block_nums, dynamic_modes=(0, 1, 2))
            if has_variable_block_nums
            else None
        )

        bsa_attn_fwd.compile_cache[compile_key] = cute.compile(
            bsa_fwd_kernel,
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
                q_kernel.detach(),
                k_kernel.detach(),
                v_kernel.detach(),
                out_kernel.detach(),
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
    bucket_size_blocks: Optional[int] = None,
    layout: str = "bhsd",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for BSA block-sparse attention.

    Paired with ``bsa_attn_fwd`` (or the blk64 C++ fwd), this recomputes
    dQ, dK, dV from stored ``out``/``lse`` and the upstream ``dout`` gradient.

    Args:
        dout: Upstream gradient w.r.t. ``out``. Shape follows ``layout``.
        q, k, v: Forward inputs. Shape is (batch, num_heads, seqlen, head_dim)
            when ``layout="bhsd"`` (default), or (batch, seqlen, num_heads,
            head_dim) when ``layout="bshd"``.
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
        bucket_size_blocks: Optional number of Q blocks per bucketed k2q CSR
            group. ``None`` selects the architecture default.
        layout: "bhsd" (default) or "bshd". Output gradients follow the same
            layout as the inputs.

    Returns:
        (dq, dk, dv): Gradients w.r.t. q, k, v in the same layout as the inputs.

    Notes:
        * Only ``head_dim == 128``, bf16, MHA (num_heads == num_heads_kv) is
          supported. No GQA/MQA, no causal/local, no varlen.
        * SM90 uses blk64. SM100/SM110 supports blk64 and blk128; blk128 routes
          to FA4's SM100 128x128 backward kernel with BSA block-sparse metadata.
    """
    assert layout in ("bhsd", "bshd"), f"layout must be 'bhsd' or 'bshd', got {layout!r}"
    q, k, v, out, dout = [maybe_contiguous(t) for t in (q, k, v, out, dout)]
    lse = maybe_contiguous(lse)

    assert q.dtype == torch.bfloat16, "bwd only supports bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
    assert lse.dtype == torch.float32
    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v, out, dout, lse))

    if layout == "bhsd":
        batch_size, num_heads, seqlen_q, head_dim = q.shape
        batch_k, num_heads_kv, seqlen_k, head_dim_k = k.shape
        batch_v, num_heads_v, seqlen_v, head_dim_v = v.shape
        q_bwd, k_bwd, v_bwd, out_bwd, dout_bwd = q, k, v, out, dout
        dq_bwd, dk_bwd, dv_bwd = dq, dk, dv
    else:
        batch_size, seqlen_q, num_heads, head_dim = q.shape
        batch_k, seqlen_k, num_heads_kv, head_dim_k = k.shape
        batch_v, seqlen_v, num_heads_v, head_dim_v = v.shape
        q_bwd, k_bwd, v_bwd, out_bwd, dout_bwd = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            out.transpose(1, 2),
            dout.transpose(1, 2),
        )
        dq_bwd = dq.transpose(1, 2) if dq is not None else None
        dk_bwd = dk.transpose(1, 2) if dk is not None else None
        dv_bwd = dv.transpose(1, 2) if dv is not None else None

    arch = _get_device_arch()
    assert arch // 10 in [9, 10, 11], "BSA bwd only supports SM90/SM100/SM110"
    if arch // 10 == 9:
        bwd_head_dim = SM90_BWD_HEAD_DIM
        sparse_block_size = SM90_BWD_SPARSE_BLOCK_SIZE
        assert head_dim == bwd_head_dim, f"sm90 bwd only supports head_dim={bwd_head_dim}, got {head_dim}"
    else:
        bwd_head_dim = SM100_BWD_HEAD_DIM
        sparse_block_size = _infer_sm100_bwd_sparse_block_size(
            batch_size=batch_size,
            num_heads=num_heads,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            q2k_block_index=q2k_block_index,
            block_sizes=block_sizes,
        )
        assert bwd_head_dim == SM100_BLK128_BWD_HEAD_DIM
        # blk128 bwd path now supports head_dim in {64, 128}; blk64 bwd path
        # still only supports head_dim=128. Force blk128 path for D=64.
        if head_dim == 64:
            assert sparse_block_size == SM100_BLK128_BWD_SPARSE_BLOCK_SIZE, (
                "head_dim=64 bwd only supports the SM100 blk128 path; pass "
                "q2k_block_index sized to sparse_block_size=128"
            )
        else:
            assert head_dim == bwd_head_dim, (
                f"sm100 bwd only supports head_dim in {{64, {bwd_head_dim}}}, got {head_dim}"
            )
    assert num_heads == num_heads_kv, "bwd does not support GQA/MQA"
    assert batch_k == batch_size and batch_v == batch_size
    assert num_heads_v == num_heads_kv
    assert seqlen_v == seqlen_k
    assert head_dim_k == head_dim and head_dim_v == head_dim
    expected_q_shape = (
        (batch_size, num_heads, seqlen_q, head_dim)
        if layout == "bhsd"
        else (batch_size, seqlen_q, num_heads, head_dim)
    )
    expected_kv_shape = (
        (batch_size, num_heads, seqlen_k, head_dim)
        if layout == "bhsd"
        else (batch_size, seqlen_k, num_heads, head_dim)
    )
    assert q.shape == out.shape == dout.shape == expected_q_shape
    assert k.shape == v.shape == expected_kv_shape
    if dq is not None:
        assert dq.shape == expected_q_shape and dq.dtype == q.dtype
        if not is_fake_mode():
            assert dq.is_cuda
    if dk is not None:
        assert dk.shape == expected_kv_shape and dk.dtype == k.dtype
        if not is_fake_mode():
            assert dk.is_cuda
    if dv is not None:
        assert dv.shape == expected_kv_shape and dv.dtype == v.dtype
        if not is_fake_mode():
            assert dv.is_cuda
    assert lse.shape == (batch_size, num_heads, seqlen_q)

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
    else:
        assert q2k_block_index.shape[-1] >= block_sparse_num, (
            f"q2k_block_index last dim ({q2k_block_index.shape[-1]}) must be >= "
            f"block_sparse_num ({block_sparse_num})"
        )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if arch // 10 != 9 and sparse_block_size == SM100_BLK128_BWD_SPARSE_BLOCK_SIZE:
        if bucket_size_blocks is None or bucket_size_blocks <= 0:
            bucket_size_blocks = sm100_blk128_bwd_default_bucketed_k2q_size_blocks(
                num_q_blocks,
                num_heads,
            )
        bucketed_k2q_offsets, bucketed_k2q_indices, _num_q_groups, _max_k2q_rows_per_group = (
            _build_bucketed_k2q_csr(
                q2k_block_index,
                block_sparse_num,
                num_kv_blocks,
                bucket_size_blocks=bucket_size_blocks,
                q2k_block_nums=q2k_block_nums,
            )
        )
        dq_out, dk_out, dv_out = bsa_sm100_blk128_bwd_bucketed_k2q_csr(
            dout_bwd,
            q_bwd,
            k_bwd,
            v_bwd,
            out_bwd,
            lse,
            bucketed_k2q_offsets,
            bucketed_k2q_indices,
            softmax_scale=softmax_scale,
            dq=dq_bwd,
            dk=dk_bwd,
            dv=dv_bwd,
        )
        if layout == "bshd":
            return (
                dq_out.transpose(1, 2),
                dk_out.transpose(1, 2),
                dv_out.transpose(1, 2),
            )
        return dq_out, dk_out, dv_out

    if bucket_size_blocks is None:
        bucket_size_blocks = (
            sm90_bwd_auto_bucketed_k2q_size_blocks(num_q_blocks)
            if arch // 10 == 9
            else sm100_bwd_auto_bucketed_k2q_size_blocks(num_q_blocks)
        )

    dq_out, dk_out, dv_out = _bsa_attn_bwd_bucketed_k2q_csr(
        dout_bwd,
        q_bwd,
        k_bwd,
        v_bwd,
        out_bwd,
        lse,
        q2k_block_index,
        block_sparse_num,
        block_sizes=block_sizes,
        q2k_block_nums=q2k_block_nums,
        softmax_scale=softmax_scale,
        dq=dq_bwd,
        dk=dk_bwd,
        dv=dv_bwd,
        bucket_size_blocks=bucket_size_blocks,
    )
    if layout == "bshd":
        return (
            dq_out.transpose(1, 2),
            dk_out.transpose(1, 2),
            dv_out.transpose(1, 2),
        )
    return dq_out, dk_out, dv_out


@triton.jit
def _bucketed_k2q_count_edges_kernel(
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
    bucket_size_blocks: tl.constexpr,
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

    q_group = q // bucket_size_blocks
    q2k_base = q2k_index + b * idx_b_s + h * idx_h_s + q * idx_q_s
    count_base = ((b * num_heads + h) * num_q_groups + q_group) * num_kv_blocks

    for i in tl.range(0, max_k):
        if i < n:
            kv = tl.load(q2k_base + i * idx_k_s)
            if (kv >= 0) & (kv < num_kv_blocks):
                tl.atomic_add(counts + count_base + kv, 1, sem="relaxed")


@triton.jit
def _bucketed_k2q_count_edges_fixed_vec_kernel(
    counts,
    q2k_index,
    idx_b_s: tl.constexpr,
    idx_h_s: tl.constexpr,
    idx_q_s: tl.constexpr,
    idx_k_s: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_blocks: tl.constexpr,
    num_q_groups: tl.constexpr,
    bucket_size_blocks: tl.constexpr,
    block_sparse_num: tl.constexpr,
    num_k_tiles: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    qk_tile = tl.program_id(2)
    q = qk_tile // num_k_tiles
    k_tile = qk_tile - q * num_k_tiles

    offs = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < block_sparse_num
    q_group = q // bucket_size_blocks
    q2k_base = q2k_index + b * idx_b_s + h * idx_h_s + q * idx_q_s
    count_base = ((b * num_heads + h) * num_q_groups + q_group) * num_kv_blocks

    kv = tl.load(q2k_base + offs * idx_k_s, mask=mask, other=0)
    tl.atomic_add(counts + count_base + kv, 1, sem="relaxed", mask=mask)


@triton.jit
def _bucketed_k2q_local_offsets_kernel(
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
def _bucketed_k2q_finalize_offsets_kernel(
    local_offsets,
    group_totals,
    bucketed_k2q_offsets,
    cursors,
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
    k2q_offset_base = (bh_group_base + g) * (num_kv_blocks + 1)
    cursor_base = (bh_group_base + g) * num_kv_blocks

    for kv in tl.range(0, num_kv_blocks):
        offset = base + tl.load(local_offsets + local_base + kv)
        tl.store(bucketed_k2q_offsets + k2q_offset_base + kv, offset)
        tl.store(cursors + cursor_base + kv, offset)
    tl.store(
        bucketed_k2q_offsets + k2q_offset_base + num_kv_blocks,
        base + tl.load(local_offsets + local_base + num_kv_blocks),
    )


@triton.jit
def _bucketed_k2q_scatter_q_indices_kernel(
    cursors,
    bucketed_k2q_indices,
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
    bucket_size_blocks: tl.constexpr,
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

    q_group = q // bucket_size_blocks
    q2k_base = q2k_index + b * idx_b_s + h * idx_h_s + q * idx_q_s
    cursor_base = ((b * num_heads + h) * num_q_groups + q_group) * num_kv_blocks
    k2q_indices_base = (b * num_heads + h) * max_edges_per_bh

    for i in tl.range(0, max_k):
        if i < n:
            kv = tl.load(q2k_base + i * idx_k_s)
            if (kv >= 0) & (kv < num_kv_blocks):
                pos = tl.atomic_add(cursors + cursor_base + kv, 1, sem="relaxed")
                tl.store(bucketed_k2q_indices + k2q_indices_base + pos, q)


@triton.jit
def _bucketed_k2q_scatter_q_indices_fixed_vec_kernel(
    cursors,
    bucketed_k2q_indices,
    q2k_index,
    idx_b_s: tl.constexpr,
    idx_h_s: tl.constexpr,
    idx_q_s: tl.constexpr,
    idx_k_s: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_blocks: tl.constexpr,
    num_q_groups: tl.constexpr,
    bucket_size_blocks: tl.constexpr,
    max_edges_per_bh: tl.constexpr,
    block_sparse_num: tl.constexpr,
    num_k_tiles: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    qk_tile = tl.program_id(2)
    q = qk_tile // num_k_tiles
    k_tile = qk_tile - q * num_k_tiles

    offs = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < block_sparse_num
    q_group = q // bucket_size_blocks
    q2k_base = q2k_index + b * idx_b_s + h * idx_h_s + q * idx_q_s
    cursor_base = ((b * num_heads + h) * num_q_groups + q_group) * num_kv_blocks
    k2q_indices_base = (b * num_heads + h) * max_edges_per_bh

    kv = tl.load(q2k_base + offs * idx_k_s, mask=mask, other=0)
    pos = tl.atomic_add(cursors + cursor_base + kv, 1, sem="relaxed", mask=mask)
    tl.store(bucketed_k2q_indices + k2q_indices_base + pos, q, mask=mask)


def _build_bucketed_k2q_csr(
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    num_kv_blocks: int,
    *,
    bucket_size_blocks: int = 1152,
    q2k_block_nums: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build bucketed k2q CSR on GPU for the backward path.

    The returned offsets have shape ``(B, H, num_q_groups, num_kv_blocks + 1)``.
    For each ``(b, h, q_group, kv_block)``, the corresponding slice in
    ``bucketed_k2q_indices`` stores the attending Q block ids.
    """
    assert q2k_block_index.dtype == torch.int32
    assert q2k_block_index.is_cuda
    B, H, num_q_blocks, max_kv = q2k_block_index.shape
    G = bucket_size_blocks
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
    max_k2q_rows_per_group = num_kv_blocks

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
    bucketed_k2q_offsets = torch.empty_like(local_offsets)
    cursors = torch.empty_like(counts)
    bucketed_k2q_indices = torch.empty(
        (B, H, max_edges), dtype=torch.int32, device=q2k_block_index.device
    )

    grid_q = (B, H, num_q_blocks)
    if has_variable_nums:
        _bucketed_k2q_count_edges_kernel[grid_q](
            counts,
            q2k_block_index,
            nums,
            q2k_block_index.stride(0),
            q2k_block_index.stride(1),
            q2k_block_index.stride(2),
            q2k_block_index.stride(3),
            nums.stride(0),
            nums.stride(1),
            nums.stride(2),
            H,
            num_kv_blocks,
            num_q_groups,
            G,
            max_k,
            int(block_sparse_num),
            has_variable_nums,
        )
    else:
        block_k = 1024
        num_k_tiles = triton.cdiv(max_k, block_k)
        grid_qk = (B, H, num_q_blocks * num_k_tiles)
        _bucketed_k2q_count_edges_fixed_vec_kernel[grid_qk](
            counts,
            q2k_block_index,
            q2k_block_index.stride(0),
            q2k_block_index.stride(1),
            q2k_block_index.stride(2),
            q2k_block_index.stride(3),
            H,
            num_kv_blocks,
            num_q_groups,
            G,
            int(block_sparse_num),
            num_k_tiles,
            BLOCK_K=block_k,
        )

    grid_group = (B, H, num_q_groups)
    _bucketed_k2q_local_offsets_kernel[grid_group](
        counts,
        local_offsets,
        group_totals,
        H,
        num_kv_blocks,
        num_q_groups,
    )
    _bucketed_k2q_finalize_offsets_kernel[grid_group](
        local_offsets,
        group_totals,
        bucketed_k2q_offsets,
        cursors,
        H,
        num_kv_blocks,
        num_q_groups,
    )

    if has_variable_nums:
        _bucketed_k2q_scatter_q_indices_kernel[grid_q](
            cursors,
            bucketed_k2q_indices,
            q2k_block_index,
            nums,
            q2k_block_index.stride(0),
            q2k_block_index.stride(1),
            q2k_block_index.stride(2),
            q2k_block_index.stride(3),
            nums.stride(0),
            nums.stride(1),
            nums.stride(2),
            H,
            num_kv_blocks,
            num_q_groups,
            G,
            max_edges,
            max_k,
            int(block_sparse_num),
            has_variable_nums,
        )
    else:
        _bucketed_k2q_scatter_q_indices_fixed_vec_kernel[grid_qk](
            cursors,
            bucketed_k2q_indices,
            q2k_block_index,
            q2k_block_index.stride(0),
            q2k_block_index.stride(1),
            q2k_block_index.stride(2),
            q2k_block_index.stride(3),
            H,
            num_kv_blocks,
            num_q_groups,
            G,
            max_edges,
            int(block_sparse_num),
            num_k_tiles,
            BLOCK_K=block_k,
        )

    return bucketed_k2q_offsets, bucketed_k2q_indices, num_q_groups, max_k2q_rows_per_group


def _bsa_attn_bwd_bucketed_k2q_csr(
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
    bucket_size_blocks: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bucketed k2q CSR backward pass for BSA block-sparse attention.

    This has the same tensor contract as :func:`bsa_attn_bwd`, but builds a
    compact bucketed k2q CSR task layout and runs the blk64 backward
    kernel. Task construction is performed on GPU with Triton on every
    call, so this path is suitable when the sparse pattern changes each
    backward.
    """
    q, k, v, out, dout = [maybe_contiguous(t) for t in (q, k, v, out, dout)]
    lse = maybe_contiguous(lse)

    assert q.dtype == torch.bfloat16, "bucketed k2q CSR bwd only supports bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
    assert lse.dtype == torch.float32
    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v, out, dout, lse))

    batch_size, num_heads, seqlen_q, head_dim = q.shape
    num_heads_kv, seqlen_k = k.shape[1], k.shape[2]
    arch = _get_device_arch()
    assert arch // 10 in [9, 10, 11], "BSA bucketed k2q CSR bwd only supports SM90/SM100/SM110"
    if arch // 10 == 9:
        bwd_head_dim = SM90_BWD_HEAD_DIM
        sparse_block_size = SM90_BWD_SPARSE_BLOCK_SIZE
    else:
        bwd_head_dim = SM100_BWD_HEAD_DIM
        sparse_block_size = SM100_BLK64_BWD_SPARSE_BLOCK_SIZE
    assert head_dim == bwd_head_dim
    assert num_heads == num_heads_kv, "bucketed k2q CSR bwd does not support GQA/MQA"
    assert k.shape == v.shape == (batch_size, num_heads, seqlen_k, head_dim)
    assert out.shape == (batch_size, num_heads, seqlen_q, head_dim)
    assert dout.shape == out.shape
    assert lse.shape == (batch_size, num_heads, seqlen_q)

    num_q_blocks = (seqlen_q + sparse_block_size - 1) // sparse_block_size
    num_kv_blocks = (seqlen_k + sparse_block_size - 1) // sparse_block_size
    if bucket_size_blocks is None or bucket_size_blocks <= 0:
        bucket_size_blocks = (
            sm90_bwd_default_bucketed_k2q_size_blocks(num_q_blocks)
            if arch // 10 == 9
            else sm100_bwd_default_bucketed_k2q_size_blocks(num_q_blocks)
        )

    assert q2k_block_index.dtype == torch.int32
    assert q2k_block_index.shape[:3] == (batch_size, num_heads, num_q_blocks)
    if q2k_block_nums is not None:
        q2k_block_nums = maybe_contiguous(q2k_block_nums)
        assert q2k_block_nums.dtype == torch.int32
        assert q2k_block_nums.shape == (batch_size, num_heads, num_q_blocks)

    bucketed_k2q_offsets, bucketed_k2q_indices, _num_q_groups, _max_k2q_rows_per_group = (
        _build_bucketed_k2q_csr(
            q2k_block_index,
            block_sparse_num,
            num_kv_blocks,
            bucket_size_blocks=bucket_size_blocks,
            q2k_block_nums=q2k_block_nums,
        )
    )

    has_block_sizes = block_sizes is not None
    if not has_block_sizes:
        variable_block_sizes = torch.empty(
            (1, 1), dtype=torch.int32, device=q.device
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

    if arch // 10 == 9:
        if dq is None:
            dq = torch.empty_like(q)
        if dk is None:
            dk = torch.empty_like(k)
        if dv is None:
            dv = torch.empty_like(v)

        block_sizes_sm90 = variable_block_sizes if has_block_sizes else None

        dtype = torch2cute_dtype_map[q.dtype]
        workspace = _empty_bwd_workspace_with_zeroed_accum(
            batch_size=batch_size,
            num_heads=num_heads,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            head_dim=head_dim,
            round_q_to=BlockSparseAttnBackwardSm90Blk64.tile_m,
            round_k_to=BlockSparseAttnBackwardSm90Blk64.tile_n,
            round_d_to=32,
            zero_dq_accum=False,
            device=q.device,
        )

        current_stream = (
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            if is_fake_mode()
            else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        )
        bwd_kernel = BlockSparseAttnBackwardSm90Blk64(dtype, head_dim, head_dim)
        problem_shape = (seqlen_q, seqlen_k, head_dim, (num_heads, batch_size))

        compile_key = (
            "sm90_bucketed_k2q",
            q.dtype,
            head_dim,
            num_heads,
            bucketed_k2q_offsets.shape[2],
            block_sizes_sm90 is not None,
            bwd_kernel.dQaccum_stage,
            bwd_kernel.PdS_stage,
            bwd_kernel.Q_stage,
            _tensor_compile_key(dout),
            _tensor_compile_key(out),
            _tensor_compile_key(q),
            _tensor_compile_key(k),
            _tensor_compile_key(v),
            _tensor_compile_key(dq),
            _tensor_compile_key(dk),
            _tensor_compile_key(dv),
            _tensor_compile_key(lse),
        )

        if compile_key not in _bsa_attn_bwd_bucketed_k2q_csr.compile_cache:
            _bsa_attn_bwd_bucketed_k2q_csr.compile_cache[compile_key] = cute.compile(
                bwd_kernel,
                problem_shape,
                _to_sm90_bwd_cute_tensor(dout),
                _to_sm90_bwd_cute_tensor(out),
                _to_sm90_bwd_cute_tensor(q),
                _to_sm90_bwd_cute_tensor(k),
                _to_sm90_bwd_cute_tensor(v),
                _to_sm90_bwd_cute_tensor(lse, assumed_align=4),
                _to_sm90_bwd_cute_tensor(dq),
                _to_sm90_bwd_cute_tensor(dk),
                _to_sm90_bwd_cute_tensor(dv),
                _to_sm90_bwd_cute_tensor(bucketed_k2q_offsets, assumed_align=4),
                _to_sm90_bwd_cute_tensor(bucketed_k2q_indices, assumed_align=4),
                (
                    _to_sm90_bwd_cute_tensor(block_sizes_sm90, assumed_align=4)
                    if block_sizes_sm90 is not None
                    else None
                ),
                _to_sm90_bwd_cute_tensor(workspace),
                Float32(softmax_scale),
                current_stream,
                options="--enable-tvm-ffi",
            )

        if not is_fake_mode():
            _bsa_attn_bwd_bucketed_k2q_csr.compile_cache[compile_key](
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
                bucketed_k2q_offsets,
                bucketed_k2q_indices,
                block_sizes_sm90,
                workspace,
                softmax_scale,
                current_stream,
            )

        return dq, dk, dv

    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)

    workspace = _empty_bwd_workspace_with_zeroed_accum(
        batch_size=batch_size,
        num_heads=num_heads,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        head_dim=head_dim,
        round_q_to=8,
        round_k_to=8,
        round_d_to=8,
        zero_dq_accum=True,
        device=q.device,
    )

    problem_shape = (seqlen_q, seqlen_k, head_dim, (num_heads, batch_size))
    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )

    compile_key = (
        "sm100_bucketed_k2q",
        q.dtype,
        head_dim,
        sparse_block_size,
        arch,
        has_block_sizes,
        fa_logging.get_fa_log_level(),
        _tensor_layout_compile_key(dout),
        _tensor_layout_compile_key(out),
        _tensor_layout_compile_key(q),
        _tensor_layout_compile_key(k),
        _tensor_layout_compile_key(v),
        _tensor_layout_compile_key(dq),
        _tensor_layout_compile_key(dk),
        _tensor_layout_compile_key(dv),
        _tensor_layout_compile_key(lse),
    )

    def convert_to_cute_tensor(t: torch.Tensor, enable_tvm_ffi: bool = True) -> cute.Tensor:
        return (
            from_dlpack(t.detach(), assumed_align=16, enable_tvm_ffi=enable_tvm_ffi)
            .mark_layout_dynamic()
            .mark_compact_shape_dynamic(
                mode=3, stride_order=t.dim_order(), divisibility=128
            )
        )

    if compile_key not in _bsa_attn_bwd_bucketed_k2q_csr.compile_cache:
        dO_t = convert_to_cute_tensor(dout)
        O_t = convert_to_cute_tensor(out)
        Q_t = convert_to_cute_tensor(q)
        K_t = convert_to_cute_tensor(k)
        V_t = convert_to_cute_tensor(v)
        dQ_t = convert_to_cute_tensor(dq)
        dK_t = convert_to_cute_tensor(dk)
        dV_t = convert_to_cute_tensor(dv)
        LSE_t = _to_cute_tensor(lse, leading_dim=2)
        bucketed_k2q_offsets_t = _to_cute_tensor(bucketed_k2q_offsets, leading_dim=3)
        bucketed_k2q_indices_t = _to_cute_tensor(bucketed_k2q_indices, leading_dim=2)
        var_bs_t = _to_cute_tensor(variable_block_sizes, leading_dim=1)
        ws_t = _to_cute_tensor(workspace, fully_dynamic=True)

        bwd_kernel = BlockSparseAttnBackwardSm100Blk64(
            sparse_block_size=sparse_block_size,
            has_block_sizes=has_block_sizes,
            full_kv_blocks=False,
        )

        _bsa_attn_bwd_bucketed_k2q_csr.compile_cache[compile_key] = cute.compile(
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
            bucketed_k2q_offsets_t,
            bucketed_k2q_indices_t,
            var_bs_t,
            ws_t,
            softmax_scale,
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        with torch.cuda.nvtx.range("bsa_attn_bwd_bucketed_k2q_csr_kernel"):
            _bsa_attn_bwd_bucketed_k2q_csr.compile_cache[compile_key](
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
                bucketed_k2q_offsets,
                bucketed_k2q_indices,
                variable_block_sizes,
                workspace,
                softmax_scale,
                current_stream,
            )

    return dq, dk, dv


_bsa_attn_bwd_bucketed_k2q_csr.compile_cache = get_jit_cache("bsa_bwd_bucketed_k2q_csr")
