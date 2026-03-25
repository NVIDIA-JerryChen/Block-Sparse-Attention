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
from utils.cache_utils import get_jit_cache
from utils.testing import is_fake_mode

from csrc.fwd.sm100 import utils
from utils import fa_logging
from csrc.fwd.sm100.cute_dsl_utils import to_cute_tensor
from csrc.fwd.sm100.flash_fwd_sm100 import FlashAttentionForwardSm100

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


def bsa_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: torch.Tensor,
    q2k_block_nums: Optional[torch.Tensor] = None,
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
        q2k_block_nums: Per-(batch, head, q_block) number of KV blocks to attend to,
            (batch, num_heads, num_q_blocks) int32, each value even and >= 2.
            When None, uses fixed block_sparse_num for all Q blocks.
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
    )

    if compile_key not in bsa_attn_fwd.compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out)
        ]
        lse_tensor = to_cute_tensor(lse, assumed_align=4) if lse is not None else None
        block_index_tensor = to_cute_tensor(q2k_block_index)
        block_sizes_tensor = to_cute_tensor(block_sizes)
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
                block_sizes.detach(),
                block_sparse_num,
                q2k_block_nums.detach() if has_variable_block_nums else None,
                current_stream,
            )

    return out, lse


bsa_attn_fwd.compile_cache = get_jit_cache("bsa_fwd")
