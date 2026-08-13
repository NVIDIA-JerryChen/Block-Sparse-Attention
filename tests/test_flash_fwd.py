"""BSA SM100 Forward Kernel — Test / Benchmark / Profile

Usage:
    python tests/test_flash_fwd.py              # quick correctness test
    python tests/test_flash_fwd.py benchmark    # performance benchmark
    python tests/test_flash_fwd.py profile      # single fwd for ncu
"""

import os
import sys
import math

# Keep direct script execution (`python tests/test_flash_fwd.py`) working.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from block_sparse_attention import (
    bsa_attn_fwd,
    bsa_attn_interface as bsa_interface,
)

# BSA_BLK env var: "64", "128", or "64,128" (default). Controls which blk sizes to test.
_BSA_BLK = os.environ.get("BSA_BLK", "64,128")
_BLK_SIZES = [int(x) for x in _BSA_BLK.split(",")]

from block_sparse_attention.utils.testing import attention_ref
from block_sparse_attention.utils.bench_utils import flops
from block_sparse_attention.utils.benchmark import benchmark_forward
from block_sparse_attention.bsa_attn_interface import (
    _sm100_blk64_requires_int64_kv_strides,
)
from block_sparse_attention.csrc.fwd.sm100_blk128.bsa_fwd_sm100 import (
    BlockSparseAttnForwardSm100Blk128,
)


# ============== Block-sparse helpers ==============

def make_dense_block_sparse_args(batch_size, seqlen_q, seqlen_k, nheads, blk_m=128, blk_n=128, device="cuda"):
    """Create block-sparse args equivalent to dense (full) attention.  For benchmark/profile.

    All block_sizes = blk_n (full blocks).  Requires seqlen_k % (2*blk_n) == 0.
    Returns q2k_block_index, block_sparse_num, block_sizes.
    """
    num_q_blocks = (seqlen_q + blk_m - 1) // blk_m
    num_kv_blocks = (seqlen_k + blk_n - 1) // blk_n

    assert num_kv_blocks >= 2 and num_kv_blocks % 2 == 0, (
        f"num_kv_blocks={num_kv_blocks} must be even and >= 2 for dense-equivalent test. "
        f"Adjust seqlen_k to be a multiple of {2 * blk_n}."
    )
    block_sparse_num = num_kv_blocks

    indices = torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
    q2k_block_index = indices.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
        batch_size, nheads, num_q_blocks, num_kv_blocks
    ).contiguous()

    block_sizes = torch.full((num_kv_blocks,), blk_n, dtype=torch.int32, device=device)
    return q2k_block_index, block_sparse_num, block_sizes


def make_random_block_sparse_args(batch_size, seqlen_q, seqlen_k, nheads, blk_m=128, blk_n=128, device="cuda"):
    """Create random block-sparse args for correctness testing.

    Random block_sparse_num (even, >= 2), random per-(batch,head,q_block) KV block
    selection, and random block_sizes in [1, blk_n].
    Returns q2k_block_index, block_sparse_num, block_sizes.
    """
    num_q_blocks = (seqlen_q + blk_m - 1) // blk_m
    num_kv_blocks = (seqlen_k + blk_n - 1) // blk_n
    assert num_kv_blocks >= 2, f"num_kv_blocks={num_kv_blocks} must be >= 2"

    max_even = num_kv_blocks if num_kv_blocks % 2 == 0 else num_kv_blocks - 1
    # Minimum bsn=4 to avoid pre-existing kernel issue with bsn=2 + large batch + small hdim
    min_bsn = min(4, max_even)
    # blk64: phantom padding allows any count >= 1. Step by 2 for variety.
    # blk128: step by 2 as before.
    step = 2
    possible_counts = list(range(min_bsn, max_even + 1, step))
    block_sparse_num = possible_counts[torch.randint(len(possible_counts), (1,)).item()]

    q2k_block_index = torch.empty(batch_size, nheads, num_q_blocks, block_sparse_num,
                                   dtype=torch.int32, device=device)
    for b in range(batch_size):
        for h in range(nheads):
            for m in range(num_q_blocks):
                perm = torch.randperm(num_kv_blocks, device=device)[:block_sparse_num]
                q2k_block_index[b, h, m] = perm.to(torch.int32)

    block_sizes = torch.randint(1, blk_n + 1, (num_kv_blocks,), dtype=torch.int32, device=device)
    last_block_actual = seqlen_k - (num_kv_blocks - 1) * blk_n
    if last_block_actual < blk_n:
        block_sizes[-1] = min(block_sizes[-1].item(), last_block_actual)

    return q2k_block_index, block_sparse_num, block_sizes


def make_random_variable_block_sparse_args(batch_size, seqlen_q, seqlen_k, nheads, blk_m=128, blk_n=128, device="cuda"):
    """Create random block-sparse args with per-(batch, head, q_block) variable block counts.

    Each Q block gets a random block_sparse_num in [0, num_kv_blocks].
    q2k_block_index is padded to num_kv_blocks with zeros (unused entries).
    Returns q2k_block_index, q2k_block_nums, block_sizes.
    """
    num_q_blocks = (seqlen_q + blk_m - 1) // blk_m
    num_kv_blocks = (seqlen_k + blk_n - 1) // blk_n
    assert num_kv_blocks >= 1, f"num_kv_blocks={num_kv_blocks} must be >= 1"

    # blk64: any count >= 1 (phantom padding handles arbitrary counts), no zero yet
    if blk_n == 64:
        possible_counts = list(range(1, num_kv_blocks + 1))
    else:
        possible_counts = list(range(0, num_kv_blocks + 1))

    # Per-(b, h, m) random block count
    q2k_block_nums = torch.empty(batch_size, nheads, num_q_blocks, dtype=torch.int32, device=device)
    q2k_block_index = torch.zeros(batch_size, nheads, num_q_blocks, num_kv_blocks,
                                   dtype=torch.int32, device=device)
    for b in range(batch_size):
        for h in range(nheads):
            for m in range(num_q_blocks):
                bsn = possible_counts[torch.randint(len(possible_counts), (1,)).item()]
                q2k_block_nums[b, h, m] = bsn
                perm = torch.randperm(num_kv_blocks, device=device)[:bsn]
                q2k_block_index[b, h, m, :bsn] = perm.to(torch.int32)

    block_sizes = torch.randint(1, blk_n + 1, (num_kv_blocks,), dtype=torch.int32, device=device)
    last_block_actual = seqlen_k - (num_kv_blocks - 1) * blk_n
    if last_block_actual < blk_n:
        block_sizes[-1] = min(block_sizes[-1].item(), last_block_actual)

    return q2k_block_index, q2k_block_nums, block_sizes


def block_sparse_to_attn_bias(q2k_block_index, block_sparse_num, block_sizes,
                               seqlen_q, seqlen_k, blk_m=128, blk_n=128,
                               q2k_block_nums=None):
    """Convert block-sparse args to additive attention bias for reference.

    Returns attn_bias (batch, nheads, seqlen_q, seqlen_k) float32:
        0.0 for attended positions, -inf for masked.

    When q2k_block_nums is provided, each (batch, head, q_block) uses its own
    block count instead of the fixed block_sparse_num.
    """
    batch_size, nheads, num_q_blocks, max_kv_blocks = q2k_block_index.shape
    num_kv_blocks = block_sizes.shape[0]
    device = q2k_block_index.device

    col_idx = torch.arange(blk_n, device=device)
    block_valid = col_idx.unsqueeze(0) < block_sizes.unsqueeze(1)  # (num_kv_blocks, blk_n)

    if q2k_block_nums is None:
        # Fixed block_sparse_num: all entries up to block_sparse_num are valid
        block_attended = torch.zeros(batch_size, nheads, num_q_blocks, num_kv_blocks,
                                      dtype=torch.bool, device=device)
        block_attended.scatter_(3, q2k_block_index[..., :block_sparse_num].long(), True)
    else:
        # Variable per-(b,h,m) block counts
        block_attended = torch.zeros(batch_size, nheads, num_q_blocks, num_kv_blocks,
                                      dtype=torch.bool, device=device)
        for b in range(batch_size):
            for h in range(nheads):
                for m in range(num_q_blocks):
                    bsn = q2k_block_nums[b, h, m].item()
                    indices = q2k_block_index[b, h, m, :bsn].long()
                    block_attended[b, h, m].scatter_(0, indices, True)

    # (batch, nheads, num_q_blocks, num_kv_blocks, blk_n) -> clip to seqlen_k
    token_valid = (block_attended.unsqueeze(-1) & block_valid).reshape(
        batch_size, nheads, num_q_blocks, -1
    )[..., :seqlen_k]

    attn_bias = torch.full((batch_size, nheads, seqlen_q, seqlen_k), float("-inf"),
                           device=device, dtype=torch.float32)
    for m in range(num_q_blocks):
        q_start = m * blk_m
        q_end = min((m + 1) * blk_m, seqlen_q)
        attn_bias[:, :, q_start:q_end] = torch.where(
            token_valid[:, :, m : m + 1], 0.0, float("-inf"),
        )

    return attn_bias


def pack_gqa_attn_bias(attn_bias_kv, nheads, qhead_per_kvhead, seqlen_q):
    """Unpack packed-GQA attention bias to per-Q-head layout.

    attn_bias_kv: (bs, nheads_kv, seqlen_q_packed, seqlen_k)
      where seqlen_q_packed = seqlen_q * qhead_per_kvhead, and packed row r
      maps to seq_pos = r // qhead_per_kvhead, head_off = r % qhead_per_kvhead.

    Returns: (bs, nheads, seqlen_q, seqlen_k)
    """
    bs, nheads_kv, seqlen_q_packed, seqlen_k = attn_bias_kv.shape
    # Build index: for each packed row r, compute seq_pos
    r = torch.arange(seqlen_q_packed, device=attn_bias_kv.device)
    seq_pos = r // qhead_per_kvhead  # (seqlen_q_packed,)
    head_off = r % qhead_per_kvhead  # (seqlen_q_packed,)
    # Expand nheads_kv to nheads: h_q = h_kv * qhead_per_kvhead + head_off
    # For each h_kv, select rows where head_off matches and gather by seq_pos
    attn_bias = torch.full((bs, nheads, seqlen_q, seqlen_k), float("-inf"),
                           device=attn_bias_kv.device, dtype=attn_bias_kv.dtype)
    valid = seq_pos < seqlen_q
    for h_off in range(qhead_per_kvhead):
        mask = (head_off == h_off) & valid  # (seqlen_q_packed,)
        src_rows = r[mask]        # packed row indices with this head_off
        dst_rows = seq_pos[mask]  # corresponding seq positions
        for h_kv in range(nheads_kv):
            h_q = h_kv * qhead_per_kvhead + h_off
            attn_bias[:, h_q, dst_rows] = attn_bias_kv[:, h_kv, src_rows]
    return attn_bias


# ============== Correctness helpers ==============

def _test_single(bs, seqlen_q, seqlen_k, nheads, nheads_kv, d, dtype=torch.bfloat16,
                  use_variable_block_nums=False, use_block_sizes=True,
                  blk_m=128, blk_n=128, use_clc=False):
    """Run a single correctness test with random block-sparse pattern.

    blk_m/blk_n: block sizes. blk_n=64 routes to the blk64 CuTe DSL kernel.
    use_block_sizes=False exercises the HasBlockSizes=false kernel path
    (block_sizes passed as None / empty tensor; all KV tokens treated as valid).
    Requires seqlen_k % blk_n == 0 so every KV block is fully populated.
    use_clc: for blk64 only, toggle the CLC persistent scheduler path
    (ignored when blk_n=128; blk128 has its own scheduling knob).
    """
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    is_blk64 = (blk_m == 64 and blk_n == 64)
    arch_major = torch.cuda.get_device_capability()[0]
    is_sm90_blk64 = is_blk64 and arch_major == 9
    is_sm120_blk64 = is_blk64 and arch_major == 12

    # blk64 constraints: skip unsupported configurations
    if is_blk64:
        if is_sm90_blk64:
            if d not in (64, 96, 128):
                pytest.skip("SM90 blk64 supports d in {64, 96, 128}")
            if dtype not in (torch.bfloat16, torch.float16):
                pytest.skip("SM90 blk64 supports bf16/fp16")
        elif is_sm120_blk64:
            if d != 128:
                pytest.skip("SM120 blk64 requires d=128")
            if dtype not in (torch.bfloat16, torch.float16):
                pytest.skip("SM120 blk64 supports bf16/fp16")
        else:
            if nheads_kv != nheads:
                pytest.skip("SM100 blk64 does not support GQA/MQA")
            if d != 128:
                pytest.skip("SM100 blk64 requires d=128")

    q_ref = torch.randn(bs, seqlen_q, nheads, d, device=device, dtype=dtype).requires_grad_()
    k_ref = torch.randn(bs, seqlen_k, nheads_kv, d, device=device, dtype=dtype).requires_grad_()
    v_ref = torch.randn(bs, seqlen_k, nheads_kv, d, device=device, dtype=dtype).requires_grad_()
    q = q_ref.detach().requires_grad_()
    k = k_ref.detach().requires_grad_()
    v = v_ref.detach().requires_grad_()

    qhead_per_kvhead = nheads // nheads_kv
    pack_gqa = (
        qhead_per_kvhead > 1
        and (128 % qhead_per_kvhead == 0)
        and not is_sm90_blk64
        and not is_sm120_blk64
    )
    nheads_q2k = nheads_kv if pack_gqa else nheads
    seqlen_q_q2k = seqlen_q * qhead_per_kvhead if pack_gqa else seqlen_q

    q2k_block_nums = None
    if use_variable_block_nums:
        q2k_block_index, q2k_block_nums, block_sizes = make_random_variable_block_sparse_args(
            bs, seqlen_q_q2k, seqlen_k, nheads_q2k, blk_m=blk_m, blk_n=blk_n, device=device,
        )
        block_sparse_num = 0
    else:
        q2k_block_index, block_sparse_num, block_sizes = make_random_block_sparse_args(
            bs, seqlen_q_q2k, seqlen_k, nheads_q2k, blk_m=blk_m, blk_n=blk_n, device=device,
        )

    # HasBlockSizes=false path: kernel receives None/empty block_sizes,
    # reference must assume every KV block is fully populated (blk_n tokens).
    if use_block_sizes:
        block_sizes_kernel = block_sizes
    else:
        assert seqlen_k % blk_n == 0, (
            f"use_block_sizes=False requires seqlen_k % blk_n == 0 "
            f"(got seqlen_k={seqlen_k}, blk_n={blk_n})"
        )
        num_kv_blocks = seqlen_k // blk_n
        block_sizes = torch.full((num_kv_blocks,), blk_n, dtype=torch.int32, device=device)
        block_sizes_kernel = None

    attn_bias_kv = block_sparse_to_attn_bias(
        q2k_block_index, block_sparse_num, block_sizes, seqlen_q_q2k, seqlen_k,
        blk_m=blk_m, blk_n=blk_n, q2k_block_nums=q2k_block_nums,
    )
    attn_bias = pack_gqa_attn_bias(attn_bias_kv, nheads, qhead_per_kvhead, seqlen_q) if pack_gqa else attn_bias_kv

    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, None, None, attn_bias=attn_bias, causal=False)
    out_pt, _ = attention_ref(
        q_ref, k_ref, v_ref, None, None, attn_bias=attn_bias, causal=False,
        upcast=False, reorder_ops=True,
    )

    # Reference LSE: logsumexp of attention scores (float32)
    softmax_scale_ref = 1.0 / math.sqrt(d)
    k_ref_expanded = k_ref.float().repeat_interleave(qhead_per_kvhead, dim=2) if qhead_per_kvhead > 1 else k_ref.float()
    scores_ref = torch.einsum("bthd,bshd->bhts", q_ref.float() * softmax_scale_ref, k_ref_expanded)
    scores_ref = scores_ref + attn_bias
    lse_ref = torch.logsumexp(scores_ref, dim=-1)  # (bs, nheads, seqlen_q)

    out_ref = torch.nan_to_num(out_ref, nan=0.0)
    out_pt = torch.nan_to_num(out_pt, nan=0.0)

    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    rtol = 2

    if is_blk64:
        softmax_scale = 1.0 / math.sqrt(d)
        bn_arg = (
            q2k_block_nums
            if q2k_block_nums is not None
            else torch.empty(0, dtype=torch.int32, device=device)
        )
        bs_arg = (
            block_sizes_kernel
            if block_sizes_kernel is not None
            else torch.empty(0, dtype=torch.int32, device=device)
        )
        # blk64 kernel consumes BHSD natively; convert from BSHD test tensors
        # at the boundary and convert the BHSD output back to BSHD for reference.
        q_bhsd = q.transpose(1, 2).contiguous()
        k_bhsd = k.transpose(1, 2).contiguous()
        v_bhsd = v.transpose(1, 2).contiguous()
        out_bhsd, lse_bhsd = bsa_attn_fwd(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            q2k_block_index,
            block_sparse_num,
            bs_arg,
            q2k_block_nums=bn_arg,
            softmax_scale=softmax_scale,
            return_lse=True,
            sparse_block_size=64,
            use_clc=use_clc,
        )
        out_from_blk64 = out_bhsd.transpose(1, 2).contiguous()
        if is_sm90_blk64 or is_sm120_blk64:
            out, lse = bsa_attn_fwd(
                q,
                k,
                v,
                q2k_block_index,
                block_sparse_num,
                block_sizes_kernel,
                q2k_block_nums=q2k_block_nums,
                return_lse=True,
                layout="bshd",
                sparse_block_size=64,
            )
            assert torch.equal(out_from_blk64, out)
            assert torch.equal(lse_bhsd, lse)
        else:
            out, lse = out_from_blk64, lse_bhsd
    else:
        out, lse = bsa_attn_fwd(
            q,
            k,
            v,
            q2k_block_index,
            block_sparse_num,
            block_sizes_kernel,
            q2k_block_nums=q2k_block_nums,
            return_lse=True,
            layout="bshd",
            sparse_block_size=128,
        )
    out = torch.nan_to_num(out, nan=0.0)

    kernel_diff = (out - out_ref).abs().max().item()
    pt_diff = (out_pt - out_ref).abs().max().item()
    tol = rtol * pt_diff + fwd_atol
    passed = kernel_diff <= tol

    # LSE validation
    lse_diff = (lse - lse_ref).abs()
    # Mask out -inf positions (empty rows) for comparison
    finite_mask = lse_ref.isfinite()
    lse_max_diff = lse_diff[finite_mask].max().item() if finite_mask.any() else 0.0
    lse_tol = 1e-3
    lse_passed = lse_max_diff <= lse_tol
    passed = passed and lse_passed

    blk_str = f" blk={blk_n}" if blk_n != 128 else ""
    mode_str = "var_bsn" if use_variable_block_nums else f"sparse_num={block_sparse_num}"
    mode_str += " no_bs" if not use_block_sizes else ""
    tag = "PASS" if passed else "FAIL"
    print(
        f"  {tag} bs={bs} sq={seqlen_q} sk={seqlen_k} h={nheads}/{nheads_kv} d={d} "
        f"{mode_str}{blk_str}: "
        f"kernel={kernel_diff:.6f} pt={pt_diff:.6f} tol={tol:.6f} "
        f"lse_diff={lse_max_diff:.6f}"
    )
    assert passed, f"kernel_diff={kernel_diff} > tol={tol}, lse_diff={lse_max_diff} > lse_tol={lse_tol}"


def _make_variable_boundary_case(
    block_counts, use_block_sizes, block_size, head_dim=128
):
    """Build deterministic variable-count metadata with live padded entries."""
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 1
    num_heads = 1
    num_q_blocks = len(block_counts)
    capacity = max(block_counts)
    seqlen_q = num_q_blocks * block_size
    seqlen_k = capacity * block_size

    torch.manual_seed(20260812)
    q = torch.randn(
        batch_size,
        num_heads,
        seqlen_q,
        head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        batch_size,
        num_heads,
        seqlen_k,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)

    block_ids = torch.arange(capacity, device=device, dtype=torch.int32)
    q2k_block_index = torch.empty(
        batch_size,
        num_heads,
        num_q_blocks,
        capacity,
        device=device,
        dtype=torch.int32,
    )
    for q_block_idx in range(num_q_blocks):
        # Keep every padded entry valid but distinct so consuming beyond the
        # runtime count changes the numerical result instead of faulting.
        q2k_block_index[0, 0, q_block_idx] = torch.roll(
            block_ids, shifts=q_block_idx
        )
    q2k_block_nums = torch.tensor(
        block_counts, device=device, dtype=torch.int32
    ).view(batch_size, num_heads, num_q_blocks)

    if use_block_sizes:
        block_sizes = block_size - (block_ids * 11) % 31
    else:
        block_sizes = None
    return q, k, v, q2k_block_index, q2k_block_nums, block_sizes


def _reference_variable_case(
    q,
    k,
    v,
    q2k_block_index,
    q2k_block_nums,
    block_sizes,
    block_size,
):
    """Compute an FP32 reference without sanitizing empty-row kernel output."""
    softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    ref_out = torch.zeros_like(q, dtype=torch.float32)
    ref_lse = torch.full(
        q.shape[:3], float("-inf"), device=q.device, dtype=torch.float32
    )
    num_q_blocks = q2k_block_nums.shape[-1]

    for q_block_idx in range(num_q_blocks):
        block_count = int(q2k_block_nums[0, 0, q_block_idx].item())
        if block_count == 0:
            continue
        selected_tokens = []
        for logical_idx in range(block_count):
            kv_block_idx = int(
                q2k_block_index[0, 0, q_block_idx, logical_idx].item()
            )
            valid_tokens = (
                block_size
                if block_sizes is None
                else int(block_sizes[kv_block_idx].item())
            )
            selected_tokens.append(
                torch.arange(
                    kv_block_idx * block_size,
                    kv_block_idx * block_size + valid_tokens,
                    device=q.device,
                )
            )
        token_indices = torch.cat(selected_tokens)
        q_begin = q_block_idx * block_size
        q_end = q_begin + block_size
        q_tile = q[:, :, q_begin:q_end].float()
        k_selected = k.index_select(2, token_indices).float()
        v_selected = v.index_select(2, token_indices).float()
        scores = torch.matmul(q_tile, k_selected.transpose(-1, -2))
        scores *= softmax_scale
        ref_out[:, :, q_begin:q_end] = torch.matmul(
            torch.softmax(scores, dim=-1), v_selected
        )
        ref_lse[:, :, q_begin:q_end] = torch.logsumexp(scores, dim=-1)
    return ref_out, ref_lse


def _assert_variable_fwd_case(
    *,
    block_counts,
    use_block_sizes,
    allow_empty_block_nums,
    use_clc,
    block_size,
    head_dim=128,
    kv_splits=1,
):
    (
        q,
        k,
        v,
        q2k_block_index,
        q2k_block_nums,
        block_sizes,
    ) = _make_variable_boundary_case(
        block_counts, use_block_sizes, block_size, head_dim
    )
    ref_out, ref_lse = _reference_variable_case(
        q,
        k,
        v,
        q2k_block_index,
        q2k_block_nums,
        block_sizes,
        block_size,
    )

    # A conflicting nonzero scalar verifies that runtime per-row counts take
    # precedence. Live entries after each valid prefix verify tail masking.
    out, lse = bsa_attn_fwd(
        q,
        k,
        v,
        q2k_block_index,
        q2k_block_index.shape[-1],
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        allow_empty_block_nums=allow_empty_block_nums,
        return_lse=True,
        sparse_block_size=block_size,
        use_clc=use_clc,
        kv_splits=kv_splits,
    )

    empty_q_blocks = q2k_block_nums[0, 0] == 0
    if empty_q_blocks.any():
        empty_rows = empty_q_blocks.repeat_interleave(block_size)
        assert torch.count_nonzero(out[0, 0, empty_rows]) == 0
        assert torch.isneginf(lse[0, 0, empty_rows]).all()

    torch.testing.assert_close(
        out.float(), ref_out, rtol=3e-2, atol=3e-2
    )
    finite_rows = ref_lse.isfinite()
    torch.testing.assert_close(
        lse[finite_rows], ref_lse[finite_rows], rtol=2e-3, atol=2e-3
    )


# ============== Pytest ==============

@pytest.mark.parametrize(
    "head_dim,head_dim_v,error",
    [
        (192, 128, "QK dim"),
        (128, 192, "value dim"),
    ],
)
def test_sm100_blk128_rejects_unsupported_head_dims(
    head_dim, head_dim_v, error
):
    with pytest.raises(AssertionError, match=error):
        BlockSparseAttnForwardSm100Blk128(head_dim, head_dim_v)


def test_flash_fwd_sm100_blk128_clc_scheduler():
    """Regression test for CTA-wide convergence when consuming CLC responses."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SM100+ required")

    _test_single(
        1,
        128,
        256,
        1,
        1,
        128,
        torch.bfloat16,
        blk_m=128,
        blk_n=128,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("use_variable_block_nums", [False, True])
@pytest.mark.parametrize("use_clc", [False, True])
@pytest.mark.parametrize("blk_n", _BLK_SIZES)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (64, 512),
        (64, 1024),
        (128, 512),
        (128, 1024),
        (256, 512),
        (256, 1024),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ],
)
def test_flash_fwd_sm100(seqlen_q, seqlen_k, d, mha_type, dtype, use_variable_block_nums, use_clc, blk_n):
    arch_major = torch.cuda.get_device_capability()[0]
    if arch_major == 9 and blk_n != 64:
        pytest.skip("SM90 fwd path is blk64 only")
    if arch_major == 9 and use_clc:
        pytest.skip("use_clc only affects SM100 blk64")
    if arch_major == 12 and use_clc:
        pytest.skip("use_clc only affects SM100 blk64")
    # use_clc only affects blk64; skip the duplicate blk128 case.
    if blk_n != 64 and use_clc:
        pytest.skip("use_clc only affects blk64")
    batch_size = 4 if seqlen_k <= 2048 else 2
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else (3 if mha_type == "gqa" else 1)
    blk_m = 64 if blk_n == 64 else 128
    _test_single(batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dtype,
                  use_variable_block_nums=use_variable_block_nums,
                  blk_m=blk_m, blk_n=blk_n, use_clc=use_clc)


# HasBlockSizes=false coverage (independent from main matrix; seqlen_k % blk_n == 0 required).
# 2 × 2 × 3 = 12 testcases per blk, covering both (no_bs, no_vbn) and (no_bs, has_vbn).
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_variable_block_nums", [False, True])
@pytest.mark.parametrize("use_clc", [False, True])
@pytest.mark.parametrize("blk_n", _BLK_SIZES)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [(128, 512), (256, 1024), (1024, 1024)],
)
def test_flash_fwd_sm100_no_block_sizes(seqlen_q, seqlen_k, dtype, use_variable_block_nums, use_clc, blk_n):
    arch_major = torch.cuda.get_device_capability()[0]
    if arch_major == 9 and blk_n != 64:
        pytest.skip("SM90 fwd path is blk64 only")
    if arch_major == 9 and use_clc:
        pytest.skip("use_clc only affects SM100 blk64")
    if arch_major == 12 and use_clc:
        pytest.skip("use_clc only affects SM100 blk64")
    if blk_n != 64 and use_clc:
        pytest.skip("use_clc only affects blk64")
    batch_size = 2
    nheads = 4
    blk_m = 64 if blk_n == 64 else 128
    _test_single(batch_size, seqlen_q, seqlen_k, nheads, nheads, 128, dtype,
                  use_variable_block_nums=use_variable_block_nums,
                  use_block_sizes=False,
                  blk_m=blk_m, blk_n=blk_n, use_clc=use_clc)


def test_sm120_blk64_odd_topk_q_tail_block_sizes():
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120-only coverage")

    torch.manual_seed(2026)
    bs, h, sq, sk, d = 1, 2, 96, 256, 128
    blk = 64
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(bs, h, sq, d, device=device, dtype=dtype)
    k = torch.randn(bs, h, sk, d, device=device, dtype=dtype)
    v = torch.randn(bs, h, sk, d, device=device, dtype=dtype)

    qtiles = (sq + blk - 1) // blk
    indices = torch.tensor([0, 2, 3], device=device, dtype=torch.int32)
    q2k_block_index = indices.view(1, 1, 1, 3).expand(bs, h, qtiles, 3).contiguous()
    q2k_block_nums = torch.empty(0, device=device, dtype=torch.int32)
    block_sizes = torch.tensor([64, 64, 17, 64], device=device, dtype=torch.int32)

    out, lse = bsa_attn_fwd(
        q,
        k,
        v,
        q2k_block_index,
        3,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        return_lse=True,
        sparse_block_size=64,
    )

    cols = list(range(0, 64)) + list(range(128, 145)) + list(range(192, 256))
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q.float(), k[:, :, cols, :].float().transpose(-1, -2)) * scale
    ref_out = torch.matmul(torch.softmax(scores, dim=-1), v[:, :, cols, :].float()).to(dtype)
    ref_lse = torch.logsumexp(scores, dim=-1)

    torch.testing.assert_close(out, ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("seqlen_q", [1, 63, 65])
def test_sm100_blk64_cutedsl_partial_tail_rows(seqlen_q):
    """Cover partial Q tiles in both the producer and split combine."""
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] not in (10, 11)
    ):
        pytest.skip("SM100/SM110 blk64 CuTe DSL test")

    torch.manual_seed(6)
    bs, h, sk, d = 1, 1, 256, 128
    blk = 64
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(bs, h, seqlen_q, d, device=device, dtype=dtype)
    k = torch.randn(bs, h, sk, d, device=device, dtype=dtype)
    v = torch.randn_like(k)
    num_q_blocks = (seqlen_q + blk - 1) // blk
    num_kv_blocks = sk // blk
    q2k_block_index = (
        torch.arange(num_kv_blocks, device=device, dtype=torch.int32)
        .view(1, 1, 1, num_kv_blocks)
        .expand(bs, h, num_q_blocks, num_kv_blocks)
        .contiguous()
    )
    q2k_block_nums = torch.full(
        (bs, h, num_q_blocks),
        num_kv_blocks,
        device=device,
        dtype=torch.int32,
    )
    block_sizes = torch.full(
        (num_kv_blocks,), blk, device=device, dtype=torch.int32
    )

    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    ref_out = torch.matmul(torch.softmax(scores, dim=-1), v.float())
    ref_lse = torch.logsumexp(scores, dim=-1)

    for kv_splits in (1, 2):
        out_buffer = torch.empty_like(q)
        lse_buffer = torch.empty_like(ref_lse)
        out, lse = bsa_attn_fwd(
            q,
            k,
            v,
            q2k_block_index,
            0,
            block_sizes,
            q2k_block_nums=q2k_block_nums,
            softmax_scale=scale,
            return_lse=True,
            out=out_buffer,
            lse=lse_buffer,
            sparse_block_size=64,
            use_clc=False,
            kv_splits=kv_splits,
        )
        assert out is out_buffer
        assert lse is lse_buffer
        torch.testing.assert_close(out.float(), ref_out, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)


def test_sm90_blk64_empty_variable_row():
    """An empty sparse row must produce O=0 and LSE=-inf on SM90."""
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 9
    ):
        pytest.skip("SM90-only coverage")

    torch.manual_seed(7)
    bs, h, sq, sk, d = 1, 1, 128, 128, 128
    blk = 64
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(bs, h, sq, d, device=device, dtype=dtype)
    k = torch.randn(bs, h, sk, d, device=device, dtype=dtype)
    v = torch.randn_like(k)
    q2k_block_index = (
        torch.arange(2, device=device, dtype=torch.int32)
        .view(1, 1, 1, 2)
        .expand(bs, h, 2, 2)
        .contiguous()
    )
    q2k_block_nums = torch.tensor([[[0, 2]]], device=device, dtype=torch.int32)
    block_sizes = torch.full((2,), blk, device=device, dtype=torch.int32)

    out, lse = bsa_attn_fwd(
        q,
        k,
        v,
        q2k_block_index,
        0,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        allow_empty_block_nums=True,
        return_lse=True,
        sparse_block_size=64,
    )

    assert torch.count_nonzero(out[:, :, :blk]) == 0
    assert torch.isneginf(lse[:, :, :blk]).all()
    scale = 1.0 / math.sqrt(d)
    scores = (
        torch.matmul(q[:, :, blk:].float(), k.float().transpose(-1, -2))
        * scale
    )
    ref_out = torch.matmul(torch.softmax(scores, dim=-1), v.float())
    ref_lse = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(
        out[:, :, blk:].float(), ref_out, rtol=3e-2, atol=3e-2
    )
    torch.testing.assert_close(lse[:, :, blk:], ref_lse, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] in (10, 11)
    ),
    reason="SM100/SM110 required",
)
@pytest.mark.parametrize(
    "use_clc,use_block_sizes,allow_empty_block_nums",
    [
        (False, True, True),
        (True, False, True),
        (False, False, False),
        (True, True, False),
    ],
    ids=[
        "static-block-sizes-empty",
        "clc-no-block-sizes-empty",
        "static-no-block-sizes-nonempty",
        "clc-block-sizes-nonempty",
    ],
)
def test_sm100_blk64_variable_count_boundaries(
    use_clc, use_block_sizes, allow_empty_block_nums
):
    """Cover phantom padding boundaries and both empty specializations."""
    block_counts = [1, 2, 3, 4, 7, 8, 9, 15, 16, 17]
    if allow_empty_block_nums:
        block_counts[0] = 0
    _assert_variable_fwd_case(
        block_counts=block_counts,
        use_block_sizes=use_block_sizes,
        allow_empty_block_nums=allow_empty_block_nums,
        use_clc=use_clc,
        block_size=64,
    )


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
    ),
    reason="SM90 required",
)
@pytest.mark.parametrize(
    "use_block_sizes,allow_empty_block_nums",
    [(True, True), (False, False)],
    ids=["block-sizes-empty", "no-block-sizes-nonempty"],
)
def test_sm90_blk64_variable_count_boundaries(
    use_block_sizes, allow_empty_block_nums
):
    """Cover zero, singleton, odd, even, and capacity counts on SM90."""
    block_counts = [1, 2, 3, 7, 8, 9, 16, 17]
    if allow_empty_block_nums:
        block_counts[0] = 0
    _assert_variable_fwd_case(
        block_counts=block_counts,
        use_block_sizes=use_block_sizes,
        allow_empty_block_nums=allow_empty_block_nums,
        use_clc=False,
        block_size=64,
    )


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] in (10, 11)
    ),
    reason="SM100/SM110 required",
)
@pytest.mark.parametrize(
    "kv_splits,use_clc",
    [(2, False), (4, True)],
    ids=["splits2-static", "splits4-clc"],
)
def test_sm100_blk64_split_kv_variable_count_boundaries(kv_splits, use_clc):
    """Exercise even and 8-block-aligned split offset branches on SM100."""
    _assert_variable_fwd_case(
        block_counts=[0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33],
        use_block_sizes=True,
        allow_empty_block_nums=True,
        use_clc=use_clc,
        block_size=64,
        kv_splits=kv_splits,
    )


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
    ),
    reason="SM90 required",
)
@pytest.mark.parametrize("kv_splits", [2, 4], ids=["splits2", "splits4"])
def test_sm90_blk64_split_kv_variable_count_boundaries(kv_splits):
    """Exercise empty splits and aligned split offsets on SM90."""
    _assert_variable_fwd_case(
        block_counts=[0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33],
        use_block_sizes=True,
        allow_empty_block_nums=True,
        use_clc=False,
        block_size=64,
        kv_splits=kv_splits,
    )


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] in (10, 11)
    ),
    reason="SM100/SM110 required",
)
@pytest.mark.parametrize(
    "head_dim,use_block_sizes,allow_empty_block_nums",
    [
        (64, True, True),
        (96, False, False),
        (128, True, False),
    ],
    ids=[
        "d64-block-sizes-empty",
        "d96-no-block-sizes-nonempty",
        "d128-block-sizes-nonempty",
    ],
)
def test_sm100_blk128_variable_count_boundaries(
    head_dim, use_block_sizes, allow_empty_block_nums
):
    """Cover odd-count padding and supported head dimensions on blk128."""
    block_counts = [1, 2, 3, 4, 5, 5]
    if allow_empty_block_nums:
        block_counts[0] = 0
    _assert_variable_fwd_case(
        block_counts=block_counts,
        use_block_sizes=use_block_sizes,
        allow_empty_block_nums=allow_empty_block_nums,
        use_clc=None,
        block_size=128,
        head_dim=head_dim,
    )


# ============== Quick test (make tt) ==============

def run_quick_tests():
    if 128 in _BLK_SIZES:
        print("Quick correctness tests (blk128)")
        print("=" * 70)
        configs_128 = [
            (1, 64, 256, 4, 4, 128),
            (1, 64, 384, 4, 4, 128),
            (1, 64, 512, 4, 4, 128),
            (1, 64, 256, 8, 1, 128),
            (1, 256, 256, 4, 4, 128),
            (1, 128, 640, 4, 4, 128),
            (1, 1024, 1024, 4, 4, 128),
            (1, 2048, 2048, 4, 4, 128),
        ]
        for bs, sq, sk, hq, hk, d in configs_128:
            _test_single(bs, sq, sk, hq, hk, d)

    if 64 in _BLK_SIZES:
        print("-" * 70)
        print("Quick correctness tests (blk64)")
        configs_64 = [
            (1, 64, 512, 4, 4, 128),
            (1, 64, 1024, 4, 4, 128),
            (1, 128, 512, 4, 4, 128),
            (1, 256, 1024, 4, 4, 128),
            (1, 1024, 1024, 4, 4, 128),
            (1, 2048, 2048, 4, 4, 128),
        ]
        for use_clc in (False, True):
            for bs, sq, sk, hq, hk, d in configs_64:
                _test_single(bs, sq, sk, hq, hk, d, blk_m=64, blk_n=64, use_clc=use_clc)

    if 128 in _BLK_SIZES:
        print("-" * 70)
        print("Variable block_sparse_num tests (blk128)")
        var_configs_128 = [
            (1, 64, 512, 4, 4, 128),
            (1, 256, 512, 4, 4, 128),
            (1, 128, 640, 4, 4, 128),
            (1, 64, 256, 8, 1, 128),     # MQA
            (4, 1024, 1024, 6, 6, 128),  # 192 tiles, MHA
            (4, 1024, 1024, 6, 3, 128),  # 192 tiles, GQA
            (2, 2048, 2048, 4, 4, 64),   # 128 tiles, d=64
            (1, 256, 512, 4, 4, 64),
            (1, 256, 512, 4, 4, 96),
        ]
        for bs, sq, sk, hq, hk, d in var_configs_128:
            _test_single(bs, sq, sk, hq, hk, d, use_variable_block_nums=True)

    if 64 in _BLK_SIZES:
        print("-" * 70)
        print("Variable block_sparse_num tests (blk64)")
        var_configs_64 = [
            (1, 64, 512, 4, 4, 128),
            (1, 128, 1024, 4, 4, 128),
            (1, 256, 1024, 4, 4, 128),
            (4, 1024, 1024, 4, 4, 128),
        ]
        for use_clc in (False, True):
            for bs, sq, sk, hq, hk, d in var_configs_64:
                _test_single(bs, sq, sk, hq, hk, d, use_variable_block_nums=True,
                              blk_m=64, blk_n=64, use_clc=use_clc)

    # block_sizes=None path (HasBlockSizes=false): requires seqlen_k % blk_n == 0
    if 128 in _BLK_SIZES:
        print("-" * 70)
        print("No-block_sizes tests (blk128)")
        no_bs_configs_128 = [
            (1, 128, 512, 4, 4, 128),                                 # fixed bsn
            (1, 256, 1024, 4, 4, 128),                                # fixed bsn
            (1, 128, 512, 4, 4, 128, True),                           # var bsn
            (1, 256, 1024, 4, 4, 128, True),                          # var bsn
        ]
        for cfg in no_bs_configs_128:
            bs, sq, sk, hq, hk, d = cfg[:6]
            use_var = cfg[6] if len(cfg) > 6 else False
            _test_single(bs, sq, sk, hq, hk, d, use_variable_block_nums=use_var,
                          use_block_sizes=False)

    if 64 in _BLK_SIZES:
        print("-" * 70)
        print("No-block_sizes tests (blk64)")
        no_bs_configs_64 = [
            (1, 128, 512, 4, 4, 128),                                 # fixed bsn, no-bs, no-vbn
            (1, 256, 1024, 4, 4, 128),                                # fixed bsn, no-bs, no-vbn
            (1, 1024, 1024, 4, 4, 128),                               # fixed bsn, no-bs, no-vbn
            (1, 128, 512, 4, 4, 128, True),                           # var bsn,   no-bs, has-vbn
            (1, 256, 1024, 4, 4, 128, True),                          # var bsn,   no-bs, has-vbn
            (4, 1024, 1024, 4, 4, 128, True),                         # var bsn,   no-bs, has-vbn
        ]
        for use_clc in (False, True):
            for cfg in no_bs_configs_64:
                bs, sq, sk, hq, hk, d = cfg[:6]
                use_var = cfg[6] if len(cfg) > 6 else False
                _test_single(bs, sq, sk, hq, hk, d, use_variable_block_nums=use_var,
                              use_block_sizes=False, blk_m=64, blk_n=64, use_clc=use_clc)

        print("-" * 70)
        print("Interface layout test (blk64)")
        _test_blk64_interface_layouts()

    print("=" * 70)
    print("All quick tests passed.")


def _test_blk64_interface_layouts():
    """Verify default BHSD and explicit BSHD wrapper paths are layout-equivalent."""
    bs, h, sq, sk, d = 1, 4, 256, 512, 128
    blk = 64
    nq = sq // blk
    nkv = sk // blk
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    q_bhsd = torch.randn(bs, h, sq, d, device=device, dtype=dtype)
    k_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)
    v_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)

    q_bshd = q_bhsd.transpose(1, 2).contiguous()
    k_bshd = k_bhsd.transpose(1, 2).contiguous()
    v_bshd = v_bhsd.transpose(1, 2).contiguous()

    q2k_block_index = torch.arange(nkv, device=device, dtype=torch.int32) \
        .view(1, 1, 1, nkv).expand(bs, h, nq, nkv).contiguous()
    q2k_block_nums = torch.full((bs, h, nq), nkv, device=device, dtype=torch.int32)
    block_sizes = torch.full((nkv,), blk, device=device, dtype=torch.int32)

    out_bhsd, lse_bhsd = bsa_attn_fwd(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_block_index,
        0,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        return_lse=True,
        sparse_block_size=64,
    )
    out_bshd, lse_bshd = bsa_attn_fwd(
        q_bshd,
        k_bshd,
        v_bshd,
        q2k_block_index,
        0,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        layout="bshd",
        return_lse=True,
        sparse_block_size=64,
    )

    assert torch.equal(out_bhsd.transpose(1, 2).contiguous(), out_bshd), \
        f"BSHD output mismatch: max diff = {(out_bhsd.transpose(1, 2).contiguous() - out_bshd).abs().max().item()}"
    assert torch.equal(lse_bhsd, lse_bshd), "LSE mismatch"
    print("  PASS default BHSD and explicit BSHD wrapper paths match")


def _make_sm90_split_kv_variable_case():
    """Create a compact variable-count case with only odd active top-k values."""
    bs, h, sq, sk, d = 1, 2, 128, 768, 128
    blk = 64
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(2029)
    q_bhsd = torch.randn(bs, h, sq, d, device=device, dtype=dtype)
    k_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)
    v_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)
    q2k_block_index, q2k_block_nums, block_sizes = (
        make_random_variable_block_sparse_args(
            bs,
            sq,
            sk,
            h,
            blk_m=blk,
            blk_n=blk,
            device=device,
        )
    )

    num_q_blocks = sq // blk
    num_kv_blocks = sk // blk
    max_topk = 7
    q2k_block_nums.copy_(
        torch.arange(1, max_topk + 1, 2, device=device, dtype=torch.int32).view(
            bs, h, num_q_blocks
        )
    )
    for head_idx in range(h):
        for q_block_idx in range(num_q_blocks):
            q2k_block_index[0, head_idx, q_block_idx] = torch.randperm(
                num_kv_blocks, device=device, dtype=torch.int32
            )
    q2k_block_index = q2k_block_index[..., :max_topk].contiguous()
    return q_bhsd, k_bhsd, v_bhsd, q2k_block_index, q2k_block_nums, block_sizes


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
    ),
    reason="SM90 required",
)
@pytest.mark.parametrize("kv_splits", [2, 4], ids=["splits2", "splits4"])
def test_sm90_blk64_split_kv_variable_odd_topk(kv_splits):
    inputs = _make_sm90_split_kv_variable_case()
    q_bhsd, k_bhsd, v_bhsd, q2k_block_index, q2k_block_nums, block_sizes = inputs

    ref_out, ref_lse = bsa_attn_fwd(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_block_index,
        0,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        return_lse=True,
        sparse_block_size=64,
        kv_splits=1,
    )
    out, lse = bsa_attn_fwd(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_block_index,
        0,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        return_lse=True,
        sparse_block_size=64,
        kv_splits=kv_splits,
    )

    torch.testing.assert_close(out, ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)

    if kv_splits == 2:
        out_bshd, lse_bshd = bsa_attn_fwd(
            q_bhsd.transpose(1, 2).contiguous(),
            k_bhsd.transpose(1, 2).contiguous(),
            v_bhsd.transpose(1, 2).contiguous(),
            q2k_block_index,
            0,
            block_sizes,
            q2k_block_nums=q2k_block_nums,
            layout="bshd",
            return_lse=True,
            sparse_block_size=64,
            kv_splits=kv_splits,
        )
        assert out_bshd.is_contiguous()
        torch.testing.assert_close(
            out_bshd.transpose(1, 2), out, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(lse_bshd, lse, rtol=0.0, atol=0.0)

        empty_block_sizes = torch.empty(0, device="cuda", dtype=torch.int32)
        ref_no_bs, ref_lse_no_bs = bsa_attn_fwd(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            q2k_block_index,
            0,
            empty_block_sizes,
            q2k_block_nums=q2k_block_nums,
            return_lse=True,
            sparse_block_size=64,
            kv_splits=1,
        )
        out_no_bs, lse_no_bs = bsa_attn_fwd(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            q2k_block_index,
            0,
            empty_block_sizes,
            q2k_block_nums=q2k_block_nums,
            return_lse=True,
            sparse_block_size=64,
            kv_splits=kv_splits,
        )
        torch.testing.assert_close(out_no_bs, ref_no_bs, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(
            lse_no_bs, ref_lse_no_bs, rtol=2e-3, atol=2e-3
        )


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
    ),
    reason="SM90 required",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sm90_blk64_split_kv_gqa(dtype):
    bs, q_heads, kv_heads, sq, sk = 2, 4, 2, 128, 512
    qk_dim, value_dim = 64, 128
    num_q_blocks = sq // 64
    num_kv_blocks = sk // 64
    topk = 5
    device = "cuda"

    torch.manual_seed(2031)
    q = torch.randn(
        bs, q_heads, sq, qk_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        bs, kv_heads, sk, qk_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        bs, kv_heads, sk, value_dim, device=device, dtype=dtype
    )
    q2k_block_index = torch.empty(
        bs,
        q_heads,
        num_q_blocks,
        topk,
        device=device,
        dtype=torch.int32,
    )
    for batch_idx in range(bs):
        for head_idx in range(q_heads):
            for q_block_idx in range(num_q_blocks):
                q2k_block_index[batch_idx, head_idx, q_block_idx] = torch.randperm(
                    num_kv_blocks, device=device, dtype=torch.int32
                )[:topk]
    q2k_block_nums = torch.full(
        (bs, q_heads, num_q_blocks),
        topk,
        device=device,
        dtype=torch.int32,
    )
    block_sizes = torch.full(
        (num_kv_blocks,), 64, device=device, dtype=torch.int32
    )

    ref_out, ref_lse = bsa_attn_fwd(
        q,
        k,
        v,
        q2k_block_index,
        0,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        return_lse=True,
        sparse_block_size=64,
        kv_splits=1,
    )
    out, lse = bsa_attn_fwd(
        q,
        k,
        v,
        q2k_block_index,
        0,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        return_lse=True,
        sparse_block_size=64,
        kv_splits=2,
    )

    torch.testing.assert_close(out, ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)

    out_bshd, lse_bshd = bsa_attn_fwd(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        q2k_block_index,
        0,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        layout="bshd",
        return_lse=True,
        sparse_block_size=64,
        kv_splits=2,
    )
    assert out_bshd.is_contiguous()
    torch.testing.assert_close(
        out_bshd.transpose(1, 2), out, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(lse_bshd, lse, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
    ),
    reason="SM90 required",
)
def test_sm90_blk64_split_kv_auto_api():
    bs, h, sq, topk, d = 1, 2, 64, 256, 128
    sk = topk * 64
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(2030)
    q_bhsd = torch.randn(bs, h, sq, d, device=device, dtype=dtype)
    k_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)
    v_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)
    q2k_block_index, block_sparse_num, block_sizes = make_dense_block_sparse_args(
        bs,
        sq,
        sk,
        h,
        blk_m=64,
        blk_n=64,
        device=device,
    )
    q2k_block_nums = torch.empty(0, device=device, dtype=torch.int32)

    explicit_out, explicit_lse = bsa_attn_fwd(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_block_index,
        block_sparse_num,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        return_lse=True,
        sparse_block_size=64,
        kv_splits=2,
    )
    auto_out, auto_lse = bsa_attn_fwd(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_block_index,
        block_sparse_num,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        return_lse=True,
        sparse_block_size=64,
        kv_splits="auto",
    )

    torch.testing.assert_close(auto_out, explicit_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(auto_lse, explicit_lse, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("use_clc", [False, True], ids=["single_tile", "clc"])
def test_sm100_blk64_split_kv_matches_single_kernel(use_clc):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    arch_major = torch.cuda.get_device_capability()[0]
    if arch_major not in (10, 11):
        pytest.skip("split-KV CuTe DSL blk64 fwd is SM100/SM110-only")
    bs, h, sq, sk, d = 2, 2, 128, 512, 128
    blk = 64
    nq = sq // blk
    nkv = sk // blk
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(2026)
    q_bhsd = torch.randn(bs, h, sq, d, device=device, dtype=dtype)
    k_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)
    v_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)

    q2k_block_index = torch.arange(nkv, device=device, dtype=torch.int32) \
        .view(1, 1, 1, nkv).expand(bs, h, nq, nkv).contiguous()
    q2k_block_nums = torch.full((bs, h, nq), nkv, device=device, dtype=torch.int32)
    q2k_block_nums[..., 0] = 2
    block_sizes = torch.full((nkv,), blk, device=device, dtype=torch.int32)

    ref_out, ref_lse = bsa_attn_fwd(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_block_index,
        0,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        return_lse=True,
        sparse_block_size=64,
        use_clc=False,
    )
    for kv_splits in (2, 4):
        out, lse = bsa_attn_fwd(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            q2k_block_index,
            0,
            block_sizes,
            q2k_block_nums=q2k_block_nums,
            return_lse=True,
            sparse_block_size=64,
            use_clc=use_clc,
            kv_splits=kv_splits,
        )
        torch.testing.assert_close(out, ref_out, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] in (10, 11)
    ),
    reason="SM100/SM110 required",
)
def test_sm100_blk64_split_kv_clc_persistent_tiles():
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    bs, h, nkv, d = 2, 3, 8, 128
    kv_splits = 3
    nq = max(32, 2 * sm_count // (bs * h * kv_splits) + 1)
    blk = 64
    sq, sk = nq * blk, nkv * blk
    device = "cuda"
    dtype = torch.bfloat16
    assert bs * h * nq * kv_splits > 2 * sm_count

    torch.manual_seed(2029)
    q_bhsd = torch.randn(bs, h, sq, d, device=device, dtype=dtype)
    k_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)
    v_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)
    q2k_block_index = torch.arange(nkv, device=device, dtype=torch.int32) \
        .view(1, 1, 1, nkv).expand(bs, h, nq, nkv).contiguous()
    empty_nums = torch.empty(0, device=device, dtype=torch.int32)
    block_sizes = torch.full((nkv,), blk, device=device, dtype=torch.int32)

    ref_out, ref_lse = bsa_attn_fwd(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_block_index,
        nkv,
        block_sizes,
        q2k_block_nums=empty_nums,
        return_lse=True,
        sparse_block_size=64,
        use_clc=False,
        kv_splits=kv_splits,
    )
    torch.cuda.synchronize()
    out, lse = bsa_attn_fwd(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_block_index,
        nkv,
        block_sizes,
        q2k_block_nums=empty_nums,
        return_lse=True,
        sparse_block_size=64,
        use_clc=True,
        kv_splits=kv_splits,
    )
    torch.cuda.synchronize()

    repeat_out, repeat_lse = bsa_attn_fwd(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_block_index,
        nkv,
        block_sizes,
        q2k_block_nums=empty_nums,
        return_lse=True,
        sparse_block_size=64,
        use_clc=True,
        kv_splits=kv_splits,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out, ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(repeat_out, out, rtol=0.0, atol=0.0)
    torch.testing.assert_close(repeat_lse, lse, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] in (10, 11)
    ),
    reason="SM100/SM110 required",
)
def test_sm100_blk64_cutedsl_large_kv_batch_stride():
    batch, heads, seqlen_q, seqlen_k, head_dim = 2, 1, 64, 512, 128
    kv_batch_stride = 1 << 27
    kv_head_stride = seqlen_k * head_dim
    storage_elems = kv_batch_stride + kv_head_stride

    q = torch.zeros(
        (batch, heads, seqlen_q, head_dim), device="cuda", dtype=torch.bfloat16
    )
    storage = torch.zeros(storage_elems, device="cuda", dtype=torch.bfloat16)
    kv_shape = (batch, heads, seqlen_k, head_dim)
    kv_stride = (kv_batch_stride, kv_head_stride, head_dim, 1)
    wide = torch.as_strided(storage, kv_shape, kv_stride)
    compact = torch.zeros(kv_shape, device="cuda", dtype=torch.bfloat16)

    num_kv_blocks = seqlen_k // 64
    q2k_block_index = (
        torch.arange(num_kv_blocks, device="cuda", dtype=torch.int32)
        .view(1, 1, 1, num_kv_blocks)
        .expand(batch, heads, seqlen_q // 64, num_kv_blocks)
        .contiguous()
    )
    block_sizes = torch.full(
        (num_kv_blocks,), 64, device="cuda", dtype=torch.int32
    )

    def run(k, v):
        return bsa_attn_fwd(
            q,
            k,
            v,
            q2k_block_index,
            num_kv_blocks,
            block_sizes,
            softmax_scale=1.0,
            return_lse=True,
            sparse_block_size=64,
            use_clc=False,
            kv_splits=2,
        )

    # Verify the wide K stride with two analytically distinct softmaxes.
    wide[1, :, seqlen_k // 2 :, 0] = 2.0
    compact[:, :, seqlen_k // 2 :, :].fill_(1.0)
    q[..., 0] = 1.0
    out, lse = run(wide, compact)
    expected_out = torch.tensor(
        [0.5, math.exp(2.0) / (1.0 + math.exp(2.0))], device="cuda"
    )
    expected_lse = torch.tensor(
        [
            math.log(seqlen_k),
            math.log((seqlen_k // 2) * (1.0 + math.exp(2.0))),
        ],
        device="cuda",
    )
    torch.testing.assert_close(
        out.float(),
        expected_out[:, None, None, None].expand_as(out),
        rtol=0.0,
        atol=1e-2,
    )
    torch.testing.assert_close(
        lse, expected_lse[:, None, None].expand_as(lse), rtol=0.0, atol=2e-3
    )

    # Verify the wide V stride with uniform attention and batch-unique values.
    wide[0].fill_(1.0)
    wide[1].fill_(2.0)
    compact.zero_()
    q.zero_()
    out, lse = run(compact, wide)
    expected_out = torch.tensor([1.0, 2.0], device="cuda")
    torch.testing.assert_close(
        out.float(),
        expected_out[:, None, None, None].expand_as(out),
        rtol=0.0,
        atol=1e-2,
    )
    torch.testing.assert_close(
        lse, torch.full_like(lse, math.log(seqlen_k)), rtol=0.0, atol=2e-3
    )


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] in (10, 11)
    ),
    reason="SM100/SM110 required",
)
def test_sm100_blk64_cutedsl_multibatch_kv_tail():
    """Keep non-aligned KV tails isolated across batches."""
    torch.manual_seed(2031)
    batch, heads, seqlen_q, seqlen_k, head_dim = 2, 2, 65, 193, 128
    block_size = 64
    num_q_blocks = (seqlen_q + block_size - 1) // block_size

    q = torch.randn(
        (batch, heads, seqlen_q, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (batch, heads, seqlen_k, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    q2k_block_index = torch.tensor(
        [
            [
                [[0, 1, 3], [1, 2, 3]],
                [[2, 0, 1], [3, 1, 0]],
            ],
            [
                [[3, 2, 0], [0, 2, 1]],
                [[1, 3, 2], [2, 1, 0]],
            ],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    assert q2k_block_index.shape == (batch, heads, num_q_blocks, 3)
    block_sizes = torch.tensor(
        [64, 17, 63, 1], device="cuda", dtype=torch.int32
    )
    empty_block_nums = torch.empty(0, device="cuda", dtype=torch.int32)

    def run(q_arg, k_arg, v_arg, indices_arg):
        return bsa_attn_fwd(
            q_arg,
            k_arg,
            v_arg,
            indices_arg,
            3,
            block_sizes,
            q2k_block_nums=empty_block_nums,
            return_lse=True,
            sparse_block_size=block_size,
            use_clc=False,
            kv_splits=1,
        )

    out, lse = run(q, k, v, q2k_block_index)
    reference = [
        run(
            q[batch_idx : batch_idx + 1],
            k[batch_idx : batch_idx + 1],
            v[batch_idx : batch_idx + 1],
            q2k_block_index[batch_idx : batch_idx + 1],
        )
        for batch_idx in range(batch)
    ]
    ref_out = torch.cat([result[0] for result in reference], dim=0)
    ref_lse = torch.cat([result[1] for result in reference], dim=0)

    torch.testing.assert_close(out, ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)


def test_sm100_blk64_int64_kv_stride_selection():
    def make_meta(batch, stride_b, stride_s=128):
        return torch.empty_strided(
            (batch, 1, 512, 128),
            (stride_b, 512 * stride_s, stride_s, 1),
            dtype=torch.bfloat16,
            device="meta",
        )

    below_limit = make_meta(2, (1 << 27) - 1)
    at_limit = make_meta(2, 1 << 27)
    inactive_batch = make_meta(1, 1 << 27)
    block_at_limit = make_meta(1, 1 << 30, stride_s=1 << 21)
    tail_multibatch = torch.empty(
        (2, 1, 593, 128), dtype=torch.bfloat16, device="meta"
    )
    tail_single_batch = torch.empty(
        (1, 1, 593, 128), dtype=torch.bfloat16, device="meta"
    )
    aligned_multibatch = torch.empty(
        (2, 1, 640, 128), dtype=torch.bfloat16, device="meta"
    )

    assert not _sm100_blk64_requires_int64_kv_strides(below_limit, below_limit)
    assert _sm100_blk64_requires_int64_kv_strides(at_limit, at_limit)
    assert not _sm100_blk64_requires_int64_kv_strides(
        inactive_batch, inactive_batch
    )
    assert _sm100_blk64_requires_int64_kv_strides(block_at_limit, block_at_limit)
    assert _sm100_blk64_requires_int64_kv_strides(
        tail_multibatch, tail_multibatch
    )
    assert _sm100_blk64_requires_int64_kv_strides(
        tail_single_batch, tail_single_batch
    )
    assert not _sm100_blk64_requires_int64_kv_strides(
        aligned_multibatch, aligned_multibatch
    )


def test_unified_blk64_lse_and_preallocated_buffer_contract(monkeypatch):
    q = torch.empty((1, 1, 64, 128), dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    q2k_block_index = torch.zeros((1, 1, 1, 1), dtype=torch.int32)
    generated_out = torch.empty_like(q)
    generated_lse = torch.empty((1, 1, 64), dtype=torch.float32)
    calls = []

    def fake_blk64(*args, out=None, lse=None, **kwargs):
        calls.append((out, lse))
        return (
            generated_out if out is None else out,
            generated_lse if lse is None else lse,
        )

    monkeypatch.setattr(bsa_interface, "_bsa_attn_fwd_blk64", fake_blk64)

    result_out, result_lse = bsa_attn_fwd(
        q,
        k,
        v,
        q2k_block_index,
        1,
        sparse_block_size=64,
    )
    assert result_out is generated_out
    assert result_lse is None

    result_out, result_lse = bsa_attn_fwd(
        q,
        k,
        v,
        q2k_block_index,
        1,
        return_lse=True,
        sparse_block_size=64,
    )
    assert result_out is generated_out
    assert result_lse is generated_lse

    result_out, result_lse = bsa_attn_fwd(
        q.requires_grad_(),
        k,
        v,
        q2k_block_index,
        1,
        sparse_block_size=64,
    )
    assert result_out is generated_out
    assert result_lse is generated_lse
    q.requires_grad_(False)

    out_buffer = torch.empty_like(q)
    lse_buffer = torch.empty_like(generated_lse)
    result_out, result_lse = bsa_attn_fwd(
        q,
        k,
        v,
        q2k_block_index,
        1,
        out=out_buffer,
        lse=lse_buffer,
        sparse_block_size=64,
    )
    assert calls[-1][0] is out_buffer
    assert calls[-1][1] is lse_buffer
    assert result_out is out_buffer
    assert result_lse is lse_buffer

    with pytest.raises(AssertionError, match="pack_gqa"):
        bsa_attn_fwd(
            q,
            k,
            v,
            q2k_block_index,
            1,
            pack_gqa=False,
            sparse_block_size=64,
        )


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] in (10, 11)
    ),
    reason="SM100/SM110 required",
)
def test_sm100_blk64_kv_bucketed_respects_fixed_block_sparse_num():
    bs, h, sq, sk, d = 1, 4, 128, 512, 128
    blk = 64
    nq = sq // blk
    nkv = sk // blk
    topk = nkv // 2
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(2027)
    q_bhsd = torch.randn(bs, h, sq, d, device=device, dtype=dtype)
    k_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)
    v_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)

    used = torch.arange(topk, device=device, dtype=torch.int32)
    tail = torch.arange(topk, nkv, device=device, dtype=torch.int32)
    q2k_compact = used.view(1, 1, 1, topk).expand(bs, h, nq, topk).contiguous()
    q2k_padded = torch.cat([
        q2k_compact,
        tail.view(1, 1, 1, nkv - topk).expand(bs, h, nq, nkv - topk),
    ], dim=-1).contiguous()
    empty_nums = torch.empty(0, device=device, dtype=torch.int32)
    block_sizes = torch.full((nkv,), blk, device=device, dtype=torch.int32)

    ref_out, ref_lse = bsa_attn_fwd(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_compact,
        topk,
        block_sizes,
        q2k_block_nums=empty_nums,
        return_lse=True,
        sparse_block_size=64,
    )
    for kv_splits in (1, 2, 4, 8):
        out, lse = bsa_attn_fwd(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            q2k_padded,
            topk,
            block_sizes,
            q2k_block_nums=empty_nums,
            return_lse=True,
            sparse_block_size=64,
            kv_splits=kv_splits,
        )
        torch.testing.assert_close(out, ref_out, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(lse, ref_lse, rtol=5e-3, atol=3e-2)


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] in (10, 11)
    ),
    reason="SM100/SM110 required",
)
def test_sm100_blk64_auto_kv_splits_api(monkeypatch):
    bs, h, sq, sk, d = 1, 4, 128, 512, 128
    blk = 64
    nq = sq // blk
    nkv = sk // blk
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(2028)
    q_bhsd = torch.randn(bs, h, sq, d, device=device, dtype=dtype)
    k_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)
    v_bhsd = torch.randn(bs, h, sk, d, device=device, dtype=dtype)

    q2k_block_index = torch.arange(nkv, device=device, dtype=torch.int32) \
        .view(1, 1, 1, nkv).expand(bs, h, nq, nkv).contiguous()
    empty_nums = torch.empty(0, device=device, dtype=torch.int32)
    block_sizes = torch.full((nkv,), blk, device=device, dtype=torch.int32)

    monkeypatch.setattr(
        bsa_interface,
        "_sm100_blk64_auto_kv_splits",
        lambda *args, **kwargs: 2,
    )
    ref_out, ref_lse = bsa_attn_fwd(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_block_index,
        nkv,
        block_sizes,
        q2k_block_nums=empty_nums,
        return_lse=True,
        sparse_block_size=64,
        kv_splits=2,
    )
    out, lse = bsa_attn_fwd(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_block_index,
        nkv,
        block_sizes,
        q2k_block_nums=empty_nums,
        return_lse=True,
        sparse_block_size=64,
        kv_splits="auto",
    )
    torch.testing.assert_close(out, ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=5e-3, atol=3e-2)


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] in (10, 11)
    ),
    reason="SM100/SM110 required",
)
def test_sm100_blk64_large_q_auto_scheduler_policy():
    """Large-Q auto mode avoids split traffic and uses persistent CLC."""
    q_large = torch.empty(
        (1, 40, 131072, 1), device="cuda", dtype=torch.int8
    )
    q_small = torch.empty((1, 4, 128, 1), device="cuda", dtype=torch.int8)
    q2k_block_index = torch.empty(
        (1, 1, 1, 2048), device="cuda", dtype=torch.int32
    )

    assert bsa_interface._sm100_blk64_auto_kv_splits(
        q_large, q2k_block_index, 2048
    ) == 1
    assert bsa_interface._sm100_blk64_auto_kv_splits(
        q_small, q2k_block_index, 2048
    ) == 8
    assert bsa_interface.choose_blk64_use_clc(q_large, 256)
    assert bsa_interface.choose_blk64_use_clc(q_large, 2048)


# ============== Kernel dispatch helper ==============

def _call_kernel(q, k, v, q2k_block_index, block_sparse_num, block_sizes, blk_n,
                 q2k_block_nums=None, softmax_scale=None, use_clc=False):
    """Dispatch to blk64 or blk128 kernel based on blk_n.

    q/k/v are BHSD (batch, heads, seq, dim). BSHD is covered by explicit
    wrapper compatibility tests, not by the benchmark helper.
    use_clc: blk64-only toggle for the CLC persistent scheduler; ignored for blk128.
    """
    if blk_n == 64:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        bn_arg = q2k_block_nums if q2k_block_nums is not None else torch.Tensor()
        bs_arg = block_sizes if block_sizes is not None else torch.Tensor()
        out_bhsd, _ = bsa_attn_fwd(
            q,
            k,
            v,
            q2k_block_index,
            block_sparse_num,
            bs_arg,
            q2k_block_nums=bn_arg,
            softmax_scale=softmax_scale,
            sparse_block_size=64,
            use_clc=use_clc,
        )
        return out_bhsd
    else:
        return bsa_attn_fwd(
            q,
            k,
            v,
            q2k_block_index,
            block_sparse_num,
            block_sizes,
            q2k_block_nums=q2k_block_nums,
            sparse_block_size=128,
        )[0]


# ============== Benchmark (make bb) ==============

def _benchmark_one(q, k, v, q2k_block_index, block_sparse_num, block_sizes, blk_n,
                   q2k_block_nums=None, niters=10, use_clc=False):
    """Warmup + benchmark a single kernel config. Returns median time in ms."""
    for _ in range(2):
        _call_kernel(q, k, v, q2k_block_index, block_sparse_num, block_sizes, blk_n,
                     q2k_block_nums=q2k_block_nums, use_clc=use_clc)
    torch.cuda.synchronize()

    evts = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(niters)
    ]
    for s, e in evts:
        s.record()
        _call_kernel(q, k, v, q2k_block_index, block_sparse_num, block_sizes, blk_n,
                     q2k_block_nums=q2k_block_nums, use_clc=use_clc)
        e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in evts])
    return times[len(times) // 2]


def run_benchmark_suite():
    # (bs, nheads, seqlen, hdim)
    configs = [
        (1, 40, 4096, 128),
        (1, 40, 8192, 128),
        (1, 40, 16384, 128),
        (1, 1,  102400, 128),
    ]

    # topK values to benchmark (0 = dense)
    topk_values = [0, 32, 64, 128, 256]

    for blk_n in _BLK_SIZES:
        blk_m = 64 if blk_n == 64 else 128
        use_clc_settings = [False, True] if blk_n == 64 else [False]
        if blk_n == 64:
            header = f"{'clc off TFLOPS':>14} {'clc on TFLOPS':>14}"
        else:
            header = f"{'TFLOPS':>14}"
        print(f"\n{'Config (blk=' + str(blk_n) + ')':<48} {header}")
        print("-" * (48 + len(header) + 2))

        for bs, nheads, seqlen, hdim in configs:
            dtype = torch.bfloat16
            q = torch.randn(bs, nheads, seqlen, hdim, device="cuda", dtype=dtype)
            k = torch.randn(bs, nheads, seqlen, hdim, device="cuda", dtype=dtype)
            v = torch.randn(bs, nheads, seqlen, hdim, device="cuda", dtype=dtype)

            num_kv_blocks = (seqlen + blk_n - 1) // blk_n

            for topk in topk_values:
                if topk == 0:
                    # Dense attention
                    q2k_block_index, bsn, bsizes = make_dense_block_sparse_args(
                        bs, seqlen, seqlen, nheads, blk_m=blk_m, blk_n=blk_n, device="cuda",
                    )
                    q2k_block_nums = None
                    label = f"bs={bs} h={nheads} sq={seqlen} d={hdim} dense"
                    effective_sk = seqlen
                else:
                    if topk > num_kv_blocks:
                        continue
                    # Round topk to even for block_sparse_num compatibility
                    topk_even = topk if topk % 2 == 0 else topk + 1
                    q2k_block_index, bsn, bsizes, q2k_block_nums = make_topk_block_sparse_args(
                        bs, seqlen, seqlen, nheads, topk_even, blk_m=blk_m, blk_n=blk_n, device="cuda",
                    )
                    label = f"bs={bs} h={nheads} sq={seqlen} d={hdim} topk={topk_even}"
                    effective_sk = topk_even * blk_n

                f = flops(bs, nheads, seqlen, effective_sk, hdim, hdim)
                print(f"  {label:<46}", end="", flush=True)
                for use_clc in use_clc_settings:
                    med = _benchmark_one(q, k, v, q2k_block_index, bsn, bsizes, blk_n,
                                         q2k_block_nums=q2k_block_nums, use_clc=use_clc)
                    tflops = f / (med * 1e-3) / 1e12
                    print(f" {tflops:>14.1f}", end="", flush=True)
                print()


# ============== Profile (make profile) ==============

def make_topk_block_sparse_args(batch_size, seqlen_q, seqlen_k, nheads, topk, blk_m=128, blk_n=128, device="cuda",
                                 use_var_block_num=False, use_block_sizes=False,
                                 block_size_mode="full"):
    """Create block-sparse args with fixed topK (each Q block attends to topK random KV blocks).

    Args:
        use_var_block_num: If True, return q2k_block_nums tensor (all entries = topk)
                           instead of using fixed block_sparse_num.
        use_block_sizes: If True, create block_sizes (actual sizes, enables masking path).
                         If False, pass None (skip block_sizes masking for faster kernel path).
        block_size_mode: "full" fills block_sizes with blk_n. "random" samples
                         each block size uniformly in [1, blk_n].

    Returns q2k_block_index, block_sparse_num, block_sizes, q2k_block_nums.
    """
    num_q_blocks = (seqlen_q + blk_m - 1) // blk_m
    num_kv_blocks = (seqlen_k + blk_n - 1) // blk_n
    assert topk <= num_kv_blocks, f"topk={topk} > num_kv_blocks={num_kv_blocks}"
    assert topk % 2 == 0, f"topk={topk} must be even"

    block_sparse_num = topk
    q2k_block_index = torch.empty(batch_size, nheads, num_q_blocks, block_sparse_num,
                                   dtype=torch.int32, device=device)
    for b in range(batch_size):
        for h in range(nheads):
            for m in range(num_q_blocks):
                perm = torch.randperm(num_kv_blocks, device=device)[:block_sparse_num]
                q2k_block_index[b, h, m] = perm.to(torch.int32)

    if use_block_sizes:
        if block_size_mode == "full":
            block_sizes = torch.full((num_kv_blocks,), blk_n, dtype=torch.int32, device=device)
        elif block_size_mode == "random":
            block_sizes = torch.randint(
                1, blk_n + 1, (num_kv_blocks,), dtype=torch.int32, device=device
            )
        else:
            raise ValueError(f"unknown block_size_mode: {block_size_mode}")
        last_block_actual = seqlen_k - (num_kv_blocks - 1) * blk_n
        if last_block_actual < blk_n:
            block_sizes[-1] = min(block_sizes[-1].item(), last_block_actual)
    else:
        block_sizes = None

    q2k_block_nums = None
    if use_var_block_num:
        q2k_block_nums = torch.full((batch_size, nheads, num_q_blocks), topk,
                                     dtype=torch.int32, device=device)

    return q2k_block_index, block_sparse_num, block_sizes, q2k_block_nums


def run_profile():
    bs, nheads, seqlen, hdim = 1, 40, 8192, 128
    # bs, nheads, seqlen, hdim = 1, 1, 1024000, 128
    topk = 64 # each Q block attends to topK KV blocks
    dtype = torch.bfloat16

    # Toggle features via env vars: BSA_VAR_BN=1  BSA_BLKSZ=1
    use_var_block_num = os.environ.get("BSA_VAR_BN", "0") == "1"
    use_block_sizes = os.environ.get("BSA_BLKSZ", "0") == "1"

    for blk_n in _BLK_SIZES:
        blk_m = 64 if blk_n == 64 else 128
        q = torch.randn(bs, nheads, seqlen, hdim, device="cuda", dtype=dtype)
        k = torch.randn(bs, nheads, seqlen, hdim, device="cuda", dtype=dtype)
        v = torch.randn(bs, nheads, seqlen, hdim, device="cuda", dtype=dtype)
        q2k_block_index, block_sparse_num, block_sizes, q2k_block_nums = make_topk_block_sparse_args(
            bs, seqlen, seqlen, nheads, topk, blk_m=blk_m, blk_n=blk_n, device="cuda",
            use_var_block_num=use_var_block_num, use_block_sizes=use_block_sizes,
        )

        # Warmup + profile run
        _call_kernel(q, k, v, q2k_block_index, block_sparse_num, block_sizes, blk_n,
                     q2k_block_nums=q2k_block_nums)
        torch.cuda.synchronize()

        flags = []
        if use_var_block_num:
            flags.append("var_bn")
        if use_block_sizes:
            flags.append("blksz")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        print(f"Profile done: bs={bs} h={nheads} sq={seqlen} d={hdim} topk={topk} blk={blk_n}{flag_str}")


# ============== Main ==============

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        run_profile()
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        run_benchmark_suite()
    else:
        run_quick_tests()
