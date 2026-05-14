"""Interface-level BSA SM100 tests and benchmark harness.

Usage:
    pytest -q test_bsa.py
    python test_bsa.py benchmark
    python test_bsa.py profile --topk 128
"""

import argparse
import os
import sys
import math
from contextlib import contextmanager

# Keep direct script execution (`python test_bsa.py ...`) working.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# BSA_BLK env var: "64", "128", or "64,128" (default). Controls which blk sizes to test.
_BSA_BLK = os.environ.get("BSA_BLK", "64,128")
_BLK_SIZES = [int(x) for x in _BSA_BLK.split(",")]

from utils.testing import attention_ref
from utils.bench_utils import flops
from bsa_attn_interface import (
    bsa_attn_bwd,
    bsa_attn_fwd,
    bsa_attn_fwd_blk64,
    convert_q2k_to_k2q_csr,
)

DEFAULT_BENCH_BATCH = 1
DEFAULT_BENCH_SEQLEN = 262144
DEFAULT_BENCH_HEADS = 40
DEFAULT_BENCH_DIM = 128
DEFAULT_BENCH_WARMUP = 5
DEFAULT_BENCH_ITERS = 20
TOPK_VALUES = (4096, 2048, 1024, 512, 256, 128, 64, 32)


class BenchmarkPattern:
    UNIFORM = "uniform"
    SINK = "sink"

# Optional: blk64 C++ AOT kernel (install via `make setup BLK=64`)
try:
    import bsa_fwd_blk64_ext
    HAS_BLK64 = True
except ImportError:
    HAS_BLK64 = False


@contextmanager
def _nvtx_range(message: str):
    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


# ============== Block-sparse helpers ==============

def make_dense_block_sparse_args(batch_size, seqlen_q, seqlen_k, nheads, blk_m=128, blk_n=128, device="cuda"):
    """Create block-sparse args equivalent to dense (full) attention.  For benchmark/profile.

    All block_sizes = blk_n (full blocks).  Requires seqlen_k % (2*blk_n) == 0.
    Returns q2k_block_index, max_topk, block_sizes.
    """
    num_q_blocks = (seqlen_q + blk_m - 1) // blk_m
    num_kv_blocks = (seqlen_k + blk_n - 1) // blk_n

    assert num_kv_blocks >= 2 and num_kv_blocks % 2 == 0, (
        f"num_kv_blocks={num_kv_blocks} must be even and >= 2 for dense-equivalent test. "
        f"Adjust seqlen_k to be a multiple of {2 * blk_n}."
    )
    max_topk = num_kv_blocks

    indices = torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
    q2k_block_index = indices.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
        batch_size, nheads, num_q_blocks, num_kv_blocks
    ).contiguous()

    block_sizes = torch.full((num_kv_blocks,), blk_n, dtype=torch.int32, device=device)
    return q2k_block_index, max_topk, block_sizes


def make_random_block_sparse_args(batch_size, seqlen_q, seqlen_k, nheads, blk_m=128, blk_n=128, device="cuda"):
    """Create random block-sparse args for correctness testing.

    Random max_topk (even, >= 2), random per-(batch,head,q_block) KV block
    selection, and random block_sizes in [1, blk_n].
    Returns q2k_block_index, max_topk, block_sizes.
    """
    num_q_blocks = (seqlen_q + blk_m - 1) // blk_m
    num_kv_blocks = (seqlen_k + blk_n - 1) // blk_n
    assert num_kv_blocks >= 2, f"num_kv_blocks={num_kv_blocks} must be >= 2"

    max_even = num_kv_blocks if num_kv_blocks % 2 == 0 else num_kv_blocks - 1
    # Minimum topK=4 to avoid pre-existing kernel issue with topK=2 + large batch + small hdim
    min_topk = min(4, max_even)
    # blk64: phantom padding allows any count >= 1. Step by 2 for variety.
    # blk128: step by 2 as before.
    step = 2
    possible_counts = list(range(min_topk, max_even + 1, step))
    max_topk = possible_counts[torch.randint(len(possible_counts), (1,)).item()]

    q2k_block_index = torch.empty(batch_size, nheads, num_q_blocks, max_topk,
                                   dtype=torch.int32, device=device)
    for b in range(batch_size):
        for h in range(nheads):
            for m in range(num_q_blocks):
                perm = torch.randperm(num_kv_blocks, device=device)[:max_topk]
                q2k_block_index[b, h, m] = perm.to(torch.int32)

    block_sizes = torch.randint(1, blk_n + 1, (num_kv_blocks,), dtype=torch.int32, device=device)
    last_block_actual = seqlen_k - (num_kv_blocks - 1) * blk_n
    if last_block_actual < blk_n:
        block_sizes[-1] = min(block_sizes[-1].item(), last_block_actual)

    return q2k_block_index, max_topk, block_sizes


def make_random_variable_block_sparse_args(batch_size, seqlen_q, seqlen_k, nheads, blk_m=128, blk_n=128, device="cuda"):
    """Create random block-sparse args with per-(batch, head, q_block) variable block counts.

    Each Q block gets a random max_topk in [0, num_kv_blocks].
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
                row_topk = possible_counts[torch.randint(len(possible_counts), (1,)).item()]
                q2k_block_nums[b, h, m] = row_topk
                perm = torch.randperm(num_kv_blocks, device=device)[:row_topk]
                q2k_block_index[b, h, m, :row_topk] = perm.to(torch.int32)

    block_sizes = torch.randint(1, blk_n + 1, (num_kv_blocks,), dtype=torch.int32, device=device)
    last_block_actual = seqlen_k - (num_kv_blocks - 1) * blk_n
    if last_block_actual < blk_n:
        block_sizes[-1] = min(block_sizes[-1].item(), last_block_actual)

    return q2k_block_index, q2k_block_nums, block_sizes


def block_sparse_to_attn_bias(q2k_block_index, max_topk, block_sizes,
                               seqlen_q, seqlen_k, blk_m=128, blk_n=128,
                               q2k_block_nums=None):
    """Convert block-sparse args to additive attention bias for reference.

    Returns attn_bias (batch, nheads, seqlen_q, seqlen_k) float32:
        0.0 for attended positions, -inf for masked.

    When q2k_block_nums is provided, each (batch, head, q_block) uses its own
    block count instead of the fixed max_topk.
    """
    batch_size, nheads, num_q_blocks, _max_topk_capacity = q2k_block_index.shape
    num_kv_blocks = block_sizes.shape[0]
    device = q2k_block_index.device

    col_idx = torch.arange(blk_n, device=device)
    block_valid = col_idx.unsqueeze(0) < block_sizes.unsqueeze(1)  # (num_kv_blocks, blk_n)

    if q2k_block_nums is None:
        # Fixed max_topk: all entries up to max_topk are valid
        block_attended = torch.zeros(batch_size, nheads, num_q_blocks, num_kv_blocks,
                                      dtype=torch.bool, device=device)
        block_attended.scatter_(3, q2k_block_index[..., :max_topk].long(), True)
    else:
        # Variable per-(b,h,m) block counts
        block_attended = torch.zeros(batch_size, nheads, num_q_blocks, num_kv_blocks,
                                      dtype=torch.bool, device=device)
        for b in range(batch_size):
            for h in range(nheads):
                for m in range(num_q_blocks):
                    row_topk = q2k_block_nums[b, h, m].item()
                    indices = q2k_block_index[b, h, m, :row_topk].long()
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


def make_topk_block_sparse_args(batch_size, seqlen_q, seqlen_k, nheads, topk, blk_m=128, blk_n=128, device="cuda",
                                 use_var_block_num=False, use_block_sizes=False,
                                 block_size_mode="full"):
    """Create block-sparse args with fixed topK for each Q block."""
    num_q_blocks = (seqlen_q + blk_m - 1) // blk_m
    num_kv_blocks = (seqlen_k + blk_n - 1) // blk_n
    assert topk <= num_kv_blocks, f"topk={topk} > num_kv_blocks={num_kv_blocks}"
    assert topk % 2 == 0, f"topk={topk} must be even"

    max_topk = topk
    q2k_block_index = torch.empty(batch_size, nheads, num_q_blocks, max_topk,
                                   dtype=torch.int32, device=device)
    for b in range(batch_size):
        for h in range(nheads):
            for m in range(num_q_blocks):
                perm = torch.randperm(num_kv_blocks, device=device)[:max_topk]
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

    return q2k_block_index, max_topk, block_sizes, q2k_block_nums


def _require_sm100():
    major, _ = torch.cuda.get_device_capability()
    if major not in (10, 11):
        pytest.skip("blk64 backward CSR kernel requires SM100/SM110")


def _torch_ref_bwd(q, k, v, dout, attn_bias, softmax_scale, upcast=True):
    dtype = torch.float32 if upcast else q.dtype
    q_ref = q.detach().to(dtype).requires_grad_()
    k_ref = k.detach().to(dtype).requires_grad_()
    v_ref = v.detach().to(dtype).requires_grad_()
    if upcast:
        scores = torch.einsum("bhtd,bhsd->bhts", q_ref * softmax_scale, k_ref)
    else:
        scores = torch.einsum("bhtd,bhsd->bhts", q_ref, k_ref * softmax_scale)
    scores = scores + attn_bias.to(scores.dtype)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhts,bhsd->bhtd", attn, v_ref)
    out = torch.nan_to_num(out, nan=0.0)
    lse = torch.logsumexp(scores, dim=-1)
    out.backward(dout.to(dtype))
    return (
        out.detach().to(q.dtype),
        lse.detach().float(),
        torch.nan_to_num(q_ref.grad.detach().float()),
        torch.nan_to_num(k_ref.grad.detach().float()),
        torch.nan_to_num(v_ref.grad.detach().float()),
    )


def _assert_bwd_close(actual, expected, bf16_baseline):
    for name, got, ref, pt in zip(("dq", "dk", "dv"), actual, expected, bf16_baseline):
        diff = (got.float() - ref).abs().max().item()
        pt_diff = (pt - ref).abs().max().item()
        bf16_eps = (ref + 0.3 - 0.3 - ref).abs().max().item()
        tol = 3 * pt_diff + 3 * bf16_eps + 5e-3
        assert diff <= tol, f"{name} diff={diff} tol={tol}"


def _expected_csr(q2k_block_index, max_topk, num_kv_blocks, q2k_block_nums):
    q2k_cpu = q2k_block_index.cpu()
    nums_cpu = q2k_block_nums.cpu() if q2k_block_nums is not None else None
    B, H, num_q_blocks, _ = q2k_cpu.shape
    row_ptr = torch.empty((B, H, num_kv_blocks + 1), dtype=torch.int32)
    q_indices = []
    offset = 0
    for b in range(B):
        for h in range(H):
            rows = [[] for _ in range(num_kv_blocks)]
            for q_block in range(num_q_blocks):
                count = (
                    int(nums_cpu[b, h, q_block].item())
                    if nums_cpu is not None
                    else int(max_topk)
                )
                for slot in range(count):
                    kv = int(q2k_cpu[b, h, q_block, slot].item())
                    if 0 <= kv < num_kv_blocks:
                        rows[kv].append(q_block)
            row_ptr[b, h, 0] = offset
            for kv, qs in enumerate(rows):
                q_indices.extend(qs)
                offset += len(qs)
                row_ptr[b, h, kv + 1] = offset
    return row_ptr, torch.tensor(q_indices, dtype=torch.int32)


def _assert_total_packed_csr(row_ptr, q_indices, expected_row_ptr, expected_q_indices):
    torch.testing.assert_close(row_ptr.cpu(), expected_row_ptr)
    torch.testing.assert_close(q_indices.cpu(), expected_q_indices)
    assert q_indices.ndim == 1
    assert int(row_ptr.reshape(-1)[-1].item()) == q_indices.numel()


def _assert_qrange_split_schedule(
    row_ptr,
    q_indices,
    schedule_metadata,
    schedule_work_counts,
    num_q_blocks,
    max_topk,
):
    from csrc.common.block_sparse_csr import _select_schedule_policy

    row_ptr_cpu = row_ptr.cpu()
    q_indices_cpu = q_indices.cpu()
    schedule_cpu = schedule_metadata.cpu()
    work_counts_cpu = schedule_work_counts.cpu()
    B, H, num_kv_blocks_plus_1 = row_ptr_cpu.shape
    num_kv_blocks = num_kv_blocks_plus_1 - 1
    qrange_blocks, target_q_blocks = _select_schedule_policy(
        B, H, num_q_blocks, num_kv_blocks, max_topk
    )
    num_q_groups = (int(num_q_blocks) + qrange_blocks - 1) // qrange_blocks

    assert schedule_metadata.shape[:2] == (B, H)
    assert schedule_metadata.shape[3] == 4
    assert schedule_work_counts.shape == (B, H)
    assert int(work_counts_cpu.max().item()) <= schedule_metadata.shape[2]

    for b in range(B):
        for h in range(H):
            work_count = int(work_counts_cpu[b, h].item())
            assert work_count >= num_q_groups * num_kv_blocks
            segments = [[] for _ in range(num_kv_blocks)]
            for work in range(work_count):
                kv = int(schedule_cpu[b, h, work, 0].item())
                start = int(schedule_cpu[b, h, work, 1].item())
                count = int(schedule_cpu[b, h, work, 2].item())
                q_group = int(schedule_cpu[b, h, work, 3].item())
                assert 0 <= kv < num_kv_blocks
                assert count <= target_q_blocks
                if count == 0:
                    continue
                row_start = int(row_ptr_cpu[b, h, kv].item())
                row_end = int(row_ptr_cpu[b, h, kv + 1].item())
                assert row_start <= start < row_end
                assert start + count <= row_end
                q_slice = q_indices_cpu[start : start + count]
                assert int(q_slice.min().item()) >= q_group * qrange_blocks
                assert int(q_slice.max().item()) < (q_group + 1) * qrange_blocks
                segments[kv].append((start, start + count))

            for kv in range(num_kv_blocks):
                row_start = int(row_ptr_cpu[b, h, kv].item())
                row_end = int(row_ptr_cpu[b, h, kv + 1].item())
                cursor = row_start
                for start, end in sorted(segments[kv]):
                    assert start == cursor
                    cursor = end
                assert cursor == row_end


# ============== Correctness helpers ==============

def _test_single(bs, seqlen_q, seqlen_k, nheads, nheads_kv, d, dtype=torch.bfloat16,
                  use_variable_block_nums=False, use_block_sizes=True,
                  blk_m=128, blk_n=128, use_clc=False):
    """Run a single correctness test with random block-sparse pattern.

    blk_m/blk_n: block sizes. blk_n=64 routes to blk64 C++ AOT kernel.
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

    # blk64 constraints: skip unsupported configurations
    if is_blk64:
        if not HAS_BLK64:
            pytest.skip("bsa_fwd_blk64_ext not built")
        if nheads_kv != nheads:
            pytest.skip("blk64 does not support GQA/MQA")
        if d != 128:
            pytest.skip("blk64 requires d=128")

    q_ref = torch.randn(bs, seqlen_q, nheads, d, device=device, dtype=dtype).requires_grad_()
    k_ref = torch.randn(bs, seqlen_k, nheads_kv, d, device=device, dtype=dtype).requires_grad_()
    v_ref = torch.randn(bs, seqlen_k, nheads_kv, d, device=device, dtype=dtype).requires_grad_()
    q = q_ref.detach().requires_grad_()
    k = k_ref.detach().requires_grad_()
    v = v_ref.detach().requires_grad_()

    qhead_per_kvhead = nheads // nheads_kv
    pack_gqa = qhead_per_kvhead > 1 and (128 % qhead_per_kvhead == 0)
    nheads_q2k = nheads_kv if pack_gqa else nheads
    seqlen_q_q2k = seqlen_q * qhead_per_kvhead if pack_gqa else seqlen_q

    q2k_block_nums = None
    if use_variable_block_nums:
        q2k_block_index, q2k_block_nums, block_sizes = make_random_variable_block_sparse_args(
            bs, seqlen_q_q2k, seqlen_k, nheads_q2k, blk_m=blk_m, blk_n=blk_n, device=device,
        )
        max_topk = q2k_block_index.shape[-1]
    else:
        q2k_block_index, max_topk, block_sizes = make_random_block_sparse_args(
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
        q2k_block_index, max_topk, block_sizes, seqlen_q_q2k, seqlen_k,
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
        bn_arg = q2k_block_nums if use_variable_block_nums else torch.Tensor()
        bs_arg = block_sizes_kernel if block_sizes_kernel is not None else torch.Tensor()
        # blk64 kernel consumes BHSD natively; convert from BSHD test tensors
        # at the boundary and convert the BHSD output back to BSHD for reference.
        q_bhsd = q.transpose(1, 2).contiguous()
        k_bhsd = k.transpose(1, 2).contiguous()
        v_bhsd = v.transpose(1, 2).contiguous()
        out_bhsd, lse = torch.ops.bsa_blk64.fwd(
            q_bhsd, k_bhsd, v_bhsd, q2k_block_index, max_topk,
            bs_arg, softmax_scale, bn_arg, use_clc)
        out = out_bhsd.transpose(1, 2).contiguous()
    else:
        out, lse = bsa_attn_fwd(q, k, v, q2k_block_index, max_topk, block_sizes_kernel,
                                 q2k_block_nums=q2k_block_nums, return_lse=True)
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
    mode_str = "variable_topk" if use_variable_block_nums else f"max_topk={max_topk}"
    mode_str += " no_bs" if not use_block_sizes else ""
    tag = "PASS" if passed else "FAIL"
    print(
        f"  {tag} bs={bs} sq={seqlen_q} sk={seqlen_k} h={nheads}/{nheads_kv} d={d} "
        f"{mode_str}{blk_str}: "
        f"kernel={kernel_diff:.6f} pt={pt_diff:.6f} tol={tol:.6f} "
        f"lse_diff={lse_max_diff:.6f}"
    )
    assert passed, f"kernel_diff={kernel_diff} > tol={tol}, lse_diff={lse_max_diff} > lse_tol={lse_tol}"


# ============== Pytest ==============

def _assert_fwd_close(out, lse, out_ref, lse_ref, out_pt, tag):
    out = torch.nan_to_num(out, nan=0.0)
    out_ref = torch.nan_to_num(out_ref, nan=0.0)
    out_pt = torch.nan_to_num(out_pt, nan=0.0)

    kernel_diff = (out - out_ref).abs().max().item()
    pt_diff = (out_pt - out_ref).abs().max().item()
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    tol = 2 * pt_diff + fwd_atol

    lse_diff = (lse - lse_ref).abs()
    finite_mask = lse_ref.isfinite()
    lse_max_diff = lse_diff[finite_mask].max().item() if finite_mask.any() else 0.0
    lse_tol = 1e-3

    print(
        f"  {tag} fwd: kernel={kernel_diff:.6f} pt={pt_diff:.6f} "
        f"tol={tol:.6f} lse_diff={lse_max_diff:.6f}"
    )
    assert kernel_diff <= tol, f"fwd diff={kernel_diff} > tol={tol}"
    assert lse_max_diff <= lse_tol, f"lse diff={lse_max_diff} > tol={lse_tol}"


def _run_blk64_fwd_bwd_single(use_variable_block_nums, use_block_sizes):
    _require_sm100()
    if not HAS_BLK64:
        pytest.skip("bsa_fwd_blk64_ext not built")

    torch.manual_seed(1)
    B, H, seqlen_q, seqlen_k, D = 1, 2, 128, 512, 128
    q = torch.randn(B, H, seqlen_q, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, seqlen_k, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, seqlen_k, D, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn_like(q)
    q2k_block_index, max_topk, block_sizes, q2k_block_nums = make_topk_block_sparse_args(
        B,
        seqlen_q,
        seqlen_k,
        H,
        topk=4,
        blk_m=64,
        blk_n=64,
        device="cuda",
        use_var_block_num=use_variable_block_nums,
        use_block_sizes=True,
        block_size_mode="full",
    )
    block_sizes_kernel = block_sizes if use_block_sizes else None
    attn_bias = block_sparse_to_attn_bias(
        q2k_block_index,
        max_topk,
        block_sizes,
        seqlen_q,
        seqlen_k,
        blk_m=64,
        blk_n=64,
        q2k_block_nums=q2k_block_nums,
    )
    softmax_scale = 1.0 / math.sqrt(D)
    out_ref, lse_ref, dq_ref, dk_ref, dv_ref = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale
    )
    out_pt, _, dq_pt, dk_pt, dv_pt = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale, upcast=False
    )

    empty = torch.empty(0, dtype=torch.int32, device=q.device)
    block_sizes_fwd = block_sizes if use_block_sizes else empty
    q2k_block_nums_fwd = q2k_block_nums if q2k_block_nums is not None else empty
    out, lse = torch.ops.bsa_blk64.fwd(
        q,
        k,
        v,
        q2k_block_index,
        max_topk,
        block_sizes_fwd,
        softmax_scale,
        q2k_block_nums_fwd,
        True,
    )
    mode = "variable" if use_variable_block_nums else "fixed"
    mode += " block_sizes" if use_block_sizes else " no_block_sizes"
    _assert_fwd_close(out, lse, out_ref, lse_ref, out_pt, f"blk64 {mode}")

    dq, dk, dv = bsa_attn_bwd(
        dout,
        q,
        k,
        v,
        out_ref,
        lse_ref,
        q2k_block_index,
        max_topk,
        block_sizes_kernel,
        q2k_block_nums=q2k_block_nums,
        softmax_scale=softmax_scale,
    )
    _assert_bwd_close((dq, dk, dv), (dq_ref, dk_ref, dv_ref), (dq_pt, dk_pt, dv_pt))
    print(f"  PASS blk64 {mode} bwd")


@pytest.mark.parametrize("blk_n", _BLK_SIZES, ids=lambda value: f"blk{value}")
@pytest.mark.parametrize(
    "use_variable_block_nums",
    [False, True],
    ids=lambda value: "variable_block_nums" if value else "fixed_block_num",
)
@pytest.mark.parametrize(
    "use_block_sizes",
    [False, True],
    ids=lambda value: "block_sizes" if value else "no_block_sizes",
)
def test_bsa_sm100(blk_n, use_variable_block_nums, use_block_sizes):
    if blk_n == 64:
        _run_blk64_fwd_bwd_single(use_variable_block_nums, use_block_sizes)
        return

    if blk_n == 128:
        blk_m = 64 if blk_n == 64 else 128
        _test_single(
            bs=2,
            seqlen_q=128,
            seqlen_k=512,
            nheads=4,
            nheads_kv=4,
            d=128,
            dtype=torch.bfloat16,
            use_variable_block_nums=use_variable_block_nums,
            use_block_sizes=use_block_sizes,
            blk_m=blk_m,
            blk_n=blk_n,
            use_clc=(blk_n == 64),
        )
        return

    raise ValueError(f"unknown block size: {blk_n}")


@pytest.mark.parametrize("use_variable_counts", [False, True])
def test_convert_q2k_to_k2q_csr(use_variable_counts):
    _require_sm100()
    torch.manual_seed(0)
    B, H, seqlen_q, seqlen_k = 2, 3, 256, 512
    num_kv_blocks = seqlen_k // 64

    if use_variable_counts:
        q2k_block_index, q2k_block_nums, _ = make_random_variable_block_sparse_args(
            B, seqlen_q, seqlen_k, H, blk_m=64, blk_n=64, device="cuda"
        )
        max_topk = q2k_block_index.shape[-1]
    else:
        q2k_block_index, max_topk, _, q2k_block_nums = make_topk_block_sparse_args(
            B, seqlen_q, seqlen_k, H, topk=4, blk_m=64, blk_n=64, device="cuda"
        )

    row_ptr, q_indices, schedule_metadata, schedule_work_counts = convert_q2k_to_k2q_csr(
        q2k_block_index,
        max_topk,
        num_kv_blocks,
        q2k_block_nums=q2k_block_nums,
        return_schedule=True,
    )
    expected_row_ptr, expected_q_indices = _expected_csr(
        q2k_block_index, max_topk, num_kv_blocks, q2k_block_nums
    )
    _assert_total_packed_csr(row_ptr, q_indices, expected_row_ptr, expected_q_indices)
    _assert_qrange_split_schedule(
        row_ptr,
        q_indices,
        schedule_metadata,
        schedule_work_counts,
        seqlen_q // 64,
        max_topk,
    )


# ============== Benchmark / Profile ==============

def _cuda_time_ms(fn, *, warmup: int, repeat: int, range_name: str | None = None) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if range_name is None:
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / repeat

    with _nvtx_range(range_name):
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        end.synchronize()
    return start.elapsed_time(end) / repeat


def _bwd_flops(batch: int, heads: int, seqlen: int, topk: int, block_size: int, dim: int) -> int:
    return 2 * batch * heads * seqlen * (topk * block_size) * (3 * dim + 2 * dim)


def _make_benchmark_topk_block_sparse_args(
    batch: int,
    seqlen: int,
    heads: int,
    topk: int,
    *,
    block_size: int,
    pattern: str,
    device: str,
):
    num_q_blocks = seqlen // block_size
    num_kv_blocks = seqlen // block_size
    if topk > num_kv_blocks:
        raise ValueError(f"topk={topk} exceeds num_kv_blocks={num_kv_blocks}")
    if topk % 2 != 0:
        raise ValueError(f"topk={topk} must be even")

    kv = torch.arange(topk, device=device, dtype=torch.int32).view(1, topk)
    if pattern == BenchmarkPattern.UNIFORM:
        q_offsets = torch.arange(num_q_blocks, device=device, dtype=torch.int32).view(num_q_blocks, 1)
        q2k_2d = (q_offsets + kv) % num_kv_blocks
    elif pattern == BenchmarkPattern.SINK:
        q2k_2d = kv.expand(num_q_blocks, topk)
    else:
        raise ValueError(f"unknown benchmark pattern: {pattern}")

    q2k_block_index = q2k_2d.view(1, 1, num_q_blocks, topk).expand(
        batch, heads, num_q_blocks, topk
    ).contiguous()
    q2k_block_nums = torch.full(
        (batch, heads, num_q_blocks), topk, dtype=torch.int32, device=device
    )
    block_sizes = torch.full((num_kv_blocks,), block_size, dtype=torch.int32, device=device)
    return q2k_block_index, topk, block_sizes, q2k_block_nums


def _build_benchmark_context(
    *,
    batch: int,
    seqlen: int,
    heads: int,
    dim: int,
    topk: int,
    seed: int,
    pattern: str,
) -> dict[str, object]:
    if not HAS_BLK64:
        raise RuntimeError("bsa_fwd_blk64_ext is not built")
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    block_size = 64
    num_kv_blocks = seqlen // block_size
    if seqlen % block_size != 0:
        raise ValueError("benchmark seqlen must be divisible by 64")
    if topk > num_kv_blocks:
        raise ValueError(f"topk={topk} exceeds num_kv_blocks={num_kv_blocks}")

    q = torch.randn(batch, heads, seqlen, dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seqlen, dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seqlen, dim, device=device, dtype=dtype)
    q2k_block_index, max_topk, block_sizes, q2k_block_nums = _make_benchmark_topk_block_sparse_args(
        batch,
        seqlen,
        heads,
        topk,
        block_size=block_size,
        pattern=pattern,
        device=device,
    )
    softmax_scale = 1.0 / math.sqrt(dim)

    def run_csr():
        return convert_q2k_to_k2q_csr(
            q2k_block_index,
            max_topk,
            num_kv_blocks,
            q2k_block_nums=q2k_block_nums,
            return_schedule=True,
        )

    k2q = run_csr()
    torch.cuda.synchronize()

    def run_fwd():
        return bsa_attn_fwd_blk64(
            q,
            k,
            v,
            q2k_block_index,
            block_sizes,
            q2k_block_nums,
            softmax_scale=softmax_scale,
            use_clc=True,
        )

    out, lse = run_fwd()
    dout = torch.randn_like(out)
    torch.cuda.synchronize()

    def run_bwd_prebuilt():
        return bsa_attn_bwd(
            dout,
            q,
            k,
            v,
            out,
            lse,
            q2k_block_index,
            max_topk,
            block_sizes,
            q2k_block_nums=q2k_block_nums,
            softmax_scale=softmax_scale,
            k2q_row_ptr=k2q[0],
            k2q_q_indices=k2q[1],
            k2q_schedule_metadata=k2q[2],
            k2q_schedule_work_counts=k2q[3],
        )

    def run_bwd_internal():
        return bsa_attn_bwd(
            dout,
            q,
            k,
            v,
            out,
            lse,
            q2k_block_index,
            max_topk,
            block_sizes,
            q2k_block_nums=q2k_block_nums,
            softmax_scale=softmax_scale,
        )

    return {
        "batch": batch,
        "seqlen": seqlen,
        "heads": heads,
        "dim": dim,
        "topk": topk,
        "block_size": block_size,
        "run_csr": run_csr,
        "run_fwd": run_fwd,
        "run_bwd_prebuilt": run_bwd_prebuilt,
        "run_bwd_internal": run_bwd_internal,
    }


def run_benchmark_suite(args: argparse.Namespace) -> None:
    topk_values = TOPK_VALUES if args.topk == "all" else tuple(int(v) for v in args.topk.split(","))
    print(
        "case: "
        f"bs={args.batch} seqlen={args.seqlen} heads={args.heads} dim={args.dim} "
        f"pattern={args.pattern} topk_values={topk_values}"
    )
    print(
        f"{'topk':>6} {'csr_ms':>10} {'fwd_ms':>10} {'bwd_ms':>10} "
        f"{'bwd_internal_ms':>16} {'fwd_tflops':>12} {'bwd_tflops':>12}"
    )
    print("-" * 84)
    for topk in topk_values:
        ctx = _build_benchmark_context(
            batch=args.batch,
            seqlen=args.seqlen,
            heads=args.heads,
            dim=args.dim,
            topk=topk,
            seed=args.seed,
            pattern=args.pattern,
        )
        csr_ms = _cuda_time_ms(
            ctx["run_csr"], warmup=args.warmup, repeat=args.iters, range_name=f"topk_{topk}_csr"
        )
        fwd_ms = _cuda_time_ms(
            ctx["run_fwd"], warmup=args.warmup, repeat=args.iters, range_name=f"topk_{topk}_fwd"
        )
        bwd_ms = _cuda_time_ms(
            ctx["run_bwd_prebuilt"], warmup=args.warmup, repeat=args.iters, range_name=f"topk_{topk}_bwd_prebuilt"
        )
        bwd_internal_ms = _cuda_time_ms(
            ctx["run_bwd_internal"], warmup=args.warmup, repeat=args.iters, range_name=f"topk_{topk}_bwd_internal"
        )
        fwd_ops = flops(args.batch, args.heads, args.seqlen, topk * 64, args.dim, args.dim)
        bwd_ops = _bwd_flops(args.batch, args.heads, args.seqlen, topk, 64, args.dim)
        print(
            f"{topk:6d} {csr_ms:10.3f} {fwd_ms:10.3f} {bwd_ms:10.3f} "
            f"{bwd_internal_ms:16.3f} "
            f"{fwd_ops / (fwd_ms * 1e-3) / 1e12:12.2f} "
            f"{bwd_ops / (bwd_ms * 1e-3) / 1e12:12.2f}"
        )
        del ctx
        torch.cuda.empty_cache()


def run_profile(args: argparse.Namespace) -> None:
    ctx = _build_benchmark_context(
        batch=args.batch,
        seqlen=args.seqlen,
        heads=args.heads,
        dim=args.dim,
        topk=args.topk,
        seed=args.seed,
        pattern=args.pattern,
    )
    with _nvtx_range(f"profile_topk_{args.topk}_csr"):
        ctx["run_csr"]()
    torch.cuda.synchronize()
    with _nvtx_range(f"profile_topk_{args.topk}_fwd"):
        ctx["run_fwd"]()
    torch.cuda.synchronize()
    with _nvtx_range(f"profile_topk_{args.topk}_bwd_prebuilt"):
        ctx["run_bwd_prebuilt"]()
    torch.cuda.synchronize()
    with _nvtx_range(f"profile_topk_{args.topk}_bwd_internal"):
        ctx["run_bwd_internal"]()
    torch.cuda.synchronize()
    print(
        "Profile done: "
        f"bs={args.batch} seqlen={args.seqlen} heads={args.heads} "
        f"dim={args.dim} pattern={args.pattern} topk={args.topk}"
    )


# ============== Main ==============

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("benchmark", help="Run BSA interface-level benchmarks")
    bench.add_argument("--batch", type=int, default=DEFAULT_BENCH_BATCH)
    bench.add_argument("--seqlen", type=int, default=DEFAULT_BENCH_SEQLEN)
    bench.add_argument("--heads", type=int, default=DEFAULT_BENCH_HEADS)
    bench.add_argument("--dim", type=int, default=DEFAULT_BENCH_DIM)
    bench.add_argument("--topk", default="all", help="Comma-separated values or 'all'")
    bench.add_argument(
        "--pattern",
        choices=(BenchmarkPattern.UNIFORM, BenchmarkPattern.SINK),
        default=BenchmarkPattern.UNIFORM,
    )
    bench.add_argument("--warmup", type=int, default=DEFAULT_BENCH_WARMUP)
    bench.add_argument("--iters", type=int, default=DEFAULT_BENCH_ITERS)
    bench.add_argument("--seed", type=int, default=42)

    profile = subparsers.add_parser("profile", help="Run one NVTX-annotated profile case")
    profile.add_argument("--batch", type=int, default=DEFAULT_BENCH_BATCH)
    profile.add_argument("--seqlen", type=int, default=DEFAULT_BENCH_SEQLEN)
    profile.add_argument("--heads", type=int, default=DEFAULT_BENCH_HEADS)
    profile.add_argument("--dim", type=int, default=DEFAULT_BENCH_DIM)
    profile.add_argument("--topk", type=int, default=128)
    profile.add_argument(
        "--pattern",
        choices=(BenchmarkPattern.UNIFORM, BenchmarkPattern.SINK),
        default=BenchmarkPattern.UNIFORM,
    )
    profile.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "benchmark":
        run_benchmark_suite(args)
    elif args.command == "profile":
        run_profile(args)


if __name__ == "__main__":
    main()
