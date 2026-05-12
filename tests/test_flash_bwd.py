"""BSA SM100 Backward Kernel — Test / Benchmark

Usage:
    python tests/test_flash_bwd.py              # quick correctness tests
    python tests/test_flash_bwd.py benchmark    # simple dense-bwd benchmark
"""

import os
import sys
import math
import gc

# Keep direct script execution (`python tests/test_flash_bwd.py`) working.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from bsa_attn_interface import (
    bsa_attn_bwd,
    bsa_attn_bwd_csr_scheduled,
    bsa_attn_bwd_qbucket,
    convert_q2k_to_k2q,
    convert_q2k_to_k2q_csr,
)
from test_flash_fwd import (
    make_dense_block_sparse_args,
    make_random_block_sparse_args,
    make_random_variable_block_sparse_args,
    make_topk_block_sparse_args,
    block_sparse_to_attn_bias,
)
from utils.bench_utils import flops, bwd_flops
from utils.block_sparse_csr import materialize_k2q_csr_to_dense


# The bwd kernel only exists for blk64 (sparse_block_size=64, head_dim=128, MHA, bf16).
BLK = 64


def _assert_total_packed_csr(
    row_ptr,
    q_indices,
    *,
    bs,
    nheads,
    num_q_blocks,
    num_kv_blocks,
    block_sparse_num,
    q2k_block_nums,
):
    assert row_ptr.shape == (bs, nheads, num_kv_blocks + 1)
    assert q_indices.ndim == 1
    flat_row_ptr = row_ptr.reshape(bs * nheads, num_kv_blocks + 1)
    if q2k_block_nums is None:
        edges_per_bh = num_q_blocks * block_sparse_num
        expected_offsets = (
            torch.arange(bs * nheads, device=row_ptr.device, dtype=torch.int64)
            * edges_per_bh
        )
        expected_edges = bs * nheads * edges_per_bh
    else:
        bh_edges = q2k_block_nums.to(torch.int64).sum(dim=2).reshape(-1)
        offsets = torch.empty((bs * nheads + 1,), device=row_ptr.device, dtype=torch.int64)
        offsets[0] = 0
        offsets[1:] = torch.cumsum(bh_edges, dim=0)
        expected_offsets = offsets[:-1]
        expected_edges = int(offsets[-1].item())
    assert q_indices.numel() == expected_edges
    torch.testing.assert_close(flat_row_ptr[:, 0].to(torch.int64), expected_offsets)
    assert int(flat_row_ptr[-1, -1].item()) == expected_edges


def _assert_csr_schedule_matches_row_ptr(
    row_ptr,
    q_indices,
    schedule_metadata,
    schedule_work_counts,
    *,
    target_q_blocks,
):
    B, H, num_kv_blocks_plus_1 = row_ptr.shape
    num_kv_blocks = num_kv_blocks_plus_1 - 1
    assert schedule_metadata.shape[:2] == (B, H)
    assert schedule_metadata.shape[3] == 4
    assert schedule_work_counts.shape == (B, H)
    assert int(schedule_work_counts.max().item()) <= schedule_metadata.shape[2]
    assert int(schedule_work_counts.min().item()) >= 0

    row_ptr_cpu = row_ptr.cpu()
    q_indices_cpu = q_indices.cpu()
    schedule_cpu = schedule_metadata.cpu()
    counts_cpu = schedule_work_counts.cpu()
    for b in range(B):
        for h in range(H):
            covered = torch.zeros(num_kv_blocks, dtype=torch.int32)
            for work in range(int(counts_cpu[b, h].item())):
                kv = int(schedule_cpu[b, h, work, 0].item())
                start = int(schedule_cpu[b, h, work, 1].item())
                count = int(schedule_cpu[b, h, work, 2].item())
                assert 0 <= kv < num_kv_blocks
                assert 0 < count <= target_q_blocks
                row_start = int(row_ptr_cpu[b, h, kv].item())
                row_end = int(row_ptr_cpu[b, h, kv + 1].item())
                assert row_start <= start < row_end
                assert start + count <= row_end
                covered_before = int(covered[kv].item())
                assert start == row_start + covered_before
                covered[kv] += count
            torch.testing.assert_close(
                covered,
                (row_ptr_cpu[b, h, 1:] - row_ptr_cpu[b, h, :-1]).to(torch.int32),
            )


def _assert_csr_qrange_schedule_matches_row_ptr(
    row_ptr,
    q_indices,
    schedule_metadata,
    schedule_work_counts,
    *,
    q_bucket_size_blocks,
    num_q_blocks,
):
    B, H, num_kv_blocks_plus_1 = row_ptr.shape
    num_kv_blocks = num_kv_blocks_plus_1 - 1
    num_q_groups = (num_q_blocks + q_bucket_size_blocks - 1) // q_bucket_size_blocks
    expected_capacity = num_q_groups * num_kv_blocks
    assert schedule_metadata.shape == (B, H, expected_capacity, 4)
    assert schedule_work_counts.shape == (B, H)
    torch.testing.assert_close(
        schedule_work_counts.cpu(),
        torch.full((B, H), expected_capacity, dtype=torch.int32),
    )

    row_ptr_cpu = row_ptr.cpu()
    q_indices_cpu = q_indices.cpu()
    schedule_cpu = schedule_metadata.cpu()
    for b in range(B):
        for h in range(H):
            for kv in range(num_kv_blocks):
                row_start = int(row_ptr_cpu[b, h, kv].item())
                row_end = int(row_ptr_cpu[b, h, kv + 1].item())
                cursor = row_start
                for q_group in range(num_q_groups):
                    work = q_group * num_kv_blocks + kv
                    meta = schedule_cpu[b, h, work]
                    start = int(meta[1].item())
                    count = int(meta[2].item())
                    assert int(meta[0].item()) == kv
                    assert int(meta[3].item()) == q_group
                    assert start == cursor
                    assert row_start <= start <= row_end
                    assert start + count <= row_end
                    if count > 0:
                        q_slice = q_indices_cpu[start:start + count]
                        assert int(q_slice.min().item()) >= q_group * q_bucket_size_blocks
                        assert int(q_slice.max().item()) < (q_group + 1) * q_bucket_size_blocks
                    cursor = start + count
                assert cursor == row_end


FROZEN_BWD_CONFIGS = [
    (1, 4, 116160, 118528, 128),
    (1, 4, 109312, 111040, 128),
    (1, 4, 216832, 219200, 128),
    (1, 1, 216832, 219200, 128),
    (1, 4, 349440, 351168, 128),
    (1, 4, 695040, 697408, 128),
    (1, 1, 695040, 697408, 128),
]


def _torch_ref_bwd(q, k, v, dout, attn_bias, softmax_scale, upcast=True):
    """Reference bwd via torch autograd. All tensors are ``BHSD``.

    When ``upcast=True`` we compute in fp32 — the "gold" reference.
    When ``upcast=False`` we stay in bf16 and reorder ops — this measures
    the per-op rounding spread of any correct bf16 implementation and is
    used as the tolerance baseline (same strategy as the fwd test).
    """
    dtype = torch.float32 if upcast else q.dtype
    q_r = q.detach().to(dtype).requires_grad_()
    k_r = k.detach().to(dtype).requires_grad_()
    v_r = v.detach().to(dtype).requires_grad_()
    if upcast:
        scores = torch.einsum("bhtd,bhsd->bhts", q_r * softmax_scale, k_r)
    else:
        scores = torch.einsum("bhtd,bhsd->bhts", q_r, k_r * softmax_scale)
    scores = scores + attn_bias.to(scores.dtype)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhts,bhsd->bhtd", attn, v_r)
    out = torch.nan_to_num(out, nan=0.0)
    lse = torch.logsumexp(scores, dim=-1)
    out.backward(dout.to(dtype))
    return (
        out.detach().to(dout.dtype),
        lse.detach().float(),
        q_r.grad.detach(),
        k_r.grad.detach(),
        v_r.grad.detach(),
    )


def _test_bwd_single(
    bs,
    seqlen_q,
    seqlen_k,
    nheads,
    *,
    use_variable_block_nums=False,
    impl="baseline",
    q_bucket_size_blocks=512,
):
    """One bwd correctness iteration."""
    device = "cuda"
    dtype = torch.bfloat16
    d = 128

    torch.manual_seed(0)
    torch.cuda.empty_cache()

    q = torch.randn(bs, nheads, seqlen_q, d, device=device, dtype=dtype)
    k = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    v = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    dout = torch.randn(bs, nheads, seqlen_q, d, device=device, dtype=dtype)

    q2k_block_nums = None
    if use_variable_block_nums:
        q2k_block_index, q2k_block_nums, block_sizes = make_random_variable_block_sparse_args(
            bs, seqlen_q, seqlen_k, nheads, blk_m=BLK, blk_n=BLK, device=device,
        )
        block_sparse_num = 0
    else:
        q2k_block_index, block_sparse_num, block_sizes = make_random_block_sparse_args(
            bs, seqlen_q, seqlen_k, nheads, blk_m=BLK, blk_n=BLK, device=device,
        )

    attn_bias = block_sparse_to_attn_bias(
        q2k_block_index, block_sparse_num, block_sizes, seqlen_q, seqlen_k,
        blk_m=BLK, blk_n=BLK, q2k_block_nums=q2k_block_nums,
    )

    softmax_scale = 1.0 / math.sqrt(d)
    out_ref, lse_ref, dq_ref, dk_ref, dv_ref = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale,
    )
    _, _, dq_pt, dk_pt, dv_pt = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale, upcast=False,
    )

    # Zero NaNs caused by 0 * (-inf) on fully-masked rows.
    dq_ref = torch.nan_to_num(dq_ref, nan=0.0, posinf=0.0, neginf=0.0)
    dk_ref = torch.nan_to_num(dk_ref, nan=0.0, posinf=0.0, neginf=0.0)
    dv_ref = torch.nan_to_num(dv_ref, nan=0.0, posinf=0.0, neginf=0.0)
    dq_pt = torch.nan_to_num(dq_pt.float(), nan=0.0, posinf=0.0, neginf=0.0)
    dk_pt = torch.nan_to_num(dk_pt.float(), nan=0.0, posinf=0.0, neginf=0.0)
    dv_pt = torch.nan_to_num(dv_pt.float(), nan=0.0, posinf=0.0, neginf=0.0)

    if impl == "baseline":
        dq, dk, dv = bsa_attn_bwd(
            dout, q, k, v, out_ref, lse_ref,
            q2k_block_index, block_sparse_num, block_sizes,
            q2k_block_nums=q2k_block_nums,
            softmax_scale=softmax_scale,
        )
    elif impl == "qbuck":
        dq, dk, dv = bsa_attn_bwd_qbucket(
            dout, q, k, v, out_ref, lse_ref,
            q2k_block_index, block_sparse_num, block_sizes,
            q2k_block_nums=q2k_block_nums,
            softmax_scale=softmax_scale,
            q_bucket_size_blocks=q_bucket_size_blocks,
        )
    elif impl == "csr":
        dq, dk, dv = bsa_attn_bwd(
            dout, q, k, v, out_ref, lse_ref,
            q2k_block_index, block_sparse_num, block_sizes,
            q2k_block_nums=q2k_block_nums,
            softmax_scale=softmax_scale,
            use_k2q_csr=True,
        )
    else:
        raise ValueError(f"unknown bwd correctness impl: {impl}")

    def _max_abs(a, b):
        return (a.float() - b.float()).abs().max().item()

    dq_diff = _max_abs(dq, dq_ref)
    dk_diff = _max_abs(dk, dk_ref)
    dv_diff = _max_abs(dv, dv_ref)

    # pt-diff baseline: a correct bf16 impl is within ~3x the ordering spread
    # of the fp32 ref (similar shape to the fwd test).
    def _tol(ref, pt):
        pt_diff = (pt - ref).abs().max().item()
        bf16_eps = (ref + 0.3 - 0.3 - ref).abs().max().item()
        return 3 * pt_diff + 3 * bf16_eps + 5e-3

    dq_tol = _tol(dq_ref, dq_pt)
    dk_tol = _tol(dk_ref, dk_pt)
    dv_tol = _tol(dv_ref, dv_pt)
    passed = dq_diff <= dq_tol and dk_diff <= dk_tol and dv_diff <= dv_tol

    mode_str = "var_bsn" if use_variable_block_nums else f"sparse_num={block_sparse_num}"
    tag = "PASS" if passed else "FAIL"
    print(
        f"  {tag} {impl} bs={bs} sq={seqlen_q} sk={seqlen_k} h={nheads} d={d} {mode_str}: "
        f"dq={dq_diff:.4f}/{dq_tol:.4f} "
        f"dk={dk_diff:.4f}/{dk_tol:.4f} "
        f"dv={dv_diff:.4f}/{dv_tol:.4f}"
    )
    assert passed, (
        f"dq_diff={dq_diff}>tol={dq_tol}, dk_diff={dk_diff}>tol={dk_tol}, "
        f"dv_diff={dv_diff}>tol={dv_tol}"
    )


def _make_dense_csr_match_case(use_variable_block_nums, block_size_mode):
    device = "cuda"
    dtype = torch.bfloat16
    bs, seqlen_q, seqlen_k, nheads, d = 2, 128, 512, 4, 128
    topk = 4

    use_block_sizes = block_size_mode != "none"
    q = torch.randn(bs, nheads, seqlen_q, d, device=device, dtype=dtype)
    k = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    v = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    dout = torch.randn(bs, nheads, seqlen_q, d, device=device, dtype=dtype)
    q2k, bsn, block_sizes, q2k_block_nums = make_topk_block_sparse_args(
        bs,
        seqlen_q,
        seqlen_k,
        nheads,
        topk,
        blk_m=BLK,
        blk_n=BLK,
        device=device,
        use_var_block_num=use_variable_block_nums,
        use_block_sizes=use_block_sizes,
        block_size_mode="full" if block_size_mode == "none" else block_size_mode,
    )
    if block_size_mode == "none":
        ref_block_sizes = torch.full(
            ((seqlen_k + BLK - 1) // BLK,), BLK, dtype=torch.int32, device=device
        )
        kernel_block_sizes = None
    else:
        ref_block_sizes = block_sizes
        kernel_block_sizes = block_sizes

    attn_bias = block_sparse_to_attn_bias(
        q2k,
        bsn,
        ref_block_sizes,
        seqlen_q,
        seqlen_k,
        blk_m=BLK,
        blk_n=BLK,
        q2k_block_nums=q2k_block_nums,
    )
    softmax_scale = 1.0 / math.sqrt(d)
    out_ref, lse_ref, _, _, _ = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale,
    )
    return (
        dout,
        q,
        k,
        v,
        out_ref,
        lse_ref,
        q2k,
        bsn,
        kernel_block_sizes,
        q2k_block_nums,
        softmax_scale,
    )


# ============== Pytest ==============

@pytest.mark.parametrize("use_variable_block_nums", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (64, 256),
        (128, 256),
        (128, 512),
        (256, 512),
        (256, 1024),
        (512, 512),
        (1024, 1024),
    ],
)
def test_flash_bwd_sm100_blk64(seqlen_q, seqlen_k, use_variable_block_nums):
    bs = 2 if seqlen_k <= 512 else 1
    nheads = 4
    _test_bwd_single(bs, seqlen_q, seqlen_k, nheads,
                     use_variable_block_nums=use_variable_block_nums)


@pytest.mark.parametrize("use_variable_block_nums", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 256),
        (256, 512),
        (1024, 1024),
    ],
)
def test_flash_bwd_qbucket_sm100_blk64(seqlen_q, seqlen_k, use_variable_block_nums):
    bs = 2 if seqlen_k <= 512 else 1
    nheads = 4
    _test_bwd_single(
        bs,
        seqlen_q,
        seqlen_k,
        nheads,
        use_variable_block_nums=use_variable_block_nums,
        impl="qbuck",
    )


@pytest.mark.parametrize("use_variable_block_nums", [False, True])
def test_flash_bwd_qbucket_multi_group_sm100_blk64(use_variable_block_nums):
    _test_bwd_single(
        1,
        512,
        768,
        4,
        use_variable_block_nums=use_variable_block_nums,
        impl="qbuck",
        q_bucket_size_blocks=2,
    )


@pytest.mark.parametrize("use_variable_block_nums", [False, True])
def test_convert_q2k_to_k2q_csr_sm100_blk64(use_variable_block_nums):
    bs, seqlen_q, seqlen_k, nheads = 2, 256, 512, 4
    if use_variable_block_nums:
        q2k, q2k_nums, _ = make_random_variable_block_sparse_args(
            bs, seqlen_q, seqlen_k, nheads, blk_m=BLK, blk_n=BLK, device="cuda",
        )
        bsn = 0
    else:
        q2k, bsn, _ = make_random_block_sparse_args(
            bs, seqlen_q, seqlen_k, nheads, blk_m=BLK, blk_n=BLK, device="cuda",
        )
        q2k_nums = None
    num_q_blocks = (seqlen_q + BLK - 1) // BLK
    num_kv_blocks = (seqlen_k + BLK - 1) // BLK

    dense_idx, dense_num = convert_q2k_to_k2q(
        q2k, bsn, num_kv_blocks, q2k_block_nums=q2k_nums,
    )
    row_ptr, q_indices = convert_q2k_to_k2q_csr(
        q2k, bsn, num_kv_blocks, q2k_block_nums=q2k_nums,
    )
    (
        sched_row_ptr,
        sched_q_indices,
        schedule_metadata,
        schedule_work_counts,
    ) = convert_q2k_to_k2q_csr(
        q2k,
        bsn,
        num_kv_blocks,
        q2k_block_nums=q2k_nums,
        return_schedule=True,
        schedule_target_q_blocks=1,
        schedule_mode="row_chunk",
    )
    (
        qrange_row_ptr,
        qrange_q_indices,
        qrange_schedule_metadata,
        qrange_schedule_work_counts,
    ) = convert_q2k_to_k2q_csr(
        q2k,
        bsn,
        num_kv_blocks,
        q2k_block_nums=q2k_nums,
        return_schedule=True,
        schedule_mode="qrange",
        schedule_q_bucket_size_blocks=2,
    )
    _assert_total_packed_csr(
        row_ptr,
        q_indices,
        bs=bs,
        nheads=nheads,
        num_q_blocks=num_q_blocks,
        num_kv_blocks=num_kv_blocks,
        block_sparse_num=bsn,
        q2k_block_nums=q2k_nums,
    )
    torch.testing.assert_close(sched_row_ptr, row_ptr)
    torch.testing.assert_close(sched_q_indices, q_indices)
    _assert_csr_schedule_matches_row_ptr(
        sched_row_ptr,
        sched_q_indices,
        schedule_metadata,
        schedule_work_counts,
        target_q_blocks=1,
    )
    torch.testing.assert_close(qrange_row_ptr, row_ptr)
    torch.testing.assert_close(qrange_q_indices, q_indices)
    _assert_csr_qrange_schedule_matches_row_ptr(
        qrange_row_ptr,
        qrange_q_indices,
        qrange_schedule_metadata,
        qrange_schedule_work_counts,
        q_bucket_size_blocks=2,
        num_q_blocks=num_q_blocks,
    )
    csr_idx, csr_num = materialize_k2q_csr_to_dense(row_ptr, q_indices, num_q_blocks)
    torch.testing.assert_close(csr_num, dense_num)
    for b in range(bs):
        for h in range(nheads):
            for kv in range(num_kv_blocks):
                n = int(dense_num[b, h, kv].item())
                torch.testing.assert_close(
                    csr_idx[b, h, kv, :n],
                    dense_idx[b, h, kv, :n],
                )


@pytest.mark.parametrize("use_variable_block_nums", [False, True])
def test_flash_bwd_csr_sm100_blk64(use_variable_block_nums):
    _test_bwd_single(
        2,
        128,
        256,
        4,
        use_variable_block_nums=use_variable_block_nums,
        impl="csr",
    )


@pytest.mark.parametrize("use_variable_block_nums", [False, True])
@pytest.mark.parametrize("block_size_mode", ["none", "full", "random"])
def test_flash_bwd_csr_matches_dense_prebuilt_sm100_blk64(
    use_variable_block_nums,
    block_size_mode,
):
    (
        dout,
        q,
        k,
        v,
        out_ref,
        lse_ref,
        q2k,
        bsn,
        block_sizes,
        q2k_block_nums,
        softmax_scale,
    ) = _make_dense_csr_match_case(use_variable_block_nums, block_size_mode)
    num_q_blocks = (q.shape[2] + BLK - 1) // BLK
    num_kv_blocks = (k.shape[2] + BLK - 1) // BLK
    dense_idx, dense_num = convert_q2k_to_k2q(
        q2k, bsn, num_kv_blocks, q2k_block_nums=q2k_block_nums,
    )
    row_ptr, q_indices = convert_q2k_to_k2q_csr(
        q2k, bsn, num_kv_blocks, q2k_block_nums=q2k_block_nums,
    )
    _assert_total_packed_csr(
        row_ptr,
        q_indices,
        bs=q.shape[0],
        nheads=q.shape[1],
        num_q_blocks=num_q_blocks,
        num_kv_blocks=num_kv_blocks,
        block_sparse_num=bsn,
        q2k_block_nums=q2k_block_nums,
    )
    csr_idx, csr_num = materialize_k2q_csr_to_dense(row_ptr, q_indices, num_q_blocks)
    torch.testing.assert_close(csr_num, dense_num)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            for kv in range(num_kv_blocks):
                n = int(dense_num[b, h, kv].item())
                torch.testing.assert_close(csr_idx[b, h, kv, :n], dense_idx[b, h, kv, :n])

    dense_workspace = _make_bwd_workspace(q)
    csr_workspace = _make_bwd_workspace(q)
    dq_dense, dk_dense, dv_dense = bsa_attn_bwd(
        dout,
        q,
        k,
        v,
        out_ref,
        lse_ref,
        q2k,
        bsn,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        softmax_scale=softmax_scale,
        use_k2q_csr=False,
        prebuilt_k2q_block_index=dense_idx,
        prebuilt_k2q_block_nums=dense_num,
        workspace=dense_workspace,
    )
    dq_csr, dk_csr, dv_csr = bsa_attn_bwd(
        dout,
        q,
        k,
        v,
        out_ref,
        lse_ref,
        q2k,
        bsn,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        softmax_scale=softmax_scale,
        use_k2q_csr=True,
        k2q_row_ptr=row_ptr,
        k2q_q_indices=q_indices,
        workspace=csr_workspace,
    )

    torch.testing.assert_close(dq_csr, dq_dense, rtol=0, atol=2e-3)
    torch.testing.assert_close(dk_csr, dk_dense, rtol=0, atol=2e-3)
    torch.testing.assert_close(dv_csr, dv_dense, rtol=0, atol=2e-3)


@pytest.mark.parametrize("use_variable_block_nums", [False, True])
@pytest.mark.parametrize("block_size_mode", ["none", "full", "random"])
def test_flash_bwd_csr_scheduled_matches_dense_prebuilt_sm100_blk64(
    use_variable_block_nums,
    block_size_mode,
):
    (
        dout,
        q,
        k,
        v,
        out_ref,
        lse_ref,
        q2k,
        bsn,
        block_sizes,
        q2k_block_nums,
        softmax_scale,
    ) = _make_dense_csr_match_case(use_variable_block_nums, block_size_mode)
    num_kv_blocks = (k.shape[2] + BLK - 1) // BLK
    dense_idx, dense_num = convert_q2k_to_k2q(
        q2k, bsn, num_kv_blocks, q2k_block_nums=q2k_block_nums,
    )
    (
        row_ptr,
        q_indices,
        schedule_metadata,
        schedule_work_counts,
    ) = convert_q2k_to_k2q_csr(
        q2k,
        bsn,
        num_kv_blocks,
        q2k_block_nums=q2k_block_nums,
        return_schedule=True,
        schedule_mode="qrange",
        schedule_q_bucket_size_blocks=1,
    )

    dense_workspace = _make_bwd_workspace(q)
    scheduled_workspace = _make_qbucket_bwd_workspace(q, k)
    dq_dense, dk_dense, dv_dense = bsa_attn_bwd(
        dout,
        q,
        k,
        v,
        out_ref,
        lse_ref,
        q2k,
        bsn,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        softmax_scale=softmax_scale,
        use_k2q_csr=False,
        prebuilt_k2q_block_index=dense_idx,
        prebuilt_k2q_block_nums=dense_num,
        workspace=dense_workspace,
    )
    dq_sched, dk_sched, dv_sched = bsa_attn_bwd_csr_scheduled(
        dout,
        q,
        k,
        v,
        out_ref,
        lse_ref,
        q2k,
        bsn,
        block_sizes,
        q2k_block_nums=q2k_block_nums,
        softmax_scale=softmax_scale,
        k2q_row_ptr=row_ptr,
        k2q_q_indices=q_indices,
        k2q_schedule_metadata=schedule_metadata,
        k2q_schedule_work_counts=schedule_work_counts,
        schedule_mode="qrange",
        schedule_q_bucket_size_blocks=1,
        workspace=scheduled_workspace,
    )

    torch.testing.assert_close(dq_sched, dq_dense, rtol=0, atol=5e-3)
    torch.testing.assert_close(dk_sched, dk_dense, rtol=0, atol=5e-3)
    torch.testing.assert_close(dv_sched, dv_dense, rtol=0, atol=5e-3)


# ============== Quick test (make ttb) ==============

def run_quick_tests():
    print("Quick correctness tests (bwd blk64)")
    print("=" * 70)
    configs = [
        (1, 64, 256, 4),
        (1, 128, 256, 4),
        (1, 128, 512, 4),
        (1, 256, 512, 4),
        (1, 256, 1024, 4),
        (2, 512, 512, 4),
        (1, 1024, 1024, 4),
    ]
    for bs, sq, sk, h in configs:
        _test_bwd_single(bs, sq, sk, h)

    print("-" * 70)
    print("Variable block_sparse_num tests (bwd blk64)")
    var_configs = [
        (1, 128, 512, 4),
        (1, 256, 1024, 4),
        (2, 512, 512, 4),
    ]
    for bs, sq, sk, h in var_configs:
        _test_bwd_single(bs, sq, sk, h, use_variable_block_nums=True)

    print("-" * 70)
    print("Q-bucket correctness tests (bwd blk64)")
    qbucket_configs = [
        (1, 128, 256, 4, False),
        (1, 256, 512, 4, False),
        (1, 128, 512, 4, True),
    ]
    for bs, sq, sk, h, use_var in qbucket_configs:
        _test_bwd_single(
            bs,
            sq,
            sk,
            h,
            use_variable_block_nums=use_var,
            impl="qbuck",
        )

    print("=" * 70)
    print("All bwd quick tests passed.")


# ============== Benchmark ==============

def _bench_bwd_one(dout, q, k, v, out, lse, q2k, bsn, bsize, q2k_block_nums=None,
                    niters=10, impl="baseline", q_bucket_size_blocks=None):
    """Warmup + time a single bwd call. Returns median ms."""
    def _run_once():
        if impl == "baseline":
            bsa_attn_bwd(dout, q, k, v, out, lse, q2k, bsn, bsize,
                         q2k_block_nums=q2k_block_nums)
        elif impl == "qbuck":
            bsa_attn_bwd_qbucket(
                dout, q, k, v, out, lse, q2k, bsn, bsize,
                q2k_block_nums=q2k_block_nums,
                q_bucket_size_blocks=q_bucket_size_blocks,
            )
        elif impl == "csr":
            bsa_attn_bwd(
                dout, q, k, v, out, lse, q2k, bsn, bsize,
                q2k_block_nums=q2k_block_nums,
                use_k2q_csr=True,
            )
        elif impl == "csr_scheduled":
            bsa_attn_bwd_csr_scheduled(
                dout, q, k, v, out, lse, q2k, bsn, bsize,
                q2k_block_nums=q2k_block_nums,
            )
        else:
            raise ValueError(f"unknown bwd benchmark impl: {impl}")

    for _ in range(int(os.environ.get("BSA_BWD_WARMUP", "5"))):
        _run_once()
    torch.cuda.synchronize()

    evts = [(torch.cuda.Event(enable_timing=True),
             torch.cuda.Event(enable_timing=True)) for _ in range(niters)]
    for s, e in evts:
        s.record()
        _run_once()
        e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in evts])
    return times[len(times) // 2]


def _cuda_median_ms(run_once, *, niters, warmup=None):
    """Time a CUDA workload with events. Compile work should be inside warmup."""
    if warmup is None:
        warmup = int(os.environ.get("BSA_BWD_WARMUP", "5"))
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    evts = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(niters)
    ]
    for start, end in evts:
        start.record()
        run_once()
        end.record()
    torch.cuda.synchronize()
    times = sorted(start.elapsed_time(end) for start, end in evts)
    return times[len(times) // 2]


def _metadata_mib(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors) / (1024.0 * 1024.0)


def _make_bwd_workspace(q):
    q_len = (q.shape[2] + 7) // 8 * 8
    d = (q.shape[3] + 7) // 8 * 8
    return torch.empty(
        (q_len, q.shape[0], q.shape[1], (d + 2) * 4),
        dtype=torch.uint8,
        device=q.device,
    )


def _make_qbucket_bwd_workspace(q, k):
    q_len = (q.shape[2] + 7) // 8 * 8
    k_len = (k.shape[2] + 7) // 8 * 8
    d = (q.shape[3] + 7) // 8 * 8
    bytes_per_bh = q_len * (d + 2) * 4 + 2 * k_len * d * 4
    return torch.empty((q.shape[0], q.shape[1], bytes_per_bh), dtype=torch.uint8, device=q.device)


def _choose_bench_topk(num_kv_blocks):
    topk_env = os.environ.get("BSA_BWD_BENCH_TOPK")
    if topk_env is None:
        topk_frac = float(os.environ.get("BSA_BWD_BENCH_TOPK_FRAC", "0.1"))
        topk = int(num_kv_blocks * topk_frac)
    else:
        topk = int(topk_env)
    topk = max(2, min(topk, num_kv_blocks))
    if topk % 2 != 0:
        topk = topk + 1 if topk < num_kv_blocks else topk - 1
    return max(2, topk)


def _make_bwd_topk_args(
    bs,
    seqlen_q,
    seqlen_k,
    nheads,
    topk,
    *,
    device,
    use_block_sizes,
    block_size_mode,
):
    pattern = os.environ.get("BSA_BWD_BENCH_PATTERN", "sliding")
    use_var_block_num = os.environ.get("BSA_BWD_BENCH_VAR_NUM", "0") == "1"
    if pattern == "random":
        return (*make_topk_block_sparse_args(
            bs,
            seqlen_q,
            seqlen_k,
            nheads,
            topk,
            blk_m=BLK,
            blk_n=BLK,
            device=device,
            use_var_block_num=use_var_block_num,
            use_block_sizes=use_block_sizes,
            block_size_mode=block_size_mode,
        ), pattern)
    if pattern != "sliding":
        raise ValueError(f"unknown BSA_BWD_BENCH_PATTERN={pattern!r}")

    num_q_blocks = (seqlen_q + BLK - 1) // BLK
    num_kv_blocks = (seqlen_k + BLK - 1) // BLK
    assert topk <= num_kv_blocks

    q = torch.arange(num_q_blocks, device=device, dtype=torch.int32).view(
        1, 1, num_q_blocks, 1
    )
    h = torch.arange(nheads, device=device, dtype=torch.int32).view(1, nheads, 1, 1)
    b = torch.arange(bs, device=device, dtype=torch.int32).view(bs, 1, 1, 1)
    offs = torch.arange(topk, device=device, dtype=torch.int32).view(1, 1, 1, topk)
    q_stride = int(os.environ.get("BSA_BWD_BENCH_Q_STRIDE", "131"))
    h_stride = int(os.environ.get("BSA_BWD_BENCH_H_STRIDE", "17"))
    b_stride = int(os.environ.get("BSA_BWD_BENCH_B_STRIDE", "29"))
    starts = (q * q_stride + h * h_stride + b * b_stride) % num_kv_blocks
    q2k_block_index = (starts + offs) % num_kv_blocks
    q2k_block_index = q2k_block_index.contiguous()

    if use_block_sizes:
        if block_size_mode == "full":
            block_sizes = torch.full((num_kv_blocks,), BLK, dtype=torch.int32, device=device)
        elif block_size_mode == "random":
            block_sizes = torch.randint(
                1, BLK + 1, (num_kv_blocks,), dtype=torch.int32, device=device
            )
        else:
            raise ValueError(f"unknown block_size_mode: {block_size_mode}")
        last_block_actual = seqlen_k - (num_kv_blocks - 1) * BLK
        if last_block_actual < BLK:
            block_sizes[-1] = min(block_sizes[-1].item(), last_block_actual)
    else:
        block_sizes = None

    q2k_block_nums = None
    if use_var_block_num:
        q2k_block_nums = torch.full(
            (bs, nheads, num_q_blocks), topk, dtype=torch.int32, device=device
        )
    return q2k_block_index, topk, block_sizes, q2k_block_nums, pattern


def _bench_convert_one(q2k, bsn, num_kv_blocks, q2k_block_nums, *, impl, niters):
    if impl == "dense":
        def _run_once():
            convert_q2k_to_k2q(
                q2k, bsn, num_kv_blocks, q2k_block_nums=q2k_block_nums,
            )
    elif impl == "csr":
        def _run_once():
            convert_q2k_to_k2q_csr(
                q2k, bsn, num_kv_blocks, q2k_block_nums=q2k_block_nums,
            )
    elif impl == "csr_schedule":
        def _run_once():
            convert_q2k_to_k2q_csr(
                q2k,
                bsn,
                num_kv_blocks,
                q2k_block_nums=q2k_block_nums,
                return_schedule=True,
                schedule_mode="qrange",
            )
    else:
        raise ValueError(f"unknown conversion impl: {impl}")
    return _cuda_median_ms(_run_once, niters=niters)


def _bench_bwd_prebuilt_one(
    dout,
    q,
    k,
    v,
    out,
    lse,
    q2k,
    bsn,
    bsize,
    q2k_block_nums,
    *,
    impl,
    dense_k2q=None,
    csr_k2q=None,
    csr_schedule_k2q=None,
    niters,
):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    workspace = (
        _make_qbucket_bwd_workspace(q, k)
        if impl == "csr_scheduled_prebuilt"
        else _make_bwd_workspace(q)
    )

    if impl == "dense_prebuilt":
        assert dense_k2q is not None
        dense_idx, dense_num = dense_k2q

        def _run_once():
            bsa_attn_bwd(
                dout,
                q,
                k,
                v,
                out,
                lse,
                q2k,
                bsn,
                bsize,
                q2k_block_nums=q2k_block_nums,
                dq=dq,
                dk=dk,
                dv=dv,
                use_k2q_csr=False,
                prebuilt_k2q_block_index=dense_idx,
                prebuilt_k2q_block_nums=dense_num,
                workspace=workspace,
            )
    elif impl == "csr_prebuilt":
        assert csr_k2q is not None
        row_ptr, q_indices = csr_k2q

        def _run_once():
            bsa_attn_bwd(
                dout,
                q,
                k,
                v,
                out,
                lse,
                q2k,
                bsn,
                bsize,
                q2k_block_nums=q2k_block_nums,
                dq=dq,
                dk=dk,
                dv=dv,
                use_k2q_csr=True,
                k2q_row_ptr=row_ptr,
                k2q_q_indices=q_indices,
                workspace=workspace,
            )
    elif impl == "csr_scheduled_prebuilt":
        assert csr_schedule_k2q is not None
        row_ptr, q_indices, schedule_metadata, schedule_work_counts = csr_schedule_k2q

        def _run_once():
            bsa_attn_bwd_csr_scheduled(
                dout,
                q,
                k,
                v,
                out,
                lse,
                q2k,
                bsn,
                bsize,
                q2k_block_nums=q2k_block_nums,
                dq=dq,
                dk=dk,
                dv=dv,
                k2q_row_ptr=row_ptr,
                k2q_q_indices=q_indices,
                k2q_schedule_metadata=schedule_metadata,
                k2q_schedule_work_counts=schedule_work_counts,
                workspace=workspace,
            )
    else:
        raise ValueError(f"unknown prebuilt bwd impl: {impl}")

    return _cuda_median_ms(_run_once, niters=niters)


def _bench_bwd_prebuilt_pair(
    dout,
    q,
    k,
    v,
    out,
    lse,
    q2k,
    bsn,
    bsize,
    q2k_block_nums,
    *,
    dense_k2q,
    csr_k2q,
    niters,
):
    dense_idx, dense_num = dense_k2q
    row_ptr, q_indices = csr_k2q
    dense_dq = torch.empty_like(q)
    dense_dk = torch.empty_like(k)
    dense_dv = torch.empty_like(v)
    dense_workspace = _make_bwd_workspace(q)
    csr_dq = torch.empty_like(q)
    csr_dk = torch.empty_like(k)
    csr_dv = torch.empty_like(v)
    csr_workspace = _make_bwd_workspace(q)

    def _run_dense():
        bsa_attn_bwd(
            dout,
            q,
            k,
            v,
            out,
            lse,
            q2k,
            bsn,
            bsize,
            q2k_block_nums=q2k_block_nums,
            dq=dense_dq,
            dk=dense_dk,
            dv=dense_dv,
            use_k2q_csr=False,
            prebuilt_k2q_block_index=dense_idx,
            prebuilt_k2q_block_nums=dense_num,
            workspace=dense_workspace,
        )

    def _run_csr():
        bsa_attn_bwd(
            dout,
            q,
            k,
            v,
            out,
            lse,
            q2k,
            bsn,
            bsize,
            q2k_block_nums=q2k_block_nums,
            dq=csr_dq,
            dk=csr_dk,
            dv=csr_dv,
            use_k2q_csr=True,
            k2q_row_ptr=row_ptr,
            k2q_q_indices=q_indices,
            workspace=csr_workspace,
        )

    warmup = int(os.environ.get("BSA_BWD_WARMUP", "5"))
    for _ in range(warmup):
        _run_dense()
        _run_csr()
    torch.cuda.synchronize()

    dense_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(niters)
    ]
    csr_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(niters)
    ]
    for i in range(niters):
        d_start, d_end = dense_events[i]
        c_start, c_end = csr_events[i]
        if i % 2 == 0:
            d_start.record()
            _run_dense()
            d_end.record()
            c_start.record()
            _run_csr()
            c_end.record()
        else:
            c_start.record()
            _run_csr()
            c_end.record()
            d_start.record()
            _run_dense()
            d_end.record()
    torch.cuda.synchronize()
    dense_times = sorted(start.elapsed_time(end) for start, end in dense_events)
    csr_times = sorted(start.elapsed_time(end) for start, end in csr_events)
    return dense_times[len(dense_times) // 2], csr_times[len(csr_times) // 2]


def _profile_start():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()


def _profile_stop():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def _prepare_frozen_bwd_case(cfg_id):
    device = "cuda"
    dtype = torch.bfloat16
    benchmark_seed = int(os.environ.get("BSA_BWD_BENCH_SEED", "0"))
    use_block_sizes = os.environ.get("BSA_BWD_BENCH_BLOCK_SIZES", "1") != "0"
    block_size_mode = os.environ.get("BSA_BWD_BENCH_BLOCK_SIZE_MODE", "full")
    if not use_block_sizes:
        block_size_mode = "none"
    bs, nheads, seqlen_q, seqlen_k, hdim = FROZEN_BWD_CONFIGS[cfg_id]
    torch.manual_seed(benchmark_seed + cfg_id)
    num_kv_blocks = (seqlen_k + BLK - 1) // BLK
    topk = _choose_bench_topk(num_kv_blocks)

    q = torch.randn(bs, nheads, seqlen_q, hdim, device=device, dtype=dtype)
    k = torch.randn(bs, nheads, seqlen_k, hdim, device=device, dtype=dtype)
    v = torch.randn(bs, nheads, seqlen_k, hdim, device=device, dtype=dtype)
    dout = torch.randn_like(q)
    out = torch.zeros_like(q)
    lse = torch.zeros(bs, nheads, seqlen_q, device=device, dtype=torch.float32)
    q2k, bsn, bsize, q2k_block_nums, pattern = _make_bwd_topk_args(
        bs,
        seqlen_q,
        seqlen_k,
        nheads,
        topk,
        device=device,
        use_block_sizes=use_block_sizes,
        block_size_mode=block_size_mode,
    )
    return {
        "cfg": (bs, nheads, seqlen_q, seqlen_k, hdim),
        "topk": topk,
        "pattern": pattern,
        "q": q,
        "k": k,
        "v": v,
        "dout": dout,
        "out": out,
        "lse": lse,
        "q2k": q2k,
        "bsn": bsn,
        "bsize": bsize,
        "q2k_block_nums": q2k_block_nums,
        "num_kv_blocks": num_kv_blocks,
    }


def run_ncu_bwd_once():
    cfg_id = int(os.environ.get("BSA_NCU_CONFIG_ID", "0"))
    impl = os.environ.get("BSA_NCU_IMPL", "csr")
    repeats = int(os.environ.get("BSA_NCU_REPEATS", "1"))
    warmup = int(os.environ.get("BSA_NCU_WARMUP", "1"))
    if impl not in ("dense", "csr", "csr_scheduled"):
        raise ValueError("BSA_NCU_IMPL must be 'dense', 'csr', or 'csr_scheduled'")

    case = _prepare_frozen_bwd_case(cfg_id)
    q = case["q"]
    if impl == "dense":
        if os.environ.get("BSA_NCU_DENSE_FROM_CSR", "1") == "1":
            dense_row_ptr, dense_q_indices = convert_q2k_to_k2q_csr(
                case["q2k"],
                case["bsn"],
                case["num_kv_blocks"],
                q2k_block_nums=case["q2k_block_nums"],
            )
            dense_idx, dense_num = materialize_k2q_csr_to_dense(
                dense_row_ptr,
                dense_q_indices,
                case["q2k"].shape[2],
            )
        else:
            dense_idx, dense_num = convert_q2k_to_k2q(
                case["q2k"],
                case["bsn"],
                case["num_kv_blocks"],
                q2k_block_nums=case["q2k_block_nums"],
            )
    elif impl == "csr":
        row_ptr, q_indices = convert_q2k_to_k2q_csr(
            case["q2k"],
            case["bsn"],
            case["num_kv_blocks"],
            q2k_block_nums=case["q2k_block_nums"],
        )
    else:
        (
            row_ptr,
            q_indices,
            schedule_metadata,
            schedule_work_counts,
        ) = convert_q2k_to_k2q_csr(
            case["q2k"],
            case["bsn"],
            case["num_kv_blocks"],
            q2k_block_nums=case["q2k_block_nums"],
            return_schedule=True,
            schedule_mode="qrange",
        )
    dq = torch.empty_like(q)
    dk = torch.empty_like(case["k"])
    dv = torch.empty_like(case["v"])
    workspace = (
        _make_qbucket_bwd_workspace(q, case["k"])
        if impl == "csr_scheduled"
        else _make_bwd_workspace(q)
    )

    def _run_dense():
        bsa_attn_bwd(
            case["dout"],
            q,
            case["k"],
            case["v"],
            case["out"],
            case["lse"],
            case["q2k"],
            case["bsn"],
            case["bsize"],
            q2k_block_nums=case["q2k_block_nums"],
            dq=dq,
            dk=dk,
            dv=dv,
            use_k2q_csr=False,
            prebuilt_k2q_block_index=dense_idx,
            prebuilt_k2q_block_nums=dense_num,
            workspace=workspace,
        )

    def _run_csr():
        bsa_attn_bwd(
            case["dout"],
            q,
            case["k"],
            case["v"],
            case["out"],
            case["lse"],
            case["q2k"],
            case["bsn"],
            case["bsize"],
            q2k_block_nums=case["q2k_block_nums"],
            dq=dq,
            dk=dk,
            dv=dv,
            use_k2q_csr=True,
            k2q_row_ptr=row_ptr,
            k2q_q_indices=q_indices,
            workspace=workspace,
        )

    def _run_csr_scheduled():
        bsa_attn_bwd_csr_scheduled(
            case["dout"],
            q,
            case["k"],
            case["v"],
            case["out"],
            case["lse"],
            case["q2k"],
            case["bsn"],
            case["bsize"],
            q2k_block_nums=case["q2k_block_nums"],
            dq=dq,
            dk=dk,
            dv=dv,
            k2q_row_ptr=row_ptr,
            k2q_q_indices=q_indices,
            k2q_schedule_metadata=schedule_metadata,
            k2q_schedule_work_counts=schedule_work_counts,
            workspace=workspace,
        )

    if impl == "dense":
        run_once = _run_dense
    elif impl == "csr":
        run_once = _run_csr
    else:
        run_once = _run_csr_scheduled
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    print(
        f"NCU bwd target cfg={cfg_id} impl={impl} shape={case['cfg']} "
        f"topk={case['topk']} pattern={case['pattern']} repeats={repeats}"
    )
    _profile_start()
    torch.cuda.nvtx.range_push(f"ncu_bwd_{impl}_cfg{cfg_id}")
    for _ in range(repeats):
        run_once()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    _profile_stop()


def run_ncu_convert_once():
    cfg_id = int(os.environ.get("BSA_NCU_CONFIG_ID", "0"))
    impl = os.environ.get("BSA_NCU_IMPL", "csr")
    repeats = int(os.environ.get("BSA_NCU_REPEATS", "1"))
    warmup = int(os.environ.get("BSA_NCU_WARMUP", "1"))
    if impl not in ("dense", "csr", "csr_schedule"):
        raise ValueError("BSA_NCU_IMPL must be 'dense', 'csr', or 'csr_schedule'")

    case = _prepare_frozen_bwd_case(cfg_id)

    def _run_dense():
        convert_q2k_to_k2q(
            case["q2k"],
            case["bsn"],
            case["num_kv_blocks"],
            q2k_block_nums=case["q2k_block_nums"],
        )

    def _run_csr():
        convert_q2k_to_k2q_csr(
            case["q2k"],
            case["bsn"],
            case["num_kv_blocks"],
            q2k_block_nums=case["q2k_block_nums"],
        )

    def _run_csr_schedule():
        convert_q2k_to_k2q_csr(
            case["q2k"],
            case["bsn"],
            case["num_kv_blocks"],
            q2k_block_nums=case["q2k_block_nums"],
            return_schedule=True,
            schedule_mode="qrange",
        )

    if impl == "dense":
        run_once = _run_dense
    elif impl == "csr":
        run_once = _run_csr
    else:
        run_once = _run_csr_schedule
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    print(
        f"NCU convert target cfg={cfg_id} impl={impl} shape={case['cfg']} "
        f"topk={case['topk']} pattern={case['pattern']} repeats={repeats}"
    )
    _profile_start()
    torch.cuda.nvtx.range_push(f"ncu_convert_{impl}_cfg{cfg_id}")
    for _ in range(repeats):
        run_once()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    _profile_stop()


def run_csr_kernel_only_benchmark_suite():
    device = "cuda"
    dtype = torch.bfloat16
    benchmark_seed = int(os.environ.get("BSA_BWD_BENCH_SEED", "0"))
    niters = int(os.environ.get("BSA_BWD_BENCH_ITERS", "5"))
    conv_niters = int(os.environ.get("BSA_BWD_BENCH_CONV_ITERS", str(niters)))
    use_block_sizes = os.environ.get("BSA_BWD_BENCH_BLOCK_SIZES", "1") != "0"
    block_size_mode = os.environ.get("BSA_BWD_BENCH_BLOCK_SIZE_MODE", "full")
    if not use_block_sizes:
        block_size_mode = "none"
    start = int(os.environ.get("BSA_BWD_BENCH_START", "0"))
    limit = int(os.environ.get("BSA_BWD_BENCH_LIMIT", str(len(FROZEN_BWD_CONFIGS) - start)))
    configs = FROZEN_BWD_CONFIGS[start:start + limit]
    bench_order = os.environ.get("BSA_BWD_BENCH_ORDER", "interleave")

    print("\nCSR prebuilt-metadata bwd benchmark (blk64)")
    print(
        f"seed={benchmark_seed} niters={niters} conv_niters={conv_niters} "
        f"warmup={os.environ.get('BSA_BWD_WARMUP', '5')} "
        f"block_sizes={block_size_mode} order={bench_order}"
    )
    print(
        f"{'shape':<38} {'pattern':>8} {'topk':>6} "
        f"{'dense':>10} {'csr':>10} {'sched':>10} "
        f"{'dCsr':>8} {'dSched':>8} "
        f"{'denseTF':>9} {'csrTF':>9} {'schedTF':>9} "
        f"{'convD':>9} {'convC':>9} {'convS':>9} "
        f"{'metaD':>9} {'metaC':>9} {'metaS':>9} {'peak':>8}"
    )
    print("-" * 185)

    for cfg_offset, (bs, nheads, seqlen_q, seqlen_k, hdim) in enumerate(configs):
        cfg_id = start + cfg_offset
        torch.manual_seed(benchmark_seed + cfg_id)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        num_kv_blocks = (seqlen_k + BLK - 1) // BLK
        topk = _choose_bench_topk(num_kv_blocks)

        q = torch.randn(bs, nheads, seqlen_q, hdim, device=device, dtype=dtype)
        k = torch.randn(bs, nheads, seqlen_k, hdim, device=device, dtype=dtype)
        v = torch.randn(bs, nheads, seqlen_k, hdim, device=device, dtype=dtype)
        dout = torch.randn_like(q)
        out = torch.zeros_like(q)
        lse = torch.zeros(bs, nheads, seqlen_q, device=device, dtype=torch.float32)
        q2k, bsn, bsize, q2k_block_nums, pattern = _make_bwd_topk_args(
            bs,
            seqlen_q,
            seqlen_k,
            nheads,
            topk,
            device=device,
            use_block_sizes=use_block_sizes,
            block_size_mode=block_size_mode,
        )
        torch.cuda.synchronize()

        dense_conv_ms = _bench_convert_one(
            q2k, bsn, num_kv_blocks, q2k_block_nums, impl="dense", niters=conv_niters,
        )
        csr_conv_ms = _bench_convert_one(
            q2k, bsn, num_kv_blocks, q2k_block_nums, impl="csr", niters=conv_niters,
        )
        sched_conv_ms = _bench_convert_one(
            q2k,
            bsn,
            num_kv_blocks,
            q2k_block_nums,
            impl="csr_schedule",
            niters=conv_niters,
        )
        dense_idx, dense_num = convert_q2k_to_k2q(
            q2k, bsn, num_kv_blocks, q2k_block_nums=q2k_block_nums,
        )
        row_ptr, q_indices = convert_q2k_to_k2q_csr(
            q2k, bsn, num_kv_blocks, q2k_block_nums=q2k_block_nums,
        )
        sched_row_ptr, sched_q_indices, schedule_metadata, schedule_work_counts = (
            convert_q2k_to_k2q_csr(
                q2k,
                bsn,
                num_kv_blocks,
                q2k_block_nums=q2k_block_nums,
                return_schedule=True,
                schedule_mode="qrange",
            )
        )
        torch.cuda.synchronize()

        def _time_dense():
            return _bench_bwd_prebuilt_one(
                dout,
                q,
                k,
                v,
                out,
                lse,
                q2k,
                bsn,
                bsize,
                q2k_block_nums,
                impl="dense_prebuilt",
                dense_k2q=(dense_idx, dense_num),
                niters=niters,
            )

        def _time_csr():
            return _bench_bwd_prebuilt_one(
                dout,
                q,
                k,
                v,
                out,
                lse,
                q2k,
                bsn,
                bsize,
                q2k_block_nums,
                impl="csr_prebuilt",
                csr_k2q=(row_ptr, q_indices),
                niters=niters,
            )

        def _time_sched():
            return _bench_bwd_prebuilt_one(
                dout,
                q,
                k,
                v,
                out,
                lse,
                q2k,
                bsn,
                bsize,
                q2k_block_nums,
                impl="csr_scheduled_prebuilt",
                csr_schedule_k2q=(
                    sched_row_ptr,
                    sched_q_indices,
                    schedule_metadata,
                    schedule_work_counts,
                ),
                niters=niters,
            )

        if bench_order == "interleave":
            dense_ms, csr_ms = _bench_bwd_prebuilt_pair(
                dout,
                q,
                k,
                v,
                out,
                lse,
                q2k,
                bsn,
                bsize,
                q2k_block_nums,
                dense_k2q=(dense_idx, dense_num),
                csr_k2q=(row_ptr, q_indices),
                niters=niters,
            )
            sched_ms = _time_sched()
        elif bench_order == "dense_first":
            dense_ms = _time_dense()
            csr_ms = _time_csr()
            sched_ms = _time_sched()
        elif bench_order == "csr_first":
            csr_ms = _time_csr()
            sched_ms = _time_sched()
            dense_ms = _time_dense()
        else:
            raise ValueError(f"unknown BSA_BWD_BENCH_ORDER={bench_order!r}")
        effective_sk = topk * BLK
        dense_tflops = (
            bwd_flops(bs, nheads, seqlen_q, effective_sk, hdim, hdim)
            / (dense_ms * 1e-3)
            / 1e12
        )
        csr_tflops = (
            bwd_flops(bs, nheads, seqlen_q, effective_sk, hdim, hdim)
            / (csr_ms * 1e-3)
            / 1e12
        )
        sched_tflops = (
            bwd_flops(bs, nheads, seqlen_q, effective_sk, hdim, hdim)
            / (sched_ms * 1e-3)
            / 1e12
        )
        csr_delta = (csr_ms / dense_ms - 1.0) * 100.0
        sched_delta = (sched_ms / dense_ms - 1.0) * 100.0
        dense_meta_mib = _metadata_mib(dense_idx, dense_num)
        csr_meta_mib = _metadata_mib(row_ptr, q_indices)
        sched_meta_mib = _metadata_mib(
            sched_row_ptr,
            sched_q_indices,
            schedule_metadata,
            schedule_work_counts,
        )
        peak_gib = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        shape_label = f"b{bs}h{nheads} q{seqlen_q} k{seqlen_k}"
        print(
            f"{shape_label:<38} {pattern:>8} {topk:>6} "
            f"{dense_ms:>8.3f}ms {csr_ms:>8.3f}ms {sched_ms:>8.3f}ms "
            f"{csr_delta:>7.2f}% {sched_delta:>7.2f}% "
            f"{dense_tflops:>9.1f} {csr_tflops:>9.1f} {sched_tflops:>9.1f} "
            f"{dense_conv_ms:>8.3f}ms {csr_conv_ms:>8.3f}ms {sched_conv_ms:>8.3f}ms "
            f"{dense_meta_mib:>8.1f}M {csr_meta_mib:>8.1f}M {sched_meta_mib:>8.1f}M "
            f"{peak_gib:>7.1f}G"
        )

        del (
            q,
            k,
            v,
            dout,
            out,
            lse,
            q2k,
            bsize,
            q2k_block_nums,
            dense_idx,
            dense_num,
            row_ptr,
            q_indices,
            sched_row_ptr,
            sched_q_indices,
            schedule_metadata,
            schedule_work_counts,
        )
        gc.collect()
        torch.cuda.empty_cache()


def run_benchmark_suite():
    device = "cuda"
    dtype = torch.bfloat16
    d = 128
    benchmark_seed = int(os.environ.get("BSA_BWD_BENCH_SEED", "0"))
    torch.manual_seed(benchmark_seed)
    bench_impls = [
        x.strip()
        for x in os.environ.get("BSA_BWD_BENCH_IMPL", "qbuck").split(",")
        if x.strip()
    ]
    q_bucket_size_blocks_env = os.environ.get("BSA_Q_BUCKET_BLOCKS")
    q_bucket_size_blocks = (
        int(q_bucket_size_blocks_env) if q_bucket_size_blocks_env else None
    )
    use_block_sizes = os.environ.get("BSA_BWD_BENCH_BLOCK_SIZES", "1") != "0"
    block_size_mode = os.environ.get("BSA_BWD_BENCH_BLOCK_SIZE_MODE", "full")
    if not use_block_sizes:
        block_size_mode = "none"
    # (bs, nheads, seqlen_q, seqlen_k, hdim)
    configs = FROZEN_BWD_CONFIGS
    # topK values to benchmark (0 = dense)
    topk_values = [0, 32, 64, 128, 256]

    print(f"\n{'Config (bwd blk=64)':<54} {'impl':>10} {'ms':>8} {'TFLOPS':>8} {'MFU':>8}")
    print("-" * 85)

    for bs, nheads, seqlen_q, seqlen_k, hdim in configs:
        q = torch.randn(bs, nheads, seqlen_q, hdim, device=device, dtype=dtype)
        k = torch.randn(bs, nheads, seqlen_k, hdim, device=device, dtype=dtype)
        v = torch.randn(bs, nheads, seqlen_k, hdim, device=device, dtype=dtype)
        dout = torch.randn_like(q)
        out = torch.zeros_like(q)
        lse = torch.zeros(bs, nheads, seqlen_q, device=device, dtype=torch.float32)

        num_kv_blocks = (seqlen_k + BLK - 1) // BLK
        topk_values = [int(num_kv_blocks * 0.1)]

        for topk in topk_values:
            if topk == 0:
                q2k, bsn, bsize = make_dense_block_sparse_args(
                    bs, seqlen_q, seqlen_k, nheads, blk_m=BLK, blk_n=BLK, device=device,
                )
                q2k_block_nums = None
                label = f"bs={bs} h={nheads} sq={seqlen_q} sk={seqlen_k} d={hdim} dense"
                effective_sk = seqlen_k
            else:
                if topk > num_kv_blocks:
                    continue
                topk_even = topk if topk % 2 == 0 else topk + 1
                q2k, bsn, bsize, q2k_block_nums = make_topk_block_sparse_args(
                    bs, seqlen_q, seqlen_k, nheads, topk_even,
                    blk_m=BLK, blk_n=BLK, device=device,
                    use_block_sizes=use_block_sizes,
                    block_size_mode=block_size_mode,
                )
                block_sizes_label = (
                    f"bsz={block_size_mode}" if use_block_sizes else "bsz=0"
                )
                label = (
                    f"bs={bs} h={nheads} sq={seqlen_q} sk={seqlen_k} "
                    f"d={hdim} topk={topk_even} {block_sizes_label}"
                )
                effective_sk = topk_even * BLK

            for impl in bench_impls:
                impl_label = impl
                if impl == "qbuck":
                    impl_label = (
                        f"qbuck{q_bucket_size_blocks}"
                        if q_bucket_size_blocks is not None
                        else "qbuckauto"
                    )
                print(f"  {label:<52} {impl_label:>10} ...", end="", flush=True)
                med = _bench_bwd_one(
                    dout, q, k, v, out, lse, q2k, bsn, bsize,
                    q2k_block_nums=q2k_block_nums,
                    impl=impl,
                    q_bucket_size_blocks=q_bucket_size_blocks,
                    niters=int(os.environ.get("BSA_BWD_BENCH_ITERS", "10")),
                )
                f = bwd_flops(bs, nheads, seqlen_q, effective_sk, hdim, hdim)
                tflops = f / (med * 1e-3) / 1e12
                mfu = tflops / 2250.0 * 100.0
                print(
                    f"\r  {label:<52} {impl_label:>10} "
                    f"{med:>8.3f}ms {tflops:>8.1f} tflops {mfu:>8.1f}%"
                )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        run_benchmark_suite()
    elif len(sys.argv) > 1 and sys.argv[1] == "csr_benchmark":
        run_csr_kernel_only_benchmark_suite()
    elif len(sys.argv) > 1 and sys.argv[1] == "ncu_bwd":
        run_ncu_bwd_once()
    elif len(sys.argv) > 1 and sys.argv[1] == "ncu_convert":
        run_ncu_convert_once()
    else:
        run_quick_tests()
