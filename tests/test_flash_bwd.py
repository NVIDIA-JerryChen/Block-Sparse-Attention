"""BSA SM100 Backward Kernel — Test / Benchmark

Usage:
    python tests/test_flash_bwd.py              # quick correctness tests
    python tests/test_flash_bwd.py benchmark    # simple dense-bwd benchmark
"""

import os
import sys
import math

# Keep direct script execution (`python tests/test_flash_bwd.py`) working.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from bsa_attn_interface import bsa_attn_bwd
from test_flash_fwd import (
    make_dense_block_sparse_args,
    make_random_block_sparse_args,
    make_random_variable_block_sparse_args,
    make_topk_block_sparse_args,
    block_sparse_to_attn_bias,
)
from utils.bench_utils import bwd_flops


# Native bwd coverage: blk64 for SM90/SM100 and blk128 for SM100/SM110.
BLK = 64
BLK128 = 128


def _full_block_sizes(seqlen_k, device="cuda"):
    num_kv_blocks = (seqlen_k + BLK - 1) // BLK
    block_sizes = torch.full((num_kv_blocks,), BLK, dtype=torch.int32, device=device)
    tail = seqlen_k - (num_kv_blocks - 1) * BLK
    block_sizes[-1] = tail
    return block_sizes


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def _sanitize_grads(*grads):
    return tuple(
        torch.nan_to_num(g.float(), nan=0.0, posinf=0.0, neginf=0.0)
        for g in grads
    )


def _grad_tols(refs, pts):
    tols = []
    for ref, pt in zip(refs, pts):
        ref = torch.nan_to_num(ref.float(), nan=0.0, posinf=0.0, neginf=0.0)
        pt = torch.nan_to_num(pt.float(), nan=0.0, posinf=0.0, neginf=0.0)
        pt_diff = (pt - ref).abs().max().item()
        bf16_eps = (ref + 0.3 - 0.3 - ref).abs().max().item()
        tols.append(3 * pt_diff + 3 * bf16_eps + 5e-3)
    return tuple(tols)


def _assert_bwd_close(label, grads, refs, tols):
    diffs = tuple(_max_abs(g, r) for g, r in zip(grads, refs))
    passed = all(diff <= tol for diff, tol in zip(diffs, tols))
    print(
        f"  {'PASS' if passed else 'FAIL'} {label}: "
        f"dq={diffs[0]:.4f}/{tols[0]:.4f} "
        f"dk={diffs[1]:.4f}/{tols[1]:.4f} "
        f"dv={diffs[2]:.4f}/{tols[2]:.4f}"
    )
    assert passed, (
        f"{label}: dq_diff={diffs[0]}>tol={tols[0]}, "
        f"dk_diff={diffs[1]}>tol={tols[1]}, dv_diff={diffs[2]}>tol={tols[2]}"
    )
    return diffs


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
    bucket_size_blocks=None,
    d=128,
    blk=None,
):
    """One bwd correctness iteration."""
    device = "cuda"
    dtype = torch.bfloat16
    if blk is None:
        blk = BLK

    torch.manual_seed(0)
    torch.cuda.empty_cache()

    q = torch.randn(bs, nheads, seqlen_q, d, device=device, dtype=dtype)
    k = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    v = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    dout = torch.randn(bs, nheads, seqlen_q, d, device=device, dtype=dtype)

    q2k_block_nums = None
    if use_variable_block_nums:
        q2k_block_index, q2k_block_nums, block_sizes = make_random_variable_block_sparse_args(
            bs, seqlen_q, seqlen_k, nheads, blk_m=blk, blk_n=blk, device=device,
        )
        block_sparse_num = 0
    else:
        q2k_block_index, block_sparse_num, block_sizes = make_random_block_sparse_args(
            bs, seqlen_q, seqlen_k, nheads, blk_m=blk, blk_n=blk, device=device,
        )

    attn_bias = block_sparse_to_attn_bias(
        q2k_block_index, block_sparse_num, block_sizes, seqlen_q, seqlen_k,
        blk_m=blk, blk_n=blk, q2k_block_nums=q2k_block_nums,
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

    dq, dk, dv = bsa_attn_bwd(
        dout, q, k, v, out_ref, lse_ref,
        q2k_block_index, block_sparse_num, block_sizes,
        q2k_block_nums=q2k_block_nums,
        softmax_scale=softmax_scale,
        bucket_size_blocks=bucket_size_blocks,
    )

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
        f"  {tag} bwd bs={bs} sq={seqlen_q} sk={seqlen_k} h={nheads} d={d} blk={blk} {mode_str}: "
        f"dq={dq_diff:.4f}/{dq_tol:.4f} "
        f"dk={dk_diff:.4f}/{dk_tol:.4f} "
        f"dv={dv_diff:.4f}/{dv_tol:.4f}"
    )
    assert passed, (
        f"dq_diff={dq_diff}>tol={dq_tol}, dk_diff={dk_diff}>tol={dk_tol}, "
        f"dv_diff={dv_diff}>tol={dv_tol}"
    )


def _make_dense_blk64_args(batch_size, seqlen_q, seqlen_k, nheads, device="cuda"):
    num_q_blocks = (seqlen_q + BLK - 1) // BLK
    num_kv_blocks = (seqlen_k + BLK - 1) // BLK
    indices = torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
    q2k_block_index = indices[None, None, None, :].expand(
        batch_size, nheads, num_q_blocks, num_kv_blocks
    ).contiguous()
    block_sizes = torch.full((num_kv_blocks,), BLK, dtype=torch.int32, device=device)
    tail = seqlen_k - (num_kv_blocks - 1) * BLK
    block_sizes[-1] = tail
    return q2k_block_index, num_kv_blocks, block_sizes


def _test_bwd_dense_single(bs, seqlen_q, seqlen_k, nheads):
    device = "cuda"
    dtype = torch.bfloat16
    d = 128

    torch.manual_seed(0)
    torch.cuda.empty_cache()

    q = torch.randn(bs, nheads, seqlen_q, d, device=device, dtype=dtype)
    k = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    v = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    dout = torch.randn_like(q)

    q2k_block_index, block_sparse_num, block_sizes = _make_dense_blk64_args(
        bs, seqlen_q, seqlen_k, nheads, device=device
    )
    attn_bias = block_sparse_to_attn_bias(
        q2k_block_index, block_sparse_num, block_sizes, seqlen_q, seqlen_k,
        blk_m=BLK, blk_n=BLK,
    )

    softmax_scale = 1.0 / math.sqrt(d)
    out_ref, lse_ref, dq_ref, dk_ref, dv_ref = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale,
    )
    _, _, dq_pt, dk_pt, dv_pt = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale, upcast=False,
    )

    dq, dk, dv = bsa_attn_bwd(
        dout, q, k, v, out_ref, lse_ref,
        q2k_block_index, block_sparse_num, block_sizes,
        softmax_scale=softmax_scale,
    )

    def _max_abs(a, b):
        return (a.float() - b.float()).abs().max().item()

    def _tol(ref, pt):
        ref = torch.nan_to_num(ref.float(), nan=0.0, posinf=0.0, neginf=0.0)
        pt = torch.nan_to_num(pt.float(), nan=0.0, posinf=0.0, neginf=0.0)
        pt_diff = (pt - ref).abs().max().item()
        bf16_eps = (ref + 0.3 - 0.3 - ref).abs().max().item()
        return 3 * pt_diff + 3 * bf16_eps + 5e-3

    dq_diff = _max_abs(dq, dq_ref)
    dk_diff = _max_abs(dk, dk_ref)
    dv_diff = _max_abs(dv, dv_ref)
    dq_tol = _tol(dq_ref, dq_pt)
    dk_tol = _tol(dk_ref, dk_pt)
    dv_tol = _tol(dv_ref, dv_pt)
    passed = dq_diff <= dq_tol and dk_diff <= dk_tol and dv_diff <= dv_tol
    print(
        f"  {'PASS' if passed else 'FAIL'} dense bs={bs} sq={seqlen_q} sk={seqlen_k} h={nheads}: "
        f"dq={dq_diff:.4f}/{dq_tol:.4f} "
        f"dk={dk_diff:.4f}/{dk_tol:.4f} "
        f"dv={dv_diff:.4f}/{dv_tol:.4f}"
    )
    assert passed, (
        f"dq_diff={dq_diff}>tol={dq_tol}, dk_diff={dk_diff}>tol={dk_tol}, "
        f"dv_diff={dv_diff}>tol={dv_tol}"
    )


def _make_topk_args_any(bs, seqlen_q, seqlen_k, nheads, topk, block_size, device):
    num_q_blocks = (seqlen_q + block_size - 1) // block_size
    num_kv_blocks = (seqlen_k + block_size - 1) // block_size
    assert topk <= num_kv_blocks
    q2k = torch.empty(
        bs, nheads, num_q_blocks, topk, dtype=torch.int32, device=device
    )
    for b in range(bs):
        for h in range(nheads):
            for q_block in range(num_q_blocks):
                q2k[b, h, q_block] = torch.randperm(
                    num_kv_blocks, device=device
                )[:topk].to(torch.int32)
    block_sizes = torch.full(
        (num_kv_blocks,), block_size, dtype=torch.int32, device=device
    )
    block_sizes[-1] = seqlen_k - (num_kv_blocks - 1) * block_size
    return q2k, topk, block_sizes


def _test_bwd_topk_blk128(
    bs,
    seqlen_q,
    seqlen_k,
    nheads,
    topk,
    use_block_sizes,
    bucket_size_blocks=None,
    d=128,
    shared_kv_blocks=False,
):
    device = "cuda"
    dtype = torch.bfloat16

    torch.manual_seed(20260519 + topk + seqlen_k)
    torch.cuda.empty_cache()

    q = torch.randn(bs, nheads, seqlen_q, d, device=device, dtype=dtype)
    k = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    v = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    dout = torch.randn_like(q)

    if shared_kv_blocks:
        assert topk == 2 and seqlen_k >= 3 * BLK128
        num_q_blocks = (seqlen_q + BLK128 - 1) // BLK128
        selected = torch.tensor([0, 2], device=device, dtype=torch.int32)
        q2k_block_index = (
            selected.view(1, 1, 1, topk)
            .expand(bs, nheads, num_q_blocks, topk)
            .contiguous()
        )
        block_sparse_num = topk
        ref_block_sizes = torch.full(
            ((seqlen_k + BLK128 - 1) // BLK128,),
            BLK128,
            device=device,
            dtype=torch.int32,
        )
    else:
        q2k_block_index, block_sparse_num, ref_block_sizes = _make_topk_args_any(
            bs, seqlen_q, seqlen_k, nheads, topk, BLK128, device
        )
    block_sizes = ref_block_sizes if use_block_sizes else None
    attn_bias = block_sparse_to_attn_bias(
        q2k_block_index,
        block_sparse_num,
        ref_block_sizes,
        seqlen_q,
        seqlen_k,
        blk_m=BLK128,
        blk_n=BLK128,
    )

    softmax_scale = 1.0 / math.sqrt(d)
    out_ref, lse_ref, dq_ref, dk_ref, dv_ref = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale,
    )
    _, _, dq_pt, dk_pt, dv_pt = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale, upcast=False,
    )
    refs = _sanitize_grads(dq_ref, dk_ref, dv_ref)
    tols = _grad_tols(refs, _sanitize_grads(dq_pt, dk_pt, dv_pt))

    dq, dk, dv = bsa_attn_bwd(
        dout,
        q,
        k,
        v,
        out_ref,
        lse_ref,
        q2k_block_index,
        block_sparse_num,
        block_sizes,
        softmax_scale=softmax_scale,
        bucket_size_blocks=bucket_size_blocks,
    )
    _assert_bwd_close(f"blk128 topk={topk}", (dq, dk, dv), refs, tols)


def _test_bwd_layout_equivalence():
    device = "cuda"
    dtype = torch.bfloat16
    bs, nheads, seqlen_q, seqlen_k, d = 1, 4, 128, 256, 128

    torch.manual_seed(2026)
    q = torch.randn(bs, nheads, seqlen_q, d, device=device, dtype=dtype)
    k = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    v = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    dout = torch.randn_like(q)

    q2k_block_index, block_sparse_num, block_sizes = make_random_block_sparse_args(
        bs, seqlen_q, seqlen_k, nheads, blk_m=BLK, blk_n=BLK, device=device,
    )
    attn_bias = block_sparse_to_attn_bias(
        q2k_block_index, block_sparse_num, block_sizes, seqlen_q, seqlen_k,
        blk_m=BLK, blk_n=BLK,
    )
    softmax_scale = 1.0 / math.sqrt(d)
    out_ref, lse_ref, _, _, _ = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale,
    )

    dq_bhsd, dk_bhsd, dv_bhsd = bsa_attn_bwd(
        dout, q, k, v, out_ref, lse_ref,
        q2k_block_index, block_sparse_num, block_sizes,
        softmax_scale=softmax_scale,
    )

    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    dout_bshd = dout.transpose(1, 2).contiguous()
    out_bshd = out_ref.transpose(1, 2).contiguous()

    dq_auto, dk_auto, dv_auto = bsa_attn_bwd(
        dout_bshd,
        q_bshd,
        k_bshd,
        v_bshd,
        out_bshd,
        lse_ref,
        q2k_block_index,
        block_sparse_num,
        block_sizes,
        softmax_scale=softmax_scale,
        layout="bshd",
    )
    # Bucketed CSR scatter and dQ accumulation use atomics, so equivalent
    # layouts can differ by a BF16 rounding step.
    layout_rtol = 1e-2
    layout_atol = 5e-4
    torch.testing.assert_close(
        dq_auto.transpose(1, 2), dq_bhsd, rtol=layout_rtol, atol=layout_atol
    )
    torch.testing.assert_close(
        dk_auto.transpose(1, 2), dk_bhsd, rtol=layout_rtol, atol=layout_atol
    )
    torch.testing.assert_close(
        dv_auto.transpose(1, 2), dv_bhsd, rtol=layout_rtol, atol=layout_atol
    )

    dq_buf = torch.empty_like(q_bshd)
    dk_buf = torch.empty_like(k_bshd)
    dv_buf = torch.empty_like(v_bshd)

    dq_bshd, dk_bshd, dv_bshd = bsa_attn_bwd(
        dout_bshd,
        q_bshd,
        k_bshd,
        v_bshd,
        out_bshd,
        lse_ref,
        q2k_block_index,
        block_sparse_num,
        block_sizes,
        softmax_scale=softmax_scale,
        dq=dq_buf,
        dk=dk_buf,
        dv=dv_buf,
        layout="bshd",
    )

    assert dq_bshd.data_ptr() == dq_buf.data_ptr()
    assert dk_bshd.data_ptr() == dk_buf.data_ptr()
    assert dv_bshd.data_ptr() == dv_buf.data_ptr()
    torch.testing.assert_close(
        dq_bshd.transpose(1, 2), dq_bhsd, rtol=layout_rtol, atol=layout_atol
    )
    torch.testing.assert_close(
        dk_bshd.transpose(1, 2), dk_bhsd, rtol=layout_rtol, atol=layout_atol
    )
    torch.testing.assert_close(
        dv_bshd.transpose(1, 2), dv_bhsd, rtol=layout_rtol, atol=layout_atol
    )


# ============== Pytest ==============

def _cuda_major():
    return torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0


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
def test_flash_bwd_blk64_sparse(seqlen_q, seqlen_k, use_variable_block_nums):
    if _cuda_major() not in [9, 10, 11]:
        pytest.skip("SM90/SM100/SM110 test")
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
def test_flash_bwd_blk64_explicit_bucket_size(seqlen_q, seqlen_k, use_variable_block_nums):
    if _cuda_major() not in [9, 10, 11]:
        pytest.skip("SM90/SM100/SM110 test")
    bs = 2 if seqlen_k <= 512 else 1
    nheads = 4
    _test_bwd_single(
        bs,
        seqlen_q,
        seqlen_k,
        nheads,
        use_variable_block_nums=use_variable_block_nums,
        bucket_size_blocks=512,
    )


@pytest.mark.parametrize("use_variable_block_nums", [False, True])
def test_flash_bwd_blk64_multi_bucket_group(use_variable_block_nums):
    if _cuda_major() not in [9, 10, 11]:
        pytest.skip("SM90/SM100/SM110 test")
    _test_bwd_single(
        1,
        512,
        768,
        4,
        use_variable_block_nums=use_variable_block_nums,
        bucket_size_blocks=2,
    )


def test_flash_bwd_blk64_sparse_path_compare_no_block_sizes():
    if _cuda_major() not in [9, 10, 11]:
        pytest.skip("SM90/SM100/SM110 test")
    device = "cuda"
    dtype = torch.bfloat16
    bs, nheads, seqlen_q, seqlen_k, d = 1, 2, 512, 768, 128

    torch.manual_seed(123)
    torch.cuda.empty_cache()

    q = torch.randn(bs, nheads, seqlen_q, d, device=device, dtype=dtype)
    k = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    v = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    dout = torch.randn_like(q)

    q2k_block_index, block_sparse_num, block_sizes, q2k_block_nums = make_topk_block_sparse_args(
        bs,
        seqlen_q,
        seqlen_k,
        nheads,
        topk=4,
        blk_m=BLK,
        blk_n=BLK,
        device=device,
        use_block_sizes=False,
    )
    assert block_sizes is None
    ref_block_sizes = _full_block_sizes(seqlen_k, device=device)
    attn_bias = block_sparse_to_attn_bias(
        q2k_block_index,
        block_sparse_num,
        ref_block_sizes,
        seqlen_q,
        seqlen_k,
        blk_m=BLK,
        blk_n=BLK,
        q2k_block_nums=q2k_block_nums,
    )
    softmax_scale = 1.0 / math.sqrt(d)
    out_ref, lse_ref, dq_ref, dk_ref, dv_ref = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale,
    )
    _, _, dq_pt, dk_pt, dv_pt = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale, upcast=False,
    )
    refs = _sanitize_grads(dq_ref, dk_ref, dv_ref)
    tols = _grad_tols(refs, _sanitize_grads(dq_pt, dk_pt, dv_pt))

    default_path = bsa_attn_bwd(
        dout,
        q,
        k,
        v,
        out_ref,
        lse_ref,
        q2k_block_index,
        block_sparse_num,
        None,
        q2k_block_nums=q2k_block_nums,
        softmax_scale=softmax_scale,
    )
    direct_bucketed = bsa_attn_bwd(
        dout,
        q,
        k,
        v,
        out_ref,
        lse_ref,
        q2k_block_index,
        block_sparse_num,
        None,
        q2k_block_nums=q2k_block_nums,
        softmax_scale=softmax_scale,
        bucket_size_blocks=2,
    )

    _assert_bwd_close("default sparse block_sizes=None", default_path, refs, tols)
    _assert_bwd_close("explicit bucket size sparse block_sizes=None", direct_bucketed, refs, tols)
    default_direct = tuple(_max_abs(a, b) for a, b in zip(default_path, direct_bucketed))
    print(
        "  compare default vs explicit bucket size: "
        f"dq={default_direct[0]:.4f} dk={default_direct[1]:.4f} dv={default_direct[2]:.4f}"
    )


def test_flash_bwd_blk64_dense_equivalent_compare_no_block_sizes():
    if _cuda_major() not in [9, 10, 11]:
        pytest.skip("SM90/SM100/SM110 test")
    device = "cuda"
    dtype = torch.bfloat16
    bs, nheads, seqlen_q, seqlen_k, d = 1, 2, 512, 768, 128

    torch.manual_seed(321)
    torch.cuda.empty_cache()

    q = torch.randn(bs, nheads, seqlen_q, d, device=device, dtype=dtype)
    k = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    v = torch.randn(bs, nheads, seqlen_k, d, device=device, dtype=dtype)
    dout = torch.randn_like(q)

    q2k_block_index, block_sparse_num, _block_sizes = _make_dense_blk64_args(
        bs, seqlen_q, seqlen_k, nheads, device=device
    )
    ref_block_sizes = _full_block_sizes(seqlen_k, device=device)
    attn_bias = block_sparse_to_attn_bias(
        q2k_block_index,
        block_sparse_num,
        ref_block_sizes,
        seqlen_q,
        seqlen_k,
        blk_m=BLK,
        blk_n=BLK,
    )
    softmax_scale = 1.0 / math.sqrt(d)
    out_ref, lse_ref, dq_ref, dk_ref, dv_ref = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale,
    )
    _, _, dq_pt, dk_pt, dv_pt = _torch_ref_bwd(
        q, k, v, dout, attn_bias, softmax_scale, upcast=False,
    )
    refs = _sanitize_grads(dq_ref, dk_ref, dv_ref)
    tols = _grad_tols(refs, _sanitize_grads(dq_pt, dk_pt, dv_pt))

    default_path = bsa_attn_bwd(
        dout,
        q,
        k,
        v,
        out_ref,
        lse_ref,
        q2k_block_index,
        block_sparse_num,
        None,
        softmax_scale=softmax_scale,
    )
    bucketed_path = bsa_attn_bwd(
        dout,
        q,
        k,
        v,
        out_ref,
        lse_ref,
        q2k_block_index,
        block_sparse_num,
        None,
        softmax_scale=softmax_scale,
        bucket_size_blocks=2,
    )

    _assert_bwd_close("default dense-equivalent block_sizes=None", default_path, refs, tols)
    _assert_bwd_close(
        "explicit bucket size dense-equivalent block_sizes=None",
        bucketed_path,
        refs,
        tols,
    )
    default_direct = tuple(_max_abs(a, b) for a, b in zip(default_path, bucketed_path))
    print(
        "  compare default vs explicit bucket size dense-equivalent: "
        f"dq={default_direct[0]:.4f} dk={default_direct[1]:.4f} dv={default_direct[2]:.4f}"
    )


def test_flash_bwd_blk64_layout_equivalence():
    if _cuda_major() not in [9, 10, 11]:
        pytest.skip("SM90/SM100/SM110 test")
    _test_bwd_layout_equivalence()


@pytest.mark.parametrize(
    "seqlen_k,topk,use_block_sizes",
    [
        (512, 2, True),
        (512, 4, True),
        (448, 4, True),
        (448, 4, False),
    ],
)
def test_flash_bwd_sm100_blk128_topk_correctness(seqlen_k, topk, use_block_sizes):
    if _cuda_major() not in [10, 11]:
        pytest.skip("SM100/SM110 blk128 bwd test")
    _test_bwd_topk_blk128(
        bs=1,
        seqlen_q=256,
        seqlen_k=seqlen_k,
        nheads=2,
        topk=topk,
        use_block_sizes=use_block_sizes,
    )


def test_flash_bwd_sm100_blk128_multi_qbucket_correctness():
    if _cuda_major() not in [10, 11]:
        pytest.skip("SM100/SM110 blk128 bwd test")
    _test_bwd_topk_blk128(
        bs=1,
        seqlen_q=512,
        seqlen_k=512,
        nheads=2,
        topk=2,
        use_block_sizes=True,
        bucket_size_blocks=2,
    )


@pytest.mark.parametrize("num_q_blocks", [1, 4])
def test_flash_bwd_sm100_blk128_dk_zero_init_transition(num_q_blocks):
    """dK must zero-initialize once, then accumulate across later Q tiles."""
    if _cuda_major() not in [10, 11]:
        pytest.skip("SM100/SM110 blk128 bwd test")
    _test_bwd_topk_blk128(
        bs=1,
        seqlen_q=num_q_blocks * BLK128,
        seqlen_k=4 * BLK128,
        nheads=1,
        topk=2,
        use_block_sizes=False,
        shared_kv_blocks=True,
    )


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
def test_flash_bwd_sm90_blk64_dense(seqlen_q, seqlen_k):
    if _cuda_major() != 9:
        pytest.skip("SM90 test")
    bs = 2 if seqlen_k <= 512 else 1
    nheads = 4
    _test_bwd_dense_single(bs, seqlen_q, seqlen_k, nheads)


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
    print("Explicit bucket-size correctness tests (bwd blk64)")
    bucketed_configs = [
        (1, 128, 256, 4, False),
        (1, 256, 512, 4, False),
        (1, 128, 512, 4, True),
    ]
    for bs, sq, sk, h, use_var in bucketed_configs:
        _test_bwd_single(
            bs,
            sq,
            sk,
            h,
            use_variable_block_nums=use_var,
            bucket_size_blocks=512,
        )

    print("-" * 70)
    print("blk128 bwd path (D=128 and D=64 via sm100_blk128)")
    blk128_configs = [
        # (bs, sq, sk, h, topk, d)
        (1, 256, 512, 2, 2, 128),
        (1, 256, 1024, 4, 4, 128),
        (1, 256, 512, 2, 2, 64),
        (1, 256, 1024, 4, 4, 64),
        (1, 512, 512, 4, 4, 64),
        (2, 256, 512, 4, 2, 64),
    ]
    for bs, sq, sk, h, topk, d in blk128_configs:
        _test_bwd_topk_blk128(
            bs=bs, seqlen_q=sq, seqlen_k=sk, nheads=h,
            topk=topk, use_block_sizes=True, d=d,
        )

    print("=" * 70)
    print("All bwd quick tests passed.")


# ============== Benchmark ==============

def _bench_bwd_one(
    dout,
    q,
    k,
    v,
    out,
    lse,
    q2k,
    bsn,
    bsize,
    q2k_block_nums=None,
    niters=10,
):
    """Warmup + time a single bwd call. Returns median ms."""
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
        )

    for _ in range(5):
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


def run_benchmark_suite():
    device = "cuda"
    dtype = torch.bfloat16
    d = 128
    torch.manual_seed(0)
    use_block_sizes = True
    block_size_mode = "full"
    # (bs, nheads, seqlen_q, seqlen_k, hdim)
    configs = [
        (1, 4, 116160, 118528, 128),
        (1, 4, 109312, 111040, 128),
        (1, 4, 216832, 219200, 128),
        (1, 1, 216832, 219200, 128),
        (1, 4, 349440, 351168, 128),
        (1, 4, 695040, 697408, 128),
        (1, 1, 695040, 697408, 128),
    ]
    # topK values to benchmark (0 = dense)
    topk_values = [0, 32, 64, 128, 256]

    print(f"\n{'Config (bwd blk=64)':<54} {'ms':>8} {'TFLOPS':>8} {'MFU':>8}")
    print("-" * 74)

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
                if not use_block_sizes:
                    bsize = None
                q2k_block_nums = None
                block_sizes_label = (
                    f"bsz={block_size_mode}" if use_block_sizes else "bsz=0"
                )
                label = (
                    f"bs={bs} h={nheads} sq={seqlen_q} sk={seqlen_k} "
                    f"d={hdim} dense {block_sizes_label}"
                )
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

            print(f"  {label:<52} ...", end="", flush=True)
            med = _bench_bwd_one(
                dout, q, k, v, out, lse, q2k, bsn, bsize,
                q2k_block_nums=q2k_block_nums,
            )
            f = bwd_flops(bs, nheads, seqlen_q, effective_sk, hdim, hdim)
            tflops = f / (med * 1e-3) / 1e12
            mfu = tflops / 2250.0 * 100.0
            print(
                f"\r  {label:<52} "
                f"{med:>8.3f}ms {tflops:>8.1f} tflops {mfu:>8.1f}%"
            )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        run_benchmark_suite()
    else:
        run_quick_tests()
