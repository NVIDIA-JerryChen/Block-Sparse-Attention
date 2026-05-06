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

from bsa_attn_interface import bsa_attn_bwd, bsa_attn_bwd_qbucket, convert_q2k_to_k2q
from test_flash_fwd import (
    make_dense_block_sparse_args,
    make_random_block_sparse_args,
    make_random_variable_block_sparse_args,
    make_topk_block_sparse_args,
    block_sparse_to_attn_bias,
)
from utils.bench_utils import flops, bwd_flops


# The bwd kernel only exists for blk64 (sparse_block_size=64, head_dim=128, MHA, bf16).
BLK = 64


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
    else:
        run_quick_tests()
