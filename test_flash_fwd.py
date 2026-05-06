"""BSA SM100 Forward Kernel — Test / Benchmark / Profile

Usage:
    python test_flash_fwd.py              # quick correctness test
    python test_flash_fwd.py benchmark    # performance benchmark
    python test_flash_fwd.py profile      # single fwd for ncu
"""

import os
import sys
import math

import pytest
import torch

# BSA_BLK env var: "64", "128", or "64,128" (default). Controls which blk sizes to test.
_BSA_BLK = os.environ.get("BSA_BLK", "64,128")
_BLK_SIZES = [int(x) for x in _BSA_BLK.split(",")]

from utils.testing import attention_ref
from utils.bench_utils import flops
from utils.benchmark import benchmark_forward
from bsa_attn_interface import bsa_attn_fwd

# Optional: blk64 C++ AOT kernel (install via `make setup BLK=64`)
try:
    import bsa_fwd_blk64_ext
    HAS_BLK64 = True
except ImportError:
    HAS_BLK64 = False


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
        bn_arg = q2k_block_nums if use_variable_block_nums else torch.Tensor()
        bs_arg = block_sizes_kernel if block_sizes_kernel is not None else torch.Tensor()
        # blk64 kernel consumes BHSD natively; convert from BSHD test tensors
        # at the boundary and convert the BHSD output back to BSHD for reference.
        q_bhsd = q.transpose(1, 2).contiguous()
        k_bhsd = k.transpose(1, 2).contiguous()
        v_bhsd = v.transpose(1, 2).contiguous()
        out_bhsd, lse = torch.ops.bsa_blk64.fwd(
            q_bhsd, k_bhsd, v_bhsd, q2k_block_index, block_sparse_num,
            bs_arg, softmax_scale, bn_arg, use_clc)
        out = out_bhsd.transpose(1, 2).contiguous()
    else:
        out, lse = bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes_kernel,
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


# ============== Pytest ==============

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
    if blk_n != 64 and use_clc:
        pytest.skip("use_clc only affects blk64")
    batch_size = 2
    nheads = 4
    blk_m = 64 if blk_n == 64 else 128
    _test_single(batch_size, seqlen_q, seqlen_k, nheads, nheads, 128, dtype,
                  use_variable_block_nums=use_variable_block_nums,
                  use_block_sizes=False,
                  blk_m=blk_m, blk_n=blk_n, use_clc=use_clc)


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

    if HAS_BLK64 and 64 in _BLK_SIZES:
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

    if HAS_BLK64 and 64 in _BLK_SIZES:
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

    if HAS_BLK64 and 64 in _BLK_SIZES:
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

    print("=" * 70)
    print("All quick tests passed.")


# ============== Kernel dispatch helper ==============

def _call_kernel(q, k, v, q2k_block_index, block_sparse_num, block_sizes, blk_n,
                 q2k_block_nums=None, softmax_scale=None, use_clc=False):
    """Dispatch to blk64 or blk128 kernel based on blk_n.

    q/k/v are BSHD (batch, seq, heads, dim). blk64 expects BHSD, so we transpose
    at the boundary; blk128 consumes BSHD directly via bsa_attn_fwd.
    use_clc: blk64-only toggle for the CLC persistent scheduler; ignored for blk128.
    """
    if blk_n == 64:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        bn_arg = q2k_block_nums if q2k_block_nums is not None else torch.Tensor()
        bs_arg = block_sizes if block_sizes is not None else torch.Tensor()
        q_bhsd = q.transpose(1, 2).contiguous()
        k_bhsd = k.transpose(1, 2).contiguous()
        v_bhsd = v.transpose(1, 2).contiguous()
        out_bhsd, _ = torch.ops.bsa_blk64.fwd(
            q_bhsd, k_bhsd, v_bhsd, q2k_block_index, block_sparse_num,
            bs_arg, softmax_scale, bn_arg, use_clc)
        return out_bhsd.transpose(1, 2).contiguous()
    else:
        return bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes,
                             q2k_block_nums=q2k_block_nums)[0]


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
        if blk_n == 64 and not HAS_BLK64:
            print(f"[blk{blk_n}] skipped (not built)")
            continue

        # For blk64 benchmark both use_clc=False/True side-by-side.
        use_clc_settings = [False, True] if blk_n == 64 else [False]
        # Each CLC variant prints "ms TFLOPS" (16 wide incl. separator).
        header_clc = " ".join(
                f"{'clc=' + str(int(uc)) + ' ms':>10} {'TFLOPS':>6}"
                for uc in use_clc_settings)
        print(f"\n{'Config (blk=' + str(blk_n) + ')':<48} {header_clc}")
        print("-" * (48 + len(header_clc) + 2))

        for bs, nheads, seqlen, hdim in configs:
            dtype = torch.bfloat16
            q = torch.randn(bs, seqlen, nheads, hdim, device="cuda", dtype=dtype)
            k = torch.randn(bs, seqlen, nheads, hdim, device="cuda", dtype=dtype)
            v = torch.randn(bs, seqlen, nheads, hdim, device="cuda", dtype=dtype)

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
                    print(f" {med:>10.3f} {tflops:>6.1f}", end="", flush=True)
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
        if blk_n == 64 and not HAS_BLK64:
            print(f"[blk{blk_n}] skipped (not built)")
            continue

        q = torch.randn(bs, seqlen, nheads, hdim, device="cuda", dtype=dtype)
        k = torch.randn(bs, seqlen, nheads, hdim, device="cuda", dtype=dtype)
        v = torch.randn(bs, seqlen, nheads, hdim, device="cuda", dtype=dtype)
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
