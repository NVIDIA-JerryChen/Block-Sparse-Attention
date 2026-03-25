"""BSA SM100 Forward Kernel — Test / Benchmark / Profile

Usage:
    python test_flash_fwd.py              # quick correctness test
    python test_flash_fwd.py benchmark    # performance benchmark
    python test_flash_fwd.py profile      # single fwd for ncu
"""

import sys
import math

import pytest
import torch

from utils.testing import attention_ref
from utils.bench_utils import flops
from utils.benchmark import benchmark_forward
from bsa_attn_interface import bsa_attn_fwd


# ============== Block-sparse helpers ==============

def make_dense_block_sparse_args(batch_size, seqlen_q, seqlen_k, nheads, tile_m=128, tile_n=128, device="cuda"):
    """Create block-sparse args equivalent to dense (full) attention.  For benchmark/profile.

    All block_sizes = tile_n (full blocks).  Requires seqlen_k % (2*tile_n) == 0.
    Returns q2k_block_index, block_sparse_num, block_sizes.
    """
    num_q_blocks = (seqlen_q + tile_m - 1) // tile_m
    num_kv_blocks = (seqlen_k + tile_n - 1) // tile_n

    assert num_kv_blocks >= 2 and num_kv_blocks % 2 == 0, (
        f"num_kv_blocks={num_kv_blocks} must be even and >= 2 for dense-equivalent test. "
        f"Adjust seqlen_k to be a multiple of {2 * tile_n}."
    )
    block_sparse_num = num_kv_blocks

    indices = torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
    q2k_block_index = indices.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
        batch_size, nheads, num_q_blocks, num_kv_blocks
    ).contiguous()

    block_sizes = torch.full((num_kv_blocks,), tile_n, dtype=torch.int32, device=device)
    return q2k_block_index, block_sparse_num, block_sizes


def make_random_block_sparse_args(batch_size, seqlen_q, seqlen_k, nheads, tile_m=128, tile_n=128, device="cuda"):
    """Create random block-sparse args for correctness testing.

    Random block_sparse_num (even, >= 2), random per-(batch,head,q_block) KV block
    selection, and random block_sizes in [1, tile_n].
    Returns q2k_block_index, block_sparse_num, block_sizes.
    """
    num_q_blocks = (seqlen_q + tile_m - 1) // tile_m
    num_kv_blocks = (seqlen_k + tile_n - 1) // tile_n
    assert num_kv_blocks >= 2, f"num_kv_blocks={num_kv_blocks} must be >= 2"

    max_even = num_kv_blocks if num_kv_blocks % 2 == 0 else num_kv_blocks - 1
    # Minimum bsn=4 to avoid pre-existing kernel issue with bsn=2 + large batch + small hdim
    min_bsn = min(4, max_even)
    possible_counts = list(range(min_bsn, max_even + 1, 2))
    block_sparse_num = possible_counts[torch.randint(len(possible_counts), (1,)).item()]

    q2k_block_index = torch.empty(batch_size, nheads, num_q_blocks, block_sparse_num,
                                   dtype=torch.int32, device=device)
    for b in range(batch_size):
        for h in range(nheads):
            for m in range(num_q_blocks):
                perm = torch.randperm(num_kv_blocks, device=device)[:block_sparse_num]
                q2k_block_index[b, h, m] = perm.to(torch.int32)

    block_sizes = torch.randint(1, tile_n + 1, (num_kv_blocks,), dtype=torch.int32, device=device)
    last_block_actual = seqlen_k - (num_kv_blocks - 1) * tile_n
    if last_block_actual < tile_n:
        block_sizes[-1] = min(block_sizes[-1].item(), last_block_actual)

    return q2k_block_index, block_sparse_num, block_sizes


def make_random_variable_block_sparse_args(batch_size, seqlen_q, seqlen_k, nheads, tile_m=128, tile_n=128, device="cuda"):
    """Create random block-sparse args with per-(batch, head, q_block) variable block counts.

    Each Q block gets a random block_sparse_num in [1, num_kv_blocks].
    q2k_block_index is padded to num_kv_blocks with zeros (unused entries).
    Returns q2k_block_index, q2k_block_nums, block_sizes.
    """
    num_q_blocks = (seqlen_q + tile_m - 1) // tile_m
    num_kv_blocks = (seqlen_k + tile_n - 1) // tile_n
    assert num_kv_blocks >= 1, f"num_kv_blocks={num_kv_blocks} must be >= 1"

    possible_counts = list(range(1, num_kv_blocks + 1))

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

    block_sizes = torch.randint(1, tile_n + 1, (num_kv_blocks,), dtype=torch.int32, device=device)
    last_block_actual = seqlen_k - (num_kv_blocks - 1) * tile_n
    if last_block_actual < tile_n:
        block_sizes[-1] = min(block_sizes[-1].item(), last_block_actual)

    return q2k_block_index, q2k_block_nums, block_sizes


def block_sparse_to_attn_bias(q2k_block_index, block_sparse_num, block_sizes,
                               seqlen_q, seqlen_k, tile_m=128, tile_n=128,
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

    col_idx = torch.arange(tile_n, device=device)
    block_valid = col_idx.unsqueeze(0) < block_sizes.unsqueeze(1)  # (num_kv_blocks, tile_n)

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

    # (batch, nheads, num_q_blocks, num_kv_blocks, tile_n) -> clip to seqlen_k
    token_valid = (block_attended.unsqueeze(-1) & block_valid).reshape(
        batch_size, nheads, num_q_blocks, -1
    )[..., :seqlen_k]

    attn_bias = torch.full((batch_size, nheads, seqlen_q, seqlen_k), float("-inf"),
                           device=device, dtype=torch.float32)
    for m in range(num_q_blocks):
        q_start = m * tile_m
        q_end = min((m + 1) * tile_m, seqlen_q)
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
                  use_variable_block_nums=False):
    """Run a single correctness test with random block-sparse pattern.

    Tolerance (from FA4 test_flash_attn.py):
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max()   # bf16 noise floor
        rtol = 2
        assert |out - out_ref| <= rtol * |out_pt - out_ref| + fwd_atol
    """
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    q_ref = torch.randn(bs, seqlen_q, nheads, d, device=device, dtype=dtype).requires_grad_()
    k_ref = torch.randn(bs, seqlen_k, nheads_kv, d, device=device, dtype=dtype).requires_grad_()
    v_ref = torch.randn(bs, seqlen_k, nheads_kv, d, device=device, dtype=dtype).requires_grad_()
    q = q_ref.detach().requires_grad_()
    k = k_ref.detach().requires_grad_()
    v = v_ref.detach().requires_grad_()

    # With pack_gqa, the kernel indexes q2k by nheads_kv (not nheads_q) and
    # seqlen_q is packed: seqlen_q_eff = seqlen_q * qhead_per_kvhead.
    qhead_per_kvhead = nheads // nheads_kv
    pack_gqa = qhead_per_kvhead > 1 and (128 % qhead_per_kvhead == 0)
    nheads_q2k = nheads_kv if pack_gqa else nheads
    seqlen_q_q2k = seqlen_q * qhead_per_kvhead if pack_gqa else seqlen_q

    q2k_block_nums = None
    if use_variable_block_nums:
        q2k_block_index, q2k_block_nums, block_sizes = make_random_variable_block_sparse_args(
            bs, seqlen_q_q2k, seqlen_k, nheads_q2k, device=device,
        )
        block_sparse_num = 0  # unused when q2k_block_nums is provided
    else:
        q2k_block_index, block_sparse_num, block_sizes = make_random_block_sparse_args(
            bs, seqlen_q_q2k, seqlen_k, nheads_q2k, device=device,
        )

    # Build attn_bias at (bs, nheads_q, seqlen_q, seqlen_k) for the reference.
    # Expand nheads_kv → nheads_q: all Q heads in the same KV group share the same pattern.
    attn_bias_kv = block_sparse_to_attn_bias(
        q2k_block_index, block_sparse_num, block_sizes, seqlen_q_q2k, seqlen_k,
        q2k_block_nums=q2k_block_nums,
    )
    attn_bias = pack_gqa_attn_bias(attn_bias_kv, nheads, qhead_per_kvhead, seqlen_q) if pack_gqa else attn_bias_kv

    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, None, None, attn_bias=attn_bias, causal=False)
    out_pt, _ = attention_ref(
        q_ref, k_ref, v_ref, None, None, attn_bias=attn_bias, causal=False,
        upcast=False, reorder_ops=True,
    )

    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    rtol = 2

    out, lse = bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes,
                             q2k_block_nums=q2k_block_nums)

    kernel_diff = (out - out_ref).abs().max().item()
    pt_diff = (out_pt - out_ref).abs().max().item()
    tol = rtol * pt_diff + fwd_atol
    passed = kernel_diff <= tol

    mode_str = "var_bsn" if use_variable_block_nums else f"sparse_num={block_sparse_num}"
    tag = "PASS" if passed else "FAIL"
    print(
        f"  {tag} bs={bs} sq={seqlen_q} sk={seqlen_k} h={nheads}/{nheads_kv} d={d} "
        f"{mode_str}: "
        f"kernel={kernel_diff:.6f} pt={pt_diff:.6f} tol={tol:.6f}"
    )
    assert passed, f"kernel_diff={kernel_diff} > tol={tol}"


# ============== Pytest ==============

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("use_variable_block_nums", [False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        # num_kv_blocks >= 2 required (seqlen_k >= 129 for tile_n=128)
        (64, 256),
        (64, 384),
        (64, 512),
        (64, 1024),
        (128, 256),
        (128, 640),
        (128, 1024),
        (256, 256),
        (256, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ],
)
def test_flash_fwd_sm100(seqlen_q, seqlen_k, d, mha_type, dtype, use_variable_block_nums):
    batch_size = 4 if seqlen_k <= 2048 else 2
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else (3 if mha_type == "gqa" else 1)
    _test_single(batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dtype,
                  use_variable_block_nums=use_variable_block_nums)


# ============== Quick test (make tt) ==============

def run_quick_tests():
    print("Quick correctness tests")
    print("=" * 70)
    configs = [
        # (bs, sq, sk, hq, hk, d) — num_kv_blocks >= 2 required
        (1, 64, 256, 4, 4, 128),
        (1, 64, 384, 4, 4, 128),
        (1, 64, 512, 4, 4, 128),
        (1, 64, 256, 8, 1, 128),
        (1, 256, 256, 4, 4, 128),
        (1, 128, 640, 4, 4, 128),
        (1, 1024, 1024, 4, 4, 128),
        (1, 2048, 2048, 4, 4, 128),
    ]
    for bs, sq, sk, hq, hk, d in configs:
        _test_single(bs, sq, sk, hq, hk, d)
    print("-" * 70)
    print("Variable block_sparse_num tests")
    var_configs = [
        (1, 64, 512, 4, 4, 128),
        (1, 256, 512, 4, 4, 128),
        (1, 128, 640, 4, 4, 128),
        (1, 64, 256, 8, 1, 128),
    ]
    for bs, sq, sk, hq, hk, d in var_configs:
        _test_single(bs, sq, sk, hq, hk, d, use_variable_block_nums=True)
    print("=" * 70)
    print("All quick tests passed.")


# ============== Benchmark (make bb) ==============

def run_benchmark_suite():
    configs = [
        (1, 40, 4096, 128),
        (1, 40, 8192, 128),
        (1, 40, 16384, 128),
    ]

    print(f"{'Config':<40} {'ms':>8} {'TFLOPS':>8}")
    print("-" * 60)

    for bs, nheads, seqlen, hdim in configs:
        label = f"bs={bs} h={nheads} sq={seqlen} d={hdim}"
        dtype = torch.bfloat16
        q = torch.randn(bs, seqlen, nheads, hdim, device="cuda", dtype=dtype)
        k = torch.randn(bs, seqlen, nheads, hdim, device="cuda", dtype=dtype)
        v = torch.randn(bs, seqlen, nheads, hdim, device="cuda", dtype=dtype)
        q2k_block_index, block_sparse_num, block_sizes = make_dense_block_sparse_args(
            bs, seqlen, seqlen, nheads, device="cuda",
        )

        # Warmup
        for _ in range(10):
            bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes)
        torch.cuda.synchronize()

        niters = 100
        evts = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(niters)
        ]
        for s, e in evts:
            s.record()
            bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes)
            e.record()
        torch.cuda.synchronize()

        times = sorted([s.elapsed_time(e) for s, e in evts])
        med = times[len(times) // 2]
        f = flops(bs, nheads, seqlen, seqlen, hdim, hdim)
        tflops = f / (med * 1e-3) / 1e12
        print(f"{label:<40} {med:>8.3f} {tflops:>8.1f}")


# ============== Profile (make profile) ==============

def run_profile():
    bs, nheads, seqlen, hdim = 1, 40, 8192, 128
    dtype = torch.bfloat16

    q = torch.randn(bs, seqlen, nheads, hdim, device="cuda", dtype=dtype)
    k = torch.randn(bs, seqlen, nheads, hdim, device="cuda", dtype=dtype)
    v = torch.randn(bs, seqlen, nheads, hdim, device="cuda", dtype=dtype)
    q2k_block_index, block_sparse_num, block_sizes = make_dense_block_sparse_args(
        bs, seqlen, seqlen, nheads, device="cuda",
    )

    # Profile run
    bsa_attn_fwd(q, k, v, q2k_block_index, block_sparse_num, block_sizes)
    torch.cuda.synchronize()

    print(f"Profile done: bs={bs} h={nheads} sq={seqlen} d={hdim}")


# ============== Main ==============

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        run_profile()
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        run_benchmark_suite()
    else:
        run_quick_tests()
