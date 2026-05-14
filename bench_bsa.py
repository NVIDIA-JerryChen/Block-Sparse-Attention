"""Benchmark script for bsa attn.

Usage:
    python bench_bsa.py
"""


import os
import sys
import math
import random
import types

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import triton
import triton.language as tl
from tabulate import tabulate

from bsa_attn_interface import (
    bsa_attn_fwd_blk64,
    bsa_attn_bwd as bsa_attn_bwd_blk64,
)

# import block_sparse_attention

def bsa_attn_fwd(
    q,
    k,
    v,
    q2k_block_index,
    max_topk,
    block_sizes=None,
    softmax_scale=None,
    q2k_block_nums=None,
    layout="bhsd",
    use_clc: bool = True,
):
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

    out, lse = torch.ops.bsa_blk64.fwd(
        q, k, v, q2k_block_index, max_topk, block_sizes, softmax_scale, q2k_block_nums, use_clc)

    # Kernel returns out as BHSD (B, H, S_q_rounded, D). Trim seqlen (dim 2).
    if out.size(2) != seqlen_q:
        out = out[:, :, :seqlen_q]
    if lse.size(2) != seqlen_q:
        lse = lse[:, :, :seqlen_q]

    if layout == "bshd":
        out = out.permute(0, 2, 1, 3).contiguous()
    return out, lse


def bsa_attn_bwd(
    dout,
    q,
    k,
    v,
    out,
    lse,
    q2k_block_index,
    max_topk,
    block_sizes=None,
    q2k_block_nums=None,
    softmax_scale=None,
    dq=None,
    dk=None,
    dv=None,
):
    return bsa_attn_bwd_blk64(
        dout, q, k, v, out, lse, q2k_block_index, max_topk,
        block_sizes, q2k_block_nums, softmax_scale, dq, dk, dv,
    )


def do_bench(func, warmup=3, runs=20):
    """Benchmark a function and return the average time in milliseconds.

    Args:
        func: Function to benchmark
        warmup: Number of warmup runs
        runs: Number of timed runs

    Returns:
        Average time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    a = torch.randn(10*1024, 4096, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)

    # Create event lists
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]

    # Record events
    for i in range(runs):
        for _ in range(10):
            a @ b
            torch.zeros(1024, 1024, 1024, device='cuda', dtype=torch.int8)
        torch.zeros(1024, 1024, 1024, device='cuda', dtype=torch.int8)
        starts[i].record()
        func()
        ends[i].record()
        torch.zeros(1024, 1024, 1024, device='cuda', dtype=torch.int8)

    torch.cuda.synchronize()

    # Compute average time
    total_time = sum(starts[i].elapsed_time(ends[i]) for i in range(runs))

    return total_time / runs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ceil_div_int(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64,
                  force_k_prefix_len=None, force_k_suffix_len=None,
                  q_mask=None, kv_mask=None):
    q_num_blocks = _ceil_div_int(q.shape[2], BLKQ)
    k_num_blocks = _ceil_div_int(k.shape[2], BLKQ)
    pooled_score = torch.randn(q.shape[0], q.shape[1], q_num_blocks, k_num_blocks, device=q.device)

    K = pooled_score.shape[-1]
    topk = min(K, max(1, int(topk_ratio * K)))
    if force_k_prefix_len is not None and force_k_prefix_len > 0:
        force_blocks = min(K, (int(force_k_prefix_len) + BLKK - 1) // BLKK)
        if force_blocks > 0:
            topk = min(K, max(topk, force_blocks))
            pooled_score = pooled_score.clone()
            pooled_score[..., :force_blocks] = torch.finfo(
                pooled_score.dtype).max
    if force_k_suffix_len is not None and force_k_suffix_len > 0:
        force_blocks = min(K, (int(force_k_suffix_len) + BLKK - 1) // BLKK)
        if force_blocks > 0:
            topk = min(K, max(topk, force_blocks))
            pooled_score = pooled_score.clone()
            pooled_score[..., -
                         force_blocks:] = torch.finfo(pooled_score.dtype).max
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices
    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    density = topk / K if K > 0 else 1.0
    return sparse_map, lut, topk, density


def get_3D_padded_mask(n_frames, height, width, block_fhw=(1, 8, 8), device=None):
    """Pad and reorder a (n_frames, height, width) video token grid using 3D block-major layout.

    Each dimension is padded to be divisible by its block size, then tokens are
    reordered in block-major order: (block_f, block_h, block_w) blocks are
    traversed contiguously.  Within each block, valid tokens are compacted to the
    front and padding tokens to the back.

    Args:
        n_frames: Number of frames (F dimension).
        height: Spatial height (H dimension).
        width: Spatial width (W dimension).
        block_fhw: Tuple (block_f, block_h, block_w) for 3D block padding/reorder.
        device: Target device.

    Returns:
        S_vis_padded: Total padded sequence length (F4 * H4 * W4).
        valid_mask: [S_vis_padded] bool, True for valid (non-padding) tokens.
    """
    block_f, block_h, block_w = block_fhw
    F, H, W = n_frames, height, width

    if F <= 0 or H <= 0 or W <= 0:
        empty_mask = torch.empty((0,), device=device, dtype=torch.bool)
        return 0, empty_mask

    F4 = _ceil_div_int(F, block_f) * block_f
    H4 = _ceil_div_int(H, block_h) * block_h
    W4 = _ceil_div_int(W, block_w) * block_w
    L_pad = F4 * H4 * W4

    # Build raster-order indices and reshape into block-major traversal
    idx = torch.arange(L_pad, device=device,
                       dtype=torch.long).reshape(F4, H4, W4)
    # (nbf, block_f, nbh, block_h, nbw, block_w) → (nbf, nbh, nbw, block_f, block_h, block_w)
    idx = (
        idx.view(F4 // block_f, block_f, H4 // block_h,
                 block_h, W4 // block_w, block_w)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(-1)
    )

    # Recover (f, h, w) coordinates from flat index in the padded grid
    hw4 = H4 * W4
    t_idx = idx // hw4
    rem = idx - t_idx * hw4
    h_idx = rem // W4
    w_idx = rem - h_idx * W4

    valid_mask = (t_idx < F) & (h_idx < H) & (w_idx < W)

    # Compact valid tokens to the front within each 3D block
    tot_block_size = block_f * block_h * block_w
    if tot_block_size > 0 and idx.numel() > 0:
        nb = L_pad // tot_block_size
        idx_blk = idx.view(nb, tot_block_size)
        m_blk = valid_mask.view(nb, tot_block_size)

        pos = torch.arange(tot_block_size, device=device,
                           dtype=torch.long).view(1, tot_block_size)
        key = (~m_blk).to(torch.long) * tot_block_size + pos
        perm = key.argsort(dim=1)

        valid_mask = m_blk.gather(1, perm).reshape(-1)

    return L_pad, valid_mask


def attn_tflops(seqlen_q, seqlen_k, nhead, head_dim, density, time_ms, factor):
    tflops = seqlen_q * seqlen_k * nhead * head_dim * \
        density * factor * (1e3 / time_ms) / 1e12
    mfu = tflops / 2250
    return tflops, mfu * 100


def prepare_inputs(
    batch_size, head_size, S_aud, S_txt, n_frames, height, width,
    head_dim, BLKQ, BLKK, topk_ratio, block_fhw=(1, 8, 8), device='cuda',
):
    """Prepare Q, K, V and block-sparse index tensors."""

    # Padding masks
    S_vis_padded, vis_mask = get_3D_padded_mask(
        n_frames, height, width, block_fhw=block_fhw, device=device)
    S_aud_padded = _ceil_div_int(S_aud, BLKK) * BLKK

    aud_mask = torch.zeros((S_aud_padded,), device=device, dtype=torch.bool)
    aud_mask[:S_aud] = True
    txt_mask = torch.ones((S_txt,), device=device, dtype=torch.bool)

    q_mask = vis_mask.clone()
    kv_mask = torch.cat([vis_mask, aud_mask, txt_mask], dim=0)

    S_q = S_vis_padded
    S_k = S_vis_padded + S_aud_padded + S_txt

    # Input tensors (BHSD layout)
    q = torch.randn(batch_size, head_size, S_q, head_dim,
                    device=device, dtype=torch.bfloat16)
    k = torch.randn(batch_size, head_size, S_k, head_dim,
                    device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, head_size, S_k, head_dim,
                    device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        q[:, :, ~q_mask] = 0
        k[:, :, ~kv_mask] = 0
        v[:, :, ~kv_mask] = 0

    # Build block-sparse map
    sparse_map, indices, real_topk, density = get_block_map(
        q, k,
        topk_ratio=topk_ratio,
        BLKQ=BLKQ,
        BLKK=BLKK,
        force_k_suffix_len=S_aud_padded + S_txt,
        q_mask=q_mask,
        kv_mask=kv_mask,
    )

    # Convert sparse_map → bsa_attn_fwd inputs
    # q2k_block_index: (batch, heads, n_q_blocks, n_kv_blocks) sorted indices
    q2k_block_index = sparse_map.argsort(
        dim=3, descending=True, stable=True).to(torch.int32)
    max_topk = real_topk
    q2k_block_nums = torch.full(
        (batch_size, head_size, q2k_block_index.size(2)),
        max_topk,
        dtype=torch.int32,
        device=device,
    )
    q2k_block_nums = None

    # block_sizes: actual valid tokens per KV block
    num_kv_blocks = _ceil_div_int(kv_mask.size(-1), BLKK)
    kv_mask_padded = F.pad(
        kv_mask, (0, num_kv_blocks * BLKK - kv_mask.size(-1)), value=False)
    block_sizes = kv_mask_padded.view(
        num_kv_blocks, BLKK).sum(dim=-1).to(torch.int32)

    return (q, k, v, q2k_block_index, max_topk, q2k_block_nums, block_sizes,
            S_q, S_k, density, kv_mask)


def bench_bsa(
    batch_size=1,
    head_size=5,
    S_aud=324,
    S_txt=1024,
    n_frames=91,
    height=30,
    width=52,
    head_dim=128,
    BLKQ=64,
    BLKK=64,
    topk_ratio=0.1,
    block_fhw=(1, 8, 8),
    device='cuda',
    warmup=5,
    runs=20,
    check_correctness=False,
):
    """Benchmark bsa_attn_fwd with given configuration.

    When check_correctness=True, also runs fa4_sla's sparse_flash_attn_cute_func
    as a baseline and reports the max absolute error.
    """
    (q, k, v, q2k_block_index, max_topk, q2k_block_nums, block_sizes,
     S_q, S_k, density, kv_mask) = prepare_inputs(
        batch_size, head_size, S_aud, S_txt, n_frames, height, width,
        head_dim, BLKQ, BLKK, topk_ratio, block_fhw=block_fhw, device=device,
    )

    # Warmup + correctness
    out, lse = bsa_attn_fwd(
        q, k, v,
        q2k_block_index=q2k_block_index,
        max_topk=max_topk,
        block_sizes=block_sizes,
        q2k_block_nums=q2k_block_nums,
        # blk_n=BLKK,
    )
    torch.cuda.synchronize()

    # ── Optional correctness check against fa4_sla baseline ──
    max_abs_err = None
    mean_abs_err = None
    if check_correctness:
        from fa4_sla import sparse_flash_attn_cute_func

        # fa4 expects BSHD layout: (B, S, H, D)
        q_bshd = q.permute(0, 2, 1, 3).contiguous()  # BHSD → BSHD
        k_bshd = k.permute(0, 2, 1, 3).contiguous()
        v_bshd = v.permute(0, 2, 1, 3).contiguous()

        # indices: topk selected KV-block indices per (batch, head, q_block)
        indices = q2k_block_index[:, :, :, :max_topk].contiguous()

        out_ref = sparse_flash_attn_cute_func(
            q_bshd, k_bshd, v_bshd,
            kv_mask,
            indices,
            blk_q=BLKQ,
            blk_k=BLKK,
        )
        # out_ref is BSHD → convert to BHSD for comparison
        out_ref_bhsd = out_ref.permute(0, 2, 1, 3)

        # bsa_attn_fwd (blk64) output is already BHSD
        diff = (out.float() - out_ref_bhsd.float()).abs()
        max_abs_err = diff.max().item()
        mean_abs_err = diff.mean().item()
        print(
            f"    Correctness check: max_abs_err = {max_abs_err:.6e}, mean_abs_err = {mean_abs_err:.6e}")

    def fwd_fn():
        return bsa_attn_fwd(
            q, k, v,
            q2k_block_index=q2k_block_index,
            max_topk=max_topk,
            block_sizes=block_sizes,
            q2k_block_nums=q2k_block_nums,
            # blk_n=BLKK,
        )

    fwd_time = do_bench(fwd_fn, warmup=warmup, runs=runs)

    tflops, mfu = attn_tflops(
        S_q, S_k, head_size, head_dim, density, fwd_time, 4)

    result = {
        'S_q': S_q,
        'S_k': S_k,
        'H': head_size,
        'D': head_dim,
        'density': density,
        'max_topk': max_topk,
        'fwd_time_ms': fwd_time,
        'tflops': tflops,
        'mfu': mfu,
    }
    if check_correctness:
        result['max_abs_err'] = max_abs_err
        result['mean_abs_err'] = mean_abs_err
    return result


def bench_bsa_bwd(
    batch_size=1,
    head_size=5,
    S_aud=324,
    S_txt=1024,
    n_frames=91,
    height=30,
    width=52,
    head_dim=128,
    BLKQ=64,
    BLKK=64,
    topk_ratio=0.1,
    block_fhw=(1, 8, 8),
    device='cuda',
    warmup=5,
    runs=20,
    check_correctness=False,
    hfu=False,
):
    """Benchmark bsa_attn_bwd with given configuration."""
    (q, k, v, q2k_block_index, max_topk, q2k_block_nums, block_sizes,
     S_q, S_k, density, kv_mask) = prepare_inputs(
        batch_size, head_size, S_aud, S_txt, n_frames, height, width,
        head_dim, BLKQ, BLKK, topk_ratio, block_fhw=block_fhw, device=device,
    )

    # Run fwd to get out and lse
    out, lse = bsa_attn_fwd(
        q, k, v,
        q2k_block_index=q2k_block_index,
        max_topk=max_topk,
        block_sizes=block_sizes,
        q2k_block_nums=q2k_block_nums,
        # blk_n=BLKK,
    )
    torch.cuda.synchronize()

    dout = torch.randn_like(out)

    # Warmup bwd
    dq, dk, dv = bsa_attn_bwd(
        dout, q, k, v, out, lse,
        q2k_block_index=q2k_block_index,
        max_topk=max_topk,
        block_sizes=block_sizes,
        # q2k_block_nums=q2k_block_nums,
    )
    torch.cuda.synchronize()

    # ── Optional correctness check against fa4_sla baseline ──
    max_abs_err_dq = None
    mean_abs_err_dq = None
    max_abs_err_dk = None
    mean_abs_err_dk = None
    max_abs_err_dv = None
    mean_abs_err_dv = None
    if check_correctness:
        from fa4_sla import sparse_flash_attn_cute_func

        # fa4 expects BSHD layout: (B, S, H, D)
        q_bshd = q.permute(0, 2, 1, 3).contiguous().requires_grad_(True)
        k_bshd = k.permute(0, 2, 1, 3).contiguous().requires_grad_(True)
        v_bshd = v.permute(0, 2, 1, 3).contiguous().requires_grad_(True)

        indices = q2k_block_index[:, :, :, :max_topk].contiguous()

        out_ref = sparse_flash_attn_cute_func(
            q_bshd, k_bshd, v_bshd,
            kv_mask,
            indices,
            blk_q=BLKQ,
            blk_k=BLKK,
        )
        dout_bshd = dout.permute(0, 2, 1, 3).contiguous()
        out_ref.backward(dout_bshd)

        dq_ref = q_bshd.grad.permute(0, 2, 1, 3)  # BSHD → BHSD
        dk_ref = k_bshd.grad.permute(0, 2, 1, 3)
        dv_ref = v_bshd.grad.permute(0, 2, 1, 3)

        diff_dq = (dq.float() - dq_ref.float()).abs()
        diff_dk = (dk.float() - dk_ref.float()).abs()
        diff_dv = (dv.float() - dv_ref.float()).abs()

        max_abs_err_dq = diff_dq.max().item()
        mean_abs_err_dq = diff_dq.mean().item()
        max_abs_err_dk = diff_dk.max().item()
        mean_abs_err_dk = diff_dk.mean().item()
        max_abs_err_dv = diff_dv.max().item()
        mean_abs_err_dv = diff_dv.mean().item()
        print(
            f"    BWD correctness: "
            f"dQ max={max_abs_err_dq:.6e} mean={mean_abs_err_dq:.6e}, "
            f"dK max={max_abs_err_dk:.6e} mean={mean_abs_err_dk:.6e}, "
            f"dV max={max_abs_err_dv:.6e} mean={mean_abs_err_dv:.6e}")

    def bwd_fn():
        return bsa_attn_bwd(
            dout, q, k, v, out, lse,
            q2k_block_index=q2k_block_index,
            max_topk=max_topk,
            block_sizes=block_sizes,
            # q2k_block_nums=q2k_block_nums,
        )

    bwd_time = do_bench(bwd_fn, warmup=warmup, runs=runs)

    # bwd flops ≈ 2x fwd (dS@V^T + dP@K + dP^T@Q + P^T@dO → factor=8)
    tflops, mfu = attn_tflops(
        S_q, S_k, head_size, head_dim, density, bwd_time, 10 if hfu else 8)

    result = {
        'S_q': S_q,
        'S_k': S_k,
        'H': head_size,
        'D': head_dim,
        'density': density,
        'max_topk': max_topk,
        'bwd_time_ms': bwd_time,
        'tflops': tflops,
        'mfu': mfu,
    }
    if check_correctness:
        result['max_abs_err_dq'] = max_abs_err_dq
        result['mean_abs_err_dq'] = mean_abs_err_dq
        result['max_abs_err_dk'] = max_abs_err_dk
        result['mean_abs_err_dk'] = mean_abs_err_dk
        result['max_abs_err_dv'] = max_abs_err_dv
        result['mean_abs_err_dv'] = mean_abs_err_dv
    return result


# ─── Resolution presets: name → (width, height) ───
RESOLUTION_PRESETS = {
    "192P":  (336,  192),
    "368P":  (640,  368),
    "480P":  (832,  480),
    "720P": (1280, 720),
}


def make_config(
    duration=30,
    fps=24,
    resolution="480P",
    height=None,
    width=None,
    head_size=5,
    head_dim=128,
    topk_ratio=0.1,
    batch_size=1,
    S_txt=1024,
    BLKQ=64,
    BLKK=64,
    block_fhw=(1, 8, 8),
):
    """Helper to build a config dict from high-level video parameters.

    Resolution can be specified either as a preset name (e.g. "480P") or
    explicit height/width.  When both are given, height/width take precedence.
    """
    if height is None or width is None:
        rw, rh = RESOLUTION_PRESETS[resolution]
        height = height or rh
        width = width or rw
    n_frames = duration * fps // 4 + 1
    h = height // 16
    w = width // 16
    S_aud = duration * 44100 // 1024
    return dict(
        batch_size=batch_size,
        head_size=head_size,
        S_aud=S_aud,
        S_txt=S_txt,
        n_frames=n_frames,
        height=h,
        width=w,
        head_dim=head_dim,
        BLKQ=BLKQ,
        BLKK=BLKK,
        topk_ratio=topk_ratio,
        block_fhw=block_fhw,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark bsa_attn_fwd")
    parser.add_argument('--bwd', action='store_true',
                        help='Benchmark bsa_attn_bwd (backward pass)')
    parser.add_argument('--hfu', action='store_true',
                        help='Use HFU (factor=10) instead of MFU (factor=8) for bwd flops calculation')
    parser.add_argument('--fwd', action='store_true',
                        help='Benchmark bsa_attn_fwd (forward pass)')
    args = parser.parse_args()

    set_seed(42)

    # ─── Define benchmark configurations ───
    configs = [
        dict(duration=30, fps=24, resolution="720P", head_size=1),
        dict(duration=30, fps=24, resolution="720P", head_size=4),
        dict(duration=15, fps=24, resolution="720P", head_size=4),
        dict(duration=30, fps=16, resolution="480P",  head_size=1),
        dict(duration=30, fps=16, resolution="480P",  head_size=4),
        dict(duration=15, fps=16, resolution="480P",  head_size=4),
        dict(duration=30, fps=16, resolution="368P",  head_size=4),
        dict(duration=30, fps=16, resolution="368P",  head_size=8),
    ]

    topk_ratios = [0.1]

    if args.fwd:
        results_table = []
        for cfg_kw in configs:
            for topk in topk_ratios:
                res_name = cfg_kw.get('resolution', 'custom')
                label = f"{res_name}-{cfg_kw['duration']}s-H{cfg_kw['head_size']}"
                cfg = make_config(**cfg_kw, topk_ratio=topk)
                tag = f"{label}-topk{topk}"
                print(f">>> Running config: {tag} ...")
                tag = f"B1-H{cfg_kw['head_size']}"
                try:
                    res = bench_bsa(**cfg)
                    row = [
                        tag,
                        res['S_q'],
                        res['S_k'],
                        f"{1 - res['density']:.3f}",
                        f"{res['fwd_time_ms']:.3f}",
                        f"{res['tflops']:.2f}",
                        f"{res['mfu']:.2f}",
                    ]
                    results_table.append(row)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    fail_row = [tag, "-", "-", "-", "FAILED", str(e)[:40], "-"]
                    results_table.append(fail_row)

        headers = [
            "Config", "S_q", "S_k",
            "Sparsity",
            "FW (ms)", "TF/s", "MFU(%)",
        ]

        print("\n" + "=" * 100)
        print(tabulate(results_table, headers=headers, tablefmt="grid"))

    # ─── Backward benchmark ───
    if args.bwd:
        bwd_results_table = []
        for cfg_kw in configs:
            for topk in topk_ratios:
                res_name = cfg_kw.get('resolution', 'custom')
                label = f"{res_name}-{cfg_kw['duration']}s-H{cfg_kw['head_size']}"
                cfg = make_config(**cfg_kw, topk_ratio=topk)
                tag = f"{label}-topk{topk}"
                print(f">>> Running BWD config: {tag} ...")
                tag = f"B1-H{cfg_kw['head_size']}"
                try:
                    res = bench_bsa_bwd(**cfg, hfu=args.hfu)
                    row = [
                        tag,
                        res['S_q'],
                        res['S_k'],
                        f"{1 - res['density']:.3f}",
                        f"{res['bwd_time_ms']:.3f}",
                        f"{res['tflops']:.2f}",
                        f"{res['mfu']:.2f}",
                    ]
                    bwd_results_table.append(row)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    fail_row = [tag, "-", "-", "-", "FAILED", str(e)[:40], "-"]
                    bwd_results_table.append(fail_row)

        bwd_headers = [
            "Config", "S_q", "S_k",
            "Sparsity",
            "BW (ms)", "TF/s", "MFU(%)",
        ]
        print("\n" + "=" * 100)
        print("Backward Benchmark:")
        print(tabulate(bwd_results_table, headers=bwd_headers, tablefmt="grid"))
