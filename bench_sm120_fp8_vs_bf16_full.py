"""Benchmark SM120 block-64 attention with BF16 and FP8 on a full mask."""

import argparse
import math
import os
import statistics
import sys
import time
from collections.abc import Callable

import torch


REPO_ROOT = "/workdir/tmp/Block-Sparse-Attention"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from bsa_attn_interface import bsa_attn_fwd  # noqa: E402


def quantize_sage(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Apply Q-token, K-16-token, and V-channel E4M3 quantization."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    q_float = q.float()
    q_descale = q_float.abs().amax(dim=-1).clamp_min(1.0e-3) / fp8_max
    q_fp8 = (q_float / q_descale[..., None]).to(torch.float8_e4m3fn)

    k_float = k.float()
    k_centered = k_float - k_float.mean(dim=2, keepdim=True)
    k_blocks = math.ceil(k.shape[2] / 16)
    k_padded = torch.nn.functional.pad(k_centered, (0, 0, 0, k_blocks * 16 - k.shape[2]))
    k_descale = (
        k_padded.view(k.shape[0], k.shape[1], k_blocks, 16, k.shape[-1])
        .abs()
        .amax(dim=(3, 4))
        .clamp_min(1.0e-3)
        / fp8_max
    )
    k_token_descale = k_descale.repeat_interleave(16, dim=-1)[..., : k.shape[2]]
    k_fp8 = (k_centered / k_token_descale[..., None]).to(torch.float8_e4m3fn)

    v_float = v.float()
    v_descale_hd = v_float.abs().amax(dim=(0, 2)).clamp_min(1.0e-3) / fp8_max
    v_fp8 = (v_float / v_descale_hd[None, :, None, :]).to(torch.float8_e4m3fn)
    return (
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale_hd.flatten().contiguous(),
    )


def compile_once(name: str, call_kernel: Callable[[], None]) -> float:
    """Time the first call, which includes CuTe compilation and one launch."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    call_kernel()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{name} first call (compile + run): {elapsed:.3f}s")
    return elapsed


def summarize(times_ms: list[float], effective_flops: int) -> dict[str, float]:
    ordered = sorted(times_ms)
    count = len(ordered)
    median_ms = statistics.median(ordered)
    return {
        "median_ms": median_ms,
        "min_ms": ordered[0],
        "p20_ms": ordered[count // 5],
        "p80_ms": ordered[(count * 4) // 5],
        "tflops": effective_flops / (median_ms * 1.0e-3) / 1.0e12,
    }


def print_result(name: str, result: dict[str, float]) -> None:
    print(
        f"{name}: median={result['median_ms']:.4f} ms, "
        f"min={result['min_ms']:.4f} ms, "
        f"p20={result['p20_ms']:.4f} ms, "
        f"p80={result['p80_ms']:.4f} ms, "
        f"effective={result['tflops']:.2f} TFLOP/s"
    )


def benchmark_pair(
    call_bf16: Callable[[], None],
    call_fp8: Callable[[], None],
    effective_flops: int,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Warm up and time both kernels in alternating launch order."""
    for _ in range(warmup):
        call_bf16()
        call_fp8()
    torch.cuda.synchronize()

    bf16_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iterations)
    ]
    fp8_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iterations)
    ]

    def record(
        events: tuple[torch.cuda.Event, torch.cuda.Event],
        call_kernel: Callable[[], None],
    ) -> None:
        start, end = events
        start.record()
        call_kernel()
        end.record()

    for idx in range(iterations):
        if idx % 2 == 0:
            record(bf16_events[idx], call_bf16)
            record(fp8_events[idx], call_fp8)
        else:
            record(fp8_events[idx], call_fp8)
            record(bf16_events[idx], call_bf16)
    torch.cuda.synchronize()

    bf16_times = [start.elapsed_time(end) for start, end in bf16_events]
    fp8_times = [start.elapsed_time(end) for start, end in fp8_events]
    return (
        summarize(bf16_times, effective_flops),
        summarize(fp8_times, effective_flops),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SM120 block-64 BF16 and FP8 full-attention performance."
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--sq", type=int, default=9450)
    parser.add_argument("--sk", type=int, default=9450)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 12:
        raise RuntimeError(f"This benchmark requires SM120, got SM{capability[0]}{capability[1]}")
    if args.head_dim != 128:
        raise ValueError("The SM120 block-64 kernel currently requires head_dim=128")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    torch.manual_seed(args.seed)
    shape_q = (args.batch, args.heads, args.sq, args.head_dim)
    shape_kv = (args.batch, args.heads, args.sk, args.head_dim)
    q_float = torch.randn(shape_q, device="cuda", dtype=torch.float32)
    k_float = torch.randn(shape_kv, device="cuda", dtype=torch.float32)
    v_float = torch.randn(shape_kv, device="cuda", dtype=torch.float32)

    q_bf16 = q_float.to(torch.bfloat16)
    k_bf16 = k_float.to(torch.bfloat16)
    v_bf16 = v_float.to(torch.bfloat16)
    q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale = quantize_sage(
        q_bf16,
        k_bf16,
        v_bf16,
    )

    block = 64
    num_q_tiles = math.ceil(args.sq / block)
    num_kv_tiles = math.ceil(args.sk / block)
    indices = torch.arange(num_kv_tiles, device="cuda", dtype=torch.int32)
    q2k_block_index = indices.view(1, 1, 1, num_kv_tiles).expand(
        args.batch,
        args.heads,
        num_q_tiles,
        num_kv_tiles,
    ).contiguous()
    block_sizes = torch.full(
        (num_kv_tiles,),
        block,
        device="cuda",
        dtype=torch.int32,
    )
    block_sizes[-1] = args.sk - (num_kv_tiles - 1) * block

    out_bf16 = torch.empty(shape_q, device="cuda", dtype=torch.bfloat16)
    out_fp8 = torch.empty_like(out_bf16)
    lse_shape = (args.batch, args.heads, args.sq)
    lse_bf16 = torch.empty(lse_shape, device="cuda", dtype=torch.float32)
    lse_fp8 = torch.empty_like(lse_bf16)

    def call_bf16() -> None:
        bsa_attn_fwd(
            q_bf16,
            k_bf16,
            v_bf16,
            q2k_block_index,
            block_sparse_num=num_kv_tiles,
            block_sizes=block_sizes,
            return_lse=True,
            out=out_bf16,
            lse=lse_bf16,
        )

    def call_fp8() -> None:
        bsa_attn_fwd(
            q_fp8,
            k_fp8,
            v_fp8,
            q2k_block_index,
            block_sparse_num=num_kv_tiles,
            block_sizes=block_sizes,
            return_lse=True,
            out=out_fp8,
            lse=lse_fp8,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    print(f"GPU: {torch.cuda.get_device_name()} (SM{capability[0]}{capability[1]})")
    print(
        f"Full mask: B={args.batch}, H={args.heads}, Sq={args.sq}, "
        f"Sk={args.sk}, D={args.head_dim}, Q_tiles={num_q_tiles}, "
        f"KV_tiles={num_kv_tiles}, last_KV={block_sizes[-1].item()}"
    )
    print(
        f"FP8 descales: q={tuple(q_descale.shape)}, "
        f"k={tuple(k_descale.shape)}, v={tuple(v_descale.shape)}"
    )

    compile_once("BF16", call_bf16)
    compile_once("FP8 E4M3FN", call_fp8)

    effective_flops = (
        4 * args.batch * args.heads * args.sq * args.sk * args.head_dim
    )
    bf16, fp8 = benchmark_pair(
        call_bf16,
        call_fp8,
        effective_flops,
        args.warmup,
        args.iterations,
    )
    print_result("BF16 full", bf16)
    print_result("FP8 E4M3FN full", fp8)
    print(f"FP8/BF16 speedup: {bf16['median_ms'] / fp8['median_ms']:.3f}x")


if __name__ == "__main__":
    main()
