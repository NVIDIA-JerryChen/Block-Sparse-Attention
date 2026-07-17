#!/usr/bin/env python3
"""Benchmark the SM90/SM100/SM110 CuTe DSL block-sparse attention kernels.

This is a standalone performance test rather than a correctness test. It uses
the complete shape matrix from ``微信图片_20260703143144_42_7.png`` and reports
steady-state forward and backward CUDA-event latency and effective sparse
TFLOPS.

Backward timing covers the complete CuTe DSL path: bucketed K-to-Q CSR
construction, preprocessing, the attention backward kernel, and postprocessing.
The unmeasured forward setup only prepares ``out`` and ``lse`` for backward.
Blk64 forward uses automatic split-KV selection for best end-to-end performance.

Usage:
    python -u tests/bench_cutedsl_performance.py
    python -u tests/bench_cutedsl_performance.py --direction bwd
    python -u tests/bench_cutedsl_performance.py --block-size 64
"""

import argparse
import gc
import statistics
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import torch


# Keep direct script execution working from any current directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from block_sparse_attention import (  # noqa: E402
    bsa_attn_bwd,
    bsa_attn_fwd,
)


BATCH_SIZE = 1
NUM_HEADS = 40
SEQLEN_Q = 131072
SEQLEN_K = 131072
HEAD_DIM = 128
DTYPE = torch.bfloat16

# The image specifies these blk64 topK values. Other block sizes preserve the
# same number of attended KV tokens, and therefore the same sparsity.
BLK64_TOPKS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
SUPPORTED_BLOCK_SIZES = (64, 128)
SUPPORTED_DIRECTIONS = ("fwd", "bwd")


@dataclass(frozen=True)
class BenchmarkResult:
    direction: str
    path: str
    block_size: int
    topk: int
    sparsity_pct: float
    time_ms: float
    tflops: float


@contextmanager
def _suppress_cutlass_pointer_deprecation() -> Iterator[None]:
    """Hide one known CuTe DSL compile warning while preserving all others."""
    original_showwarning = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        if (
            issubclass(category, DeprecationWarning)
            and str(message)
            == "Use explicit `struct.scalar.ptr` for pointer instead."
        ):
            return
        original_showwarning(
            message,
            category,
            filename,
            lineno,
            file=file,
            line=line,
        )

    warnings.showwarning = showwarning
    try:
        yield
    finally:
        warnings.showwarning = original_showwarning


def _topks_for_block_size(block_size: int) -> tuple[int, ...]:
    assert block_size in SUPPORTED_BLOCK_SIZES
    assert all(topk * 64 % block_size == 0 for topk in BLK64_TOPKS)
    topks = tuple(topk * 64 // block_size for topk in BLK64_TOPKS)
    return topks


def _device_arch() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _device_arch_major() -> int:
    return _device_arch() // 10


def _supported_block_sizes_for_device() -> tuple[int, ...]:
    return (64,) if _device_arch_major() == 9 else SUPPORTED_BLOCK_SIZES


def _make_qkv(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create BHSD BF16 inputs in the same manner as bench_bsa.py."""
    torch.manual_seed(seed)
    shape = (BATCH_SIZE, NUM_HEADS, SEQLEN_Q, HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=DTYPE)
    k = torch.randn(shape, device="cuda", dtype=DTYPE)
    v = torch.randn(shape, device="cuda", dtype=DTYPE)
    return q, k, v


def _make_block_sparse_inputs(
    block_size: int,
    topk: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sorted random topK indices and full block sizes.

    ``bench_bsa.py`` selects blocks from random pooled scores and then places
    selected block IDs in ascending order. Sorting the selected IDs directly
    reproduces that active prefix without materializing a dense sparse map.
    """
    num_q_blocks = (SEQLEN_Q + block_size - 1) // block_size
    num_kv_blocks = (SEQLEN_K + block_size - 1) // block_size
    assert 0 < topk <= num_kv_blocks

    if topk == num_kv_blocks:
        block_ids = torch.arange(num_kv_blocks, device="cuda", dtype=torch.int32)
        q2k_block_index = block_ids.view(1, 1, 1, -1).expand(
            BATCH_SIZE,
            NUM_HEADS,
            num_q_blocks,
            num_kv_blocks,
        ).contiguous()
    else:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        pooled_score = torch.randn(
            BATCH_SIZE,
            NUM_HEADS,
            num_q_blocks,
            num_kv_blocks,
            device="cuda",
            generator=generator,
        )
        selected = torch.topk(
            pooled_score,
            topk,
            dim=-1,
            sorted=False,
        ).indices
        q2k_block_index = selected.sort(dim=-1).values.to(torch.int32).contiguous()
        del pooled_score, selected

    block_sizes = torch.full(
        (num_kv_blocks,),
        block_size,
        device="cuda",
        dtype=torch.int32,
    )
    return q2k_block_index, block_sizes


def _make_fwd_kernel_call(
    block_size: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sizes: torch.Tensor,
    topk: int,
) -> tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    # The same callable prepares out/LSE for backward, so every path requests
    # LSE explicitly instead of depending on input grad state.
    arch = _device_arch()
    arch_major = _device_arch_major()
    if arch_major == 9:
        assert block_size == 64
        path_name = "sm90_blk64_cutedsl_fwd"
        num_q_blocks = (SEQLEN_Q + block_size - 1) // block_size
        q2k_block_nums = torch.full(
            (BATCH_SIZE, NUM_HEADS, num_q_blocks),
            topk,
            dtype=torch.int32,
            device=q.device,
        )
        block_sizes_bh = (
            block_sizes.view(1, 1, -1)
            .expand(BATCH_SIZE, NUM_HEADS, -1)
            .contiguous()
        )

        def run_kernel() -> tuple[torch.Tensor, torch.Tensor]:
            return bsa_attn_fwd(
                q,
                k,
                v,
                q2k_block_index,
                topk,
                block_sizes_bh,
                q2k_block_nums=q2k_block_nums,
                return_lse=True,
                layout="bhsd",
                kv_splits="auto",
                sparse_block_size=64,
            )

        return path_name, run_kernel

    if block_size == 64:
        path_name = f"sm{arch}_blk64_cutedsl_fwd"

        def run_kernel() -> tuple[torch.Tensor, torch.Tensor]:
            return bsa_attn_fwd(
                q,
                k,
                v,
                q2k_block_index,
                topk,
                block_sizes,
                q2k_block_nums=None,
                return_lse=True,
                layout="bhsd",
                use_clc=None,
                kv_splits="auto",
                sparse_block_size=64,
            )

        return path_name, run_kernel

    assert block_size == 128
    path_name = f"sm{arch}_blk128_cutedsl_fwd"

    def run_kernel() -> tuple[torch.Tensor, torch.Tensor]:
        return bsa_attn_fwd(
            q,
            k,
            v,
            q2k_block_index,
            topk,
            block_sizes,
            q2k_block_nums=None,
            return_lse=True,
            layout="bhsd",
            sparse_block_size=128,
        )

    return path_name, run_kernel


def _make_bwd_kernel_call(
    block_size: int,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sizes: torch.Tensor,
    topk: int,
) -> tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    assert block_size in SUPPORTED_BLOCK_SIZES
    path_name = f"sm{_device_arch()}_blk{block_size}_cutedsl_bwd_pipeline"

    def run_kernel() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return bsa_attn_bwd(
            dout,
            q,
            k,
            v,
            out,
            lse,
            q2k_block_index,
            topk,
            block_sizes,
            q2k_block_nums=None,
            dq=dq,
            dk=dk,
            dv=dv,
            layout="bhsd",
            sparse_block_size=block_size,
        )

    return path_name, run_kernel


def _benchmark_cuda_events(
    fn: Callable[[], tuple[torch.Tensor, ...]],
    warmup: int,
    runs: int,
) -> float:
    """Return steady-state median latency in milliseconds."""
    # The first call performs JIT compilation. Compilation is intentionally
    # excluded from the reported steady-state kernel latency.
    with _suppress_cutlass_pointer_deprecation():
        result = fn()
    del result
    torch.cuda.synchronize()

    for _ in range(warmup):
        result = fn()
        del result
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    for start, end in zip(starts, ends):
        start.record()
        result = fn()
        end.record()
        del result
    torch.cuda.synchronize()

    times_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return statistics.median(times_ms)


def _effective_tflops(
    direction: str,
    block_size: int,
    topk: int,
    time_ms: float,
) -> float:
    assert direction in SUPPORTED_DIRECTIONS
    effective_seqlen_k = topk * block_size
    # Forward uses QK + PV (factor 4 for D == Dv). Backward HFU includes the
    # score recomputation and four gradient GEMMs (factor 10).
    factor = 4 if direction == "fwd" else 10
    flop_count = (
        factor
        * BATCH_SIZE
        * NUM_HEADS
        * SEQLEN_Q
        * effective_seqlen_k
        * HEAD_DIM
    )
    return flop_count / (time_ms * 1e-3) / 1e12


def _benchmark_case(
    directions: Sequence[str],
    block_size: int,
    topk: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bwd_tensors: Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    warmup: int,
    runs: int,
    seed: int,
) -> Iterator[BenchmarkResult]:
    assert all(direction in SUPPORTED_DIRECTIONS for direction in directions)
    q2k_block_index, block_sizes = _make_block_sparse_inputs(
        block_size,
        topk,
        seed,
    )
    fwd_kernel_name, run_fwd = _make_fwd_kernel_call(
        block_size,
        q,
        k,
        v,
        q2k_block_index,
        block_sizes,
        topk,
    )
    sparsity_pct = (1.0 - topk * block_size / SEQLEN_K) * 100.0
    if "fwd" in directions:
        time_ms = _benchmark_cuda_events(run_fwd, warmup, runs)
        yield BenchmarkResult(
            direction="fwd",
            path=fwd_kernel_name,
            block_size=block_size,
            topk=topk,
            sparsity_pct=sparsity_pct,
            time_ms=time_ms,
            tflops=_effective_tflops("fwd", block_size, topk, time_ms),
        )

    if "bwd" in directions:
        assert bwd_tensors is not None
        with _suppress_cutlass_pointer_deprecation():
            out, lse = run_fwd()
        torch.cuda.synchronize()
        dout, dq, dk, dv = bwd_tensors
        bwd_path_name, run_bwd = _make_bwd_kernel_call(
            block_size,
            dout,
            q,
            k,
            v,
            out,
            lse,
            dq,
            dk,
            dv,
            q2k_block_index,
            block_sizes,
            topk,
        )
        time_ms = _benchmark_cuda_events(run_bwd, warmup, runs)
        yield BenchmarkResult(
            direction="bwd",
            path=bwd_path_name,
            block_size=block_size,
            topk=topk,
            sparsity_pct=sparsity_pct,
            time_ms=time_ms,
            tflops=_effective_tflops("bwd", block_size, topk, time_ms),
        )
        del run_bwd, out, lse

    del run_fwd, q2k_block_index, block_sizes
    gc.collect()
    torch.cuda.empty_cache()


def _validate_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the CuTe DSL performance test")
    major, minor = torch.cuda.get_device_capability()
    if major not in (9, 10, 11):
        raise RuntimeError(
            "This performance matrix requires an SM90/SM100/SM110 GPU; "
            f"found compute capability {major}.{minor}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction",
        choices=SUPPORTED_DIRECTIONS,
        action="append",
        dest="directions",
        help="direction to benchmark; repeat to select both (default: fwd and bwd)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        choices=SUPPORTED_BLOCK_SIZES,
        action="append",
        dest="block_sizes",
        help=(
            "block size to benchmark; repeat to select multiple "
            "(default: all supported by the current GPU)"
        ),
    )
    parser.add_argument("--warmup", type=int, default=3, help="warmup launches per case")
    parser.add_argument("--runs", type=int, default=10, help="timed launches per case")
    parser.add_argument("--seed", type=int, default=42, help="random input seed")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.directions is None:
        args.directions = list(SUPPORTED_DIRECTIONS)
    else:
        args.directions = list(dict.fromkeys(args.directions))
    if args.block_sizes is not None:
        args.block_sizes = list(dict.fromkeys(args.block_sizes))
    return args


def run_benchmark_suite(
    directions: Sequence[str],
    block_sizes: Optional[Sequence[int]],
    warmup: int,
    runs: int,
    seed: int,
) -> None:
    _validate_environment()
    supported_block_sizes = _supported_block_sizes_for_device()
    if block_sizes is None:
        block_sizes = supported_block_sizes
    unsupported_block_sizes = set(block_sizes) - set(supported_block_sizes)
    if unsupported_block_sizes:
        raise ValueError(
            f"block size(s) {sorted(unsupported_block_sizes)} are unsupported on "
            f"SM{_device_arch()}; supported: {list(supported_block_sizes)}"
        )

    device_name = torch.cuda.get_device_name()
    print(
        f"GPU: {device_name} (SM{_device_arch()}); "
        f"B={BATCH_SIZE}, Hq=Hkv={NUM_HEADS}, "
        f"Sq=Sk={SEQLEN_Q}, D={HEAD_DIM}, dtype=bf16",
        flush=True,
    )
    print(
        "TFLOPS convention: fwd factor=4; bwd factor=10 (HFU)",
        flush=True,
    )
    print(
        f"{'dir':<4} {'path':<34} {'blk':>5} {'topK':>6} {'sparsity':>10} "
        f"{'median GPU ms':>14} {'TFLOPS':>12}",
        flush=True,
    )
    print("-" * 93, flush=True)

    q, k, v = _make_qkv(seed)
    bwd_tensors = None
    if "bwd" in directions:
        torch.manual_seed(seed + 1)
        bwd_tensors = (
            torch.randn_like(q),
            torch.empty_like(q),
            torch.empty_like(k),
            torch.empty_like(v),
        )
    try:
        for block_size in block_sizes:
            for topk in _topks_for_block_size(block_size):
                for result in _benchmark_case(
                    directions,
                    block_size,
                    topk,
                    q,
                    k,
                    v,
                    bwd_tensors,
                    warmup,
                    runs,
                    seed + block_size + topk,
                ):
                    print(
                        f"{result.direction:<4} {result.path:<34} "
                        f"{result.block_size:>5} {result.topk:>6} "
                        f"{result.sparsity_pct:>9.1f}% {result.time_ms:>14.3f} "
                        f"{result.tflops:>12.1f}",
                        flush=True,
                    )
    finally:
        del bwd_tensors, q, k, v
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    args = _parse_args()
    run_benchmark_suite(
        directions=args.directions,
        block_sizes=args.block_sizes,
        warmup=args.warmup,
        runs=args.runs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
