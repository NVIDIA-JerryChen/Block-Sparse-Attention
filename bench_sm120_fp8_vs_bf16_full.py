"""Benchmark SM120 block-64 BF16 and FP8 attention on full and 10% masks."""

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


BLOCK_SIZE = 64
SPARSE_KEEP_FRACTION = 0.10


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


def make_random_sparse_block_index(
    batch: int,
    heads: int,
    num_q_tiles: int,
    num_kv_tiles: int,
    keep_fraction: float,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Select a fixed fraction of unique KV blocks for every Q block."""
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")

    block_sparse_num = max(1, int(num_kv_tiles * keep_fraction))
    random_scores = torch.rand(
        (batch, heads, num_q_tiles, num_kv_tiles),
        device=device,
        generator=generator,
    )
    q2k_block_index = random_scores.topk(
        block_sparse_num,
        dim=-1,
        largest=False,
    ).indices
    q2k_block_index = q2k_block_index.sort(dim=-1).values.to(torch.int32)
    return q2k_block_index, block_sparse_num


def calculate_effective_flops(
    q2k_block_index: torch.Tensor,
    block_sizes: torch.Tensor,
    seqlen_q: int,
    head_dim: int,
) -> int:
    """Count QK and PV FLOPs for the valid tokens selected by the block map."""
    num_q_tiles = q2k_block_index.shape[2]
    q_block_sizes = torch.full(
        (num_q_tiles,),
        BLOCK_SIZE,
        device=q2k_block_index.device,
        dtype=torch.int64,
    )
    q_block_sizes[-1] = seqlen_q - (num_q_tiles - 1) * BLOCK_SIZE
    selected_k_tokens = block_sizes[q2k_block_index.long()].sum(dim=-1)
    token_pairs = (
        selected_k_tokens * q_block_sizes.view(1, 1, num_q_tiles)
    ).sum()
    return 4 * head_dim * int(token_pairs.item())


def make_query_samples(seqlen_q: int, sample_count: int) -> list[int]:
    """Choose evenly spaced query rows, including the first and last rows."""
    sample_count = min(sample_count, seqlen_q)
    if sample_count == 1:
        return [0]
    return [
        sample_idx * (seqlen_q - 1) // (sample_count - 1)
        for sample_idx in range(sample_count)
    ]


def make_random_partial_block_sizes(
    base_block_sizes: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    query_indices: list[int],
    partial_block_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shorten random full blocks used by the sampled sparse query rows."""
    partial_block_sizes = base_block_sizes.clone()
    if partial_block_count == 0:
        return partial_block_sizes, torch.empty(
            0,
            device=base_block_sizes.device,
            dtype=torch.int64,
        )

    if query_indices:
        query_tiles = torch.tensor(
            sorted({query_idx // BLOCK_SIZE for query_idx in query_indices}),
            device=q2k_block_index.device,
            dtype=torch.int64,
        )
        candidate_blocks = q2k_block_index.index_select(2, query_tiles)[
            ..., :block_sparse_num
        ]
    else:
        candidate_blocks = q2k_block_index[..., :block_sparse_num]

    candidate_blocks = torch.unique(candidate_blocks).long()
    candidate_blocks = candidate_blocks[
        base_block_sizes[candidate_blocks] == BLOCK_SIZE
    ]
    selected_count = min(partial_block_count, candidate_blocks.numel())
    if selected_count == 0:
        return partial_block_sizes, candidate_blocks

    selection = torch.randperm(
        candidate_blocks.numel(),
        device=candidate_blocks.device,
        generator=generator,
    )[:selected_count]
    selected_blocks = candidate_blocks[selection]
    valid_token_counts = torch.randint(
        1,
        BLOCK_SIZE,
        (selected_count,),
        device=base_block_sizes.device,
        dtype=torch.int32,
        generator=generator,
    )
    selected_blocks, order = selected_blocks.sort()
    valid_token_counts = valid_token_counts[order]
    partial_block_sizes[selected_blocks] = valid_token_counts
    return partial_block_sizes, selected_blocks


def expand_kv_token_indices(
    block_indices: torch.Tensor,
    block_sizes: torch.Tensor,
) -> torch.Tensor:
    """Expand KV block indices into valid token indices."""
    block_indices = block_indices.long()
    token_offsets = torch.arange(
        BLOCK_SIZE,
        device=block_indices.device,
        dtype=torch.int64,
    )
    token_indices = block_indices[:, None] * BLOCK_SIZE + token_offsets[None, :]
    valid_tokens = token_offsets[None, :] < block_sizes[block_indices, None]
    return token_indices[valid_tokens]


def sampled_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: torch.Tensor,
    query_indices: list[int],
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute independent PyTorch references for selected query rows."""
    is_fp8 = q_descale is not None
    if is_fp8 != (k_descale is not None and v_descale is not None):
        raise ValueError("q_descale, k_descale, and v_descale must be provided together")

    batch, heads, _, head_dim = q.shape
    reference_out = torch.empty(
        (batch, heads, len(query_indices), head_dim),
        device=q.device,
        dtype=torch.float32,
    )
    reference_lse = torch.empty(
        (batch, heads, len(query_indices)),
        device=q.device,
        dtype=torch.float32,
    )
    v_descale_hd = v_descale.view(heads, head_dim) if is_fp8 else None
    softmax_scale = head_dim ** -0.5

    for batch_idx in range(batch):
        for head_idx in range(heads):
            for sample_idx, query_idx in enumerate(query_indices):
                query_tile = query_idx // BLOCK_SIZE
                selected_blocks = q2k_block_index[
                    batch_idx,
                    head_idx,
                    query_tile,
                    :block_sparse_num,
                ]
                kv_token_indices = expand_kv_token_indices(
                    selected_blocks,
                    block_sizes,
                )

                q_row = q[batch_idx, head_idx, query_idx].float()
                k_rows = k[batch_idx, head_idx, kv_token_indices].float()
                v_rows = v[batch_idx, head_idx, kv_token_indices].float()
                if is_fp8:
                    q_row = q_row * q_descale[batch_idx, head_idx, query_idx]
                    k_rows = k_rows * k_descale[
                        batch_idx,
                        head_idx,
                        kv_token_indices // 16,
                    ][:, None]
                    v_rows = v_rows * v_descale_hd[head_idx][None, :]

                scores = torch.mv(k_rows, q_row) * softmax_scale
                probabilities = torch.softmax(scores, dim=0)
                reference_out[batch_idx, head_idx, sample_idx] = torch.matmul(
                    probabilities,
                    v_rows,
                )
                reference_lse[batch_idx, head_idx, sample_idx] = torch.logsumexp(
                    scores,
                    dim=0,
                )

    return reference_out, reference_lse


def check_sampled_correctness(
    name: str,
    out: torch.Tensor,
    lse: torch.Tensor,
    reference_out: torch.Tensor,
    reference_lse: torch.Tensor,
    query_indices: list[int],
    out_rtol: float,
    out_atol: float,
    lse_rtol: float,
    lse_atol: float,
) -> None:
    """Check sampled kernel results and print compact error statistics."""
    sampled_out = out[:, :, query_indices, :].float()
    sampled_lse = lse[:, :, query_indices].float()
    out_diff = (sampled_out - reference_out).abs()
    lse_diff = (sampled_lse - reference_lse).abs()

    torch.testing.assert_close(
        sampled_out,
        reference_out,
        rtol=out_rtol,
        atol=out_atol,
    )
    torch.testing.assert_close(
        sampled_lse,
        reference_lse,
        rtol=lse_rtol,
        atol=lse_atol,
    )
    print(
        f"{name} correctness: PASS, rows/head={len(query_indices)}, "
        f"out_max={out_diff.max().item():.6f}, "
        f"out_mean={out_diff.mean().item():.6f}, "
        f"lse_max={lse_diff.max().item():.6f}"
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
        description=(
            "Compare SM120 block-64 BF16 and FP8 performance with full and 10% masks."
        )
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--sq", type=int, default=75600)
    parser.add_argument("--sk", type=int, default=75600)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--correctness-rows",
        type=int,
        default=4,
        help="Number of sampled query rows per batch/head; use 0 to disable checks.",
    )
    parser.add_argument(
        "--partial-blocks",
        type=int,
        default=8,
        help="Number of randomly shortened KV blocks in the partial-token case.",
    )
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
    if args.correctness_rows < 0:
        raise ValueError("correctness-rows must be non-negative")
    if args.partial_blocks < 0:
        raise ValueError("partial-blocks must be non-negative")

    torch.manual_seed(args.seed)
    query_indices = make_query_samples(args.sq, args.correctness_rows)
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

    num_q_tiles = math.ceil(args.sq / BLOCK_SIZE)
    num_kv_tiles = math.ceil(args.sk / BLOCK_SIZE)
    indices = torch.arange(num_kv_tiles, device="cuda", dtype=torch.int32)
    q2k_block_index_full = indices.view(1, 1, 1, num_kv_tiles).expand(
        args.batch,
        args.heads,
        num_q_tiles,
        num_kv_tiles,
    ).contiguous()
    block_sizes = torch.full(
        (num_kv_tiles,),
        BLOCK_SIZE,
        device="cuda",
        dtype=torch.int32,
    )
    block_sizes[-1] = args.sk - (num_kv_tiles - 1) * BLOCK_SIZE

    sparse_generator = torch.Generator(device="cuda")
    sparse_generator.manual_seed(args.seed)
    q2k_block_index_sparse, sparse_block_num = make_random_sparse_block_index(
        args.batch,
        args.heads,
        num_q_tiles,
        num_kv_tiles,
        SPARSE_KEEP_FRACTION,
        q_bf16.device,
        sparse_generator,
    )

    out_bf16 = torch.empty(shape_q, device="cuda", dtype=torch.bfloat16)
    out_fp8 = torch.empty_like(out_bf16)
    lse_shape = (args.batch, args.heads, args.sq)
    lse_bf16 = torch.empty(lse_shape, device="cuda", dtype=torch.float32)
    lse_fp8 = torch.empty_like(lse_bf16)

    def make_kernel_calls(
        q2k_block_index: torch.Tensor,
        block_sparse_num: int,
        case_block_sizes: torch.Tensor,
    ) -> tuple[Callable[[], None], Callable[[], None]]:
        def call_bf16() -> None:
            bsa_attn_fwd(
                q_bf16,
                k_bf16,
                v_bf16,
                q2k_block_index,
                block_sparse_num=block_sparse_num,
                block_sizes=case_block_sizes,
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
                block_sparse_num=block_sparse_num,
                block_sizes=case_block_sizes,
                return_lse=True,
                out=out_fp8,
                lse=lse_fp8,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )

        return call_bf16, call_fp8

    dense_effective_flops = (
        4 * args.batch * args.heads * args.sq * args.sk * args.head_dim
    )
    partial_generator = torch.Generator(device="cuda")
    partial_generator.manual_seed(args.seed + 1)
    partial_block_sizes, partial_block_indices = make_random_partial_block_sizes(
        block_sizes,
        q2k_block_index_sparse,
        sparse_block_num,
        query_indices,
        args.partial_blocks,
        partial_generator,
    )

    def check_case_correctness(
        name: str,
        q2k_block_index: torch.Tensor,
        block_sparse_num: int,
        case_block_sizes: torch.Tensor,
        partial_token_case: bool,
    ) -> None:
        if not query_indices:
            return

        reference_out_bf16, reference_lse_bf16 = sampled_attention_reference(
            q_bf16,
            k_bf16,
            v_bf16,
            q2k_block_index,
            block_sparse_num,
            case_block_sizes,
            query_indices,
        )
        check_sampled_correctness(
            f"BF16 {name}",
            out_bf16,
            lse_bf16,
            reference_out_bf16,
            reference_lse_bf16,
            query_indices,
            out_rtol=2.0e-2 if partial_token_case else 3.0e-2,
            out_atol=5.0e-4 if partial_token_case else 3.0e-2,
            lse_rtol=1.0e-3 if partial_token_case else 5.0e-3,
            lse_atol=1.0e-3 if partial_token_case else 3.0e-2,
        )

        reference_out_fp8, reference_lse_fp8 = sampled_attention_reference(
            q_fp8,
            k_fp8,
            v_fp8,
            q2k_block_index,
            block_sparse_num,
            case_block_sizes,
            query_indices,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        check_sampled_correctness(
            f"FP8 E4M3FN {name}",
            out_fp8,
            lse_fp8,
            reference_out_fp8,
            reference_lse_fp8,
            query_indices,
            out_rtol=2.0e-2 if partial_token_case else 5.0e-2,
            out_atol=3.0e-3 if partial_token_case else 2.0e-2,
            lse_rtol=1.0e-3,
            lse_atol=1.0e-3,
        )

    def run_case(
        name: str,
        q2k_block_index: torch.Tensor,
        block_sparse_num: int,
        case_block_sizes: torch.Tensor,
        partial_token_case: bool = False,
    ) -> None:
        effective_flops = calculate_effective_flops(
            q2k_block_index,
            case_block_sizes,
            args.sq,
            args.head_dim,
        )
        effective_density = effective_flops / dense_effective_flops
        print(
            f"\n{name}: blocks={block_sparse_num}/{num_kv_tiles} "
            f"({block_sparse_num / num_kv_tiles:.2%}), "
            f"effective token density={effective_density:.2%}"
        )

        call_bf16, call_fp8 = make_kernel_calls(
            q2k_block_index,
            block_sparse_num,
            case_block_sizes,
        )
        compile_once(f"BF16 {name}", call_bf16)
        compile_once(f"FP8 E4M3FN {name}", call_fp8)
        check_case_correctness(
            name,
            q2k_block_index,
            block_sparse_num,
            case_block_sizes,
            partial_token_case,
        )

        bf16, fp8 = benchmark_pair(
            call_bf16,
            call_fp8,
            effective_flops,
            args.warmup,
            args.iterations,
        )
        print_result(f"BF16 {name}", bf16)
        print_result(f"FP8 E4M3FN {name}", fp8)
        print(f"FP8/BF16 {name} speedup: {bf16['median_ms'] / fp8['median_ms']:.3f}x")

    print(f"GPU: {torch.cuda.get_device_name()} (SM{capability[0]}{capability[1]})")
    print(
        f"Shape: B={args.batch}, H={args.heads}, Sq={args.sq}, "
        f"Sk={args.sk}, D={args.head_dim}, Q_tiles={num_q_tiles}, "
        f"KV_tiles={num_kv_tiles}, last_KV={block_sizes[-1].item()}"
    )
    print(
        f"FP8 descales: q={tuple(q_descale.shape)}, "
        f"k={tuple(k_descale.shape)}, v={tuple(v_descale.shape)}"
    )
    if partial_block_indices.numel() > 0:
        partial_blocks = partial_block_indices.tolist()
        partial_sizes = partial_block_sizes[partial_block_indices].tolist()
        partial_summary = ", ".join(
            f"{block_idx}:{valid_tokens}"
            for block_idx, valid_tokens in zip(partial_blocks, partial_sizes)
        )
        print(f"Partial KV blocks (block:valid_tokens): {partial_summary}")

    run_case(
        "full",
        q2k_block_index_full,
        num_kv_tiles,
        block_sizes,
    )
    run_case(
        "10% blocks",
        q2k_block_index_sparse,
        sparse_block_num,
        block_sizes,
    )
    if partial_block_indices.numel() > 0:
        run_case(
            "10% blocks + partial tokens",
            q2k_block_index_sparse,
            sparse_block_num,
            partial_block_sizes,
            partial_token_case=True,
        )


if __name__ == "__main__":
    main()
