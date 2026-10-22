# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native SM120 NVFP4-QK BSA with Sage FP8 BSA.

Run on an idle SM120 GPU: ``python tests/bench_sm120_nvfp4_blk64.py``. Shapes
are given as BSHD (batch, sequence, heads, head_dim) and run as self-attention.
NVFP4 blk64 and Sage FP8 blk64 share exactly the same sparse mask. Attention
and quantizer times use rotating CUDA-graph replays; attention runs on
pre-quantized inputs and the quantizers are reported separately: Sage FP8 Q/K/V
via this repo's Triton ``quantize_sage_bhsd``, and the NVFP4 path's NVFP4 Q/K +
Sage FP8 V via the fused ``quantize_nvfp4_qk_sage_v_bhsd``.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bsa_attn_interface as _interface  # noqa: E402
from bsa_fp8_quant import quantize_sage_bhsd  # noqa: E402
from bsa_nvfp4_quant import quantize_nvfp4_qk_sage_v_bhsd  # noqa: E402


def _capture(fn, warmup):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(max(1, warmup)):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    return graph, output


def _rotating_times(graphs, repeats):
    """Replay every graph once per repeat, rotating the order so no kernel always runs first."""
    events = [[] for _ in graphs]
    for repeat in range(repeats):
        for offset in range(len(graphs)):
            index = (repeat + offset) % len(graphs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[index].replay()
            end.record()
            events[index].append((start, end))
    torch.cuda.synchronize()
    return [[start.elapsed_time(end) for start, end in pairs] for pairs in events]


def _block_indices(heads, num_q_blocks, num_kv_blocks, topk, device):
    """Distinct random K blocks per (head, Q block); all blocks, in order, when topk == num_kv_blocks."""
    if topk == num_kv_blocks:
        return torch.arange(num_kv_blocks, device=device, dtype=torch.int32).expand(1, heads, num_q_blocks, -1).contiguous()
    order = torch.rand((1, heads, num_q_blocks, num_kv_blocks), device=device).argsort(-1)
    return order[..., :topk].sort(-1).values.to(torch.int32).contiguous()


def _attention_flops(indices, seqlen, block_size, head_dim):
    """Exact QK^T + PV FLOPs of the mask, counting partial tail blocks and tail Q rows."""
    num_kv_blocks = (seqlen + block_size - 1) // block_size
    sizes = torch.full((num_kv_blocks,), block_size, device=indices.device)
    sizes[-1] = seqlen - (num_kv_blocks - 1) * block_size
    keys_per_q_block = sizes[indices.long()].sum(-1).double()
    rows = torch.full((indices.shape[2],), block_size, device=indices.device, dtype=torch.float64)
    rows[-1] = seqlen - (indices.shape[2] - 1) * block_size
    return 4.0 * head_dim * (keys_per_q_block * rows).sum().item()


def _relative_l2(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)).item()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=("1,6045,4,128", "1,51200,4,128"), help="BSHD shapes, e.g. 1,6045,4,128")
    parser.add_argument("--sparsities", type=float, nargs="+", default=(0.0, 0.9), help="fraction of K blocks skipped per Q block")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise RuntimeError("this benchmark requires SM120")
    if any(not 0 <= sparsity < 1 for sparsity in args.sparsities) or args.repeats < 1 or args.warmup < 0:
        raise ValueError("sparsity must be in [0, 1); repeats must be positive; warmup must be nonnegative")
    shapes = [tuple(int(value) for value in shape.split(",")) for shape in args.shapes]
    if any(len(shape) != 4 or shape[0] != 1 or shape[3] != 128 for shape in shapes):
        raise ValueError("shapes must be BSHD with batch 1 and head_dim 128")

    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cutlass": str(_interface.cutlass.__version__),
        "timing": "rotating CUDA-graph replay of pre-quantized inputs; medians; milliseconds",
        "cases": [],
    }
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}), flush=True)
    softmax_scale = 128**-0.5
    for batch, seqlen, heads, head_dim in shapes:
        torch.manual_seed(120464)
        # BSHD inputs, transposed to the contiguous BHSD layout the kernels take.
        q, k, v = (torch.randn((batch, seqlen, heads, head_dim), dtype=torch.bfloat16, device="cuda").transpose(1, 2).contiguous() for _ in range(3))
        fp8_64 = quantize_sage_bhsd(q, k, v)
        q4, q_sf, k4, k_sf, v8, v_scale = quantize_nvfp4_qk_sage_v_bhsd(q, k, v)

        # Quantizers run as CUDA graphs too, so the times are GPU work, not CPU launch gaps.
        quantizer_graphs = [
            _capture(fn, args.warmup)[0]
            for fn in (
                lambda: quantize_sage_bhsd(q, k, v),
                lambda: quantize_nvfp4_qk_sage_v_bhsd(q, k, v),
            )
        ]
        quantizer_ms = [statistics.median(times) for times in _rotating_times(quantizer_graphs, args.repeats)]
        del quantizer_graphs

        for sparsity in args.sparsities:
            nq64 = nk64 = (seqlen + 63) // 64
            topk64 = max(1, round((1 - sparsity) * nk64))
            indices64 = _block_indices(heads, nq64, nk64, topk64, q.device)
            kernels = {
                "nvfp4_blk64": (
                    lambda: _interface._bsa_attn_fwd_sm120_nvfp4_blk64(q4, k4, v8, q_sf, k_sf, v_scale, indices64, topk64, softmax_scale)[0],
                    _attention_flops(indices64, seqlen, 64, head_dim),
                ),
                "fp8_blk64": (
                    lambda: _interface._bsa_attn_fwd_sm120_fp8_blk64(*fp8_64, indices64, topk64, softmax_scale)[0],
                    _attention_flops(indices64, seqlen, 64, head_dim),
                ),
            }
            graphs, outputs = [], {}
            for name, (fn, _) in kernels.items():
                graph, outputs[name] = _capture(fn, args.warmup)
                graphs.append(graph)
            samples = _rotating_times(graphs, args.repeats)
            case = {
                "shape_bshd": [batch, seqlen, heads, head_dim],
                "sparsity": sparsity,
                "topk64": topk64,
                "effective_density64": topk64 / nk64,
                "quantize_fp8_sage_qkv_ms": quantizer_ms[0],
                "quantize_nvfp4_qk_sage_v_ms": quantizer_ms[1],
            }
            for (name, (_, flops)), times in zip(kernels.items(), samples):
                median = statistics.median(times)
                case[name] = {"ms": median, "tflops": flops / median / 1e9, "min_ms": min(times), "max_ms": max(times)}
            case["nvfp4_vs_fp8_blk64_speedup"] = case["fp8_blk64"]["ms"] / case["nvfp4_blk64"]["ms"]
            # Same mask, so NVFP4 and FP8 blk64 should agree to quantization error.
            case["nvfp4_vs_fp8_blk64_relative_l2"] = _relative_l2(outputs["nvfp4_blk64"], outputs["fp8_blk64"])
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
            del graphs, outputs, graph
        del q, k, v, fp8_64, q4, k4, v8
        torch.cuda.empty_cache()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
