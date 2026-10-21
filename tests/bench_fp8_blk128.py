"""FP8/BF16/FP16 blk128 benchmark; one JSON per case.

Run from a source checkout or an installed package environment:
    python tests/bench_fp8_blk128.py --suite all --fp16
    python tests/bench_fp8_blk128.py --suite all --fp16 --mode graph

The default event mode interleaves public API calls. Graph mode captures
the same calls, then times sustained runs after warming each path.

Large cases reuse the blk64 comparison's sequence lengths and head counts.
The block map is rebuilt for blk128, and top-k is rounded down to an even
number for the existing half-precision blk128 uniform-count contract.
"""

import argparse
import datetime
import gc
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import cutlass

# Support direct script execution from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from block_sparse_attention import bsa_attn_fwd, bsa_fp8_fwd, quantize_sage_bhsd
from block_sparse_attention import bsa_fp8_quant
from block_sparse_attention.csrc.fwd.sm100_blk128 import bsa_fwd_sm100_fp8
from block_sparse_attention.csrc.fwd.sm100_blk128 import bsa_fwd_sm100_fp8_fused_pv

_REFERENCE_KERNEL_MODULE = bsa_fwd_sm100_fp8

LARGE_CASES = [
    (1, h, sq, sk, 2 * max(1, int(math.ceil(sk / 128) * 0.1) // 2))
    for sq, sk in [
        (119040, 120320),
        (234240, 235712),
        (349440, 351168),
        (464640, 466560),
        (579840, 581952),
        (695040, 697408),
    ]
    for h in (4, 8)
]
SMALL_CASES = [
    (4 if sk <= 2048 else 2, 6, sq, sk, min(8, sk // 128))
    for sq, sk in [
        (64, 512),
        (64, 1024),
        (128, 512),
        (128, 1024),
        (256, 512),
        (256, 1024),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]
]


def indices(batch, heads, sq, sk, topk):
    nq, nk = math.ceil(sq / 128), math.ceil(sk / 128)
    step = max(1, nk // 2 - 1)
    while math.gcd(step, nk) != 1:
        step -= 1
    rows = torch.arange(batch * heads * nq, device="cuda").reshape(batch, heads, nq, 1)
    return ((rows + torch.arange(topk, device="cuda") * step) % nk).int()


def measure(fn):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    out = fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b), out


def errors(out, ref):
    # Reduce in chunks to avoid allocating several full FP32 outputs at once.
    max_abs, abs_sum, ref_sum = 0.0, 0.0, 0.0
    for start in range(0, out.shape[2], 4096):
        r = ref[:, :, start : start + 4096].float()
        e = (out[:, :, start : start + 4096].float() - r).abs()
        max_abs = max(max_abs, e.max().item())
        abs_sum += e.sum().item()
        ref_sum += r.abs().sum().item()
    return {"abs_max": max_abs, "relative_mean": abs_sum / max(ref_sum, 1e-8)}


def case(shape, a):
    batch, heads, sq, sk, topk = shape
    seed = 20261008 + heads + topk + getattr(a, "seed_offset", 0)
    torch.manual_seed(seed)
    q = torch.randn(batch, heads, sq, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, heads, sk, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    index = indices(*shape)
    # Explicit extents also protect the non-128-aligned customer KV tails in
    # the existing half-precision implementation.
    sizes = (sk - torch.arange(math.ceil(sk / 128), device="cuda") * 128).clamp(0, 128).int()
    quantized = quantize_sage_bhsd(q, k, v)

    def half(q, k, v):
        return bsa_attn_fwd(q, k, v, index, topk, sizes, sparse_block_size=128)[0]

    def fp8(args=quantized):
        return bsa_fp8_fwd(*args, index, topk, sparse_block_size=128, block_sizes=sizes)

    funcs = {
        "bf16": lambda: half(q, k, v),
        "fp8": fp8,
        "quantize": lambda: quantize_sage_bhsd(q, k, v),
        "fp8_e2e": lambda: fp8(quantize_sage_bhsd(q, k, v)),
    }
    if a.fp16:
        q16, k16, v16 = (x.half() for x in (q, k, v))
        funcs["fp16"] = lambda: half(q16, k16, v16)
    # Materialize every compiled path before warmup or CUDA-event timing.
    for fn in funcs.values():
        fn()
        torch.cuda.synchronize()
    for _ in range(a.warmup):
        for fn in funcs.values():
            fn()
    torch.cuda.synchronize()
    graphs = {}
    graph_batch = 10 if max(sq, sk) <= 4096 else 1
    if a.mode == "graph":
        for name, fn in funcs.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(graph_batch):
                    captured = fn()
            graphs[name] = (graph, captured)
    rounds = []
    for rnd in range(a.rounds):
        samples = {name: [] for name in funcs}
        names = list(funcs)
        if a.mode == "event":
            for i in range(a.runs):
                order = names if (i + rnd) % 2 == 0 else names[::-1]
                for name in order:
                    t, _ = measure(funcs[name])
                    samples[name].append(t)
        else:
            # Measure sustained runs after warming each path individually.
            # Reverse round order to balance thermal and frequency drift.
            for name in names if rnd % 2 == 0 else names[::-1]:
                graph, _ = graphs[name]
                for _ in range(max(a.warmup, 10)):
                    graph.replay()
                events = [
                    (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    for _ in range(a.runs)
                ]
                for start, end in events:
                    start.record()
                    graph.replay()
                    end.record()
                events[-1][1].synchronize()
                samples[name] = [start.elapsed_time(end) / graph_batch for start, end in events]
        rounds.append(
            {
                "median_ms": {n: statistics.median(t) for n, t in samples.items()},
                "samples_ms": samples,
            }
        )
    out = fp8()
    if a.mode == "graph":
        for name in ("fp8", "fp8_e2e"):
            assert torch.equal(graphs[name][1], out), f"{name}: graph/eager mismatch"
    out_half = half(q, k, v)
    if a.mode == "graph":
        assert torch.equal(graphs["bf16"][1], out_half), "bf16: graph/eager mismatch"
        if a.fp16:
            assert torch.equal(graphs["fp16"][1], half(q16, k16, v16)), "fp16: graph/eager mismatch"
    total_error = errors(out, out_half)
    del out_half
    q8, k8, v8, qs, ks, vs = quantized
    qd = (q8.float() * qs[..., None]).bfloat16()
    kd = (k8.float() * ks.repeat_interleave(16, dim=-1)[..., :sk, None]).bfloat16()
    vd = (v8.float() * vs[None, :, None, :]).bfloat16()
    ref = half(qd, kd, vd)
    kernel_error = errors(out, ref)
    assert torch.isfinite(out).all(), "nonfinite output"
    assert kernel_error["relative_mean"] < 0.029 and kernel_error["abs_max"] < 0.15, kernel_error
    assert total_error["relative_mean"] < 0.08 and total_error["abs_max"] < 0.3, total_error
    repeat = fp8()
    assert torch.equal(out, repeat), "nondeterministic output"
    median = {n: statistics.mean(r["median_ms"][n] for r in rounds) for n in funcs}
    # Record the public implementation selected at compile time. Private
    # comparison harnesses can still supply their explicit module identity.
    kernel_module = bsa_fwd_sm100_fp8
    capability = torch.cuda.get_device_capability()
    if kernel_module is _REFERENCE_KERNEL_MODULE and capability in ((10, 0), (10, 3)):
        kernel_module = bsa_fwd_sm100_fp8_fused_pv
    result = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(capability),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass": cutlass.__version__,
        "kernel_sha256": hashlib.sha256(Path(kernel_module.__file__).read_bytes()).hexdigest(),
        "kernel_reference_sha256": hashlib.sha256(Path(_REFERENCE_KERNEL_MODULE.__file__).read_bytes()).hexdigest(),
        "quantizer_sha256": hashlib.sha256(Path(bsa_fp8_quant.__file__).read_bytes()).hexdigest(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "shape": list(shape),
        "seed": seed,
        "warmup": a.warmup,
        "mode": a.mode,
        "graph_batch": graph_batch if a.mode == "graph" else None,
        "graph_warmup": max(a.warmup, 10) if a.mode == "graph" else None,
        "runs": a.runs,
        "rounds": rounds,
        "median_ms": median,
        "speedup_kernel": median["bf16"] / median["fp8"],
        "speedup_e2e": median["bf16"] / median["fp8_e2e"],
        "kernel_error": kernel_error,
        "with_quantization_error": total_error,
    }
    print(json.dumps(result, sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["small", "large", "all"], default="all")
    p.add_argument("--case", type=int, action="append")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument(
        "--mode", choices=["event", "graph"], default="event",
        help="interleaved API timing or sustained CUDA Graph timing",
    )
    a = p.parse_args()
    assert a.runs % 2 == 0, "use an even run count to balance launch order"
    cases = (
        SMALL_CASES
        if a.suite == "small"
        else LARGE_CASES if a.suite == "large" else SMALL_CASES + LARGE_CASES
    )
    if a.case is not None:
        cases = [cases[i] for i in a.case]
    for c in cases:
        case(c, a)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
