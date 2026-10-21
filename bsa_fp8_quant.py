"""Sage FP8 quantization that keeps tensors in contiguous BHSD storage."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _kv_partial_stats(
    mK, mV, mKSum, mVMax,
    SEQ: tl.constexpr, HEADS: tl.constexpr, SPLITS: tl.constexpr,
    ROWS: tl.constexpr, COLS: tl.constexpr, FINAL_STATS: tl.constexpr = False,
):
    split = tl.program_id(0)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    bh = tl.program_id(2)
    rows = tl.arange(0, ROWS)
    # Every split owns at least one tile. Initialize from that tile directly;
    # short reductions need neither zero accumulators nor a runtime loop.
    seq = split * ROWS + rows
    offset = (bh.to(tl.int64) * SEQ + seq[:, None]) * 128 + cols[None, :]
    tKSum = tl.load(mK + offset, seq[:, None] < SEQ, 0).to(tl.float32)
    tVMax = tl.abs(tl.load(mV + offset, seq[:, None] < SEQ, 0).to(tl.float32))
    if tl.cdiv(SEQ, ROWS) > SPLITS:
        for tile in range(split + SPLITS, tl.cdiv(SEQ, ROWS), SPLITS):
            seq = tile * ROWS + rows
            offset = (bh.to(tl.int64) * SEQ + seq[:, None]) * 128 + cols[None, :]
            tK = tl.load(mK + offset, seq[:, None] < SEQ, 0).to(tl.float32)
            tV = tl.load(mV + offset, seq[:, None] < SEQ, 0).to(tl.float32)
            tKSum += tK
            tVMax = tl.maximum(tVMax, tl.abs(tV))
    offset = (bh * SPLITS + split) * 128 + cols
    tKStat = tl.sum(tKSum, axis=0)
    tVStat = tl.max(tVMax, axis=0)
    if FINAL_STATS:
        tKStat = tKStat / SEQ
        tVStat = tl.maximum(tVStat, 1e-3) / 448.0
    tl.store(mKSum + offset, tKStat)
    tl.store(mVMax + offset, tVStat)


@triton.jit
def _kv_finish_stats(
    mKSum, mVMax, mKMean, mVScale,
    SEQ: tl.constexpr, BATCH: tl.constexpr, HEADS: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_BATCH: tl.constexpr, BLOCK_SPLITS: tl.constexpr,
    COLS: tl.constexpr,
):
    head = tl.program_id(0)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    batch = tl.arange(0, BLOCK_BATCH)
    split = tl.arange(0, BLOCK_SPLITS)
    offset = (
        ((batch[:, None, None] * HEADS + head) * SPLITS + split[None, :, None]) * 128
        + cols[None, None, :]
    )
    valid = (batch[:, None, None] < BATCH) & (split[None, :, None] < SPLITS)
    tKSum = tl.load(mKSum + offset, valid, 0)
    tVMax = tl.load(mVMax + offset, valid, 0)
    tKMean = tl.sum(tKSum, axis=1) / SEQ
    tVScale = tl.maximum(tl.max(tl.max(tVMax, axis=1), axis=0), 1e-3) / 448.0
    tl.store(mKMean + (batch[:, None] * HEADS + head) * 128 + cols[None, :],
             tKMean, batch[:, None] < BATCH)
    tl.store(mVScale + head * 128 + cols, tVScale)


@triton.jit
def _quantize_qkv(
    mQ, mK, mV, mQ8, mK8, mV8, mQScale, mKScale, mVScale, mKMean,
    SQ: tl.constexpr, SK: tl.constexpr, HEADS: tl.constexpr, ROWS: tl.constexpr,
):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    seq = tile * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, 128)
    if tile * ROWS < SQ:
        offset = (bh.to(tl.int64) * SQ + seq[:, None]) * 128 + cols[None, :]
        tQ = tl.load(mQ + offset, seq[:, None] < SQ, 0)
        tScale = tl.maximum(tl.max(tl.abs(tQ.to(tl.float32)), axis=1), 1e-3) / 448.0
        # Preserve the customer's BF16 rounding of Q scale, reciprocal and product.
        tReciprocal = (1.0 / tScale.to(tl.bfloat16)).to(tl.bfloat16)
        tQScaled = (tQ * tReciprocal[:, None]).to(tl.bfloat16)
        tl.store(mQ8 + offset, tQScaled.to(mQ8.dtype.element_ty), seq[:, None] < SQ)
        tl.store(mQScale + bh * SQ + seq, tScale, seq < SQ)
    if tile * ROWS < SK:
        offset = (bh.to(tl.int64) * SK + seq[:, None]) * 128 + cols[None, :]
        tKMean = tl.load(mKMean + bh * 128 + cols)
        tK = tl.load(mK + offset, seq[:, None] < SK, 0).to(tl.float32) - tKMean[None, :]
        # Keep zero-padded tail lanes in the K scale reduction, as in the original recipe.
        tKGroups = tl.reshape(tK, (ROWS // 16, 16 * 128))
        tKScale = tl.maximum(tl.max(tl.abs(tKGroups), axis=1), 1e-3) / 448.0
        tKScaled = tl.reshape(tKGroups / tKScale[:, None], (ROWS, 128))
        tl.store(mK8 + offset, tKScaled.to(mK8.dtype.element_ty), seq[:, None] < SK)
        group = tile * (ROWS // 16) + tl.arange(0, ROWS // 16)
        tl.store(mKScale + bh * tl.cdiv(SK, 16) + group, tKScale, group < tl.cdiv(SK, 16))

        tVScale = tl.load(mVScale + (bh % HEADS) * 128 + cols)
        tV = tl.load(mV + offset, seq[:, None] < SK, 0)
        # The original compiled V recipe keeps reciprocal and multiply in FP32.
        tVReciprocal = 1.0 / tVScale
        tVScaled = tV.to(tl.float32) * tVReciprocal[None, :]
        tl.store(mV8 + offset, tVScaled.to(mV8.dtype.element_ty), seq[:, None] < SK)


_native_quant_enabled = os.environ.get("BSA_FP8_NATIVE_QUANT", "1") != "0"
_native_quant_plans = {}
_triton_runtime = getattr(getattr(triton, "knobs", None), "runtime", None)
_triton_hook_chain = getattr(getattr(triton, "knobs", None), "HookChain", ())


def _native_hooks_clear():
    if _triton_runtime is None:
        return False
    enter = _triton_runtime.launch_enter_hook
    leave = _triton_runtime.launch_exit_hook
    return ((enter is None or (type(enter) is _triton_hook_chain and not enter.calls))
            and (leave is None or (type(leave) is _triton_hook_chain and not leave.calls)))


def quantize_sage_bhsd(
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
):
    """Quantize BF16 BHSD Q/K/V for :func:`bsa_fp8_fwd` (blk64 or blk128).

    Q scales are per token, smoothed K scales per 16 tokens, and V scales
    per head/channel across batch and sequence. Reductions and conversion
    use three Triton kernels without layout copies or torch.compile. A single
    batch with one statistics tile needs only two launches, since no partial
    results or cross-batch maxima remain to combine. When a host C++ compiler
    is available, warmed calls allocate and launch through one native host
    function; the GPU kernels and rounding stay unchanged. Set
    ``BSA_FP8_NATIVE_QUANT=0`` before import to use the regular Triton launcher.
    """
    key = None
    if (_native_quant_enabled and _native_hooks_clear()
            and all(type(t) is torch.Tensor for t in (q_bhsd, k_bhsd, v_bhsd))):
        # Triton specializes input pointers by 16-byte alignment. Include that
        # property, not addresses or tensor values, in the host plan identity.
        aligned = tuple(t.data_ptr() % 16 == 0 for t in (q_bhsd, k_bhsd, v_bhsd))
        key = (q_bhsd.device, q_bhsd.shape, k_bhsd.shape, aligned)
        plan = _native_quant_plans.get(key)
        if plan is not None:
            return plan[0](q_bhsd, k_bhsd, v_bhsd)
    return _quantize_sage_bhsd_triton(q_bhsd, k_bhsd, v_bhsd, _plan_key=key)


def _quantize_sage_bhsd_triton(q_bhsd, k_bhsd, v_bhsd, *, _plan_key=None):
    tensors = (q_bhsd, k_bhsd, v_bhsd)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("Q, K, and V must be rank-4 BHSD tensors")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise TypeError("Sage FP8 quantization currently requires BF16 inputs")
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Q, K, and V must be CUDA tensors")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Q, K, and V must use contiguous BHSD storage")
    batch, heads, seqlen_q, head_dim = q_bhsd.shape
    if batch < 1 or heads < 1:
        raise ValueError("Sage FP8 quantization requires positive batch and head counts")
    if head_dim != 128:
        raise ValueError("Sage FP8 quantization requires D=128")
    if k_bhsd.shape[:2] != (batch, heads) or k_bhsd.shape[-1] != head_dim:
        raise ValueError("K must match Q batch, heads, and head dimension")
    if v_bhsd.shape != k_bhsd.shape:
        raise ValueError("V must have the same shape as K")
    seqlen_k = k_bhsd.shape[2]
    if seqlen_q < 1 or seqlen_k < 1:
        raise ValueError("Sage FP8 quantization requires positive sequence lengths")
    if k_bhsd.device != q_bhsd.device or v_bhsd.device != q_bhsd.device:
        raise ValueError("Q, K, and V must be on the same CUDA device")

    device = q_bhsd.device
    q_fp8, k_fp8, v_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn) for x in tensors)
    q_scale = torch.empty((batch, heads, seqlen_q), device=device, dtype=torch.float32)
    k_scale = torch.empty((batch, heads, triton.cdiv(seqlen_k, 16)), device=device, dtype=torch.float32)
    v_scale = torch.empty((heads, 128), device=device, dtype=torch.float32)
    k_mean = torch.empty((batch, heads, 128), device=device, dtype=torch.float32)
    splits = min(triton.cdiv(seqlen_k, 128), 128)
    with torch.cuda.device(device):
        kernel_finish = None
        if batch == 1 and splits == 1:
            kernel_stats = _kv_partial_stats[(1, 4, heads)](
                k_bhsd, v_bhsd, k_mean, v_scale,
                seqlen_k, heads, 1, 128, 32, FINAL_STATS=True, num_warps=4,
            )
        else:
            partials = torch.empty((2, batch, heads, splits, 128), device=device, dtype=torch.float32)
            k_sum, v_max = partials.unbind(0)
            kernel_stats = _kv_partial_stats[(splits, 4, batch * heads)](
                k_bhsd, v_bhsd, k_sum, v_max,
                seqlen_k, heads, splits, 128, 32, num_warps=4,
            )
            cols = max(1, 32 // triton.next_power_of_2(batch))
            kernel_finish = _kv_finish_stats[(heads, 128 // cols)](
                k_sum, v_max, k_mean, v_scale, seqlen_k, batch, heads, splits,
                triton.next_power_of_2(batch), triton.next_power_of_2(splits), cols, num_warps=4,
            )
        rows = 32
        kernel_quant = _quantize_qkv[(triton.cdiv(max(seqlen_q, seqlen_k), rows), batch * heads)](
            q_bhsd, k_bhsd, v_bhsd, q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, k_mean,
            seqlen_q, seqlen_k, heads, rows, num_warps=4,
        )
    if _plan_key is not None and _plan_key not in _native_quant_plans:
        from block_sparse_attention.csrc.fp8_quant_runtime import make_quant_plan

        plan = make_quant_plan(
            (kernel_stats, kernel_finish, kernel_quant),
            (batch, heads, seqlen_q, seqlen_k), device.index,
        )
        # Bound retained kernel handles without retaining any user tensors.
        if len(_native_quant_plans) >= 128:
            del _native_quant_plans[next(iter(_native_quant_plans))]
        _native_quant_plans[_plan_key] = plan
    return q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale


__all__ = ["quantize_sage_bhsd"]
