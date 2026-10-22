# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVFP4 quantization of BHSD Q/K for the SM120 NVFP4-QK BSA kernel.

``quantize_nvfp4_qk_sage_v_bhsd`` is the fused Q/K/V entry point of the NVFP4 path: NVFP4 Q/K
plus this repo's Sage FP8 V, in the same three Triton launches as ``quantize_sage_bhsd``.
``quantize_nvfp4_bhsd`` quantizes one tensor with CuTe DSL kernels.

Each row is split into 16-element head-dim groups that share one E4M3 scale
(``amax / 6``); values round to nearest-even E2M1 and two codes pack into one
byte, low nibble first. Scales are stored row-major per token and zero-padded
to whole 64-token tiles so the kernel can TMA one 512-byte tile per block.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass import BFloat16, Float8E4M3FN, Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

from cutlass.utils import SmemAllocator

from block_sparse_attention.csrc.utils.cute_dsl_utils import to_cute_tensor

try:
    from .bsa_fp8_quant import _kv_finish_stats, _kv_partial_stats
except ImportError:
    from bsa_fp8_quant import _kv_finish_stats, _kv_partial_stats

NVFP4_SF_VEC_SIZE = 16
NVFP4_TILE_ROWS = 64
_E2M1_MAX = 6.0
# Midpoints between adjacent E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values (already clamped to [-6, 6]) to 4-bit E2M1 codes."""
    midpoints = torch.tensor(_E2M1_MIDPOINTS, device=x.device, dtype=torch.float32)
    magnitude = x.abs()
    down = torch.bucketize(magnitude, midpoints, right=False)
    up = torch.bucketize(magnitude, midpoints, right=True)
    # A tie sits between codes down and up = down + 1; keep the even code.
    code = torch.where(down == up, down, torch.where(down % 2 == 0, down, up))
    sign = (x < 0) & (code != 0)
    return (code | (sign.to(code.dtype) << 3)).to(torch.uint8)


def nvfp4_dequantize_reference(packed: torch.Tensor, scale: torch.Tensor, seqlen: int) -> torch.Tensor:
    """Expand packed E2M1 bytes and padded E4M3 scales back to fp32 BHSD."""
    values = torch.tensor(_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    decoded = values[codes & 0x7] * torch.where(codes >= 8, -1.0, 1.0)
    head_dim = decoded.shape[-1]
    scales = scale[..., :seqlen, :].float().repeat_interleave(NVFP4_SF_VEC_SIZE, dim=-1)
    return decoded[..., :seqlen, :head_dim] * scales


def quantize_nvfp4_bhsd_reference(x: torch.Tensor, *, smooth: bool = False):
    """PyTorch reference for ``quantize_nvfp4_bhsd``; defines the bit-level contract.

    ``smooth=True`` subtracts the per-(batch, head) token mean first (Sage K
    smoothing); it shifts every score of a query row by the same constant,
    so softmax is unchanged.

    Returns ``(packed, scale)``: ``packed`` is uint8 ``[B, H, S, D/2]`` and
    ``scale`` is float8_e4m3fn ``[B, H, ceil(S/64)*64, D/16]``.
    """
    if x.ndim != 4:
        raise ValueError("NVFP4 quantization expects a rank-4 BHSD tensor")
    batch, heads, seqlen, head_dim = x.shape
    if head_dim % NVFP4_SF_VEC_SIZE:
        raise ValueError(f"head_dim must be a multiple of {NVFP4_SF_VEC_SIZE}")
    xf = x.float()
    if smooth:
        xf = xf - xf.mean(dim=-2, keepdim=True)
    groups = xf.view(batch, heads, seqlen, head_dim // NVFP4_SF_VEC_SIZE, NVFP4_SF_VEC_SIZE)
    scale = (groups.abs().amax(dim=-1) / _E2M1_MAX).clamp(max=448.0).to(torch.float8_e4m3fn)
    scale_f = scale.float()
    scaled = groups / torch.where(scale_f > 0, scale_f, 1.0)[..., None]
    codes = _round_to_e2m1_codes(scaled.clamp(-_E2M1_MAX, _E2M1_MAX)).view(batch, heads, seqlen, head_dim)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()

    padded_seqlen = (seqlen + NVFP4_TILE_ROWS - 1) // NVFP4_TILE_ROWS * NVFP4_TILE_ROWS
    padded_scale = torch.zeros(
        (batch, heads, padded_seqlen, head_dim // NVFP4_SF_VEC_SIZE),
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    padded_scale[:, :, :seqlen] = scale
    return packed, padded_scale


def _to_dynamic_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return to_cute_tensor(
        tensor.view(-1),
        assumed_align=16,
        fully_dynamic=True,
        enable_tvm_ffi=False,
    )


@cute.jit
def _e2m1x8(values) -> Uint32:
    """Pack 8 fp32 values (already divided by the group scale) into one word, element i in nibble i.

    ``cvt.rn.satfinite.e2m1x2.f32 d, a, b`` rounds to nearest-even, saturates to +/-6 and puts
    ``a`` in the upper nibble. Codes whose magnitude rounds to zero drop their sign so the bytes
    match ``quantize_nvfp4_bhsd_reference`` (which never emits -0).
    """
    word = Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value() for value in values],
            "{ .reg .b8 b0, b1, b2, b3;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;\n"
            "mov.b32 $0, {b0, b1, b2, b3}; }",
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    # Bit 3 of (magnitude + 7) is set exactly when the 3-bit magnitude is nonzero; no nibble carries.
    nonzero = ((word & Uint32(0x77777777)) + Uint32(0x77777777)) & Uint32(0x88888888)
    return word & (nonzero | Uint32(0x77777777))


class _Nvfp4QuantizerSm120:
    """Quantize contiguous BF16 BHSD rows (D=128) to packed NVFP4 plus tile-padded E4M3 scales.

    One thread owns one 16-element scale group: a 32-byte vector load, an E4M3 scale, and an
    8-byte packed store. A 512-thread CTA covers one 64-token scale tile and also zero-fills the
    tile's padded scale rows. With ``smooth`` the per-(batch, head) token mean is subtracted first;
    it is reduced by a partial-sum kernel plus a tiny finalize kernel, so K is read twice.
    """

    HEAD_DIM = 128
    GROUPS_PER_ROW = HEAD_DIM // NVFP4_SF_VEC_SIZE
    THREADS = NVFP4_TILE_ROWS * GROUPS_PER_ROW
    MEAN_VEC = 8
    MEAN_COL_LANES = HEAD_DIM // MEAN_VEC
    MEAN_THREADS = 256
    MEAN_ROW_LANES = MEAN_THREADS // MEAN_COL_LANES
    MEAN_ROWS_PER_CHUNK = 128
    FINALIZE_DIMS = 32
    FINALIZE_LANES = 32

    def __init__(self, smooth: bool):
        self.smooth = smooth

    def _rows_view(self, flat: cute.Tensor, num_rows, width):
        # Static row stride, so per-thread slices keep 16-byte alignment and vectorize.
        return cute.make_tensor(flat.iterator, cute.make_layout((num_rows, width), stride=(width, 1)))

    @cute.kernel
    def _mean_partial_kernel(self, mX: cute.Tensor, mPartial: cute.Tensor, seqlen: Int32, num_chunks: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, group_idx, _ = cute.arch.block_idx()
        col_lane = tidx % self.MEAN_COL_LANES
        row_lane = tidx // self.MEAN_COL_LANES
        sums = SmemAllocator().allocate_tensor(Float32, cute.make_layout((self.MEAN_ROW_LANES, self.HEAD_DIM), stride=(self.HEAD_DIM, 1)), byte_alignment=16)
        acc = cute.make_rmem_tensor(cute.make_layout((self.MEAN_VEC,)), Float32)
        acc.fill(0.0)
        values = cute.make_rmem_tensor(cute.make_layout((1, self.MEAN_VEC)), BFloat16)
        for step in cutlass.range_constexpr(self.MEAN_ROWS_PER_CHUNK // self.MEAN_ROW_LANES):
            seq_idx = chunk_idx * self.MEAN_ROWS_PER_CHUNK + step * self.MEAN_ROW_LANES + row_lane
            if seq_idx < seqlen:
                cute.autovec_copy(cute.local_tile(mX, (1, self.MEAN_VEC), (group_idx * seqlen + seq_idx, col_lane)), values)
                for elem in cutlass.range_constexpr(self.MEAN_VEC):
                    acc[elem] += values[0, elem].to(Float32)
        for elem in cutlass.range_constexpr(self.MEAN_VEC):
            sums[row_lane, col_lane * self.MEAN_VEC + elem] = acc[elem]
        cute.arch.barrier()
        if tidx < self.HEAD_DIM:
            total = Float32(0.0)
            for lane in cutlass.range_constexpr(self.MEAN_ROW_LANES):
                total += sums[lane, tidx]
            mPartial[(group_idx * num_chunks + chunk_idx) * self.HEAD_DIM + tidx] = total

    @cute.kernel
    def _mean_finalize_kernel(self, mPartial: cute.Tensor, mMean: cute.Tensor, num_chunks: Int32, seqlen_float: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        group_idx, dim_block, _ = cute.arch.block_idx()
        dim_idx = dim_block * self.FINALIZE_DIMS + tidx % self.FINALIZE_DIMS
        lane = tidx // self.FINALIZE_DIMS
        sums = SmemAllocator().allocate_tensor(
            Float32, cute.make_layout((self.FINALIZE_LANES, self.FINALIZE_DIMS), stride=(self.FINALIZE_DIMS, 1)), byte_alignment=16
        )
        # Many lanes stride over the chunks so the loads are independent and pipeline; a
        # one-thread-per-channel serial loop was latency-bound (~67 us for 400 chunks).
        # The fixed-order shared-memory reduction keeps the mean deterministic (no atomics).
        total = Float32(0.0)
        for chunk_idx in cutlass.range(lane, num_chunks, self.FINALIZE_LANES, unroll=4):
            total += mPartial[(group_idx * num_chunks + chunk_idx) * self.HEAD_DIM + dim_idx]
        sums[lane, tidx % self.FINALIZE_DIMS] = total
        cute.arch.barrier()
        if tidx < self.FINALIZE_DIMS:
            total = Float32(0.0)
            for lane_idx in cutlass.range_constexpr(self.FINALIZE_LANES):
                total += sums[lane_idx, tidx]
            mMean[group_idx * self.HEAD_DIM + dim_idx] = total / seqlen_float

    @cute.kernel
    def _quantize_kernel(
        self,
        mX: cute.Tensor,
        mPacked: cute.Tensor,
        mScale: cute.Tensor,
        mMean: cute.Tensor,
        seqlen: Int32,
        padded_seqlen: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tile_idx, group_idx, _ = cute.arch.block_idx()
        col = tidx % self.GROUPS_PER_ROW
        seq_idx = tile_idx * NVFP4_TILE_ROWS + tidx // self.GROUPS_PER_ROW
        scale_e4m3 = Float32(0.0).to(Float8E4M3FN)
        if seq_idx < seqlen:
            row = group_idx * seqlen + seq_idx
            raw = cute.make_rmem_tensor(cute.make_layout((1, NVFP4_SF_VEC_SIZE)), BFloat16)
            cute.autovec_copy(cute.local_tile(mX, (1, NVFP4_SF_VEC_SIZE), (row, col)), raw)
            values = cute.make_rmem_tensor(cute.make_layout((NVFP4_SF_VEC_SIZE,)), Float32)
            if cutlass.const_expr(self.smooth):
                mean = cute.make_rmem_tensor(cute.make_layout((1, NVFP4_SF_VEC_SIZE)), Float32)
                cute.autovec_copy(cute.local_tile(mMean, (1, NVFP4_SF_VEC_SIZE), (group_idx, col)), mean)
                for elem in cutlass.range_constexpr(NVFP4_SF_VEC_SIZE):
                    values[elem] = raw[0, elem].to(Float32) - mean[0, elem]
            else:
                for elem in cutlass.range_constexpr(NVFP4_SF_VEC_SIZE):
                    values[elem] = raw[0, elem].to(Float32)
            amax = Float32(0.0)
            for elem in cutlass.range_constexpr(NVFP4_SF_VEC_SIZE):
                amax = cute.arch.fmax(amax, cute.arch.fmax(values[elem], -values[elem]))
            scale_e4m3 = cute.arch.fmin(amax / Float32(_E2M1_MAX), Float32(448.0)).to(Float8E4M3FN)
            scale = scale_e4m3.to(Float32)
            divisor = Float32(1.0)
            if scale > Float32(0.0):
                divisor = scale
            # IEEE division (div.rn), as in the reference: a reciprocal multiply or the approximate
            # div.full lands a hair above E2M1 midpoints such as 5.0 and flips the tie to the odd code.
            for elem in cutlass.range_constexpr(NVFP4_SF_VEC_SIZE):
                values[elem] = values[elem] / divisor
            packed = cute.make_rmem_tensor(cute.make_layout((1, 2)), Uint32)
            packed[0, 0] = _e2m1x8([values[elem] for elem in range(8)])
            packed[0, 1] = _e2m1x8([values[elem] for elem in range(8, 16)])
            cute.autovec_copy(packed, cute.local_tile(mPacked, (1, 2), (row, col)))
        # Rows past seqlen in the last tile store a zero scale: the padding contract.
        mScale[group_idx * padded_seqlen + seq_idx, col] = scale_e4m3

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mPacked: cute.Tensor,
        mScale: cute.Tensor,
        mPartial: cute.Tensor,
        mMean: cute.Tensor,
        num_groups: Int32,
        seqlen: Int32,
        num_chunks: Int32,
        seqlen_float: Float32,
        stream: cuda.CUstream,
    ):
        num_tiles = (seqlen + NVFP4_TILE_ROWS - 1) // NVFP4_TILE_ROWS
        padded_seqlen = num_tiles * NVFP4_TILE_ROWS
        mX = self._rows_view(mX, num_groups * seqlen, self.HEAD_DIM)
        mPacked = self._rows_view(cute.recast_tensor(mPacked, Uint32), num_groups * seqlen, self.HEAD_DIM // 8)
        mScale = self._rows_view(mScale, num_groups * padded_seqlen, self.GROUPS_PER_ROW)
        mMeanRows = self._rows_view(mMean, num_groups, self.HEAD_DIM)
        if cutlass.const_expr(self.smooth):
            self._mean_partial_kernel(mX, mPartial, seqlen, num_chunks).launch(
                grid=(num_chunks, num_groups, 1),
                block=(self.MEAN_THREADS, 1, 1),
                smem=self.MEAN_ROW_LANES * self.HEAD_DIM * 4,
                stream=stream,
            )
            self._mean_finalize_kernel(mPartial, mMean, num_chunks, seqlen_float).launch(
                grid=(num_groups, self.HEAD_DIM // self.FINALIZE_DIMS, 1),
                block=(self.FINALIZE_LANES * self.FINALIZE_DIMS, 1, 1),
                smem=self.FINALIZE_LANES * self.FINALIZE_DIMS * 4,
                stream=stream,
            )
        self._quantize_kernel(mX, mPacked, mScale, mMeanRows, seqlen, padded_seqlen).launch(
            grid=(num_tiles, num_groups, 1),
            block=(self.THREADS, 1, 1),
            stream=stream,
        )


def quantize_nvfp4_bhsd(x: torch.Tensor, *, smooth: bool = False):
    """Quantize a contiguous BF16 BHSD tensor (D=128) to NVFP4 on SM120 with fused CuTe DSL kernels.

    Bit-identical to ``quantize_nvfp4_bhsd_reference`` for ``smooth=False``; with ``smooth=True``
    the mean is summed in a different order, so a code may differ where the centered value sits
    on an E2M1 rounding boundary. Returns ``(packed, scale)`` with the reference's shapes.
    """
    if x.ndim != 4:
        raise ValueError("NVFP4 quantization expects a rank-4 BHSD tensor")
    batch, heads, seqlen, head_dim = x.shape
    if head_dim != _Nvfp4QuantizerSm120.HEAD_DIM:
        raise ValueError(f"the NVFP4 quantization kernel requires head_dim == {_Nvfp4QuantizerSm120.HEAD_DIM}")
    if x.dtype != torch.bfloat16:
        raise TypeError("the NVFP4 quantization kernel requires BF16 input")
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("the NVFP4 quantization kernel requires a contiguous CUDA tensor")
    if batch < 1 or heads < 1 or seqlen < 1:
        raise ValueError("NVFP4 quantization requires positive batch, head, and sequence counts")
    if torch.cuda.get_device_capability(x.device)[0] != 12:
        raise NotImplementedError("the NVFP4 quantization kernel requires SM120")

    num_groups = batch * heads
    padded_seqlen = (seqlen + NVFP4_TILE_ROWS - 1) // NVFP4_TILE_ROWS * NVFP4_TILE_ROWS
    num_chunks = (seqlen + _Nvfp4QuantizerSm120.MEAN_ROWS_PER_CHUNK - 1) // _Nvfp4QuantizerSm120.MEAN_ROWS_PER_CHUNK
    with torch.cuda.device(x.device):
        packed = torch.empty((batch, heads, seqlen, head_dim // 2), dtype=torch.uint8, device=x.device)
        scale = torch.empty((batch, heads, padded_seqlen, head_dim // NVFP4_SF_VEC_SIZE), dtype=torch.float8_e4m3fn, device=x.device)
        mean_partial = torch.empty((num_groups, num_chunks if smooth else 1, head_dim), dtype=torch.float32, device=x.device)
        mean = torch.empty((num_groups, head_dim), dtype=torch.float32, device=x.device)
        cute_tensors = tuple(_to_dynamic_cute_tensor(tensor) for tensor in (x, packed, scale, mean_partial, mean))
        scalars = (Int32(num_groups), Int32(seqlen), Int32(num_chunks), Float32(seqlen))
        compile_key = (torch.cuda.get_device_capability(x.device), smooth)
        if compile_key not in quantize_nvfp4_bhsd.compile_cache:
            quantize_nvfp4_bhsd.compile_cache[compile_key] = cute.compile(
                _Nvfp4QuantizerSm120(smooth),
                *cute_tensors,
                *scalars,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            )
        quantize_nvfp4_bhsd.compile_cache[compile_key](*cute_tensors, *scalars, cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream))
    return packed, scale


quantize_nvfp4_bhsd.compile_cache = {}


@triton.jit
def _e2m1x2(lo, hi):
    """Pack two fp32 values (already divided by the group scale) into one E2M1 byte, ``lo`` in the low nibble.

    ``cvt.rn.satfinite.e2m1x2.f32`` rounds to nearest-even and saturates to +/-6 exactly like
    ``quantize_nvfp4_bhsd_reference``; zero magnitudes then drop their sign, since the
    reference never emits -0. Bit 3 of (magnitude + 7) is set exactly when the magnitude is nonzero.
    """
    byte = tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t; }",
        "=r,f,f",
        [lo, hi],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    nonzero = ((byte & 0x77) + 0x77) & 0x88
    return (byte & (nonzero | 0x77)).to(tl.uint8)


@triton.jit
def _quantize_nvfp4_rows(tX, mPacked, mScale, bh, seq, SEQ: tl.constexpr, SEQ_PAD: tl.constexpr, ROWS: tl.constexpr):
    """NVFP4-quantize a [ROWS, 128] fp32 tile; rows in [SEQ, SEQ_PAD) are zero and store zero scales."""
    tGroups = tl.reshape(tX, (ROWS, 8, 16))
    tScale = tl.minimum(tl.math.div_rn(tl.max(tl.abs(tGroups), axis=2), 6.0), 448.0).to(tl.float8e4nv)
    tScaleF = tScale.to(tl.float32)
    tDivisor = tl.where(tScaleF > 0, tScaleF, 1.0)[:, :, None]
    # x / s rounded to nearest, as the reference's true division, without a per-element div.rn
    # (it dominated the latency of short sequences): with r = RN(1/s), one FMA residual step
    # q0 + (x - q0 * s) * r is correctly rounded (Markstein). Checked against div.rn over every
    # BF16 value x every E4M3 scale and 2e8 random FP32 values: no E2M1 code differs.
    tReciprocal = tl.math.div_rn(1.0, tDivisor)
    tQuotient = tGroups * tReciprocal
    tQuotient = tl.fma(tl.fma(-tQuotient, tDivisor, tGroups), tReciprocal, tQuotient)
    tLo, tHi = tl.split(tl.reshape(tQuotient, (ROWS, 64, 2)))
    tPacked = _e2m1x2(tLo, tHi)
    packed_cols = tl.arange(0, 64)
    tl.store(mPacked + (bh.to(tl.int64) * SEQ + seq[:, None]) * 64 + packed_cols[None, :], tPacked, seq[:, None] < SEQ)
    scale_cols = tl.arange(0, 8)
    tl.store(mScale + (bh.to(tl.int64) * SEQ_PAD + seq[:, None]) * 8 + scale_cols[None, :], tScale, seq[:, None] < SEQ_PAD)


@triton.jit
def _quantize_nvfp4_qk_fp8_v(
    mQ, mK, mV, mQ4, mQSF, mK4, mKSF, mV8, mVScale, mKMean,
    SQ: tl.constexpr, SK: tl.constexpr, SQ_PAD: tl.constexpr, SK_PAD: tl.constexpr, HEADS: tl.constexpr, ROWS: tl.constexpr,
):
    # Q, K and V tiles run in separate CTAs, so short sequences pay the latency of one tensor's
    # tile rather than all three in sequence. The part is the fastest grid axis: a tile's Q/K/V
    # CTAs launch together and K/V are re-read while the statistics pass left them in L2.
    part = tl.program_id(0)
    tile = tl.program_id(1)
    bh = tl.program_id(2)
    seq = tile * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, 128)
    if part == 0:
        if tile * ROWS < SQ_PAD:
            offset = (bh.to(tl.int64) * SQ + seq[:, None]) * 128 + cols[None, :]
            tQ = tl.load(mQ + offset, seq[:, None] < SQ, 0).to(tl.float32)
            _quantize_nvfp4_rows(tQ, mQ4, mQSF, bh, seq, SQ, SQ_PAD, ROWS)
    elif part == 1:
        if tile * ROWS < SK_PAD:
            offset = (bh.to(tl.int64) * SK + seq[:, None]) * 128 + cols[None, :]
            tKMean = tl.load(mKMean + bh * 128 + cols)
            # Smooth K with the same mean the Sage FP8 path uses; padded rows stay zero.
            tK = tl.where(seq[:, None] < SK, tl.load(mK + offset, seq[:, None] < SK, 0).to(tl.float32) - tKMean[None, :], 0.0)
            _quantize_nvfp4_rows(tK, mK4, mKSF, bh, seq, SK, SK_PAD, ROWS)
    elif tile * ROWS < SK:
        offset = (bh.to(tl.int64) * SK + seq[:, None]) * 128 + cols[None, :]
        # Same V expressions as bsa_fp8_quant._quantize_qkv, so V matches quantize_sage_bhsd bit for bit.
        tVScale = tl.load(mVScale + (bh % HEADS) * 128 + cols)
        tV = tl.load(mV + offset, seq[:, None] < SK, 0)
        tVReciprocal = 1.0 / tVScale
        tVScaled = tV.to(tl.float32) * tVReciprocal[None, :]
        tl.store(mV8 + offset, tVScaled.to(mV8.dtype.element_ty), seq[:, None] < SK)


def quantize_nvfp4_qk_sage_v_bhsd(q_bhsd: torch.Tensor, k_bhsd: torch.Tensor, v_bhsd: torch.Tensor):
    """Quantize BF16 BHSD Q/K to NVFP4 and V to Sage FP8 for ``_bsa_attn_fwd_sm120_nvfp4_blk64``.

    Returns ``(q_fp4, q_sf, k_fp4, k_sf, v_fp8, v_scale)``. Q/K follow ``quantize_nvfp4_bhsd_reference``
    (K smoothed by its per-(batch, head) token mean, summed in this repo's Sage order); V and its
    scale are bit-identical to ``quantize_sage_bhsd``. The K-mean/V-amax statistics are the Sage
    path's own kernels, so this runs the same three launches while writing fewer bytes.
    """
    tensors = (q_bhsd, k_bhsd, v_bhsd)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("Q, K, and V must be rank-4 BHSD tensors")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise TypeError("NVFP4 Q/K + Sage FP8 V quantization requires BF16 inputs")
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Q, K, and V must be CUDA tensors")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Q, K, and V must use contiguous BHSD storage")
    batch, heads, seqlen_q, head_dim = q_bhsd.shape
    if batch < 1 or heads < 1:
        raise ValueError("NVFP4 quantization requires positive batch and head counts")
    if head_dim != 128:
        raise ValueError("NVFP4 Q/K + Sage FP8 V quantization requires D=128")
    if k_bhsd.shape[:2] != (batch, heads) or k_bhsd.shape[-1] != head_dim:
        raise ValueError("K must match Q batch, heads, and head dimension")
    if v_bhsd.shape != k_bhsd.shape:
        raise ValueError("V must have the same shape as K")
    seqlen_k = k_bhsd.shape[2]
    if seqlen_q < 1 or seqlen_k < 1:
        raise ValueError("NVFP4 quantization requires positive sequence lengths")
    if k_bhsd.device != q_bhsd.device or v_bhsd.device != q_bhsd.device:
        raise ValueError("Q, K, and V must be on the same CUDA device")

    device = q_bhsd.device
    pad_q = triton.cdiv(seqlen_q, NVFP4_TILE_ROWS) * NVFP4_TILE_ROWS
    pad_k = triton.cdiv(seqlen_k, NVFP4_TILE_ROWS) * NVFP4_TILE_ROWS
    q_fp4 = torch.empty((batch, heads, seqlen_q, 64), device=device, dtype=torch.uint8)
    k_fp4 = torch.empty((batch, heads, seqlen_k, 64), device=device, dtype=torch.uint8)
    q_sf = torch.empty((batch, heads, pad_q, 8), device=device, dtype=torch.float8_e4m3fn)
    k_sf = torch.empty((batch, heads, pad_k, 8), device=device, dtype=torch.float8_e4m3fn)
    v_fp8 = torch.empty_like(v_bhsd, dtype=torch.float8_e4m3fn)
    v_scale = torch.empty((heads, 128), device=device, dtype=torch.float32)
    k_mean = torch.empty((batch, heads, 128), device=device, dtype=torch.float32)
    # K-mean and V-amax statistics: identical launches to bsa_fp8_quant.quantize_sage_bhsd.
    splits = min(triton.cdiv(seqlen_k, 128), 128)
    with torch.cuda.device(device):
        if batch == 1 and splits == 1:
            _kv_partial_stats[(1, 4, heads)](
                k_bhsd, v_bhsd, k_mean, v_scale,
                seqlen_k, heads, 1, 128, 32, FINAL_STATS=True, num_warps=4,
            )
        else:
            partials = torch.empty((2, batch, heads, splits, 128), device=device, dtype=torch.float32)
            k_sum, v_max = partials.unbind(0)
            _kv_partial_stats[(splits, 4, batch * heads)](
                k_bhsd, v_bhsd, k_sum, v_max,
                seqlen_k, heads, splits, 128, 32, num_warps=4,
            )
            cols = max(1, 32 // triton.next_power_of_2(batch))
            _kv_finish_stats[(heads, 128 // cols)](
                k_sum, v_max, k_mean, v_scale, seqlen_k, batch, heads, splits,
                triton.next_power_of_2(batch), triton.next_power_of_2(splits), cols, num_warps=4,
            )
        rows = 32
        _quantize_nvfp4_qk_fp8_v[(3, triton.cdiv(max(pad_q, pad_k), rows), batch * heads)](
            q_bhsd, k_bhsd, v_bhsd, q_fp4, q_sf, k_fp4, k_sf, v_fp8, v_scale, k_mean,
            seqlen_q, seqlen_k, pad_q, pad_k, heads, rows, num_warps=4,
        )
    return q_fp4, q_sf, k_fp4, k_sf, v_fp8, v_scale
