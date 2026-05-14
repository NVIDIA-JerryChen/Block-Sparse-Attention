# Copyright (c) 2025, Tri Dao.

from typing import Callable, TypeAlias

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


MaskGenFn: TypeAlias = Callable[[int], Uint32]
MASK_R2P_CHUNK_SIZE: int = 32


@dsl_user_op
def shr_u32(val: Uint32, shift: Uint32, *, loc=None, ip=None) -> Uint32:
    """Unsigned right shift via PTX, avoiding LLVM shift-by-32 UB."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(val).ir_value(loc=loc, ip=ip),
                Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shr.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def r2p_bitmask_below(limit: Int32, s: int) -> Uint32:
    """32-bit R2P bitmask keeping positions < limit."""
    m = max((s + 1) * MASK_R2P_CHUNK_SIZE - limit, 0)
    return shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def mask_r2p_lambda(
    X: cute.Tensor,
    mask_gen_fn: cutlass.Constexpr[MaskGenFn],
    rank1: bool = False,
) -> None:
    """Apply R2P-style masking with a custom 32-bit bitmask generator."""
    ncol = const_expr(cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape))
    chunk_size = MASK_R2P_CHUNK_SIZE
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, chunk_size)):
        mask = mask_gen_fn(s)
        for i in cutlass.range_constexpr(min(chunk_size, ncol - s * chunk_size)):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * chunk_size + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def apply_block_size_mask(
    acc_S: cute.Tensor,
    block_size: Int32,
    n_block_size: cutlass.Constexpr[int] = 128,
) -> None:
    """Mask positions >= block_size inside a tile using a 32-bit predicate mask."""
    if block_size < n_block_size:
        mask_r2p_lambda(
            acc_S,
            lambda s: r2p_bitmask_below(block_size, s),
            rank1=True,
        )
