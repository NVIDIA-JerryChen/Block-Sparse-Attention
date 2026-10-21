"""Public import shim for forward-only SM100/SM103 FP8 blk128 BSA."""

if __package__:
    from .bsa_attn_interface import bsa_fp8_blk128_fwd
    from .bsa_fp8_quant import quantize_sage_bhsd
else:
    from bsa_attn_interface import bsa_fp8_blk128_fwd
    from bsa_fp8_quant import quantize_sage_bhsd

__all__ = ["bsa_fp8_blk128_fwd", "quantize_sage_bhsd"]
