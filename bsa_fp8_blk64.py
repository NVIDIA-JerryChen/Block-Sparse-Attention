"""Public import shim for the forward-only SM100 FP8 blk64 BSA."""

if __package__:
    from .bsa_attn_interface import bsa_fp8_blk64_fwd
    from .bsa_fp8_quant import quantize_sage_bhsd
else:
    from bsa_attn_interface import bsa_fp8_blk64_fwd
    from bsa_fp8_quant import quantize_sage_bhsd

__all__ = ["bsa_fp8_blk64_fwd", "quantize_sage_bhsd"]
