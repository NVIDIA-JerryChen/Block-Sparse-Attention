"""Public Sage FP8 forward API with selectable sparse block size."""

if __package__:
    from .bsa_attn_interface import bsa_fp8_blk128_fwd, bsa_fp8_fwd
    from .bsa_fp8_quant import quantize_sage_bhsd
else:
    from bsa_attn_interface import bsa_fp8_blk128_fwd, bsa_fp8_fwd
    from bsa_fp8_quant import quantize_sage_bhsd

__all__ = ["bsa_fp8_blk128_fwd", "bsa_fp8_fwd", "quantize_sage_bhsd"]
