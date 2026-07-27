"""Public API for Block Sparse Attention."""

from pathlib import Path


_package_dir = Path(__file__).absolute().parent
_source_dir = _package_dir.parent
_is_source_layout = (
    not (_package_dir / "bsa_attn_interface.py").is_file()
    and (_source_dir / "pyproject.toml").is_file()
    and (_source_dir / "bsa_attn_interface.py").is_file()
    and (_source_dir / "csrc" / "__init__.py").is_file()
    and (_source_dir / "utils" / "__init__.py").is_file()
)
_source_path = str(_source_dir) if _is_source_layout else None
if _is_source_layout:
    __path__.append(_source_path)


__all__ = [
    "bsa_attn_bwd",
    "bsa_attn_fwd",
    "bsa_fp8_blk64_fwd",
    "quantize_sage_bhsd",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    interface = import_module(".bsa_attn_interface", __name__)
    quantization = import_module(".bsa_fp8_quant", __name__)
    import_module(".bsa_fp8_blk64", __name__)
    for api_name in ("bsa_attn_bwd", "bsa_attn_fwd", "bsa_fp8_blk64_fwd"):
        globals()[api_name] = getattr(interface, api_name)
    globals()["quantize_sage_bhsd"] = quantization.quantize_sage_bhsd

    global _source_path
    if _source_path is not None:
        __path__.remove(_source_path)
        _source_path = None
    return globals()[name]


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


del Path, _is_source_layout, _package_dir, _source_dir
