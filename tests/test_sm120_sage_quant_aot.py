import os
from pathlib import Path

import cutlass
import pytest
import torch
import triton

from block_sparse_attention.csrc.fwd.sm120_blk64.aot_runtime import (
    SM120_AOT_DIR_ENV,
    SM120_AOT_ONLY_ENV,
    Sm120AotArtifactError,
)
from block_sparse_attention.csrc.fwd.sm120_blk64.aot_utils import (
    SM120_AOT_SCHEMA_VERSION,
    compute_sm120_aot_source_fingerprint,
    get_cuda_runtime_version,
    get_host_cpu_arch,
    sha256_file,
    write_sm120_aot_manifest,
)
from block_sparse_attention.csrc.fwd.sm120_blk64 import quant_aot_runtime
from block_sparse_attention.csrc.fwd.sm120_blk64.quant_aot_runtime import (
    clear_sm120_sage_quant_aot_runtime_cache,
    get_sm120_sage_quant_aot,
)


_RUNTIME_TEST_ENABLED = os.getenv("BSA_TEST_SM120_AOT_RUNTIME") == "1"
_IS_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


class _FakeFunction:
    def __init__(self) -> None:
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return 0


class _FakeLibrary:
    def __init__(self, functions: dict[str, str]) -> None:
        for function_name in functions.values():
            setattr(self, function_name, _FakeFunction())


def _write_quant_manifest(
    root: Path,
    *,
    triton_version: str | None = None,
    include_quantization: bool = True,
) -> tuple[Path, dict]:
    artifact_dir = root / get_host_cpu_arch() / "sm_120f"
    artifact_dir.mkdir(parents=True)
    functions = {
        "q": "quant_q",
        "stats_partial": "quant_stats_partial",
        "stats_finalize": "quant_stats_finalize",
        "kv": "quant_kv",
    }
    shared_library = artifact_dir / "quant.so"
    shared_library.write_bytes(b"quant-aot-library")
    sources = []
    headers = []
    for name in functions:
        source = artifact_dir / f"{name}.c"
        header = artifact_dir / f"{name}.h"
        source.write_bytes(b"source")
        header.write_bytes(b"header")
        sources.append(source.name)
        headers.append(header.name)
    manifest = {
        "schema_version": SM120_AOT_SCHEMA_VERSION,
        "target_arch": "sm_120f",
        "cpu_arch": get_host_cpu_arch(),
        "cutlass_dsl_version": str(cutlass.__version__),
        "cuda_python_version": "test",
        "cuda_runtime_version": get_cuda_runtime_version(),
        "python_abi": "test",
        "source_fingerprint": compute_sm120_aot_source_fingerprint(),
        "variants": {},
    }
    if include_quantization:
        manifest["quantization"] = {
            "triton_version": triton_version or str(triton.__version__),
            "shared_library": shared_library.name,
            "sha256": sha256_file(shared_library),
            "functions": functions,
            "sources": sources,
            "headers": headers,
        }
    manifest_path = write_sm120_aot_manifest(artifact_dir, manifest)
    return manifest_path, functions


def test_sm120_sage_quant_aot_loader_binds_and_caches(
    tmp_path: Path,
    monkeypatch,
):
    _, functions = _write_quant_manifest(tmp_path)
    libraries = []

    def fake_cdll(path):
        library = _FakeLibrary(functions)
        libraries.append((path, library))
        return library

    clear_sm120_sage_quant_aot_runtime_cache()
    monkeypatch.setattr(quant_aot_runtime.ctypes, "CDLL", fake_cdll)
    first = get_sm120_sage_quant_aot(120, required=True, root=tmp_path)
    second = get_sm120_sage_quant_aot(120, required=True, root=tmp_path)

    assert first is second
    assert len(libraries) == 1
    library = libraries[0][1]
    assert len(getattr(library, functions["q"]).argtypes) == 14
    assert len(getattr(library, functions["stats_partial"]).argtypes) == 17
    assert len(getattr(library, functions["stats_finalize"]).argtypes) == 16
    assert len(getattr(library, functions["kv"]).argtypes) == 28
    clear_sm120_sage_quant_aot_runtime_cache()


def test_sm120_sage_quant_aot_only_rejects_missing_entry(tmp_path: Path):
    _write_quant_manifest(tmp_path, include_quantization=False)
    clear_sm120_sage_quant_aot_runtime_cache()

    with pytest.raises(Sm120AotArtifactError, match="missing Sage quantization"):
        get_sm120_sage_quant_aot(120, required=True, root=tmp_path)
    clear_sm120_sage_quant_aot_runtime_cache()


def test_sm120_sage_quant_aot_rejects_checksum_mismatch(tmp_path: Path):
    manifest_path, _ = _write_quant_manifest(tmp_path)
    shared_library = manifest_path.parent / "quant.so"
    shared_library.write_bytes(b"modified")
    clear_sm120_sage_quant_aot_runtime_cache()

    with pytest.raises(Sm120AotArtifactError, match="checksum mismatch"):
        get_sm120_sage_quant_aot(120, required=True, root=tmp_path)
    clear_sm120_sage_quant_aot_runtime_cache()


def test_sm120_sage_quant_aot_rejects_triton_mismatch(tmp_path: Path):
    _write_quant_manifest(tmp_path, triton_version="0.0.test")
    clear_sm120_sage_quant_aot_runtime_cache()

    with pytest.raises(Sm120AotArtifactError, match="Triton version mismatch"):
        get_sm120_sage_quant_aot(120, required=True, root=tmp_path)
    clear_sm120_sage_quant_aot_runtime_cache()


@pytest.mark.skipif(
    not _RUNTIME_TEST_ENABLED or not _IS_SM120,
    reason="Set BSA_TEST_SM120_AOT_RUNTIME=1 and run on SM120",
)
def test_sm120_sage_quant_aot_only_matches_jit(monkeypatch):
    artifact_root = os.getenv(SM120_AOT_DIR_ENV)
    assert artifact_root, f"{SM120_AOT_DIR_ENV} must be set"
    import block_sparse_attention.bsa_sage_quant as sage_quant

    monkeypatch.delenv(SM120_AOT_DIR_ENV)
    monkeypatch.delenv(SM120_AOT_ONLY_ENV, raising=False)
    clear_sm120_sage_quant_aot_runtime_cache()
    torch.manual_seed(1207)
    q = torch.randn((2, 3, 137, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((2, 3, 309, 128), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    expected = sage_quant.quantize_sage_qkv_sm120(q, k, v)
    torch.cuda.synchronize()

    monkeypatch.setenv(SM120_AOT_DIR_ENV, artifact_root)
    monkeypatch.setenv(SM120_AOT_ONLY_ENV, "1")
    clear_sm120_sage_quant_aot_runtime_cache()

    class FailJit:
        def __getitem__(self, grid):
            raise AssertionError(f"Triton JIT path used for grid {grid}")

    for name in (
        "_quantize_sage_q_kernel",
        "_sage_kv_stats_partial_kernel",
        "_sage_kv_stats_finalize_kernel",
        "_quantize_sage_kv_kernel",
    ):
        monkeypatch.setattr(sage_quant, name, FailJit())
    actual = sage_quant.quantize_sage_qkv_sm120(q, k, v)
    torch.cuda.synchronize()

    for actual_tensor, expected_tensor in zip(actual, expected):
        if actual_tensor.dtype == torch.float8_e4m3fn:
            assert torch.equal(
                actual_tensor.view(torch.uint8),
                expected_tensor.view(torch.uint8),
            )
        else:
            assert torch.equal(actual_tensor, expected_tensor)
    clear_sm120_sage_quant_aot_runtime_cache()
