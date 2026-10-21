"""Cached native AOT loading; strict mode never consults the JIT cache."""

import functools
import json
import os
from pathlib import Path

import cutlass
import cutlass.cute as cute

from block_sparse_attention.csrc.fwd.sm100_blk64.aot_runtime import (
    Sm100AotArtifactError,
    is_sm100_aot_only,
)
from block_sparse_attention.csrc.fwd.sm100_blk128.aot_utils import (
    MANIFEST_NAME,
    artifact_dir,
    get_cuda_runtime_version,
    get_host_cpu_arch,
    sha256_file,
    source_fingerprint,
    validate_manifest,
)

AOT_DIR_ENV = "BSA_FP8_BLK128_AOT_DIR"
AOT_ONLY_ENV = "BSA_FP8_BLK128_AOT_ONLY"
_CACHE_GENERATION = 0


def is_fp8_blk128_aot_only():
    return is_sm100_aot_only() or os.getenv(AOT_ONLY_ENV, "").strip().lower() in (
        "1", "true", "yes", "on"
    )


def dispatch_token():
    """Configuration identity, independent of tensor contents and JIT entries.

    Call clear_fp8_blk128_aot_runtime_cache after replacing artifacts at the
    same path. Hot dispatch does not repeatedly read manifests or source files.
    """
    return (os.getenv(AOT_DIR_ENV), is_fp8_blk128_aot_only(), _CACHE_GENERATION)


@functools.cache
def _source_fingerprint():
    return source_fingerprint()


@functools.cache
def _load_manifest(path, target_arch):
    try:
        manifest = validate_manifest(json.loads(Path(path).read_text()))
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise Sm100AotArtifactError(f"Invalid FP8 blk128 AOT manifest {path}: {error}") from error
    expected = {
        "target_arch": target_arch,
        "cpu_arch": get_host_cpu_arch(),
        "cutlass_dsl_version": str(cutlass.__version__),
        "cuda_runtime_version": get_cuda_runtime_version(),
        "source_fingerprint": _source_fingerprint(),
    }
    for field, value in expected.items():
        if manifest[field] != value:
            raise Sm100AotArtifactError(
                f"FP8 blk128 AOT {field} mismatch: artifact {manifest[field]!r}, runtime {value!r}"
            )
    return manifest


@functools.cache
def _load_bound_kernel(path, name, checksum, device_index):
    if not Path(path).is_file() or sha256_file(path) != checksum:
        raise Sm100AotArtifactError(f"FP8 blk128 AOT missing library or checksum mismatch: {path}")
    try:
        module = cute.runtime.load_module(path, enable_tvm_ffi=False)
        executor = getattr(module, name).to(device_index)
    except Exception as error:
        raise Sm100AotArtifactError(f"Unable to load FP8 blk128 AOT function {name}: {error}") from error
    # Keep the native module alive for the lifetime of the device executor.
    return module, executor


@functools.cache
def _resolve_bundle(variant, device_arch, device_index, token):
    target_arch = {100: "sm_100a", 103: "sm_103a"}.get(device_arch)
    if target_arch is None:
        raise Sm100AotArtifactError(f"FP8 blk128 AOT requires SM100/SM103, got {device_arch}")
    configured_root, required, _ = token
    root = Path(configured_root).expanduser().resolve() if configured_root else Path(__file__).parent / "aot_artifacts"
    directory = root if (root / MANIFEST_NAME).is_file() else artifact_dir(root, target_arch)
    path = directory / MANIFEST_NAME
    if not path.is_file():
        if required:
            raise Sm100AotArtifactError(f"FP8 blk128 AOT-only requires {path}; build native artifacts first")
        return None
    manifest = _load_manifest(str(path), target_arch)
    names = [("variants", variant.name)]
    if variant.kv_splits > 1:
        names.append(("combine_variants", variant.combine_name))
    entries = [manifest[section].get(name) for section, name in names]
    if any(entry is None for entry in entries):
        if required:
            missing = [name for (_, name), entry in zip(names, entries) if entry is None]
            raise Sm100AotArtifactError(f"FP8 blk128 AOT-only missing producer/combine: {missing}")
        return None
    functions = []
    for entry in entries:
        _, executor = _load_bound_kernel(
            str(directory / entry["shared_library"]), entry["function_name"],
            entry["sha256"], int(device_index),
        )
        functions.append(executor)
    return functions[0], functions[1] if len(functions) > 1 else None


def get_fp8_blk128_aot_bundle(variant, device_arch, device_index):
    return _resolve_bundle(variant, int(device_arch), int(device_index), dispatch_token())


def clear_fp8_blk128_aot_runtime_cache():
    global _CACHE_GENERATION
    _CACHE_GENERATION += 1
    _resolve_bundle.cache_clear()
    _load_manifest.cache_clear()
    _load_bound_kernel.cache_clear()
    _source_fingerprint.cache_clear()
