import functools
import os
from pathlib import Path
from typing import Iterable, Optional

import cutlass
import cutlass.cute as cute
import torch

import block_sparse_attention.utils.cache_utils  # noqa: F401  # Preload runtime symbols for AOT modules.
from block_sparse_attention.csrc.fwd.sm90_blk64.aot_utils import (
    SM90_AOT_MANIFEST,
    Sm90AotCombineVariant,
    Sm90AotVariant,
    compute_sm90_aot_source_fingerprint,
    get_cuda_runtime_version,
    get_host_cpu_arch,
    get_sm90_aot_artifact_dir,
    load_sm90_aot_manifest,
    sha256_file,
)


SM90_AOT_DIR_ENV = "BSA_SM90_AOT_DIR"
SM90_AOT_ONLY_ENV = "BSA_SM90_AOT_ONLY"
_SM90_AOT_LEADING_DIMS = (1, 1, 0, 1, 0, 0, 0, 0, 0)


class Sm90AotArtifactError(RuntimeError):
    pass


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def is_sm90_aot_only() -> bool:
    return _env_flag(SM90_AOT_ONLY_ENV)


def sm90_aot_target_arch(device_arch: int) -> str:
    if int(device_arch) != 90:
        raise Sm90AotArtifactError(
            f"SM90 AOT artifacts require compute capability 9.0, got {device_arch}"
        )
    return "sm_90a"


def _default_sm90_aot_root() -> Path:
    return Path(__file__).resolve().parent / "aot_artifacts"


def resolve_sm90_aot_artifact_dir(
    target_arch: str,
    root: Optional[Path | str] = None,
) -> Path:
    if root is None:
        root = os.getenv(SM90_AOT_DIR_ENV) or _default_sm90_aot_root()
    root = Path(root).expanduser().resolve()
    if (root / SM90_AOT_MANIFEST).is_file():
        return root
    return get_sm90_aot_artifact_dir(root, target_arch)


def is_sm90_aot_layout_supported(tensors: Iterable[torch.Tensor]) -> bool:
    tensors = tuple(tensors)
    if len(tensors) != len(_SM90_AOT_LEADING_DIMS):
        return False
    for tensor, leading_dim in zip(tensors, _SM90_AOT_LEADING_DIMS):
        if tensor.stride(leading_dim) != 1:
            return False
        if any(stride == 0 for stride in tensor.stride()):
            return False
    return True


@functools.cache
def _current_sm90_aot_source_fingerprint() -> str:
    return compute_sm90_aot_source_fingerprint()


def _validate_manifest_compatibility(manifest: dict, target_arch: str) -> None:
    if manifest["target_arch"] != target_arch:
        raise Sm90AotArtifactError(
            f"SM90 AOT target mismatch: manifest has {manifest['target_arch']}, "
            f"runtime requires {target_arch}"
        )
    cpu_arch = get_host_cpu_arch()
    if manifest["cpu_arch"] != cpu_arch:
        raise Sm90AotArtifactError(
            f"SM90 AOT CPU architecture mismatch: manifest has "
            f"{manifest['cpu_arch']}, runtime requires {cpu_arch}"
        )
    dsl_version = str(cutlass.__version__)
    if manifest.get("cutlass_dsl_version") != dsl_version:
        raise Sm90AotArtifactError(
            f"SM90 AOT CUTLASS DSL version mismatch: manifest has "
            f"{manifest.get('cutlass_dsl_version')}, runtime has {dsl_version}"
        )
    cuda_version = get_cuda_runtime_version()
    manifest_cuda_version = manifest.get("cuda_runtime_version", "unknown")
    if "unknown" not in (cuda_version, manifest_cuda_version):
        if manifest_cuda_version != cuda_version:
            raise Sm90AotArtifactError(
                f"SM90 AOT CUDA version mismatch: manifest has "
                f"{manifest_cuda_version}, runtime has {cuda_version}"
            )
    source_fingerprint = _current_sm90_aot_source_fingerprint()
    if manifest.get("source_fingerprint") != source_fingerprint:
        raise Sm90AotArtifactError(
            "SM90 AOT source fingerprint mismatch; rebuild the AOT artifacts "
            "from the current BSA source tree"
        )


@functools.cache
def _load_manifest_cached(manifest_path: str) -> dict:
    return load_sm90_aot_manifest(manifest_path)


@functools.cache
def _load_kernel_cached(
    shared_library_path: str,
    function_name: str,
    expected_sha256: str,
):
    actual_sha256 = sha256_file(shared_library_path)
    if actual_sha256 != expected_sha256:
        raise Sm90AotArtifactError(
            f"SM90 AOT checksum mismatch for {shared_library_path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    module = cute.runtime.load_module(shared_library_path, enable_tvm_ffi=False)
    try:
        function = getattr(module, function_name)
    except AttributeError as error:
        raise Sm90AotArtifactError(
            f"SM90 AOT function {function_name!r} is missing from "
            f"{shared_library_path}"
        ) from error
    return module, function


def clear_sm90_aot_runtime_cache() -> None:
    _load_manifest_cached.cache_clear()
    _load_kernel_cached.cache_clear()
    _current_sm90_aot_source_fingerprint.cache_clear()


def _get_sm90_aot_entry(
    variant: Sm90AotVariant | Sm90AotCombineVariant,
    device_arch: int,
    section: str,
    *,
    required: Optional[bool],
    root: Optional[Path | str],
):
    if required is None:
        required = is_sm90_aot_only()
    target_arch = sm90_aot_target_arch(device_arch)
    artifact_dir = resolve_sm90_aot_artifact_dir(target_arch, root)
    manifest_path = artifact_dir / SM90_AOT_MANIFEST
    if not manifest_path.is_file():
        if required:
            raise Sm90AotArtifactError(
                f"SM90 AOT-only mode requires {manifest_path}; build artifacts "
                "with `make aot-sm90` or set BSA_SM90_AOT_DIR"
            )
        return None
    try:
        manifest = _load_manifest_cached(str(manifest_path))
    except (OSError, ValueError) as error:
        raise Sm90AotArtifactError(
            f"Failed to load SM90 AOT manifest {manifest_path}: {error}"
        ) from error
    _validate_manifest_compatibility(manifest, target_arch)
    entry = manifest[section].get(variant.name)
    if entry is None:
        if required:
            label = "combine variant" if section == "combine_variants" else "variant"
            raise Sm90AotArtifactError(
                f"SM90 AOT-only mode is missing {label} {variant.name} in "
                f"{manifest_path}"
            )
        return None
    shared_library_path = artifact_dir / entry["shared_library"]
    if not shared_library_path.is_file():
        raise Sm90AotArtifactError(
            f"SM90 AOT shared library is missing: {shared_library_path}"
        )
    _, function = _load_kernel_cached(
        str(shared_library_path),
        entry["function_name"],
        entry["sha256"],
    )
    return function


def get_sm90_aot_kernel(
    variant: Sm90AotVariant,
    device_arch: int,
    tensors: Iterable[torch.Tensor],
    *,
    required: Optional[bool] = None,
    root: Optional[Path | str] = None,
):
    if required is None:
        required = is_sm90_aot_only()
    if not is_sm90_aot_layout_supported(tensors):
        if required:
            raise Sm90AotArtifactError(
                "SM90 AOT-only mode requires non-broadcast dynamic layouts "
                "with unit leading strides"
            )
        return None
    return _get_sm90_aot_entry(
        variant,
        device_arch,
        "variants",
        required=required,
        root=root,
    )


def get_sm90_aot_combine_kernel(
    variant: Sm90AotCombineVariant,
    device_arch: int,
    *,
    required: Optional[bool] = None,
    root: Optional[Path | str] = None,
):
    return _get_sm90_aot_entry(
        variant,
        device_arch,
        "combine_variants",
        required=required,
        root=root,
    )
