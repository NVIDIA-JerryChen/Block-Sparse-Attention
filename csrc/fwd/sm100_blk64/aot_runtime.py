import functools
import os
from pathlib import Path
from typing import Iterable, Optional

import cutlass
import cutlass.cute as cute
import torch

import block_sparse_attention.utils.cache_utils  # noqa: F401
from block_sparse_attention.csrc.fwd.sm100_blk64.aot_utils import (
    SM100_AOT_MANIFEST,
    Sm100AotCombineVariant,
    Sm100AotVariant,
    compute_sm100_aot_source_fingerprint,
    get_cuda_runtime_version,
    get_host_cpu_arch,
    get_sm100_aot_artifact_dir,
    load_sm100_aot_manifest,
    sha256_file,
)


SM100_AOT_DIR_ENV = "BSA_SM100_AOT_DIR"
SM100_AOT_ONLY_ENV = "BSA_SM100_AOT_ONLY"
_SM100_AOT_TARGET_BY_DEVICE_ARCH = {
    100: "sm_100a",
    103: "sm_103a",
}


class Sm100AotArtifactError(RuntimeError):
    pass


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def is_sm100_aot_only() -> bool:
    return _env_flag(SM100_AOT_ONLY_ENV)


def sm100_aot_target_arch(device_arch: int) -> str:
    try:
        return _SM100_AOT_TARGET_BY_DEVICE_ARCH[int(device_arch)]
    except KeyError as error:
        raise Sm100AotArtifactError(
            "SM100/SM103 AOT artifacts require compute capability 10.0 or "
            f"10.3, got {device_arch}"
        ) from error


def _default_sm100_aot_root() -> Path:
    return Path(__file__).resolve().parent / "aot_artifacts"


def resolve_sm100_aot_artifact_dir(
    target_arch: str,
    root: Optional[Path | str] = None,
) -> Path:
    if root is None:
        root = os.getenv(SM100_AOT_DIR_ENV) or _default_sm100_aot_root()
    root = Path(root).expanduser().resolve()
    if (root / SM100_AOT_MANIFEST).is_file():
        return root
    return get_sm100_aot_artifact_dir(root, target_arch)


def is_sm100_aot_layout_supported(
    variant: Sm100AotVariant,
    tensors: Iterable[Optional[torch.Tensor]],
) -> bool:
    tensors = tuple(tensors)
    if len(tensors) != 9:
        return False
    output_dtype = torch.float32 if variant.kv_splits > 1 else torch.bfloat16
    expected_specs = (
        (True, 4, torch.bfloat16, 16),
        (True, 4, torch.bfloat16, 16),
        (True, 4, torch.bfloat16, 16),
        (True, 4, output_dtype, 16),
        (True, 3, torch.float32, 4),
        (True, 4, torch.int32, 16),
        (variant.has_block_sizes, 1, torch.int32, 16),
        (variant.has_block_nums, 3, torch.int32, 16),
        (
            variant.kv_splits > 1
            and (variant.has_block_nums or variant.use_clc_scheduler),
            4,
            torch.int32,
            16,
        ),
    )
    device = tensors[0].device if tensors[0] is not None else None
    for tensor, (expected, rank, dtype, alignment) in zip(
        tensors,
        expected_specs,
    ):
        if (tensor is not None) != expected:
            return False
        if tensor is None:
            continue
        if tensor.ndim != rank or tensor.dtype != dtype:
            return False
        if not tensor.is_cuda or tensor.device != device:
            return False
        if tensor.stride(-1) != 1:
            return False
        if any(stride == 0 for stride in tensor.stride()):
            return False
        if getattr(tensor, "fake_mode", None) is None:
            try:
                if tensor.data_ptr() % alignment != 0:
                    return False
            except (RuntimeError, TypeError):
                return False
    return True


@functools.cache
def _current_sm100_aot_source_fingerprint() -> str:
    return compute_sm100_aot_source_fingerprint()


def _validate_manifest_compatibility(manifest: dict, target_arch: str) -> None:
    if manifest["target_arch"] != target_arch:
        raise Sm100AotArtifactError(
            f"SM100 AOT target mismatch: manifest has {manifest['target_arch']}, "
            f"runtime requires {target_arch}"
        )
    cpu_arch = get_host_cpu_arch()
    if manifest["cpu_arch"] != cpu_arch:
        raise Sm100AotArtifactError(
            f"SM100 AOT CPU architecture mismatch: manifest has "
            f"{manifest['cpu_arch']}, runtime requires {cpu_arch}"
        )
    runtime_dsl_version = str(cutlass.__version__)
    manifest_dsl_version = manifest.get("cutlass_dsl_version")
    if manifest_dsl_version != runtime_dsl_version:
        raise Sm100AotArtifactError(
            f"SM100 AOT CUTLASS DSL version mismatch: manifest has "
            f"{manifest_dsl_version}, runtime has {runtime_dsl_version}"
        )
    cuda_version = get_cuda_runtime_version()
    manifest_cuda_version = manifest.get("cuda_runtime_version", "unknown")
    if "unknown" not in (cuda_version, manifest_cuda_version):
        if manifest_cuda_version != cuda_version:
            raise Sm100AotArtifactError(
                f"SM100 AOT CUDA version mismatch: manifest has "
                f"{manifest_cuda_version}, runtime has {cuda_version}"
            )
    source_fingerprint = _current_sm100_aot_source_fingerprint()
    if manifest.get("source_fingerprint") != source_fingerprint:
        raise Sm100AotArtifactError(
            "SM100 AOT source fingerprint mismatch; rebuild the artifacts "
            "from the current BSA source tree"
        )


@functools.cache
def _load_manifest_cached(
    manifest_path: str,
    modified_ns: int,
    size: int,
    inode: int,
) -> dict:
    manifest = load_sm100_aot_manifest(manifest_path)
    stat = Path(manifest_path).stat()
    if (stat.st_mtime_ns, stat.st_size, stat.st_ino) != (
        modified_ns,
        size,
        inode,
    ):
        raise Sm100AotArtifactError(
            f"SM100 AOT manifest changed while loading: {manifest_path}"
        )
    return manifest


@functools.cache
def _load_kernel_cached(
    shared_library_path: str,
    function_name: str,
    expected_sha256: str,
):
    actual_sha256 = sha256_file(shared_library_path)
    if actual_sha256 != expected_sha256:
        raise Sm100AotArtifactError(
            f"SM100 AOT checksum mismatch for {shared_library_path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        module = cute.runtime.load_module(
            shared_library_path,
            enable_tvm_ffi=False,
        )
        function = getattr(module, function_name)
    except Exception as error:
        raise Sm100AotArtifactError(
            f"Failed to load SM100 AOT function {function_name!r} from "
            f"{shared_library_path}: {error}"
        ) from error
    return module, function


@functools.cache
def _bind_kernel_cached(
    shared_library_path: str,
    function_name: str,
    expected_sha256: str,
    device_index: int,
):
    module, function = _load_kernel_cached(
        shared_library_path,
        function_name,
        expected_sha256,
    )
    try:
        executor = function.to(device_index)
    except Exception as error:
        raise Sm100AotArtifactError(
            f"Failed to bind SM100 AOT function {function_name!r} to CUDA "
            f"device {device_index}: {error}"
        ) from error
    return module, executor


def clear_sm100_aot_runtime_cache() -> None:
    _load_manifest_cached.cache_clear()
    _load_kernel_cached.cache_clear()
    _bind_kernel_cached.cache_clear()
    _current_sm100_aot_source_fingerprint.cache_clear()


def _get_sm100_aot_entry(
    variant: Sm100AotVariant | Sm100AotCombineVariant,
    device_arch: int,
    section: str,
    *,
    required: Optional[bool],
    root: Optional[Path | str],
    device_index: int,
):
    if required is None:
        required = is_sm100_aot_only()
    target_arch = sm100_aot_target_arch(device_arch)
    artifact_dir = resolve_sm100_aot_artifact_dir(target_arch, root)
    manifest_path = artifact_dir / SM100_AOT_MANIFEST
    if not manifest_path.is_file():
        if required:
            raise Sm100AotArtifactError(
                f"SM100 AOT-only mode requires {manifest_path}; build artifacts "
                "with `make aot-sm100` or set BSA_SM100_AOT_DIR"
            )
        return None
    try:
        stat = manifest_path.stat()
        manifest = _load_manifest_cached(
            str(manifest_path), stat.st_mtime_ns, stat.st_size, stat.st_ino
        )
    except (OSError, ValueError, Sm100AotArtifactError) as error:
        raise Sm100AotArtifactError(
            f"Failed to load SM100 AOT manifest {manifest_path}: {error}"
        ) from error
    _validate_manifest_compatibility(manifest, target_arch)
    entry = manifest[section].get(variant.name)
    if entry is None:
        if required:
            label = "combine variant" if section == "combine_variants" else "variant"
            raise Sm100AotArtifactError(
                f"SM100 AOT-only mode is missing {label} {variant.name} in "
                f"{manifest_path}"
            )
        return None
    shared_library_path = artifact_dir / entry["shared_library"]
    if not shared_library_path.is_file():
        raise Sm100AotArtifactError(
            f"SM100 AOT shared library is missing: {shared_library_path}"
        )
    _, executor = _bind_kernel_cached(
        str(shared_library_path),
        entry["function_name"],
        entry["sha256"],
        int(device_index),
    )
    return executor


def get_sm100_aot_kernel(
    variant: Sm100AotVariant,
    device_arch: int,
    tensors: Iterable[Optional[torch.Tensor]],
    *,
    required: Optional[bool] = None,
    root: Optional[Path | str] = None,
):
    if required is None:
        required = is_sm100_aot_only()
    tensors = tuple(tensors)
    if not is_sm100_aot_layout_supported(variant, tensors):
        if required:
            raise Sm100AotArtifactError(
                "SM100 AOT-only mode requires the compiled tensor ranks and "
                "dtypes on one CUDA device, matching optional tensors, "
                "non-broadcast layouts, aligned pointers, and unit trailing "
                "strides"
            )
        return None
    return _get_sm100_aot_entry(
        variant,
        device_arch,
        "variants",
        required=required,
        root=root,
        device_index=int(tensors[0].device.index),
    )


def get_sm100_aot_combine_kernel(
    variant: Sm100AotCombineVariant,
    device_arch: int,
    *,
    required: Optional[bool] = None,
    root: Optional[Path | str] = None,
    device_index: Optional[int] = None,
):
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _get_sm100_aot_entry(
        variant,
        device_arch,
        "combine_variants",
        required=required,
        root=root,
        device_index=int(device_index),
    )
