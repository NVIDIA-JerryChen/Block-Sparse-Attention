"""Runtime loader for SM120 Sage Triton quantization AOT artifacts."""

import ctypes
import functools
from pathlib import Path
from typing import Optional

import torch
import triton

from block_sparse_attention.csrc.fwd.sm120_blk64.aot_runtime import (
    Sm120AotArtifactError,
    _load_manifest_cached,
    _validate_manifest_compatibility,
    clear_sm120_aot_runtime_cache,
    is_sm120_aot_only,
    resolve_sm120_aot_artifact_dir,
    sm120_aot_target_arch,
)
from block_sparse_attention.csrc.fwd.sm120_blk64.aot_utils import (
    SM120_AOT_MANIFEST,
    sha256_file,
)


_STREAM = ctypes.c_void_p
_DEVICE_POINTER = ctypes.c_uint64
_STRIDE = ctypes.c_int64
_SCALAR = ctypes.c_int32


def _kernel_argtypes(
    pointer_count: int,
    stride_count: int,
    scalar_count: int,
) -> list:
    return [
        _STREAM,
        *(_DEVICE_POINTER for _ in range(pointer_count)),
        *(_STRIDE for _ in range(stride_count)),
        *(_SCALAR for _ in range(scalar_count)),
    ]


class Sm120SageQuantAotRuntime:
    """Typed ctypes bindings for the four Sage quantization kernels."""

    def __init__(self, library, functions: dict[str, str]) -> None:
        self._library = library
        self._q = self._bind(functions["q"], _kernel_argtypes(3, 8, 2))
        self._stats_partial = self._bind(
            functions["stats_partial"],
            _kernel_argtypes(4, 9, 3),
        )
        self._stats_finalize = self._bind(
            functions["stats_finalize"],
            _kernel_argtypes(4, 7, 4),
        )
        self._kv = self._bind(functions["kv"], _kernel_argtypes(7, 18, 2))

    def _bind(self, name: str, argtypes: list):
        try:
            function = getattr(self._library, name)
        except AttributeError as error:
            raise Sm120AotArtifactError(
                f"SM120 Sage quantization AOT function {name!r} is missing"
            ) from error
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        return function

    @staticmethod
    def _pointer(tensor: torch.Tensor) -> int:
        return int(tensor.data_ptr())

    @staticmethod
    def _launch(function, device: torch.device, *args) -> None:
        with torch.cuda.device(device):
            stream = int(torch.cuda.current_stream(device).cuda_stream)
            result = int(function(stream, *args))
        if result != 0:
            raise RuntimeError(
                f"SM120 Sage quantization AOT launch failed with CUDA error {result}"
            )

    def quantize_q(
        self,
        q: torch.Tensor,
        q_int8: torch.Tensor,
        q_scale: torch.Tensor,
    ) -> None:
        self._launch(
            self._q,
            q.device,
            self._pointer(q),
            self._pointer(q_int8),
            self._pointer(q_scale),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q_int8.stride(0),
            q_int8.stride(1),
            q_int8.stride(2),
            q_scale.stride(0),
            q_scale.stride(1),
            q.shape[2],
            q.shape[0],
        )

    def stats_partial(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_partial: torch.Tensor,
        v_partial: torch.Tensor,
    ) -> None:
        self._launch(
            self._stats_partial,
            k.device,
            self._pointer(k),
            self._pointer(v),
            self._pointer(k_partial),
            self._pointer(v_partial),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            k_partial.stride(0),
            k_partial.stride(1),
            k_partial.stride(2),
            k.shape[2],
            k.shape[1],
            k.shape[0],
        )

    def stats_finalize(
        self,
        k_partial: torch.Tensor,
        v_partial: torch.Tensor,
        k_mean: torch.Tensor,
        v_scale: torch.Tensor,
        seqlen_k: int,
    ) -> None:
        self._launch(
            self._stats_finalize,
            k_partial.device,
            self._pointer(k_partial),
            self._pointer(v_partial),
            self._pointer(k_mean),
            self._pointer(v_scale),
            k_partial.stride(0),
            k_partial.stride(1),
            k_partial.stride(2),
            k_mean.stride(0),
            k_mean.stride(1),
            v_scale.stride(0),
            v_scale.stride(1),
            k_partial.shape[2],
            seqlen_k,
            k_partial.shape[1],
            k_partial.shape[0],
        )

    def quantize_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_mean: torch.Tensor,
        v_scale: torch.Tensor,
        k_int8: torch.Tensor,
        v_fp8: torch.Tensor,
        k_scale: torch.Tensor,
    ) -> None:
        self._launch(
            self._kv,
            k.device,
            self._pointer(k),
            self._pointer(v),
            self._pointer(k_mean),
            self._pointer(v_scale),
            self._pointer(k_int8),
            self._pointer(v_fp8),
            self._pointer(k_scale),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            k_mean.stride(0),
            k_mean.stride(1),
            v_scale.stride(0),
            v_scale.stride(1),
            k_int8.stride(0),
            k_int8.stride(1),
            k_int8.stride(2),
            v_fp8.stride(0),
            v_fp8.stride(1),
            v_fp8.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            k.shape[2],
            k.shape[0],
        )


@functools.cache
def _load_quant_runtime_cached(
    shared_library_path: str,
    expected_sha256: str,
    q_function: str,
    stats_partial_function: str,
    stats_finalize_function: str,
    kv_function: str,
) -> Sm120SageQuantAotRuntime:
    actual_sha256 = sha256_file(shared_library_path)
    if actual_sha256 != expected_sha256:
        raise Sm120AotArtifactError(
            f"SM120 Sage quantization AOT checksum mismatch for "
            f"{shared_library_path}: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        library = ctypes.CDLL(shared_library_path)
    except OSError as error:
        raise Sm120AotArtifactError(
            f"Failed to load SM120 Sage quantization AOT library "
            f"{shared_library_path}: {error}"
        ) from error
    return Sm120SageQuantAotRuntime(
        library,
        {
            "q": q_function,
            "stats_partial": stats_partial_function,
            "stats_finalize": stats_finalize_function,
            "kv": kv_function,
        },
    )


def clear_sm120_sage_quant_aot_runtime_cache() -> None:
    _load_quant_runtime_cached.cache_clear()
    clear_sm120_aot_runtime_cache()


def get_sm120_sage_quant_aot(
    device_arch: int,
    *,
    required: Optional[bool] = None,
    root: Optional[Path | str] = None,
) -> Optional[Sm120SageQuantAotRuntime]:
    if required is None:
        required = is_sm120_aot_only()
    target_arch = sm120_aot_target_arch(device_arch)
    artifact_dir = resolve_sm120_aot_artifact_dir(target_arch, root)
    manifest_path = artifact_dir / SM120_AOT_MANIFEST
    if not manifest_path.is_file():
        if required:
            raise Sm120AotArtifactError(
                f"SM120 AOT-only mode requires {manifest_path}; build artifacts "
                "with `make aot-sm120` or set BSA_SM120_AOT_DIR"
            )
        return None
    try:
        manifest = _load_manifest_cached(str(manifest_path))
    except (OSError, ValueError) as error:
        raise Sm120AotArtifactError(
            f"Failed to load SM120 AOT manifest {manifest_path}: {error}"
        ) from error
    _validate_manifest_compatibility(manifest, target_arch)
    entry = manifest.get("quantization")
    if entry is None:
        if required:
            raise Sm120AotArtifactError(
                f"SM120 AOT-only mode is missing Sage quantization artifacts in "
                f"{manifest_path}"
            )
        return None
    if entry["triton_version"] != str(triton.__version__):
        raise Sm120AotArtifactError(
            f"SM120 Sage quantization Triton version mismatch: manifest has "
            f"{entry['triton_version']}, runtime has {triton.__version__}"
        )
    shared_library_path = artifact_dir / entry["shared_library"]
    if not shared_library_path.is_file():
        raise Sm120AotArtifactError(
            f"SM120 Sage quantization AOT library is missing: "
            f"{shared_library_path}"
        )
    functions = entry["functions"]
    return _load_quant_runtime_cached(
        str(shared_library_path),
        entry["sha256"],
        functions["q"],
        functions["stats_partial"],
        functions["stats_finalize"],
        functions["kv"],
    )


__all__ = [
    "Sm120SageQuantAotRuntime",
    "clear_sm120_sage_quant_aot_runtime_cache",
    "get_sm120_sage_quant_aot",
]
