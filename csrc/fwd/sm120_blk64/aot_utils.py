import hashlib
import json
import os
import platform
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from cuda.bindings import runtime as cuda_runtime


SM120_AOT_SCHEMA_VERSION = 1
SM120_AOT_MANIFEST = "manifest.json"
SM120_AOT_LAYOUT_MODE = "dynamic_strided_nonbroadcast"
SM120_QUANTIZED_AOT_MIN_CUTLASS_DSL_VERSION = "4.5.0"


@dataclass(frozen=True)
class Sm120AotVariant:
    dtype: str
    gqa_ratio: int
    has_block_nums: bool
    block_sizes_mode: int
    layout_mode: str = SM120_AOT_LAYOUT_MODE

    def __post_init__(self) -> None:
        if self.dtype not in ("bf16", "fp16", "fp8", "sage"):
            raise ValueError(f"Unsupported SM120 AOT dtype: {self.dtype}")
        if self.gqa_ratio < 1:
            raise ValueError("gqa_ratio must be >= 1")
        if self.block_sizes_mode not in (0, 1, 2, 3):
            raise ValueError("block_sizes_mode must be one of 0, 1, 2, or 3")
        if self.layout_mode != SM120_AOT_LAYOUT_MODE:
            raise ValueError(f"Unsupported SM120 AOT layout mode: {self.layout_mode}")
        if self.dtype in ("fp8", "sage"):
            if self.gqa_ratio != 1:
                raise ValueError(
                    "SM120 quantized AOT variants currently require gqa_ratio=1"
                )

    @property
    def has_block_sizes(self) -> bool:
        return self.block_sizes_mode != 0

    @property
    def is_fp8(self) -> bool:
        return self.dtype == "fp8"

    @property
    def is_sage(self) -> bool:
        return self.dtype == "sage"

    @property
    def name(self) -> str:
        block_nums = int(self.has_block_nums)
        return (
            f"bsa_sm120_blk64_{self.dtype}_gqa{self.gqa_ratio}_"
            f"bn{block_nums}_bs{self.block_sizes_mode}_dyn"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Sm120AotVariant":
        return cls(
            dtype=str(value["dtype"]),
            gqa_ratio=int(value["gqa_ratio"]),
            has_block_nums=bool(value["has_block_nums"]),
            block_sizes_mode=int(value["block_sizes_mode"]),
            layout_mode=str(value.get("layout_mode", SM120_AOT_LAYOUT_MODE)),
        )


def iter_sm120_aot_variants(
    dtypes: Iterable[str],
    gqa_ratios: Iterable[int],
    has_block_nums_values: Iterable[bool],
    block_sizes_modes: Iterable[int],
) -> tuple[Sm120AotVariant, ...]:
    variants = set()
    for dtype in dtypes:
        for gqa_ratio in gqa_ratios:
            for has_block_nums in has_block_nums_values:
                for block_sizes_mode in block_sizes_modes:
                    if dtype in ("fp8", "sage") and gqa_ratio != 1:
                        continue
                    variants.add(
                        Sm120AotVariant(
                            dtype,
                            gqa_ratio,
                            has_block_nums,
                            block_sizes_mode,
                        )
                    )
    return tuple(sorted(variants, key=lambda variant: variant.name))


def get_host_cpu_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    return machine


def get_cuda_runtime_version() -> str:
    error, version = cuda_runtime.cudaRuntimeGetVersion()
    if int(error) != 0:
        return "unknown"
    major = int(version) // 1000
    minor = (int(version) % 1000) // 10
    return f"{major}.{minor}"


def require_sm120_aot_cutlass_dsl_version(
    version: str,
    variants: Iterable[Sm120AotVariant],
) -> None:
    if not any(variant.is_fp8 or variant.is_sage for variant in variants):
        return
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(f"Cannot parse CUTLASS DSL version: {version!r}")
    current = tuple(int(value) for value in match.groups())
    minimum = tuple(
        int(value)
        for value in SM120_QUANTIZED_AOT_MIN_CUTLASS_DSL_VERSION.split(".")
    )
    if current < minimum:
        raise RuntimeError(
            "SM120 FP8/Sage AOT requires nvidia-cutlass-dsl>="
            f"{SM120_QUANTIZED_AOT_MIN_CUTLASS_DSL_VERSION}, got {version}"
        )


def get_sm120_aot_artifact_dir(root: Path | str, target_arch: str) -> Path:
    return Path(root) / get_host_cpu_arch() / target_arch


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_sm120_aot_source_fingerprint(repo_root: Path | None = None) -> str:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    source_paths = [
        repo_root / "bsa_attn_interface.py",
        repo_root / "bsa_sage_blk64.py",
        repo_root / "bsa_sage_quant.py",
        repo_root / "utils" / "cache_utils.py",
        *sorted((repo_root / "csrc" / "fwd" / "sm120_blk64").glob("*.py")),
        *sorted((repo_root / "csrc" / "utils").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for source_path in source_paths:
        relative_path = source_path.relative_to(repo_root).as_posix()
        content = source_path.read_bytes()
        digest.update(relative_path.encode())
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def validate_sm120_aot_manifest(manifest: dict) -> dict:
    schema_version = int(manifest.get("schema_version", -1))
    if schema_version != SM120_AOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported SM120 AOT manifest schema {schema_version}; "
            f"expected {SM120_AOT_SCHEMA_VERSION}"
        )
    if not manifest.get("target_arch"):
        raise ValueError("SM120 AOT manifest is missing target_arch")
    if not manifest.get("cpu_arch"):
        raise ValueError("SM120 AOT manifest is missing cpu_arch")
    variants = manifest.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("SM120 AOT manifest variants must be an object")
    for name, entry in variants.items():
        variant = Sm120AotVariant.from_dict(entry["config"])
        if name != variant.name:
            raise ValueError(
                f"SM120 AOT manifest variant name mismatch: {name} != {variant.name}"
            )
        for field in ("function_name", "object", "header", "shared_library", "sha256"):
            if not entry.get(field):
                raise ValueError(f"SM120 AOT variant {name} is missing {field}")
    quantization = manifest.get("quantization")
    if quantization is not None:
        if not isinstance(quantization, dict):
            raise ValueError("SM120 AOT manifest quantization must be an object")
        for field in (
            "triton_version",
            "shared_library",
            "sha256",
            "functions",
            "sources",
            "headers",
        ):
            if not quantization.get(field):
                raise ValueError(
                    f"SM120 AOT quantization entry is missing {field}"
                )
        functions = quantization["functions"]
        if not isinstance(functions, dict):
            raise ValueError("SM120 AOT quantization functions must be an object")
        for name in ("q", "stats_partial", "stats_finalize", "kv"):
            if not functions.get(name):
                raise ValueError(
                    f"SM120 AOT quantization functions are missing {name}"
                )
        for field in ("sources", "headers"):
            paths = quantization[field]
            if not isinstance(paths, list) or not all(
                isinstance(path, str) and path for path in paths
            ):
                raise ValueError(
                    f"SM120 AOT quantization {field} must be a non-empty string list"
                )
    return manifest


def load_sm120_aot_manifest(path: Path | str) -> dict:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / SM120_AOT_MANIFEST
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    return validate_sm120_aot_manifest(manifest)


def write_sm120_aot_manifest(artifact_dir: Path | str, manifest: dict) -> Path:
    validate_sm120_aot_manifest(manifest)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / SM120_AOT_MANIFEST
    temporary_path = artifact_dir / f".{SM120_AOT_MANIFEST}.{os.getpid()}.tmp"
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest_path)
    return manifest_path
