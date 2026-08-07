import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from cuda.bindings import runtime as cuda_runtime


SM90_AOT_SCHEMA_VERSION = 1
SM90_AOT_MANIFEST = "manifest.json"
SM90_AOT_LAYOUT_MODE = "dynamic_strided_nonbroadcast"
SM90_AOT_COMBINE_TILE_M = 16
SM90_AOT_COMBINE_K_BLOCK_SIZE = 64
SM90_AOT_COMBINE_NUM_THREADS = 128
SM90_AOT_COMBINE_STAGES = 4


@dataclass(frozen=True)
class Sm90AotVariant:
    dtype: str
    qk_dim: int
    value_dim: int
    gqa_ratio: int
    has_block_sizes: bool
    kv_splits: int
    allow_empty_block_nums: bool
    layout_mode: str = SM90_AOT_LAYOUT_MODE

    def __post_init__(self) -> None:
        if self.dtype not in ("bf16", "fp16"):
            raise ValueError(f"Unsupported SM90 AOT dtype: {self.dtype}")
        if self.qk_dim not in (64, 96, 128):
            raise ValueError("qk_dim must be one of 64, 96, or 128")
        if self.value_dim not in (64, 96, 128):
            raise ValueError("value_dim must be one of 64, 96, or 128")
        if self.gqa_ratio < 1:
            raise ValueError("gqa_ratio must be >= 1")
        if not 1 <= self.kv_splits <= 256:
            raise ValueError("kv_splits must be in [1, 256]")
        if self.kv_splits > 1 and self.allow_empty_block_nums:
            raise ValueError(
                "allow_empty_block_nums is only valid when kv_splits is 1"
            )
        if self.layout_mode != SM90_AOT_LAYOUT_MODE:
            raise ValueError(f"Unsupported SM90 AOT layout mode: {self.layout_mode}")

    @property
    def name(self) -> str:
        block_sizes = int(self.has_block_sizes)
        allow_empty = int(self.allow_empty_block_nums)
        return (
            f"bsa_sm90_blk64_{self.dtype}_qk{self.qk_dim}_v{self.value_dim}_"
            f"gqa{self.gqa_ratio}_bs{block_sizes}_split{self.kv_splits}_"
            f"empty{allow_empty}_dyn"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Sm90AotVariant":
        return cls(
            dtype=str(value["dtype"]),
            qk_dim=int(value["qk_dim"]),
            value_dim=int(value["value_dim"]),
            gqa_ratio=int(value["gqa_ratio"]),
            has_block_sizes=bool(value["has_block_sizes"]),
            kv_splits=int(value["kv_splits"]),
            allow_empty_block_nums=bool(value["allow_empty_block_nums"]),
            layout_mode=str(value.get("layout_mode", SM90_AOT_LAYOUT_MODE)),
        )


def iter_sm90_aot_variants(
    dtypes: Iterable[str],
    qk_dims: Iterable[int],
    value_dims: Iterable[int],
    gqa_ratios: Iterable[int],
    has_block_sizes_values: Iterable[bool],
    kv_splits_values: Iterable[int],
    allow_empty_block_nums_values: Iterable[bool],
) -> tuple[Sm90AotVariant, ...]:
    variants = {
        Sm90AotVariant(
            dtype,
            qk_dim,
            value_dim,
            gqa_ratio,
            has_block_sizes,
            kv_splits,
            allow_empty_block_nums,
        )
        for dtype in dtypes
        for qk_dim in qk_dims
        for value_dim in value_dims
        for gqa_ratio in gqa_ratios
        for has_block_sizes in has_block_sizes_values
        for kv_splits in kv_splits_values
        for allow_empty_block_nums in allow_empty_block_nums_values
        if kv_splits == 1 or not allow_empty_block_nums
    }
    return tuple(sorted(variants, key=lambda variant: variant.name))


@dataclass(frozen=True)
class Sm90AotCombineVariant:
    dtype: str
    value_dim: int
    log_max_splits: int
    layout_mode: str = SM90_AOT_LAYOUT_MODE

    def __post_init__(self) -> None:
        if self.dtype not in ("bf16", "fp16"):
            raise ValueError(f"Unsupported SM90 AOT combine dtype: {self.dtype}")
        if self.value_dim not in (64, 96, 128):
            raise ValueError("value_dim must be one of 64, 96, or 128")
        if not 1 <= self.log_max_splits <= 8:
            raise ValueError("log_max_splits must be in [1, 8]")
        if self.layout_mode != SM90_AOT_LAYOUT_MODE:
            raise ValueError(f"Unsupported SM90 AOT layout mode: {self.layout_mode}")

    @property
    def name(self) -> str:
        return (
            f"bsa_sm90_blk64_combine_{self.dtype}_v{self.value_dim}_"
            f"logsplit{self.log_max_splits}_dyn"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Sm90AotCombineVariant":
        return cls(
            dtype=str(value["dtype"]),
            value_dim=int(value["value_dim"]),
            log_max_splits=int(value["log_max_splits"]),
            layout_mode=str(value.get("layout_mode", SM90_AOT_LAYOUT_MODE)),
        )


def get_sm90_aot_combine_variant(variant: Sm90AotVariant) -> Sm90AotCombineVariant:
    if variant.kv_splits <= 1:
        raise ValueError("A combine variant requires kv_splits > 1")
    return Sm90AotCombineVariant(
        dtype=variant.dtype,
        value_dim=variant.value_dim,
        log_max_splits=(variant.kv_splits - 1).bit_length(),
    )


def iter_sm90_aot_combine_variants(
    variants: Iterable[Sm90AotVariant],
) -> tuple[Sm90AotCombineVariant, ...]:
    combine_variants = {
        get_sm90_aot_combine_variant(variant)
        for variant in variants
        if variant.kv_splits > 1
    }
    return tuple(sorted(combine_variants, key=lambda variant: variant.name))


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


def get_sm90_aot_artifact_dir(root: Path | str, target_arch: str) -> Path:
    return Path(root) / get_host_cpu_arch() / target_arch


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_sm90_aot_source_fingerprint(repo_root: Path | None = None) -> str:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    source_paths = [
        repo_root / "bsa_attn_interface.py",
        repo_root / "utils" / "cache_utils.py",
        repo_root
        / "csrc"
        / "fwd"
        / "sm100_blk64"
        / "cutedsl"
        / "bsa_fwd_combine.py",
        *sorted((repo_root / "csrc" / "fwd" / "sm90_blk64").glob("*.py")),
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


def validate_sm90_aot_manifest(manifest: dict) -> dict:
    schema_version = int(manifest.get("schema_version", -1))
    if schema_version != SM90_AOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported SM90 AOT manifest schema {schema_version}; "
            f"expected {SM90_AOT_SCHEMA_VERSION}"
        )
    if not manifest.get("target_arch"):
        raise ValueError("SM90 AOT manifest is missing target_arch")
    if not manifest.get("cpu_arch"):
        raise ValueError("SM90 AOT manifest is missing cpu_arch")
    variants = manifest.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("SM90 AOT manifest variants must be an object")
    for name, entry in variants.items():
        variant = Sm90AotVariant.from_dict(entry["config"])
        if name != variant.name:
            raise ValueError(
                f"SM90 AOT manifest variant name mismatch: {name} != {variant.name}"
            )
        for field in ("function_name", "object", "header", "shared_library", "sha256"):
            if not entry.get(field):
                raise ValueError(f"SM90 AOT variant {name} is missing {field}")
        if entry["function_name"] != name:
            raise ValueError(
                f"SM90 AOT variant {name} function_name mismatch: "
                f"{entry['function_name']}"
            )
    combine_variants = manifest.get("combine_variants")
    if not isinstance(combine_variants, dict):
        raise ValueError("SM90 AOT manifest combine_variants must be an object")
    for name, entry in combine_variants.items():
        variant = Sm90AotCombineVariant.from_dict(entry["config"])
        if name != variant.name:
            raise ValueError(
                "SM90 AOT manifest combine variant name mismatch: "
                f"{name} != {variant.name}"
            )
        for field in ("function_name", "object", "header", "shared_library", "sha256"):
            if not entry.get(field):
                raise ValueError(f"SM90 AOT combine variant {name} is missing {field}")
        if entry["function_name"] != name:
            raise ValueError(
                f"SM90 AOT combine variant {name} function_name mismatch: "
                f"{entry['function_name']}"
            )
    return manifest


def load_sm90_aot_manifest(path: Path | str) -> dict:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / SM90_AOT_MANIFEST
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    return validate_sm90_aot_manifest(manifest)


def write_sm90_aot_manifest(artifact_dir: Path | str, manifest: dict) -> Path:
    validate_sm90_aot_manifest(manifest)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / SM90_AOT_MANIFEST
    temporary_path = artifact_dir / f".{SM90_AOT_MANIFEST}.{os.getpid()}.tmp"
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest_path)
    return manifest_path
