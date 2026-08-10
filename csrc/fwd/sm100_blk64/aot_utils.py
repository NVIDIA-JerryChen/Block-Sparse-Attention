import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from cuda.bindings import runtime as cuda_runtime


SM100_AOT_SCHEMA_VERSION = 1
SM100_AOT_MANIFEST = "manifest.json"
SM100_AOT_LAYOUT_MODE = "dynamic_strided_nonbroadcast"
SM100_AOT_TARGET_ARCHES = ("sm_100a", "sm_103a")
SM100_AOT_COMBINE_TILE_M = 16
SM100_AOT_COMBINE_K_BLOCK_SIZE = 64
SM100_AOT_COMBINE_NUM_THREADS = 128
SM100_AOT_COMBINE_FAST_NUM_THREADS = 256
SM100_AOT_COMBINE_STAGES = 4


@dataclass(frozen=True)
class Sm100AotVariant:
    dtype: str
    has_block_nums: bool
    allow_empty_block_nums: bool
    has_block_sizes: bool
    kv_splits: int
    use_clc_scheduler: bool
    use_int64_kv_strides: bool
    layout_mode: str = SM100_AOT_LAYOUT_MODE

    def __post_init__(self) -> None:
        if self.dtype != "bf16":
            raise ValueError("SM100/SM103 AOT supports BF16 only")
        if not 1 <= self.kv_splits <= 256:
            raise ValueError("kv_splits must be in [1, 256]")
        if self.kv_splits > 1 and not self.allow_empty_block_nums:
            raise ValueError("split-KV variants must allow empty buckets")
        if (
            self.kv_splits == 1
            and self.allow_empty_block_nums
            and not self.has_block_nums
        ):
            raise ValueError(
                "allow_empty_block_nums requires runtime block counts for "
                "non-split variants"
            )
        if self.layout_mode != SM100_AOT_LAYOUT_MODE:
            raise ValueError(f"Unsupported SM100 AOT layout mode: {self.layout_mode}")

    @property
    def name(self) -> str:
        return (
            f"bsa_sm100_blk64_{self.dtype}_bn{int(self.has_block_nums)}_"
            f"empty{int(self.allow_empty_block_nums)}_"
            f"bs{int(self.has_block_sizes)}_split{self.kv_splits}_"
            f"clc{int(self.use_clc_scheduler)}_"
            f"i64{int(self.use_int64_kv_strides)}_dyn"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Sm100AotVariant":
        if not isinstance(value, dict):
            raise ValueError("SM100 AOT variant config must be an object")
        bool_fields = (
            "has_block_nums",
            "allow_empty_block_nums",
            "has_block_sizes",
            "use_clc_scheduler",
            "use_int64_kv_strides",
        )
        if any(type(value.get(field)) is not bool for field in bool_fields):
            raise ValueError("SM100 AOT variant flags must be booleans")
        if type(value.get("kv_splits")) is not int:
            raise ValueError("SM100 AOT kv_splits must be an integer")
        if not isinstance(value.get("dtype"), str):
            raise ValueError("SM100 AOT dtype must be a string")
        return cls(
            dtype=value["dtype"],
            has_block_nums=value["has_block_nums"],
            allow_empty_block_nums=value["allow_empty_block_nums"],
            has_block_sizes=value["has_block_sizes"],
            kv_splits=value["kv_splits"],
            use_clc_scheduler=value["use_clc_scheduler"],
            use_int64_kv_strides=value["use_int64_kv_strides"],
            layout_mode=value.get("layout_mode", SM100_AOT_LAYOUT_MODE),
        )


def iter_sm100_aot_variants(
    has_block_nums_values: Iterable[bool],
    allow_empty_block_nums_values: Iterable[bool],
    has_block_sizes_values: Iterable[bool],
    kv_splits_values: Iterable[int],
    use_clc_scheduler_values: Iterable[bool],
    use_int64_kv_strides_values: Iterable[bool],
) -> tuple[Sm100AotVariant, ...]:
    variants = set()
    for has_block_nums in has_block_nums_values:
        for requested_allow_empty in allow_empty_block_nums_values:
            for has_block_sizes in has_block_sizes_values:
                for kv_splits in kv_splits_values:
                    allow_empty = (
                        bool(requested_allow_empty) and bool(has_block_nums)
                    ) or int(kv_splits) > 1
                    for use_clc_scheduler in use_clc_scheduler_values:
                        for use_int64_kv_strides in use_int64_kv_strides_values:
                            variants.add(
                                Sm100AotVariant(
                                    dtype="bf16",
                                    has_block_nums=bool(has_block_nums),
                                    allow_empty_block_nums=allow_empty,
                                    has_block_sizes=bool(has_block_sizes),
                                    kv_splits=int(kv_splits),
                                    use_clc_scheduler=bool(use_clc_scheduler),
                                    use_int64_kv_strides=bool(
                                        use_int64_kv_strides
                                    ),
                                )
                            )
    return tuple(sorted(variants, key=lambda variant: variant.name))


@dataclass(frozen=True)
class Sm100AotCombineVariant:
    dtype: str
    value_dim: int
    log_max_splits: int
    num_threads: int
    layout_mode: str = SM100_AOT_LAYOUT_MODE

    def __post_init__(self) -> None:
        if self.dtype != "bf16":
            raise ValueError("SM100/SM103 AOT combine supports BF16 only")
        if self.value_dim != 128:
            raise ValueError("SM100/SM103 AOT combine requires value_dim=128")
        if not 1 <= self.log_max_splits <= 8:
            raise ValueError("log_max_splits must be in [1, 8]")
        if self.num_threads not in (
            SM100_AOT_COMBINE_NUM_THREADS,
            SM100_AOT_COMBINE_FAST_NUM_THREADS,
        ):
            raise ValueError("num_threads must be 128 or 256")
        if (
            self.num_threads == SM100_AOT_COMBINE_FAST_NUM_THREADS
            and self.log_max_splits != 4
        ):
            raise ValueError("the 256-thread combine variant requires 16 splits")
        if self.layout_mode != SM100_AOT_LAYOUT_MODE:
            raise ValueError(f"Unsupported SM100 AOT layout mode: {self.layout_mode}")

    @property
    def name(self) -> str:
        return (
            f"bsa_sm100_blk64_combine_{self.dtype}_v{self.value_dim}_"
            f"logsplit{self.log_max_splits}_t{self.num_threads}_dyn"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Sm100AotCombineVariant":
        if not isinstance(value, dict):
            raise ValueError("SM100 AOT combine config must be an object")
        if not isinstance(value.get("dtype"), str):
            raise ValueError("SM100 AOT combine dtype must be a string")
        for field in ("value_dim", "log_max_splits", "num_threads"):
            if type(value.get(field)) is not int:
                raise ValueError(
                    f"SM100 AOT combine {field} must be an integer"
                )
        return cls(
            dtype=value["dtype"],
            value_dim=value["value_dim"],
            log_max_splits=value["log_max_splits"],
            num_threads=value["num_threads"],
            layout_mode=value.get("layout_mode", SM100_AOT_LAYOUT_MODE),
        )


def get_sm100_aot_combine_variant(
    variant: Sm100AotVariant,
    num_threads: int = SM100_AOT_COMBINE_NUM_THREADS,
) -> Sm100AotCombineVariant:
    if variant.kv_splits <= 1:
        raise ValueError("A combine variant requires kv_splits > 1")
    return Sm100AotCombineVariant(
        dtype=variant.dtype,
        value_dim=128,
        log_max_splits=(variant.kv_splits - 1).bit_length(),
        num_threads=int(num_threads),
    )


def iter_sm100_aot_combine_variants(
    variants: Iterable[Sm100AotVariant],
) -> tuple[Sm100AotCombineVariant, ...]:
    combine_variants = set()
    for variant in variants:
        if variant.kv_splits <= 1:
            continue
        combine_variants.add(get_sm100_aot_combine_variant(variant))
        if variant.kv_splits == 16:
            combine_variants.add(
                get_sm100_aot_combine_variant(
                    variant,
                    SM100_AOT_COMBINE_FAST_NUM_THREADS,
                )
            )
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


def get_sm100_aot_artifact_dir(root: Path | str, target_arch: str) -> Path:
    if target_arch not in SM100_AOT_TARGET_ARCHES:
        raise ValueError(f"Unsupported SM100 AOT target: {target_arch}")
    return Path(root) / get_host_cpu_arch() / target_arch


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_sm100_aot_source_fingerprint(
    repo_root: Path | None = None,
) -> str:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    source_paths = [
        repo_root / "bsa_attn_interface.py",
        repo_root / "utils" / "cache_utils.py",
        *sorted(
            (repo_root / "csrc" / "fwd" / "sm100_blk64").glob(
                "bsa_fwd*.py"
            )
        ),
        *sorted((repo_root / "csrc" / "fwd" / "sm100_blk64").glob("aot_*.py")),
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


def _validate_manifest_entries(
    entries: object,
    variant_type,
    label: str,
) -> None:
    if not isinstance(entries, dict):
        raise ValueError(f"SM100 AOT manifest {label} must be an object")
    for name, entry in entries.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError(f"SM100 AOT {label} entry {name} must be an object")
        try:
            variant = variant_type.from_dict(entry.get("config"))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"SM100 AOT {label} entry {name!r} has invalid config: {error}"
            ) from error
        if name != variant.name:
            raise ValueError(
                f"SM100 AOT {label} name mismatch: {name} != {variant.name}"
            )
        for field in (
            "function_name",
            "object",
            "header",
            "shared_library",
            "sha256",
        ):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ValueError(f"SM100 AOT {label} {name} is missing {field}")
        if entry["function_name"] != name:
            raise ValueError(
                f"SM100 AOT {label} {name} function_name mismatch: "
                f"{entry['function_name']}"
            )
        for field in ("object", "header", "shared_library"):
            if Path(entry[field]).name != entry[field]:
                raise ValueError(
                    f"SM100 AOT {label} {name} has unsafe {field} path"
                )
        checksum = entry["sha256"]
        if len(checksum) != 64 or any(
            character not in "0123456789abcdef" for character in checksum.lower()
        ):
            raise ValueError(
                f"SM100 AOT {label} {name} has invalid sha256"
            )


def validate_sm100_aot_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise ValueError("SM100 AOT manifest must be an object")
    schema_version = manifest.get("schema_version")
    if schema_version != SM100_AOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported SM100 AOT manifest schema {schema_version}; "
            f"expected {SM100_AOT_SCHEMA_VERSION}"
        )
    if manifest.get("target_arch") not in SM100_AOT_TARGET_ARCHES:
        raise ValueError(
            f"Unsupported SM100 AOT target: {manifest.get('target_arch')!r}"
        )
    if not isinstance(manifest.get("cpu_arch"), str) or not manifest["cpu_arch"]:
        raise ValueError("SM100 AOT manifest is missing cpu_arch")
    if (
        not isinstance(manifest.get("cutlass_dsl_version"), str)
        or not manifest["cutlass_dsl_version"]
    ):
        raise ValueError("SM100 AOT manifest is missing cutlass_dsl_version")
    source_fingerprint = manifest.get("source_fingerprint")
    if (
        not isinstance(source_fingerprint, str)
        or len(source_fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_fingerprint.lower()
        )
    ):
        raise ValueError("SM100 AOT manifest has invalid source_fingerprint")
    _validate_manifest_entries(
        manifest.get("variants"),
        Sm100AotVariant,
        "variants",
    )
    _validate_manifest_entries(
        manifest.get("combine_variants"),
        Sm100AotCombineVariant,
        "combine_variants",
    )
    return manifest


def load_sm100_aot_manifest(path: Path | str) -> dict:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / SM100_AOT_MANIFEST
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    return validate_sm100_aot_manifest(manifest)


def write_sm100_aot_manifest(artifact_dir: Path | str, manifest: dict) -> Path:
    validate_sm100_aot_manifest(manifest)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / SM100_AOT_MANIFEST
    temporary_path = artifact_dir / f".{SM100_AOT_MANIFEST}.{os.getpid()}.tmp"
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest_path)
    return manifest_path
