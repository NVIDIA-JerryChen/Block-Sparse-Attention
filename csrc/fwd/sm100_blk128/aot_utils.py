"""Native FP8 blk128 artifact identities and compatibility metadata."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from block_sparse_attention.csrc.fwd.sm100_blk64.aot_utils import (
    get_cuda_runtime_version as get_cuda_runtime_version,
    get_host_cpu_arch,
    sha256_file as sha256_file,
)

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
TARGET_ARCHES = ("sm_100a", "sm_103a")


@dataclass(frozen=True)
class Sm100Fp8Blk128AotVariant:
    has_block_nums: bool = False
    block_sizes_mode: int = 0
    kv_splits: int = 1

    def __post_init__(self):
        if type(self.has_block_nums) is not bool:
            raise ValueError("has_block_nums must be boolean")
        if type(self.block_sizes_mode) is not int or self.block_sizes_mode not in (0, 1, 2, 3):
            raise ValueError("block_sizes_mode must be 0, 1, 2 or 3")
        if type(self.kv_splits) is not int or not 1 <= self.kv_splits <= 256:
            raise ValueError("kv_splits must be an integer in [1, 256]")

    @property
    def name(self):
        return (f"bsa_fp8_blk128_bn{int(self.has_block_nums)}_"
                f"bs{self.block_sizes_mode}_split{self.kv_splits}")

    @property
    def combine_name(self):
        return f"bsa_fp8_blk128_combine_split{self.kv_splits}"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def artifact_dir(root, target_arch):
    if target_arch not in TARGET_ARCHES:
        raise ValueError(f"Unsupported FP8 blk128 AOT target: {target_arch}")
    return Path(root) / get_host_cpu_arch() / target_arch


def source_fingerprint(repo_root=None):
    """Hash the actual producer, inherited helpers, combine and native ABI."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    paths = [root / "bsa_attn_interface.py", root / "utils/cache_utils.py"]
    for directory in ("csrc/fwd/sm100_blk128", "csrc/fwd/sm100_blk64", "csrc/utils"):
        paths.extend(sorted((root / directory).glob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported FP8 blk128 AOT manifest schema")
    if manifest.get("target_arch") not in TARGET_ARCHES:
        raise ValueError("Invalid FP8 blk128 AOT target")
    for field in ("cpu_arch", "cutlass_dsl_version", "cuda_runtime_version", "source_fingerprint"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"Missing FP8 blk128 AOT manifest field: {field}")
    fingerprint = manifest["source_fingerprint"]
    if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("Invalid FP8 blk128 source fingerprint")
    for section in ("variants", "combine_variants"):
        entries = manifest.get(section)
        if not isinstance(entries, dict):
            raise ValueError(f"Invalid FP8 blk128 AOT section: {section}")
        for name, entry in entries.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid FP8 blk128 AOT entry: {name}")
            variant = Sm100Fp8Blk128AotVariant.from_dict(entry["config"])
            expected = variant.name if section == "variants" else variant.combine_name
            if name != expected or entry.get("function_name") != name:
                raise ValueError(f"FP8 blk128 AOT entry name mismatch: {name}")
            if section == "combine_variants" and variant.kv_splits <= 1:
                raise ValueError("Combine artifacts require more than one split")
            for field in ("object", "header", "shared_library"):
                filename = entry.get(field)
                if not isinstance(filename, str) or not filename or Path(filename).name != filename:
                    raise ValueError(f"Unsafe FP8 blk128 AOT {field}")
            checksum = entry.get("sha256", "")
            if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                raise ValueError("Invalid FP8 blk128 AOT checksum")
    return manifest


def write_manifest(directory, manifest):
    validate_manifest(manifest)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{MANIFEST_NAME}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    path = directory / MANIFEST_NAME
    os.replace(temporary, path)
    return path
