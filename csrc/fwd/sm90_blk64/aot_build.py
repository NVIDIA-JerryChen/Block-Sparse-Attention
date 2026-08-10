import argparse
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import cutlass
import cutlass.cute as cute

import block_sparse_attention.utils.cache_utils  # noqa: F401  # Preload runtime symbols for export_to_c.
from block_sparse_attention.csrc.fwd.sm100_blk64.bsa_fwd_combine import (
    BlockSparseAttnForwardCombine,
)
from block_sparse_attention.csrc.fwd.sm90_blk64.aot_utils import (
    SM90_AOT_COMBINE_K_BLOCK_SIZE,
    SM90_AOT_COMBINE_NUM_THREADS,
    SM90_AOT_COMBINE_STAGES,
    SM90_AOT_COMBINE_TILE_M,
    SM90_AOT_SCHEMA_VERSION,
    Sm90AotCombineVariant,
    Sm90AotVariant,
    compute_sm90_aot_source_fingerprint,
    get_cuda_runtime_version,
    get_host_cpu_arch,
    get_sm90_aot_artifact_dir,
    iter_sm90_aot_combine_variants,
    iter_sm90_aot_variants,
    sha256_file,
    write_sm90_aot_manifest,
)
from block_sparse_attention.csrc.fwd.sm90_blk64.bsa_fwd_sm90 import (
    BlockSparseAttnForwardSm90Blk64,
)


_DTYPES = {
    "bf16": cutlass.BFloat16,
    "fp16": cutlass.Float16,
}


def _make_dynamic_fake_tensor(dtype, rank: int, leading_dim: int, alignment):
    shape = tuple(cute.sym_int() for _ in range(rank))
    stride = tuple(
        1 if mode == leading_dim else cute.sym_int64()
        for mode in range(rank)
    )
    return cute.runtime.make_fake_tensor(
        dtype,
        shape=shape,
        stride=stride,
        assumed_align=alignment,
    )


def _compact_stride_ranks(dim_order: tuple[int, ...]) -> tuple[int, ...]:
    """Convert outer-to-inner tensor dim order to fake compact stride ranks."""
    rank = len(dim_order)
    stride_ranks = [0] * rank
    for position, mode in enumerate(dim_order):
        stride_ranks[mode] = rank - position - 1
    return tuple(stride_ranks)


def _make_dynamic_compact_fake_tensor(
    dtype,
    shape: tuple,
    dim_order: tuple[int, ...],
    alignment: int,
):
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=_compact_stride_ranks(dim_order),
        assumed_align=alignment,
    )


def make_sm90_aot_fake_args(variant: Sm90AotVariant):
    dtype = _DTYPES[variant.dtype]
    output_dtype = cutlass.Float32 if variant.kv_splits > 1 else dtype
    q = _make_dynamic_fake_tensor(dtype, 4, 1, 128)
    k = _make_dynamic_fake_tensor(dtype, 4, 1, 128)
    v = _make_dynamic_fake_tensor(dtype, 4, 0, 128)
    out = _make_dynamic_fake_tensor(output_dtype, 4, 1, 128)
    lse = _make_dynamic_fake_tensor(cutlass.Float32, 3, 0, 4)
    indices = _make_dynamic_fake_tensor(cutlass.Int32, 4, 0, None)
    block_nums = _make_dynamic_fake_tensor(cutlass.Int32, 3, 0, None)
    block_sizes = (
        _make_dynamic_fake_tensor(cutlass.Int32, 3, 0, None)
        if variant.has_block_sizes
        else block_nums
    )
    split_offsets = (
        _make_dynamic_fake_tensor(cutlass.Int32, 4, 0, None)
        if variant.kv_splits > 1
        else block_nums
    )
    return (
        q,
        k,
        v,
        out,
        lse,
        indices,
        block_nums,
        block_sizes,
        split_offsets,
        cutlass.Float32(variant.qk_dim**-0.5),
        cute.runtime.make_fake_stream(),
    )


def make_sm90_aot_combine_fake_args(variant: Sm90AotCombineVariant):
    dynamic = cute.sym_int
    o_partial = _make_dynamic_compact_fake_tensor(
        cutlass.Float32,
        (dynamic(), dynamic(), dynamic(), dynamic(), dynamic()),
        (1, 0, 3, 2, 4),
        16,
    )
    lse_partial = _make_dynamic_compact_fake_tensor(
        cutlass.Float32,
        (dynamic(), dynamic(), dynamic(), dynamic()),
        (1, 0, 3, 2),
        4,
    )
    out = _make_dynamic_compact_fake_tensor(
        _DTYPES[variant.dtype],
        (dynamic(), dynamic(), dynamic(), dynamic()),
        (0, 1, 2, 3),
        16,
    )
    lse = _make_dynamic_compact_fake_tensor(
        cutlass.Float32,
        (dynamic(), dynamic(), dynamic()),
        (0, 1, 2),
        4,
    )
    return (
        o_partial,
        lse_partial,
        out,
        lse,
        None,
        None,
        None,
        None,
        None,
        cute.runtime.make_fake_stream(),
    )


def compile_sm90_aot_variant(variant: Sm90AotVariant, target_arch: str):
    kernel = BlockSparseAttnForwardSm90Blk64(
        gqa_ratio=variant.gqa_ratio,
        head_dim=variant.qk_dim,
        value_dim=variant.value_dim,
        dtype=_DTYPES[variant.dtype],
        acc_dtype=cutlass.Float32,
        has_block_sizes=variant.has_block_sizes,
        num_splits=variant.kv_splits,
        allow_empty_block_nums=variant.allow_empty_block_nums,
    )
    return cute.compile(
        kernel,
        *make_sm90_aot_fake_args(variant),
        options=f"--gpu-arch {target_arch} --opt-level 3",
    )


def compile_sm90_aot_combine_variant(
    variant: Sm90AotCombineVariant,
    target_arch: str,
):
    kernel = BlockSparseAttnForwardCombine(
        dtype=_DTYPES[variant.dtype],
        head_dim=variant.value_dim,
        tile_m=SM90_AOT_COMBINE_TILE_M,
        k_block_size=SM90_AOT_COMBINE_K_BLOCK_SIZE,
        log_max_splits=variant.log_max_splits,
        num_threads=SM90_AOT_COMBINE_NUM_THREADS,
        stages=SM90_AOT_COMBINE_STAGES,
    )
    return cute.compile(
        kernel,
        *make_sm90_aot_combine_fake_args(variant),
        options=f"--gpu-arch {target_arch} --opt-level 3",
    )


def _export_compiled_variant(compiled, variant, artifact_dir: Path, cc: str) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    compiled.export_to_c(
        file_path=str(artifact_dir),
        file_name=variant.name,
        function_prefix=variant.name,
    )
    object_path = artifact_dir / f"{variant.name}.o"
    header_path = artifact_dir / f"{variant.name}.h"
    shared_library_path = artifact_dir / f"{variant.name}.so"
    subprocess.run(
        [cc, "-shared", "-o", str(shared_library_path), str(object_path)],
        check=True,
    )
    return {
        "config": variant.to_dict(),
        "function_name": variant.name,
        "object": object_path.name,
        "header": header_path.name,
        "shared_library": shared_library_path.name,
        "sha256": sha256_file(shared_library_path),
    }


def export_sm90_aot_variant(
    variant: Sm90AotVariant,
    target_arch: str,
    artifact_dir: Path,
    cc: str,
) -> dict:
    compiled = compile_sm90_aot_variant(variant, target_arch)
    return _export_compiled_variant(compiled, variant, artifact_dir, cc)


def export_sm90_aot_combine_variant(
    variant: Sm90AotCombineVariant,
    target_arch: str,
    artifact_dir: Path,
    cc: str,
) -> dict:
    compiled = compile_sm90_aot_combine_variant(variant, target_arch)
    return _export_compiled_variant(compiled, variant, artifact_dir, cc)


def _package_version(distribution: str, fallback: str = "unknown") -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return fallback


def build_sm90_aot_artifacts(
    output_root: Path | str,
    variants: tuple[Sm90AotVariant, ...],
    target_arch: str = "sm_90a",
    cc: str = "gcc",
) -> Path:
    if target_arch != "sm_90a":
        raise ValueError(
            f"Unsupported SM90 AOT target: {target_arch}; expected sm_90a"
        )
    if not variants:
        raise ValueError("At least one SM90 AOT variant is required")
    artifact_dir = get_sm90_aot_artifact_dir(output_root, target_arch)
    entries = {}
    for index, variant in enumerate(variants, start=1):
        print(f"[main {index}/{len(variants)}] Building {variant.name}", flush=True)
        entries[variant.name] = export_sm90_aot_variant(
            variant,
            target_arch,
            artifact_dir,
            cc,
        )

    combine_variants = iter_sm90_aot_combine_variants(variants)
    combine_entries = {}
    for index, variant in enumerate(combine_variants, start=1):
        print(
            f"[combine {index}/{len(combine_variants)}] Building {variant.name}",
            flush=True,
        )
        combine_entries[variant.name] = export_sm90_aot_combine_variant(
            variant,
            target_arch,
            artifact_dir,
            cc,
        )

    manifest = {
        "schema_version": SM90_AOT_SCHEMA_VERSION,
        "target_arch": target_arch,
        "cpu_arch": get_host_cpu_arch(),
        "cutlass_dsl_version": str(cutlass.__version__),
        "cuda_python_version": _package_version("cuda-python"),
        "cuda_runtime_version": get_cuda_runtime_version(),
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "source_fingerprint": compute_sm90_aot_source_fingerprint(),
        "variants": entries,
        "combine_variants": combine_entries,
    }
    manifest_path = write_sm90_aot_manifest(artifact_dir, manifest)
    print(f"SM90 AOT manifest: {manifest_path}", flush=True)
    return manifest_path


def _parse_csv(value: str, convert):
    items = tuple(convert(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list")
    return items


def _parse_bool_matrix(value: str) -> tuple[bool, ...]:
    if value == "both":
        return (False, True)
    return (value == "enabled",)


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SM90 blk64 CuTe DSL AOT artifacts")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agent/agent_space/sm90_aot"),
    )
    parser.add_argument("--target-arch", default="sm_90a")
    parser.add_argument("--dtypes", default="bf16,fp16")
    parser.add_argument("--qk-dims", default="128")
    parser.add_argument("--value-dims", default="128")
    parser.add_argument("--gqa-ratios", default="1,2,4,8,16,32,64")
    parser.add_argument("--kv-splits", default="1,2,4,8")
    parser.add_argument(
        "--block-sizes",
        choices=("both", "disabled", "enabled"),
        default="both",
    )
    parser.add_argument(
        "--allow-empty-block-nums",
        choices=("both", "disabled", "enabled"),
        default="both",
    )
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _create_argument_parser().parse_args(argv)
    variants = iter_sm90_aot_variants(
        _parse_csv(args.dtypes, str),
        _parse_csv(args.qk_dims, int),
        _parse_csv(args.value_dims, int),
        _parse_csv(args.gqa_ratios, int),
        _parse_bool_matrix(args.block_sizes),
        _parse_csv(args.kv_splits, int),
        _parse_bool_matrix(args.allow_empty_block_nums),
    )
    combine_variants = iter_sm90_aot_combine_variants(variants)
    if args.dry_run:
        for variant in variants:
            print(variant.name)
        for variant in combine_variants:
            print(variant.name)
        print(f"Main variants: {len(variants)}")
        print(f"Combine variants: {len(combine_variants)}")
        return 0
    build_sm90_aot_artifacts(
        args.output_dir,
        variants,
        target_arch=args.target_arch,
        cc=args.cc,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
