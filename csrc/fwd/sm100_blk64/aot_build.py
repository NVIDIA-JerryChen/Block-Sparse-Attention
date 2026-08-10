import argparse
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import cutlass
import cutlass.cute as cute

import block_sparse_attention.utils.cache_utils  # noqa: F401
from block_sparse_attention.csrc.fwd.sm100_blk64.aot_utils import (
    SM100_AOT_COMBINE_K_BLOCK_SIZE,
    SM100_AOT_COMBINE_STAGES,
    SM100_AOT_COMBINE_TILE_M,
    SM100_AOT_SCHEMA_VERSION,
    SM100_AOT_TARGET_ARCHES,
    Sm100AotCombineVariant,
    Sm100AotVariant,
    compute_sm100_aot_source_fingerprint,
    get_cuda_runtime_version,
    get_host_cpu_arch,
    get_sm100_aot_artifact_dir,
    iter_sm100_aot_combine_variants,
    iter_sm100_aot_variants,
    sha256_file,
    write_sm100_aot_manifest,
)
from block_sparse_attention.csrc.fwd.sm100_blk64.bsa_fwd_combine import (
    BlockSparseAttnForwardCombine,
)
from block_sparse_attention.csrc.fwd.sm100_blk64.bsa_fwd_sm100 import (
    BlockSparseAttnForwardSm100Blk64,
)


def _make_dynamic_fake_tensor(dtype, rank: int, alignment: int):
    shape = tuple(cute.sym_int() for _ in range(rank))
    stride = tuple(
        1 if mode == rank - 1 else cute.sym_int64()
        for mode in range(rank)
    )
    return cute.runtime.make_fake_tensor(
        dtype,
        shape=shape,
        stride=stride,
        assumed_align=alignment,
    )


def _compact_stride_ranks(dim_order: tuple[int, ...]) -> tuple[int, ...]:
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


def make_sm100_aot_fake_args(variant: Sm100AotVariant):
    output_dtype = cutlass.Float32 if variant.kv_splits > 1 else cutlass.BFloat16
    mQ = _make_dynamic_fake_tensor(cutlass.BFloat16, 4, 16)
    mK = _make_dynamic_fake_tensor(cutlass.BFloat16, 4, 16)
    mV = _make_dynamic_fake_tensor(cutlass.BFloat16, 4, 16)
    mO = _make_dynamic_fake_tensor(output_dtype, 4, 16)
    mLSE = _make_dynamic_fake_tensor(cutlass.Float32, 3, 4)
    mBlockIndex = _make_dynamic_fake_tensor(cutlass.Int32, 4, 16)
    mBlockSizes = (
        _make_dynamic_fake_tensor(cutlass.Int32, 1, 16)
        if variant.has_block_sizes
        else None
    )
    mBlockNums = (
        _make_dynamic_fake_tensor(cutlass.Int32, 3, 16)
        if variant.has_block_nums
        else None
    )
    has_split_offsets = variant.kv_splits > 1 and (
        variant.has_block_nums or variant.use_clc_scheduler
    )
    mSplitOffsets = (
        _make_dynamic_fake_tensor(cutlass.Int32, 4, 16)
        if has_split_offsets
        else None
    )
    return (
        mQ,
        mK,
        mV,
        mO,
        mLSE,
        None,
        None,
        None,
        cutlass.Float32(128**-0.5),
        mBlockIndex,
        mBlockSizes,
        cutlass.Int32(4),
        mBlockNums,
        mSplitOffsets,
        cute.runtime.make_fake_stream(),
    )


def make_sm100_aot_combine_fake_args(variant: Sm100AotCombineVariant):
    dynamic = cute.sym_int
    mO_partial = _make_dynamic_compact_fake_tensor(
        cutlass.Float32,
        (dynamic(), dynamic(), dynamic(), dynamic(), dynamic()),
        (1, 0, 3, 2, 4),
        16,
    )
    mLSE_partial = _make_dynamic_compact_fake_tensor(
        cutlass.Float32,
        (dynamic(), dynamic(), dynamic(), dynamic()),
        (1, 0, 3, 2),
        4,
    )
    mO = _make_dynamic_compact_fake_tensor(
        cutlass.BFloat16,
        (dynamic(), dynamic(), dynamic(), dynamic()),
        (0, 1, 2, 3),
        16,
    )
    mLSE = _make_dynamic_compact_fake_tensor(
        cutlass.Float32,
        (dynamic(), dynamic(), dynamic()),
        (0, 1, 2),
        4,
    )
    return (
        mO_partial,
        mLSE_partial,
        mO,
        mLSE,
        None,
        None,
        None,
        None,
        None,
        cute.runtime.make_fake_stream(),
    )


def compile_sm100_aot_variant(variant: Sm100AotVariant, target_arch: str):
    kernel = BlockSparseAttnForwardSm100Blk64(
        128,
        128,
        qhead_per_kvhead=1,
        pack_gqa=False,
        m_block_size=64,
        n_block_size=256,
        sparse_block_size=64,
        is_persistent=variant.use_clc_scheduler,
        use_clc_scheduler=variant.use_clc_scheduler,
        allow_empty_block_nums=variant.allow_empty_block_nums,
        has_block_sizes=variant.has_block_sizes,
        num_splits=variant.kv_splits,
        use_int64_kv_strides=variant.use_int64_kv_strides,
    )
    return cute.compile(
        kernel,
        *make_sm100_aot_fake_args(variant),
        options=f"--gpu-arch {target_arch} --opt-level 3",
    )


def compile_sm100_aot_combine_variant(
    variant: Sm100AotCombineVariant,
    target_arch: str,
):
    kernel = BlockSparseAttnForwardCombine(
        dtype=cutlass.BFloat16,
        head_dim=variant.value_dim,
        tile_m=SM100_AOT_COMBINE_TILE_M,
        k_block_size=SM100_AOT_COMBINE_K_BLOCK_SIZE,
        log_max_splits=variant.log_max_splits,
        num_threads=variant.num_threads,
        stages=SM100_AOT_COMBINE_STAGES,
    )
    return cute.compile(
        kernel,
        *make_sm100_aot_combine_fake_args(variant),
        options=f"--gpu-arch {target_arch} --opt-level 3",
    )


def _export_compiled_variant(
    compiled,
    variant,
    artifact_dir: Path,
    cc: str,
) -> dict:
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


def _package_version(distribution: str, fallback: str = "unknown") -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return fallback


def _require_configured_target_arch(target_arch: str) -> None:
    from cutlass.base_dsl.dsl import BaseDSL

    configured_arch = str(BaseDSL._get_dsl().envar.arch)
    if configured_arch != target_arch:
        raise RuntimeError(
            f"SM100 AOT target {target_arch} requires a fresh process with "
            f"CUTE_DSL_ARCH={target_arch}; current DSL arch is {configured_arch}"
        )


def build_sm100_aot_artifacts(
    output_root: Path | str,
    variants: tuple[Sm100AotVariant, ...],
    target_arch: str,
    cc: str = "gcc",
) -> Path:
    if target_arch not in SM100_AOT_TARGET_ARCHES:
        raise ValueError(
            f"Unsupported SM100 AOT target: {target_arch}; expected one of "
            f"{', '.join(SM100_AOT_TARGET_ARCHES)}"
        )
    if not variants:
        raise ValueError("At least one SM100 AOT variant is required")
    _require_configured_target_arch(target_arch)
    artifact_dir = get_sm100_aot_artifact_dir(output_root, target_arch)
    entries = {}
    for index, variant in enumerate(variants, start=1):
        print(
            f"[{target_arch} main {index}/{len(variants)}] Building "
            f"{variant.name}",
            flush=True,
        )
        compiled = compile_sm100_aot_variant(variant, target_arch)
        entries[variant.name] = _export_compiled_variant(
            compiled,
            variant,
            artifact_dir,
            cc,
        )

    combine_variants = iter_sm100_aot_combine_variants(variants)
    combine_entries = {}
    for index, variant in enumerate(combine_variants, start=1):
        print(
            f"[{target_arch} combine {index}/{len(combine_variants)}] Building "
            f"{variant.name}",
            flush=True,
        )
        compiled = compile_sm100_aot_combine_variant(variant, target_arch)
        combine_entries[variant.name] = _export_compiled_variant(
            compiled,
            variant,
            artifact_dir,
            cc,
        )

    manifest = {
        "schema_version": SM100_AOT_SCHEMA_VERSION,
        "target_arch": target_arch,
        "cpu_arch": get_host_cpu_arch(),
        "cutlass_dsl_version": str(cutlass.__version__),
        "cuda_python_version": _package_version("cuda-python"),
        "cuda_runtime_version": get_cuda_runtime_version(),
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "source_fingerprint": compute_sm100_aot_source_fingerprint(),
        "variants": entries,
        "combine_variants": combine_entries,
    }
    manifest_path = write_sm100_aot_manifest(artifact_dir, manifest)
    print(f"SM100 AOT manifest: {manifest_path}", flush=True)
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
    parser = argparse.ArgumentParser(
        description="Build SM100/SM103 blk64 BF16 CuTe DSL AOT artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agent/agent_space/sm100_aot"),
    )
    parser.add_argument(
        "--target-arch",
        choices=SM100_AOT_TARGET_ARCHES,
        default=os.environ.get("CUTE_DSL_ARCH", "sm_100a"),
    )
    parser.add_argument("--kv-splits", default="1,2,4,8")
    parser.add_argument(
        "--block-nums",
        choices=("both", "disabled", "enabled"),
        default="both",
    )
    parser.add_argument(
        "--allow-empty-block-nums",
        choices=("both", "disabled", "enabled"),
        default="both",
    )
    parser.add_argument(
        "--block-sizes",
        choices=("both", "disabled", "enabled"),
        default="both",
    )
    parser.add_argument(
        "--use-clc",
        choices=("both", "disabled", "enabled"),
        default="both",
    )
    parser.add_argument(
        "--int64-kv-strides",
        choices=("both", "disabled", "enabled"),
        default="both",
    )
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _create_argument_parser().parse_args(argv)
    variants = iter_sm100_aot_variants(
        _parse_bool_matrix(args.block_nums),
        _parse_bool_matrix(args.allow_empty_block_nums),
        _parse_bool_matrix(args.block_sizes),
        _parse_csv(args.kv_splits, int),
        _parse_bool_matrix(args.use_clc),
        _parse_bool_matrix(args.int64_kv_strides),
    )
    combine_variants = iter_sm100_aot_combine_variants(variants)
    if args.dry_run:
        for variant in variants:
            print(f"{args.target_arch}/{variant.name}")
        for variant in combine_variants:
            print(f"{args.target_arch}/{variant.name}")
        print(f"Target: {args.target_arch}")
        print(f"Main variants: {len(variants)}")
        print(f"Combine variants: {len(combine_variants)}")
        return 0
    build_sm100_aot_artifacts(
        args.output_dir,
        variants,
        target_arch=args.target_arch,
        cc=args.cc,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
