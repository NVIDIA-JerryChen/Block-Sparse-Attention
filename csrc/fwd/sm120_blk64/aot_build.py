import argparse
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import cutlass
import cutlass.cute as cute

import block_sparse_attention.utils.cache_utils  # noqa: F401  # Preload runtime symbols for export_to_c.
from block_sparse_attention.csrc.fwd.sm120_blk64.aot_utils import (
    SM120_AOT_SCHEMA_VERSION,
    Sm120AotVariant,
    compute_sm120_aot_source_fingerprint,
    get_cuda_runtime_version,
    get_host_cpu_arch,
    get_sm120_aot_artifact_dir,
    iter_sm120_aot_variants,
    require_sm120_aot_cutlass_dsl_version,
    sha256_file,
    write_sm120_aot_manifest,
)
from block_sparse_attention.csrc.fwd.sm120_blk64.bsa_fwd_sm120 import (
    BlockSparseAttnForwardSm120Blk64,
)
from block_sparse_attention.csrc.fwd.sm120_blk64.bsa_fwd_sm120_fp8 import (
    BlockSparseAttnForwardFp8Sm120Blk64,
)


_DTYPES = {
    "bf16": cutlass.BFloat16,
    "fp16": cutlass.Float16,
    "fp8": cutlass.Float8E4M3FN,
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


def make_sm120_aot_fake_args(variant: Sm120AotVariant):
    dtype = _DTYPES[variant.dtype]
    q = _make_dynamic_fake_tensor(dtype, 4, 1, 128)
    k = _make_dynamic_fake_tensor(dtype, 4, 1, 128)
    v = _make_dynamic_fake_tensor(dtype, 4, 0, 128)
    out_dtype = cutlass.BFloat16 if variant.is_fp8 else dtype
    out = _make_dynamic_fake_tensor(out_dtype, 4, 1, 128)
    lse = _make_dynamic_fake_tensor(cutlass.Float32, 3, 0, 4)
    indices = _make_dynamic_fake_tensor(cutlass.Int32, 4, 0, None)
    if variant.has_block_nums:
        block_nums = _make_dynamic_fake_tensor(cutlass.Int32, 3, 0, None)
    else:
        block_nums = indices
    if variant.block_sizes_mode == 0:
        block_sizes = block_nums
    else:
        block_sizes = _make_dynamic_fake_tensor(
            cutlass.Int32,
            variant.block_sizes_mode,
            0,
            None,
        )
    if variant.is_fp8:
        q_scale = _make_dynamic_fake_tensor(cutlass.Float32, 3, 0, 4)
        k_scale = _make_dynamic_fake_tensor(cutlass.Float32, 3, 0, 4)
        v_scale = _make_dynamic_fake_tensor(cutlass.Float32, 2, 0, 4)
        return (
            q,
            k,
            v,
            out,
            lse,
            q_scale,
            k_scale,
            v_scale,
            indices,
            block_nums,
            cutlass.Int32(1),
            block_sizes,
            cutlass.Float32(128**-0.5),
            cute.runtime.make_fake_stream(),
        )
    return (
        q,
        k,
        v,
        out,
        lse,
        indices,
        block_nums,
        cutlass.Int32(1),
        block_sizes,
        cutlass.Float32(128**-0.5),
        cute.runtime.make_fake_stream(),
    )


def compile_sm120_aot_variant(variant: Sm120AotVariant, target_arch: str):
    if variant.is_fp8:
        kernel = BlockSparseAttnForwardFp8Sm120Blk64(
            gqa_ratio=variant.gqa_ratio,
            head_dim=128,
            value_dim=128,
            dtype=_DTYPES[variant.dtype],
            acc_dtype=cutlass.Float32,
            has_block_sizes=variant.has_block_sizes,
            has_block_nums=variant.has_block_nums,
            block_sizes_mode=variant.block_sizes_mode,
        )
        return cute.compile(
            kernel,
            *make_sm120_aot_fake_args(variant),
            options=f"--gpu-arch {target_arch} --opt-level 3",
        )
    kernel = BlockSparseAttnForwardSm120Blk64(
        gqa_ratio=variant.gqa_ratio,
        head_dim=128,
        value_dim=128,
        dtype=_DTYPES[variant.dtype],
        acc_dtype=cutlass.Float32,
        has_block_sizes=variant.has_block_sizes,
        has_block_nums=variant.has_block_nums,
        block_sizes_mode=variant.block_sizes_mode,
    )
    return cute.compile(
        kernel,
        *make_sm120_aot_fake_args(variant),
        options=f"--gpu-arch {target_arch} --opt-level 3",
    )


def export_sm120_aot_variant(
    variant: Sm120AotVariant,
    target_arch: str,
    artifact_dir: Path,
    cc: str,
) -> dict:
    compiled = compile_sm120_aot_variant(variant, target_arch)
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


def build_sm120_aot_artifacts(
    output_root: Path | str,
    variants: tuple[Sm120AotVariant, ...],
    target_arch: str = "sm_120f",
    cc: str = "gcc",
) -> Path:
    require_sm120_aot_cutlass_dsl_version(str(cutlass.__version__))
    if target_arch != "sm_120f":
        raise ValueError(
            f"Unsupported SM120 AOT target: {target_arch}; expected sm_120f"
        )
    if not variants:
        raise ValueError("At least one SM120 AOT variant is required")
    artifact_dir = get_sm120_aot_artifact_dir(output_root, target_arch)
    entries = {}
    for index, variant in enumerate(variants, start=1):
        print(f"[{index}/{len(variants)}] Building {variant.name}", flush=True)
        entries[variant.name] = export_sm120_aot_variant(
            variant,
            target_arch,
            artifact_dir,
            cc,
        )
    manifest = {
        "schema_version": SM120_AOT_SCHEMA_VERSION,
        "target_arch": target_arch,
        "cpu_arch": get_host_cpu_arch(),
        "cutlass_dsl_version": str(cutlass.__version__),
        "cuda_python_version": _package_version("cuda-python"),
        "cuda_runtime_version": get_cuda_runtime_version(),
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "source_fingerprint": compute_sm120_aot_source_fingerprint(),
        "variants": entries,
    }
    manifest_path = write_sm120_aot_manifest(artifact_dir, manifest)
    print(f"SM120 AOT manifest: {manifest_path}", flush=True)
    return manifest_path


def _parse_csv(value: str, convert):
    items = tuple(convert(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list")
    return items


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SM120 blk64 CuTe DSL AOT artifacts")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agent/agent_space/sm120_aot"),
    )
    parser.add_argument("--target-arch", default="sm_120f")
    parser.add_argument("--dtypes", default="bf16,fp16,fp8")
    parser.add_argument("--gqa-ratios", default="1,2,4,8,16,32,64")
    parser.add_argument("--block-sizes-modes", default="0,1,2,3")
    parser.add_argument(
        "--block-nums",
        choices=("both", "fixed", "variable"),
        default="both",
    )
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _create_argument_parser().parse_args(argv)
    dtypes = _parse_csv(args.dtypes, str)
    gqa_ratios = _parse_csv(args.gqa_ratios, int)
    block_sizes_modes = _parse_csv(args.block_sizes_modes, int)
    if args.block_nums == "both":
        has_block_nums_values = (False, True)
    else:
        has_block_nums_values = (args.block_nums == "variable",)
    variants = iter_sm120_aot_variants(
        dtypes,
        gqa_ratios,
        has_block_nums_values,
        block_sizes_modes,
    )
    if args.dry_run:
        for variant in variants:
            print(variant.name)
        print(f"Total variants: {len(variants)}")
        return 0
    build_sm120_aot_artifacts(
        args.output_dir,
        variants,
        target_arch=args.target_arch,
        cc=args.cc,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
