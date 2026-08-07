import argparse
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import cutlass
import cutlass.cute as cute
import triton
from triton.backends import compiler as triton_backend_compiler
from triton.backends.nvidia import driver as triton_cuda_driver
from triton.tools.compile import CompileArgs, compile_kernel

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
from block_sparse_attention.csrc.fwd.sm120_blk64.bsa_fwd_sm120_sage import (
    BlockSparseAttnForwardSageSm120Blk64,
)


_DTYPES = {
    "bf16": cutlass.BFloat16,
    "fp16": cutlass.Float16,
    "fp8": cutlass.Float8E4M3FN,
    "sage": cutlass.Int8,
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
    v_dtype = cutlass.Float8E4M3FN if variant.is_sage else dtype
    v_leading_dim = 1 if variant.is_sage else 0
    v = _make_dynamic_fake_tensor(v_dtype, 4, v_leading_dim, 128)
    out_dtype = cutlass.BFloat16 if variant.is_fp8 or variant.is_sage else dtype
    out = _make_dynamic_fake_tensor(out_dtype, 4, 1, 128)
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
    if variant.is_sage:
        q_scale = _make_dynamic_fake_tensor(cutlass.Float32, 3, 0, 4)
        k_scale = _make_dynamic_fake_tensor(cutlass.Float32, 3, 0, 4)
        v_scale = _make_dynamic_fake_tensor(cutlass.Float32, 3, 0, 4)
        return (
            q,
            k,
            v,
            out,
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
    lse = _make_dynamic_fake_tensor(cutlass.Float32, 3, 0, 4)
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
    if variant.is_sage:
        kernel = BlockSparseAttnForwardSageSm120Blk64(
            gqa_ratio=variant.gqa_ratio,
            head_dim=128,
            value_dim=128,
            has_block_sizes=variant.has_block_sizes,
            has_block_nums=variant.has_block_nums,
            block_sizes_mode=variant.block_sizes_mode,
        )
        return cute.compile(
            kernel,
            *make_sm120_aot_fake_args(variant),
            options=f"--gpu-arch {target_arch} --opt-level 3",
        )
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


def _compile_sm120_triton_kernel(args: CompileArgs):
    original_target = triton_backend_compiler.GPUTarget

    def make_target(backend, arch, warp_size):
        return original_target(backend, int(arch), int(warp_size))

    triton_backend_compiler.GPUTarget = make_target
    try:
        return compile_kernel(args)
    finally:
        triton_backend_compiler.GPUTarget = original_target


def _cuda_include_dirs() -> tuple[Path, ...]:
    candidates = [Path(path) for path in triton_cuda_driver.include_dirs]
    for environment_variable in ("CUDA_HOME", "CUDA_PATH"):
        cuda_root = os.getenv(environment_variable)
        if cuda_root:
            candidates.append(Path(cuda_root) / "include")
    candidates.append(Path("/usr/local/cuda/include"))
    include_dirs = tuple(dict.fromkeys(path.resolve() for path in candidates if path.is_dir()))
    if not include_dirs:
        raise RuntimeError(
            "Cannot find cuda.h; set CUDA_HOME or CUDA_PATH before building "
            "SM120 Sage quantization AOT artifacts"
        )
    return include_dirs


def _triton_linker_flags() -> tuple[str, ...]:
    flags = []
    for library_dir in triton_cuda_driver.library_dirs():
        if Path(library_dir).is_dir():
            flags.append(f"-L{library_dir}")
    for library in triton_cuda_driver.libraries:
        if library.startswith("lib") and ".so" in library:
            flags.append(f"-l:{library}")
        else:
            flags.append(f"-l{library}")
    return tuple(flags)


def export_sm120_sage_quant_aot(
    target_arch: str,
    artifact_dir: Path,
    cc: str,
) -> dict:
    if target_arch != "sm_120f":
        raise ValueError(
            f"Unsupported Sage quantization AOT target: {target_arch}; "
            "expected sm_120f"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = Path(__file__).with_name("bsa_quant_sm120_sage.py")
    target = "cuda:120:32"
    specs = (
        (
            "q",
            "_quantize_sage_q_kernel",
            (
                "*bf16:16",
                "*i8:16",
                "*fp32:16",
                *("i64",) * 8,
                "i32",
                "i32",
                "128",
                "32",
            ),
            "((seqlen_q + 127) / 128) * 4,q_stride_b / q_stride_h,batch_size",
            8,
        ),
        (
            "stats_partial",
            "_sage_kv_stats_partial_kernel",
            (
                "*bf16:16",
                "*bf16:16",
                "*fp32:16",
                "*fp32:16",
                *("i64",) * 9,
                "i32",
                "i32",
                "i32",
                "128",
                "256",
                "32",
            ),
            "(seqlen_k + 255) / 256,4,batch_size * heads",
            8,
        ),
        (
            "stats_finalize",
            "_sage_kv_stats_finalize_kernel",
            (
                "*fp32:16",
                "*fp32:16",
                "*bf16:16",
                "*fp32:16",
                *("i64",) * 7,
                "i32",
                "i32",
                "i32",
                "i32",
                "32",
                "16",
                "2.25",
            ),
            "4,batch_size * heads,1",
            4,
        ),
        (
            "kv",
            "_quantize_sage_kv_kernel",
            (
                "*bf16:16",
                "*bf16:16",
                "*bf16:16",
                "*fp32:16",
                "*i8:16",
                "*fp8e4nv:16",
                "*fp32:16",
                *("i64",) * 18,
                "i32",
                "i32",
                "128",
                "64",
            ),
            "(seqlen_k + 63) / 64,k_stride_b / k_stride_h,batch_size",
            8,
        ),
    )
    functions = {}
    sources = []
    headers = []
    for key, kernel_name, signature, grid, num_warps in specs:
        output_name = f"bsa_sm120_sage_quant_{key}"
        function_name, output_files = _compile_sm120_triton_kernel(
            CompileArgs(
                path=str(kernel_path),
                kernel_name=kernel_name,
                signature=",".join(signature),
                grid=grid,
                target=target,
                num_warps=num_warps,
                num_stages=3,
                out_name=output_name,
                out_path=artifact_dir / output_name,
            )
        )
        functions[key] = function_name
        for output_file in output_files:
            if output_file.suffix == ".c":
                sources.append(output_file)
            elif output_file.suffix == ".h":
                headers.append(output_file)
    if len(sources) != len(specs) or len(headers) != len(specs):
        raise RuntimeError(
            "Triton AOT compiler did not produce one C source and header per "
            "Sage quantization kernel"
        )

    shared_library_path = artifact_dir / "bsa_sm120_sage_quant.so"
    command = [
        cc,
        "-shared",
        "-fPIC",
        "-O2",
        "-Wno-psabi",
        *(str(path) for path in sources),
        "-o",
        str(shared_library_path),
        *(f"-I{path}" for path in _cuda_include_dirs()),
        *_triton_linker_flags(),
    ]
    subprocess.run(command, check=True)
    return {
        "triton_version": str(triton.__version__),
        "shared_library": shared_library_path.name,
        "sha256": sha256_file(shared_library_path),
        "functions": functions,
        "sources": [path.name for path in sources],
        "headers": [path.name for path in headers],
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
    include_sage_quant: bool = True,
) -> Path:
    require_sm120_aot_cutlass_dsl_version(str(cutlass.__version__), variants)
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
    quantization = None
    if include_sage_quant:
        print("Building SM120 Sage quantization kernels", flush=True)
        quantization = export_sm120_sage_quant_aot(
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
    if quantization is not None:
        manifest["quantization"] = quantization
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
    parser.add_argument("--dtypes", default="bf16,fp16,fp8,sage")
    parser.add_argument("--gqa-ratios", default="1,2,4,8,16,32,64")
    parser.add_argument("--block-sizes-modes", default="0,1,2,3")
    parser.add_argument(
        "--block-nums",
        choices=("both", "fixed", "variable"),
        default="both",
    )
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    parser.add_argument("--no-sage-quant", action="store_true")
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
        print(f"Sage quantization: {'disabled' if args.no_sage_quant else 'enabled'}")
        return 0
    build_sm120_aot_artifacts(
        args.output_dir,
        variants,
        target_arch=args.target_arch,
        cc=args.cc,
        include_sage_quant=not args.no_sage_quant,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
