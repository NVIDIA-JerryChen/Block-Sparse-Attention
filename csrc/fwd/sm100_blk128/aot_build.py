"""Export native FP8 blk128 producers and split combines."""

import argparse
import itertools
import subprocess

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.dsl import BaseDSL

import block_sparse_attention.utils.cache_utils  # noqa: F401
from block_sparse_attention.csrc.fwd.sm100_blk128.aot_utils import (
    SCHEMA_VERSION,
    TARGET_ARCHES,
    Sm100Fp8Blk128AotVariant,
    artifact_dir,
    get_cuda_runtime_version,
    get_host_cpu_arch,
    sha256_file,
    source_fingerprint,
    write_manifest,
)
from block_sparse_attention.csrc.fwd.sm100_blk128.bsa_fwd_sm100_fp8_fused_pv import (
    BlockSparseAttnForwardFp8FusedPvSm100Blk128,
)
from block_sparse_attention.csrc.fwd.sm100_blk64.bsa_fwd_combine import (
    BlockSparseAttnForwardCombine,
)


def _compact_tensor(dtype, shape, order=None, alignment=4):
    rank = len(shape)
    order = tuple(range(rank)) if order is None else order
    ranks = tuple(rank - 1 - order.index(i) for i in range(rank))
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=ranks, assumed_align=alignment
    )


def make_fp8_blk128_aot_args(variant):
    def dynamic_tensor(dtype, rank, alignment=4):
        shape = tuple(cute.sym_int() for _ in range(rank))
        return _compact_tensor(dtype, shape, alignment=alignment)

    mQ, mK, mV = [dynamic_tensor(cutlass.Float8E4M3FN, 4, alignment=16) for _ in range(3)]
    mO = dynamic_tensor(cutlass.Float32 if variant.kv_splits > 1 else cutlass.BFloat16, 4, alignment=16)
    mLSE = dynamic_tensor(cutlass.Float32, 3) if variant.kv_splits > 1 else None
    return (
        mQ, mK, mV, mO, mLSE,
        dynamic_tensor(cutlass.Float32, 3),
        dynamic_tensor(cutlass.Float32, 3),
        dynamic_tensor(cutlass.Float32, 2),
        cutlass.Float32(128**-0.5),
        dynamic_tensor(cutlass.Int32, 4),
        dynamic_tensor(cutlass.Int32, variant.block_sizes_mode) if variant.block_sizes_mode else None,
        cutlass.Int32(4),
        dynamic_tensor(cutlass.Int32, 3) if variant.has_block_nums else None,
        cute.runtime.make_fake_stream(),
    )


def make_fp8_blk128_combine_args():
    # Native runtime tensors expose D and its dependent strides dynamically.
    # Mirror that ABI even though the public wrapper enforces D=128; baking D
    # into the fake descriptor changes which scalar layout fields are passed.
    d = cute.sym_int
    return (
        _compact_tensor(cutlass.Float32, (d(), d(), d(), d(), d()), (1, 0, 3, 2, 4), 16),
        _compact_tensor(cutlass.Float32, (d(), d(), d(), d()), (1, 0, 3, 2), 4),
        _compact_tensor(cutlass.BFloat16, (d(), d(), d(), d()), alignment=16),
        _compact_tensor(cutlass.Float32, (d(), d(), d())),
        None, None, None, None, None, cute.runtime.make_fake_stream(),
    )


def _export(compiled, variant, directory, cc, *, combine=False):
    name = variant.combine_name if combine else variant.name
    compiled.export_to_c(file_path=str(directory), file_name=name, function_prefix=name)
    library = directory / f"{name}.so"
    subprocess.run([cc, "-shared", "-o", str(library), str(directory / f"{name}.o")], check=True)
    return {
        "config": variant.to_dict(), "function_name": name,
        "object": f"{name}.o", "header": f"{name}.h", "shared_library": library.name,
        "sha256": sha256_file(library),
    }


def build_fp8_blk128_aot_artifacts(output_root, variants, target_arch, cc="gcc"):
    if target_arch not in TARGET_ARCHES:
        raise ValueError(f"Unsupported target: {target_arch}")
    configured = str(BaseDSL._get_dsl().envar.arch)
    if configured != target_arch:
        raise RuntimeError(f"Use a fresh process with CUTE_DSL_ARCH={target_arch}; current is {configured}")
    variants = tuple(sorted(set(variants), key=lambda v: v.name))
    if not variants:
        raise ValueError("At least one FP8 blk128 producer variant is required")
    directory = artifact_dir(output_root, target_arch)
    directory.mkdir(parents=True, exist_ok=True)
    producers, combines = {}, {}
    for variant in variants:
        print(f"[{target_arch}] Export {variant.name}", flush=True)
        kernel = BlockSparseAttnForwardFp8FusedPvSm100Blk128(
            128, allow_empty_block_nums=variant.has_block_nums,
            has_block_sizes=variant.block_sizes_mode != 0,
            block_sizes_mode=variant.block_sizes_mode, num_splits=variant.kv_splits,
        )
        compiled = cute.compile(
            kernel, *make_fp8_blk128_aot_args(variant),
            options=f"--gpu-arch {target_arch} --opt-level 3",
        )
        producers[variant.name] = _export(compiled, variant, directory, cc)
    for split in sorted({v.kv_splits for v in variants if v.kv_splits > 1}):
        variant = Sm100Fp8Blk128AotVariant(kv_splits=split)
        print(f"[{target_arch}] Export {variant.combine_name}", flush=True)
        kernel = BlockSparseAttnForwardCombine(
            dtype=cutlass.BFloat16, head_dim=128, tile_m=16, k_block_size=64,
            log_max_splits=(split - 1).bit_length(), num_threads=128, stages=4,
        )
        compiled = cute.compile(
            kernel, *make_fp8_blk128_combine_args(),
            options=f"--gpu-arch {target_arch} --opt-level 3",
        )
        combines[variant.combine_name] = _export(compiled, variant, directory, cc, combine=True)
    manifest = {
        "schema_version": SCHEMA_VERSION, "target_arch": target_arch,
        "cpu_arch": get_host_cpu_arch(), "cutlass_dsl_version": str(cutlass.__version__),
        "cuda_runtime_version": get_cuda_runtime_version(), "source_fingerprint": source_fingerprint(),
        "variants": producers, "combine_variants": combines,
    }
    path = write_manifest(directory, manifest)
    print(f"FP8 blk128 AOT manifest: {path}", flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-arch", choices=TARGET_ARCHES, required=True)
    parser.add_argument("--kv-splits", default="1,2,4,8")
    parser.add_argument("--block-sizes-modes", default="0,1,2,3")
    parser.add_argument("--block-nums", choices=("both", "yes", "no"), default="both")
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    counts = (False, True) if args.block_nums == "both" else (args.block_nums == "yes",)
    variants = tuple(
        Sm100Fp8Blk128AotVariant(has_block_nums=n, block_sizes_mode=m, kv_splits=s)
        for n, m, s in itertools.product(
            counts, map(int, args.block_sizes_modes.split(",")), map(int, args.kv_splits.split(","))
        )
    )
    if args.dry_run:
        for variant in variants:
            print(f"{args.target_arch}/{variant.name}")
        return
    build_fp8_blk128_aot_artifacts(args.output_dir, variants, args.target_arch, args.cc)


if __name__ == "__main__":
    main()
