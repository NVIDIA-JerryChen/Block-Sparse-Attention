import os
import shutil
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CUTLASS_ROOT = PROJECT_ROOT / "third_party" / "cutlass"
SRC_DIR = Path(__file__).resolve().parent

# Kernel instantiation sources (separate TUs for register allocation isolation)
instantiation_sources = [
    str(SRC_DIR / "instantiations" / f"flash_fwd_varblk{v}_bs{b}.cu")
    for v in [0, 1] for b in [0, 1]
]

nvcc_flags = [
    "-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--use_fast_math",
    "-Wno-deprecated-declarations",
    "--threads=4",
    "-DCUDA_CTA_RECONFIG_ACTIVATED=1",
    "--ptxas-options=-v",
    "-gencode=arch=compute_100a,code=sm_100a",
] + [f"-D{d}" for d in os.environ.get("BSA_EXTRA_DEFINES", "").split(",") if d]

setup(
    name="bsa_fwd_blk64_ext",
    ext_modules=[
        CUDAExtension(
            name="bsa_fwd_blk64_ext",
            sources=[
                str(SRC_DIR / "bindings.cpp"),
                str(SRC_DIR / "flash_fwd_launch_template.cu"),
            ] + instantiation_sources,
            include_dirs=[
                str(SRC_DIR),
                str(CUTLASS_ROOT / "include"),
                str(CUTLASS_ROOT / "tools/util/include"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-Wno-deprecated-declarations"],
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=shutil.which("ninja") is not None)},
)
