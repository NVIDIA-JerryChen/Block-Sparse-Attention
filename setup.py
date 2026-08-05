import shutil
from pathlib import Path

from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py as _BuildPy


_CSRC_EXCLUDE = (
    "fwd.sm100_blk64.cpp*",
    "fwd.sm100_blk64.instantiations*",
    "*.__pycache__*",
    "*.egg-info*",
)


class BuildPy(_BuildPy):
    """Install root API modules inside block_sparse_attention."""

    user_options = [
        *_BuildPy.user_options,
        ("dry-run", None, "show build actions without changing files"),
    ]
    boolean_options = [*_BuildPy.boolean_options, "dry-run"]

    def run(self) -> None:
        if not self.dry_run:
            build_lib = Path(self.build_lib)
            for relative_path in (
                "block_sparse_attention",
                "bsa_attn_interface.py",
                "csrc",
                "utils",
            ):
                stale_output = build_lib / relative_path
                if stale_output.is_symlink() or stale_output.is_file():
                    stale_output.unlink()
                elif stale_output.is_dir():
                    shutil.rmtree(stale_output)
        super().run()

    def find_package_modules(self, package: str, package_dir: str) -> list[tuple]:
        modules = super().find_package_modules(package, package_dir)
        if package == "block_sparse_attention":
            modules.extend(
                (
                    (package, "bsa_attn_interface", "bsa_attn_interface.py"),
                    (package, "bsa_fp8_blk64", "bsa_fp8_blk64.py"),
                    (package, "bsa_fp8_quant", "bsa_fp8_quant.py"),
                    (package, "bsa_sage_blk64", "bsa_sage_blk64.py"),
                    (package, "bsa_sage_quant", "bsa_sage_quant.py"),
                )
            )
        return modules


_csrc_packages = find_namespace_packages(
    where="csrc",
    exclude=_CSRC_EXCLUDE,
)

setup(
    packages=[
        "block_sparse_attention",
        "block_sparse_attention.csrc",
        "block_sparse_attention.utils",
        *(f"block_sparse_attention.csrc.{name}" for name in _csrc_packages),
    ],
    package_dir={
        "block_sparse_attention": "block_sparse_attention",
        "block_sparse_attention.csrc": "csrc",
        "block_sparse_attention.utils": "utils",
    },
    cmdclass={"build_py": BuildPy},
)
