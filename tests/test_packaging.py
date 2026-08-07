from email.parser import BytesParser
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        **kwargs,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"Command failed with exit code {completed.returncode}: {command}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


@pytest.fixture(scope="module")
def packaging_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source_dir = tmp_path_factory.mktemp("source")
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    for dirname in ("block_sparse_attention", "csrc", "utils"):
        shutil.copytree(
            REPO_ROOT / dirname,
            source_dir / dirname,
            ignore=ignored,
        )
    for filename in (
        "MANIFEST.in",
        "README.md",
        "bsa_attn_interface.py",
        "bsa_fp8_blk64.py",
        "bsa_fp8_quant.py",
        "bsa_sage_blk64.py",
        "bsa_sage_quant.py",
        "pyproject.toml",
        "setup.py",
    ):
        shutil.copy2(REPO_ROOT / filename, source_dir / filename)
    return source_dir


@pytest.fixture(scope="module")
def built_wheel(
    packaging_source: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    wheel_dir = tmp_path_factory.mktemp("wheel")
    stale_outputs = (
        packaging_source / "build" / "lib" / "bsa_attn_interface.py",
        packaging_source / "build" / "lib" / "csrc" / "stale.py",
        packaging_source / "build" / "lib" / "utils" / "stale.py",
        packaging_source
        / "build"
        / "lib"
        / "block_sparse_attention"
        / "csrc"
        / "fwd"
        / "sm100_blk64"
        / "cpp"
        / "stale.txt",
    )
    for stale_output in stale_outputs:
        stale_output.parent.mkdir(parents=True, exist_ok=True)
        stale_output.write_text("stale build output")
    _run(
        [sys.executable, "setup.py", "build_py", "--dry-run"],
        cwd=packaging_source,
    )
    assert all(stale_output.is_file() for stale_output in stale_outputs)
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(packaging_source),
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
        ]
    )
    wheels = tuple(wheel_dir.glob("block_sparse_attention-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


@pytest.fixture(scope="module")
def built_sdist(
    packaging_source: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    sdist_dir = tmp_path_factory.mktemp("sdist")
    _run(
        [
            sys.executable,
            "setup.py",
            "sdist",
            "--dist-dir",
            str(sdist_dir),
        ],
        cwd=packaging_source,
    )
    sdists = tuple(sdist_dir.glob("block_sparse_attention-*.tar.gz"))
    assert len(sdists) == 1
    return sdists[0]


def test_source_tree_keeps_existing_layout() -> None:
    assert (REPO_ROOT / "bsa_attn_interface.py").is_file()
    assert (REPO_ROOT / "bsa_fp8_blk64.py").is_file()
    assert (REPO_ROOT / "bsa_fp8_quant.py").is_file()
    assert (REPO_ROOT / "bsa_sage_blk64.py").is_file()
    assert (REPO_ROOT / "bsa_sage_quant.py").is_file()
    assert (REPO_ROOT / "csrc" / "__init__.py").is_file()
    assert (REPO_ROOT / "utils" / "__init__.py").is_file()
    assert not (REPO_ROOT / "block_sparse_attention" / "bsa_attn_interface.py").exists()
    assert not (REPO_ROOT / "block_sparse_attention" / "csrc").exists()
    assert not (REPO_ROOT / "block_sparse_attention" / "utils").exists()


def test_wheel_contains_single_import_package(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as wheel:
        names = set(wheel.namelist())
        top_level_name = next(
            name for name in names if name.endswith(".dist-info/top_level.txt")
        )
        top_level = wheel.read(top_level_name).decode().splitlines()

    assert top_level == ["block_sparse_attention"]
    assert {
        "block_sparse_attention/__init__.py",
        "block_sparse_attention/bsa_attn_interface.py",
        "block_sparse_attention/bsa_fp8_blk64.py",
        "block_sparse_attention/bsa_fp8_quant.py",
        "block_sparse_attention/bsa_sage_blk64.py",
        "block_sparse_attention/bsa_sage_quant.py",
        "block_sparse_attention/csrc/fwd/sm90_blk64/aot_build.py",
        "block_sparse_attention/csrc/fwd/sm90_blk64/aot_runtime.py",
        "block_sparse_attention/csrc/fwd/sm90_blk64/aot_utils.py",
        "block_sparse_attention/csrc/fwd/sm120_blk64/quant_aot_runtime.py",
        "block_sparse_attention/csrc/fwd/sm100_blk64/cutedsl/bsa_fwd_sm100.py",
        "block_sparse_attention/utils/cache_utils.py",
    } <= names
    assert "bsa_attn_interface.py" not in names
    assert not any(name.startswith(("csrc/", "utils/")) for name in names)
    assert not any(
        name.startswith("block_sparse_attention/csrc/fwd/sm100_blk64/cpp/")
        for name in names
    )


def test_wheel_uses_local_cutedsl_version(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as wheel:
        metadata_name = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(wheel.read(metadata_name))

    assert "nvidia-cutlass-dsl" in metadata.get_all("Requires-Dist")


def test_sdist_contains_mapped_sources_without_legacy_cpp(built_sdist: Path) -> None:
    with tarfile.open(built_sdist) as sdist:
        names = {
            name.partition("/")[2]
            for name in sdist.getnames()
            if "/" in name
        }

    assert {
        "block_sparse_attention/__init__.py",
        "bsa_attn_interface.py",
        "bsa_fp8_blk64.py",
        "bsa_fp8_quant.py",
        "bsa_sage_blk64.py",
        "bsa_sage_quant.py",
        "csrc/fwd/sm90_blk64/aot_build.py",
        "csrc/fwd/sm90_blk64/aot_runtime.py",
        "csrc/fwd/sm90_blk64/aot_utils.py",
        "csrc/fwd/sm120_blk64/quant_aot_runtime.py",
        "csrc/fwd/sm100_blk64/cutedsl/bsa_fwd_sm100.py",
        "setup.py",
        "utils/cache_utils.py",
    } <= names
    assert not any(name.startswith("csrc/fwd/sm100_blk64/cpp/") for name in names)


def test_wheel_exports_public_api_from_isolated_install(
    built_wheel: Path,
    tmp_path: Path,
) -> None:
    install_dir = tmp_path / "install"
    outside_repo = tmp_path / "outside-repo"
    outside_repo.mkdir()
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_dir),
            str(built_wheel),
        ]
    )

    smoke_test = """
import importlib
import os
from pathlib import Path

import block_sparse_attention
import block_sparse_attention.bsa_attn_interface as interface
from block_sparse_attention import (
    bsa_attn_bwd,
    bsa_attn_fwd,
    bsa_fp8_blk64_fwd,
    bsa_sage_blk64_fwd,
    quantize_sage_bhsd,
    quantize_sage_kv_sm120,
    quantize_sage_q_sm120,
    quantize_sage_qkv_sm120,
    sage_sm120_kv_quant_workspace_size,
)
from block_sparse_attention.bsa_fp8_blk64 import (
    bsa_fp8_blk64_fwd as module_bsa_fp8_blk64_fwd,
)
from block_sparse_attention.csrc.fwd.sm90_blk64.aot_utils import (
    compute_sm90_aot_source_fingerprint,
)
from block_sparse_attention.csrc.fwd.sm120_blk64.aot_utils import (
    compute_sm120_aot_source_fingerprint,
)
from block_sparse_attention.utils.cache_utils import _compute_source_fingerprint

public_apis = {
    "bsa_attn_bwd": bsa_attn_bwd,
    "bsa_attn_fwd": bsa_attn_fwd,
    "bsa_fp8_blk64_fwd": bsa_fp8_blk64_fwd,
    "bsa_sage_blk64_fwd": bsa_sage_blk64_fwd,
    "quantize_sage_bhsd": quantize_sage_bhsd,
    "quantize_sage_kv_sm120": quantize_sage_kv_sm120,
    "quantize_sage_q_sm120": quantize_sage_q_sm120,
    "quantize_sage_qkv_sm120": quantize_sage_qkv_sm120,
    "sage_sm120_kv_quant_workspace_size": sage_sm120_kv_quant_workspace_size,
}
assert set(block_sparse_attention.__all__) == set(public_apis)
assert len(block_sparse_attention.__dir__()) == len(set(block_sparse_attention.__dir__()))
for name, api in public_apis.items():
    assert callable(api)
for name in ("bsa_attn_bwd", "bsa_attn_fwd", "bsa_fp8_blk64_fwd"):
    api = public_apis[name]
    assert api is getattr(interface, name)
assert module_bsa_fp8_blk64_fwd is bsa_fp8_blk64_fwd
for old_name in ("bsa_attn_fwd_blk64", "bsa_attn_fwd_blk64_cutedsl"):
    assert old_name not in block_sparse_attention.__all__
    assert not hasattr(block_sparse_attention, old_name)
    assert not hasattr(interface, old_name)

for module_name in (
    "block_sparse_attention.csrc.bwd.sm90_blk64.bsa_bwd_sm90",
    "block_sparse_attention.csrc.fwd.sm90_blk64.aot_runtime",
    "block_sparse_attention.csrc.fwd.sm100_blk128.bsa_fwd_sm100",
    "block_sparse_attention.csrc.fwd.sm120_blk64.aot_runtime",
    "block_sparse_attention.csrc.fwd.sm120_blk64.quant_aot_runtime",
    "block_sparse_attention.utils.cache_utils",
):
    importlib.import_module(module_name)

install_dir = Path(os.environ["BSA_TEST_INSTALL_DIR"]).resolve()
package_file = Path(block_sparse_attention.__file__).resolve()
assert len(_compute_source_fingerprint()) == 64
aot_fingerprint = compute_sm120_aot_source_fingerprint()
assert len(aot_fingerprint) == 64
assert compute_sm120_aot_source_fingerprint(package_file.parent) == aot_fingerprint
assert install_dir in package_file.parents
sm90_aot_fingerprint = compute_sm90_aot_source_fingerprint()
assert len(sm90_aot_fingerprint) == 64
assert (
    compute_sm90_aot_source_fingerprint(package_file.parent)
    == sm90_aot_fingerprint
)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(install_dir)
    env["BSA_TEST_INSTALL_DIR"] = str(install_dir)
    _run(
        [sys.executable, "-c", smoke_test],
        cwd=outside_repo,
        env=env,
    )

    assert not (install_dir / "bsa_attn_interface.py").exists()
    assert not (install_dir / "csrc").exists()
    assert not (install_dir / "utils").exists()


@pytest.mark.parametrize("editable_mode", ["lenient", "strict"])
def test_editable_install_exports_public_api_from_outside_repo(
    packaging_source: Path,
    editable_mode: str,
    tmp_path: Path,
) -> None:
    install_dir = tmp_path / "install"
    outside_repo = tmp_path / "outside-repo"
    outside_repo.mkdir()
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "--no-deps",
        "--target",
        str(install_dir),
        "--editable",
        str(packaging_source),
    ]
    if editable_mode == "strict":
        command.extend(["--config-settings", "editable_mode=strict"])
    _run(command)

    smoke_test = """
import os
from pathlib import Path
import site

site.addsitedir(os.environ["BSA_TEST_INSTALL_DIR"])

import block_sparse_attention
import block_sparse_attention.bsa_attn_interface as interface
from block_sparse_attention import (
    bsa_attn_bwd,
    bsa_attn_fwd,
    bsa_fp8_blk64_fwd,
    bsa_sage_blk64_fwd,
    quantize_sage_bhsd,
    quantize_sage_kv_sm120,
    quantize_sage_q_sm120,
    quantize_sage_qkv_sm120,
    sage_sm120_kv_quant_workspace_size,
)
from block_sparse_attention.bsa_fp8_blk64 import (
    bsa_fp8_blk64_fwd as module_bsa_fp8_blk64_fwd,
)
from block_sparse_attention.csrc.fwd.sm90_blk64 import aot_utils as sm90_aot_utils
from block_sparse_attention.csrc.fwd.sm120_blk64 import aot_utils as sm120_aot_utils
from block_sparse_attention.utils import cache_utils

for api in (
    bsa_attn_bwd,
    bsa_attn_fwd,
    bsa_fp8_blk64_fwd,
    bsa_sage_blk64_fwd,
    quantize_sage_bhsd,
    quantize_sage_kv_sm120,
    quantize_sage_q_sm120,
    quantize_sage_qkv_sm120,
    sage_sm120_kv_quant_workspace_size,
):
    assert callable(api)
assert module_bsa_fp8_blk64_fwd is bsa_fp8_blk64_fwd
assert set(block_sparse_attention.__all__) == {
    "bsa_attn_bwd",
    "bsa_attn_fwd",
    "bsa_fp8_blk64_fwd",
    "bsa_sage_blk64_fwd",
    "quantize_sage_bhsd",
    "quantize_sage_kv_sm120",
    "quantize_sage_q_sm120",
    "quantize_sage_qkv_sm120",
    "sage_sm120_kv_quant_workspace_size",
}
for old_name in ("bsa_attn_fwd_blk64", "bsa_attn_fwd_blk64_cutedsl"):
    assert not hasattr(block_sparse_attention, old_name)
    assert not hasattr(interface, old_name)

source_dir = Path(os.environ["BSA_TEST_SOURCE_DIR"]).resolve()
assert Path(block_sparse_attention.__file__).resolve() == (
    source_dir / "block_sparse_attention" / "__init__.py"
)
assert Path(interface.__file__).resolve() == source_dir / "bsa_attn_interface.py"
assert source_dir / "csrc" in Path(sm90_aot_utils.__file__).resolve().parents
assert source_dir / "csrc" in Path(sm120_aot_utils.__file__).resolve().parents
assert source_dir / "utils" in Path(cache_utils.__file__).resolve().parents
"""
    env = os.environ.copy()
    env["BSA_TEST_INSTALL_DIR"] = str(install_dir)
    env["BSA_TEST_SOURCE_DIR"] = str(packaging_source)
    env.pop("PYTHONPATH", None)
    _run(
        [sys.executable, "-c", smoke_test],
        cwd=outside_repo,
        env=env,
    )
