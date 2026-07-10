import math
import os
from pathlib import Path
from types import SimpleNamespace

import cutlass
import pytest
import torch

import bsa_attn_interface
from bsa_attn_interface import bsa_attn_fwd_blk64
from csrc.fwd.sm120_blk64.aot_build import build_sm120_aot_artifacts
from csrc.fwd.sm120_blk64 import aot_runtime
from csrc.fwd.sm120_blk64.aot_runtime import (
    SM120_AOT_DIR_ENV,
    SM120_AOT_ONLY_ENV,
    Sm120AotArtifactError,
    clear_sm120_aot_runtime_cache,
    get_sm120_aot_kernel,
)

from csrc.fwd.sm120_blk64.aot_utils import (
    SM120_AOT_LAYOUT_MODE,
    SM120_AOT_MIN_CUTLASS_DSL_VERSION,
    SM120_AOT_SCHEMA_VERSION,
    Sm120AotVariant,
    compute_sm120_aot_source_fingerprint,
    get_cuda_runtime_version,
    get_host_cpu_arch,
    iter_sm120_aot_variants,
    load_sm120_aot_manifest,
    require_sm120_aot_cutlass_dsl_version,
    sha256_file,
    write_sm120_aot_manifest,
)


_RUNTIME_TEST_ENABLED = os.getenv("BSA_TEST_SM120_AOT_RUNTIME") == "1"
_IS_SM120 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12
_SKIP_RUNTIME_TEST = not _RUNTIME_TEST_ENABLED or not _IS_SM120


def _write_compatible_sm120_artifact(root: Path, variant: Sm120AotVariant) -> Path:
    artifact_dir = root / get_host_cpu_arch() / "sm_120f"
    artifact_dir.mkdir(parents=True)
    object_path = artifact_dir / f"{variant.name}.o"
    header_path = artifact_dir / f"{variant.name}.h"
    shared_library_path = artifact_dir / f"{variant.name}.so"
    object_path.write_bytes(b"object")
    header_path.write_bytes(b"header")
    shared_library_path.write_bytes(b"shared-library")
    manifest = {
        "schema_version": SM120_AOT_SCHEMA_VERSION,
        "target_arch": "sm_120f",
        "cpu_arch": get_host_cpu_arch(),
        "cutlass_dsl_version": str(cutlass.__version__),
        "cuda_python_version": "test",
        "cuda_runtime_version": get_cuda_runtime_version(),
        "python_abi": "test",
        "source_fingerprint": compute_sm120_aot_source_fingerprint(),
        "variants": {
            variant.name: {
                "config": variant.to_dict(),
                "function_name": variant.name,
                "object": object_path.name,
                "header": header_path.name,
                "shared_library": shared_library_path.name,
                "sha256": sha256_file(shared_library_path),
            }
        },
    }
    return write_sm120_aot_manifest(artifact_dir, manifest)


def test_sm120_aot_variant_matrix_is_deterministic():
    variants = iter_sm120_aot_variants(
        ("fp16", "bf16"),
        (2, 1, 2),
        (True, False),
        (3, 0),
    )

    assert len(variants) == 16
    assert tuple(variant.name for variant in variants) == tuple(
        sorted(variant.name for variant in variants)
    )
    assert all("split" not in variant.name for variant in variants)


def test_sm120_aot_variant_round_trip_and_validation():
    variant = Sm120AotVariant(
        dtype="bf16",
        gqa_ratio=8,
        has_block_nums=True,
        block_sizes_mode=3,
    )

    assert variant.has_block_sizes
    assert variant.layout_mode == SM120_AOT_LAYOUT_MODE
    assert Sm120AotVariant.from_dict(variant.to_dict()) == variant
    assert variant.name == "bsa_sm120_blk64_bf16_gqa8_bn1_bs3_dyn"

    with pytest.raises(ValueError, match="dtype"):
        Sm120AotVariant("fp32", 1, False, 0)
    with pytest.raises(ValueError, match="gqa_ratio"):
        Sm120AotVariant("bf16", 0, False, 0)
    with pytest.raises(ValueError, match="block_sizes_mode"):
        Sm120AotVariant("bf16", 1, False, 4)


def test_sm120_aot_requires_dynamic_layout_capable_dsl():
    assert SM120_AOT_MIN_CUTLASS_DSL_VERSION == "4.5.2"
    require_sm120_aot_cutlass_dsl_version("4.5.2")
    require_sm120_aot_cutlass_dsl_version("4.6.0.dev0")

    with pytest.raises(RuntimeError, match="nvidia-cutlass-dsl>=4.5.2"):
        require_sm120_aot_cutlass_dsl_version("4.4.2")
    with pytest.raises(RuntimeError, match="Cannot parse"):
        require_sm120_aot_cutlass_dsl_version("unknown")


def test_sm120_aot_manifest_round_trip(tmp_path: Path):
    variant = Sm120AotVariant("bf16", 1, False, 0)
    for suffix in (".o", ".h", ".so"):
        (tmp_path / f"{variant.name}{suffix}").write_bytes(suffix.encode())
    manifest = {
        "schema_version": SM120_AOT_SCHEMA_VERSION,
        "target_arch": "sm_120f",
        "cpu_arch": get_host_cpu_arch(),
        "cutlass_dsl_version": "test",
        "cuda_python_version": "test",
        "cuda_runtime_version": "test",
        "python_abi": "test",
        "source_fingerprint": "test",
        "variants": {
            variant.name: {
                "config": variant.to_dict(),
                "function_name": variant.name,
                "object": f"{variant.name}.o",
                "header": f"{variant.name}.h",
                "shared_library": f"{variant.name}.so",
                "sha256": sha256_file(tmp_path / f"{variant.name}.so"),
            }
        },
    }

    manifest_path = write_sm120_aot_manifest(tmp_path, manifest)
    loaded = load_sm120_aot_manifest(manifest_path)

    assert loaded == manifest
    assert len(compute_sm120_aot_source_fingerprint()) == 64


def test_sm120_aot_loader_verifies_and_caches_artifact(tmp_path: Path, monkeypatch):
    variant = Sm120AotVariant("bf16", 1, False, 0)
    _write_compatible_sm120_artifact(tmp_path, variant)
    kernel = object()
    load_calls = []

    def fake_load_module(path, enable_tvm_ffi):
        load_calls.append((path, enable_tvm_ffi))
        return SimpleNamespace(**{variant.name: kernel})

    clear_sm120_aot_runtime_cache()
    monkeypatch.setattr(aot_runtime.cute.runtime, "load_module", fake_load_module)
    tensor = torch.empty((2, 3))

    first = get_sm120_aot_kernel(variant, 120, (tensor,), root=tmp_path)
    second = get_sm120_aot_kernel(variant, 120, (tensor,), root=tmp_path)

    assert first is kernel
    assert second is kernel
    assert len(load_calls) == 1
    assert load_calls[0][1] is False
    clear_sm120_aot_runtime_cache()


def test_sm120_aot_only_rejects_missing_artifact(tmp_path: Path):
    variant = Sm120AotVariant("bf16", 1, False, 0)

    with pytest.raises(Sm120AotArtifactError, match="requires"):
        get_sm120_aot_kernel(
            variant,
            120,
            (torch.empty((1,)),),
            required=True,
            root=tmp_path,
        )


def test_sm120_aot_loader_rejects_checksum_mismatch(tmp_path: Path):
    variant = Sm120AotVariant("bf16", 1, False, 0)
    manifest_path = _write_compatible_sm120_artifact(tmp_path, variant)
    manifest = load_sm120_aot_manifest(manifest_path)
    shared_library = manifest["variants"][variant.name]["shared_library"]
    (manifest_path.parent / shared_library).write_bytes(b"corrupted")
    clear_sm120_aot_runtime_cache()

    with pytest.raises(Sm120AotArtifactError, match="checksum mismatch"):
        get_sm120_aot_kernel(
            variant,
            120,
            (torch.empty((1,)),),
            root=tmp_path,
        )
    clear_sm120_aot_runtime_cache()


def test_sm120_aot_only_rejects_broadcast_layout(tmp_path: Path):
    variant = Sm120AotVariant("bf16", 1, False, 0)
    broadcast = torch.empty((1, 4)).expand(3, 4)

    with pytest.raises(Sm120AotArtifactError, match="stride-0"):
        get_sm120_aot_kernel(
            variant,
            120,
            (broadcast,),
            required=True,
            root=tmp_path,
        )


def test_sm120_aot_callable_resolution_never_jits_on_hit(monkeypatch):
    variant = Sm120AotVariant("bf16", 1, False, 0)
    aot_kernel = object()
    compile_cache = {}

    monkeypatch.setattr(
        bsa_attn_interface,
        "get_sm120_aot_kernel",
        lambda *args, **kwargs: aot_kernel,
    )

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run on an AOT hit")

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fail_compile)
    resolved = bsa_attn_interface._resolve_sm120_fwd_callable(
        variant,
        120,
        (torch.empty((1,)),),
        ("compile-key",),
        object(),
        (),
        compile_cache,
    )

    assert resolved is aot_kernel
    assert not compile_cache


def test_sm120_aot_callable_resolution_caches_jit_fallback(monkeypatch):
    variant = Sm120AotVariant("bf16", 1, False, 0)
    jit_kernel = object()
    compile_calls = []
    compile_cache = {}
    monkeypatch.setattr(
        bsa_attn_interface,
        "get_sm120_aot_kernel",
        lambda *args, **kwargs: None,
    )

    def fake_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))
        return jit_kernel

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fake_compile)
    call_args = (
        variant,
        120,
        (torch.empty((1,)),),
        ("compile-key",),
        object(),
        (),
        compile_cache,
    )

    assert bsa_attn_interface._resolve_sm120_fwd_callable(*call_args) is jit_kernel
    assert bsa_attn_interface._resolve_sm120_fwd_callable(*call_args) is jit_kernel
    assert len(compile_calls) == 1


def test_sm120_aot_only_resolution_never_falls_back_to_jit(
    tmp_path: Path,
    monkeypatch,
):
    variant = Sm120AotVariant("bf16", 1, False, 0)
    monkeypatch.setenv(aot_runtime.SM120_AOT_ONLY_ENV, "1")
    monkeypatch.setenv(aot_runtime.SM120_AOT_DIR_ENV, str(tmp_path))

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run in AOT-only mode")

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fail_compile)
    with pytest.raises(Sm120AotArtifactError, match="requires"):
        bsa_attn_interface._resolve_sm120_fwd_callable(
            variant,
            120,
            (torch.empty((1,)),),
            ("compile-key",),
            object(),
            (),
            {},
        )


def test_sm120_aot_loader_rejects_dsl_version_mismatch(tmp_path: Path):
    variant = Sm120AotVariant("bf16", 1, False, 0)
    manifest_path = _write_compatible_sm120_artifact(tmp_path, variant)
    manifest = load_sm120_aot_manifest(manifest_path)
    manifest["cutlass_dsl_version"] = "incompatible"
    write_sm120_aot_manifest(manifest_path.parent, manifest)
    clear_sm120_aot_runtime_cache()

    with pytest.raises(Sm120AotArtifactError, match="version mismatch"):
        get_sm120_aot_kernel(
            variant,
            120,
            (torch.empty((1,)),),
            root=tmp_path,
        )
    clear_sm120_aot_runtime_cache()


def test_sm120_aot_loader_rejects_cuda_runtime_mismatch(
    tmp_path: Path,
    monkeypatch,
):
    variant = Sm120AotVariant("bf16", 1, False, 0)
    _write_compatible_sm120_artifact(tmp_path, variant)
    monkeypatch.setattr(aot_runtime, "get_cuda_runtime_version", lambda: "99.0")
    clear_sm120_aot_runtime_cache()

    with pytest.raises(Sm120AotArtifactError, match="CUDA version mismatch"):
        get_sm120_aot_kernel(
            variant,
            120,
            (torch.empty((1,)),),
            root=tmp_path,
        )
    clear_sm120_aot_runtime_cache()


@pytest.mark.skipif(
    os.getenv("BSA_TEST_SM120_AOT_BUILD") != "1",
    reason="Set BSA_TEST_SM120_AOT_BUILD=1 to run the SM120 cross-compile test",
)
def test_build_sm120_aot_artifact(tmp_path: Path):
    variant = Sm120AotVariant("bf16", 1, False, 0)

    manifest_path = build_sm120_aot_artifacts(
        tmp_path,
        (variant,),
        target_arch="sm_120f",
    )
    manifest = load_sm120_aot_manifest(manifest_path)
    entry = manifest["variants"][variant.name]
    artifact_dir = manifest_path.parent

    assert (artifact_dir / entry["object"]).is_file()
    assert (artifact_dir / entry["header"]).is_file()
    shared_library = artifact_dir / entry["shared_library"]
    assert shared_library.is_file()
    assert sha256_file(shared_library) == entry["sha256"]

    clear_sm120_aot_runtime_cache()
    kernel = get_sm120_aot_kernel(
        variant,
        120,
        (torch.empty((1,)),),
        required=True,
        root=tmp_path,
    )
    assert callable(kernel)
    clear_sm120_aot_runtime_cache()


def _make_runtime_block_sizes(
    mode: int,
    batch: int,
    q_heads: int,
    seqlen_k: int,
    device: torch.device,
) -> torch.Tensor:
    num_blocks = math.ceil(seqlen_k / 64)
    physical_sizes = torch.tensor(
        [min(64, seqlen_k - block * 64) for block in range(num_blocks)],
        dtype=torch.int32,
        device=device,
    )
    if mode == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    if mode == 1:
        reductions = torch.arange(num_blocks, device=device, dtype=torch.int32) % 11
        return torch.clamp(physical_sizes - reductions, min=1)
    if mode == 2:
        values = physical_sizes.unsqueeze(0).expand(batch, -1).clone()
        reductions = torch.arange(batch, device=device, dtype=torch.int32).unsqueeze(1)
        return torch.clamp(values - reductions, min=1)
    values = physical_sizes.view(1, 1, -1).expand(batch, q_heads, -1).clone()
    reductions = (
        torch.arange(batch, device=device, dtype=torch.int32).view(-1, 1, 1)
        + torch.arange(q_heads, device=device, dtype=torch.int32).view(1, -1, 1)
    ) % 7
    return torch.clamp(values - reductions, min=1)


def _runtime_block_size(
    block_sizes: torch.Tensor,
    mode: int,
    batch_idx: int,
    head_idx: int,
    physical_idx: int,
    seqlen_k: int,
) -> int:
    if mode == 0:
        return min(64, seqlen_k - physical_idx * 64)
    if mode == 1:
        return int(block_sizes[physical_idx])
    if mode == 2:
        return int(block_sizes[batch_idx, physical_idx])
    return int(block_sizes[batch_idx, head_idx, physical_idx])


def _reference_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    block_sparse_num: int,
    block_nums: torch.Tensor,
    block_sizes: torch.Tensor,
    block_sizes_mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, q_heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[2]
    gqa_ratio = q_heads // k.shape[1]
    out = torch.empty_like(q)
    lse = torch.empty(
        (batch, q_heads, seqlen_q),
        dtype=torch.float32,
        device=q.device,
    )
    scale = head_dim**-0.5
    for batch_idx in range(batch):
        for head_idx in range(q_heads):
            kv_head_idx = head_idx // gqa_ratio
            for q_block_idx in range(math.ceil(seqlen_q / 64)):
                count = (
                    int(block_nums[batch_idx, head_idx, q_block_idx])
                    if block_nums.numel()
                    else block_sparse_num
                )
                token_ids = []
                for physical_idx in indices[
                    batch_idx, head_idx, q_block_idx, :count
                ].tolist():
                    size = _runtime_block_size(
                        block_sizes,
                        block_sizes_mode,
                        batch_idx,
                        head_idx,
                        physical_idx,
                        seqlen_k,
                    )
                    token_ids.extend(
                        range(physical_idx * 64, physical_idx * 64 + size)
                    )
                token_ids_tensor = torch.tensor(
                    token_ids,
                    dtype=torch.long,
                    device=q.device,
                )
                q_start = q_block_idx * 64
                q_end = min(q_start + 64, seqlen_q)
                q_tile = q[batch_idx, head_idx, q_start:q_end].float()
                k_tile = k[batch_idx, kv_head_idx, token_ids_tensor].float()
                v_tile = v[batch_idx, kv_head_idx, token_ids_tensor].float()
                scores = torch.matmul(q_tile, k_tile.transpose(0, 1)) * scale
                probabilities = torch.softmax(scores, dim=-1)
                out[batch_idx, head_idx, q_start:q_end] = torch.matmul(
                    probabilities,
                    v_tile,
                ).to(q.dtype)
                lse[batch_idx, head_idx, q_start:q_end] = torch.logsumexp(
                    scores,
                    dim=-1,
                )
    return out, lse


def _run_sm120_aot_runtime_case(
    dtype: torch.dtype,
    batch: int,
    q_heads: int,
    kv_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    capacity: int,
    has_block_nums: bool,
    block_sizes_mode: int,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(
        batch * 1000
        + q_heads * 100
        + seqlen_q
        + capacity * 10
        + block_sizes_mode
    )
    q = torch.randn(
        (batch, q_heads, seqlen_q, 128),
        dtype=dtype,
        device=device,
    )
    k = torch.randn(
        (batch, kv_heads, seqlen_k, 128),
        dtype=dtype,
        device=device,
    )
    v = torch.randn_like(k)
    num_q_blocks = math.ceil(seqlen_q / 64)
    num_kv_blocks = math.ceil(seqlen_k / 64)
    assert capacity <= num_kv_blocks
    indices = torch.empty(
        (batch, q_heads, num_q_blocks, capacity),
        dtype=torch.int32,
        device=device,
    )
    base_indices = torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
    for batch_idx in range(batch):
        for head_idx in range(q_heads):
            for q_block_idx in range(num_q_blocks):
                shift = (batch_idx + head_idx + q_block_idx) % num_kv_blocks
                indices[batch_idx, head_idx, q_block_idx] = torch.roll(
                    base_indices,
                    shifts=shift,
                )[:capacity]
    if has_block_nums:
        block_nums = torch.empty(
            (batch, q_heads, num_q_blocks),
            dtype=torch.int32,
            device=device,
        )
        for batch_idx in range(batch):
            for head_idx in range(q_heads):
                for q_block_idx in range(num_q_blocks):
                    block_nums[batch_idx, head_idx, q_block_idx] = 1 + (
                        batch_idx + head_idx + q_block_idx
                    ) % capacity
    else:
        block_nums = torch.empty(0, dtype=torch.int32, device=device)
    block_sizes = _make_runtime_block_sizes(
        block_sizes_mode,
        batch,
        q_heads,
        seqlen_k,
        device,
    )

    out, lse = bsa_attn_fwd_blk64(
        q,
        k,
        v,
        indices,
        block_sizes,
        block_nums,
        kv_splits=1,
        block_sparse_num=capacity,
    )
    ref_out, ref_lse = _reference_sparse_attention(
        q,
        k,
        v,
        indices,
        capacity,
        block_nums,
        block_sizes,
        block_sizes_mode,
    )
    torch.testing.assert_close(out, ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=0.0, atol=2e-3)


@pytest.mark.skipif(
    _SKIP_RUNTIME_TEST,
    reason="Set BSA_TEST_SM120_AOT_RUNTIME=1 and run on SM120",
)
@pytest.mark.parametrize(
    (
        "dtype",
        "batch",
        "q_heads",
        "kv_heads",
        "seqlen_q",
        "seqlen_k",
        "capacity",
        "has_block_nums",
        "block_sizes_mode",
    ),
    [
        (torch.bfloat16, 1, 4, 2, 65, 193, 3, False, 0),
        (torch.bfloat16, 2, 8, 4, 129, 257, 4, False, 0),
        (torch.float16, 1, 2, 2, 96, 256, 4, True, 1),
        (torch.bfloat16, 2, 4, 2, 70, 193, 3, False, 2),
        (torch.float16, 2, 6, 1, 129, 257, 4, True, 3),
    ],
)
def test_sm120_aot_only_forward_matrix(
    dtype,
    batch,
    q_heads,
    kv_heads,
    seqlen_q,
    seqlen_k,
    capacity,
    has_block_nums,
    block_sizes_mode,
    monkeypatch,
):
    assert os.getenv(SM120_AOT_DIR_ENV), f"{SM120_AOT_DIR_ENV} must be set"
    monkeypatch.setenv(SM120_AOT_ONLY_ENV, "1")

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run during AOT-only validation")

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fail_compile)
    _run_sm120_aot_runtime_case(
        dtype,
        batch,
        q_heads,
        kv_heads,
        seqlen_q,
        seqlen_k,
        capacity,
        has_block_nums,
        block_sizes_mode,
    )


@pytest.mark.skipif(
    _SKIP_RUNTIME_TEST,
    reason="Set BSA_TEST_SM120_AOT_RUNTIME=1 and run on SM120",
)
def test_sm120_split_kv_is_disabled(monkeypatch):
    monkeypatch.setenv(SM120_AOT_ONLY_ENV, "1")
    q = torch.randn((1, 1, 64, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((1, 1, 64, 128), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    indices = torch.zeros((1, 1, 1, 1), dtype=torch.int32, device="cuda")
    empty = torch.empty(0, dtype=torch.int32, device="cuda")

    with pytest.raises(AssertionError, match="does not support split-KV"):
        bsa_attn_fwd_blk64(
            q,
            k,
            v,
            indices,
            empty,
            empty,
            kv_splits=2,
            block_sparse_num=1,
        )
