import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from block_sparse_attention import bsa_attn_fwd
from block_sparse_attention.csrc.fwd.sm100_blk64 import aot_runtime
from block_sparse_attention.csrc.fwd.sm100_blk64.aot_build import (
    build_sm100_aot_artifacts,
    main as aot_build_main,
)
from block_sparse_attention.csrc.fwd.sm100_blk64.aot_runtime import (
    SM100_AOT_DIR_ENV,
    SM100_AOT_ONLY_ENV,
    Sm100AotArtifactError,
    clear_sm100_aot_runtime_cache,
    get_sm100_aot_combine_kernel,
    get_sm100_aot_kernel,
    is_sm100_aot_layout_supported,
)
from block_sparse_attention.csrc.fwd.sm100_blk64.aot_utils import (
    SM100_AOT_COMBINE_FAST_NUM_THREADS,
    SM100_AOT_COMBINE_NUM_THREADS,
    SM100_AOT_SCHEMA_VERSION,
    Sm100AotCombineVariant,
    Sm100AotVariant,
    compute_sm100_aot_source_fingerprint,
    get_cuda_runtime_version,
    get_host_cpu_arch,
    get_sm100_aot_combine_variant,
    iter_sm100_aot_combine_variants,
    iter_sm100_aot_variants,
    load_sm100_aot_manifest,
    sha256_file,
    write_sm100_aot_manifest,
)


_RUNTIME_TEST_ENABLED = os.getenv("BSA_TEST_SM100_AOT_RUNTIME") == "1"
_DEVICE_ARCH = (
    sum(
        value * factor
        for value, factor in zip(torch.cuda.get_device_capability(), (10, 1))
    )
    if torch.cuda.is_available()
    else 0
)
_IS_SM100_OR_SM103 = _DEVICE_ARCH in (100, 103)


class _LayoutTensor:
    def __init__(
        self,
        shape,
        dtype,
        *,
        strides=None,
        device="cuda:0",
        data_ptr=256,
    ):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        if strides is None:
            running = 1
            values = []
            for extent in reversed(self.shape):
                values.append(running)
                running *= extent
            strides = tuple(reversed(values))
        self._strides = tuple(strides)
        self._data_ptr = int(data_ptr)

    def stride(self, dim=None):
        return self._strides if dim is None else self._strides[dim]

    def data_ptr(self):
        return self._data_ptr


def _variant(**overrides) -> Sm100AotVariant:
    values = {
        "dtype": "bf16",
        "has_block_nums": False,
        "allow_empty_block_nums": False,
        "has_block_sizes": False,
        "kv_splits": 1,
        "use_clc_scheduler": False,
        "use_int64_kv_strides": False,
    }
    values.update(overrides)
    return Sm100AotVariant(**values)


def _artifact_entry(artifact_dir: Path, variant) -> dict:
    object_path = artifact_dir / f"{variant.name}.o"
    header_path = artifact_dir / f"{variant.name}.h"
    shared_library_path = artifact_dir / f"{variant.name}.so"
    object_path.write_bytes(b"object")
    header_path.write_bytes(b"header")
    shared_library_path.write_bytes(variant.name.encode())
    return {
        "config": variant.to_dict(),
        "function_name": variant.name,
        "object": object_path.name,
        "header": header_path.name,
        "shared_library": shared_library_path.name,
        "sha256": sha256_file(shared_library_path),
    }


def _write_compatible_artifact(
    root: Path,
    target_arch: str,
    variants: tuple[Sm100AotVariant, ...],
    combine_variants: tuple[Sm100AotCombineVariant, ...] = (),
) -> Path:
    artifact_dir = root / get_host_cpu_arch() / target_arch
    artifact_dir.mkdir(parents=True)
    manifest = {
        "schema_version": SM100_AOT_SCHEMA_VERSION,
        "target_arch": target_arch,
        "cpu_arch": get_host_cpu_arch(),
        "cutlass_dsl_version": str(aot_runtime.cutlass.__version__),
        "cuda_python_version": "test",
        "cuda_runtime_version": get_cuda_runtime_version(),
        "python_abi": "test",
        "source_fingerprint": compute_sm100_aot_source_fingerprint(),
        "variants": {
            variant.name: _artifact_entry(artifact_dir, variant)
            for variant in variants
        },
        "combine_variants": {
            variant.name: _artifact_entry(artifact_dir, variant)
            for variant in combine_variants
        },
    }
    return write_sm100_aot_manifest(artifact_dir, manifest)


def _make_runtime_tensors(
    variant: Sm100AotVariant,
    *,
    device: str = "cuda:0",
):
    output_dtype = torch.float32 if variant.kv_splits > 1 else torch.bfloat16
    tensors = [
        _LayoutTensor((1, 2, 64, 128), torch.bfloat16, device=device),
        _LayoutTensor((1, 2, 256, 128), torch.bfloat16, device=device),
        _LayoutTensor((1, 2, 256, 128), torch.bfloat16, device=device),
        _LayoutTensor(
            (1, variant.kv_splits * 2, 64, 128),
            output_dtype,
            device=device,
        ),
        _LayoutTensor(
            (1, variant.kv_splits * 2, 64),
            torch.float32,
            device=device,
        ),
        _LayoutTensor((1, 2, 1, 4), torch.int32, device=device),
        (
            _LayoutTensor((4,), torch.int32, device=device)
            if variant.has_block_sizes
            else None
        ),
        (
            _LayoutTensor((1, 2, 1), torch.int32, device=device)
            if variant.has_block_nums
            else None
        ),
        (
            _LayoutTensor(
                (1, 2, 1, variant.kv_splits + 1),
                torch.int32,
                device=device,
            )
            if variant.kv_splits > 1
            and (variant.has_block_nums or variant.use_clc_scheduler)
            else None
        ),
    ]
    return tuple(tensors)


def test_sm100_aot_default_matrix_is_deterministic_and_bf16_only(capsys):
    variants = iter_sm100_aot_variants(
        (False, True),
        (False, True),
        (False, True),
        (1, 2, 4, 8),
        (False, True),
        (False, True),
    )
    assert len(variants) == 72
    assert tuple(variant.name for variant in variants) == tuple(
        sorted(variant.name for variant in variants)
    )
    assert {variant.dtype for variant in variants} == {"bf16"}
    assert all(
        variant.kv_splits == 1 or variant.allow_empty_block_nums
        for variant in variants
    )
    combine_variants = iter_sm100_aot_combine_variants(variants)
    assert len(combine_variants) == 3
    assert {variant.log_max_splits for variant in combine_variants} == {1, 2, 3}
    assert {variant.num_threads for variant in combine_variants} == {
        SM100_AOT_COMBINE_NUM_THREADS
    }

    assert (
        aot_build_main(
            [
                "--target-arch",
                "sm_100a",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Target: sm_100a" in output
    assert "Main variants: 72" in output
    assert "Combine variants: 3" in output


def test_sm100_aot_variant_round_trip_and_validation():
    variant = _variant(
        has_block_nums=True,
        allow_empty_block_nums=True,
        has_block_sizes=True,
        use_clc_scheduler=True,
        use_int64_kv_strides=True,
    )
    assert Sm100AotVariant.from_dict(variant.to_dict()) == variant
    assert variant.name == (
        "bsa_sm100_blk64_bf16_bn1_empty1_bs1_split1_clc1_i641_dyn"
    )

    split_variant = _variant(kv_splits=16, allow_empty_block_nums=True)
    combines = iter_sm100_aot_combine_variants((split_variant,))
    assert {variant.num_threads for variant in combines} == {
        SM100_AOT_COMBINE_NUM_THREADS,
        SM100_AOT_COMBINE_FAST_NUM_THREADS,
    }
    combine = get_sm100_aot_combine_variant(split_variant)
    assert Sm100AotCombineVariant.from_dict(combine.to_dict()) == combine

    with pytest.raises(ValueError, match="BF16"):
        _variant(dtype="fp8")
    with pytest.raises(ValueError, match="kv_splits"):
        _variant(kv_splits=0)
    with pytest.raises(ValueError, match="empty buckets"):
        _variant(kv_splits=2)
    with pytest.raises(ValueError, match="runtime block counts"):
        _variant(allow_empty_block_nums=True)
    with pytest.raises(ValueError, match="16 splits"):
        Sm100AotCombineVariant("bf16", 128, 3, 256)
    with pytest.raises(ValueError, match="flags must be booleans"):
        Sm100AotVariant.from_dict(
            {**variant.to_dict(), "has_block_nums": "false"}
        )


def test_sm100_aot_loader_rejects_cuda_runtime_mismatch(
    tmp_path: Path,
    monkeypatch,
):
    variant = _variant()
    _write_compatible_artifact(tmp_path, "sm_100a", (variant,))
    clear_sm100_aot_runtime_cache()
    monkeypatch.setattr(aot_runtime, "get_cuda_runtime_version", lambda: "99.0")

    with pytest.raises(Sm100AotArtifactError, match="CUDA version mismatch"):
        get_sm100_aot_kernel(
            variant,
            100,
            _make_runtime_tensors(variant),
            required=True,
            root=tmp_path,
        )
    clear_sm100_aot_runtime_cache()


def test_sm100_aot_loader_uses_local_dsl_version(tmp_path: Path, monkeypatch):
    variant = _variant()
    _write_compatible_artifact(tmp_path, "sm_100a", (variant,))
    clear_sm100_aot_runtime_cache()
    monkeypatch.setattr(aot_runtime.cutlass, "__version__", "different-local-version")

    with pytest.raises(Sm100AotArtifactError, match="CUTLASS DSL version mismatch"):
        get_sm100_aot_kernel(
            variant,
            100,
            _make_runtime_tensors(variant),
            required=True,
            root=tmp_path,
        )
    clear_sm100_aot_runtime_cache()


@pytest.mark.parametrize(
    ("device_arch", "target_arch"),
    ((100, "sm_100a"), (103, "sm_103a")),
)
def test_sm100_aot_manifest_loader_and_cache(
    tmp_path: Path,
    monkeypatch,
    device_arch: int,
    target_arch: str,
):
    variant = _variant(kv_splits=2, allow_empty_block_nums=True)
    combine = get_sm100_aot_combine_variant(variant)
    manifest_path = _write_compatible_artifact(
        tmp_path,
        target_arch,
        (variant,),
        (combine,),
    )
    manifest = load_sm100_aot_manifest(manifest_path)
    assert tuple(manifest["variants"]) == (variant.name,)
    assert tuple(manifest["combine_variants"]) == (combine.name,)

    main_kernel = object()
    main_kernel_device_1 = object()
    combine_kernel = object()
    bind_calls = []
    load_calls = []

    class FakeFunction:
        def __init__(self, executors):
            self.executors = executors

        def to(self, device_index):
            executor = self.executors[device_index]
            bind_calls.append((executor, device_index))
            return executor

    def fake_load_module(path, enable_tvm_ffi):
        load_calls.append((path, enable_tvm_ffi))
        name = variant.name if path.endswith(f"{variant.name}.so") else combine.name
        executors = (
            {0: main_kernel, 1: main_kernel_device_1}
            if name == variant.name
            else {0: combine_kernel}
        )
        function = FakeFunction(executors)
        return SimpleNamespace(**{name: function})

    clear_sm100_aot_runtime_cache()
    monkeypatch.setattr(aot_runtime.cute.runtime, "load_module", fake_load_module)
    tensors = _make_runtime_tensors(variant)
    assert get_sm100_aot_kernel(
        variant,
        device_arch,
        tensors,
        root=tmp_path,
    ) is main_kernel
    assert get_sm100_aot_kernel(
        variant,
        device_arch,
        _make_runtime_tensors(variant, device="cuda:1"),
        root=tmp_path,
    ) is main_kernel_device_1
    assert get_sm100_aot_kernel(
        variant,
        device_arch,
        tensors,
        root=tmp_path,
    ) is main_kernel
    assert get_sm100_aot_combine_kernel(
        combine,
        device_arch,
        root=tmp_path,
        device_index=0,
    ) is combine_kernel
    assert len(load_calls) == 2
    assert bind_calls == [
        (main_kernel, 0),
        (main_kernel_device_1, 1),
        (combine_kernel, 0),
    ]
    assert all(not enable_tvm_ffi for _, enable_tvm_ffi in load_calls)
    clear_sm100_aot_runtime_cache()


def test_sm100_aot_layout_checks_optional_presence_and_strides():
    variant = _variant(
        has_block_nums=True,
        has_block_sizes=True,
        kv_splits=2,
        allow_empty_block_nums=True,
    )
    tensors = list(_make_runtime_tensors(variant))
    assert is_sm100_aot_layout_supported(variant, tensors)

    tensors[6] = None
    assert not is_sm100_aot_layout_supported(variant, tensors)
    tensors = list(_make_runtime_tensors(variant))
    tensors[0] = _LayoutTensor(
        (1, 2, 64, 128),
        torch.bfloat16,
        strides=(0, 8192, 128, 1),
    )
    assert not is_sm100_aot_layout_supported(variant, tensors)

    tensors = list(_make_runtime_tensors(variant))
    tensors[1] = _LayoutTensor((1, 2, 256, 128), torch.float16)
    assert not is_sm100_aot_layout_supported(variant, tensors)

    tensors = list(_make_runtime_tensors(variant))
    tensors[5] = _LayoutTensor((1, 2, 4), torch.int32)
    assert not is_sm100_aot_layout_supported(variant, tensors)

    tensors = list(_make_runtime_tensors(variant))
    tensors[0] = _LayoutTensor(
        (1, 2, 64, 128),
        torch.bfloat16,
        device="cpu",
    )
    assert not is_sm100_aot_layout_supported(variant, tensors)


def test_sm100_aot_only_rejects_missing_or_corrupt_artifact(
    tmp_path: Path,
    monkeypatch,
):
    variant = _variant()
    with pytest.raises(Sm100AotArtifactError, match="requires"):
        get_sm100_aot_kernel(
            variant,
            100,
            _make_runtime_tensors(variant),
            required=True,
            root=tmp_path,
        )

    manifest_path = _write_compatible_artifact(
        tmp_path,
        "sm_100a",
        (variant,),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shared_library = manifest["variants"][variant.name]["shared_library"]
    (manifest_path.parent / shared_library).write_bytes(b"corrupt")
    clear_sm100_aot_runtime_cache()
    with pytest.raises(Sm100AotArtifactError, match="checksum mismatch"):
        get_sm100_aot_kernel(
            variant,
            100,
            _make_runtime_tensors(variant),
            required=True,
            root=tmp_path,
        )
    clear_sm100_aot_runtime_cache()


def test_sm100_aot_manifest_cache_observes_atomic_replacement(
    tmp_path: Path,
    monkeypatch,
):
    variant = _variant()
    manifest_path = _write_compatible_artifact(
        tmp_path,
        "sm_100a",
        (variant,),
    )
    kernel = object()
    clear_sm100_aot_runtime_cache()
    monkeypatch.setattr(
        aot_runtime,
        "_bind_kernel_cached",
        lambda *args: (None, kernel),
    )
    assert get_sm100_aot_kernel(
        variant,
        100,
        _make_runtime_tensors(variant),
        required=True,
        root=tmp_path,
    ) is kernel

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_fingerprint"] = "0" * 64
    write_sm100_aot_manifest(manifest_path.parent, manifest)
    with pytest.raises(Sm100AotArtifactError, match="source fingerprint mismatch"):
        get_sm100_aot_kernel(
            variant,
            100,
            _make_runtime_tensors(variant),
            required=True,
            root=tmp_path,
        )


@pytest.mark.skipif(
    os.getenv("BSA_TEST_SM100_AOT_BUILD") != "1",
    reason="Set BSA_TEST_SM100_AOT_BUILD=1 in the target DSL environment",
)
def test_build_sm100_aot_artifacts(tmp_path: Path):
    variant = _variant()
    target_arch = os.getenv("CUTE_DSL_ARCH", "sm_100a")
    manifest_path = build_sm100_aot_artifacts(
        tmp_path,
        (variant,),
        target_arch,
    )
    manifest = load_sm100_aot_manifest(manifest_path)
    entry = manifest["variants"][variant.name]
    shared_library = manifest_path.parent / entry["shared_library"]
    assert shared_library.is_file()
    assert sha256_file(shared_library) == entry["sha256"]


@pytest.mark.skipif(
    not _RUNTIME_TEST_ENABLED or not _IS_SM100_OR_SM103,
    reason="Set BSA_TEST_SM100_AOT_RUNTIME=1 and run on SM100 or SM103",
)
@pytest.mark.parametrize("kv_splits", (1, 2))
def test_sm100_aot_only_public_api_matches_reference(monkeypatch, kv_splits):
    artifact_root = os.getenv(SM100_AOT_DIR_ENV)
    assert artifact_root, f"{SM100_AOT_DIR_ENV} must be set"
    monkeypatch.setenv(SM100_AOT_ONLY_ENV, "1")

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run during AOT validation")

    import block_sparse_attention.bsa_attn_interface as interface

    monkeypatch.setattr(interface.cute, "compile", fail_compile)
    torch.manual_seed(20260807)
    q = torch.randn((1, 1, 64, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    indices = torch.tensor([0, 2], device="cuda", dtype=torch.int32).view(
        1, 1, 1, 2
    )
    softmax_scale = 0.17
    out, lse = bsa_attn_fwd(
        q,
        k,
        v,
        indices,
        2,
        softmax_scale=softmax_scale,
        use_clc=False,
        kv_splits=kv_splits,
        return_lse=True,
    )
    torch.cuda.synchronize()

    selected = torch.cat(
        (
            torch.arange(64, device="cuda"),
            torch.arange(128, 192, device="cuda"),
        )
    )
    selected_k = k.index_select(2, selected)
    selected_v = v.index_select(2, selected)
    scores = (
        torch.matmul(q.float(), selected_k.float().transpose(-1, -2))
        * softmax_scale
    )
    ref_out = torch.matmul(torch.softmax(scores, dim=-1), selected_v.float())
    ref_lse = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(out.float(), ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)
