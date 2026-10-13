import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from block_sparse_attention import bsa_attn_fwd, bsa_fp8_blk64_fwd
from block_sparse_attention.bsa_fp8_quant import quantize_sage_bhsd
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
    get_sm100_aot_manifest_token,
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
    iter_sm100_fp8_aot_variants,
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
        "block_sizes_mode": 0,
        "use_sage_ldred_rowmax": False,
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
    input_dtype = torch.float8_e4m3fn if variant.is_fp8 else torch.bfloat16
    output_dtype = torch.float32 if variant.kv_splits > 1 else torch.bfloat16
    block_sizes_shapes = {
        1: (4,),
        2: (1, 4),
        3: (1, 2, 4),
    }
    tensors = [
        _LayoutTensor((1, 2, 64, 128), input_dtype, device=device),
        _LayoutTensor((1, 2, 256, 128), input_dtype, device=device),
        _LayoutTensor((1, 2, 256, 128), input_dtype, device=device),
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
        (
            _LayoutTensor((1, 2, 64), torch.float32, device=device)
            if variant.is_fp8
            else None
        ),
        (
            _LayoutTensor((1, 2, 16), torch.float32, device=device)
            if variant.is_fp8
            else None
        ),
        (
            _LayoutTensor((2, 128), torch.float32, device=device)
            if variant.is_fp8
            else None
        ),
        _LayoutTensor((1, 2, 1, 4), torch.int32, device=device),
        (
            _LayoutTensor(
                block_sizes_shapes[variant.block_sizes_mode],
                torch.int32,
                device=device,
            )
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


def test_sm100_aot_default_matrix_is_deterministic_and_includes_fp8(capsys):
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

    fp8_variants = iter_sm100_fp8_aot_variants(
        (False, True),
        (0, 1, 2, 3),
        (1, 4, 8, 16),
    )
    assert len(fp8_variants) == 32
    assert {variant.dtype for variant in fp8_variants} == {"fp8"}
    assert {variant.block_sizes_mode for variant in fp8_variants} == {
        0,
        1,
        2,
        3,
    }
    assert all(not variant.use_clc_scheduler for variant in fp8_variants)
    assert all(not variant.use_int64_kv_strides for variant in fp8_variants)

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
    assert "Main variants: 104" in output
    assert "Dtype variants: bf16=72, fp8=32" in output
    assert "Combine variants: 5" in output

    assert (
        aot_build_main(
            [
                "--target-arch",
                "sm_103a",
                "--dtypes",
                "fp8",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Main variants: 40" in output
    assert "Dtype variants: fp8=40" in output


def test_sm100_aot_variant_round_trip_and_validation():
    variant = _variant(
        has_block_nums=True,
        allow_empty_block_nums=True,
        has_block_sizes=True,
        block_sizes_mode=1,
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

    fp8_variant = _variant(
        dtype="fp8",
        has_block_sizes=True,
        block_sizes_mode=3,
        use_sage_ldred_rowmax=True,
    )
    assert Sm100AotVariant.from_dict(fp8_variant.to_dict()) == fp8_variant
    assert fp8_variant.name == (
        "bsa_sm100_blk64_fp8_bn0_empty0_bs3_split1_clc0_i640_"
        "ldred1_dyn"
    )

    with pytest.raises(ValueError, match="Unsupported"):
        _variant(dtype="fp16")
    with pytest.raises(ValueError, match="block_sizes_mode"):
        _variant(has_block_sizes=True, block_sizes_mode=0)
    with pytest.raises(ValueError, match="FP8"):
        _variant(use_sage_ldred_rowmax=True)
    with pytest.raises(ValueError, match="unsplit"):
        _variant(
            dtype="fp8",
            kv_splits=2,
            allow_empty_block_nums=True,
            use_sage_ldred_rowmax=True,
        )
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


def test_sm100_aot_manifest_token_changes_when_cache_is_cleared(
    tmp_path: Path,
    monkeypatch,
):
    artifact_dir = tmp_path / get_host_cpu_arch() / "sm_100a"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv(SM100_AOT_DIR_ENV, str(tmp_path))

    token_before = get_sm100_aot_manifest_token(100)
    clear_sm100_aot_runtime_cache()
    token_after = get_sm100_aot_manifest_token(100)

    assert token_before is not None
    assert token_after is not None
    assert token_before[:-1] == token_after[:-1]
    assert token_before[-1] != token_after[-1]


def test_sm100_aot_layout_checks_optional_presence_and_strides():
    variant = _variant(
        has_block_nums=True,
        has_block_sizes=True,
        block_sizes_mode=2,
        kv_splits=2,
        allow_empty_block_nums=True,
    )
    tensors = list(_make_runtime_tensors(variant))
    assert is_sm100_aot_layout_supported(variant, tensors)

    tensors[9] = None
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
    tensors[8] = _LayoutTensor((1, 2, 4), torch.int32)
    assert not is_sm100_aot_layout_supported(variant, tensors)

    tensors = list(_make_runtime_tensors(variant))
    tensors[0] = _LayoutTensor(
        (1, 2, 64, 128),
        torch.bfloat16,
        device="cpu",
    )
    assert not is_sm100_aot_layout_supported(variant, tensors)

    fp8_variant = _variant(dtype="fp8")
    tensors = list(_make_runtime_tensors(fp8_variant))
    assert is_sm100_aot_layout_supported(fp8_variant, tensors)
    tensors[5] = None
    assert not is_sm100_aot_layout_supported(fp8_variant, tensors)


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
@pytest.mark.parametrize("dtype", ("bf16", "fp8"))
def test_build_sm100_aot_artifacts(tmp_path: Path, dtype: str):
    variant = _variant(dtype=dtype)
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


@pytest.mark.skipif(
    not _RUNTIME_TEST_ENABLED or not _IS_SM100_OR_SM103,
    reason="Set BSA_TEST_SM100_AOT_RUNTIME=1 and run on SM100 or SM103",
)
@pytest.mark.parametrize("seqlen_q,seqlen_k", ((65, 127), (128, 128)))
def test_sm100_aot_only_fp8_public_api_matches_reference(
    monkeypatch,
    seqlen_q: int,
    seqlen_k: int,
):
    artifact_root = os.getenv(SM100_AOT_DIR_ENV)
    assert artifact_root, f"{SM100_AOT_DIR_ENV} must be set"
    monkeypatch.setenv(SM100_AOT_ONLY_ENV, "1")

    import block_sparse_attention.bsa_attn_interface as interface

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run during FP8 AOT validation")

    monkeypatch.setattr(interface.cute, "compile", fail_compile)
    torch.manual_seed(20260909)
    batch, heads = 1, 2
    q = torch.randn(
        (batch, heads, seqlen_q, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (batch, heads, seqlen_k, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale = quantize_sage_bhsd(
        q,
        k,
        v,
    )
    indices = torch.tensor(
        [[[[0, 1], [1, 0]], [[1, 0], [0, 1]]]],
        dtype=torch.int32,
        device="cuda",
    )
    block_nums = torch.tensor(
        [[[2, 1], [1, 2]]],
        dtype=torch.int32,
        device="cuda",
    )
    block_sizes = torch.tensor(
        [[[64, 64], [64, 47]]],
        dtype=torch.int32,
        device="cuda",
    )
    softmax_scale = 128**-0.5
    out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        indices,
        0,
        softmax_scale,
        block_sizes=block_sizes,
        q2k_block_nums=block_nums,
    )
    torch.cuda.synchronize()

    q_dequant = q_fp8.float() * q_scale.unsqueeze(-1)
    k_dequant = k_fp8.float() * k_scale.repeat_interleave(
        16,
        dim=-1,
    )[..., :seqlen_k].unsqueeze(-1)
    v_dequant = v_fp8.float() * v_scale.view(1, heads, 1, 128)
    reference = torch.empty_like(out, dtype=torch.float32)
    for head_idx in range(heads):
        for q_block_idx in range(2):
            count = int(block_nums[0, head_idx, q_block_idx])
            token_ids = []
            for block_idx in indices[
                0,
                head_idx,
                q_block_idx,
                :count,
            ].tolist():
                valid_tokens = min(
                    int(block_sizes[0, head_idx, block_idx]),
                    seqlen_k - block_idx * 64,
                )
                token_ids.extend(
                    range(block_idx * 64, block_idx * 64 + valid_tokens)
                )
            selected = torch.tensor(token_ids, device="cuda")
            q_start = q_block_idx * 64
            q_stop = min(q_start + 64, seqlen_q)
            q_tile = q_dequant[0, head_idx, q_start:q_stop]
            k_tile = k_dequant[0, head_idx].index_select(0, selected)
            v_tile = v_dequant[0, head_idx].index_select(0, selected)
            scores = q_tile @ k_tile.transpose(0, 1) * softmax_scale
            reference[0, head_idx, q_start:q_stop] = (
                torch.softmax(scores, dim=-1) @ v_tile
            )

    diff = (out.float() - reference).abs()
    assert diff.max().item() < 0.15
    assert (diff.mean() / reference.abs().mean()).item() < 0.035


@pytest.mark.skipif(
    not _RUNTIME_TEST_ENABLED or not _IS_SM100_OR_SM103,
    reason="Set BSA_TEST_SM100_AOT_RUNTIME=1 and run on SM100 or SM103",
)
def test_sm100_aot_only_fp8_split_matches_reference(monkeypatch):
    artifact_root = os.getenv(SM100_AOT_DIR_ENV)
    assert artifact_root, f"{SM100_AOT_DIR_ENV} must be set"
    monkeypatch.setenv(SM100_AOT_ONLY_ENV, "1")

    import block_sparse_attention.bsa_attn_interface as interface

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run during FP8 AOT validation")

    monkeypatch.setattr(interface.cute, "compile", fail_compile)
    # Keep this ABI test independent of unrelated jobs consuming most of a
    # shared test GPU. The case itself needs less than one MiB of workspace.
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device=None: (2 << 30, 8 << 30),
    )
    torch.manual_seed(20260910)
    seqlen_q, seqlen_k = 65, 70
    q = torch.randn(
        (1, 2, seqlen_q, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (1, 2, seqlen_k, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale = quantize_sage_bhsd(
        q,
        k,
        v,
    )
    indices = torch.tensor(
        [[[[0, 1], [1, 0]], [[1, 0], [0, 1]]]],
        dtype=torch.int32,
        device="cuda",
    )
    softmax_scale = 128**-0.5
    out, _ = interface._bsa_attn_fwd_sm100_blk64(
        q_fp8,
        k_fp8,
        v_fp8,
        indices,
        None,
        block_sparse_num=2,
        softmax_scale=softmax_scale,
        use_clc=False,
        kv_splits=4,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()

    q_dequant = q_fp8.float() * q_scale.unsqueeze(-1)
    k_dequant = k_fp8.float() * k_scale.repeat_interleave(
        16,
        dim=-1,
    )[..., :seqlen_k].unsqueeze(-1)
    v_dequant = v_fp8.float() * v_scale.view(1, 2, 1, 128)
    reference = torch.empty_like(out, dtype=torch.float32)
    for head_idx in range(2):
        for q_block_idx in range(2):
            token_ids = []
            for block_idx in indices[0, head_idx, q_block_idx].tolist():
                block_stop = min((block_idx + 1) * 64, seqlen_k)
                token_ids.extend(range(block_idx * 64, block_stop))
            selected = torch.tensor(token_ids, device="cuda")
            q_start = q_block_idx * 64
            q_stop = min(q_start + 64, seqlen_q)
            scores = (
                q_dequant[0, head_idx, q_start:q_stop]
                @ k_dequant[0, head_idx]
                .index_select(0, selected)
                .transpose(0, 1)
                * softmax_scale
            )
            reference[0, head_idx, q_start:q_stop] = (
                torch.softmax(scores, dim=-1)
                @ v_dequant[0, head_idx].index_select(0, selected)
            )

    diff = (out.float() - reference).abs()
    assert diff.max().item() < 0.15
    assert (diff.mean() / reference.abs().mean()).item() < 0.035


@pytest.mark.skipif(
    not _RUNTIME_TEST_ENABLED or _DEVICE_ARCH != 103,
    reason="Set BSA_TEST_SM100_AOT_RUNTIME=1 and run on SM103",
)
def test_sm103_aot_only_fp8_ldred_matches_reference(monkeypatch):
    artifact_root = os.getenv(SM100_AOT_DIR_ENV)
    assert artifact_root, f"{SM100_AOT_DIR_ENV} must be set"
    monkeypatch.setenv(SM100_AOT_ONLY_ENV, "1")

    import block_sparse_attention.bsa_attn_interface as interface

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run during FP8 AOT validation")

    monkeypatch.setattr(interface.cute, "compile", fail_compile)
    torch.manual_seed(20260911)
    seqlen_q = 512 * 64
    q = torch.randn(
        (1, 1, seqlen_q, 128),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.5
    k = torch.randn(
        (1, 1, 127, 128),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.5
    v = torch.randn_like(k) * 0.5
    q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale = quantize_sage_bhsd(
        q,
        k,
        v,
    )
    indices = (
        torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        .view(1, 1, 1, 2)
        .expand(1, 1, 512, 2)
        .contiguous()
    )
    softmax_scale = 128**-0.5
    out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        indices,
        2,
        softmax_scale,
    )
    torch.cuda.synchronize()

    def fail_lookup(*args, **kwargs):
        raise AssertionError("a warmed FP8 AOT call must use the fast cache")

    monkeypatch.setattr(interface, "get_sm100_aot_kernel", fail_lookup)
    cached_out = bsa_fp8_blk64_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        indices,
        2,
        softmax_scale,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(cached_out, out, rtol=0.0, atol=0.0)

    q_dequant = q_fp8.float() * q_scale.unsqueeze(-1)
    k_dequant = k_fp8.float() * k_scale.repeat_interleave(
        16,
        dim=-1,
    )[..., : k_fp8.shape[2]].unsqueeze(-1)
    v_dequant = v_fp8.float() * v_scale.view(1, 1, 1, 128)
    scores = q_dequant @ k_dequant.transpose(-1, -2) * softmax_scale
    reference = torch.softmax(scores, dim=-1) @ v_dequant
    diff = (out.float() - reference).abs()
    assert diff.max().item() < 0.15
    assert (diff.mean() / reference.abs().mean()).item() < 0.035
