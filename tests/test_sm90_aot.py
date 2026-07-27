import math
import os
from pathlib import Path
from types import SimpleNamespace

import cutlass
import pytest
import torch

from block_sparse_attention import bsa_attn_fwd, bsa_attn_interface
from block_sparse_attention.csrc.fwd.sm90_blk64 import aot_runtime
from block_sparse_attention.csrc.fwd.sm90_blk64.aot_build import (
    build_sm90_aot_artifacts,
)
from block_sparse_attention.csrc.fwd.sm90_blk64.aot_runtime import (
    SM90_AOT_DIR_ENV,
    SM90_AOT_ONLY_ENV,
    Sm90AotArtifactError,
    clear_sm90_aot_runtime_cache,
    get_sm90_aot_combine_kernel,
    get_sm90_aot_kernel,
)
from block_sparse_attention.csrc.fwd.sm90_blk64.aot_utils import (
    SM90_AOT_LAYOUT_MODE,
    SM90_AOT_MIN_CUTLASS_DSL_VERSION,
    SM90_AOT_SCHEMA_VERSION,
    Sm90AotCombineVariant,
    Sm90AotVariant,
    compute_sm90_aot_source_fingerprint,
    get_cuda_runtime_version,
    get_host_cpu_arch,
    get_sm90_aot_combine_variant,
    iter_sm90_aot_combine_variants,
    iter_sm90_aot_variants,
    load_sm90_aot_manifest,
    require_sm90_aot_cutlass_dsl_version,
    sha256_file,
    write_sm90_aot_manifest,
)


_RUNTIME_TEST_ENABLED = os.getenv("BSA_TEST_SM90_AOT_RUNTIME") == "1"
_IS_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
_SKIP_RUNTIME_TEST = not _RUNTIME_TEST_ENABLED or not _IS_SM90


def _variant(**overrides) -> Sm90AotVariant:
    values = {
        "dtype": "bf16",
        "qk_dim": 128,
        "value_dim": 128,
        "gqa_ratio": 1,
        "has_block_sizes": False,
        "kv_splits": 1,
        "allow_empty_block_nums": False,
    }
    values.update(overrides)
    return Sm90AotVariant(**values)


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


def _write_compatible_sm90_artifact(
    root: Path,
    variants: tuple[Sm90AotVariant, ...],
    combine_variants: tuple[Sm90AotCombineVariant, ...] = (),
) -> Path:
    artifact_dir = root / get_host_cpu_arch() / "sm_90a"
    artifact_dir.mkdir(parents=True)
    manifest = {
        "schema_version": SM90_AOT_SCHEMA_VERSION,
        "target_arch": "sm_90a",
        "cpu_arch": get_host_cpu_arch(),
        "cutlass_dsl_version": str(cutlass.__version__),
        "cuda_python_version": "test",
        "cuda_runtime_version": get_cuda_runtime_version(),
        "python_abi": "test",
        "source_fingerprint": compute_sm90_aot_source_fingerprint(),
        "variants": {
            variant.name: _artifact_entry(artifact_dir, variant)
            for variant in variants
        },
        "combine_variants": {
            variant.name: _artifact_entry(artifact_dir, variant)
            for variant in combine_variants
        },
    }
    return write_sm90_aot_manifest(artifact_dir, manifest)


def _make_runtime_tensors(variant: Sm90AotVariant) -> tuple[torch.Tensor, ...]:
    dtype = torch.bfloat16 if variant.dtype == "bf16" else torch.float16
    q = torch.empty((1, 2, 64, variant.qk_dim), dtype=dtype).permute(2, 3, 1, 0)
    k = torch.empty((1, 1, 128, variant.qk_dim), dtype=dtype).permute(2, 3, 1, 0)
    v = torch.empty((1, 1, 128, variant.value_dim), dtype=dtype).permute(3, 2, 1, 0)
    output_dtype = torch.float32 if variant.kv_splits > 1 else dtype
    out = torch.empty(
        (1, variant.kv_splits * 2, 64, variant.value_dim),
        dtype=output_dtype,
    ).permute(2, 3, 1, 0)
    lse = torch.empty((1, variant.kv_splits * 2, 64), dtype=torch.float32).permute(
        2, 1, 0
    )
    indices = torch.empty((1, 2, 1, 2), dtype=torch.int32).permute(3, 2, 1, 0)
    block_nums = torch.empty((1, 2, 1), dtype=torch.int32).permute(2, 1, 0)
    block_sizes = torch.empty((2, 2, 1), dtype=torch.int32)
    split_offsets = (
        torch.empty((1, 2, 1, variant.kv_splits + 1), dtype=torch.int32).permute(
            3, 2, 1, 0
        )
        if variant.kv_splits > 1
        else block_nums
    )
    return (
        q,
        k,
        v,
        out,
        lse,
        indices,
        block_nums,
        block_sizes if variant.has_block_sizes else block_nums,
        split_offsets,
    )


def test_sm90_aot_variant_matrix_and_combine_derivation_are_deterministic():
    variants = iter_sm90_aot_variants(
        ("fp16", "bf16"),
        (128, 64),
        (96,),
        (2, 1, 2),
        (True, False),
        (4, 1),
        (True, False),
    )

    assert len(variants) == 48
    assert tuple(variant.name for variant in variants) == tuple(
        sorted(variant.name for variant in variants)
    )
    assert all(
        variant.kv_splits == 1 or not variant.allow_empty_block_nums
        for variant in variants
    )
    combine_variants = iter_sm90_aot_combine_variants(variants)
    assert len(combine_variants) == 2
    assert {variant.log_max_splits for variant in combine_variants} == {2}


def test_sm90_aot_variant_round_trip_and_validation():
    variant = _variant(
        dtype="fp16",
        qk_dim=64,
        value_dim=96,
        gqa_ratio=8,
        has_block_sizes=True,
        kv_splits=4,
    )
    assert variant.layout_mode == SM90_AOT_LAYOUT_MODE
    assert Sm90AotVariant.from_dict(variant.to_dict()) == variant
    assert variant.name == (
        "bsa_sm90_blk64_fp16_qk64_v96_gqa8_bs1_split4_empty0_dyn"
    )
    combine = get_sm90_aot_combine_variant(variant)
    assert combine == Sm90AotCombineVariant("fp16", 96, 2)
    assert Sm90AotCombineVariant.from_dict(combine.to_dict()) == combine

    with pytest.raises(ValueError, match="dtype"):
        _variant(dtype="fp32")
    with pytest.raises(ValueError, match="qk_dim"):
        _variant(qk_dim=32)
    with pytest.raises(ValueError, match="value_dim"):
        _variant(value_dim=192)
    with pytest.raises(ValueError, match="gqa_ratio"):
        _variant(gqa_ratio=0)
    with pytest.raises(ValueError, match="kv_splits"):
        _variant(kv_splits=257)
    with pytest.raises(ValueError, match="allow_empty"):
        _variant(kv_splits=2, allow_empty_block_nums=True)


def test_sm90_aot_requires_dynamic_layout_capable_dsl():
    assert SM90_AOT_MIN_CUTLASS_DSL_VERSION == "4.5.2"
    require_sm90_aot_cutlass_dsl_version("4.5.2")
    require_sm90_aot_cutlass_dsl_version("4.6.0.dev0")

    with pytest.raises(RuntimeError, match="nvidia-cutlass-dsl>=4.5.2"):
        require_sm90_aot_cutlass_dsl_version("4.4.2")
    with pytest.raises(RuntimeError, match="Cannot parse"):
        require_sm90_aot_cutlass_dsl_version("unknown")


def test_sm90_aot_manifest_round_trip(tmp_path: Path):
    variant = _variant(kv_splits=2)
    combine = get_sm90_aot_combine_variant(variant)
    manifest_path = _write_compatible_sm90_artifact(
        tmp_path,
        (variant,),
        (combine,),
    )
    manifest = load_sm90_aot_manifest(manifest_path)

    assert tuple(manifest["variants"]) == (variant.name,)
    assert tuple(manifest["combine_variants"]) == (combine.name,)
    assert len(compute_sm90_aot_source_fingerprint()) == 64


def test_sm90_aot_manifest_rejects_function_name_mismatch(tmp_path: Path):
    variant = _variant(kv_splits=2)
    combine = get_sm90_aot_combine_variant(variant)
    manifest_path = _write_compatible_sm90_artifact(
        tmp_path,
        (variant,),
        (combine,),
    )

    for section, name in (
        ("variants", variant.name),
        ("combine_variants", combine.name),
    ):
        manifest = load_sm90_aot_manifest(manifest_path)
        manifest[section][name]["function_name"] = "wrong_function"
        with pytest.raises(ValueError, match="function_name mismatch"):
            write_sm90_aot_manifest(manifest_path.parent, manifest)


def test_sm90_aot_loader_verifies_and_caches_main_and_combine(
    tmp_path: Path,
    monkeypatch,
):
    variant = _variant(kv_splits=2)
    combine = get_sm90_aot_combine_variant(variant)
    _write_compatible_sm90_artifact(tmp_path, (variant,), (combine,))
    main_kernel = object()
    combine_kernel = object()
    load_calls = []
    fingerprint_calls = []

    source_fingerprint = compute_sm90_aot_source_fingerprint()

    def fake_source_fingerprint():
        fingerprint_calls.append(True)
        return source_fingerprint

    def fake_load_module(path, enable_tvm_ffi):
        load_calls.append((path, enable_tvm_ffi))
        name = variant.name if path.endswith(f"{variant.name}.so") else combine.name
        function = main_kernel if name == variant.name else combine_kernel
        return SimpleNamespace(**{name: function})

    clear_sm90_aot_runtime_cache()
    monkeypatch.setattr(aot_runtime.cute.runtime, "load_module", fake_load_module)
    monkeypatch.setattr(
        aot_runtime,
        "compute_sm90_aot_source_fingerprint",
        fake_source_fingerprint,
    )
    tensors = _make_runtime_tensors(variant)

    assert get_sm90_aot_kernel(variant, 90, tensors, root=tmp_path) is main_kernel
    assert get_sm90_aot_kernel(variant, 90, tensors, root=tmp_path) is main_kernel
    assert (
        get_sm90_aot_combine_kernel(combine, 90, root=tmp_path) is combine_kernel
    )
    assert (
        get_sm90_aot_combine_kernel(combine, 90, root=tmp_path) is combine_kernel
    )
    assert len(load_calls) == 2
    assert len(fingerprint_calls) == 1
    assert all(not enable_tvm_ffi for _, enable_tvm_ffi in load_calls)
    clear_sm90_aot_runtime_cache()


def test_sm90_aot_only_rejects_missing_artifact_and_combine(tmp_path: Path):
    variant = _variant(kv_splits=2)
    combine = get_sm90_aot_combine_variant(variant)
    with pytest.raises(Sm90AotArtifactError, match="requires"):
        get_sm90_aot_kernel(
            variant,
            90,
            _make_runtime_tensors(variant),
            required=True,
            root=tmp_path,
        )

    _write_compatible_sm90_artifact(tmp_path, (variant,))
    with pytest.raises(Sm90AotArtifactError, match="combine variant"):
        get_sm90_aot_combine_kernel(
            combine,
            90,
            required=True,
            root=tmp_path,
        )


def test_sm90_aot_loader_rejects_checksum_mismatch(tmp_path: Path):
    variant = _variant()
    manifest_path = _write_compatible_sm90_artifact(tmp_path, (variant,))
    manifest = load_sm90_aot_manifest(manifest_path)
    shared_library = manifest["variants"][variant.name]["shared_library"]
    (manifest_path.parent / shared_library).write_bytes(b"corrupted")
    clear_sm90_aot_runtime_cache()

    with pytest.raises(Sm90AotArtifactError, match="checksum mismatch"):
        get_sm90_aot_kernel(
            variant,
            90,
            _make_runtime_tensors(variant),
            root=tmp_path,
        )
    clear_sm90_aot_runtime_cache()


@pytest.mark.parametrize("layout", ["broadcast", "non_unit_leading"])
def test_sm90_aot_only_rejects_unsupported_layout(
    tmp_path: Path,
    layout: str,
):
    variant = _variant()
    tensors = list(_make_runtime_tensors(variant))
    if layout == "broadcast":
        tensors[0] = torch.empty((64, 128, 1, 1), dtype=torch.bfloat16).expand(
            64, 128, 2, 1
        )
    else:
        storage = torch.empty(64 * 128 * 2 * 2, dtype=torch.bfloat16)
        tensors[0] = torch.as_strided(storage, (64, 128, 2, 1), (512, 2, 256, 1))

    with pytest.raises(Sm90AotArtifactError, match="unit leading strides"):
        get_sm90_aot_kernel(
            variant,
            90,
            tensors,
            required=True,
            root=tmp_path,
        )


def test_sm90_aot_callable_resolution_never_jits_on_hit(monkeypatch):
    variant = _variant()
    aot_kernel = object()
    compile_cache = {}
    monkeypatch.setattr(
        bsa_attn_interface,
        "get_sm90_aot_kernel",
        lambda *args, **kwargs: aot_kernel,
    )

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run on an AOT hit")

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fail_compile)
    resolved = bsa_attn_interface._resolve_sm90_fwd_callable(
        variant,
        90,
        _make_runtime_tensors(variant),
        ("compile-key",),
        object(),
        (),
        compile_cache,
    )
    assert resolved is aot_kernel
    assert not compile_cache


def test_sm90_aot_callable_resolution_caches_jit_fallback(monkeypatch):
    variant = _variant()
    jit_kernel = object()
    compile_calls = []
    compile_cache = {}
    monkeypatch.setattr(
        bsa_attn_interface,
        "get_sm90_aot_kernel",
        lambda *args, **kwargs: None,
    )

    def fake_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))
        return jit_kernel

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fake_compile)
    call_args = (
        variant,
        90,
        _make_runtime_tensors(variant),
        ("compile-key",),
        object(),
        (),
        compile_cache,
    )
    assert bsa_attn_interface._resolve_sm90_fwd_callable(*call_args) is jit_kernel
    assert bsa_attn_interface._resolve_sm90_fwd_callable(*call_args) is jit_kernel
    assert len(compile_calls) == 1


def test_sm90_aot_only_resolution_never_falls_back_to_jit(
    tmp_path: Path,
    monkeypatch,
):
    variant = _variant()
    monkeypatch.setenv(SM90_AOT_ONLY_ENV, "1")
    monkeypatch.setenv(SM90_AOT_DIR_ENV, str(tmp_path))

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run in AOT-only mode")

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fail_compile)
    with pytest.raises(Sm90AotArtifactError, match="requires"):
        bsa_attn_interface._resolve_sm90_fwd_callable(
            variant,
            90,
            _make_runtime_tensors(variant),
            ("compile-key",),
            object(),
            (),
            {},
        )


def test_sm90_split_combine_aot_hit_never_jits(monkeypatch):
    aot_kernel = object()
    monkeypatch.setattr(bsa_attn_interface, "_get_device_arch", lambda: 90)
    monkeypatch.setattr(bsa_attn_interface, "is_fake_mode", lambda: True)
    monkeypatch.setattr(bsa_attn_interface, "BlockSparseAttnForwardCombine", None)
    monkeypatch.setattr(
        bsa_attn_interface,
        "get_sm90_aot_combine_kernel",
        lambda *args, **kwargs: aot_kernel,
    )

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run on an AOT combine hit")

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fail_compile)
    q = torch.empty((1, 2, 64, 128), dtype=torch.bfloat16)
    o_partial = torch.empty((1, 4, 64, 128), dtype=torch.float32)
    lse_partial = torch.empty((1, 4, 64), dtype=torch.float32)
    out, lse = bsa_attn_interface._combine_blk64_kv_bucketed_partials(
        q,
        o_partial,
        lse_partial,
        2,
    )
    assert out.shape == q.shape
    assert lse.shape == q.shape[:3]


@pytest.mark.parametrize("mismatch", ["dsl", "cuda", "source"])
def test_sm90_aot_loader_rejects_incompatible_manifest(
    tmp_path: Path,
    monkeypatch,
    mismatch: str,
):
    variant = _variant()
    manifest_path = _write_compatible_sm90_artifact(tmp_path, (variant,))
    manifest = load_sm90_aot_manifest(manifest_path)
    if mismatch == "dsl":
        manifest["cutlass_dsl_version"] = "incompatible"
        error = "version mismatch"
    elif mismatch == "cuda":
        monkeypatch.setattr(aot_runtime, "get_cuda_runtime_version", lambda: "99.0")
        error = "CUDA version mismatch"
    else:
        manifest["source_fingerprint"] = "incompatible"
        error = "source fingerprint mismatch"
    write_sm90_aot_manifest(manifest_path.parent, manifest)
    clear_sm90_aot_runtime_cache()

    with pytest.raises(Sm90AotArtifactError, match=error):
        get_sm90_aot_kernel(
            variant,
            90,
            _make_runtime_tensors(variant),
            root=tmp_path,
        )
    clear_sm90_aot_runtime_cache()


@pytest.mark.skipif(
    os.getenv("BSA_TEST_SM90_AOT_BUILD") != "1",
    reason="Set BSA_TEST_SM90_AOT_BUILD=1 to run the SM90 cross-compile test",
)
def test_build_sm90_split_aot_artifacts(tmp_path: Path):
    variant = _variant(kv_splits=2)
    combine = get_sm90_aot_combine_variant(variant)
    manifest_path = build_sm90_aot_artifacts(tmp_path, (variant,))
    manifest = load_sm90_aot_manifest(manifest_path)
    artifact_dir = manifest_path.parent

    for section, name in (("variants", variant.name), ("combine_variants", combine.name)):
        entry = manifest[section][name]
        assert (artifact_dir / entry["object"]).is_file()
        assert (artifact_dir / entry["header"]).is_file()
        shared_library = artifact_dir / entry["shared_library"]
        assert shared_library.is_file()
        assert sha256_file(shared_library) == entry["sha256"]

    clear_sm90_aot_runtime_cache()
    assert callable(
        get_sm90_aot_kernel(
            variant,
            90,
            _make_runtime_tensors(variant),
            required=True,
            root=tmp_path,
        )
    )
    assert callable(
        get_sm90_aot_combine_kernel(
            combine,
            90,
            required=True,
            root=tmp_path,
        )
    )
    clear_sm90_aot_runtime_cache()


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
        return physical_sizes
    if mode == 2:
        return physical_sizes.unsqueeze(0).expand(batch, -1).clone()
    return physical_sizes.view(1, 1, -1).expand(batch, q_heads, -1).clone()


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
    batch, q_heads, seqlen_q, _ = q.shape
    seqlen_k = k.shape[2]
    gqa_ratio = q_heads // k.shape[1]
    out = torch.empty(
        (*q.shape[:-1], v.shape[-1]),
        dtype=q.dtype,
        device=q.device,
    )
    lse = torch.empty(
        (batch, q_heads, seqlen_q),
        dtype=torch.float32,
        device=q.device,
    )
    scale = q.shape[-1] ** -0.5
    for batch_idx in range(batch):
        for head_idx in range(q_heads):
            kv_head_idx = head_idx // gqa_ratio
            for q_block_idx in range(seqlen_q // 64):
                count = (
                    int(block_nums[batch_idx, head_idx, q_block_idx])
                    if block_nums.numel()
                    else block_sparse_num
                )
                q_start = q_block_idx * 64
                q_end = q_start + 64
                if count == 0:
                    out[batch_idx, head_idx, q_start:q_end].zero_()
                    lse[batch_idx, head_idx, q_start:q_end].fill_(float("-inf"))
                    continue
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


def _run_sm90_aot_runtime_case(
    dtype: torch.dtype,
    qk_dim: int,
    value_dim: int,
    q_heads: int,
    kv_heads: int,
    has_block_nums: bool,
    allow_empty_block_nums: bool,
    block_sizes_mode: int,
    kv_splits: int,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(
        qk_dim + value_dim + q_heads * 10 + block_sizes_mode + kv_splits
    )
    batch, seqlen_q, seqlen_k, capacity = 1, 128, 257, 4
    q = torch.randn(
        (batch, q_heads, seqlen_q, qk_dim), dtype=dtype, device=device
    )
    k = torch.randn(
        (batch, kv_heads, seqlen_k, qk_dim), dtype=dtype, device=device
    )
    v = torch.randn(
        (batch, kv_heads, seqlen_k, value_dim), dtype=dtype, device=device
    )
    num_q_blocks = seqlen_q // 64
    num_kv_blocks = math.ceil(seqlen_k / 64)
    indices = torch.empty(
        (batch, q_heads, num_q_blocks, capacity),
        dtype=torch.int32,
        device=device,
    )
    base_indices = torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
    for head_idx in range(q_heads):
        for q_block_idx in range(num_q_blocks):
            indices[0, head_idx, q_block_idx] = torch.roll(
                base_indices,
                shifts=head_idx + q_block_idx,
            )[:capacity]
    if has_block_nums:
        block_nums = torch.full(
            (batch, q_heads, num_q_blocks),
            capacity,
            dtype=torch.int32,
            device=device,
        )
        if allow_empty_block_nums:
            block_nums[0, 0, 0] = 0
        block_sparse_num = 0
    else:
        block_nums = torch.empty(0, dtype=torch.int32, device=device)
        block_sparse_num = capacity
    block_sizes = _make_runtime_block_sizes(
        block_sizes_mode,
        batch,
        q_heads,
        seqlen_k,
        device,
    )

    out, lse = bsa_attn_fwd(
        q,
        k,
        v,
        indices,
        block_sparse_num,
        block_sizes,
        q2k_block_nums=block_nums,
        allow_empty_block_nums=allow_empty_block_nums,
        kv_splits=kv_splits,
        return_lse=True,
        sparse_block_size=64,
    )
    ref_out, ref_lse = _reference_sparse_attention(
        q,
        k,
        v,
        indices,
        block_sparse_num,
        block_nums,
        block_sizes,
        block_sizes_mode,
    )
    torch.testing.assert_close(out, ref_out, rtol=3e-2, atol=3e-2)
    finite = ref_lse.isfinite()
    torch.testing.assert_close(lse[finite], ref_lse[finite], rtol=2e-3, atol=2e-3)
    assert torch.equal(torch.isneginf(lse), torch.isneginf(ref_lse))


@pytest.mark.skipif(
    _SKIP_RUNTIME_TEST,
    reason="Set BSA_TEST_SM90_AOT_RUNTIME=1 and run on SM90",
)
@pytest.mark.parametrize(
    (
        "dtype",
        "qk_dim",
        "value_dim",
        "q_heads",
        "kv_heads",
        "has_block_nums",
        "allow_empty_block_nums",
        "block_sizes_mode",
        "kv_splits",
    ),
    [
        (torch.bfloat16, 128, 128, 4, 2, False, False, 0, 1),
        (torch.bfloat16, 128, 128, 4, 2, True, False, 1, 2),
        (torch.float16, 64, 96, 2, 2, True, False, 2, 4),
        (torch.bfloat16, 96, 64, 4, 1, True, False, 3, 8),
        (torch.bfloat16, 128, 128, 1, 1, True, True, 1, 1),
    ],
)
def test_sm90_aot_only_forward_matrix(
    dtype,
    qk_dim,
    value_dim,
    q_heads,
    kv_heads,
    has_block_nums,
    allow_empty_block_nums,
    block_sizes_mode,
    kv_splits,
    monkeypatch,
):
    assert os.getenv(SM90_AOT_DIR_ENV), f"{SM90_AOT_DIR_ENV} must be set"
    monkeypatch.setenv(SM90_AOT_ONLY_ENV, "1")

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run during AOT-only validation")

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fail_compile)
    _run_sm90_aot_runtime_case(
        dtype,
        qk_dim,
        value_dim,
        q_heads,
        kv_heads,
        has_block_nums,
        allow_empty_block_nums,
        block_sizes_mode,
        kv_splits,
    )
