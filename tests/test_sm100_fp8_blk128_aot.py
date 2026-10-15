"""Real exported FP8 blk128 artifacts, strict dispatch and compatibility checks."""

import json
import os
import shutil
from pathlib import Path

import pytest
import torch
import cutlass.cute as cute

from block_sparse_attention import bsa_fp8_blk128_fwd, bsa_fp8_fwd, quantize_sage_bhsd
from block_sparse_attention.csrc.fwd.sm100_blk128 import aot_runtime, aot_utils
from block_sparse_attention.csrc.fwd.sm100_blk64.aot_runtime import Sm100AotArtifactError
from test_bsa_fp8_blk128 import assert_accuracy, dequantize, make_inputs, make_metadata, sparse_reference

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="FP8 blk128 native artifacts require SM100/SM103",
)


def fail_compile(*args, **kwargs):
    raise AssertionError("cute.compile is prohibited during AOT-only execution")


@pytest.fixture
def artifacts(monkeypatch):
    root = os.getenv("BSA_TEST_FP8_BLK128_AOT_DIR")
    if not root:
        pytest.skip("Set BSA_TEST_FP8_BLK128_AOT_DIR to actual exported artifacts")
    device_arch = torch.cuda.get_device_capability()
    target = f"sm_{device_arch[0]}{device_arch[1]}a"
    directory = Path(root)
    if not (directory / "manifest.json").is_file():
        directory = aot_utils.artifact_dir(root, target)
    assert (directory / "manifest.json").is_file()
    monkeypatch.setenv(aot_runtime.AOT_DIR_ENV, str(directory))
    monkeypatch.setenv(aot_runtime.AOT_ONLY_ENV, "1")
    aot_runtime.clear_fp8_blk128_aot_runtime_cache()
    yield directory
    aot_runtime.clear_fp8_blk128_aot_runtime_cache()


@pytest.mark.parametrize("value", [0, -1, 257, 1.5, True])
def test_invalid_split_variant(value):
    with pytest.raises(ValueError):
        aot_utils.Sm100Fp8Blk128AotVariant(kv_splits=value)


@requires_sm100
@pytest.mark.parametrize("kv_splits", [1, 2, 4, 8])
@pytest.mark.parametrize("sizes_mode", [0, 1, 2, 3])
@pytest.mark.parametrize("variable", [False, True])
def test_exported_strict_dynamic(artifacts, monkeypatch, kv_splits, sizes_mode, variable):
    monkeypatch.setattr(cute, "compile", fail_compile)
    torch.manual_seed(4130 + sizes_mode)
    for batch, heads, sq, sk, topk in [(1, 1, 1, 1, 1), (2, 3, 257, 913, 7)]:
        q, k, v = make_inputs(batch, heads, sq, sk, gain=0.5)
        index, nums, sizes = make_metadata(
            batch, heads, sq, sk, topk, variable=variable, sizes_mode=sizes_mode
        )
        quantized = quantize_sage_bhsd(q, k, v)
        kwargs = dict(sparse_block_size=128, kv_splits=kv_splits,
                      block_sizes=sizes, q2k_block_nums=nums)
        out = bsa_fp8_fwd(*quantized, index, topk, **kwargs)
        assert_accuracy(out, sparse_reference(*dequantize(quantized), index, topk, nums, sizes))
        assert_accuracy(out, sparse_reference(q, k, v, index, topk, nums, sizes), relative=0.08, absolute=0.3)
        assert torch.equal(out, bsa_fp8_fwd(*quantized, index, topk, **kwargs))
        quantized = list(quantized)
        quantized[2] = torch.zeros_like(quantized[2])
        assert torch.count_nonzero(bsa_fp8_fwd(*quantized, index, topk, **kwargs)) == 0


def small_call(kv_splits=1):
    q, k, v = make_inputs(1, 1, 129, 257, gain=0.5)
    index, _, _ = make_metadata(1, 1, 129, 257, 3)
    args = (*quantize_sage_bhsd(q, k, v), index, 3)
    return args, dict(sparse_block_size=128, kv_splits=kv_splits)


@requires_sm100
def test_strict_cached_dispatch(artifacts, monkeypatch):
    args, kwargs = small_call(2)
    monkeypatch.setattr(cute, "compile", fail_compile)
    out = bsa_fp8_fwd(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(aot_runtime, "_load_manifest", fail_compile)
        patch.setattr(aot_runtime, "_load_bound_kernel", fail_compile)
        assert torch.equal(out, bsa_fp8_fwd(*args, **kwargs))


@requires_sm100
def test_missing_artifact_fallback_and_strict_jit_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("BSA_SM100_AOT_ONLY", raising=False)
    monkeypatch.setenv(aot_runtime.AOT_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(aot_runtime.AOT_ONLY_ENV, "0")
    aot_runtime.clear_fp8_blk128_aot_runtime_cache()
    args, kwargs = small_call(2)
    out = bsa_fp8_fwd(*args, **kwargs)
    assert torch.isfinite(out).all()
    # A populated JIT cache must not bypass the missing native artifact.
    monkeypatch.setenv(aot_runtime.AOT_ONLY_ENV, "1")
    monkeypatch.setattr(cute, "compile", fail_compile)
    with pytest.raises(Sm100AotArtifactError, match="AOT-only requires"):
        bsa_fp8_fwd(*args, **kwargs)
    aot_runtime.clear_fp8_blk128_aot_runtime_cache()


@requires_sm100
@pytest.mark.parametrize("change", ["target", "source", "dsl", "missing", "combine", "checksum"])
def test_reject_invalid_artifacts(artifacts, monkeypatch, tmp_path, change):
    directory = tmp_path / "artifacts"
    shutil.copytree(artifacts, directory)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    variant = aot_utils.Sm100Fp8Blk128AotVariant(kv_splits=2)
    if change == "target":
        manifest["target_arch"] = "sm_103a" if manifest["target_arch"] == "sm_100a" else "sm_100a"
    elif change == "source":
        manifest["source_fingerprint"] = "0" * 64
    elif change == "dsl":
        manifest["cutlass_dsl_version"] = "incompatible-test-version"
    elif change == "missing":
        del manifest["variants"][variant.name]
    elif change == "combine":
        del manifest["combine_variants"][variant.combine_name]
    else:
        library = directory / manifest["variants"][variant.name]["shared_library"]
        with library.open("ab") as file:
            file.write(b"corrupt-test-artifact")
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv(aot_runtime.AOT_DIR_ENV, str(directory))
    aot_runtime.clear_fp8_blk128_aot_runtime_cache()
    args, kwargs = small_call(2)
    monkeypatch.setattr(cute, "compile", fail_compile)
    with pytest.raises(Sm100AotArtifactError):
        bsa_fp8_fwd(*args, **kwargs)


@requires_sm100
def test_native_current_stream_and_scale(artifacts, monkeypatch):
    args, kwargs = small_call(4)
    kwargs.pop("sparse_block_size")
    monkeypatch.setattr(cute, "compile", fail_compile)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        changed = list(args)
        changed[2] = torch.zeros_like(changed[2])
        out = bsa_fp8_blk128_fwd(*changed, softmax_scale=0.17, **kwargs)
        copied = out.clone()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.count_nonzero(copied) == 0
    scaled = bsa_fp8_blk128_fwd(*args, softmax_scale=0.17, **kwargs)
    reference = sparse_reference(*dequantize(args[:6]), args[6], args[7], scale=0.17)
    assert_accuracy(scaled, reference)
