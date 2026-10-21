from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch._subclasses.fake_tensor as fake_tensor_module
from torch._subclasses.fake_tensor import FakeTensorMode

from block_sparse_attention import bsa_attn_interface
from block_sparse_attention.csrc.fwd.sm100_blk64.aot_runtime import (
    Sm100AotArtifactError,
)


def _make_fake_bf16_inputs(device="cuda"):
    q = torch.empty((1, 2, 64, 128), dtype=torch.bfloat16, device=device)
    k = torch.empty((1, 2, 128, 128), dtype=torch.bfloat16, device=device)
    v = torch.empty_like(k)
    q2k = torch.empty((1, 2, 1, 2), dtype=torch.int32, device=device)
    return q, k, v, q2k


def _make_fake_fp8_inputs(device="cuda"):
    q = torch.empty((1, 4, 64, 128), dtype=torch.float8_e4m3fn, device=device)
    k = torch.empty((1, 4, 128, 128), dtype=torch.float8_e4m3fn, device=device)
    v = torch.empty_like(k)
    q_scale = torch.empty((1, 4, 64), dtype=torch.float32, device=device)
    k_scale = torch.empty((1, 4, 8), dtype=torch.float32, device=device)
    v_scale = torch.empty((4, 128), dtype=torch.float32, device=device)
    q2k = torch.empty((1, 4, 1, 2), dtype=torch.int32, device=device)
    return q, k, v, q_scale, k_scale, v_scale, q2k


def test_device_arch_lookup_uses_explicit_device(monkeypatch):
    device_indices = []

    def fake_get_device_arch_for_device(device_index):
        device_indices.append(device_index)
        return 103

    monkeypatch.setattr(
        bsa_attn_interface,
        "_get_device_arch_for_device",
        fake_get_device_arch_for_device,
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(AssertionError("current device must not be read")),
    )

    assert bsa_attn_interface._get_device_arch(torch.device("cuda:7")) == 103
    assert device_indices == [7]


def test_sm100_main_aot_uses_tensor_device_arch_and_stream(monkeypatch):
    arch_devices = []
    stream_devices = []
    native_streams = []

    def fake_get_device_arch(device=None):
        arch_devices.append(device)
        return 103

    def fake_current_stream(device=None):
        stream_devices.append(device)
        return SimpleNamespace(cuda_stream=17)

    def fake_make_args(*args, **kwargs):
        native_streams.append(args[-1])
        return ()

    monkeypatch.setattr(bsa_attn_interface, "_get_device_arch", fake_get_device_arch)
    monkeypatch.setattr(bsa_attn_interface, "is_fake_mode", lambda: False)
    monkeypatch.setattr(bsa_attn_interface, "is_sm100_aot_only", lambda: True)
    monkeypatch.setattr(
        bsa_attn_interface, "get_sm100_aot_kernel", lambda *args, **kwargs: lambda: None
    )
    monkeypatch.setattr(
        bsa_attn_interface, "_make_sm100_blk64_cute_args", fake_make_args
    )
    monkeypatch.setattr(torch.cuda, "current_stream", fake_current_stream)
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(fake_tensor_module, "init_gpu_context", lambda _device: None)

    with FakeTensorMode():
        q, k, v, q2k = _make_fake_bf16_inputs("cuda:1")
        out, _ = bsa_attn_interface.bsa_attn_fwd(
            q,
            k,
            v,
            q2k,
            2,
            use_clc=False,
        )

    assert out.device == torch.device("cuda:1")
    assert arch_devices == [torch.device("cuda:1"), torch.device("cuda:1")]
    assert stream_devices == [torch.device("cuda:1")]
    assert len(native_streams) == 1


def test_sm100_combine_aot_uses_tensor_device_arch_and_stream(monkeypatch):
    arch_devices = []
    stream_devices = []
    native_streams = []

    def fake_get_device_arch(device=None):
        arch_devices.append(device)
        return 103

    def fake_current_stream(device=None):
        stream_devices.append(device)
        return SimpleNamespace(cuda_stream=19)

    def fake_make_args(*args, **kwargs):
        native_streams.append(args[-1])
        return ()

    monkeypatch.setattr(bsa_attn_interface, "_get_device_arch", fake_get_device_arch)
    monkeypatch.setattr(bsa_attn_interface, "is_fake_mode", lambda: False)
    monkeypatch.setattr(
        bsa_attn_interface,
        "get_sm100_aot_combine_kernel",
        lambda *args, **kwargs: lambda: None,
    )
    monkeypatch.setattr(
        bsa_attn_interface, "_make_blk64_combine_cute_args", fake_make_args
    )
    monkeypatch.setattr(torch.cuda, "current_stream", fake_current_stream)
    monkeypatch.setattr(fake_tensor_module, "init_gpu_context", lambda _device: None)

    with FakeTensorMode():
        q = torch.empty((1, 2, 64, 128), dtype=torch.bfloat16, device="cuda:1")
        o_partial = torch.empty(
            (1, 4, 64, 128), dtype=torch.float32, device="cuda:1"
        )
        lse_partial = torch.empty((1, 4, 64), dtype=torch.float32, device="cuda:1")
        out, _ = bsa_attn_interface._combine_blk64_kv_bucketed_partials(
            q,
            o_partial,
            lse_partial,
            2,
        )

    assert out.device == torch.device("cuda:1")
    assert arch_devices == [torch.device("cuda:1")]
    assert stream_devices == [torch.device("cuda:1")]
    assert len(native_streams) == 1


@pytest.mark.parametrize("device_arch", [100, 103])
def test_sm100_bf16_aot_hit_precedes_jit(monkeypatch, device_arch):
    captured = {}
    aot_kernel = object()
    monkeypatch.setattr(
        bsa_attn_interface, "_get_device_arch", lambda _device=None: device_arch
    )
    monkeypatch.setattr(bsa_attn_interface, "is_sm100_aot_only", lambda: False)

    def fake_get_aot(variant, arch, tensors):
        captured["variant"] = variant
        captured["arch"] = arch
        captured["tensors"] = tuple(tensors)
        return aot_kernel

    monkeypatch.setattr(bsa_attn_interface, "get_sm100_aot_kernel", fake_get_aot)

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run on an SM100 AOT hit")

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fail_compile)
    with FakeTensorMode():
        q, k, v, q2k = _make_fake_bf16_inputs()
        out, lse = bsa_attn_interface.bsa_attn_fwd(
            q,
            k,
            v,
            q2k,
            2,
            use_clc=False,
            return_lse=True,
        )

    variant = captured["variant"]
    assert captured["arch"] == device_arch
    assert variant.dtype == "bf16"
    assert variant.has_block_nums is False
    assert variant.allow_empty_block_nums is False
    assert variant.has_block_sizes is False
    assert variant.kv_splits == 1
    assert variant.use_clc_scheduler is False
    assert variant.use_int64_kv_strides is False
    assert len(captured["tensors"]) == 12
    assert captured["tensors"][5:8] == (None, None, None)
    assert out.shape == q.shape
    assert lse.shape == q.shape[:3]


def test_sm100_aot_only_missing_forward_never_jits(monkeypatch):
    monkeypatch.setattr(
        bsa_attn_interface, "_get_device_arch", lambda _device=None: 100
    )
    monkeypatch.setattr(bsa_attn_interface, "is_sm100_aot_only", lambda: True)
    monkeypatch.setattr(
        bsa_attn_interface,
        "get_sm100_aot_kernel",
        lambda *args, **kwargs: None,
    )

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run in SM100 AOT-only mode")

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fail_compile)
    with FakeTensorMode(), pytest.raises(Sm100AotArtifactError, match="missing"):
        q, k, v, q2k = _make_fake_bf16_inputs()
        bsa_attn_interface.bsa_attn_fwd(
            q,
            k,
            v,
            q2k,
            2,
            use_clc=False,
        )


@pytest.mark.parametrize("device_arch", [100, 103])
def test_sm100_fp8_aot_hit_precedes_jit(monkeypatch, device_arch):
    captured = {}
    aot_kernel = object()
    monkeypatch.setattr(
        bsa_attn_interface, "_get_device_arch", lambda _device=None: device_arch
    )
    monkeypatch.setattr(bsa_attn_interface, "is_sm100_aot_only", lambda: False)

    def fake_get_aot(variant, arch, tensors):
        captured["variant"] = variant
        captured["arch"] = arch
        captured["tensors"] = tuple(tensors)
        return aot_kernel

    monkeypatch.setattr(bsa_attn_interface, "get_sm100_aot_kernel", fake_get_aot)

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run on an FP8 AOT hit")

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fail_compile)
    with FakeTensorMode():
        q, k, v, q_scale, k_scale, v_scale, q2k = _make_fake_fp8_inputs()
        out, _ = bsa_attn_interface._bsa_attn_fwd_sm100_blk64(
            q,
            k,
            v,
            q2k,
            None,
            block_sparse_num=2,
            use_clc=False,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    variant = captured["variant"]
    assert captured["arch"] == device_arch
    assert variant.dtype == "fp8"
    assert variant.block_sizes_mode == 0
    assert variant.use_sage_ldred_rowmax is False
    assert len(captured["tensors"]) == 12
    assert all(
        actual is expected
        for actual, expected in zip(
            captured["tensors"][5:8],
            (q_scale, k_scale, v_scale),
        )
    )
    assert out.dtype == torch.bfloat16


def test_sm100_aot_only_missing_fp8_never_uses_fast_cache_or_jit(monkeypatch):
    monkeypatch.setattr(
        bsa_attn_interface, "_get_device_arch", lambda _device=None: 100
    )
    monkeypatch.setattr(bsa_attn_interface, "is_sm100_aot_only", lambda: True)
    monkeypatch.setattr(
        bsa_attn_interface,
        "get_sm100_aot_kernel",
        lambda *args, **kwargs: None,
    )

    class FailOnCacheLookup(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("AOT-only FP8 must not use the JIT fast cache")

    monkeypatch.setattr(
        bsa_attn_interface,
        "_bsa_fp8_blk64_fast_cache",
        FailOnCacheLookup(),
    )

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run in AOT-only mode")

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fail_compile)
    with FakeTensorMode(), pytest.raises(Sm100AotArtifactError, match="missing"):
        q, k, v, q_scale, k_scale, v_scale, q2k = _make_fake_fp8_inputs()
        bsa_attn_interface.bsa_fp8_blk64_fwd(
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            q2k,
            2,
        )


def test_sm100_fp8_aot_fast_hit_uses_native_producer_and_combine(monkeypatch):
    producer = object()
    combine = object()
    captured = {}
    monkeypatch.setattr(
        bsa_attn_interface, "_get_device_arch", lambda _device=None: 100
    )
    monkeypatch.setattr(
        bsa_attn_interface, "_sm100_blk64_auto_fp8_kv_splits", lambda *args: 4
    )
    monkeypatch.setattr(
        bsa_attn_interface,
        "_get_sm100_fp8_aot_fast_entry",
        lambda *args: (producer, combine),
    )
    monkeypatch.setattr(
        bsa_attn_interface,
        "_bsa_fp8_blk64_aot_fast_layout",
        lambda *args: True,
    )
    monkeypatch.setattr(bsa_attn_interface, "is_fake_mode", lambda: False)
    monkeypatch.setattr(bsa_attn_interface, "is_sm100_aot_only", lambda: True)

    def fake_launch(compiled_fn, q, *args, **kwargs):
        captured.update(
            compiled_fn=compiled_fn,
            native_abi=kwargs["native_abi"],
            combine_compiled=kwargs["combine_compiled"],
        )
        return torch.empty_like(q, dtype=torch.bfloat16)

    monkeypatch.setattr(
        bsa_attn_interface, "_bsa_fp8_blk64_launch_cached", fake_launch
    )
    monkeypatch.setattr(
        bsa_attn_interface,
        "_bsa_attn_fwd_sm100_blk64",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("an AOT fast hit must bypass general dispatch")
        ),
    )

    with FakeTensorMode():
        q, k, v, q_scale, k_scale, v_scale, q2k = _make_fake_fp8_inputs()
        out = bsa_attn_interface.bsa_fp8_blk64_fwd(
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            q2k,
            2,
        )

    assert captured == {
        "compiled_fn": producer,
        "native_abi": True,
        "combine_compiled": combine,
    }
    assert out.dtype == torch.bfloat16


def test_sm100_fp8_aot_fast_cache_rejects_stale_manifest(monkeypatch):
    fast_key = ("fp8", 4)
    cache = {fast_key: (("old",), object(), object())}
    monkeypatch.setattr(
        bsa_attn_interface,
        "_bsa_fp8_blk64_aot_fast_cache",
        cache,
    )
    monkeypatch.setattr(
        bsa_attn_interface,
        "get_sm100_aot_manifest_token",
        lambda _arch: ("new",),
    )

    assert (
        bsa_attn_interface._get_sm100_fp8_aot_fast_entry(fast_key, 103)
        is None
    )
    assert fast_key not in cache


def test_sm100_non_strict_fp8_aot_miss_falls_back_to_jit(monkeypatch):
    monkeypatch.setattr(
        bsa_attn_interface, "_get_device_arch", lambda _device=None: 100
    )
    monkeypatch.setattr(bsa_attn_interface, "is_sm100_aot_only", lambda: False)
    monkeypatch.setattr(
        bsa_attn_interface,
        "get_sm100_aot_kernel",
        lambda *args, **kwargs: None,
    )
    compile_calls = []

    def fake_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fake_compile)
    monkeypatch.setattr(
        bsa_attn_interface._bsa_attn_fwd_sm100_blk64,
        "compile_cache",
        {},
    )
    with FakeTensorMode():
        q, k, v, q_scale, k_scale, v_scale, q2k = _make_fake_fp8_inputs()
        out, _ = bsa_attn_interface._bsa_attn_fwd_sm100_blk64(
            q,
            k,
            v,
            q2k,
            None,
            block_sparse_num=2,
            use_clc=False,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    assert out.dtype == torch.bfloat16
    assert len(compile_calls) == 1
    assert compile_calls[0][1]["options"] == "--enable-tvm-ffi"


@pytest.mark.parametrize("device_arch", [100, 103])
@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_sm100_split_combine_aot_hit_never_jits(
    monkeypatch,
    device_arch,
    q_dtype,
):
    captured = {}
    aot_kernel = object()
    monkeypatch.setattr(
        bsa_attn_interface, "_get_device_arch", lambda _device=None: device_arch
    )
    monkeypatch.setattr(bsa_attn_interface, "is_fake_mode", lambda: True)
    monkeypatch.setattr(bsa_attn_interface, "is_sm100_aot_only", lambda: False)

    def fake_get_aot(variant, arch, **kwargs):
        captured["variant"] = variant
        captured["arch"] = arch
        captured["device_index"] = kwargs["device_index"]
        return aot_kernel

    monkeypatch.setattr(
        bsa_attn_interface,
        "get_sm100_aot_combine_kernel",
        fake_get_aot,
    )

    def fail_compile(*args, **kwargs):
        raise AssertionError("cute.compile must not run on an SM100 combine AOT hit")

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fail_compile)
    q = torch.empty((1, 2, 64, 128), dtype=q_dtype)
    o_partial = torch.empty((1, 4, 64, 128), dtype=torch.float32)
    lse_partial = torch.empty((1, 4, 64), dtype=torch.float32)
    out, lse = bsa_attn_interface._combine_blk64_kv_bucketed_partials(
        q,
        o_partial,
        lse_partial,
        2,
        output_dtype=torch.bfloat16,
    )

    variant = captured["variant"]
    assert captured["arch"] == device_arch
    assert captured["device_index"] is None
    assert variant.dtype == "bf16"
    assert variant.value_dim == 128
    assert variant.log_max_splits == 1
    assert variant.num_threads == 128
    assert out.shape == q.shape
    assert lse.shape == q.shape[:3]


def test_sm100_native_args_disable_tvm_ffi(monkeypatch):
    calls = []

    def fake_to_cute_tensor(tensor, **kwargs):
        calls.append((tensor, kwargs))
        return ("cute", len(calls))

    monkeypatch.setattr(bsa_attn_interface, "_to_cute_tensor", fake_to_cute_tensor)
    tensors = [object() for _ in range(7)]
    args = bsa_attn_interface._make_sm100_blk64_cute_args(
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        tensors[4],
        None,
        None,
        None,
        0.125,
        tensors[5],
        tensors[6],
        2,
        None,
        None,
        "stream",
        enable_tvm_ffi=False,
    )

    assert len(calls) == 7
    assert all(call_kwargs["enable_tvm_ffi"] is False for _, call_kwargs in calls)
    assert calls[4][1]["assumed_align"] == 4
    assert args[5:8] == (None, None, None)
    assert args[8] == 0.125
    assert args[11] == 2
    assert args[-1] == "stream"


def test_sm100_bf16_rejects_mixed_kv_dtype_before_aot_lookup(monkeypatch):
    monkeypatch.setattr(
        bsa_attn_interface, "_get_device_arch", lambda _device=None: 100
    )

    def fail_aot_lookup(*args, **kwargs):
        raise AssertionError("mixed-dtype inputs must not reach the AOT resolver")

    monkeypatch.setattr(bsa_attn_interface, "get_sm100_aot_kernel", fail_aot_lookup)
    with FakeTensorMode(), pytest.raises(AssertionError, match="all use bf16"):
        q = torch.empty((1, 2, 64, 128), dtype=torch.bfloat16, device="cuda")
        k = torch.empty((1, 2, 128, 128), dtype=torch.float16, device="cuda")
        v = torch.empty_like(k)
        q2k = torch.empty((1, 2, 1, 2), dtype=torch.int32, device="cuda")
        bsa_attn_interface.bsa_attn_fwd(q, k, v, q2k, 2, use_clc=False)


def test_sm100_non_strict_aot_miss_falls_back_to_tvm_ffi_jit(monkeypatch):
    monkeypatch.setattr(
        bsa_attn_interface, "_get_device_arch", lambda _device=None: 100
    )
    monkeypatch.setattr(bsa_attn_interface, "is_sm100_aot_only", lambda: False)
    monkeypatch.setattr(
        bsa_attn_interface,
        "get_sm100_aot_kernel",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        bsa_attn_interface._bsa_attn_fwd_sm100_blk64,
        "compile_cache",
        {},
    )
    calls = []

    def fake_compile(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fake_compile)
    with FakeTensorMode():
        q, k, v, q2k = _make_fake_bf16_inputs()
        out, lse = bsa_attn_interface.bsa_attn_fwd(
            q,
            k,
            v,
            q2k,
            2,
            use_clc=False,
            return_lse=True,
        )

    assert len(calls) == 1
    assert calls[0][1]["options"] == "--enable-tvm-ffi"
    assert out.shape == q.shape
    assert lse.shape == q.shape[:3]
