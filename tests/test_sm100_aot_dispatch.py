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
    assert len(captured["tensors"]) == 9
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


def test_sm100_aot_only_rejects_sage_without_lookup_or_jit(monkeypatch):
    monkeypatch.setattr(
        bsa_attn_interface, "_get_device_arch", lambda _device=None: 100
    )
    monkeypatch.setattr(bsa_attn_interface, "is_sm100_aot_only", lambda: True)

    def fail_aot_lookup(*args, **kwargs):
        raise AssertionError("Sage/FP8 must not use the BF16 AOT resolver")

    def fail_compile(*args, **kwargs):
        raise AssertionError("Sage/FP8 must not JIT in SM100 AOT-only mode")

    monkeypatch.setattr(bsa_attn_interface, "get_sm100_aot_kernel", fail_aot_lookup)
    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fail_compile)
    with FakeTensorMode(), pytest.raises(Sm100AotArtifactError, match="BF16"):
        q = torch.empty(
            (1, 4, 64, 128), dtype=torch.float8_e4m3fn, device="cuda"
        )
        k = torch.empty(
            (1, 4, 128, 128), dtype=torch.float8_e4m3fn, device="cuda"
        )
        v = torch.empty_like(k)
        q_scale = torch.empty((1, 4, 64), dtype=torch.float32, device="cuda")
        k_scale = torch.empty((1, 4, 8), dtype=torch.float32, device="cuda")
        v_scale = torch.empty((4, 128), dtype=torch.float32, device="cuda")
        q2k = torch.empty((1, 4, 1, 2), dtype=torch.int32, device="cuda")
        bsa_attn_interface._bsa_attn_fwd_sm100_blk64(
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


def test_sm100_aot_only_rejects_sage_before_fast_cache(monkeypatch):
    monkeypatch.setattr(
        bsa_attn_interface, "_get_device_arch", lambda _device=None: 100
    )
    monkeypatch.setattr(bsa_attn_interface, "is_sm100_aot_only", lambda: True)

    def fail_cache_lookup(*args, **kwargs):
        raise AssertionError("AOT-only mode must reject Sage before cache lookup")

    monkeypatch.setattr(
        bsa_attn_interface,
        "_sm100_blk64_auto_fp8_kv_splits",
        fail_cache_lookup,
    )
    with FakeTensorMode(), pytest.raises(Sm100AotArtifactError, match="BF16"):
        q = torch.empty(
            (1, 4, 64, 128), dtype=torch.float8_e4m3fn, device="cuda"
        )
        k = torch.empty(
            (1, 4, 128, 128), dtype=torch.float8_e4m3fn, device="cuda"
        )
        v = torch.empty_like(k)
        q_scale = torch.empty((1, 4, 64), dtype=torch.float32, device="cuda")
        k_scale = torch.empty((1, 4, 8), dtype=torch.float32, device="cuda")
        v_scale = torch.empty((4, 128), dtype=torch.float32, device="cuda")
        q2k = torch.empty((1, 4, 1, 2), dtype=torch.int32, device="cuda")
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


def test_sm100_non_strict_sage_remains_jit_only(monkeypatch):
    monkeypatch.setattr(
        bsa_attn_interface, "_get_device_arch", lambda _device=None: 100
    )
    monkeypatch.setattr(bsa_attn_interface, "is_sm100_aot_only", lambda: False)

    def fail_aot_lookup(*args, **kwargs):
        raise AssertionError("Sage/FP8 must not use the BF16 AOT resolver")

    compile_calls = []

    def fake_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(bsa_attn_interface, "get_sm100_aot_kernel", fail_aot_lookup)
    monkeypatch.setattr(bsa_attn_interface.cute, "compile", fake_compile)
    monkeypatch.setattr(
        bsa_attn_interface._bsa_attn_fwd_sm100_blk64,
        "compile_cache",
        {},
    )
    with FakeTensorMode():
        q = torch.empty(
            (1, 4, 64, 128), dtype=torch.float8_e4m3fn, device="cuda"
        )
        k = torch.empty(
            (1, 4, 128, 128), dtype=torch.float8_e4m3fn, device="cuda"
        )
        v = torch.empty_like(k)
        q_scale = torch.empty((1, 4, 64), dtype=torch.float32, device="cuda")
        k_scale = torch.empty((1, 4, 8), dtype=torch.float32, device="cuda")
        v_scale = torch.empty((4, 128), dtype=torch.float32, device="cuda")
        q2k = torch.empty((1, 4, 1, 2), dtype=torch.int32, device="cuda")
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
def test_sm100_split_combine_aot_hit_never_jits(monkeypatch, device_arch):
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
    q = torch.empty((1, 2, 64, 128), dtype=torch.bfloat16)
    o_partial = torch.empty((1, 4, 64, 128), dtype=torch.float32)
    lse_partial = torch.empty((1, 4, 64), dtype=torch.float32)
    out, lse = bsa_attn_interface._combine_blk64_kv_bucketed_partials(
        q,
        o_partial,
        lse_partial,
        2,
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
