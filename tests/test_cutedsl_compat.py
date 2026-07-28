import contextlib
from dataclasses import dataclass
from typing import Tuple

import cutlass
import cutlass.cute._tvm_ffi_args_spec_converter as converter
from cutlass.base_dsl.tvm_ffi_builder import spec
import pytest

from block_sparse_attention.csrc.utils import copy_utils, cute_dsl_utils


@pytest.mark.parametrize(
    ("version", "expected"),
    (
        ("4.5.2", False),
        ("4.6.0", True),
        ("4.6.1", True),
        ("4.7.0.dev0", True),
    ),
)
def test_cute_dsl_bulk_copy_self_election_version(
    monkeypatch,
    version,
    expected,
):
    monkeypatch.setattr(copy_utils.cutlass, "__version__", version)
    assert copy_utils._cute_dsl_bulk_copy_self_elects() is expected


@pytest.mark.parametrize("version", ("unknown", None))
def test_cute_dsl_bulk_copy_self_election_rejects_unknown_version(
    monkeypatch,
    version,
):
    monkeypatch.setattr(copy_utils.cutlass, "__version__", version)
    with pytest.raises(RuntimeError, match="Cannot parse CUTLASS DSL version"):
        copy_utils._cute_dsl_bulk_copy_self_elects()


def test_bulk_copy_elect_one_uses_version_selected_context(monkeypatch):
    monkeypatch.setattr(copy_utils, "_BULK_COPY_SELF_ELECTS", True)
    assert isinstance(copy_utils.bulk_copy_elect_one(), contextlib.nullcontext)

    elected = contextlib.nullcontext()
    calls = []

    def fake_elect_one():
        calls.append(None)
        return elected

    monkeypatch.setattr(copy_utils, "_BULK_COPY_SELF_ELECTS", False)
    monkeypatch.setattr(copy_utils.cute.arch, "elect_one", fake_elect_one)
    assert copy_utils.bulk_copy_elect_one() is elected
    assert len(calls) == 1


def test_constexpr_tvm_ffi_converter_handles_tuple_field():
    @dataclass
    class Params:
        tile_shape: cutlass.Constexpr[Tuple[int, int]]

    converted = converter._convert_single_arg(
        Params((64, 128)),
        "params",
        Params,
        converter.ConverterContext(),
    )

    assert isinstance(converted, spec.TupleParam)
    assert len(converted.params) == 1
    assert isinstance(converted.params[0], spec.ConstNone)


def test_constexpr_tvm_ffi_converter_forwards_new_signature(monkeypatch):
    calls = []
    sentinel = object()

    def original(arg, arg_name, arg_type, ctx, *, is_constexpr=False):
        calls.append((arg, arg_name, arg_type, ctx, is_constexpr))
        return sentinel

    monkeypatch.setattr(converter, "_convert_single_arg", original)
    cute_dsl_utils._install_constexpr_tvm_ffi_converter()
    wrapped = converter._convert_single_arg
    ctx = object()

    assert wrapped(3, "value", int, ctx, is_constexpr=True) is sentinel
    assert calls == [(3, "value", int, ctx, True)]


def test_constexpr_tvm_ffi_converter_forwards_legacy_signature(monkeypatch):
    calls = []
    sentinel = object()

    def original(arg, arg_name, arg_type, ctx):
        calls.append((arg, arg_name, arg_type, ctx))
        return sentinel

    monkeypatch.setattr(converter, "_convert_single_arg", original)
    cute_dsl_utils._install_constexpr_tvm_ffi_converter()
    wrapped = converter._convert_single_arg
    ctx = object()

    assert wrapped(3, "value", int, ctx, is_constexpr=True) is sentinel
    assert calls == [(3, "value", int, ctx)]

    constexpr_type = cutlass.Constexpr[Tuple[int, int]]
    converted = wrapped(
        (64, 128),
        "tile_shape",
        constexpr_type,
        ctx,
        is_constexpr=True,
    )
    assert isinstance(converted, spec.ConstNone)
    assert calls == [(3, "value", int, ctx)]
