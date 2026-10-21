"""Public blk128 parity with the existing blk64 FP8 contract."""
import inspect
import pytest
import torch

from block_sparse_attention import bsa_fp8_blk128_fwd, bsa_fp8_fwd, quantize_sage_bhsd
from test_bsa_fp8_blk128 import assert_accuracy, dequantize, make_inputs, make_metadata, sparse_reference

requires_target = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="FP8 blk128 requires SM100/SM103",
)


def test_named_public_signature():
    signature = inspect.signature(bsa_fp8_blk128_fwd)
    expected = ["q_fp8", "k_fp8", "v_fp8", "q_sfs", "k_sfs", "v_sfs",
                "q2k_block_index", "topk_num", "softmax_scale",
                "block_sizes", "q2k_block_nums", "kv_splits"]
    assert list(signature.parameters) == expected
    assert signature.parameters["softmax_scale"].default is None
    for name in ("block_sizes", "q2k_block_nums", "kv_splits"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["block_sizes"].default is None
    assert signature.parameters["q2k_block_nums"].default is None
    assert signature.parameters["kv_splits"].default == 1
    from block_sparse_attention import bsa_fp8_blk64_fwd
    blk64_parameters = inspect.signature(bsa_fp8_blk64_fwd).parameters
    assert list(signature.parameters.values())[:-1] == list(blk64_parameters.values())
    from block_sparse_attention.bsa_fp8_blk128 import bsa_fp8_blk128_fwd as shim
    from block_sparse_attention.bsa_fp8 import bsa_fp8_blk128_fwd as generic_shim
    assert shim is generic_shim is bsa_fp8_blk128_fwd


def strided_copy(tensor):
    storage = torch.empty((*tensor.shape, 2), device=tensor.device, dtype=tensor.dtype)
    storage[..., 0].copy_(tensor)
    return storage[..., 0]


@requires_target
@pytest.mark.parametrize("splits", [1, 4])
@pytest.mark.parametrize("sizes_mode", [1, 2, 3])
def test_named_views_metadata_and_current_stream(splits, sizes_mode):
    torch.manual_seed(7013 + sizes_mode)
    q, k, v = make_inputs(2, 3, 129, 641, gain=0.5)
    data = quantize_sage_bhsd(q, k, v)
    index, nums, sizes = make_metadata(2, 3, 129, 641, 5, variable=True, sizes_mode=sizes_mode)
    expected = bsa_fp8_fwd(*data, index, 5, 0.17, sparse_block_size=128,
                           block_sizes=sizes, q2k_block_nums=nums, kv_splits=splits)
    # The original wrapper reshapes valid scale views and consumes the live
    # per-row prefix. The scalar count is ignored when row counts are present.
    views = [strided_copy(t) for t in data]
    views[3:] = [t.reshape(-1) for t in views[3:]]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = bsa_fp8_blk128_fwd(*views, strided_copy(index), 0, 0.17,
            block_sizes=strided_copy(sizes), q2k_block_nums=strided_copy(nums), kv_splits=splits)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(actual, expected)
    assert_accuracy(actual, sparse_reference(*dequantize(data), index, 5, nums, sizes, scale=0.17))
    assert actual.shape == q.shape and actual.dtype == torch.bfloat16
    assert actual.device == q.device


@requires_target
def test_named_default_policy_and_empty_optional_metadata():
    q, k, v = make_inputs(1, 2, 65, 257, gain=0.5)
    data = quantize_sage_bhsd(q, k, v)
    index, _, _ = make_metadata(1, 2, 65, 257, 3)
    default = bsa_fp8_blk128_fwd(*data, index, 3)
    explicit = bsa_fp8_blk128_fwd(*data, index, 3, kv_splits=1)
    empty = torch.empty(0, dtype=torch.int32, device=q.device)
    optional = bsa_fp8_blk128_fwd(*data, index, 3,
                                 block_sizes=empty, q2k_block_nums=empty)
    assert torch.equal(default, explicit) and torch.equal(default, optional)
    assert_accuracy(default, sparse_reference(*dequantize(data), index, 3))


@requires_target
@pytest.mark.parametrize("invalid", ["dtype", "dimension", "heads", "cpu_scale",
    "scale_dtype", "index_dtype", "index_rank", "count_dtype", "count_shape",
    "size_dtype", "size_shape", "topk", "split_zero", "split_float", "split_bool"])
def test_named_rejects_unsupported_inputs(invalid):
    q, k, v = make_inputs(1, 2, 65, 257, gain=0.5)
    data = list(quantize_sage_bhsd(q, k, v))
    index, nums, sizes = make_metadata(1, 2, 65, 257, 3)
    topk = 3
    kwargs = {}
    if invalid == "dtype": data[0] = data[0].bfloat16()
    elif invalid == "dimension": data[0] = data[0][..., :64].contiguous()
    elif invalid == "heads": data[1] = data[1][:, :1].contiguous()
    elif invalid == "cpu_scale": data[3] = data[3].cpu()
    elif invalid == "scale_dtype": data[4] = data[4].bfloat16()
    elif invalid == "index_dtype": index = index.long()
    elif invalid == "index_rank": index = index.flatten()
    elif invalid == "count_dtype": kwargs["q2k_block_nums"] = torch.ones(1, 2, 1, device=q.device)
    elif invalid == "count_shape": kwargs["q2k_block_nums"] = torch.ones(1, 2, 2, device=q.device, dtype=torch.int32)
    elif invalid == "size_dtype": kwargs["block_sizes"] = torch.ones(3, device=q.device)
    elif invalid == "size_shape": kwargs["block_sizes"] = torch.ones(2, device=q.device, dtype=torch.int32)
    elif invalid == "topk": topk = 0
    elif invalid == "split_zero": kwargs["kv_splits"] = 0
    elif invalid == "split_float": kwargs["kv_splits"] = 1.5
    elif invalid == "split_bool": kwargs["kv_splits"] = True
    with pytest.raises(AssertionError):
        bsa_fp8_blk128_fwd(*data, index, topk, **kwargs)
