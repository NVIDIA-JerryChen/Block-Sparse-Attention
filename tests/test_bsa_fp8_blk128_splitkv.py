"""Native FP8 split producers and FP32 combine on SM100/SM103."""

import pytest
import torch

from block_sparse_attention import bsa_fp8_fwd, quantize_sage_bhsd
from test_bsa_fp8_blk128 import (
    assert_accuracy,
    dequantize,
    make_inputs,
    make_metadata,
    sparse_reference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Native FP8 blk128 split-KV requires SM100/SM103",
)


@pytest.mark.parametrize("kv_splits", [1, 2, 4, 8])
@pytest.mark.parametrize("sizes_mode", [0, 1, 2, 3])
@pytest.mark.parametrize("variable", [False, True])
def test_split_dynamic_metadata(kv_splits, sizes_mode, variable):
    torch.manual_seed(3700 + sizes_mode)
    # Change every dynamic extent while reusing the compiled variant. Include
    # empty splits, odd counts, physical tails, and poisoned inactive indices.
    for batch, heads, sq, sk, topk in [
        (1, 1, 1, 1, 1),
        (2, 3, 257, 913, 7),
        (1, 2, 389, 1153, 9),
    ]:
        q, k, v = make_inputs(batch, heads, sq, sk, gain=0.5)
        index, nums, sizes = make_metadata(
            batch, heads, sq, sk, topk, variable=variable, sizes_mode=sizes_mode
        )
        quantized = quantize_sage_bhsd(q, k, v)
        kwargs = dict(
            sparse_block_size=128, kv_splits=kv_splits,
            block_sizes=sizes, q2k_block_nums=nums,
        )
        out = bsa_fp8_fwd(*quantized, index, topk, **kwargs)
        ref = sparse_reference(*dequantize(quantized), index, topk, nums, sizes)
        assert_accuracy(out, ref)
        original_ref = sparse_reference(q, k, v, index, topk, nums, sizes)
        assert_accuracy(out, original_ref, relative=0.08, absolute=0.3)
        repeated = bsa_fp8_fwd(*quantized, index, topk, **kwargs)
        assert torch.equal(out, repeated)
        if nums is not None:
            empty = nums.repeat_interleave(128, dim=-1)[..., :sq] == 0
            assert torch.count_nonzero(out[empty]) == 0
        # A cached split entry must consume new inputs rather than old outputs.
        replacement = list(quantized)
        replacement[2] = torch.zeros_like(quantized[2])
        zero = bsa_fp8_fwd(*replacement, index, topk, **kwargs)
        assert torch.count_nonzero(zero) == 0


@pytest.mark.parametrize("kv_splits", [2, 3, 4, 8])
def test_split_rescaling_and_zero_extents(kv_splits):
    torch.manual_seed(3821)
    q, k, v = make_inputs(1, 2, 257, 1665, gain=0.5)
    # Unequal split normalizers and mixed masked blocks exercise log-sum-exp
    # combine weights, including the producer's rounded-P factor of 128.
    k[:, :, 0:512] *= 4
    q[:, :, 0:128] *= 2
    index, nums, sizes = make_metadata(1, 2, 257, 1665, 13, variable=True, sizes_mode=3)
    nums.fill_(13)
    nums[0, 0, 0] = 0
    nums[0, 1, 1] = 1
    # Rebuild indices because the variable helper poisons initially dead slots.
    index.copy_(torch.arange(13, device="cuda", dtype=torch.int32))
    sizes[..., ::3] = 0
    sizes[0, 0].zero_()
    quantized = quantize_sage_bhsd(q, k, v)
    out = bsa_fp8_fwd(
        *quantized, index, 13, sparse_block_size=128,
        kv_splits=kv_splits, block_sizes=sizes, q2k_block_nums=nums,
    )
    ref = sparse_reference(*dequantize(quantized), index, 13, nums, sizes)
    assert_accuracy(out, ref)
    assert torch.count_nonzero(out[0, 0]) == 0
    repeated = bsa_fp8_fwd(
        *quantized, index, 13, sparse_block_size=128,
        kv_splits=kv_splits, block_sizes=sizes, q2k_block_nums=nums,
    )
    assert torch.equal(out, repeated)
