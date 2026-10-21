"""Exercise warp-role handoffs under Compute Sanitizer synccheck.

Numerical agreement alone does not detect aligned barriers in divergent roles.
Run this file with ``compute-sanitizer --tool synccheck --error-exitcode=1``.
"""

import pytest
import torch

from block_sparse_attention import bsa_fp8_blk128_fwd, quantize_sage_bhsd
from test_bsa_fp8_blk128 import assert_accuracy, dequantize, make_inputs, sparse_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Native FP8 blk128 requires SM100/SM103",
)


@pytest.mark.parametrize("kv_splits", [1, 4])
def test_subset_barriers_with_graph_and_dynamic_metadata(kv_splits):
    torch.manual_seed(1752)
    q, k, v = make_inputs(1, 2, 257, 641, gain=0.5)
    data = quantize_sage_bhsd(q, k, v)
    dequantized = dequantize(data)
    order = torch.tensor([5, 0, 3, 1, 4, 2], device="cuda", dtype=torch.int32)
    index = order.expand(1, 2, 3, 6).clone()
    counts = torch.tensor([[[6, 0, 2], [1, 4, 6]]], device="cuda", dtype=torch.int32)
    sizes = torch.tensor([128, 64, 0, 127, 128, 1], device="cuda", dtype=torch.int32)

    def attention():
        return bsa_fp8_blk128_fwd(
            *data, index, 6, block_sizes=sizes,
            q2k_block_nums=counts, kv_splits=kv_splits,
        )

    expected = attention()
    assert_accuracy(expected, sparse_reference(*dequantized, index, 6, counts, sizes))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        repeated = attention()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = attention()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(expected, repeated)

    # Reuse the graph while rows switch between empty and nonempty workloads.
    # Six blocks wrap the pipeline; the last Q and KV blocks have physical tails.
    for new_counts in (
        [[[6, 0, 2], [1, 4, 6]]],
        [[[0, 5, 1], [6, 0, 3]]],
        [[[0, 0, 0], [0, 0, 0]]],
    ):
        counts.copy_(torch.tensor(new_counts, device="cuda", dtype=torch.int32))
        index.copy_(order)
        inactive = torch.arange(6, device="cuda") >= counts[..., None]
        index.masked_fill_(inactive, 0x3FFFFFFF)
        graph.replay()
        reference = sparse_reference(*dequantized, index, 6, counts, sizes)
        assert_accuracy(captured, reference)
        empty = counts.repeat_interleave(128, dim=-1)[..., :q.shape[2]] == 0
        assert torch.count_nonzero(captured[empty]) == 0
