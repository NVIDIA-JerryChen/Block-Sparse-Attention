# BSA blk64 CSR Scheduled Backward Plan

## Goal

Provide one production backward path for blk64 BSA:

- compact total-packed CSR k2q metadata
- fused qrange-split schedule generation in the CSR builder
- high-performance scheduled backward kernel
- no blk64 backward attention kernel regression beyond the 2% gate on frozen and customer shapes

## Production Contract

- Input sparsity remains q2k: `q2k_block_index [B, H, Q_blocks, max_kv]`.
- CSR output:
  - `k2q_row_ptr: int32 [B, H, num_kv_blocks + 1]`
  - `k2q_q_indices: int32 [total_edges]`
- Schedule output:
  - `k2q_schedule_metadata: int32 [B, H, work, 4]`
  - fields: `(kv_block, q_indices_start, q_count, q_group)`
  - `k2q_schedule_work_counts: int32 [B, H]`
- Schedule policy is fixed in production:
  - qrange width: `1024` Q blocks
  - split target: `128` Q blocks per work item
  - split only qrange slices whose `q_count > 128`

## Current Implementation Plan

1. Keep `bsa_attn_bwd` as the only public blk64 backward entry.
2. Build CSR plus schedule internally when prebuilt metadata is absent.
3. Allow all four prebuilt CSR/schedule tensors to be passed together for benchmark isolation.
4. Keep fwd correctness, bwd correctness, benchmark, and profile in root-level `test_bsa.py`.
5. Use the benchmark default case `seqlen=262144`, `bs=1`, `heads=40`, `topK=(4096, 2048, 1024, 512, 256, 128, 64, 32)`.
6. Remove obsolete alternate backward entry points, experimental schedule modes, and debug-only helpers from production code.

## Validation Plan

1. Compile check:
   - `python -m py_compile bsa_attn_interface.py csrc/common/block_sparse_csr.py test_bsa.py`
2. CSR correctness:
   - fixed topK and `q2k_block_nums` paths match a CPU reference CSR builder
   - schedule segments cover every CSR row exactly once
3. Backward correctness:
   - use `test_bsa_sm100` to check fwd correctness and compare `bsa_attn_bwd` against torch autograd reference
   - cover fixed and variable `q2k_block_nums` paths
   - cover `block_sizes=None` and explicit `block_sizes` paths
   - use torch reference `out/lse` for bwd so bwd correctness does not depend on fwd kernel output
4. Performance:
   - use `python test_bsa.py benchmark` for customer runtime benchmarks
   - use `python test_bsa.py profile --topk <value>` for NCU/NVTX profiling
   - use `--pattern sink` when profiling hotspot/attention-sink style q2k layouts
   - use NCU as the final kernel-level gate for conversion and backward kernels

## Remaining TODO

- Optimize or hide the one D2H scalar sync used by variable-count total-edge sizing.
- Add int64 CSR offset support if production shapes can exceed int32 total edge capacity.
- Continue NCU tuning for the CSR conversion kernel on the largest customer shapes.
- Re-run the full customer-shape NCU gate after every non-trivial builder or scheduled backward change.
