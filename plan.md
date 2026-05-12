# BSA blk64 CSR K2Q High-Performance Plan

## Summary

Implement a BSA-specific runtime-topK `q2k -> k2q CSR` conversion path and make the blk64 backward attention kernel consume CSR metadata directly. The implementation must reduce metadata memory relative to dense `k2q_index` while keeping the blk64 backward attention kernel-only runtime within 2% of the current path on the frozen benchmark shapes.

MiniMax's CSR builder is the algorithmic reference, but BSA cannot use its fixed-topK contract directly. In BSA, `block_sparse_num/topK` is a runtime value, `q2k_block_nums` must be supported, and blk64 is the target.

## Frozen Benchmark Shapes

Use the current `tests/test_flash_bwd.py` long-sequence benchmark set as the hard-gate baseline:

| bs | heads | seqlen_q | seqlen_k | dim | topK policy |
|---:|---:|---:|---:|---:|---|
| 1 | 4 | 116160 | 118528 | 128 | runtime `int(num_kv_blocks * 0.1)` |
| 1 | 4 | 109312 | 111040 | 128 | runtime `int(num_kv_blocks * 0.1)` |
| 1 | 4 | 216832 | 219200 | 128 | runtime `int(num_kv_blocks * 0.1)` |
| 1 | 1 | 216832 | 219200 | 128 | runtime `int(num_kv_blocks * 0.1)` |
| 1 | 4 | 349440 | 351168 | 128 | runtime `int(num_kv_blocks * 0.1)` |
| 1 | 4 | 695040 | 697408 | 128 | runtime `int(num_kv_blocks * 0.1)` |
| 1 | 1 | 695040 | 697408 | 128 | runtime `int(num_kv_blocks * 0.1)` |

The 2% gate applies to the blk64 backward attention kernel-only runtime. CSR conversion latency, end-to-end backward latency, and peak memory are measured and optimized separately.

## Phased Implementation

### Phase A: Baseline And Tracking

- Rewrite `progress.md` from scratch and keep it current.
- Record current dense/qbucket backward kernel-only time, current q2k->dense-k2q conversion time, end-to-end time, and peak memory for the frozen shapes.
- Add benchmark hooks for CSR conversion latency and CSR bwd kernel-only timing.

### Phase B: CSR Contract And Reference

- Add BSA CSR metadata with rows keyed by `(batch, head, kv_block)`.
- Use:
  - `k2q_row_ptr`: `int32 [B, H, num_kv_blocks + 1]`, storing global offsets into `k2q_q_indices`
  - `k2q_q_indices`: `int32 [total_edges]`
- Store attending `q_block_idx` values in each row.
- Keep row payloads ascending by `q_block_idx`.
- Support runtime `block_sparse_num`, runtime `max_kv`, and optional `q2k_block_nums`.
- Treat duplicate edges as unsupported in v1; test inputs must avoid duplicates.
- Fixed-count allocation uses `total_edges = B * H * Q_blocks * block_sparse_num`.
- Variable-count allocation computes per-BH edge counts and prefix offsets on GPU, then performs one D2H scalar copy of `total_edges` for exact packed allocation.

### Phase C: Runtime-topK CSR Builder Kernel

- Implement a BSA-specific CUDA extension inspired by MiniMax's histogram, prefix, tile-prefix, scatter pipeline.
- Do not specialize on fixed topK values such as `{4, 8, 16, 32}`.
- Support blk64 and current BSA q2k layout `[B, H, Q_blocks, max_kv]`.
- Use one runtime-topK design for fixed and variable block counts.
- Optimize with Nsight Compute after correctness is established.

### Phase D: blk64 Backward CSR Path

- Add an explicit CSR path to `bsa_attn_bwd`.
- Allow callers to pass prebuilt CSR metadata to avoid rebuilding.
- When CSR metadata is absent and CSR path is requested, build it from q2k before launching bwd.
- Modify the blk64 backward kernel metadata reads from dense:
  - `iter_count = k2q_num[kv]`
  - `q_block_idx = k2q_index[kv, iter]`
- To CSR:
  - `start = row_ptr[kv]`
  - `end = row_ptr[kv + 1]`
  - `iter_count = end - start`
  - `q_block_idx = q_indices[start + iter]`
- Keep the existing dense/qbucket paths as fallback and A/B baselines.

### Phase D2: Fused CSR Schedule And Scheduled Backward

- Extend the BSA CSR builder to optionally emit backward schedule chunks in the same build pass.
- Use MiniMax's fused schedule generation as the reference pattern, but keep BSA's runtime-topK and blk64 contracts.
- Emit per-(batch, head) schedule metadata:
  - `k2q_schedule_metadata`: `int32 [B, H, work_capacity_per_bh, 4]`
  - fields: `(kv_block, q_indices_start, q_count, reserved)`
  - `k2q_schedule_work_counts`: `int32 [B, H]`
- Support two schedule modes:
  - `row_chunk`: split long CSR rows by runtime `schedule_target_q_blocks`.
  - `qrange`: emit qbucket-like `(q_range, kv_block)` tasks over total-packed CSR rows; this is the default for customer sink/random-topK shapes.
- Add a CSR scheduled bwd variant that reuses qbucket's fp32 accumulator flow for split-row dK/dV accumulation.
- Keep the unscheduled CSR bwd path as the 2% regression baseline until the scheduled variant passes full NCU validation.

### Phase E: NCU/SASS Driven Optimization

- Profile every runnable CSR builder and CSR bwd version with Nsight Compute.
- Track DRAM and L2 throughput, SM occupancy, warp stalls, atomic pressure, shared-memory conflicts, branch divergence, eligible warps, issue active, and register pressure.
- Export CUBIN/SASS for CuTeDSL kernels when needed and inspect actual memory, atomic, barrier, and async-copy instructions.
- Consult NVIDIA CUDA, PTX, and Blackwell documentation during optimization.
- Allow large kernel rewrites if NCU shows a meaningful payoff.

## Acceptance Criteria

- Correctness matches the current dense converter and blk64 backward outputs.
- CSR metadata materially reduces peak metadata memory versus dense `k2q_index`.
- CSR builder supports runtime topK and `q2k_block_nums`.
- Frozen-shape blk64 backward attention kernel-only regression is <= 2%.
- `progress.md` contains benchmark and NCU evidence for the final path.
- CSR scheduled bwd has targeted correctness, full frozen/customer NCU data, and no >2% main-kernel regression before replacing the unscheduled CSR path for hotspot-shaped workloads.

## Current Follow-up TODO

- Budget and optimize the remaining variable-count sizing sync.
  - Current target design uses GPU-side `q2k_block_nums.sum(dim=2)` + prefix offsets, with one D2H scalar copy of `total_edges` for exact allocation.
  - Candidate production options: caller-supplied total edge budget, persistent scratch allocator capacity, or an async allocator path that hides the scalar sync.
- Add int64 CSR offset support if production shapes can exceed `int32` total edge capacity.
- Continue NCU tuning for CSR conversion on the largest frozen shapes; cfg5 was previously dominated by hist/scatter and measured around 3.3 ms total builder kernel time.
- Continue tuning internal-build benchmark overhead for `csr_scheduled`; current qrange kernels are competitive, but short customer shapes still expose temporary allocation/rebuild cost when CSR is rebuilt inside every bwd call.
- Keep NCU as the final bwd performance gate:
  - Event timing is useful for prebuilt-wrapper regressions but can show order/frequency noise.
  - NCU main `kernel_cutlass_bwd` duration is the authoritative 2% gate for CSR metadata load overhead.
