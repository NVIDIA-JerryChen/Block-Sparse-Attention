# BSA blk64 CSR K2Q Progress

## Current Status

- Status: total-packed CSR refactor implemented; fused qrange CSR schedule output and CSR scheduled bwd implemented. Targeted correctness passes, customer prebuilt-CSR benchmark passes the 2% per-shape qbucket gate, and NCU confirms the scheduled CSR main bwd/builder kernels are faster than qbucket on checked customer hotspot shapes.
- Branch/worktree note: `CLAUDE.md` has pre-existing unrelated modifications and is intentionally untouched.
- `progress.md` was rewritten from scratch per request.

## Locked Requirements

- Implement CSR compressed k2q metadata for blk64 backward.
- Implement a high-performance q2k -> k2q CSR conversion kernel.
- `topK/block_sparse_num` is a runtime value; do not rely on fixed `{4, 8, 16, 32}` specialization.
- Support `q2k_block_nums` variable counts.
- Use MiniMax's CSR builder as an algorithmic reference, not a direct drop-in dependency.
- Keep blk64 backward attention kernel-only performance within 2% of the current path on frozen benchmark shapes.
- Use Nsight Compute and, when useful, CUBIN/SASS inspection to guide optimization.

## Frozen Benchmark Shapes

| bs | heads | seqlen_q | seqlen_k | dim |
|---:|---:|---:|---:|---:|
| 1 | 4 | 116160 | 118528 | 128 |
| 1 | 4 | 109312 | 111040 | 128 |
| 1 | 4 | 216832 | 219200 | 128 |
| 1 | 1 | 216832 | 219200 | 128 |
| 1 | 4 | 349440 | 351168 | 128 |
| 1 | 4 | 695040 | 697408 | 128 |
| 1 | 1 | 695040 | 697408 | 128 |

## Task Tracker

| Phase | Task | Status | Notes |
|---|---|---|---|
| A | Write plan.md | done | Captures phased implementation and optimization gates. |
| A | Rewrite progress.md | done | This file is the new progress log. |
| B | Add CSR Python API and reference helpers | done | Added `convert_q2k_to_k2q_csr` and CSR materialization helper. |
| C | Add runtime-topK CUDA CSR builder | done, needs more tuning | BSA layout, blk64, fixed/variable counts; no fixed-topK specialization. |
| D | Add blk64 bwd CSR metadata path | done, preliminary gate pass | Explicit CSR path, prebuilt dense metadata, prebuilt CSR metadata, reusable workspace support, and total-packed q_indices support added. |
| E | Run correctness tests | targeted pass | Total-packed CSR builder/bwd/prebuilt parity tests pass; broader regression suite still pending. |
| E | Add kernel-only/prebuilt benchmark harness | done | `python tests/test_flash_bwd.py csr_benchmark`; default uses interleaved dense/CSR timing, prebuilt metadata, and reusable workspace. |
| E | Run frozen benchmark shapes | done, preliminary | Interleaved event timing is mostly within 2%, but event timing has order/frequency noise; NCU main-kernel gate is authoritative. |
| F | NCU validate bwd dense vs CSR | done | Post-refactor full 7-shape gate passes; all main `kernel_cutlass_bwd` deltas are within 2%. |
| F | NCU optimize CSR conversion | done, continue later | Added adaptive Q partitioning for small edge-count shapes; cfg0/cfg1 CSR builder NCU total improved. |
| G | Production interface and docs | done | Explicit CSR/prebuilt-dense/prebuilt-CSR/workspace API and benchmark/NCU commands recorded here. |
| H | Refactor CSR storage to total-packed layout | done | `q_indices` is now `[total_edges]`; `row_ptr` stores global offsets. |
| H | Remove max-per-BH variable allocation | done, residual scalar sync | Variable path now computes per-BH offsets on GPU and does one D2H `total_edges` scalar copy for allocation. |
| I | Add int64 CSR offset support | TODO | Current row_ptr remains int32 and fails fast if `total_edges > int32_max`. |
| J | Fuse schedule generation into CSR builder | done, needs NCU tuning | Builder can return `[B,H,work_capacity,4]` schedule metadata and `[B,H]` work counts in the same CSR build. |
| J | Add CSR scheduled bwd variant | done, prototype | `bsa_attn_bwd_csr_scheduled` reuses the qbucket fp32 accumulator path and consumes total-packed CSR q-indices plus fused schedule metadata. |
| J | Wire tests/benchmark/NCU entrypoints | done | Added scheduled correctness tests, `BSA_NCU_IMPL=csr_scheduled`, `BSA_NCU_IMPL=csr_schedule` for conversion, and `csr_scheduled` benchmark impl. |
| J | Add qrange CSR schedule mode | done | Schedule capacity is `num_q_groups * num_kv_blocks`; default qrange width is now 1024 Q-blocks and remains runtime-tunable via `BSA_BWD_CSR_QRANGE_BLOCKS`. |
| J | NCU optimize scheduled builder/bwd | done, continue tuning | Customer cfg3/cfg5 NCU shows scheduled CSR main bwd and builder kernels faster than qbucket. Internal-build customer event timing still has small allocation-related regressions on two short shapes. |
| K | Add customer prebuilt CSR benchmark mode | done | `bench_bsa.py --bwd-impl all_prebuilt` compares qbucket with CSR metadata already stored in compressed form. |
| K | Optimize internal CSR builder allocation/reuse | TODO | The qrange kernel path is fast, but customer `do_bench` internal-build mode still pays temporary tensor allocation/rebuild overhead on short shapes. |

## Performance Log

- Latest qrange CSR schedule update:
  - Implemented CSR schedule mode `qrange`, which emits qbucket-like `(q_range, kv)` work over total-packed CSR rows. This replaces the earlier row-chunk schedule for the default scheduled CSR path.
  - Default qrange width changed to `1024` Q blocks, with runtime override `BSA_BWD_CSR_QRANGE_BLOCKS=<blocks>`.
  - Correctness:
    - `python -m py_compile bench_bsa.py bsa_attn_interface.py utils/block_sparse_csr.py csrc/bwd/sm100_blk64/build_k2q_csr/__init__.py tests/test_flash_bwd.py`: passed.
    - `git diff --check`: passed.
    - `pytest -q tests/test_flash_bwd.py::test_convert_q2k_to_k2q_csr_sm100_blk64 tests/test_flash_bwd.py::test_flash_bwd_csr_scheduled_matches_dense_prebuilt_sm100_blk64`: passed 8 cases.
  - Customer benchmark, CSR storage prebuilt:
    - Command: `BSA_BWD_CSR_QRANGE_BLOCKS=1024 python bench_bsa.py --bwd --bwd-impl all_prebuilt --warmup 5 --runs 20`.
    - `csr_scheduled_prebuilt` vs qbucket geomean: `-0.33%` faster.
    - Per-shape deltas vs qbucket: `[-0.03%, -0.05%, -0.60%, -1.59%, -0.86%, +1.06%, +0.35%, -0.90%]`.
    - No customer shape regresses by more than 2% when CSR metadata is already stored in compressed form.
  - Customer benchmark, internal CSR rebuild each bwd:
    - Command: `BSA_BWD_CSR_QRANGE_BLOCKS=1024 python bench_bsa.py --bwd --bwd-impl all --warmup 5 --runs 20`.
    - `csr_scheduled` vs qbucket geomean: `+0.08%` slower.
    - Per-shape deltas vs qbucket: `[-0.55%, -0.18%, -0.65%, -2.22%, -0.82%, +3.23%, +2.18%, -0.22%]`.
    - Interpretation: the scheduled CSR kernel path is competitive, but short-shape internal rebuild mode still exposes allocation/rebuild overhead. Keep allocation/reuse optimization as a follow-up if the customer requires q2k-to-CSR rebuild inside every bwd call.
  - Customer NCU hotspot checks:
    - Log directory: `/tmp/bsa_ncu_customer_qrange_20260512_100449`.
    - cfg3 `480P-30s-H1`: qbucket main bwd `18.623 ms`, scheduled CSR main bwd `17.896 ms`; qbucket builder kernels `0.891 ms`, scheduled CSR builder+qrange schedule kernels `0.174 ms`.
    - cfg5 `480P-15s-H4`: qbucket main bwd `18.570 ms`, scheduled CSR main bwd `18.399 ms`; qbucket builder kernels `0.488 ms`, scheduled CSR builder+qrange schedule kernels `0.129 ms`.
    - NCU supports the conclusion that remaining internal-build event regressions are not from the main scheduled bwd kernel.
- Note: the latest post-refactor full NCU bwd gate is recorded below. Older frozen-shape benchmark and conversion tables remain as historical context.
- Correctness:
  - Total-packed refactor verification:
    - `python -m py_compile bsa_attn_interface.py utils/block_sparse_csr.py csrc/bwd/sm100_blk64/flash_bwd_sm100.py tests/test_flash_bwd.py csrc/bwd/sm100_blk64/build_k2q_csr/__init__.py`: passed.
    - `pytest -q tests/test_flash_bwd.py::test_convert_q2k_to_k2q_csr_sm100_blk64 --tb=short -s`: passed fixed and variable counts; asserts `q_indices.ndim == 1`, exact `numel == total_edges`, and global `row_ptr` starts.
  - `pytest -q tests/test_flash_bwd.py::test_flash_bwd_csr_sm100_blk64 --tb=short -s`: passed fixed and variable counts.
  - `pytest -q tests/test_flash_bwd.py::test_flash_bwd_csr_matches_dense_prebuilt_sm100_blk64 --tb=short -s`: passed 6 cases.
  - `pytest -q tests/test_flash_bwd.py::test_flash_bwd_sm100_blk64[128-256-False] --tb=short -s`: baseline path passes after CuTeDSL `tmem_holding_buf` compatibility fix.
  - Qbucket atomic-add lowering issue fixed by passing `ptr.llvm_ptr` to `cute.arch.atomic_add`.
  - `pytest -q tests/test_flash_bwd.py::test_flash_bwd_qbucket_sm100_blk64 tests/test_flash_bwd.py::test_flash_bwd_qbucket_multi_group_sm100_blk64`: passed 8 cases.
  - `pytest -q tests/test_flash_bwd.py::test_convert_q2k_to_k2q_csr_sm100_blk64`: passed fixed and variable counts, including fused schedule metadata contiguity checks.
  - `pytest -q tests/test_flash_bwd.py::test_flash_bwd_csr_scheduled_matches_dense_prebuilt_sm100_blk64`: passed 6 cases. The scheduled path uses `5e-3` absolute tolerance against dense-prebuilt because split-row fp32 atomic accumulation changes accumulation order.
  - `python -m py_compile bsa_attn_interface.py utils/block_sparse_csr.py csrc/bwd/sm100_blk64/flash_bwd_sm100.py csrc/bwd/sm100_blk64/flash_bwd_sm100_qbucket.py tests/test_flash_bwd.py`: passed.
  - `python -m py_compile bsa_attn_interface.py utils/block_sparse_csr.py tests/test_flash_bwd.py csrc/bwd/sm100_blk64/flash_bwd_sm100_qbucket.py csrc/bwd/sm100_blk64/build_k2q_csr/__init__.py`: passed after scheduled path wiring.
  - Dense-prebuilt vs CSR-prebuilt bwd smoke with valid torch-reference `out/lse`: `dq maxdiff=4.8828125e-4`, `dk/dv maxdiff=0`.
  - Scheduled smoke against dense-prebuilt on fixed/full case: `dq maxdiff=1.52587890625e-05`, `dk/dv maxdiff=0`.
- Conversion timing sample on B200, shape `B=1,H=1,Q_blocks=1024,KV_blocks=2048,topK=128`:
  - dense converter median: `0.1417 ms`
  - CSR converter median: `0.0486 ms`
  - dense metadata elements: `2,099,200 int32` including index and counts
  - CSR metadata elements: `133,121 int32` including row_ptr and q_indices
- Frozen-shape prebuilt-metadata bwd benchmark on B200:
  - Command: `BSA_BWD_BENCH_ITERS=5 BSA_BWD_BENCH_CONV_ITERS=1 BSA_BWD_WARMUP=2 python tests/test_flash_bwd.py csr_benchmark`
  - Harness details: `order=interleave`, deterministic `sliding` topK pattern, `topK=int(0.1 * num_kv_blocks)` rounded even, `block_sizes=full`, dense and CSR metadata are prebuilt before bwd timing, workspace is preallocated and reused.
  - Caveat: this is event timing of the prebuilt bwd wrapper path, not NCU-isolated main bwd kernel duration. It still includes required zeroing and helper kernels, but excludes q2k->k2q conversion.

| bs | heads | seqlen_q | seqlen_k | topK | dense prebuilt | CSR prebuilt | CSR delta | dense meta | CSR meta | CSR conv |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 116160 | 118528 | 186 | 11.547 ms | 11.675 ms | +1.11% | 51.3 MiB | 5.2 MiB | 0.099 ms |
| 1 | 4 | 109312 | 111040 | 174 | 10.398 ms | 10.318 ms | -0.77% | 45.2 MiB | 4.6 MiB | 0.089 ms |
| 1 | 4 | 216832 | 219200 | 342 | 39.656 ms | 40.113 ms | +1.15% | 177.1 MiB | 17.7 MiB | 0.175 ms |
| 1 | 1 | 216832 | 219200 | 342 | 9.901 ms | 9.833 ms | -0.69% | 44.3 MiB | 4.4 MiB | 0.090 ms |
| 1 | 4 | 349440 | 351168 | 548 | 104.505 ms | 104.681 ms | +0.17% | 457.2 MiB | 45.7 MiB | 0.451 ms |
| 1 | 4 | 695040 | 697408 | 1090 | 435.878 ms | 438.299 ms | +0.56% | 1805.9 MiB | 180.8 MiB | 2.208 ms |
| 1 | 1 | 695040 | 697408 | 1090 | 104.296 ms | 104.839 ms | +0.52% | 451.5 MiB | 45.2 MiB | 0.568 ms |

  - Metadata compression is about 10x for these topK=10% shapes.
  - CSR conversion is about 7x-16x faster than the dense k2q conversion in the frozen-shape harness.
  - Non-interleaved dense-first / CSR-first runs showed order-dependent timing noise, so the default harness was changed to interleave the implementations and alternate per-iteration order.
  - After CSR builder adaptive Q partitioning, one latest event run showed an event-timing outlier on cfg1 (`+6.02%`) while NCU main-kernel duration remained `+0.06%`; use NCU main-kernel duration for the bwd 2% gate.
- Post-refactor NCU bwd kernel-duration gate on B200:
  - Command template: `BSA_NCU_CONFIG_ID=<id> BSA_NCU_IMPL=<dense|csr> BSA_NCU_REPEATS=1 BSA_NCU_WARMUP=1 ncu --profile-from-start off --target-processes all --kernel-name regex:kernel_cutlass_ --metrics gpu__time_duration.sum --print-units base --print-summary per-kernel python tests/test_flash_bwd.py ncu_bwd`
  - NCU prints a benign Python shutdown `LookupError: unknown encoding: utf-8-sig` after profiling in this environment; commands returned usable kernel reports.
  - Log directory: `/tmp/bsa_ncu_post_refactor_20260512_063518`.
  - Harness note: dense NCU bwd now defaults to `BSA_NCU_DENSE_FROM_CSR=1`, which builds equivalent dense metadata via the CSR builder/materializer before `cudaProfilerStart()`. This avoids an unrelated Triton driver initialization failure under NCU; the profiled dense bwd kernel still consumes dense k2q metadata.

| cfg | bs | heads | seqlen_q | seqlen_k | topK | dense main bwd | CSR main bwd | CSR delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 4 | 116160 | 118528 | 186 | 19.717 ms | 19.483 ms | -1.18% |
| 1 | 1 | 4 | 109312 | 111040 | 174 | 17.248 ms | 17.127 ms | -0.70% |
| 2 | 1 | 4 | 216832 | 219200 | 342 | 66.167 ms | 65.510 ms | -0.99% |
| 3 | 1 | 1 | 216832 | 219200 | 342 | 17.042 ms | 16.827 ms | -1.26% |
| 4 | 1 | 4 | 349440 | 351168 | 548 | 169.719 ms | 167.698 ms | -1.19% |
| 5 | 1 | 4 | 695040 | 697408 | 1090 | 667.350 ms | 659.303 ms | -1.21% |
| 6 | 1 | 1 | 695040 | 697408 | 1090 | 167.691 ms | 165.607 ms | -1.24% |

  - The post-refactor NCU-isolated main bwd kernel gate passes; no shape shows a CSR regression beyond the 2% threshold.
  - `sum_OdO` and final `convert` kernels are metadata-independent and showed matching durations between dense and CSR.
- Scheduled CSR NCU smoke on cfg0:
  - Bwd command: `BSA_NCU_CONFIG_ID=0 BSA_NCU_IMPL=csr_scheduled BSA_NCU_REPEATS=1 BSA_NCU_WARMUP=1 ncu --profile-from-start off --target-processes all --kernel-name regex:kernel_cutlass_ --metrics gpu__time_duration.sum --print-units base --print-summary per-kernel python tests/test_flash_bwd.py ncu_bwd`
  - Scheduled qbucket main bwd: `19.641 ms`.
  - Compared with the earlier cfg0 dense main bwd `19.717 ms`, scheduled is `-0.39%`; compared with unscheduled CSR `19.483 ms`, scheduled is `+0.81%`.
  - Same cfg0 qbucket main bwd smoke: `20.202 ms`; scheduled CSR is about `2.8%` faster than qbucket for the main bwd kernel on this shape.
  - Builder command: `BSA_NCU_CONFIG_ID=0 BSA_NCU_IMPL=csr_schedule BSA_NCU_REPEATS=1 BSA_NCU_WARMUP=1 ncu --profile-from-start off --target-processes all --kernel-name regex:k2q_ --metrics gpu__time_duration.sum --print-units base --print-summary per-kernel python tests/test_flash_bwd.py ncu_convert`
  - Scheduled builder cfg0 kernels: hist `0.034 ms`, row prefix + schedule `0.013 ms`, tile prefix `0.046 ms`, scatter `0.046 ms`, total `0.140 ms`.
  - Relative to unscheduled cfg0 builder total `0.135 ms`, fused schedule output adds about `0.005 ms` in this smoke, mostly row-prefix schedule emission.
- Full 7-shape scheduled CSR NCU comparison:
  - Log directory: `/tmp/bsa_ncu_customer_shapes_20260512_072206`.
  - Bwd command template: `BSA_NCU_CONFIG_ID=<id> BSA_NCU_IMPL=<dense|csr|csr_scheduled> BSA_NCU_REPEATS=1 BSA_NCU_WARMUP=1 ncu --profile-from-start off --target-processes all --kernel-name regex:kernel_cutlass_ --metrics gpu__time_duration.sum --print-units base --print-summary per-kernel python tests/test_flash_bwd.py ncu_bwd`.
  - Qbucket used the same frozen case generator and profiled only `kernel_cutlass_` kernels.

| cfg | bs | heads | seqlen_q | seqlen_k | topK | dense | CSR | qbucket | CSR scheduled | sched vs dense | sched vs CSR | sched vs qbucket |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 4 | 116160 | 118528 | 186 | 19.711 ms | 19.482 ms | 20.199 ms | 19.648 ms | -0.32% | +0.85% | -2.73% |
| 1 | 1 | 4 | 109312 | 111040 | 174 | 17.249 ms | 17.112 ms | 17.839 ms | 17.282 ms | +0.19% | +0.99% | -3.12% |
| 2 | 1 | 4 | 216832 | 219200 | 342 | 66.199 ms | 65.501 ms | 68.220 ms | 66.644 ms | +0.67% | +1.74% | -2.31% |
| 3 | 1 | 1 | 216832 | 219200 | 342 | 17.028 ms | 16.821 ms | 17.135 ms | 16.863 ms | -0.97% | +0.25% | -1.59% |
| 4 | 1 | 4 | 349440 | 351168 | 548 | 169.674 ms | 167.682 ms | 176.286 ms | 170.913 ms | +0.73% | +1.93% | -3.05% |
| 5 | 1 | 4 | 695040 | 697408 | 1090 | 667.410 ms | 659.268 ms | 697.380 ms | 671.859 ms | +0.67% | +1.91% | -3.66% |
| 6 | 1 | 1 | 695040 | 697408 | 1090 | 167.699 ms | 165.707 ms | 174.421 ms | 168.094 ms | +0.24% | +1.44% | -3.63% |

  - Geomean: CSR scheduled vs dense `+0.17%`; CSR scheduled vs unscheduled CSR `+1.30%`; CSR scheduled vs qbucket `-2.87%`.
  - Interpretation: scheduled CSR is consistently faster than qbucket on these shapes, but it is consistently slower than the unscheduled CSR bwd path. It remains within the 2% dense-regression gate.
  - Full 7-shape builder NCU comparison:

| cfg | CSR builder | CSR schedule builder | schedule overhead |
|---:|---:|---:|---:|
| 0 | 0.135 ms | 0.139 ms | +2.55% |
| 1 | 0.115 ms | 0.119 ms | +3.57% |
| 2 | 0.288 ms | 0.322 ms | +11.72% |
| 3 | 0.156 ms | 0.190 ms | +21.84% |
| 4 | 0.720 ms | 0.797 ms | +10.65% |
| 5 | 3.310 ms | 3.568 ms | +7.80% |
| 6 | 0.882 ms | 1.134 ms | +28.53% |

  - Builder geomean overhead from fused schedule output: `+12.04%`.
  - Absolute builder overhead is small on the short shapes but reaches `+0.258 ms` on cfg5 and `+0.252 ms` on cfg6; row-prefix schedule emission dominates the delta.
- Customer `bench_bsa.py` end-to-end bwd benchmark:
  - Harness source: `/home/scratch.cjerry_sw/BSA/bench_bsa.py`.
  - Method: reused customer `configs`, `make_config`, `prepare_inputs`, `bsa_attn_fwd`, `do_bench`, and `attn_tflops`; compared `client_default`, direct `qbuck`, `csr`, and `csr_scheduled`.
  - Settings: `warmup=5`, `runs=20`, `topk_ratio=0.1`, `seed=42`.
  - Important: this is customer end-to-end wrapper timing, not NCU main-kernel-only timing. It includes metadata build, workspace allocation/zeroing, and the customer's memory-pressure code in `do_bench`. The sparse pattern is random topK from `get_block_map`, not the previous frozen sliding pattern.

| cfg | config | S_q | S_k | H | topK | client default | qbucket | CSR | CSR scheduled |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 720P-30s-H1 | 695040 | 697408 | 1 | 1089 | 102.932 ms | 104.236 ms | 180.457 ms | 181.951 ms |
| 1 | 720P-30s-H4 | 695040 | 697408 | 4 | 1089 | 417.281 ms | 418.461 ms | 726.321 ms | 739.922 ms |
| 2 | 720P-15s-H4 | 349440 | 351168 | 4 | 548 | 107.428 ms | 107.798 ms | 172.106 ms | 173.492 ms |
| 3 | 480P-30s-H1 | 216832 | 219200 | 1 | 342 | 10.042 ms | 10.401 ms | 14.329 ms | 14.227 ms |
| 4 | 480P-30s-H4 | 216832 | 219200 | 4 | 342 | 41.172 ms | 41.435 ms | 56.998 ms | 60.729 ms |
| 5 | 480P-15s-H4 | 109312 | 111040 | 4 | 173 | 12.210 ms | 10.593 ms | 12.454 ms | 11.633 ms |
| 6 | 368P-30s-H4 | 116160 | 118528 | 4 | 185 | 13.972 ms | 12.182 ms | 14.314 ms | 13.602 ms |
| 7 | 368P-30s-H8 | 116160 | 118528 | 8 | 185 | 29.124 ms | 24.686 ms | 27.848 ms | 27.213 ms |

  - Geomean from customer benchmark:
    - `csr_scheduled` vs qbucket: `+38.40%` slower.
    - `csr` vs qbucket: `+39.29%` slower.
    - `csr_scheduled` vs `csr`: `-0.64%` faster, effectively parity overall with shape-dependent swings.
    - qbucket vs client default: `-4.69%` faster overall; default does not force qbucket on smaller shapes.
  - Interpretation: for the customer's random-topK end-to-end benchmark, qbucket is currently the best-performing bwd path. CSR/scheduled needs a kernel-level investigation on this random pattern before it can be recommended for performance, although it still provides the compressed metadata format.
  - Customer cfg0 pattern diagnostics:
    - `force_k_suffix_len=S_aud_padded + S_txt` creates block-level sink rows. For `720P-30s-H1`, k2q row counts have `p50=1052`, `p99=1130`, `max=10860`, and `37` rows exceed `5000`.
    - The previous frozen sliding cfg with the same long sequence is balanced: row counts are roughly `1081-1090`, `max=1090`.
    - Current CSR scheduled splits hot rows, but its default chunking still differs from qbucket: customer cfg0 has qbucket `108970` non-empty tasks with median iter count `110`; scheduled CSR has `53914` valid tasks with median iter count `256`.
    - Customer cfg0 NCU main-kernel check on the actual `bench_bsa.py` input: qbucket `175.458 ms`, unscheduled CSR `205.851 ms`, scheduled CSR `196.793 ms`. Scheduled improves CSR by about `4.4%`, but remains about `12.2%` slower than qbucket on the main bwd kernel.
- Total-packed refactor cfg0 conversion NCU smoke:
  - `BSA_NCU_CONFIG_ID=0 BSA_NCU_IMPL=csr ... python tests/test_flash_bwd.py ncu_convert`: hist `0.033984 ms`, row prefix `0.008160 ms`, tile prefix `0.046496 ms`, scatter `0.046432 ms`, total `0.135072 ms`.
- NCU sample on same shape:
  - Initial builder split produced only 8 CTAs for hist/scatter; NCU reported severe grid underutilization.
  - After changing Q-block partitioning to target SM-scale grids, hist improved from about `104.7 us` to `15.8 us`; scatter improved from about `173.2 us` to `15.6 us`.
  - Tile-prefix increased to about `39.1 us` because `G_total` increased; this is the next conversion-kernel optimization target.
  - NCU command used: `ncu --target-processes all --set basic --kernel-name regex:k2q_ --launch-count 4 ...`
- NCU CSR builder profile after adaptive Q partitioning:
  - Command template: `BSA_NCU_CONFIG_ID=<id> BSA_NCU_IMPL=csr BSA_NCU_REPEATS=1 BSA_NCU_WARMUP=1 ncu --profile-from-start off --target-processes all --kernel-name regex:k2q_ --metrics gpu__time_duration.sum --print-units base --print-summary per-kernel python tests/test_flash_bwd.py ncu_convert`
  - Optimization: small per-(B,H) edge-count shapes now target about 1 CTA/SM instead of 2 CTA/SM, reducing `G_total` and tile-prefix work. Override: `BSA_K2Q_CSR_CTAS_PER_SM=<N>`.

| cfg | hist | row prefix | tile prefix | scatter | total k2q kernels |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.034 ms | 0.008 ms | 0.047 ms | 0.046 ms | 0.135 ms |
| 1 | 0.029 ms | 0.008 ms | 0.042 ms | 0.036 ms | 0.115 ms |
| 2 | 0.069 ms | 0.010 ms | 0.138 ms | 0.072 ms | 0.289 ms |
| 3 | 0.043 ms | 0.010 ms | 0.046 ms | 0.057 ms | 0.156 ms |
| 4 | 0.202 ms | 0.013 ms | 0.226 ms | 0.280 ms | 0.720 ms |
| 5 | 1.078 ms | 0.022 ms | 0.405 ms | 1.808 ms | 3.312 ms |
| 6 | 0.275 ms | 0.021 ms | 0.122 ms | 0.462 ms | 0.880 ms |

  - Before this tuning, cfg0/cfg1 NCU totals were `0.153 ms` / `0.141 ms`; after tuning they are `0.135 ms` / `0.115 ms`.
  - Large shapes are dominated by hist/scatter; small shapes were tile-prefix heavy before adaptive partitioning.
  - SpeedOfLight sample on cfg5 CSR main bwd: memory throughput `80.23%`, L1/TEX `80.39%`, L2 `44.34%`, SM compute `53.54%`.
  - SpeedOfLight sample on cfg5 CSR builder: hist `1.08 ms`, scatter `1.81 ms`; both have low DRAM throughput and are dominated by sparse-indexing/atomic/control overhead, not raw DRAM bandwidth.

## Production Interface Notes

- `convert_q2k_to_k2q_csr(q2k_block_index, block_sparse_num, num_kv_blocks, q2k_block_nums=None)` returns:
  - `k2q_row_ptr`: `int32 [B, H, num_kv_blocks + 1]`
  - `k2q_q_indices`: `int32 [total_edges]`
- `convert_q2k_to_k2q_csr(..., return_schedule=True)` returns the two CSR tensors plus:
  - `k2q_schedule_metadata`: `int32 [B, H, work_capacity_per_bh, 4]`
  - `k2q_schedule_work_counts`: `int32 [B, H]`
  - schedule metadata fields are `(kv_block, q_indices_start, q_count, reserved)`
- `k2q_row_ptr` values are global offsets into `k2q_q_indices`.
- Fixed path uses `total_edges = B * H * Q_blocks * block_sparse_num`.
- Variable path computes per-BH edge counts and prefix offsets on GPU, then copies one `total_edges` scalar D2H for exact allocation.
- `bsa_attn_bwd(..., use_k2q_csr=True)` builds CSR metadata internally when prebuilt CSR is not supplied.
- `bsa_attn_bwd(..., k2q_row_ptr=row_ptr, k2q_q_indices=q_indices)` uses prebuilt CSR metadata and avoids conversion in the bwd call.
- `bsa_attn_bwd_csr_scheduled(...)` uses fused CSR schedule metadata and the qbucket fp32 accumulator path. It is currently a prototype path pending scheduled NCU tuning.
- `bsa_attn_bwd(..., prebuilt_k2q_block_index=dense_idx, prebuilt_k2q_block_nums=dense_num, use_k2q_csr=False)` is the dense prebuilt baseline path.
- `workspace` is an optional reusable scratch buffer for allocator-stable benchmarking or external scratch management; it is not CSR-specific and is zeroed before each launch.
- Benchmark entrypoints:
  - `python tests/test_flash_bwd.py csr_benchmark`
  - `python tests/test_flash_bwd.py ncu_bwd`
  - `python tests/test_flash_bwd.py ncu_convert`
  - scheduled bwd NCU: `BSA_NCU_IMPL=csr_scheduled python tests/test_flash_bwd.py ncu_bwd`
  - scheduled builder NCU: `BSA_NCU_IMPL=csr_schedule python tests/test_flash_bwd.py ncu_convert`

## Known Risks

- Runtime-topK CSR scatter must preserve q-sorted row payloads without fixed topK specialization.
- Variable `q2k_block_nums` now avoids max-per-BH sizing but still has one D2H scalar sync for `total_edges`; benchmark and, if needed, hide or replace this sync.
- Int64 CSR offset support is TODO; current int32 row_ptr rejects `total_edges > 2^31 - 1`.
- CSR metadata reads in the blk64 bwd kernel passed NCU main-kernel duration gate, but should be rechecked after any bwd kernel scheduling or metadata-load rewrite.
- CSR scheduled bwd has only targeted correctness coverage so far. It still needs full frozen-shape NCU validation, schedule target tuning, and SASS/NCU inspection before production use.
- Large-shape CSR builder hist/scatter remain the next conversion optimization target if conversion latency becomes end-to-end critical.
- Dense prebuilt k2q metadata remains useful as the stable A/B baseline for kernel-only benchmarks without conversion time.
