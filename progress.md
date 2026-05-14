# BSA blk64 CSR Scheduled Backward Progress

## Current State

- Public blk64 backward API is `bsa_attn_bwd`.
- `bsa_attn_bwd` always uses total-packed CSR k2q metadata plus the fused qrange-split schedule.
- Production schedule defaults to the adaptive qrange-split policy in
  `csrc/common/block_sparse_csr.py`.
- Validation-only schedule rollback is available with
  `BSA_CSR_SCHEDULE_POLICY=qbuck`, which uses qbucket-compatible qrange sizing
  and disables per-row qrange splitting.
- `convert_q2k_to_k2q_csr(..., return_schedule=True)` returns CSR plus schedule metadata.
- Fixed-count CSR allocation uses exact `B * H * Q_blocks * max_topk`.
- Variable-count CSR allocation uses GPU prefix sizing plus one D2H scalar read of `total_edges`.
- CSR row offsets remain int32; int64 support is tracked as TODO.

## Code Clean Status

| Item | Status | Notes |
|---|---|---|
| Public backward interface | done | Only `bsa_attn_bwd` is exposed for blk64 backward. |
| CSR builder schedule modes | done | Builder now emits only production qrange-split schedule. |
| Dense k2q converter | done | Removed dense Triton q2k->k2q helper. |
| Old backward kernels | done | Removed obsolete alternate blk64 backward files. |
| Tests/benchmark/profile | done | Root-level `test_bsa.py` follows the MiniMax-style single-file structure; pytest surface is `test_bsa_sm100` plus `test_convert_q2k_to_k2q_csr`. `test_bsa_sm100` checks fwd and bwd correctness in the same test case. |

## Latest Performance Conclusions

- Previous customer runtime event timing showed CSR qrange-split within the 2% gate versus the previous comparison baselines.
- Kernel-level NCU remained the authoritative gate for backward kernel regression.
- The largest CSR conversion cost previously observed was on the largest frozen/customer-style shape and remains a tuning TODO.
- dK/dV accumulation now uses a MiniMax-style TMA reduce path instead of per-element fp32 atomics:
  - tmem fragment is staged once into per-WG shared memory.
  - each compute WG owns 32 KV rows and reduces only those rows.
  - TMA reduce remains 32 fp32 columns per bulk op; no full/non-full TMA special case is used.
  - per-WG named barriers replace the previous cross-WG shared staging barrier.
- Current topK=32 uniform benchmark improved bwd runtime from ~85.9 ms to 59.521 ms after the per-WG row-owned staging and dKV postprocess change.
- Current bwd-only NCU for topK=32 uniform:
  - Report: `ncu_reports/bsa_blk64_bwd_topk32_postprocess_dkv.ncu-rep`
  - Duration: 79.49 ms under NCU replay
  - Memory Throughput: 60.09%
  - Compute Throughput: 37.94%
  - No Eligible: 81.59%
  - Stall Long Scoreboard: 8.95 inst
  - Stall Barrier: 3.72 inst, reduced from 7.45 inst in the earlier shared full-D staging profile.
- dQ accumulator now follows the FlashAttention/CUTLASS TMA-reduce structure:
  - global dQAccm workspace is linear row-major `(Q_acc, D, (H, B))`;
  - the reduce warp path slices the 128-row dQ tile into upper/lower 64-row q blocks;
  - each 64x32 half-tile is staged through the swizzled shared-memory view and written with `CopyReduceBulkTensorTileS2GOp`;
  - final dQ postprocess is a simple linear fp32 accumulator load plus bf16 store.
- The MiniMax-style 64-column dQ reduce stage was explored, but CuTe's TMEM load fragment profile changes for this BSA tile shape. The production path stays with the FA/CUTLASS 32-column stage because it preserves correctness and restores the previous ~59 ms bwd runtime.

## Remaining TODO

- Re-run full customer-shape NCU gate after confirming correctness.
- Optionally revisit a 64-column dQAccm reduce stage only if it can be made faster than the FA/CUTLASS 32-column TMA path without adding Q/K/V shape to the compile key.
- Optimize or hide the variable-count one-scalar D2H sizing sync.
- Add int64 CSR row offset support if production total edges can exceed int32.

## Latest Validation

- `python -m py_compile test_bsa.py bsa_attn_interface.py csrc/common/block_sparse_csr.py csrc/common/build_k2q_csr/__init__.py csrc/bwd/sm100_blk64/flash_bwd_sm100.py`: passed.
- `python test_bsa.py --help`: passed and exposes `benchmark` and `profile`.
- `BSA_BLK=64 pytest -q test_bsa.py --tb=short -s`: passed 6 tests.
- `BSA_BLK=64,128 pytest -q test_bsa.py --tb=short -s`: passed 10 tests.
- `python test_bsa.py profile --seqlen 512 --heads 2 --topk 4 --batch 1 --dim 128 --pattern sink`: passed smoke.
- `python test_bsa.py benchmark --seqlen 512 --heads 2 --topk 4 --warmup 0 --iters 1 --pattern uniform`: passed smoke.
- `CUDA_LAUNCH_BLOCKING=1 BSA_BLK=64 pytest -q test_bsa.py::test_bsa_sm100 --tb=short -s -x`: passed 4 tests after dKV TMA reduce tuning.
- `python test_bsa.py benchmark --topk 32 --warmup 2 --iters 5 --pattern uniform`: bwd 64.978 ms, bwd_internal 65.917 ms.
- `ncu --target-processes all --kernel-name-base function --kernel-name 'regex:kernel_cutlass_bwd_csrcbwdsm100_blk64flash_bwd_sm100.*' --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats ...`: report `/tmp/bsa_dkv_perwg_rows_final_bwd_detail.ncu-rep`.
- `python -m py_compile csrc/bwd/sm100_blk64/flash_bwd_sm100.py && pytest -q test_bsa.py::test_bsa_sm100 --tb=short -s -x`: passed 8 tests.
- `python test_bsa.py benchmark --topk 32 --warmup 2 --iters 5 --pattern uniform`: csr 0.713 ms, fwd 15.521 ms, bwd 59.521 ms, bwd_internal 61.133 ms.
- `ncu --target-processes all --kernel-name-base function --kernel-name 'regex:kernel_cutlass_bwd_csrcbwdsm100_blk64flash_bwd_sm100.*' --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --force-overwrite -o ncu_reports/bsa_blk64_bwd_topk32_postprocess_dkv python test_bsa.py profile --topk 32 --pattern uniform`: report `ncu_reports/bsa_blk64_bwd_topk32_postprocess_dkv.ncu-rep`.
- `pytest -q test_bsa.py::test_bsa_sm100 --tb=short -s -x`: passed 8 tests after the FA-linear/chunked dQ reduce experiment.
- `python test_bsa.py benchmark --topk 32 --warmup 2 --iters 5 --pattern uniform`: csr 0.715 ms, fwd 15.559 ms, bwd 70.034 ms, bwd_internal 70.763 ms for the previous chunked dQ experiment.
- `pytest -q test_bsa.py::test_bsa_sm100 --tb=short -s -x`: passed 8 tests after switching dQAccm to the FA/CUTLASS TMA-reduce path.
- `python test_bsa.py benchmark --topk 32 --warmup 2 --iters 5 --pattern uniform`: csr 0.710 ms, fwd 15.601 ms, bwd 59.436 ms, bwd_internal 60.973 ms.
- `python test_bsa.py benchmark --warmup 2 --iters 5 --pattern uniform`: completed the default customer topK sweep:
  - topK 4096: csr 22.772 ms, fwd 1758.525 ms, bwd 5201.161 ms, bwd_internal 5241.084 ms
  - topK 2048: csr 9.968 ms, fwd 805.988 ms, bwd 2622.739 ms, bwd_internal 2632.874 ms
  - topK 1024: csr 4.564 ms, fwd 388.480 ms, bwd 1321.398 ms, bwd_internal 1325.509 ms
  - topK 512: csr 2.990 ms, fwd 196.802 ms, bwd 667.871 ms, bwd_internal 670.555 ms
  - topK 256: csr 2.263 ms, fwd 100.697 ms, bwd 339.912 ms, bwd_internal 342.169 ms
  - topK 128: csr 1.032 ms, fwd 52.767 ms, bwd 179.984 ms, bwd_internal 181.722 ms
  - topK 64: csr 0.812 ms, fwd 28.635 ms, bwd 101.705 ms, bwd_internal 100.614 ms
  - topK 32: csr 0.723 ms, fwd 15.898 ms, bwd 61.325 ms, bwd_internal 62.040 ms
- `ncu --target-processes all --kernel-name-base function --kernel-name 'regex:kernel_cutlass_bwd_csrcbwdsm100_blk64flash_bwd_sm100.*' --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --force-overwrite -o ncu_reports/bsa_blk64_bwd_topk32_dqaccm_tma_linear python test_bsa.py profile --topk 32 --pattern uniform`: report `ncu_reports/bsa_blk64_bwd_topk32_dqaccm_tma_linear.ncu-rep`; Duration 79.76 ms under replay, Memory Throughput 59.91%, Compute Throughput 37.81%, No Eligible 81.65%.
- `bench_bsa.py` 8-case bwd A/B vs original qbucket (`335bec1` plus minimal CuTe API compatibility patch, forced `bsa_attn_bwd_qbucket`, warmup=1/runs=3):
  - 720P-30s-H1: current 89.242 ms vs qbucket 89.262 ms, +0.02%
  - 720P-30s-H4: current 374.774 ms vs qbucket 370.496 ms, -1.15%
  - 720P-15s-H4: current 94.891 ms vs qbucket 94.335 ms, -0.59%
  - 480P-30s-H1: current 9.490 ms vs qbucket 9.212 ms, -3.02%
  - 480P-30s-H4: current 37.065 ms vs qbucket 36.721 ms, -0.94%
  - 480P-15s-H4: current 9.773 ms vs qbucket 9.218 ms, -6.02%
  - 368P-30s-H4: current 11.017 ms vs qbucket 10.431 ms, -5.62%
  - 368P-30s-H8: current 21.385 ms vs qbucket 21.032 ms, -1.68%
  - Conclusion: current CSR scheduled path is not broadly faster than original qbucket on these video-pattern cases; large cases are near parity, smaller cases still need schedule/kernel tuning if the target is to exceed qbucket.
- Schedule rollback validation:
  - Added `BSA_CSR_SCHEDULE_POLICY=qbuck` to test the qbucket-compatible
    schedule independently of the dQ/dKV epilogue implementation.
  - `BSA_CSR_SCHEDULE_POLICY=qbuck pytest -q test_bsa.py::test_convert_q2k_to_k2q_csr --tb=short -s`: passed.
  - Same checkout, same `bench_bsa.py` 8-case bwd harness (`warmup=1/runs=3`):
    - 720P-30s-H1: adaptive 89.508 ms vs qbucket-schedule 90.164 ms, -0.73%
    - 720P-30s-H4: adaptive 374.974 ms vs qbucket-schedule 373.664 ms, +0.35%
    - 720P-15s-H4: adaptive 94.922 ms vs qbucket-schedule 93.730 ms, +1.26%
    - 480P-30s-H1: adaptive 9.392 ms vs qbucket-schedule 9.929 ms, -5.72%
    - 480P-30s-H4: adaptive 36.667 ms vs qbucket-schedule 37.671 ms, -2.74%
    - 480P-15s-H4: adaptive 9.796 ms vs qbucket-schedule 9.969 ms, -1.77%
    - 368P-30s-H4: adaptive 11.052 ms vs qbucket-schedule 11.293 ms, -2.18%
    - 368P-30s-H8: adaptive 21.231 ms vs qbucket-schedule 21.313 ms, -0.39%
  - Conclusion: qbucket-compatible schedule is not the primary source of the
    customer benchmark regression. Next rollback target is dKV TMA reduce versus
    the original qbucket atomic dK/dV epilogue.
- dKV rollback validation:
  - Added `BSA_BWD_DKV_EPILOGUE` as a validation-only switch:
    - `tma`: current dKV TMA reduce with staged accumulator layout.
    - `atomic`: atomic add into the current staged dKV accumulator layout.
    - `atomic_linear`: qbucket-style linear dK/dV accumulator layout plus atomic
      add and linear postprocess.
  - `BSA_BWD_DKV_EPILOGUE=atomic pytest -q test_bsa.py::test_bsa_sm100 --tb=short -s -x`: passed.
  - `BSA_BWD_DKV_EPILOGUE=atomic_linear pytest -q test_bsa.py::test_bsa_sm100 --tb=short -s -x`: passed.
  - Customer 8-case bwd, adaptive schedule, `warmup=1/runs=3`:
    - 720P-30s-H1: tma 89.508 ms, atomic-stage 89.142 ms, atomic-linear 88.709 ms.
    - 720P-30s-H4: tma 374.974 ms, atomic-stage 372.630 ms, atomic-linear 372.520 ms.
    - 720P-15s-H4: tma 94.922 ms, atomic-stage 94.442 ms, atomic-linear 94.266 ms.
    - 480P-30s-H1: tma 9.392 ms, atomic-stage 9.382 ms, atomic-linear 9.409 ms.
    - 480P-30s-H4: tma 36.667 ms, atomic-stage 36.744 ms, atomic-linear 36.522 ms.
    - 480P-15s-H4: tma 9.796 ms, atomic-stage 9.625 ms, atomic-linear 9.466 ms.
    - 368P-30s-H4: tma 11.052 ms, atomic-stage 10.993 ms, atomic-linear 10.824 ms.
    - 368P-30s-H8: tma 21.231 ms, atomic-stage 21.297 ms, atomic-linear 21.184 ms.
  - Atomic-stage only recovers a small part of the regression. Atomic-linear
    recovers more, especially on short/small cases, so the staged dKV accumulator
    layout plus postprocess is a real contributor.
  - Relative to original qbucket, atomic-linear is within or faster than qbucket
    on 720P-H1, 720P-15s-H4, and 480P-30s-H4, but still slower on:
    - 480P-30s-H1: -2.14%
    - 480P-15s-H4: -2.69%
    - 368P-30s-H4: -3.77%
    - 368P-30s-H8: -0.72%
  - `BSA_CSR_SCHEDULE_POLICY=qbuck BSA_BWD_DKV_EPILOGUE=atomic_linear` does not
    improve the small cases, so the residual is not fixed by qbucket qrange
    sizing. The remaining suspect is the current CSR scheduled main-kernel task
    layout/metadata path versus qbucket's native `(q_group, kv)` offsets layout.
  - Conversion/schedule build is not the primary source:
    - Standalone CSR build+schedule event time: 720P-H1 0.663 ms, 720P-H4
      2.473 ms, 720P-15s-H4 0.520 ms, 480P-H1 0.133 ms, 480P-H4 0.235 ms,
      480P-15s-H4 0.127 ms, 368P-H4 0.136 ms, 368P-H8 0.160 ms.
  - NCU on 368P-30s-H4 main bwd:
    - tma/staged: `/tmp/bsa_ncu_368_h4_tma.csv`, duration 15.548 ms.
    - atomic-linear: `/tmp/bsa_ncu_368_h4_atomic_linear.csv`, duration 15.286 ms.
    - qbucket NCU on the 335bec1 worktree failed under NCU due a Triton driver
      `/sbin/ldconfig` abort before profiling; runtime event baseline remains
      the qbucket comparison for this step.
