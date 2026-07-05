# SM100 blk64 Sage FP8 最终验证结果

## 结论

当前实现支持固定长度、前向、BHSD 布局的 Sage FP8 block-sparse
attention：`B=1`、`H=4/8`、`D=128`、`64x64` 稀疏块、uniform top-k，
输出为 BF16。

在 B300/SM103 上，客户 PDF 的 12 组输入规模全部通过精度门槛。FP8 和
BF16 使用相同的自动 split 后，两轮正式测试的 FP8/BF16 延迟几何平均为
`0.95585`，即 FP8 综合耗时少 `4.41%`。12 组两轮平均结果中 FP8 赢 8 组。

## 客户 PDF 输入

- `B=1`，`D=128`，block size `64x64`，稀疏率 `0.9`。
- PDF 给出 6 组 `(Sq, Sk)`，每组测试 `H=4` 和 `H=8`，共 12 组。
- top-k 按客户脚本计算：`floor((Sk / 64) * 0.1)`，得到
  `188, 368, 548, 729, 909, 1089`。
- FP8 和 BF16 每组使用完全相同的输入、稀疏索引和 split。
- 自动 split 随 top-k 使用 `1, 2, 4, 4, 8, 8`。

## 逐 case 精度和性能

性能值是两轮正式测试 median 的平均。最后一列为 FP8 相对 BF16 的耗时
变化，正数表示 FP8 更快。精度为
`mean(abs(FP8 - reference)) / mean(abs(reference))`，reference 是反量化后的
BF16 blk64 路径。

| H | Sq | Sk | top-k | split | BF16 ms | FP8 ms | FP8 快慢 | 相对平均误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 119040 | 120320 | 188 | 1 | 2.660 | 2.540 | 快 4.51% | 2.6327% |
| 8 | 119040 | 120320 | 188 | 1 | 5.267 | 5.077 | 快 3.61% | 2.6348% |
| 4 | 234240 | 235712 | 368 | 2 | 10.396 | 9.961 | 快 4.18% | 2.6014% |
| 8 | 234240 | 235712 | 368 | 2 | 20.223 | 20.379 | 慢 0.77% | 2.6253% |
| 4 | 349440 | 351168 | 548 | 4 | 22.535 | 23.228 | 慢 3.07% | 2.6229% |
| 8 | 349440 | 351168 | 548 | 4 | 46.645 | 46.897 | 慢 0.55% | 2.6239% |
| 4 | 464640 | 466560 | 729 | 4 | 40.681 | 41.319 | 慢 1.59% | 2.6746% |
| 8 | 464640 | 466560 | 729 | 4 | 86.334 | 77.173 | 快 10.61% | 2.6126% |
| 4 | 579840 | 581952 | 909 | 8 | 65.636 | 64.784 | 快 1.30% | 2.5971% |
| 8 | 579840 | 581952 | 909 | 8 | 134.878 | 121.531 | 快 9.90% | 2.6214% |
| 4 | 695040 | 697408 | 1089 | 8 | 95.435 | 87.484 | 快 8.33% | 2.6342% |
| 8 | 695040 | 697408 | 1089 | 8 | 194.416 | 165.904 | 快 14.66% | 2.6566% |

汇总：

- PDF 规模精度：12/12 finite、shape/dtype 正确、重复运行一致并通过门槛。
- PDF 规模平均相对误差：`2.6281%`。
- PDF 规模最差相对误差：`2.6746%`，低于客户 `2.9%` 门槛。
- PDF 规模最大绝对误差：`0.000763`，低于客户 `0.15` 门槛。
- 性能 run 1 FP8/BF16 geomean：`0.95930`。
- 性能 run 2 FP8/BF16 geomean：`0.95241`。
- 两轮综合 geomean：`0.95585`，FP8 耗时少 `4.41%`。

更广的 28 组精度回归同样全部通过：28/28 finite、shape/dtype 正确、重复
运行一致、最大绝对误差通过，最差相对平均误差为 `2.81195%`。

## 当前保留的有效优化

1. **原生 Blackwell FP8 tensor core 路径**
   - QK 和 PV 使用 E4M3 MMA，FP32 累加，最终输出 BF16。
   - Q/K/V 不在 kernel 外生成完整反量化张量。

2. **Sage scale 融合**
   - Q 使用 per-token scale，K 使用 per-16-token scale，V 使用
     per-channel scale。
   - Q/K scale 融合进 score 计算，V scale 融合进输出 epilogue。

3. **概率 `P * 448` 融合**
   - `log2(448)` 直接加入 exp2 bias，删除单独的逐元素 `* 448`。
   - P 和 softmax 分母携带相同的 448，归一化时自然抵消，删除输出端
     `/ 448`。
   - 保持一次原生 FP8 PV MMA，避免旧的双 PV error-feedback 路径。

4. **V scale tile 缓存**
   - 每个 work tile 只把 128 个 V channel scale 载入 shared memory 一次，
     多个输出行复用。

5. **长短 Q 分开的 split-KV 调度**
   - 短 Q 保留为补足 GPU 并行度调优过的 split 策略。
   - 当独立 Q tile 已达到 512 个时，使用较小的 `1/2/4/8` split，避免
     客户长序列产生过量 partial workspace 和 combine 开销。

6. **split combine 和热路径优化**
   - uniform top-k 的 split 边界在 kernel 内计算，避免额外 offset kernel。
   - 16 split 的已知非空场景使用 256-thread combine specialization。
   - 已编译的主 kernel/combine callable 使用专用缓存；重复调用绕过通用
     compile-key 和布局检查热路径。
   - 编译与运行使用当前 CUDA stream 语义，非默认 stream 回归通过。

以下实验没有保留在交付代码中：把 V scale 移到 combine、合并四个 K scale
读取、四 stage pipeline，以及所有诊断/ablation 开关。它们没有稳定改善完整
前向延迟或增加了额外约束。

## 代码检查和回归

- `git diff --check`：通过。
- 6 个交付 Python 文件 `py_compile`：通过。
- `tests/test_bsa_fp8_fwd.py`：28 passed。
- CuTeDSL compile-key、partial-tail 和大 stride 定向回归：10 passed，1 skipped。
- 扩展 28 组 FP8 精度：28/28 通过。
- 客户 PDF 精度：12/12 通过。
- 非默认 CUDA stream 的 split 1/16：event 完成、输出有限、ordered V-zero
  检查为精确 0。

全仓 `pytest tests` 在 PyTorch 26.05 环境中为 228 passed、309 skipped、
15 failed。失败都来自旧 BF16 AOT `torch.ops.bsa_blk64`，错误为
`Cannot access data pointer of Tensor that doesn't have storage`。失败 case 单独
运行通过；在远端原版本 worktree 上整文件运行同样复现 19 个同类失败。因此
这是现有 AOT 扩展/测试顺序与当前 PyTorch 环境的问题，不是本次 FP8/CuTeDSL
改动造成的回归。

## 环境和测试方法

- GPU：NVIDIA B300 SXM6 AC，SM103。
- Driver：595.58.03；CUDA：13.2。
- Container：`nvcr.io/nvidia/pytorch:26.05-py3`。
- PyTorch：`2.12.0a0+5aff3928d8.nv26.05`。
- CUTLASS DSL：4.4.1。
- 正式性能：10 对 warmup，30 个交错 CUDA-event 样本，两轮独立执行。
- 计时排除输入生成、量化、稀疏 map 生成和 JIT 编译；包含输出分配、主
  kernel、split partial 和 combine。
- 正式计时前 GPU 利用率为 0%，显存占用为 0 MiB。

PDF 只提供形状、head 数、块大小和稀疏率，没有提供真实 Q/K/V 与 sparse
index。因此这里是客户 PDF 精确形状验证，不是客户线上张量回放。
