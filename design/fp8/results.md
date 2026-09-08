# SM100 blk64 Sage FP8 最终验证结果

## 2026-09-08：与 SM120 对齐动态输入能力

SM100 和 SM103 的 Sage FP8 接口已解除原先 `B=1`、`H in {4,8}`、Q/KV
长度必须是 64 倍数、uniform top-k 和 rank-1 `block_sizes` 的限制。现在支持：

- 任意正数 batch 和 MHA head 数；
- 非 64 对齐的 Q/KV 长度（SM100/SM103 在后端边界补齐并屏蔽，SM120 原生处理）；
- `[B,H,Q_blocks]` 的 `q2k_block_nums`，SM100/SM103 仍支持空行；
- `[N]`、`[B,N]`、`[B,H,N]` 三种 `block_sizes`。

真机定向验证覆盖 B200/SM100 和 B300/SM103，包括 `B=2`、`H=1/3/5`、
Q/KV 尾块、三种 block-size 作用域、可变/空 top-k，以及 H=1/5 下
1/4/8/16 路 split。完整 FP8/量化回归在 B200 上为 67 passed、14 skipped，
在 B300 上为 68 passed、13 skipped；AOT dispatch 和动态 compile-key 回归为
30 passed、2 skipped。SM100/SM103 仍是 FP8 JIT-only；SM120 的 FP8 支持 AOT。

以下早期性能结果中的 `B=1`、`H=4/8` 是当时的基准输入范围，不再是当前
接口限制。

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

## 2026-07-09：B300 kernel-only 1.5x 优化

本轮只优化 Blackwell FP8 attention kernel，没有修改客户的 Q/K/V 量化函数。
在 SLA 0709 的 10 个形状上，起点的 kernel-only 几何平均约为 `1.30x`。

### Barrier-safe 耗时拆分

通过保留 barrier、pipeline phase 和 TMEM 生命周期，只跳过对应计算，得到
以下关键路径下界。各项彼此重叠，不能直接相加：

| 路径 | 对完整 kernel 的边际影响 |
|---|---:|
| Q/K scale 读取和计算 | 约 9%～11% |
| score 读取、max、exp 和 sum | 约 30%～34% |
| 在线 softmax 的中间 O 重缩放 | 约 23%～25% |
| PV 和输出路径整体 | 约 27%～28% |
| 原始 QK MMA 增量 | 约 3%～6% |
| 最终 correction/输出 | 约 3%～4% |

原版本 NCU 的主要等待是 long scoreboard（约 `41.7%`）和 barrier
（约 `18.3%`）。删除 Q/K scale 路径时，全局 load sector 从约
`14.76M` 降到 `2.51M`，说明这些 scale 大多命中 cache，但每一行重复执行的
load、地址计算和依赖等待仍然很重。

### 最终保留的三项优化

1. 将内部 FP8 概率 P 的标尺从 448 逐步降到 128，并利用 E4M3 到 448 的
   数值余量延后 online-softmax 的 reference-max 更新。P 和分母仍携带相同
   因子，归一化公式不变；这不是输入 Q/K/V 量化改动。该项显著减少 O 在
   TMEM 中的读、乘、写重缩放。
2. 普通非 split 路径在 correction 的第一次 BF16 转换前乘 V-scale，删除
   epilogue 中“读 BF16、转 FP32、乘 scale、再转 BF16”的第二轮处理。
   split-KV 的 FP32 partial 路径保持原实现。
3. 每个 KV tile 的 16 个 K-scale 改由 MMA warp 只读取一次并发布到 shared
   memory，64 个 query 行共同复用。原实现是每个 softmax 线程重复读取 8 个
   K-scale。与上一版交错 A/B：480P-15s-H4 快 `3.20%`（80/80 赢），
   720P-15s-H4 快 `3.33%`（60/60 赢）。

没有保留的实验包括：P=64（与 P=128 持平）、K-scale warp shuffle 广播
（明显变慢）、128-bit 全局 scale load（约慢 3%）、跨迭代 K-scale 寄存器
双缓冲（约慢 40%）以及 V-scale shared 向量 load（持平）。

### P128 阶段的历史性能（旧顺序计时）

第二轮完整测试使用 warmup 10、30 次计时：

| Sq | Sk | H | top-k | BF16 ms | FP8 kernel ms | SpeedK |
|---:|---:|---:|---:|---:|---:|---:|
| 116160 | 118528 | 4 | 186 | 2.447 | 1.878 | 1.303x |
| 116160 | 118528 | 8 | 186 | 5.555 | 3.563 | 1.559x |
| 109312 | 111040 | 4 | 174 | 2.114 | 1.774 | 1.192x |
| 109312 | 111040 | 8 | 174 | 4.743 | 3.186 | 1.489x |
| 216832 | 219200 | 4 | 342 | 9.727 | 5.886 | 1.653x |
| 216832 | 219200 | 8 | 342 | 19.948 | 11.724 | 1.701x |
| 349440 | 351168 | 4 | 548 | 26.012 | 16.258 | 1.600x |
| 349440 | 351168 | 8 | 548 | 51.709 | 32.737 | 1.580x |
| 695040 | 697408 | 4 | 1090 | 103.295 | 65.691 | 1.572x |
| 695040 | 697408 | 8 | 1090 | 205.502 | 131.901 | 1.558x |

- 完整表 run 1 kernel-only 几何平均：`1.533x`
- 完整表 run 2 kernel-only 几何平均：`1.513x`
- 两轮综合 kernel-only 几何平均：约 `1.523x`
- 客户原量化不变时，两轮含量化几何平均：约 `1.113x`
- 正确性：kernel 回归 `36/36`，扩展精度回归 `28/28`
- 扩展精度最差相对平均误差：`2.86455%`，低于 `2.9%` 门槛

按当时“先测完 BF16、再测完 FP8”的顺序计时口径，B300 的 kernel-only
`1.5x` 目标在 P128 阶段达到。这个数字不是当前 P256 的公平交错 A/B
headline；当前结果见下一节。

## 2026-07-09：P256 对 BF16 重新验证

当前代码为 `P_SCALE=256`，安全延迟缩放阈值为
`log2(448 / 256) = 0.807354922`。使用 SLA 0709 的相同 10 个形状、客户脚本
相同的 mean-pool/top-k/完整 K-block 索引布局，并让 FP8 与 BF16 使用完全
相同的 Q/K/V 和索引。每轮 20 对 warmup、60 个交错 CUDA-event 样本、每个
样本 3 次 inner launch；A/B 交错后整批同步，保持热态同时避免顺序偏差。

两轮结果：

- kernel-only 几何平均：`1.37373x`、`1.37612x`。
- 两轮综合 kernel-only：`1.37492x`，FP8 在 10/10 行都快。
- 真实把未修改的客户 Sage Q1/K16/V1 量化与 FP8 kernel 连起来计时：
  `1.02448x`、`1.02532x`。
- 两轮综合真实含量化：`1.02490x`，FP8 在 5/10 行更快。
- 大形状（720P-H8/30s）kernel-only 为约 `1.50x～1.52x`，真实含量化为
  约 `1.31x～1.39x`；小形状的量化成本会吃掉 kernel 收益。

为兼容历史口径，当前 P256 也重新运行了旧的非交错脚本：kernel-only
`1.47375x`，单独量化中位数与 kernel 中位数相加后为 `1.08156x`。因此旧
`1.523x` 与当前公平 headline 的差异不只来自 P128→P256 的约 1% 回退，主要
还来自旧脚本的顺序计时与采样口径。当前对外应使用 `1.37492x` kernel-only
和 `1.02490x` 真实含量化结果。

## 2026-07-09：直接 BHSD 量化与客户 PDF 复测

客户 helper 原先先将 Q/K/V 从 BHSD 实体转置为 BSHD，量化后再做三次实体
转置返回 BHSD。新的 `quantize_sage_bhsd` 保持相同的 Sage Q1/K16/V1 和 K
smoothing 数学契约，直接按 BHSD backing stride 读写，删除这六次全张量
copy。Q 量化进一步将 16 个 token 合并到一个 Triton program；相对旧的
one-program-per-token 实现，H8 的 109312/695040 两个长度分别从
`0.4650/2.8775 ms` 降到 `0.0635/0.3177 ms`，输出逐元素一致。

在 SLA 0709 全部 10 个形状上，使用相同驻留输入、20 对 warmup、60 个交错
CUDA-event 样本比较客户 helper 与新量化器：

- 客户量化中位数的几何平均：`2.9954 ms`。
- 新量化中位数的几何平均：`0.5138 ms`。
- 新/旧几何平均比：`0.17153`，即新量化约 `5.83x`；10/10 行都更快。
- 最大 H8 形状从 `12.4532 ms` 降到 `1.8733 ms`。

最大 H8 形状的单独分解为 Q `0.3170 ms`、K mean `0.2484 ms`、K quant
`0.4528 ms`、V scale `0.3268 ms`、V quant `0.4014 ms`；完整 API 为
`1.6849 ms`。按 Q/K/V 必需读写估算约搬运 9.3 GB，有效数据率约
`5.5 TB/s`，剩余路径主要受全张量内存流量约束。三流并发在短形状慢
`4.3%`、最大形状只快 `1.0%`，因此没有保留。

未修改客户脚本文件，只在运行时将其 `quantize_sage` 替换为新公开 API 后，
按脚本原生的 5 warmup/20 runs 重跑全部 10 行：

- 量化耗时几何平均从基线 `3.1533 ms` 降到 `0.5307 ms`，约 `5.94x`。
- 含量化 `Speed+Q` 几何平均从 `1.0809x` 提升到 `1.3984x`。
- 10/10 行的含量化 FP8 路径都快于 BF16；最小/最大行为
  `1.060x/1.608x`。

客户 PDF B300 表与当前结果的完整逐组对比如下。两边使用相同的 10 个
SLA 0709 形状和 top-k；`SpeedK` 只包含 attention kernel，`Speed+Q` 包含
Q/K/V 量化：

| 配置 | PDF Quant ms | 当前 Quant ms | PDF SpeedK | 当前 SpeedK | PDF Speed+Q | 当前 Speed+Q |
|---|---:|---:|---:|---:|---:|---:|
| 368P-30s-H4 | 1.063 | 0.200 | 1.066x | 1.198x | 0.737x | 1.091x |
| 368P-30s-H8 | 2.169 | 0.347 | 1.236x | 1.516x | 0.830x | 1.384x |
| 480P-15s-H4 | 1.166 | 0.209 | 0.905x | 1.187x | 0.600x | 1.060x |
| 480P-15s-H8 | 2.082 | 0.345 | 1.051x | 1.417x | 0.704x | 1.282x |
| 480P-30s-H4 | 2.070 | 0.366 | 1.316x | 1.637x | 1.030x | 1.542x |
| 480P-30s-H8 | 3.998 | 0.706 | 1.344x | 1.705x | 1.057x | 1.608x |
| 720P-15s-H4 | 3.260 | 0.594 | 1.315x | 1.617x | 1.126x | 1.560x |
| 720P-15s-H8 | 6.360 | 1.100 | 1.329x | 1.583x | 1.144x | 1.531x |
| 720P-30s-H4 | 6.355 | 1.020 | 1.355x | 1.564x | 1.252x | 1.541x |
| 720P-30s-H8 | 14.572 | 2.056 | 1.317x | 1.551x | 1.206x | 1.527x |
| **几何平均** | **3.125** | **0.531** | **1.213x** | **1.487x** | **0.941x** | **1.398x** |

量化耗时几何平均相对 PDF 降低到 `16.98%`（约 `5.89x`），含量化整体从
PDF 的 6/10 行快于 BF16 提升到当前 10/10 行都快。

最终验证：仓库 FP8/量化测试 `40/40` 通过，扩展精度集 `28/28` 通过。
客户 `sla-fp8.pdf` 的 12 个精确形状也为 `12/12` 通过，最差相对平均误差
`2.7341%`（门槛 `<2.9%`），最大绝对误差 `0.000778`（门槛 `<0.15`）。

调用方式：

```python
from bsa_fp8_blk64 import bsa_fp8_blk64_fwd, quantize_sage_bhsd

q8, k8, v8, q_scale, k_scale, v_scale = quantize_sage_bhsd(q, k, v)
out = bsa_fp8_blk64_fwd(
    q8, k8, v8, q_scale, k_scale, v_scale, q2k_block_index, topk
)
```

## 2026-07-09：合并 master 后的 CUTLASS DSL 4.5.2 复验

将 `origin/master` 的统一 CuTe DSL wheel 打包路径合入 `bsa_fp8` 后，在最终
工作树和 `nvidia-cutlass-dsl==4.5.2` 下重新验证。wheel 元数据限制为
`nvidia-cutlass-dsl>=4.5.2,<4.6`，从隔离安装目录导入
`bsa_fp8_blk64_fwd` 和 `quantize_sage_bhsd` 通过；FP8/量化测试再次为
`40/40` 通过。CuTe DSL compile-key/SM120 AOT 定向测试为 20 passed、
9 skipped，SM100 blk64 split/unified-dispatch 的 3 个定向测试全部通过。

性能仍使用未修改的客户十组脚本，只将 `quantize_sage` 替换为公开的直接
BHSD API。每组保持 5 次 warmup、20 次 CUDA-event 计时，连续运行两轮。
下表是两轮 median 的算术平均；`SpeedK` 和 `Speed+Q` 由平均延迟重新计算：

| 配置 | BF16 ms | FP8 kernel ms | Quant ms | SpeedK | Speed+Q |
|---|---:|---:|---:|---:|---:|
| 368P-30s-H4 | 2.405 | 2.077 | 0.206 | 1.158x | 1.054x |
| 368P-30s-H8 | 5.508 | 3.772 | 0.346 | 1.460x | 1.338x |
| 480P-15s-H4 | 2.081 | 1.776 | 0.216 | 1.171x | 1.044x |
| 480P-15s-H8 | 4.614 | 3.425 | 0.356 | 1.347x | 1.220x |
| 480P-30s-H4 | 9.793 | 6.053 | 0.377 | 1.618x | 1.523x |
| 480P-30s-H8 | 20.198 | 11.881 | 0.706 | 1.700x | 1.605x |
| 720P-15s-H4 | 26.325 | 16.176 | 0.600 | 1.627x | 1.569x |
| 720P-15s-H8 | 52.627 | 32.831 | 1.127 | 1.603x | 1.550x |
| 720P-30s-H4 | 103.151 | 66.047 | 1.059 | 1.562x | 1.537x |
| 720P-30s-H8 | 206.135 | 132.117 | 1.984 | 1.560x | 1.537x |

- Run 1：`SpeedK=1.46716x`，`Speed+Q=1.38111x`。
- Run 2：`SpeedK=1.47012x`，`Speed+Q=1.38082x`。
- 两轮综合：`SpeedK=1.46864x`，`Speed+Q=1.38097x`，量化耗时几何平均
  `0.53915 ms`。
- 两轮都是 10/10 行 `Speed+Q > 1`；最小值分别为 `1.042x` 和 `1.041x`。
- 相对上一轮记录的 `1.487x/1.398x`，两项几何平均分别变化 `-1.23%` 和
  `-1.22%`，在预设的 3% 复验波动门槛内。

环境为 NVIDIA B300 SXM6 AC（SM103）、driver 595.58.03、PyTorch
`2.12.0a0+5aff3928d8.nv26.05`、CUDA 13.2、CUTLASS DSL 4.5.2。两轮开始前
GPU 均为 0 MiB、0% utilization，第二轮结束后也恢复为 0 MiB、0%。

## 2026-07-26：合入最新 master 前的回归验证

将 `origin/master@7e0f854` 合入 `bsa_fp8@83bfba3`，保留 FP8 kernel、直接
BHSD 量化、统一 Python 包 API，同时接入 master 的 SM90 AOT、SM120 AOT 和
SM100 blk64 CLC split-KV 调度。冲突解决后，在同一台 B300、同一
`nvcr.io/nvidia/pytorch:26.05-py3` 容器和 CUTLASS DSL 4.5.2 下验证：

- 客户 PDF 的 12 个精确形状全部通过；最差相对平均误差为 `2.71331%`
  （门槛 `<2.9%`），最大绝对误差为 `0.000763`（门槛 `<0.15`）。
- FP8 kernel 定向回归 `36/36` 通过；直接 BHSD 量化为 2 passed、2 skipped，
  skipped 项依赖未安装的私有 `flashinfer_vx`。
- 包安装/导入测试 `6/6` 通过，SM100 blk64 的 CLC、single-tile、
  persistent、large-stride 和 auto-split 定向测试 `10/10` 通过。
- 全仓正式测试目录结果为 `296 passed, 325 skipped, 0 failed`。

性能使用 10 个 SLA 形状、5 次 warmup、20 个交错 CUDA-event 样本，对合并前
`83bfba3` 和合并后工作树做同机 A/B。这里的稀疏索引是固定随机种子的滚动
sparse map，用于判断合并回归，不替代前文客户 mean-pool map 的 headline：

- 合并后相对 BF16：kernel-only 几何平均 `1.32954x`，含直接 BHSD 量化为
  `1.27970x`。
- 合并前相对 BF16：kernel-only 几何平均 `1.32527x`，含量化为
  `1.27751x`。
- 合并后/合并前的 FP8 绝对延迟比：kernel-only `1.00050`（`+0.05%`），
  含量化 `1.00201`（`+0.20%`），属于计时噪声，没有可测的合并回退。

全量测试还暴露并修正了 master 中 SM100 blk128 的架构判断：CUTLASS DSL
把 B300 表示为 `sm_103a`，原数值上界 `<= sm_110f` 会错误拒绝它；改为与
blk64 一致的 SM100/SM110 family 判断后，相关回归全部通过。
