# SM100/SM103 FP8 验证结果

## 2026-10-21：两处同步修复的完整复测

复测内核固定为 `86e40ad`，修复前基线为 `c514e9a`。两处正向修复的专项检查
通过，但扩大检查后确认 **BF16 反向 blk64 和 blk128 仍有原有的同步错误**。
因此本次结果不能解释为整个项目已经没有 bug；本轮没有修改反向内核。
性能方面，B200 的波动项复测通过；SM103 测试设备确认发生热降频，稳定性未通过。

环境：PyTorch 26.05 / CUDA 13.2、CUTLASS DSL 4.5.2、apache-tvm-ffi
0.1.14.post1、Compute Sanitizer 2026.1.1。SM100 使用 B200；SM103 使用
GB110 bring-up board（运行时 capability=(10,3)），不能将其绝对耗时当作 B300 数据。

### 正确性与预编译验证

| 检查 | B200 / SM100 | GB110 / SM103 |
|---|---|---|
| 两处修复的数值回归 | 18 passed | 18 passed |
| 同一组用例的 synccheck | 18 passed，0 errors | 18 passed，0 errors |
| 同一组用例的 memcheck | 18 passed，0 errors | 18 passed，0 errors |
| 尾块、多 batch/head、动态 metadata 的 memcheck | 37 passed，0 errors | 37 passed，0 errors |
| 原始完整 pytest 回归 | 588 passed，2 failed，425 skipped | 589 passed，2 failed，424 skipped |
| 修正旧兼容性测试后，相关测试重跑 | 11 passed | 11 passed |
| 重新生成 blk64 AOT 后的测试 | 15 passed，3 skipped | 16 passed，2 skipped |
| 重新生成 blk128 AOT 后的测试 | 46 passed | 46 passed |
| blk128 AOT 专项 synccheck | 2 passed，0 errors | 2 passed，0 errors |

完整回归的两个失败均来自 `tests/test_cutedsl_compat.py` 的旧预期：4.6.2 以后的
bulk-copy election 策略，以及 DSL 对静态 dataclass 字段的省略方式。相同失败在
`c514e9a` 上也存在；测试修正提交为 `5d11217`，没有改动内核。表中保留原始
完整运行的失败数；修正后只重跑受影响的 11 项，并未将它们冒充为第二次全量运行。
跳过项包括其他架构、当前不支持的 GQA/head dimension、双 GPU、可选量化依赖
和未开启的构建集成检查；AOT 运行检查已在后续独立启用。

旧版 blk64 首个用例触发 synccheck 后会破坏当前 CUDA context，因此另开进程检查
旧版 FP8 blk128。B200 的旧 blk64 / blk128 分别报告 224 / 416 errors，SM103
分别为 192 / 640 errors；对应修复版均为 0。错误条数受运行调度影响，不用于
比较问题严重程度。AOT 均重新生成，blk128 覆盖 variable counts 开/关、四种
block-sizes 形式与 split=1/2/4/8，共 32 个 producer 变体。

### 扩大检查发现的反向旧问题

两个反向用例分别在独立进程中运行，避免首个错误影响第二个用例。初次检查还
遇到工具无法跟踪 PyTorch reference 的 cuBLAS `nvjet_` kernel；重新运行时只
排除这些 reference kernel，BSA kernel 仍完整接受检查，且工具内部错误数为 0。

| 独立 synccheck | B200：旧版 / 修复版 | SM103：旧版 / 修复版 |
|---|---|---|
| BF16 backward blk64 | 2528 / 3040 errors | 2976 / 1760 errors |
| BF16 backward blk128 | 672 / 736 errors | 576 / 480 errors |

报错均为 BSA 反向 kernel 内的 `Divergent thread(s) in block`，旧版与修复版
均包含相同的 kernel 指令偏移（blk64 `+0xa110`，blk128 `+0xa2a0`）；
SM103 的部分运行还报告其他同步点。
两次正向修复没有改动 `csrc/bwd/`。普通梯度数值回归通过不能排除这些同步错误。

```bash
compute-sanitizer --tool synccheck --error-exitcode=86 \
  --kernel-name-exclude kns=nvjet_ python -m pytest -v \
  'tests/test_flash_bwd.py::test_flash_bwd_blk64_sparse[64-256-False]'
compute-sanitizer --tool synccheck --error-exitcode=86 \
  --kernel-name-exclude kns=nvjet_ python -m pytest -v \
  'tests/test_flash_bwd.py::test_flash_bwd_sm100_blk128_topk_correctness[256-512-2-True]'
```

### 性能与证据

每次 A/B 对比在同一 GPU 上交替运行旧版和修复版，完整矩阵各 4 轮；编译、
参考校验与预热均在计时外。覆盖 BF16 blk64、FP8
blk64/blk128，以及量化加 FP8 的 GPU 总耗时，另记录 Python eager 调用耗时。
9 组输入包括短输入、动态 metadata、blk128 split=4、39756-token 客户形状，
以及 119040 / 349440 / 695040-token 长输入。blk64 FP8 公开接口保留自动 split
策略，显式 split=4 只传给 blk128。

B200 的 9 组已完成，每组均通过采样 FP32 reference、全输出 finite、重复输出
一致性及 A/B 输入摘要匹配。下表是 9 组 GPU 耗时比值的几何平均，正数表示修复版
更慢。量化加计算采用 CUDA Graph，包含量化的 GPU 工作，不包括 Python 调度。

| 路径 | B200 修复版相对旧版 |
|---|---:|
| BF16 blk64 | -0.73% |
| FP8 blk64 | -0.20% |
| FP8 blk128 | +0.90% |
| 量化 + FP8 blk64 | +0.00% |
| 量化 + FP8 blk128 | +1.22% |

首次 B200 矩阵中，39756-token 与 119040-token/H=4 两组出现 10% 以上的轮间波动，
个别量化加计算耗时比旧版高 6.49%。保留这些原始数据，并在另一张 B200 上将
每个路径的持续预热延长至 500ms，交替运行各 8 轮（作业 `2331121`）。复测中
10 个 GPU 指标的最大轮间跨度为 2.00%，修复版相对旧版的最大增幅为 0.65%，
未复现原来的退化。轮间跨度定义为 `(max(各轮中位数)-min(各轮中位数))/中位数`。
换卡和延长预热同时发生，不能仅据此断言最初波动只由预热不足造成。

SM103 初次性能设备 `2u2g-gen-0468` 在持续负载下达到 87℃，SM 时钟下降到
数百 MHz；旧版和修复版均出现明显波动。该轮在最终长输入计时中停止，保留原始
结果作为环境异常记录，不作为性能稳定的证明。完整矩阵已转移至 B300 PCIe
分区的 `2u2g-gen-0149`（作业 `5180997`），并对每个路径持续预热 500ms；
运行时设备名仍为 `NVIDIA Graphics Device`，capability=(10,3)，因此本报告
按 SM103 设备记录，分区标签不作为具体产品规格的证明。

新节点的 9 组输入、每版各 4 轮已全部跑完，数值检查通过。各路径的 GPU 耗时
几何平均变化如下；这些是在变频条件下取得的原始观测，不能用于证明稳定性。

| 路径 | SM103 修复版相对旧版 |
|---|---:|
| BF16 blk64 | +0.06% |
| FP8 blk64 | -0.10% |
| FP8 blk128 | +0.40% |
| 量化 + FP8 blk64 | +0.04% |
| 量化 + FP8 blk128 | -0.63% |

虽然没有观察到一致的平均性能退化，119040-token/H=4 的修复版仍有
10.73%～19.05% 的轮间跨度，695040-token/H=8 的 FP8 blk128 为 12.54%。
新节点最高 89℃，繁忙时最低采样 SM 时钟为 682MHz；现场额外查询确认
`clocks_event_reasons.sw_thermal_slowdown=Active`，当时为 88℃、885MHz，
power-cap slowdown 未激活。故 **SM103 的整体性能稳定性本轮不予确认**。
需要散热与频率稳定的设备才能继续作稳定性判断；此次没有更改机器的功耗、
时钟或散热设置。

完整数值回归作业：B200 `2330834`、SM103 `5179562`；补充检查及 AOT 作业：
B200 `2330990`、SM103 `5180217`。首次性能脚本误把显式 split 参数传入 blk64
公开接口，已修正；完整性能重新安排为 B200 `2331063`、SM103 `5180550`，
此前的局部计时不作为完整性能结论。
本轮申请的作业均已退出，GPU 已释放。最终验收记录为
`agent/agent_profiles/full-retest-20261021/acceptance.json`，明确保留整体未通过状态。

原始日志、JUnit XML、环境、GPU telemetry 与分阶段返回码保存在
`agent/agent_profiles/full-retest-20261021/`；固定源码归档及 SHA256 在
`agent/agent_space/full-retest-20261021/manifest.json`，性能脚本在
`agent/agent_benchmark/full-retest-20261021/benchmark.py`，持续预热复测脚本为
同目录的 `benchmark_stability.py`；三个最终性能数据集另导出了 CSV。
测试期间另修复了 `tools/ssh-gw` 在 cluster scratch 为空时忽略显式 workdir 的
作业文件路径问题（独立工具提交 `5cc9ae4`，69 项测试通过，CPU 作业验证成功）。

## 2026-10-21：blk128 warp 子集同步修复（cuDNN PR #1752 P1）

[cuDNN review](https://github.com/NVIDIA/cudnn-frontend/pull/1752#pullrequestreview-5594779337)
指出的 P1 同样存在于本仓库 FP8 blk128 路径。TMEM 的 416 个线程、S-read 的
160 个线程，以及统计量交接中的 64/256 个线程从不同 warp 分支到达同步点，
原 `bar.sync` / `bar.arrive` 的 aligned 约束不成立。

修复覆盖 TMEM 分配和释放、S-read、统计量和 terminal handoff，以及 inline
row-max publication。使用已有 blk64 unaligned helpers，保留 barrier ID、线程数
与调用顺序；全 CTA 的初始化 barrier 保持原实现。直接调用 TMEM barrier，避免
`TmemAllocator.wait_for_alloc()` 重新构造 aligned barrier。

基线为 `b1e4cc9`。环境为 PyTorch 26.05（CUDA 13.2）、CUTLASS DSL 4.5.2、
apache-tvm-ffi 0.1.14.post1、Compute Sanitizer 2026.1.1。

| 设备 | 基线 synccheck（split=1） | 修复后 synccheck | 修复后 memcheck | 原有回归 |
|---|---|---|---|---|
| B200 / SM100 | 704 errors，含 divergent barrier | 2 passed，0 errors | 2 passed，0 errors | 22 passed |
| GB110 开发板 / SM103 | 320 errors，含 divergent barrier | 2 passed，0 errors | 2 passed，0 errors | 22 passed |

B200 作业为 `5175441`（`umbriel-b200-068`）；SM103 作业为 `5175669`
（`2u2g-gen-0471`）。后者设备名为 NVIDIA Graphics Device，运行时确认
capability=(10,3)，为 GB110 bring-up board。原计划 B300 作业因资源排队取消。

新增 `tests/test_sm100_blk128_sync.py` 以 split=1/4 覆盖 CUDA Graph 和缓存重用，
并在重放之间切换动态 block counts。输入包含空行、零长度块、Q/KV 尾块、pipeline
环绕和无效闲置索引；结果对照 FP32 reference。旧版在同一输入上触发 synccheck，
说明单看数值测试不足以发现该问题。

原有 22 项测试覆盖不传及 rank-1/2/3 `block_sizes`、online rescaling、空块、
scale views、独立 stream/graph，以及 split=2/3/4/8。本次验证 JIT 路径，已有
blk128 AOT bundle 需重新生成。

```bash
compute-sanitizer --tool synccheck --error-exitcode=1 \
  python -m pytest -q tests/test_sm100_blk128_sync.py
compute-sanitizer --tool memcheck --error-exitcode=1 \
  python -m pytest -q tests/test_sm100_blk128_sync.py
```

## 2026-10-21：blk64 warp 分工下的同步检查修复

### 原因与修改

TMEM 分配/释放和 softmax 统计量交换只同步指定的 warp，但原实现调用的是
CTA-aligned named barrier（`bar.sync` / `bar.arrive`）。这些指令要求参与者的
控制流满足 aligned 约束，不能用于这里分开的 warp 执行路径。改为
`barrier.sync` / `barrier.arrive`（CUTLASS 的 unaligned API），保留原 barrier
编号、参与线程数和调用次序。这里的 unaligned 指控制流，不是内存地址对齐。

另一个问题在输出合并：某些行的 softmax 权重下溢为 0 时，原来的逐行判断会让
部分 lane 跳过整 warp 必须共同执行的 TMEM load。现在由 warp 投票决定是否读取，
再将零权重行写为 0；整 warp 都为空时仍跳过未初始化的 TMEM。

同步约束依据：[PTX named barrier 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)。

### 已完成的验证

基线为 `c514e9a`，环境为 PyTorch 26.05（CUDA 13.2）、CUTLASS DSL 4.5.2、
apache-tvm-ffi 0.1.14.post1；Compute Sanitizer 2026.1.1。

- **最小复现（B300）**：不含 attention 的 416-thread named barrier 例子，aligned
  指令在复用和不复用 barrier 两种情况下都触发 synccheck；仅改为 unaligned
  指令后，两种情况均为 0 错误。
- **B300 原问题复现**：旧内核在动态候选数、含空块和极端权重输入上触发
  `Divergent thread(s) in block`；修复后同样输入输出校验通过，synccheck 0 错误。
- **新增 16 例回归（B300）**：FP8/BF16 × split 1/2 × CLC 开/关 × Q 长度 65/128。
  同时覆盖逐行权重下溢、候选数为 0、奇数候选数、部分 KV 块和 Q 尾块；
  检查 O、LSE 以及空行的 O=0、LSE=-inf。synccheck 和 memcheck 各 16 passed，
  均 0 错误；混合候选数用例 racecheck 为 0 errors / 0 warnings。
- **B300 原有测试**：FP8 forward/quant、编译缓存键和 AOT dispatch 共 146 passed、
  16 skipped；`tests/test_flash_fwd.py -k sm100` 为 190 passed、294 skipped、
  12 deselected。无失败。
- **B200（SM100）针对性复查**：同一组新增 16 例全部通过，O/LSE 校验通过，
  synccheck 为 0 错误；覆盖 FP8/BF16、split 1/2、CLC 开关和 Q 尾块。
  作业 `5175059`，相同软件版本；此处没有重复 B300 的完整旧测试套件。

复现回归：

```bash
compute-sanitizer --tool synccheck --error-exitcode=1 \
  python -m pytest -q tests/test_sm100_blk64_sync.py
compute-sanitizer --tool memcheck --error-exitcode=1 \
  python -m pytest -q tests/test_sm100_blk64_sync.py
```

源码指纹变化，已有 blk64 AOT bundle 需要重新生成。

### 性能抽查（B300）

同一张 B300 上依次运行 `c514e9a` 和修复版本，每个用例用 CUDA Graph 重放
5 次调用，测量 11 轮取中位数。FP8 输入预先量化，表中只计 attention，不计
量化和 Python 调用开销。未锁定时钟；这组数据未见明显性能回退，不据此声称加速。
4 个用例的新旧 O 和 LSE 均逐位一致。

| 用例（B=1，D=128） | 修复前 ms | 修复后 ms | 变化 |
|---|---:|---:|---:|
| FP8，H=14，Q=KV=39756，top-k≤160，split=1 | 1.98107 | 1.97516 | -0.3% |
| BF16，同上 | 2.68296 | 2.71678 | +1.3% |
| FP8，H=8，Q=KV=8192，固定 top-k=32，split=1 | 0.07939 | 0.07917 | -0.3% |
| FP8，H=4，Q=64，KV=32768，top-k≤128，split=4 | 0.04303 | 0.04269 | -0.8% |

变长候选数用例包含 0，所有用例均包含短 `block_sizes`。记录来自 B300 作业
`5174215` 的 `candidate5` 检查阶段和一组修复前/后性能结果。GPU 验证使用 JIT
路径；本次未重建 AOT bundle。

## 2026-10-21：BF16 blk64 KV 尾块越界读取

### 问题

BF16 blk64 默认把 K/V 看成 rank-6 稀疏块视图 `(64, 64, 2, H, ceil(S/64), B)`，
其中没有真实序列长度。KV 长度不是 64 的倍数时：

- 最后一个短块会把下一 head 的前几行当成 key/value 读进来；不传
  `block_sizes` 时这些行参与 softmax，结果错误。客户形状
  `B=1,H=14,S=39756` 不传 `block_sizes` 时 14 个 head 里 13 个相对误差约 5%。
- 最后一个 head 的短块读到 K/V 分配之外。传了 `block_sizes` 也只能把分数
  置零，越界读到的 V 若是 NaN 仍会污染输出；若那段地址未映射则
  illegal memory access，或者 TMA 永远不到账导致卡死。此前在客户形状、
  `kv_splits=2` 序列中观察到的卡死即属此类：把 K/V 放在映射区末尾、后面
  地址不映射时，旧代码四种配置（有无 `block_sizes` × split 1/2）全部
  illegal memory access。
- 实测 B>1 时从第二个 batch 起整个 head 都错（即使传了 `block_sizes`），与
  rank-6 视图下 batch 基址不再是 64 行块整数倍一致。

FP8 早已改为保留真实序列长度的视图，不受影响。

### 修复

- KV 长度不是 64 的倍数时，BF16 改用已有的 exact 序列视图（原本只在 stride
  超出 Int32 TMA 范围时启用），TMA 按真实长度对尾块零填充；同时打开与 FP8
  相同的物理尾块掩码，不依赖调用方传 `block_sizes`。对齐长度仍走 rank-6
  视图，代码路径与结果不变。
- exact 视图原来用 `zipped_divide` 取块，load warp 每次取块都要对动态块数做
  一次除法取模，整体慢约 15%；改为只切序列维的 `local_tile`，与 FP8 写法一致，
  速度与 rank-6 持平。大 stride（Int64）路径同样受益。
- AOT 无需新增变体：exact 视图对应已有的 `use_int64_kv_strides=True` 变体。

### 验证（B200 与 B300 各一遍）

- **精度矩阵**：8 个形状（含 B>1、H>1、S_k=1/70/1000/1999/4097 及对齐长度）×
  `block_sizes` 无/rank-1/rank-3 随机短块 × top-k 固定/变长含 0 × CLC 开关 ×
  split 1/2，共 192 例，与 FP32 精确参考比较 O 最大误差 <3e-2、LSE <2e-3，空行
  O=0、LSE=-inf：新代码 0 失败；旧代码 137 例失败。
- **越界探针**：K/V 紧贴映射区末尾，新代码四种配置正常且无 NaN。
- **客户形状**：S=39756 有无 `block_sizes`、split 1/2 全部正确；此前卡死的调用
  序列 90 例全部完成；900 例完整矩阵无卡死、无 NaN、重复运行一致。
- **对齐路径不变**：同种子、同进程顺序下，6 个对齐形状共 540 例（FP8 108、
  BF16 432）新旧输出逐位一致；非对齐形状上 FP8 逐位一致，BF16 只在旧代码出错
  的用例上不同。旧代码在非对齐矩阵第 48 例再次卡死，新代码 270 例全部完成。
- **compute-sanitizer**：racecheck、memcheck 0 错误；synccheck 仍为既有的
  correction warp divergent 报告，与本改动无关。该告警随后由上方的
  blk64 warp 同步修复消除。
- **测试套件**：FP8 相关 B200 257、B300 258 passed；`test_flash_fwd.py -k sm100`
  190 passed（含新增 4 个尾块多 head 回归用例）；AOT 测试 34 passed；AOT-only
  660 例与 JIT 逐位一致（未构建的 BF16 rank-3 `block_sizes` 变体除外）。

### 性能（B200，客户形状 BF16 blk64，两轮）

| S_k | `block_sizes` | 修复前 ms | 修复后 ms |
|---:|---|---:|---:|
| 39756 | 无 | 3.02 / 3.03（结果错误） | 3.06 / 2.97 |
| 39756 | 有 | 3.14 / 3.13 | 3.09 / 3.02 |
| 39744 | 无 | 3.05 / 3.01 | 3.06 / 3.00 |

## 2026-10-20：blk64 softmax 推迟等待统计量槽位

起因是客户在 4×B200、Ulysses=4 的视频生成流程中观察到 FP8 慢于 BF16。
按客户单卡形状 `B=1,H=14,S=39756,D=128`、blk64 top-k 160 单独复测，FP8 本身
并不慢：B200 上 BF16 blk64 3.00 ms，FP8 含新鲜量化 2.69 ms；FP8 wrapper 每次
调用的 host 入队约 30–60 µs，没有隐藏的 host 同步。客户端到端变慢需要结合其
集成代码和 attention 内部算子拆分再查。

### 改动

原实现中 softmax 每个 step 末尾都等 correction 读走上一份 `acc_scale` 才进入
下一步；NCU 采样里这两个等待点约占全部 warp 采样的 15%。现在改到下一步算完
row max、真正要写 `sScale` 之前再等，最后一步之后的等待移到写最终 row sum/max
之前，与 blk128 已有做法一致。每个 tile 的等待次数、先后顺序和 named barrier
arrive 次数都不变；只改同步时机，不改数值。BF16 blk64 共用这段代码。

### 性能

同一进程内先编译预热，CUDA event 每轮 30–40 次取中位数；`f953680` 与新树交替
运行两轮。时钟未锁定。

| GPU | 路径 | 形状 `B,H,S,top-k` | `f953680` ms | 本次 ms | 变化 |
|---|---|---|---:|---:|---:|
| B200 | FP8 blk64 | 1,14,39756,160（客户） | 2.4365 | 2.3708 | -2.7% |
| B200 | BF16 blk64 | 1,14,39756,160（客户） | 3.1391 | 3.0214 | -3.8% |
| B300 | FP8 blk64 | 1,14,39756,160（客户） | 2.1504 | 2.0777 | -3.4% |
| B300 | BF16 blk64 | 1,14,39756,160（客户） | 2.8606 | 2.8258 | -1.2%（两轮 -2.8%/+0.3%，噪声内） |
| B200 | FP8 blk64 | 1,8,16384,64 | 0.2769 | 0.2701 | -2.5% |
| B200 | FP8 blk64 | 1,24,8192,32 | 0.2605 | 0.2585 | -0.8% |
| B200 | FP8 blk64 | 1,4,216832,342 | 7.8621 | 7.6554 | -2.6% |

`B=2,H=6,S=4096,top-k=8` 单次调用约 45 µs，低于 CUDA event 的可靠分辨率；
nsys 统计 604 次 kernel，B200 平均 36.90→36.34 µs（-1.5%）。

### 正确性与稳定性验证（B200 与 B300 各一遍）

- **逐位一致**：10 个形状（含客户形状、Q/KV 尾块、top-k=1/2/奇数、B>1）×
  普通/大动态范围数据 × `block_sizes` 无/rank-1/rank-3 × top-k 固定/变长/含 0，
  FP8 与 BF16（CLC 开关、split 1/2）共 857 个用例，新旧输出逐位一致；
  每例重复 3–5 次结果一致，无 NaN/Inf。
- **FP8 精度**：与 FP32 精确参考（反量化输入、相同稀疏索引）相比相对平均误差
  ≤2.3%，满足 <2.9% 合同。
- **compute-sanitizer**：racecheck、memcheck 新旧均 0 错误。synccheck 在
  `q2k_block_nums` 变体上新旧都报 correction warp 的 "Divergent thread(s) in
  block"（固定 top-k 变体干净），属于既有问题，与本改动无关。该告警随后由
  本页开头的 warp 同步修复消除。
- **AOT**：重建精简 blk64 bundle（BF16 split 1/2、FP8 split 1/4、
  `block_sizes` 0/1/3）后官方 AOT 测试 B200 34 passed、B300 35 passed，0 failed；
  AOT-only 模式下 35 个用例与 JIT 输出逐位一致。
- **测试套件**：FP8 相关 B200 257 passed、B300 258 passed；
  `tests/test_flash_fwd.py -k sm100` 两卡均 186 passed；均 0 failed。

### 验证中发现的既有问题（`f953680` 同样存在；1、2、3 已在 2026-10-21 修复）

1. BF16 blk64 在客户形状、大动态范围数据、`kv_splits=2`、无 `block_sizes`
   的调用上，按特定调用历史可稳定卡死（GPU 100%、停在 synchronize），单独
   调用不复现。
2. BF16 blk64 在 KV 长度不是 64 的倍数且 B×H>1 时结果错误或出现 NaN
   （即使传了 `block_sizes`）；对齐长度正常。FP8 路径不受影响。
3. 上述 synccheck 报告。
4. 曾试验的 blk128 配置组合（SM100 用 SM103 的缓冲开关，或 SM103 用 16 列
   correction/epilogue tile）会算错；生产配置不受影响。

源码指纹变化，现有 blk64 AOT bundle 需要重新构建；blk128 未改动。环境：
`nvcr.io/nvidia/pytorch:26.05-py3`（Torch `2.12.0a0+5aff3928d8.nv26.05`、
CUDA 13.2）、CUTLASS DSL 4.5.2、apache-tvm-ffi 0.1.14.post1。

## 2026-09-09：SM100/SM103 Sage FP8 AOT

SM100 和 SM103 的 Sage FP8 forward 现已接入现有 CuTe native AOT artifact
bundle。AOT 覆盖固定或逐 Q block 的 top-k、三种 `block_sizes` 作用域、
split-KV producer/combine，以及 SM103 的软件和 16-token `ld.red` row-max
两种路径。未命中 artifact 时仍可回退 JIT；`BSA_SM100_AOT_ONLY=1` 会严格
禁止运行时调用 `cute.compile()`。

最终代码已在 B200 和 B300 上真实导出并加载 AOT `.so`，分别覆盖动态
top-k/三维 `block_sizes`、split-KV producer/combine 和 SM103 `ld.red`。
三条 AOT-only 精度用例均通过，并在测试中显式禁止了 `cute.compile()`。

最终回归在 B200 上为 123 passed、24 skipped，在 B300 上为 124 passed、
23 skipped，均为 0 failed。最终源码重新生成的普通 FP8、split-KV、SM103
`ld.red` 和原有 BF16 AOT 均通过 AOT-only 真机精度测试。

### 尾块性能回退和修复

早期实现把 KV 尾块长度的动态计算放进 SM100 的 softmax 热循环。SM100 有
8 个 softmax warp，同一份判断会被重复执行并放大编译器控制流；B200 常用
unsplit 用例因此从约 2.42 ms 退化到约 3.78 ms。两版内核都是 128 个寄存器
且没有 spill，但 cubin 从 132264 bytes 增至 193536 bytes，反汇编从约 5218
行增至 6378 行，确认问题是热循环代码膨胀而不是寄存器压力。

最终实现不在 host 生成或修改 `block_sizes`，也不补齐 Q/K scale。一个
SM100 FP8 kernel 根据运行时 K 长度做统一判断：64 对齐时直接执行原 hot
loop；确有尾块时，同一 kernel 内计算最后一个物理块的有效长度、收紧用户
提供的 block size 并完成 score masking。split-KV 的尾部 K-scale 也在同一
判断下安全索引。

性能回归使用 `origin/master` (`8a9e8d7`) 作为基线，在同一张 B200/B300
上测试下方 12 组客户规模。每组预热 10 次、计时 21 次取中位数，前后版本
交换顺序跑两轮，再取几何平均：

- B200 最终 JIT 相对 master 快 0.871%，12 项范围为 -2.482% 到 +0.096%；
- B300 最终 JIT 相对 master 快 2.883%，12 项范围为 -4.334% 到 -0.623%；
- B200 原先约 2.42 ms 到 3.78 ms 的回退已消失，最终为 2.388 ms，
  同场 master 为 2.416 ms；
- 输出抽样汇总相对 master 的最大绝对差为 `2.0981e-05`，完整精度门槛
  全部通过。

AOT 的主要价值是免去首次编译：当前源码的首个 unsplit 调用在 B200/B300
上需要约 34.78/37.03 秒 JIT，预编译后首次加载并执行为约 0.053/0.072 秒。
同一 kernel 同时包含 aligned 和 tail 的内部实现，因此 JIT 编译时间高于
master 的约 11.35/12.20 秒；这是一次性的编译开销，不是上述稳态执行时间。
B200 上 AOT native ABI 的 12 项稳态几何平均比同一最终 JIT 慢 1.153%，
主要来自调用开销。B300 bring-up 板在连续 AOT 轮次中触发了明显温度/时钟
波动，因此不把其 AOT 稳态数字用于性能结论。

## 2026-09-08：与 SM120 对齐动态输入能力

SM100 和 SM103 的 Sage FP8 接口已解除原先 `B=1`、`H in {4,8}`、Q/KV
长度必须是 64 倍数、uniform top-k 和 rank-1 `block_sizes` 的限制。现在支持：

- 任意正数 batch 和 MHA head 数；
- 非 64 对齐的 Q/KV 长度（Q/K/V 和 scale 均保持真实长度；SM100/SM103
  kernel 根据运行时长度直接判断并屏蔽尾块）；
- `[B,H,Q_blocks]` 的 `q2k_block_nums`，SM100/SM103 仍支持空行；
- `[N]`、`[B,N]`、`[B,H,N]` 三种 `block_sizes`。

真机定向验证覆盖 B200/SM100 和 B300/SM103，包括 `B=2`、`H=1/3/5`、
Q/KV 尾块、三种 block-size 作用域、可变/空 top-k，以及 H=1/5 下
1/4/8/16 路 split。完整 FP8/量化回归在 B200 上为 68 passed、14 skipped，
在 B300 上为 69 passed、13 skipped；AOT、动态 compile-key 和 packaging
定向回归为 49 passed、9 skipped。SM100/SM103 和 SM120 的 FP8 均支持 AOT
或 JIT。

以下早期性能结果中的 `B=1`、`H=4/8` 是当时的基准输入范围，不再是当前
接口限制。

## 早期固定形状性能基准

以下性能数据使用前向、BHSD 布局的 Sage FP8 block-sparse attention：
`B=1`、`H=4/8`、`D=128`、`64x64` 稀疏块、uniform top-k，输出为
BF16。这些是基准输入，不是当前接口限制。

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
