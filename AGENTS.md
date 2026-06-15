# Block Sparse Attention

## 编码规范

- **张量前缀**：`m*`(GMEM)、`g*`(GMEM tile)、`s*`(SMEM)、`t*`(线程视图)、`acc_*`(累加器)
- **类名** PascalCase + 架构后缀（如 `SparseAttentionUniversal`）；**方法名** snake_case，内部方法 `_` 前缀
- **Kernel 类结构**：`__init__`(host 配置) → `@cute.jit __call__`(launch) → `@cute.kernel`(设备主体)
- **注释英文**，代码 4 空格缩进，类型标注使用 `cute.Tensor` / `Optional[cute.Tensor]` / `cutlass.Constexpr[...]`
- **导入顺序**：`cutlass` → `cutlass.cute as cute` → `cutlass.cute.nvgpu` → `cuda.bindings`

## 验证流程

1. 修改 kernel 代码后，**先运行对应的单元测试**
2. 确认**全部用例 PASS** 后再 commit
3. 若修改了 `src/common/` 的共享组件，需运行**所有测试**都通过
4. 新增功能必须有对应测试覆盖
5. **如果单个 test case 或 demo 运行超过 30s，则认定为死锁**，死锁问题的解决见 skill: `AI/DEBUG_2CTA.md`

## 开发测试规范

- **编译计时**：每次 `cute.compile(...)` 后必须打印编译耗时，区分编译慢和 kernel 死锁：
  ```python
  import time
  t0 = time.time()
  compiled = cute.compile(demo, ...)
  print(f"Compiled in {time.time() - t0:.1f}s")
  compiled(...)  # If this times out, treat it as a deadlock.
  ```
- **超时 30s = 死锁**：kernel 运行超过 30s 认定为死锁，不是编译慢（编译通常 < 10s）

## 文件组织规范

除非用户明确指定路径，agent 创建的临时文件必须按以下约定存放：

- **测试/复现文件**：`/home/scratch.cjerry_sw/BSA/agent/agent_tests/`
- **Benchmark 文件与结果**：`/home/scratch.cjerry_sw/BSA/agent/agent_benchmark/`
- **Profile/NCU/NSYS/Perfsim 文件**：`/home/scratch.cjerry_sw/BSA/agent/agent_profiles/`
- **临时工作区、构建缓存、调试数据**：`/home/scratch.cjerry_sw/BSA/agent/agent_space/`

`/home/scratch.cjerry_sw/BSA/agent/` 整体不被 git 追踪；不要再向
`tests/temp/` 或 `benchmarks/temp/` 新增 agent 临时文件。

## 工作流程

- 每完成一项子任务，必须进行 git commit
- Commit message 应简洁描述完成的子任务内容
- 提交前必须通过相关测试（见上方验证流程）
- 每次 commit 之前，检查相关文档是否需要同步更新
