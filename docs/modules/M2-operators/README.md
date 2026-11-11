# M2 · 算子优化

> 状态：**大纲**。本模块的完整内容（逐节讲解与练习）尚未写出，先把学习路径、资源与每节的产出定下来。
>
> 前置：M0（Roofline、profiling、压测脚本）与 M1 第 1–2 周（vLLM 服务已跑起来）。M2 的终点不是“会写 kernel”，而是“自己写的 kernel 跑进了推理引擎，端到端变快了”。

## 为什么单开一个模块

M1 让你会用引擎、会调参数，但引擎内部的 attention、matmul、量化 kernel 对你还是黑盒。M2 的目标是打开这个盒子：先学会写 kernel，再把 kernel 接回引擎，最后用 M0 的压测脚本量出端到端的变化。现有的练习资源（GPU Puzzles、Triton Puzzles、LeetGPU）都在 kernel 层面收尾，**“接入引擎”这一步目前没有现成课程**，是本模块要补的关键一节。

## 课程安排（5 周，每周 5 节、每节 30 分钟，沿用 M1 的节奏）

### 第 1 周 · CUDA 基础：先理解 GPU 再写代码

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | GPU 执行模型：thread、block、grid，显存层级 | [GPU Puzzles](https://github.com/srush/GPU-Puzzles) 前几题；[Stanford CS336 第 5 讲](https://www.youtube.com/watch?v=6OBtO9niT00) | 能画出一次 kernel 启动的数据流向 |
| 2 | 向量加法与访存合并（coalescing） | [GPU Puzzles](https://github.com/srush/GPU-Puzzles) | 跑通 3 道谜题，理解为什么连续访存快 |
| 3 | Shared memory 与分块（tiling） | [GPU Puzzles](https://github.com/srush/GPU-Puzzles)；[GPU MODE 讲座](https://github.com/gpu-mode/lectures) | 一份矩阵乘法的分块实现与耗时对比 |
| 4 | 占用率与 launch 开销：什么时候 kernel 启动本身就是瓶颈 | [GPU MODE 讲座](https://github.com/gpu-mode/lectures) | 用 profiler 指出一次推理里 launch 开销的占比 |
| 5 | 小结：写一份“CUDA 性能检查清单” | — | 检查清单，后面每节对照 |

**硬件**：需要一块 NVIDIA GPU（租卡或本地卡均可；LeetGPU 提供免费 GPU 评测，见第 3 周）。无卡时先读题、看题解，不要跳过——第 2 周的 Triton 部分无需 GPU。

### 第 2 周 · Triton 入门：用 Python 写 kernel

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | Triton 语法与 block 编程模型 | [Triton 官方教程](https://github.com/triton-lang/triton/tree/release/3.6.x/python/tutorials)（向量加法 → fused softmax） | 第一个能跑的 Triton kernel |
| 2 | 谜题 1–3：逐元素、归约、矩阵乘 | [Triton Puzzles](https://github.com/gpu-mode/Triton-Puzzles)（可用解释器，**无需 GPU**）；轻量版 [Triton-Puzzles-Lite](https://github.com/SiriusNEO/Triton-Puzzles-Lite) | 3 道谜题通过 |
| 3 | Fused softmax 与数值稳定性 | [Triton 官方教程](https://github.com/triton-lang/triton/tree/release/3.6.x/python/tutorials) | 一个 fused softmax，与 PyTorch 版对上数值 |
| 4 | 中文讲解对照 | [Datawhale《大模型算法与系统教程》Part 03 Triton 部分](https://github.com/datawhalechina/llm-algo-leetcode)（15 节，中文 Notebook） | 把本周概念用中文复述一遍 |
| 5 | 小结：Triton 什么时候够用，什么时候必须回 CUDA | — | 一页取舍笔记 |

**硬件**：本周围绕解释器与纯 Python 可做；跑性能对比时再上 GPU。

### 第 3 周 · Fused Attention 与量化 Matmul

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | FlashAttention 的思路：分块 + 在线 softmax | [Triton 官方教程](https://github.com/triton-lang/triton/tree/release/3.6.x/python/tutorials)（fused attention） | 读懂分块 attention 的数据流 |
| 2 | 动手写 fused attention | 同上；[GPU MODE 讲座](https://github.com/gpu-mode/lectures)中 FlashAttention 相关章节 | 一个能跑的 fused attention，与 PyTorch 数值对齐 |
| 3 | 量化 matmul：INT8/FP8 的访存收益 | [GPU MODE 讲座](https://github.com/gpu-mode/lectures) | 一份量化前后访存量的手算 |
| 4 | 刷题验收 | [LeetGPU](https://leetgpu.com)（题库：[leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges)，提供免费 GPU 运行提交） | 至少 3 道 attention / matmul 相关题通过 |
| 5 | 小结：把本周 kernel 的 Roofline 点标到 M0 的图上 | M0 的 Roofline 图 | 更新后的 Roofline 图 |

**硬件**：本周需要 GPU 跑性能对比；无卡可用 LeetGPU 的免费 GPU 评测先做正确性验证。

### 第 4 周 · FlashAttention 与 CUTLASS：读工业级实现

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | FlashAttention 2/3 的工程取舍 | [GPU MODE 讲座](https://github.com/gpu-mode/lectures) | 一份“论文做法 vs 工程做法”对照笔记 |
| 2 | CUTLASS / CuTe 的角色：模板化高性能 GEMM | [GPU MODE 讲座](https://github.com/gpu-mode/lectures) | 能讲清 CUTLASS 解决什么问题 |
| 3 | 读 vLLM / FlashInfer 里一个 attention kernel 的调用链 | vLLM 仓库源码；[vLLM CustomOp 设计文档](https://docs.vllm.ai/en/latest/design/custom_op/) | 一张调用链图：请求 → 调度 → kernel |
| 4 | 对比：Triton 版 vs CUTLASS 版 attention 的性能与可维护性 | 第 3 周的实测数据 | 一页对比表 |
| 5 | 小结：什么时候自己写 kernel，什么时候直接用库 | — | 取舍笔记 |

**硬件**：以读代码为主，无卡可做；有卡可复现第 3 周的对比。

### 第 5 周 · 收口：把自写 kernel 接入推理引擎

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | vLLM 的 CustomOp 机制：算子如何被替换 | [vLLM CustomOp 设计文档](https://docs.vllm.ai/en/latest/design/custom_op/) | 能讲清插件算子的注册与调用路径 |
| 2 | 动手：把第 3 周的 fused attention 包成 CustomOp | 同上 | 可加载的 CustomOp |
| 3 | 端到端验证：同一模型、同一评测集，接入前后的 TTFT / ITL / 吞吐 | M0 的 `scripts/bench_openai.py` 与基线报告 | 一份接入前后对比表 |
| 4 | 负面结果也要写下来：没变快的原因分析（launch 开销？访存没省下来？batch 太小？） | M0 第 3 节的 profiling 方法 | 一份原因分析 |
| 5 | 小结：M2 毕业——一份“自写 kernel 端到端加速报告” | — | **M2 的可交付物** |

**硬件**：本周需要 GPU（租卡）。无卡时先完成第 1–2 节的机制学习，端到端验证标注“待有卡后补测”。

## 本模块检查清单

- [ ] GPU Puzzles 核心谜题通过，能讲清访存合并与分块
- [ ] Triton Puzzles（解释器版）通过，写出过 fused attention
- [ ] LeetGPU 上至少 3 道相关题通过
- [ ] 能画出 vLLM 里一个 attention kernel 的调用链
- [ ] 有一份“自写 kernel 接入引擎前后”的端到端对比报告（含负面结果）
