# LLM 推理工程：原理与方法 { .home-title }

**从单条请求的延迟拆解，到推理引擎调度、GPU 算子优化与国产芯片适配。**

大语言模型推理是一门贴近物理极限的系统工程：换一块算力翻倍的显卡为什么解码速度几乎不动？并发翻倍后为什么吞吐见顶而尾延迟骤增？PagedAttention、量化、投机解码、FlashAttention 和国产 NPU 适配各自到底在省什么？本书用直观图解、手算账本与可复用的工程方法，打通 LLM 推理全栈。

[从第 1 章开始](docs/chapters/chapter-01.md){ .md-button .md-button--primary }
[按角色选路线](#reading-paths){ .md-button }
[硬件与模型速查](docs/toolkit/hardware-models.md){ .md-button }

<figure class="book-figure">
  <img src="assets/figures/fullstack-map.svg" alt="LLM 推理工程四层全栈地图" loading="lazy">
  <figcaption><span class="book-figure__label">全景图</span>从上层服务接入到底层异构硬件，贯穿一条「性能度量与 Profiling」主轴，对应全书 5 篇 23 章。</figcaption>
</figure>

## 按角色选择阅读路线 { #reading-paths }

不必线性读完所有章节。结合你当前的工程目标选一条主线，遇到底层概念再随时按图索骥。

<div class="grid cards part-cards" markdown>

- **推理框架与部署优化**

    ---

    适合推理平台、模型部署与性能调优工程师。

    **推荐路径**：第 1–11 章（第一篇 + 第二篇）。

    **核心收获**：掌握 Roofline 与模型账本、延迟分布与 Profiling 读法，吃透 vLLM / SGLang / TensorRT-LLM 核心机制、量化、投机解码、PD 解耦与成本路由。

    [第一篇：直觉与度量](docs/parts/part-1.md) · [第二篇：引擎与优化](docs/parts/part-2.md)

- **GPU 算子工程**

    ---

    适合希望打开引擎黑盒、编写与接入高性能 kernel 的工程师。

    **推荐路径**：第 1–4 章 → 第 12–17 章（第一篇 + 第三篇）。

    **核心收获**：建立 GPU 执行模型与存储层级心智，走完 GEMM 四版优化阶梯、Triton Block 编程、FlashAttention 在线 Softmax、量化 Matmul、CUTLASS 与引擎注册分发。

    [第一篇：直觉与度量](docs/parts/part-1.md) · [第三篇：算子优化](docs/parts/part-3.md)

- **国产芯片适配**

    ---

    适合负责异构算力迁移、昇腾生态部署与跨芯片对比的工程师。

    **推荐路径**：第 1–6 章 → 第 18–23 章（基础 + 引擎主干 + 第四、五篇）。

    **核心收获**：看懂国产芯片四面墙与五层软件栈，深入昇腾 CANN / AscendC、`vllm-ascend` 插件机制、算子覆盖与回退定位、跨芯片算子迁移与精度性能归因。

    [第四篇：国产芯片](docs/parts/part-4.md) · [第五篇：综合归因](docs/parts/part-5.md)

</div>

## 全书结构总览

全书按「5 篇 23 章主线 + 进阶专题库 + 参考手册与工具箱」组织，既支持系统顺读，也方便日常速查。

### 五篇主线

| 篇目 | 章节范围 | 核心问题与方法主线 |
|---|---|---|
| [第一篇 性能直觉与度量方法](docs/parts/part-1.md) | 第 1–4 章 | 用 Roofline 三区间与模型 M/F/R 账本定瓶颈；拆解 TTFT/ITL/吞吐/排队与双泳道 Profiling |
| [第二篇 推理引擎与部署优化](docs/parts/part-2.md) | 第 5–11 章 | 拆解 vLLM、SGLang、TensorRT-LLM 架构；系统讲透监控拐点、量化、投机解码、PD 解耦与成本路由 |
| [第三篇 GPU 算子优化与引擎接入](docs/parts/part-3.md) | 第 12–17 章 | 从 GPU 硬件执行到 GEMM 四版阶梯、Triton、Fused Attention、量化 Matmul、CUTLASS 与 CustomOp 接入 |
| [第四篇 国产芯片适配原理与方法](docs/parts/part-4.md) | 第 18–22 章 | 以华为昇腾为主线，打通软件栈地图、`vllm-ascend`、CANN/AscendC、覆盖回退诊断与跨芯片迁移 |
| [第五篇 综合分析与方法收口](docs/parts/part-5.md) | 第 23 章 | 建立跨芯片配对精度回归、五步性能归因闭环、完整分析报告范式与系统论文三遍读法 |

### 进阶专题与参考工具箱

| 板块 | 页面 | 适用场景 |
|---|---|---|
| 进阶专题库 | [专题 A：前沿长上下文与混合架构拆账](docs/topics/advanced-architectures.md) | 深度对比稠密、MoE、MLA、压缩稀疏与递推混合五类前沿架构的显存与 1M 上下文账 |
| 进阶专题库 | [专题 B：大规模集群 PD 与分布式 KV 池](docs/topics/cluster-pd-mooncake.md) | 拆解 Mooncake 以 KV 为中心的集群架构、跨节点传输账、池配比与容错设计 |
| 进阶专题库 | [专题 C：端侧与浏览器推理运行时](docs/topics/edge-runtimes.md) | 梳理 llama.cpp、ONNX Runtime、MLC/WebLLM 等本地与端侧部署边界 |
| 进阶专题库 | [专题 D：各章核心论文精读与数据口径](docs/topics/paper-notes.md) | 集中查阅 PagedAttention、FlashAttention、SmoothQuant、Speculative Decoding 等关键论文要点 |
| 进阶专题库 | [专题 E：主流国产芯片生态速查](docs/topics/domestic-ecosystems.md) | 对比昇腾、海光 DCU、寒武纪、壁仞、沐曦、摩尔线程等软件栈与 vLLM 适配路径 |
| 参考工具箱 | [硬件规格与模型账本速查](docs/toolkit/hardware-models.md) | 查阅 A100 / H100 / Ascend 910B 硬件参数与 Llama-3.1 / Qwen3 等模型手算表 |
| 参考工具箱 | [测量口径与对比分析方法](docs/toolkit/methodology.md) | 统一压测条件、分位数口径、精度检验与跨配置对比边界 |
| 参考工具箱 | [工程脚本与配置参考](docs/toolkit/tools-and-configs.md) | 获取 SSE 流式压测脚本、Prometheus 监控配置与常见引擎启动参数参考 |

## 阅读说明

为保证全书公式与数量级直觉前后一致，主线算例统一采用 **NVIDIA A100 80GB SXM**（FP16 312 TFLOPS，HBM 2 TB/s，脊点 156 FLOP/B）与 **NVIDIA H100 80GB SXM**（FP16 989.5 TFLOPS，HBM 3.35 TB/s，脊点 295 FLOP/B）作为 GPU 基准，以 **Llama-3.1-8B**（FP16 权重 16 GB，KV 128 KiB/token）与 **Qwen3-8B**（BF16 权重 16.38 GB，KV 144 KiB/token）作为主线模型，国产芯片主线采用 **昇腾 Ascend 910B（64 GB HBM）**。文中的手算数字旨在建立清晰的数量级直觉与优化方向感，换用其他卡型或模型时可按同一套公式直接代换；完整规格速查见 [硬件规格与模型账本速查](docs/toolkit/hardware-models.md)。

## 站点使用提示

- **目录与导航**：左侧栏按篇目与章节跳转，页面底部提供上一章 / 下一章连续阅读入口。
- **术语悬停**：正文首次出现的专业术语带虚线下划线，悬停即可查看精简释义，点击跳转 [术语表](docs/glossary.md)。
- **搜索与进度**：顶部搜索框支持全文与代码检索；[阅读进度](docs/progress.md) 支持在浏览器本地勾选记录已读章节并导出同步。
- **右侧笔记栏**：每章配备独立笔记区，默认保存在浏览器本地；若配置 GitHub 同步，请使用私有仓库，切勿在公开仓库写入密钥或敏感内容。

[阅读进度](docs/progress.md) · [术语表](docs/glossary.md) · [来源、贡献与许可](docs/colophon.md)
