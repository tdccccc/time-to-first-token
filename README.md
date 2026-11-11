<p align="center">
  <img src="assets/banner.png" alt="边做边学 LLM 推理服务：交付一个真实的推理服务">
</p>

# 边做边学 LLM 推理服务：交付一个真实的推理服务

一份 10 周、每天 30 分钟的路线图，给想真正跑起 LLM 推理、而不只是读一读的工程师。另有 1 处可选加深（第 8 周（续）大规模集群 PD 优化），不计入 10 周主线。

一共 50 节课。每一节都服务于同一个产物：一个 OpenAI 兼容的推理服务——你把它部署到租来的 GPU 上，做埋点、压测到 1000+ 并发，用量化和投机解码优化，在前面放一个成本感知的路由器，最后把它作为一份可复现的 benchmark 发表出来。

另一种做法是做十七个互不相干的小实验，大部分时间都花在搭环境上。一个不断长大的服务能覆盖同样的知识点，最后你手里还留下一个能拿得出手的东西。

> **本 fork 的改进**（相对上游 [patchy631/time-to-first-token](https://github.com/patchy631/time-to-first-token)）
>
> - **中文版**：主 README 翻译成中文，课程结构、做/读/略读标记、时间投入和 GPU 成本说明全部保留；英文原文保存在 [docs/original-README.md](docs/original-README.md)。
> - **模块地图**：本计划扩展为 M0–M4 五个模块，见 [docs/MODULES.md](docs/MODULES.md)。M1 就是下面这份 10 周路线图；M0 是前置的基础与度量，M2 是算子优化，M3 是国产芯片适配，M4 是跨芯片综合项目。
> - **算子与国产芯片**：新增 [M2 算子优化](docs/modules/M2-operators/README.md)与 [M3 国产芯片适配](docs/modules/M3-domestic-chips/README.md)的课程大纲，补上原路线图没有的两条线：自己写 kernel 并接进推理引擎、国产品牌芯片上跑通同一模型。
- **第 8 周加深**：新增第 8 周（续）：大规模集群 PD 优化——Mooncake 的 KV 中心架构、跨节点 KV 池、KV 传输账、池配比与容错。论文一律作选读，核心内容写进正文。
> - **原项目授权与出处不变**：本仓库沿用上游的 CC BY 4.0 协议，原作者与原仓库链接见文末。

---

## 适合谁

你应该熟悉 Python、了解 Transformer 的架构、会用命令行。不需要有 serving、Kubernetes 或 CUDA 的经验。

如果你已经了解某个主题，标记为**略读**的课可以压缩。**做**的课不要压缩，它们才是重点。

## 学完之后你将得到什么

- 一套你自己配置、埋点、调优过的推理服务
- Grafana 看板，展示 TTFT（首 token 延迟）、token 间延迟、吞吐、排队深度和每次请求的成本
- 一个压测工具，能稳定地打出 1000+ 并发
- FP16、FP8、INT4、投机解码、KV 淘汰等多个变体的 benchmark 数据
- 一个按成本、延迟、质量选后端、带每请求 token 预算的路由器
- 一份发表出来的 benchmark 写作，版本固定、命令可复现
- （可选第 8 周（续））一张大规模集群 PD 架构图、KV 传输账表与“信号→动作→验证方法”策略表

## 时间投入

| | |
|---|---|
| 每节课时长 | 30 分钟 |
| 每周节数 | 5 节，另有 2 天缓冲 |
| 总计 | 10 周，50 节，25 小时（核心）；另有第 8 周（续）5 节可选加深 |
| GPU 成本 | 大约一块 24GB 显卡按半小时租用，另加两次 H100 |

缓冲日是用来补课的，不是用来加新内容的。周二缺了一节，不至于让整个计划垮掉。

---

## 路线图为什么是这个顺序

这个顺序是刻意排的，和大多数人写清单的方式不一样。背后有五个决定。

**Roofline 模型放在最前面。** 后面每一次优化，都是同一张图上的一个移动。量化打的是解码阶段受显存带宽限制的问题；连续批处理把算术强度往计算上限推；投机解码用便宜的 FLOPs（在显存带宽受限的情况下 FLOPs 是便宜的）换更少的顺序访存；解耦式部署（disaggregation）存在，是因为 prefill 受计算限制、解码受带宽限制，两者在一块 GPU 上打架。没有这个模型，后面就是一袋子技巧。

**度量先于优化。** 埋点在第 3 周，压测在第 5 周，远在调优旋钮之前。没有它们，后面的一切都无法验证；一份公开的 benchmark，大部分就是一个可信的度量工具加一个模型。

**PagedAttention 和连续批处理不值得自己重写。** vLLM 已经实现了这两者，分块 prefill（chunked prefill）是 vLLM 和 SGLang 默认的调度策略。重写它们，不如读懂 block manager 和调度器，直到你能从代码讲清楚：为什么显存浪费能降到 4% 以下，为什么 GPU 在请求之间不再空转。

**路由器放在单位经济学旁边。** 一个按成本、延迟、质量选后端的路由器，是唯一一个把“多少钱”和“延迟预算”写进代码的组件。同一周学经济学，读到的东西会变成路由策略，而不是一篇你点头同意的博客。

**端侧部署是可选的，放在最后。** ONNX Runtime、TensorRT-LLM、WebLLM 是客户端和嵌入式运行时，和前面九周搭建的数据中心那套几乎没有共同的操作面。

---

## 环境准备

### GPU

7–8B 的模型大约需要一块 24GB 显卡。按半小时租，用完关机。

| 服务商 | 适合 |
|---|---|
| [RunPod](https://www.runpod.io) | 按秒计费，开机器快 |
| [Modal](https://modal.com) | Serverless，最适合跑 benchmark 扫描脚本 |
| [Lambda](https://lambda.ai) / [vast.ai](https://vast.ai) | 按需和市场价 GPU，便宜 |
| [Colab](https://colab.research.google.com) | 纯 Python 的课，不用部署服务 |

第 1 周全部、第 3 周大部分、第 6 周的讲座日、第 9 周全部阅读都不需要 GPU。把租卡集中在动手的那几天。只有两节课必须用 H100：第 5 周的 1000 并发测试，以及第 8 周如果你真的要跑解耦式部署。第 8 周（续）的多节点实测需要多卡集群，租不到就只做部署验证、性能对比待补测。

### 需要安装的工具

```bash
pip install vllm
pip install guidellm
pip install sglang
```

另外还需要 Docker（跑 Prometheus 和 Grafana），以及第 8 周用的 `kind` 或一个小的托管集群。

---

## 路线图

图例：**读**表示这节课是输入背景知识，**做**表示这节课要产出东西。**略读**表示相关概念你可能已经知道，价值在动手部分。

---

### 第 1 周：心智模型

**重点。** 装上后面一切都挂在上面的那个模型。Roofline、算术强度，以及为什么解码等显存、prefill 等计算。

**形式。** 以视频为主。Roofline 的推导配上现场画图和示意图更容易理解。

**产出。** 手算出你要服务的那个模型的算术强度，并且能说清哪个阶段受带宽限制、为什么。

| 天 | 课程 |
|---|---|
| 周一 | *读* · [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)（Horace He），前半部分：计算、显存带宽、开销三种区间。 |
| 周二 | *读* · 读完这篇文章。算子融合、开销，以及为什么给一个受显存带宽限制的 kernel 加 FLOPs 没有用。 |
| 周三 | *读* · [Stanford CS336 第 5 讲：GPUs](https://www.youtube.com/watch?v=6OBtO9niT00)（Percy Liang、Tatsunori Hashimoto），前 30 分钟：执行模型与显存层级。[讲义](https://github.com/stanford-cs336/spring2025-lectures)。 |
| 周四 | *读* · CS336 第 5 讲后 30 分钟：算术强度、Roofline、为什么数据搬运占主导。文字版：[Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)（kipply）。 |
| 周五 | *做，略读* · 手算目标模型的算术强度。价值在于你自己算出这个数，而不是再读一遍解释。 |
| 缓冲 | [CS336 第 10 讲：Inference](https://www.youtube.com/watch?v=fcgPYo3OtV0)，然后读 [Databricks 那篇](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)开头 prefill 与 decode 的部分。 |

---

### 第 2 周：vLLM——先部署，再读内部实现

**重点。** 先把服务跑起来，然后钻进 block manager 和调度器，直到 paging 的设计从代码里看出来是显然的。

**形式。** PagedAttention 以文字为主，论文比视频更精确；V1 架构以视频为主。

**产出。** 一个跑着 7–8B 模型的 OpenAI 兼容接口，加上你自己写的、从源码讲清 PagedAttention 和 V1 调度器的笔记。

| 天 | 课程 |
|---|---|
| 周一 | *读，略读* · [PagedAttention 发布文章](https://blog.vllm.ai/2023/06/20/vllm.html)。以前的系统在 KV 显存碎片和过度预留上浪费 60–80%，固定大小的 block 把浪费压到 4% 以下。然后略读 [SOSP 论文](https://arxiv.org/abs/2309.06180)的 block table 部分。 |
| 周二 | *做* · 用 vLLM 的 OpenAI 兼容服务跑起 Llama-3.1-8B-Instruct 或 Qwen2.5-7B。把启动命令存下来，十周都要复用。[Modal 示例](https://modal.com/docs/examples/vllm_inference)。 |
| 周三 | *读* · [vLLM Office Hours 22：Intro to vLLM V1](https://www.youtube.com/watch?v=jmzIvQZCLZM)（Michael Goin，Red Hat）。文字版：[vLLM V1 架构](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)。 |
| 周四 | *读* · [Inside vLLM: Anatomy of a High-Throughput Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)，调度器与 block manager 部分。 |
| 周五 | *做* · 跟着 [这个仓库](https://github.com/vllm-project/vllm)里的 block table 代码走一遍。写三句话，讲清一个请求的 block 是怎么被查到的。写不出来就再读一遍。 |
| 缓冲 | 对照代码重读 V1 博客的调度器部分，记下博客做了哪些简化。 |

---

### 第 3 周：度量基础设施

**重点。** 先造出“看”的工具，再做优化。这一周之后的所有内容，都是因为这一周才看得清楚。

**形式。** 以文字为主。关于 benchmark 方法论的文字，比现有的任何视频都强。

**产出。** 一个实时 Grafana 看板，在你自己的服务上展示 TTFT、token 间延迟、吞吐和排队深度。

| 天 | 课程 |
|---|---|
| 周一 | *读* · [vLLM 指标设计文档](https://docs.vllm.ai/en/latest/design/v1/metrics/)。弄清 `num_requests_running`、`num_requests_waiting` 和延迟直方图到底在量什么。 |
| 周二 | *做* · 按 [Prometheus 和 Grafana 教程](https://docs.vllm.ai/en/latest/examples/online_serving/prometheus_grafana/)把监控搭在你的服务上。 |
| 周三 | *做* · 导入看板 JSON，确认在少量负载下面板真的在动。抓取配置现在就修好，别等第 5 周 1000 并发的时候再修。 |
| 周四 | *读* · [Databricks 性能工程](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)和 [NVIDIA benchmarking 基础](https://developer.nvidia.com/blog/llm-inference-benchmarking-fundamental-concepts/)。把它们定义的每个指标对应到一个面板上。没有定义的面板删掉。 |
| 周五 | *读* · [How to Benchmark LLM Engines](https://modal.com/llm-almanac/how-to-benchmark)（Modal）。为什么要扫请求速率而不是只测一个点，为什么饱和点要丢掉。 |
| 缓冲 | 可选：[vLLM Office Hours 21：Production Stack Deep Dive](https://www.youtube.com/watch?v=0ZVu0A4wWQg)，看已部署场景下的可观测性。 |

---

### 第 4 周：SGLang、RadixAttention 与批处理内部

**重点。** 通过对比学设计空间：固定 block 分页 vs 树形前缀复用。

**形式。** 混合。RadixAttention 的讲解来自项目负责人。分块 prefill 只有文字资料。

**产出。** 同一个服务的 SGLang 版本，以及第一份与 vLLM 的前缀复用对比。

| 天 | 课程 |
|---|---|
| 周一 | *读* · [RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/)。在前缀很重的负载上吞吐最高可达 5 倍，没有缓存命中时也没有明显开销，这就告诉你它赢在哪些负载上。 |
| 周二 | *做* · 用 [SGLang](https://docs.sglang.io)部署和第 2 周相同的模型。同一硬件、同一 prompt，唯一的变量是引擎。 |
| 周三 | *读* · [Efficient LLM Inference with SGLang](https://www.youtube.com/watch?v=Ny4xxErgFgQ)（Lianmin Zheng，SGLang 负责人）。论文：[SGLang，NeurIPS 2024](https://arxiv.org/abs/2312.07104)。 |
| 周四 | *读，略读* · [Continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)（Anyscale），然后 [Orca，OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)。重点是迭代级调度和重尾长度假设——正是它让那些数字成立。 |
| 周五 | *读* · [Sarathi-Serve](https://arxiv.org/abs/2403.02310)，关于分块 prefill 和无停顿调度。这就是分块 prefill 成为两个引擎默认策略的原因。 |
| 缓冲 | [SGLang v0.4 零开销调度器](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)，然后用一个共享 system prompt 的负载同时跑两个引擎，看 TTFT 的差距。 |

---

### 第 5 周：压测到 1000 并发

**重点。** 搭建可复现的压测工具，学会什么叫一份站得住脚的 benchmark——在你有东西可辩护之前。

**形式。** 以文字为主，只用文档和动手。

**产出。** 一份脚本化的并发扫描，超过 1000 并发，输出 p50、p95、p99，并用两个独立工具交叉验证。

| 天 | 课程 |
|---|---|
| 周一 | *读* · [GuideLLM 介绍](https://developers.redhat.com/articles/2025/06/20/guidellm-evaluate-llm-deployments-real-world-inference)并安装。它从同步基线扫到饱和，而不是只测一个任意的负载点。 |
| 周二 | *做* · 用 [GuideLLM](https://github.com/vllm-project/guidellm)跑一次扫描，输入和输出 token 长度固定。固定长度决定了 KV 缓存大小，也就决定了整个结果。 |
| 周三 | *做* · 在同一个负载上跑 [`vllm bench serve`](https://docs.vllm.ai/en/latest/benchmarking/cli/) 做交叉验证。两个工具结果不一致是信息，不是问题。 |
| 周四 | *做* · [genai-perf](https://github.com/triton-inference-server/perf_analyzer)并发扫描，从 1、2、4 一直到 128。找到吞吐饱和、延迟恶化的点，那个拐点是整条曲线上唯一有意思的点。 |
| 周五 | *做* · 只在这节课租一台 H100。冲到 1000 并发以上，同时盯着 KV 缓存利用率和 `num_requests_waiting` 看有没有抢占。跑的时候读 [vLLM 调优策略](https://developers.redhat.com/articles/2026/03/03/practical-strategies-vllm-performance-tuning)。 |
| 缓冲 | 把发表前的检查清单写进你的 benchmark 仓库。 |

---

### 第 6 周：量化取舍

**重点。** 第一个调优旋钮，也是你的压测工具第一次真正派上用场的周。

**形式。** 以视频为主。讲座是入口，论文是深挖。

**产出。** FP8 和 INT4/AWQ 变体，在吞吐和质量代理指标上都与 FP16 基线做了对比。

| 天 | 课程 |
|---|---|
| 周一 | *读* · [MIT 6.5940 第 5 讲：Quantization Part I](https://www.youtube.com/watch?v=TSc_BibWRhM)（Song Han），前半部分。[课程主页](https://hanlab.mit.edu/courses/2024-fall-65940)。 |
| 周二 | *读* · 读完第 5 讲，开始第 6 讲，深入 AWQ、GPTQ、SmoothQuant。SmoothQuant 和 AWQ 都是这个实验室的成果。 |
| 周三 | *读，略读* · [Lilian Weng 的综述](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)里的量化部分。写下来：你跑的每个引擎实际上实现了其中哪种方法。 |
| 周四 | *做* · 跑一个 AWQ 或 GPTQ 检查点和一个 FP8 变体。在与第 5 周相同的扫描下记录吞吐和显存。[vLLM 量化文档](https://docs.vllm.ai/en/latest/features/quantization/)。 |
| 周五 | *做* · 在 FP16、FP8、INT4 上跑一组固定评测集或困惑度代理指标。这张表才是产出，不是加速比那个数字。 |
| 缓冲 | [QLoRA 讲座](https://www.youtube.com/watch?v=fQirE9N5q_Y)（Tim Dettmers，LLM.int8() 和 QLoRA 的作者），然后 [LLM Compressor office hours](https://www.youtube.com/watch?v=WVenRmF4dPY)。 |

---

### 第 7 周：投机解码与 KV 淘汰

**重点。** 两个概念上广为人知、实现细节上少有人知的旋钮。

**形式。** 投机解码以视频为主，由写了 vLLM 实现的人来讲；KV 淘汰以文字为主。

**产出。** 一个投机解码变体和一个长上下文淘汰配置，都做了 benchmark，包括负面结果。

| 天 | 课程 |
|---|---|
| 周一 | *读，略读* · [投机解码文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)里的方法选择表，然后读 [vLLM 的 benchmark 文章](https://blog.vllm.ai/2024/10/17/spec-decode.html)：QPS 为 1 时最高 2.8 倍，但高 QPS 时会慢 1.4–1.8 倍。交叉点才是发现，不是加速比。 |
| 周二 | *读* · [GPU MODE 第 22 讲：Hacker's Guide to Speculative Decoding in vLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4)（Cade Daniel，Anyscale），前半部分：提议器、打分器、验证器、草稿模型 worker。 |
| 周三 | *做* · 打开投机解码，在低 QPS 和高 QPS 下分别测 token 间延迟。找到你自己的交叉点，这就是这节课。 |
| 周四 | *读* · [StreamingLLM](https://arxiv.org/abs/2309.17453)（Guangxuan Xiao 等，ICLR 2024）：相对滑动窗口重算最高 22.2 倍加速，超过 400 万 token 依然稳定。可选视频：[StreamingLLM and DuoAttention](https://www.csail.mit.edu/event/scale-ml-guangxuan-xiao-streamingllm-and-duoattention-efficient-and-effective-long-sequence)（MIT CSAIL）。 |
| 周五 | *做* · 配一个长上下文负载和一套淘汰策略，测显存和 TTFT。长上下文是淘汰从理论变成现实的地方。 |
| 缓冲 | 读完 GPU MODE 第 22 讲，然后略读 [speculators](https://github.com/vllm-project/speculators) 和[它的 office hours](https://www.youtube.com/watch?v=2ISAr_JVGLs)。 |

---

### 第 8 周：解耦式服务与 Kubernetes

**重点。** 第 1 周的 prefill/decode 不对称变成一种架构，再加上运行它的运维层。

**形式。** 解耦式部署以视频为主；Kubernetes 以文字为主，它变化太快，视频跟不上。

**产出。** 你的服务跑在 Kubernetes 上，用排队深度而不是 CPU 驱动自动扩缩容。

| 天 | 课程 |
|---|---|
| 周一 | *读* · [DistServe at OSDI 2024](https://www.youtube.com/watch?v=WwJvecXOeUA)（Yinmin Zhong，报告人）。报告称可多服务 7.4 倍请求，或把 SLO 收紧 12.6 倍。[论文](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)。 |
| 周二 | *读* · [Splitwise](https://arxiv.org/abs/2311.18677)（Microsoft，ISCA 2024）：prefill 用 H100、decode 用降功耗 A100，吞吐 1.4 倍、成本低 20%。然后读 [vLLM 解耦式 prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/)。 |
| 周三 | *读* · [Mooncake](https://arxiv.org/abs/2407.00079)（FAST 2025 最佳论文），以 KV 缓存为中心，已在 Kimi 生产使用。然后读 [SGLang PD 解耦](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)。比较每个引擎怎么拆分这两个阶段。 |
| 周四 | *做* · 用 [vLLM production stack](https://github.com/vllm-project/production-stack) 的 Helm chart 部署，在托管集群或 `kind` 上。 |
| 周五 | *做* · 用 `num_requests_waiting` 这个自定义指标做扩缩容，[而不是 CPU](https://developers.redhat.com/articles/2025/09/23/how-set-kserve-autoscaling-vllm-keda)。在 GPU 服务上按 CPU 扩缩容，正是这节课要防止的错误。 |
| 缓冲 | 把生态图谱补上：[PyTorch 解耦式推理](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/)、[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)、[llm-d](https://llm-d.ai)。 |

---

### 第 8 周（续）：大规模集群 PD 优化

**重点。** 第 8 周把 prefill 和 decode 拆开了，这一续周回答拆开之后在真实集群上要解决的问题：KV 怎么跨节点走、池子怎么共享、两池配比怎么定、节点挂了怎么办。

**形式。** 以文字为主。论文一律是选读：每篇论文的核心思想、核心逻辑、关键机制和关键数字都已经写进下面正文，不读原文也能掌握深度；原文是想再深挖时的材料。数字是论文宣称的，没有独立复现。

**产出。** 一张 Mooncake 架构图、一张“prompt 长度 × 带宽 → 传输时间占 TTFT 比例”的表、一张“信号→动作→验证方法”的扩缩容与容错策略表，以及一份注明实际跑了多大集群的解耦 vs 同机部署对比。

#### 本周论文精读（正文；论文原文选读）

**Mooncake：以 KVCache 为中心的解耦架构。** 出处：[Mooncake 论文（选读）](https://arxiv.org/abs/2407.00079)，FAST 2025 最佳论文；Kimi 的服务平台。开源实现见 [Mooncake 仓库](https://github.com/kvcache-ai/Mooncake)，文档见 [Mooncake 文档](https://kvcache-ai.github.io/Mooncake/)。

- **核心思想。** 把 KVCache 当成系统调度的中心对象，而不是把请求或模型当中心。prefill 和 decode 拆到不同的集群；GPU 服务器里闲置的 CPU、DRAM、SSD 拼成一个集群范围的分布式 KV 缓存池（Mooncake Store）；调度器围着 KV 的位置和取回成本做决策。
- **核心逻辑链。** 长上下文下 KV 巨大，prefill 要算力、decode 要显存带宽，两者诉求相反，合在一个实例里互相拖累，所以拆开；拆开的代价是 KV 必须跨节点搬，搬的成本决定了解耦划不划算，所以要有高速传输引擎和可复用的共享池；池子让 KV 变成可复用的资产，前缀命中就不用重算，prefill 只算增量部分；调度器按 SLO 做准入和放置，接不下的请求提前拒绝，而不是放进来排队拖垮所有人。
- **关键机制。** 全局调度器 Conductor：对每个到达的请求，估算它在每个候选 prefill 节点上的 TTFT = 排队时间 + prefill 计算时间 + KV 传输时间（prefill 时间用离线测量拟合的多项式回归预测，输入是请求长度、前缀命中长度、传输时间），放到总和最小的节点；没有节点能满足 SLO 就直接拒绝，同样估算 TBT，decode 满足不了也提前拒绝。Mooncake Store：分布式 KV 缓存，KV 块按前缀感知的哈希寻址（论文评估里每块 256 个 token），热块在多节点复制，冷块按 LRU 淘汰；热前缀会自动迁移——最匹配的缓存如果在过载节点上，被选中的节点会拉一份副本，所以系统提示词这类共享前缀最后几乎每个 prefill 实例都有一份。请求流：Conductor 选 prefill 和 decode 实例 → 可复用的缓存块流进 prefill 节点 → 增量 prefill 逐层算，新 KV 边算边流向 decode 节点 → 请求加入 decode 侧的连续批处理；长上下文时 prefill 用分块流水线并行跨节点，只在块边界传 activation。Transfer Engine：统一的批量数据搬运接口，多传输协议、拓扑感知路由、多网卡带宽聚合、自动 failover。
- **开放问题（论文自己承认的）。** 预测过载导致过度节流，接着负载不足又导致过度准入，两阶段之间会来回振荡；一个缓解思路是系统级预测——假设全系统统一的 decode 时间，用下一时隙预测的 decode 负载指导当前 prefill 的节流。
- **关键结论与数字（论文宣称）。** 模拟场景下满足 SLO 的同时吞吐最高提升 525%；真实负载下 Kimi 多处理 75% 的请求；长上下文场景收益最大。注意这些数字的适用边界：可复现的实验部分用的是替代模型加回放 trace（论文评估用 16 个 8 卡节点），收益随前缀命中率变化（公开 trace 里 40–66%），“有效容量”只统计系统选择接纳的请求——读原文可以核对实验设计。
- **为什么重要。** 它是 KVCache 中心学派的开山之作：把 KV 缓存从“引擎内部的优化”变成“集群一级的共享基础设施”。后面的 vLLM MooncakeConnector、SGLang HiCache、llm-d 的 Mooncake PD 部署指南都是这个思路的落地。

**Splitwise：按阶段拆硬件。** 出处：[Splitwise 论文（选读）](https://arxiv.org/abs/2311.18677)，Microsoft，ISCA 2024。

- **核心思想。** 一次 LLM 推理有两个阶段，prompt 计算是计算密集型，token 生成是访存密集型，两者的延迟、吞吐、显存、功耗特征完全不同；token 生成阶段用不满最新 GPU 的算力，用便宜、低功耗的卡就能跑。所以把两个阶段拆到不同的机器上，各自用适合的硬件、各自独立配资源。
- **核心逻辑链。** 表征先行——论文先做了大量实测，证明 decode 阶段不需要最新 GPU 的算力；拆开后 prefill 用高算力卡，decode 用降功耗的便宜卡；拆开的代价是状态（KV）要跨机器传，用 GPU 集群里现成的高速背板互联来传；最后按三个目标设计集群：吞吐、成本、功耗。
- **关键结论与数字（论文宣称）。** 吞吐 1.4 倍、成本低 20%；或者相同成本和功耗下吞吐 2.35 倍。
- **为什么重要。** 最早把“按阶段拆硬件”做成完整集群设计的工作，第 8 周解耦思路的源头之一；它证明了解耦的收益首先来自硬件错配的纠正，不只是调度。

| 天 | 课程 |
|---|---|
| 周一 | *读* · 按上面的 Mooncake 总结精读：核心思想、Conductor 的 TTFT 估算式、早期拒绝。画一张架构图：prefill 池、decode 池、KV 池（Mooncake Store）、Conductor、传输引擎，标出请求和 KV 各走哪条路。[Mooncake 论文（选读）](https://arxiv.org/abs/2407.00079)。 |
| 周二 | *做* · 手算传输账：7B 模型（32 层、8 个 KV 头、head dim 128、BF16）每个 token 的 KV 是 128KB；4096 token 的 prompt 是 512MB；100Gb/s 约 40ms、400Gb/s InfiniBand 约 10ms、10Gb/s TCP 约 400ms。做一张“prompt 长度 × 带宽 → 传输时间占 TTFT 比例”的表，解耦划不划算从这个数开始。Transfer Engine 的机制见 [Mooncake 仓库](https://github.com/kvcache-ai/Mooncake)的 mooncake-transfer-engine。 |
| 周三 | *读* · 跨节点 KV 池与复用：Mooncake Store 的分布式缓存、内容寻址、跨实例前缀复用、热块复制与 LRU 淘汰；再对照 [vLLM 解耦式 prefill 文档](https://docs.vllm.ai/en/latest/features/disagg_prefill/)里 9 种 connector 的分工（NixlConnector 走 UCX/GDS，MooncakeConnector 走 Transfer Engine，LMCacheConnectorV1 走 NIXL）。画出 GPU→CPU→远端三级层级，写清每一级命中意味着什么。 |
| 周四 | *做* · 多节点解耦部署：用 [vLLM production-stack](https://github.com/vllm-project/production-stack) 的 Helm chart 装两套独立部署——prefill 一套、decode 一套（两份 values，两个 release；chart 原生是否支持解耦部署以当前版本为准，两套独立 release 是不依赖 chart 新特性的做法），按 vLLM 解耦文档配 `--kv-transfer-config`（NixlConnector 或 MooncakeConnector，照文档示例写）。再用 M0 的 `docs/modules/M0-foundations/scripts/bench_openai.py` 在同一负载下测解耦 vs 同机部署的 TTFT 和吞吐。注意：kind 多节点只能验证拓扑和配置，测不出性能；真实多节点 GPU 集群才能测传输开销，租不到就记“部署验证完成，性能对比待补测”。 |
| 周五 | *读* · 池配比、扩缩容与容错：先用第 5 周的负载特征判断 prefill-heavy 还是 decode-heavy，再定两池的卡数；扩缩容信号用 `num_requests_waiting`、TTFT 和队列深度；容错对照 [SGLang PD 解耦文档](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)的心跳与超时（heartbeat 5 秒、连续 2 次失败摘掉 prefill 节点、KV 等待超时 300 秒），节点挂了 KV 丢了要重算，路由器负责摘掉故障节点；参考 [llm-d](https://llm-d.ai) 的智能路由与预测延迟调度。产出：一张“信号→动作→验证方法”的策略表。 |
| 缓冲 | 对照 [Mooncake 文档](https://kvcache-ai.github.io/Mooncake/)和 vLLM 解耦文档把周四的配置再走一遍；或在 [llm-d 文档](https://llm-d.ai)里找 Mooncake PD 部署指南（llm-d 2026 年 8 月新增）。 |

---

### 第 9 周：单位经济学与路由器

**重点。** 把成本的阅读变成代码。路线图从这里开始不只是关于延迟，也关于利润。

**形式。** 以文字为主。

**产出。** 一个按成本、延迟、质量选后端的路由器，加上按请求计 token 预算、把成本打到看板上的中间件。

| 天 | 课程 |
|---|---|
| 周一 | *读* · [LLM Inference Economics from First Principles](https://www.tensoreconomics.com/p/llm-inference-economics-from-first)。GPU 小时到 token 的换算，以及为什么利用率而不是每小时单价决定成本。 |
| 周二 | *做* · 用公式（每小时价格 ÷（峰值 token/秒 × 3600 × 利用率）× 100 万）结合你第 5 周的吞吐，重新算你自己那块 GPU 每百万 token 的成本。[示例](https://www.digitalocean.com/community/tutorials/llm-inference-cost)。 |
| 周三 | *读* · [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai-2/)。通过权重、激活、KV 缓存的 int8 加上多查询注意力和跨层 KV 共享，服务成本降了 33 倍。每一个杠杆都是你在第 6、7 周学过的。 |
| 周四 | *做* · 搭建路由器。一个瘦服务，按成本/延迟/质量策略在便宜的小模型和贵的的大模型之间选择，并把选择的路径和成本记到 Prometheus。 |
| 周五 | *做* · Token 预算中间件：按请求封顶并记账。把每次请求的成本放到看板上，和延迟并排。 |
| 缓冲 | [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)，把第 1–9 周做一次整合。 |

---

### 第 10 周：发表，然后养成阅读习惯

**重点。** 把产物发出去，并建立让它保持新鲜的输入渠道。

**产出。** 一份发表出来的仓库和写作，版本固定、命令可复现，外加日历上每周一次的研究时间。

| 天 | 课程 |
|---|---|
| 周一 | *做* · 起草写作：硬件、模型、精确的输入输出 token 长度、并发扫描、TTFT 和 ITL 的 p50/p95/p99、吞吐。先拿下面的检查清单过一遍。 |
| 周二 | *做* · 把量化、投机解码、KV 淘汰作为有标签的变体加进去，附可复现的命令和固定版本。发表。 |
| 周三 | *读* · 建立每周的阅读习惯。[arXiv cs.DC](https://arxiv.org/list/cs.DC/recent)、[MLSys](https://mlsys.org)、OSDI、NeurIPS 效率方向。先排三篇。 |
| 周四 | *读* · 第一次习惯性阅读。30 分钟读一篇论文的摘要、方法图和实验，不读全文。 |
| 周五 | *可选* · 端侧采样：[ONNX Runtime](https://onnxruntime.ai)、[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)、[WebLLM](https://webllm.mlc.ai)。注意这些是客户端运行时，不是你所建东西的延伸。 |
| 缓冲 | 把 benchmark 拿去征求反馈。能回答的批评，证明你的压测工具是真的。 |

---

## 跟踪进度

三种方式，按需要配置的多少排序。

**1. 网页。** 打开 `index.html`，或 GitHub Pages 版本，完成一节就打勾。进度存在浏览器的 local storage 里，关掉标签页或重启机器都不会丢。按浏览器、按设备，没有账号、没有服务器。

要把进度带到另一台设备，用**复制同步链接**，它把你的状态打包成 URL 里一个 12 字符的码；或者用**导出**，下载一个 JSON 文件，之后再导入。这两个也是你清掉浏览器数据后的备份。

如果你在隐私浏览里打开页面，或在沙箱 frame 里打开，local storage 可能被禁用。页面会检测并提示你，打勾在当次访问里仍然能用。

**2. GitHub issue。** Fork 本仓库，从 progress 模板新建一个 issue，在里面打勾。GitHub 把任务列表的状态存在它自己的服务器上，和你的账号绑定，所以在任何你登录的设备上都有效，还能留下一份你何时完成每一周的、带日期的评论记录。如果你想让进度比浏览器配置活得久，这是最好的选择。

**3. 直接改 README。** Fork 本仓库，在你自己的副本里勾选清单。你的提交记录就是进度记录。

### 关于用数据库

服务器端存储（SQLite 或别的）在这里是错误的工具，值得解释一下为什么。它需要托管、需要账号（服务器得知道进度是谁的）、会话处理、你现在持有数据的隐私政策，以及持续维护——而这个项目的全部价值就是一份链接清单。SQLite 尤其不适合部署成 web 应用，因为大多数便宜的托管服务的磁盘是临时的，重新部署就把数据库文件丢了。

Local storage 对单个读者能达到同样的效果，而且什么都不需要。它唯一给不了的是跨设备同步，而同步链接和 JSON 导出已经覆盖了这种情况，不需要后端。

如果你确实想要一个托管版本，比如一群人一起学、共用排行榜，现实的方案是 GitHub OAuth 做身份加一个托管的 Postgres 或 Turso 实例，而不是 web 服务器上的一个 SQLite 文件。

## 发表 benchmark 之前

这些是公开的推理 benchmark 最容易被拆台的地方。第 5 周先读，第 10 周再用。

- [ ] 扫描请求速率。只测一个并发级别是轶事，不是 benchmark。
- [ ] 报告精确的输入和输出 token 长度。KV 缓存大小、进而吞吐，完全取决于它们。
- [ ] 报告 p50、p95、p99。平均值会藏住用户实际感受到的长尾。
- [ ] TTFT 和 token 间延迟分开报。把 prefill 和 decode 压成一个数，会毁掉结论。
- [ ] 不要在服务器已经抢占的饱和点上测。报拐点，不报崩溃点。
- [ ] 别用均匀长度的合成流量。均匀长度会藏住让连续批处理有意义的重尾分布。
- [ ] 比较每秒 token 数之前，先处理不同引擎之间的分词器差异。
- [ ] 固定引擎版本，公布精确命令，让批评者可以复现而不是猜测。

## 决策点

这些时刻，盲目跟进度是错的。

| 如果 | 那么 |
|---|---|
| **第 5 周。** 还没到目标并发，TTFT 就飙升、`num_requests_waiting` 爬升。 | 服务器在抢占。先停下来调 `--max-num-seqs`、`--gpu-memory-utilization` 和分块 prefill。在抢占点上测出来的数据不能发表。 |
| **第 6 周。** INT4 换来的吞吐，让质量代理指标掉了超过 1–2%。 | 保留 FP16 或 FP8 作默认。把 INT4 当作缓解显存压力的杠杆，不当作默认的吞吐手段。 |
| **第 7 周。** 在你真实的请求速率下，投机解码反而变慢了。 | 预期之内。收益出现在低查询率下，GPU 被计算打满时就消失了。把交叉点发表出来，它比加速比更有用。 |
| **第 8 周。** Kubernetes 开始整节整节地吃掉你搭集群的时间。 | 用托管集群，或者退回单节点 Docker 加 KEDA 那篇阅读。自动扩缩容的信号比自己搭的集群重要。 |
| **第 8 周（续）。** 解耦后 TTFT 反而变差。 | 先算传输账：KV 传输时间超过了 prefill 排队省下的时间，解耦在你的 prompt 长度和带宽下不划算。先缩短传输路径（同机、更高速的互联）或退回同机部署，再谈扩池子。 |

---

## 参考资料

### 论文

| 论文 | 会议 | 主题 |
|---|---|---|
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | SOSP 2023 | KV 缓存分页 |
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) | OSDI 2022 | 连续批处理 |
| [SGLang](https://arxiv.org/abs/2312.07104) | NeurIPS 2024 | RadixAttention、前缀复用 |
| [SARATHI](https://arxiv.org/abs/2308.16369) / [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | OSDI 2024 | 分块 prefill |
| [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | OSDI 2024 | prefill/decode 解耦 |
| [Splitwise](https://arxiv.org/abs/2311.18677) | ISCA 2024 | 按阶段拆分到不同 GPU |
| [Mooncake](https://arxiv.org/abs/2407.00079) | FAST 2025 | 以 KV 缓存为中心的服务 |
| [StreamingLLM](https://arxiv.org/abs/2309.17453) | ICLR 2024 | 注意力汇、长上下文 |
| [EAGLE](https://arxiv.org/abs/2401.15077) | arXiv | 投机采样 |

GPTQ、AWQ、SmoothQuant、LLM.int8()、Medusa、H2O、SnapKV 和 FlashAttention 系列，请用 [Lilian Weng 的综述](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)和 [NVIDIA 优化综述](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)，它们直接引用了这些工作。搜精确标题，不要猜 arXiv 编号。

### 课程与系列

- [Stanford CS336：Language Modeling from Scratch](https://cs336.stanford.edu/spring2025/)，Percy Liang 与 Tatsunori Hashimoto 主讲。第 5 讲（GPU）、第 6 讲（Kernel）、第 7–8 讲（并行）、第 10 讲（推理）。
- [MIT 6.5940 EfficientML.ai](https://hanlab.mit.edu/courses/2024-fall-65940)，Song Han 主讲。2023 秋和 2024 秋的录像都可以用。
- [vLLM Office Hours](https://www.youtube.com/@vllm-project)，双周，来自引擎团队。
- [GPU MODE](https://www.youtube.com/@GPUMODE)，kernel 与系统级讲座。
- [USENIX](https://www.usenix.org/conferences)，OSDI、NSDI、ATC 上论文作者的报告。

### 工具

| 工具 | 用途 |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | 主要服务引擎 |
| [SGLang](https://docs.sglang.io) | 前缀复用对比 |
| [GuideLLM](https://github.com/vllm-project/guidellm) | SLO 感知的速率扫描 |
| [genai-perf](https://github.com/triton-inference-server/perf_analyzer) | TTFT 与 ITL 测量 |
| [LLMPerf](https://github.com/ray-project/llmperf) | 托管端点的 benchmark |
| [production-stack](https://github.com/vllm-project/production-stack) | Kubernetes 部署 |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) | 分布式 KV 缓存与传输引擎 |
| [llm-d](https://llm-d.ai) | Kubernetes 上的分布式推理编排与智能路由 |

### 阅读习惯

每周 30 分钟：扫 arXiv cs.DC 和系统会议的标题，只读一篇论文的摘要、方法图和实验，并在 [vLLM](https://blog.vllm.ai) 和 [LMSYS](https://lmsys.org/blog/) 博客以及 [Modal Almanac](https://modal.com/llm-almanac/summary) 上跟踪应用型写作。读系统论文，跳过模型发布的新闻稿。

---

## 关于来源的说明

这里的每个视频都有具名的讲者——来自大学、引擎团队或会议——并且配有覆盖同样内容的文字材料，所以你可以读而不是看。没有视频达到这个标准的地方，这一周就是纯文字，而不是硬凑。

还有几点要注意：

- 时间戳是目标值。讲座会被重新上传、重新剪辑，用视频里的章节标记来定位分割点。
- MIT 6.5940 在 2025 秋没有开课。2023 秋和 2024 秋的录像都存在，任选其一。
- vLLM 的博客有两种 URL 形式，文档随版本变化。如果链接 404，搜精确标题，不要改路径。
- 这里引用的性能数字是论文和博客各自宣称的，没有独立复现。把它当作那个来源愿意为之辩护的数字，而不是你硬件上的事实。
- 每百万 token 的成本和 GPU 每小时价格变化很快。经济学那部分阅读的价值在于方法。每个数字都要按你自己的硬件重新算。

## 参与贡献

欢迎纠错和补充，尤其欢迎：

- 失效或搬家的链接
- 取代了这里某项内容的新的一手资料
- 达到标准的视频（具名讲者、可信机构、配有文字），给目前标记为纯文字的周
- 如果你走完了路线图，你自己的 benchmark 写作

开 issue 或 PR。

## 许可与出处

本仓库基于 [patchy631/time-to-first-token](https://github.com/patchy631/time-to-first-token) fork 并改进，沿用上游的 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 协议。英文原文见 [docs/original-README.md](docs/original-README.md)。

可以自由使用、改编、教学。
