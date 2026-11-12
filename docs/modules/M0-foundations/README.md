# M0 · 基础与度量

M1 的 10 周路线图默认你已经会“看”系统：知道什么受显存带宽限制、什么受计算限制，知道 TTFT、token 间延迟、吞吐各自在量什么，并有一套能复现的压测方法。本模块就是把这些前置能力补上，**一周，5 节，每节 30 分钟**，做完之后再进 M1。

> 硬件门槛：本模块**不需要 GPU**。第 1–3 节是纸面推导与概念，第 4 节写压测脚本（纯 Python），第 5 节跑基线时才需要一台跑着推理服务的机器——可以用 M1 第 2 周搭起来的 vLLM 服务，也可以先用模块自带的 mock 思路验证脚本本身（见第 5 节）。

**产出**：一份手算的 Roofline 图、一张度量指标对照表、一套 profiling 读数方法，以及 `scripts/bench_openai.py` 这套后续所有模块共用的压测脚本和一份基线报告。

## 第 1 节 · Roofline：先算，再优化

**目标**：给定模型和硬件，不跑代码就能判断瓶颈在哪。

**怎么做**（30 分钟）：

1. 读 [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)（Horace He）前半部分：计算、显存带宽、开销三种区间。
2. 记住两个公式。算术强度 = FLOPs / 搬运字节数；运行时间 ≈ max（计算时间，搬运时间）。
3. 手算一个例子：7B 模型，FP16，batch=1 做解码。每生成一个 token，要把全部权重读一遍：约 7×10⁹ 参数 × 2 字节 ≈ 14 GB；计算量约 2 × 7×10⁹ ≈ 14 GFLOP。算术强度 ≈ 1 FLOP/字节。
4. 对照硬件：以 A100 80GB 为例，FP16 稠密算力约 312 TFLOPS，HBM 带宽约 2 TB/s。算时间：14 GFLOP / 312 TFLOPS ≈ 0.045 ms；搬时间：14 GB / 2 TB/s ≈ 7 ms。搬运时间是计算的 150 多倍——**解码是显存带宽受限**，理论上限约 140 token/s。硬件参数以厂商公开规格为准，不同型号数字不同，结论不变。
5. 再算 prefill：输入越长，注意力部分的计算量随长度增长，算术强度随之上升，逐步从带宽受限滑向计算受限。这就是为什么 prefill 和 decode 要分开看，也是 M1 第 8 周解耦式部署的出发点。

**产出**：一张手画的 Roofline 图，标出你目标模型的 prefill 点与 decode 点，并写一句话：哪个阶段受什么限制。

**延伸**：[Stanford CS336 第 5 讲](https://www.youtube.com/watch?v=6OBtO9niT00)（GPU 与显存层级）、文字版 [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)。

## 第 2 节 · 推理度量：四个数，一个分布

**目标**：把“快”拆成四个不会互相掩盖的数，并知道分布比均值重要。

**概念**（先读，再记到表里）：

| 指标 | 英文 | 定义 | 它回答的问题 |
|---|---|---|---|
| 首 token 延迟 | TTFT | 从发请求到收到第一个 token 的时间 | 用户要等多久才看到东西 |
| token 间延迟 | ITL / TPOT | 相邻两个 token 之间的平均时间 | 输出流畅不流畅 |
| 吞吐 | throughput | 单位时间生成的 token 总数 | 系统整体产出 |
| 排队深度 | queue depth | 正在等待的请求数 | 系统是不是已经满了 |

**为什么只报均值是错的**：用户感受到的是长尾。报 p50、p95、p99——p95 是“95% 的请求比它快”，p99 是尾巴。两个系统均值相同、p99 差一倍，用户体验完全不同。TTFT 和 ITL 必须分开报，压成一个“延迟”就把 prefill 和 decode 的结论毁了。

**固定长度原则**：输入和输出 token 长度固定下来再测。KV 缓存大小完全由它们决定，长度一变，吞吐跟着变，数字就没法比较。`bench_openai.py` 默认固定 prompt、固定 `--max-tokens`，就是这个原因。

**产出**：填好上面这张表，并写下你自己的基线要报哪几个数（建议：TTFT p50/p95/p99、ITL p50、吞吐、成功请求数）。

**延伸**：[NVIDIA：LLM 推理 benchmark 基础概念](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)、[Modal：How to Benchmark LLM Engines](https://modal.com/llm-almanac/how-to-benchmark)。

## 第 3 节 · Profiling：看时间花在哪

**目标**：拿到一个推理请求的 timeline，能指出来瓶颈在哪个 kernel、哪次访存、哪段等待上。

**工具对照**：

| 工具 | 看什么 | 什么时候用 |
|---|---|---|
| PyTorch Profiler（`torch.profiler`） | 每个算子的耗时、显存、CPU 与 GPU 时间线 | 先看这里：它能告诉你时间花在 attention、matmul 还是 launch 开销上 |
| `nsys`（Nsight Systems） | 系统级时间线：CPU、GPU、拷贝、核函数启动 | 跨进程、跨库的问题，比如服务框架的调度开销 |
| `ncu`（Nsight Compute） | 单个 kernel 的细节：占用率、访存效率、指令 mix | 已经定位到某个 kernel，想知道它为什么慢 |

**动手**（30 分钟，无 GPU 也可读文档做）：

1. 用 `torch.profiler` 跑一次小模型的前向（CPU 也行），导出 Chrome trace（`chrome://tracing` 可打开），找到最耗时的三个算子。
2. 对照第 1 节的 Roofline：最耗时的算子是算术强度高还是低？如果一个低算术强度的算子占了大头，优化方向是减少搬运（融合、量化），不是加算力。
3. 记下两条坑：`ncu` 需要 CUPTI 权限，在很多云主机上要管理员权限或特定配置，跑不起来先查权限而不是怀疑工具；profiling 本身有开销，采样结果用来定位问题，不用来报最终性能数字。

**产出**：一份一页纸的 profiling 笔记——工具、命令、你看到的最耗时算子、以及你判断的瓶颈类型。

**延伸**：PyTorch 官方 Profiler 文档、[GPU MODE 讲座](https://github.com/gpu-mode/lectures)中 profiling 相关章节。

## 第 4 节 · 写压测脚本：`bench_openai.py`

**目标**：有一套自己的、可复现的压测工具，后续所有模块都用它，数字之间才能比较。

**怎么做**：读 [scripts/bench_openai.py](scripts/bench_openai.py)，它是本模块的产物，只用 Python 标准库，不需要 GPU 也可以跑通逻辑：

- 打 OpenAI 兼容接口的 `/v1/chat/completions`（流式），测 TTFT、总延迟、ITL 和 token 数。
- token 数优先读流式最后一个 chunk 的 `usage.completion_tokens`（vLLM 支持 `stream_options.include_usage`）；服务器不支持时按字符数估算，并在结果里标明是估计值。
- 固定 prompt 与 `--max-tokens`，按 `--concurrency` 列表逐级扫并发，每级固定 `--requests` 个请求，输出 JSON 与 Markdown 表格。

**验证脚本本身**（无 GPU 可做）：用一个本地 mock SSE 服务跑通一次——本仓库的作者在提交前就是这么验证的：mock 服务每 10ms 吐一个 chunk，脚本测出的 ITL 应为约 0.011s，TTFT 接近 0，并发 4 时的吞吐应约为并发 1 的数倍（mock 服务无真实瓶颈，仅验证测量逻辑）。

**产出**：在你自己的机器上跑通 `python3 scripts/bench_openai.py --help`，并理解输出表格里每一列的含义。

## 第 5 节 · 跑出第一份基线

**目标**：给后面的所有优化一个“优化前”的数字。没有基线，任何加速比都是空话。

**怎么做**：

1. 起一个推理服务：M1 第 2 周的 vLLM 服务，或任何 OpenAI 兼容的服务（SGLang、llama.cpp 的 server 模式都可以）。
2. 跑三级并发扫描，例如：`python3 scripts/bench_openai.py --base-url http://localhost:8000 --model <你的模型名> --concurrency 1 8 32 --requests 20 --output bench-baseline.json`。
3. 把结果填进 [baseline-report-template.md](baseline-report-template.md)：硬件、模型、引擎版本、精确的输入输出长度、TTFT p50/p95/p99、ITL、吞吐、以及——最重要的一点——**拐点**：吞吐不再涨、延迟开始恶化的那个并发数。饱和点的数据要丢掉，不要报。
4. 如果这周还没有 GPU：先用 mock 服务验证脚本（第 4 节的方法），基线报告的硬件栏写“待 M1 第 2 周服务就绪后补测”，不要编数字。

**产出**：一份填好的基线报告。它是 M1 第 5 周压测、第 6 周量化对比、第 7 周投机解码对比的共同参照物。

---

## 本模块检查清单

- [ ] 手算过目标模型的 decode 算术强度，能说清为什么解码受带宽限制
- [ ] 能区分 TTFT、ITL、吞吐、排队深度，并知道为什么报 p95/p99
- [ ] 用 profiler 找到过一次最耗时的算子，并判断了瓶颈类型
- [ ] `bench_openai.py` 在你机器上跑通过，能解释输出的每一列
- [ ] 有一份基线报告（或明确标注了待补测）
