# 第 1 部分 · 基础与度量

第二部分假定读者已经会“看”系统：知道什么受显存带宽限制、什么受计算限制，知道 TTFT、token 间延迟、吞吐各自在量什么，并有一套能复现的压测方法。本部分把这五项前置能力补齐，共 5 章。

概念在正文里讲透，推导一步不跳，数字例子全程手算。读完一章，你应当能不看笔记、拿一张纸把这一章的核心算式重新推一遍。第二部分里每一次“先定位瓶颈再动手”的判断，用的都是这里的工具：Roofline 判断瓶颈在算还是在搬运，四个度量说明“快”快在哪，profiling 定位时间花在哪个算子上。第 4 章的压测脚本与第 5 章的基线报告，把这三样东西变成后面每一部分都能复用的测量习惯。

> 起步：先按[快速上手](../../quickstart.md)跑完整标准库 mock，或按第 8 章第 2 节起真实服务，再回到本部分解释指标。纸面推导与 mock 客户端无需 GPU；真实模型基线必须有已运行的推理服务，GPU/NPU 的性能不能由 mock 代替。

**本部分产出**：一份手算的 Roofline 图、一张度量指标对照表、一套 profiling 读数方法，以及 `scripts/bench_openai.py` 与一份基线报告。

## 第 1 章 · Roofline：先算，再优化

**本章目标。** 给定模型和硬件，不运行推理服务也能判断瓶颈更可能在算力、显存带宽还是启动开销。

### 问题：换更快的卡，为什么有时没用

先看一个具体问题：一个 7B 模型、FP16 权重、batch=1 的解码请求，生成下一个 token 时，GPU 主要在算什么？瓶颈在显存带宽，不在算力：矩阵乘还在做，但每个权重只参与一次乘加，时间主要花在把权重从显存读进来。这个判断不用跑服务，用参数量、精度和硬件的峰值算力、带宽就能算出来。

每生成一个 token，模型都要把约 70 亿个权重参数从显存读进计算单元，每个参数只参与一次乘加：计算量很小，搬运量很大。换一块算力翻倍、显存带宽不变的卡，计算时间减半，总时间却几乎不变。

Roofline 模型就是把这类判断变成算术。它用两个量描述一个 kernel：

- **算术强度（arithmetic intensity）**：每搬运 1 字节数据，做了多少次浮点运算，单位是 FLOP/字节。
- **脊点（ridge point）**：硬件峰值算力除以峰值显存带宽，单位也是 FLOP/字节。

把 kernel 的算术强度和硬件的脊点相比：低于脊点，时间主要花在搬运上，叫**显存带宽受限（memory-bound）**；高于脊点，时间主要花在计算上，叫**计算受限（compute-bound）**。还有第三类边界：kernel 太小，计算和搬运都只要微秒级，真正占时间的是启动和同步，叫**开销受限（overhead-bound）**。开销不在 Roofline 的两根线上，需要单独判断。

两个手算公式：

```text
算术强度 = FLOPs / 搬运字节数
运行时间 ≈ max(计算时间, 搬运时间)   # 大 kernel 的近似
```

这两个公式给的是上限和方向，不是精确预测：真实 kernel 还会有启动、同步、访存合并和重叠执行的开销。

### 例子：7B 模型解码的一次时间账

取 A40 48GB 的公开规格做主算例：FP16 稠密算力约 74.8 TFLOPS，GDDR6 带宽约 696 GB/s，脊点约 107 FLOP/字节。A100 80GB 作为对照：FP16 稠密约 312 TFLOPS、HBM 约 2 TB/s、脊点 156 FLOP/字节。这是公开规格、用于示例，不是实测；不同卡的数字不同，判断方法不变。

**decode，batch=1。** 每生成一个 token：

- 权重搬运：约 70 亿参数 × 2 字节/参数 ≈ 14 GB。
- 计算量：每个参数一次乘加，一次乘加算 2 FLOP，约 2 × 70 亿 ≈ 14 GFLOP。
- 算术强度：14 GFLOP ÷ 14 GB ≈ 1 FLOP/字节。
- A40 脊点：74.8 TFLOPS ÷ 696 GB/s ≈ 107 FLOP/字节（A100 80GB 对照为 312 TFLOPS ÷ 2 TB/s = 156）。
- 计算时间：14 GFLOP ÷ 74.8 TFLOPS ≈ 0.187 ms。
- 搬运时间：14 GB ÷ 696 GB/s ≈ 20.1 ms。

搬运时间是计算时间的约 107 倍，decode 落在带宽受限一侧。**只算权重搬运、且一步只生成一个 token** 时，单请求吞吐上限约 1 ÷ 0.0201 s ≈ 50 token/s；这个上限不含 KV 读取、调度、kernel 启动与采样，真实服务只会更低。如果测出超过 50 token/s 的单请求吞吐，先检查测量口径（batch 是否真的为 1、是不是一步吐了多个 token、token 数怎么数）。把算力翻倍，0.187 ms 只变成 0.094 ms，总时间仍由 20.1 ms 的搬运决定：算力不是没用，是当前瓶颈不在它那里。

### 用 batch 与 prefill 看 Roofline 上的位置变化

**增大 batch。** batch=8 时，一次 decode 步骤仍只读一遍 14 GB 权重，但计算量变成 8 × 14 = 112 GFLOP：

- 算术强度：112 GFLOP ÷ 14 GB = 8 FLOP/字节。
- 计算时间：112 GFLOP ÷ 74.8 TFLOPS ≈ 1.50 ms。
- 搬运时间：仍约 20.1 ms。

总吞吐约 8 ÷ 0.0201 s ≈ 398 token/s（只算权重，不含 KV），是 batch=1 的 8 倍；但每个请求仍约每 20.1 ms 得到一个 token，单请求的 token 间隔没有缩短。batch 提高的是权重复用次数，不是单个请求能用的带宽。

那 batch 要多大才会翻到计算受限？只算权重时，令 B × 14 GFLOP / 74.8 TFLOPS = 20.1 ms，得 B ≈ 107。batch 约 107 时计算时间追上搬运时间；更大以后，计算才成为主要成本。这个数是模型和硬件共同决定的，不是通用常数。

**prefill。** 它一次处理整段 prompt，权重仍只读一遍，但计算量随输入长度 N 线性增长。矩阵乘部分约为 2 × N × 70 亿 FLOPs：

| 输入长度 N | 权重计算量 | 注意力部分（估算） | 总计算量（估算） | 算术强度（约） | 判断 |
|---:|---:|---:|---:|---:|---|
| 128 | 1.79 TFLOP | 0.004 TFLOP | 1.80 TFLOP | 128.6 FLOP/字节 | 刚过脊点，计算受限（A100 上为带宽受限） |
| 512 | 7.17 TFLOP | 0.069 TFLOP | 7.24 TFLOP | 517 FLOP/字节 | 计算受限 |
| 1024 | 14.34 TFLOP | 0.27 TFLOP | 14.61 TFLOP | 1044 FLOP/字节 | 计算受限 |
| 4096 | 57.34 TFLOP | 4.40 TFLOP | 61.74 TFLOP | 4410 FLOP/字节 | 计算受限 |

表里的“注意力部分”按 2 × N² × 4096 × 32 估算：32 层、hidden 4096，只计 QK^T 一项，PV、softmax、mask 未计，所以它只表示 N² 的量级，不是完整 attention 的实测 FLOPs。总计算量是权重项加这一项；算术强度用总计算量除以 14 GB 权重搬运量，未计激活与 KV 写入。

令权重计算时间追上权重搬运时间：2N × 7e9 / 74.8e12 = 14e9 / 696e9，解得 N ≈ 107.5（A100 80GB 对照为 156）。N 小于约 107 的 prefill 仍受带宽限制，更长的 prompt 转为计算受限；注意力项随 N² 增长，长上下文下会进一步推高计算占比。只算权重时，batch 阈值、prefill 交叉点和脊点都落在约 107：每多一个 token，算术强度就加 1 FLOP/字节，加到脊点就翻到计算侧。

**补上 KV 缓存。** 上面只算了权重，decode 还要读历史 token 的 K、V。以 Mistral-7B 为例：32 层、8 个 KV 头（GQA，grouped-query attention，分组查询注意力：多个 query 头共享一组 K/V 头）、头维 128、FP16，每 token 的 KV 为：

```text
2（K 和 V）× 32 层 × 8 个 KV 头 × 128 维 × 2 字节 = 131072 字节 = 128 KiB
```

上下文 1024 token、batch=1 时，KV 约 128 MiB，和 14 GB 权重相比很小；batch=8、上下文 1024 时，KV 约 1 GiB，decode 一步的搬运量从 14 GB 变成约 15.1 GB，时间约从 20.1 ms 变成 21.7 ms。上下文越长、batch 越大，KV 在搬运账里的占比越高。

如果是 MHA（multi-head attention，多头注意力：每个 query 头各存一份 K/V），同一模型的 KV 会是 GQA 的 4 倍。所以看到一个模型的 KV 数字，先确认注意力结构与头数，不能只拿参数量套。

把 KV 计入搬运量，上面只算权重的上限会下降：batch=8、上下文 1024 时，吞吐上限约 369 token/s，而不是约 398 token/s。本章的约 398 是只算权重的口径；比较不同来源的数字，先对齐这一点。

### 把这套账套到 Qwen3.5-9B

上面的 7B 算例用 Llama-3.1-8B/Mistral-7B 的 GQA 几何讲最简版本；本书默认的运行模型是 Qwen3.5-9B。只加载语言模型时（vLLM 启动加 `--language-model-only`），16 位（BF16）权重为 18.394 GB（17.13 GiB）。下面的数字都是纸面算例和下界，不是实测。

**decode。** 单请求一步仍只读一遍权重：18.394 GB ÷ 696 GB/s ≈ 26.4 ms/token，对应单请求吞吐上限约 38 token/s；计算下界为 18.394 GFLOP ÷ 74.8 TFLOPS ≈ 0.246 ms。算术强度约 18.394 GFLOP ÷ 18.394 GB = 1 FLOP/字节，和 7B 算例一样远在 A40 脊点 107 的左侧，decode 仍是带宽受限。

**混合结构与状态。** Qwen3.5-9B 的 32 层由 8 层 full attention 和 24 层 Gated DeltaNet 组成。full attention 每 token 的 KV 为 2 × 8 层 × 4 个 KV 头 × 256 维 × 2 字节 = 32 KiB；24 层 Gated DeltaNet 另有每请求约 49 MiB 的固定递推状态（48 MiB FP32 状态加约 1.1–1.5 MiB 卷积槽位，合计约 49 MiB），不能由 block table 在请求间共享。把上下文加进去，单请求的状态账是：

| 上下文 | KV（32 KiB/token） | 固定状态 | 合计 |
|---:|---:|---:|---:|
| 500 token | 16 MiB | 49 MiB | 约 65 MiB |
| 8K | 256 MiB | 49 MiB | 约 305 MiB |
| 32K | 1 GiB | 49 MiB | 约 1.05 GiB |

短上下文由固定状态主导：500 token 的 KV 只有 16 MiB，49 MiB 的递推状态反而是大头；到 8K 以后 KV 才逐渐占主导。这部分固定状态随请求数线性增长，估并发时要逐条计入，不能当成可共享的 KV block。

**prefill。** 只算权重项，512 token 约为 2 × 9.2e9 × 512 ≈ 9.42 TFLOP，除以 74.8 TFLOPS 约 126 ms；8K 约 150.7 TFLOP ÷ 74.8 TFLOPS ≈ 2.0 s。两者都远高于约 26.4 ms 的权重搬运下界，prefill 是计算受限；序列越长，注意力项还会继续抬高计算量。

这些数字用来在启动服务前判断量级，不是实测；第 5 章给出 A40 + Qwen3.5-9B 的基线命令，第 8 章给出服务启动配置。实际值以启动日志的 KV cache size 和压测记录为准。

### 边界：Roofline 不负责预测什么

Roofline 的用途是排除方向错误，不是替代测量。

- **它不告诉你实测有多快。** 上限和实测之间的差距，来自访存是否合并、占用率、kernel 启动、同步和调度。第 3 章用 profiling 找差距，第 5 章用压测报数字。
- **它不覆盖小 kernel 的启动开销。** 一个只有几十 KB 的逐元素 kernel，计算和搬运可能都远小于微秒，时间由启动与调度决定。推理服务里大量 RMSNorm、激活、采样小算子属于这一类，第 6 章会展开。
- **容量和带宽是两个问题。** 显存容量决定模型和 KV 装不装得下，显存带宽决定 decode 读权重有多快。一块容量大但带宽低的卡，可能装得下模型却仍然跑得慢。
- **算术强度会随 batch、序列长度和缓存结构变化。** 本章的点是示例；实际服务要按自己的 batch、prompt 长度和 KV 配置重算。

### 与其他章节的关系

本章的账会在后面反复出现：

- 第 6 章把“计算、带宽、启动”三种区间变成更完整的 kernel 心智模型。
- 第 7 章补 MHA、GQA、MLA、MoE 的 KV 与参数账本。
- 第 8 章解释 PagedAttention 怎样减少 KV 浪费。
- 第 12 章量化通过减少权重搬运量影响带宽受限的 decode。
- 第 14 章解耦式服务利用 prefill 偏计算、decode 偏带宽的差异。

读完后做三件事，把结果合成一张标出目标模型 prefill 点与 decode 点的 Roofline 草图和一页手算记录：

1. 先用参数量与精度算只含权重的 decode 算术强度，再用层数、KV 头数、头维和上下文长度把 KV 计入，比较两个结果，并写下各自的假设。
2. 算至少三个 prompt 长度对应的 prefill 算术强度，标出脊点把哪一段分到带宽侧、哪一段分到计算侧。
3. 在草图上标出这些点，并在每个点旁写一句边界条件：这个判断忽略了什么（KV、启动、调度、访存效率），下一步用哪条测量验证。

三步做完，每个点都有算术强度、判断依据和一条边界条件。这张图和这页记录的价值不在数字精确，而在换模型、换卡、换 batch 时，你能按同一套算法重算，并在动手前知道该改哪一侧。

**延伸**：[Stanford CS336 第 5 讲](https://www.youtube.com/watch?v=6OBtO9niT00)（GPU 与显存层级）、[Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)（kipp.ly）、[Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)（Horace He）。

## 第 2 章 · 推理度量：四个数，一个分布

**本章目标。** 把“快”拆成口径不同的 TTFT、ITL、吞吐与排队深度：从请求时间线读时延，结合产出窗口与队列观测读吞吐/排队，并说清均值、分位数与样本量的边界。

设一条流式请求在 t=0 发出，30 ms 后进入调度，预填充与第一个 token 的采样共用 120 ms（回传时间忽略不计），第一个 token 在 150 ms 到达客户端；之后每 25 ms 到达一个 token，共 100 个输出 token。由这组时间戳可以直接读出四个数：首 token 延迟是 150 ms；token 间延迟是 25 ms；总延迟是 150 + 99 × 25 = 2,625 ms；单请求输出吞吐是 100 ÷ 2.625 ≈ 38.1 token/s。这组数是算例，不是实测，但它把“延迟”“吞吐”从形容词变成了可以复算的量。

只报一个“延迟”，上面四段——排队、预填充、解码、网络——会被加成一个数。两个服务都报 100 ms，可能一个卡在排队，一个卡在解码，优化动作完全不同。本章先定义四个数各自的起止边界，再说明为什么必须看分布，最后给出一张能直接用于压测报告的读数表。

### 从一条时间线读出四个数

| 指标 | 英文 | 定义（从哪到哪） | 单位 | 回答的问题 |
|---|---|---|---|---|
| 首 token 延迟 | TTFT | 从客户端发出请求到首个非空内容 chunk 到达 | ms | 用户多久看到输出内容 |
| token 间延迟 | ITL（TPOT 为平均） | 概念上为相邻 token 间隔；本脚本按内容 chunk 间隔近似 | ms | 输出是否流畅、有没有停顿 |
| 吞吐 | throughput | 窗口内输出 token 数除以窗口时长 | token/s（另报 req/s） | 系统单位时间产出多少 |
| 排队深度 | queue depth | 等待调度的请求数 | 个 | 系统离饱和还有多远 |

本书客户端 TTFT（time to first token）从发出请求算到**首个非空 `delta.content` chunk 到达**，包含上传、排队、prefill/采样、回传与解析。首个 SSE JSON 可能只含角色或 usage，不能触发 TTFT；一个内容 chunk 也可能包含多个 token，所以它是首内容时间，不是精确的 GPU 首 token 时间。服务端指标起点与网络范围不同，不能直接混比。输入长度、前缀命中和排队都会改变 TTFT，具体归因需服务端指标或对照实验。

概念上的 ITL（inter-token latency）是逐个 token 间隔，TPOT（time per output token）是输出跨度除以 `N−1`，N>1 时才有定义。本脚本只能观察内容 chunk 到达时间，因此 ITL 是 **chunk 级近似**，不为多 token chunk 虚构逐 token 时间；TPOT 也会受传输缓冲影响。只有一个输出 token 或一个内容 chunk 时，TPOT 写 `null` 而不是 0。报告分别给间隔的 p50/p95/max 与可定义的 TPOT 分布，并写清 token 来源。首内容之前属于 TTFT，末尾 usage/关闭开销属于总延迟，不混入内容生成跨度。

吞吐是窗口内的成功产出除以**完整窗口时长**：`X_out = 成功请求的输出 token 总数 / wall_s`；请求吞吐另报 `成功请求数 / wall_s`。失败的部分输出不进分子，但失败与超时占用的时间仍在分母内，成功/失败/超时数量一起报告。输入和输出 token/s 分开，附长度、token 来源、并发及窗口定义。单请求内容生成速度可用有定义的 `1/TPOT` 粗估，不等于包含 TTFT 的端到端吞吐，更不等于整台服务的吞吐。

排队深度是等待调度的请求数（gauge）；`num_requests_waiting` 与正在执行的 `num_requests_running` 一起看，报告窗口内的 p50/p95/max。waiting 为零不单独证明还有容量：客户端或准入限制也可能让请求没进入队列。

Little 定律在一致、稳定的观测边界下写为 `L = λW`：L 是**时间平均**系统请求数，λ 是完成率，W 是**平均**停留时间，不是分位数。只有客户端确实在稳态维持 C 个请求、且没有思考时间等额外阶段时，才可把 L 换成 C。本脚本只设线程池上限，有限批次包含启动与排空，不能以配置 C 直接反查 W；客户端在途与服务端等待/运行也必须分别测量。

### 均值、分位数与长尾

均值把每个样本加总再除，一个极端值就能把它拉高；分位数描述的是排序后的位置。p50 是中位数；p95 表示 95% 的样本不超过它；p99 表示 99% 的样本不超过它。分位数的算法不止一种：`bench_openai.py` 用线性插值，秩 `r = (n − 1) × q`，0 对应最小值；nearest-rank 用 `ceil(n × q)`。同一个样本集换算法，p99 可能差一个样本，报告里要写用的是哪一种。

看一个均值与 p99 都相同的例子。系统 A 的 100 个请求里 99 个 TTFT 是 100 ms、1 个是 200 ms；系统 B 的 100 个请求全是 101 ms。A 的均值是 (99 × 100 + 200) ÷ 100 = 101 ms，B 也是 101 ms。按线性插值，A 的 p99 秩是 99 × 0.99 = 98.01，落在第 99 与第 100 个样本之间，插值结果是 101 ms；B 的 p99 也是 101 ms。两个系统的均值与 p99 完全一样，但 A 的最慢一次是 200 ms，B 是 101 ms。只报均值或只报 p99，都会把 A 的那 1% 长尾藏掉；要看见它，至少再加一个最大值，或报“最慢 1% 的平均”。

任何非空样本都能算经验 p99，但小样本不能可靠估计总体的 1% 尾部。20 个请求的 p95 秩为 18.05、p99 为 18.81，都落在最后两个样本之间，适合作冒烟描述，不适合作容量承诺。若真实尾部概率是 1%，100 个独立请求期望只有 1 个尾部样本，而且一个都没遇到的概率仍约 `0.99^100 ≈ 36.6%`；1000 个的期望是 10 个，也不是“达到数量就稳定”的保证。正式扫描记录样本量并重复，关注区间与异常请求；本书 20 请求只用于 C=1/4 的 mock 冒烟。

ITL 也要看分布。设 100 个间隔里 99 个是 25 ms、1 个是 300 ms，平均 TPOT 是 (99 × 25 + 300) ÷ 100 = 27.75 ms，看起来只比 25 ms 慢一点；但那个 300 ms 的间隔会让用户看到一次明显停顿。只报 TPOT 会把这个停顿平均掉，ITL 至少要报 p50、p95 与最大间隔。吞吐也一样：同一配置重复跑三次，报中位数与范围，不要只报最好的一次。

### 测量条件：长度、并发、样本量

固定输入与输出长度是第一条。输入长度决定 prefill 的计算量与 KV 缓存占用，输出长度决定 decode 步数、总时长与 KV 增长；两者一起决定 batch 能开多大。长度一变，TTFT、ITL、吞吐同时变，两次测量之差就分不清是优化还是负载变化。`bench_openai.py` 固定 prompt 与 `--max-tokens`，但 `--max-tokens` 只是上限：EOS（end of sequence，序列结束符）可能提前结束，实际输出长度要从 `usage.completion_tokens` 读；报告里写实际长度的分布，不只写上限。

并发要写清是客户端并发还是服务端并发。客户端开 32 路，服务端可能只同时跑 8 路，其余在队列里；闭环压测（每个客户端收到响应才发下一个）与开环压测（固定到达率）给出的排队与延迟也不同。报告里写并发数、到达方式、每级请求数、超时设置。

预热与采样窗口也要固定。服务刚启动的前几个请求会触发编译、autotune、显存分配与缓存预热，耗时高于稳定态；先跑预热请求，再开始计时。测量窗口内不要改模型、并发或长度；窗口太短会把启动抖动算进去，太长则可能跨过版本或负载变化。

样本量应同时满足负载设计和尾部统计要求：本脚本强制 N≥每档 C，正式示例使用 N=1024，而默认 N=128 仅避免参数自相矛盾。建议相同条件至少重复 3 次，报中位数与范围；不要把任何一个请求数门槛当成 p99 已可靠的证明。成功延迟与失败/超时数量分开，失败判定耗时另列，避免只看幸存请求的分位数。

| 记录项 | 为什么必须写 | 示例 |
|---|---|---|
| 输入 / 输出长度 | 决定 prefill、decode 与 KV 缓存 | 输入 512，输出 128（实际 completion_tokens 的 p50/p95） |
| 并发与到达方式 | 决定排队 | 有限请求上限 C=32、N=1024；观测实际在途，非恒定并发 |
| 模型、引擎、版本 | 决定基线 | Qwen3.5-9B，vLLM 版本号 |
| 硬件 | 决定上限 | A40 48GB |
| 预热与窗口 | 排除冷启动 | 预热 10 个请求，计时 60 s |
| 成功 / 失败 / 超时 | 决定分母 | 100 / 0 / 0 |
| 分位数算法 | 决定 p99 | 线性插值 (n − 1) × q |

### 四个数一起读：一条并发扫描

把四个数放进同一张表，才看得出系统在哪一档从“还有余量”翻到“开始排队”。下表是一组算例（A40、Qwen3.5-9B、输入 512、输出 128、每级 100 个请求），不是实测：

| 并发 | 吞吐（token/s） | TTFT p50 / p95（ms） | ITL p50 / p95（ms） | 排队深度 p95 | 成功 / 总数 |
|---:|---:|---:|---:|---:|---:|
| 1 | 35 | 180 / 210 | 28 / 35 | 0 | 100 / 100 |
| 8 | 240 | 220 / 320 | 33 / 60 | 2 | 100 / 100 |
| 32 | 620 | 450 / 900 | 48 / 120 | 15 | 100 / 100 |
| 64 | 700 | 1,100 / 2,400 | 85 / 300 | 60 | 99 / 100 |

读表先看吞吐增幅：1→8 约 6.9 倍，8→32 约 2.6 倍，32→64 只有 1.13 倍；后一段的 TTFT/ITL 与排队一起恶化，所以在 32–64 档附近加密验证。若 SLO 是 TTFT p95≤1 s、ITL p95≤150 ms、成功率100%，示意表中 32 档满足而 64 档不满足；这是特定负载下的候选配置，不是“系统普遍支持32个并发用户”的实测容量结论。

这张表用于演示曲线形状，不能据配置并发反推出稳态平均延迟。Little 定律检查需要独立观测的时间平均请求数、完成率与平均停留时间；p50 不能替代均值，有限批次的 C 也不能替代平均在途。第 4 章保留实际客户端在途轨迹，第 5 章说明如何在报告中区分有限扫描和独立采得的稳态窗口。

这张表也说明“最大吞吐”不是“容量”。容量是满足 SLO 的最大负载；64 并发的吞吐更高，但 TTFT p95 已经超出预算，用户体验不达标。容量要同时看四个数，缺一个都可能把系统推到拐点之后。

### 报告里最容易犯的六个错

1. 只报均值或只报一个 p99，不报最大值与样本量。
2. 把 20 个请求的经验 p99 当成可靠的总体 1% 尾部估计。
3. 吞吐不写输入 / 输出长度、并发与成功数。
4. 客户端与服务端延迟混着比；TTFT 含不含网络没说清。
5. 只报最好一次，或把预热算进稳态。
6. 把 TPOT 当成每个间隔，忽略 ITL 的最大停顿。

本章的交付物是一张四数对照表与一份报告清单：TTFT（p50/p95/p99 + 样本量）、ITL（p50/p95/max）、吞吐（token/s 与 req/s，附长度、并发、成功数）、排队深度（waiting/running 的 p50/p95/max），再加输入输出长度、预热与窗口、分位数算法。第 3 章用 profile 解释这些数花在哪，第 4 章把测量固化成脚本，第 5 章跑出第一份基线。

**延伸**：[NVIDIA：LLM 推理 benchmark 基础概念](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)、[Modal：How to Benchmark LLM Engines](https://modal.com/llm-almanac/how-to-benchmark)。

## 第 3 章 · Profiling：看时间花在哪

**本章目标。** 拿到一次推理请求的时间线，能指出时间花在哪个 kernel、哪类访存或哪段等待上，并据此选出下一步动作。

### 三个工具：先看整体，再钻 kernel

上限和外部观测之间还缺一环：一次请求内部的时间去向。Roofline 说 decode 受显存带宽限制，四个度量说 TTFT 是几百毫秒，但都不会告诉你这几百毫秒里哪个算子占了大头、GPU 有多少时间在空等。Profiling（性能分析）补这一环：把一次推理展开成时间线（timeline，按时间排列的调用记录，横轴是时间，纵轴是 CPU 与 GPU 两条泳道），每一段对应一个算子、一次拷贝或一段等待。瓶颈从一句“受带宽限制”变成一个具体的名字：是 attention 里的 softmax，是某个 matmul，还是 GPU 在等 CPU 发下一个 kernel。

三个工具的分工按粒度分层，不重叠：

| 工具 | 看什么 | 什么时候用 |
|---|---|---|
| PyTorch Profiler（`torch.profiler`） | 每个算子的耗时、显存、CPU 与 GPU 时间线 | 第一步。先拿算子级占比：top-3 占得够多才继续往下钻；占比低，说明时间可能不在算子里 |
| `nsys`（Nsight Systems） | 系统级时间线：CPU、GPU、拷贝、kernel 启动，跨进程与库 | 第二步。算子表解释不了时间去向时用它：框架调度、跨进程、启动间隔 |
| `ncu`（Nsight Compute） | 单个 kernel 的细节：占用率、访存效率、指令构成 | 第三步。已经有一个具体 kernel，且它占的时间足够大，才需要问“它为什么慢” |

顺序不能反，理由是一笔账。举例：一个算子占总时间的 5%，把它加速一倍，总时间只从 100% 降到 97.5%，端到端时间少 2.5%；如果 top-3 加起来只有 15%，剩下 85% 分散在大量小算子、启动间隔与空白里，只盯 top-3 找不到答案。先把占大头的那一段找出来，再决定用多细的工具；一上来就钻 5% 的 kernel，花的是最细的工具的时间，省的是最小的一块。这个顺序是一般情况，不是教条：如果症状已经明显在框架层，比如 GPU 空白与请求到达节奏对齐、延迟里有一大段既不在算子里也不在拷贝里，可以直接从 `nsys` 开始。工具由问题决定，不由顺序决定。

`ncu` 的细节读法在第 19 章展开：先读计算单元利用率与显存带宽利用率两个数，判断 kernel 是算受限、访存受限还是延迟受限，再按判断改 kernel。本章的要求到“会读、会定位”为止：`ncu` 回答“这个 kernel 为什么慢”，不回答“哪个 kernel 慢”——后者是 `torch.profiler` 的活。

最小的一次采集长这样：

```python
import torch
from torch.profiler import ProfilerActivity, profile

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

model.eval()
# 先空跑几轮，等每步时间稳定
with torch.no_grad(), profile(activities=activities) as prof:
    model(input_ids)
prof.export_chrome_trace("trace.json")
```

这里用一次前向把工具跑通；换成 `generate` 或引擎的请求入口，读法不变。导出的 `trace.json` 用 `chrome://tracing` 或 Perfetto 打开。有 GPU 时保留 `ProfilerActivity.CUDA`，才能看到 GPU 线；没有 GPU 时只剩 CPU 线，它仍然能告诉你算子调用顺序、dispatch 开销和框架层的空白，但显存带宽、kernel 效率和拷贝都看不到。`record_shapes`、`profile_memory`、`with_stack` 这些开关默认先关：它们改的是采集范围，范围越大，开销越大；定位到具体算子后再按需打开。

### 读 trace：CPU 与 GPU 两条线

trace（一次 profiling 录下来的时间线记录）上有两条线，读法不同。CPU 线的宽度是 host 端的时间：Python 调用、算子分发、kernel 启动、同步。GPU 线的宽度是设备端真正执行的时间：kernel 跑了多久、拷贝跑了多久。同一个算子在两条线上的宽度经常不一样；把 CPU 宽度当成计算时间，是最常见的读错。

看四种形状：

- CPU 长、GPU 短：时间花在启动与调度上，不在计算上。kernel 越小，这个比例越难看。
- GPU 一直是大块、块间没有空白：时间确实花在 kernel 执行上，不在等待上。这时排名前几的 kernel 才需要看。
- GPU 有空白，空白正下方 CPU 在 dispatch 或 Python 调用：GPU 在等 CPU 发下一个 kernel，流水线断了。
- GPU 有空白，空白正下方是拷贝或同步：GPU 在等数据搬完，或在等一次同步的结果。

空白本身不是结论，空白的邻居才是。把两条线上下对齐：空白正下方是 Python 调用，说明 CPU 还没把 kernel 发下来；正下方是 `cudaMemcpy` 或 `synchronize`，说明 GPU 在等数据或等同步；正下方什么都没有，可能是依赖前一个 kernel 的结果，也可能是 profiler 的采样窗口断了。单请求的 trace 里有空白很正常；一段空白是不是浪费，看它是否在每一步重复、是否随 batch 增大而消失——batch 一加就消失的空白是流水线气泡，batch 加了还在的空白，才是结构性的等待。

服务里的 trace 比单请求复杂：连续批处理把多个请求的 kernel 混在同一条 GPU 线上，一段 kernel 不一定属于某一个请求。要给请求归因，先让引擎在请求边界打上 NVTX 标记（NVIDIA Tools Extension，NVIDIA 的时间线标记接口：给一段代码起名，用 `nsys` 采时按名字找），或者把 batch 固定为 1 采一次；没有标记时，只能看整体时间去向，不要硬把某段 kernel 算给某个请求。

没有 GPU 也能读一半：在 CPU 上跑一次小模型前向，trace 只有 CPU 线，仍能看到算子调用顺序、dispatch 开销和同步点；GPU 线的读法先按上面的四种形状建立，等有卡再用同一段代码打开 CUDA activity 验证。

### 从最热的算子到下一步动作

找到 top-3 只是中间结果。下一步要回答两件事：这个算子为什么慢，以及改它值不值。

先看它属于哪一类。低算术强度（arithmetic intensity，每搬运 1 字节做多少次浮点运算）的算子，每个元素读进来、算一两次、写回去，时间主要花在搬运上；高算术强度的算子，数据在片上被复用多次，时间主要花在计算上。前者优化搬运，后者优化计算，方向相反。

| 时间线上的样子 | 说明 | 下一步动作 |
|---|---|---|
| 低算术强度算子占大头 | 逐元素、归一化、decode 的 attention：读一次、算一两次、写回 | 融合、量化、减少中间结果写回 |
| 高算术强度算子顶住算力 | matmul、长上下文 prefill 的 attention：数据在片上复用多次 | 换 kernel 实现、调整分块大小、用更合适的精度 |
| 小 kernel 密集、GPU 空白多 | CPU 发射时间接近或超过 kernel 执行时间 | 融合小算子、CUDA Graph、减少每步 Python 开销 |
| GPU 空白下方是拷贝或同步 | 主机与设备之间的拷贝、`.item()`、日志、采样 | 异步拷贝、去掉不必要的同步、把数据留在 GPU |

同一个 attention 不能只按名字分类。prefill 一次处理 N 个 token，Q、K、V 都是 N×d，矩阵乘的复用随 N 增长；decode 一次只处理 1 个 token，却要读整段 KV，每个 KV 元素只参与少量乘加，算术强度低，上下文越长，读的 KV 越多，搬运量跟着涨。batch 增大时线性层的权重复用变多，但 attention 的 KV 读取也随 batch 增长；判断仍要回到具体 kernel 的实测。时间线上看到 attention 占大头，先分清它是 prefill 还是 decode 的 attention，再看 batch 与上下文长度，才能判断是搬运问题还是计算问题。

低算术强度这一侧的动作是减少搬运：融合把几个逐元素算子合成一个 kernel，中间结果不写回显存；量化把权重从 FP16 压到 FP8 或 INT4，同样的参数少搬一半到四分之三；FlashAttention 不减少 FLOPs，减少的是 attention 中间矩阵的显存读写（第 21 章推导）。高算术强度这一侧的动作是让计算更接近峰值：换更好的 kernel 实现、调整分块大小与流水线、用硬件支持的更低精度矩阵乘；如果已经贴着 Roofline 的算力上限，换实现的空间很小，要么换算法减少工作量，要么接受它，去改占比更大的等待。

判断顺序可以固定成四步：先看占比，top-3 加起来不到两成，先别钻 kernel，去看空白与启动开销；再看类型，低算术强度、高算术强度、启动受限还是等待；然后用 Roofline 估一个上限，看还有多少空间；最后只改一个变量，用同一负载重测。改完不回到时间线验证，就不知道改动有没有命中瓶颈——优化动作和瓶颈类型对不上，数字不会动。

### 两个常见误读与操作细节

第一个误读：“profiler 显示某个算子占 80%，所以它就是瓶颈。”这句话缺了分母。80% 可以是占 GPU 时间的 80%，也可以是占端到端延迟的 80%。算例：端到端 100 ms，GPU 真正忙 50 ms，某个算子占 GPU 时间的 80%，即 40 ms，它只占端到端的 40%；另外 60 ms 在 GPU 空白、CPU 调度或排队里。换一组条件，GPU 忙 95 ms，同样 80% 就是 76 ms，占端到端 76%。同一个百分比，分母不同，结论差近一倍。读数前先问：这个百分比的分母是 GPU 时间、CPU 时间，还是端到端延迟？

`torch.profiler` 的表里还有一层分母：`Self CUDA`（自己 kernel 的时间）与 `CUDA total`（含子算子的时间）不是一回事。按 total 排序再把几行相加，父算子会把子算子的时间重复计一遍；排名用 Self，追一个阶段的总时间才用 total。

第二个误读：“GPU 时间线上有空白，说明 GPU 在偷懒。”空白只说明这一段时间没有 kernel 在跑，原因可能是等 CPU 发下一个 kernel、等拷贝、等同步，也可能是依赖上一个 kernel 的结果。单请求的 trace 里有空白是常态；判断它是不是浪费，看两件事：空白是否在每一步重复，以及增大 batch 后空白是否消失。batch 一加就消失的空白是流水线气泡，说明之前 GPU 没吃饱；batch 加了还在的空白，才是结构性的等待，要去查同步、拷贝或调度。

四条操作细节：

1. 先热身再采。第一次前向有 JIT 编译、cuDNN autotune、显存分配和时钟爬升，采到的是冷启动。先空跑几轮，等每步时间稳定，再开 profiler。
2. 带 profiler 的数字不报成绩。profiler 要记录每个 kernel 的起止，`ncu` 还要读硬件计数器、重放 kernel，这些动作本身有开销；重放会扰动缓存状态与时钟，`ncu` 的耗时更不能当性能数字。定位用带 profiler 的 trace，报性能关掉 profiler 用同一负载重跑。
3. 采样频率与采集范围分开调。频率是隔多久采一次（CPU 采样、硬件计数器采样），范围是记录哪些活动、哪些指标、是否带 shape 与调用栈。不同工具的叫法不同，先分清旋钮属于哪一类。频率越高，单位时间记录越密；范围越大，每条记录越重。两者都会让开销上升，但调的是两个旋钮：`torch.profiler` 的 activities、record_shapes、with_stack 是范围，schedule 的 active 窗口是采哪一段；`ncu --set full` 是范围，`--launch-count` 是采几个 kernel；`nsys -t` 是范围，CPU 采样是否开启与采样周期是频率那一侧。
4. `ncu` 读的是 GPU 性能计数器，需要驱动层放开的权限。很多容器与云主机默认禁用（驱动参数限制，或容器权限不足），典型报错是 `ERR_NVGPUCTRPERM`。跑不起来先查权限与容器配置，再查命令，最后才怀疑工具。`torch.profiler` 与 `nsys` 通常只需要 CUPTI 的活动记录，不需要性能计数器；CUPTI（CUDA Profiling Tools Interface，CUDA 的性能分析采集接口：向上提供活动记录与计数器）是这些工具共用的底层接口。

一份能用的 profiling 笔记至少有三行：最热的算子及其占 GPU 时间的比例；GPU 时间线上空白的位置与成因；下一步只改一个变量的动作，以及改完用什么指标验证。三行齐了，才算回答了本章的问题：时间花在哪，下一步改什么。

**延伸**：[PyTorch Profiler 文档](https://pytorch.org/docs/stable/profiler.html)、[Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)、[GPU MODE lectures](https://github.com/gpu-mode/lectures)（profiling 相关讲座）。

## 第 4 章 · 写压测脚本：`bench_openai.py`

**本章目标。** 用标准库脚本跑通一次完整 SSE 请求与有限并发扫描；从原始请求记录复算 TTFT、chunk 间隔、TPOT 与吞吐，并辨认失败流和 token 估算的边界。

先按[快速上手](../../quickstart.md)跑 mock，或按第二部分第 8 章第 2 节起一个真实服务，再使用 [`bench_openai.py`](scripts/bench_openai.py)。客户端只发 HTTP、解析流并记时间，不需要 GPU；真实模型计算发生在服务端。实验编号与条件记录从这一步开始，约定见[实验记录](../../experiments.md)。

### 为什么必须是流式

脚本向 `/v1/chat/completions` 发送 `stream=true` 请求。非流式响应只能给客户端总耗时；流式 SSE 才能观察内容何时首次到达。**本书客户端 TTFT 一律从发请求算到第一个非空 `delta.content` 的 chunk 到达**，不是首个 JSON、首个 role 块或首个 usage 块。空字符串不触发计时，包含空格的非空内容仍算内容。

SSE 的 `data:` 行以空行结束一个事件，脚本支持多行数据和注释；成功流必须读到 `[DONE]`，并至少有一个内容 chunk。HTTP 错误、超时、错误事件、非法 JSON/UTF-8、结束标记缺失或事件被截断，都记录为失败；已经收到的内容时间仍保留，部分回答不进入成功吞吐。

chunk 是传输单位，可能含多个 token。代理缓冲、网络和解析会影响到达时间，因此这里测的是**客户端首内容时间**，不是 GPU 首 token 产生的精确时刻。第 10、12、13 章换被测对象时，请保留同一接口与计时定义。

### 四个时间量怎么算

所有请求时间使用 `time.perf_counter()`，单位秒。`first_content_s`、`last_content_s` 与 `content_times_s` 都是相对**本次请求发出**的时间；它们不包含线程池中尚未开始请求的等待。

| 量 | 脚本口径 | 边界 |
|---|---|---|
| TTFT | `ttft_s = first_content_s` | 首个非空内容 chunk；角色、usage 与注释不计 |
| 总延迟 | `total_s` | 发出请求到完整流结束或失败判定；包括 HTTP、解析与末尾协议开销 |
| ITL | 相邻 `content_times_s` 之差，保存在 `itl_samples` | **chunk 级近似**；不把多 token chunk 拆成虚构的逐 token 时间 |
| TPOT | `(last_content_s - first_content_s) / (completion_tokens - 1)` | 仅成功请求、token 数 > 1 且至少两个内容 chunk 时有定义 |

只有一个输出 token 时没有 token 间隔；只有一个内容 chunk 时也无法观察内容生成跨度。两种情况的 `tpot_s` 均为 `null`，原因分别写入 `tpot_unavailable_reason`，汇总不把它们当 0。服务端的 usage 能改善 token 数分母，却不能消除 chunk 合并和代理缓冲对时间的影响。

算例：首、末内容在 0.8 s 与 12.8 s 到达，实际输出 128 token，则客户端 TPOT 约 `(12.8−0.8)/127 ≈ 0.0945 s`。若末内容之后还有 usage 或连接关闭开销，`total_s` 会更大，不能用它替换末内容时间。ITL 分布与每请求 TPOT 分开报：前者量停顿，后者量平均生成速度。内部 prefill、decode、排队与网络的归因仍需第 3 章的 profiler 与服务端指标。

### token 数与固定长度

`completion_tokens` 优先使用 `usage.completion_tokens`，来源写为 `usage.completion_tokens`。usage 缺失时，脚本对拼接内容使用 `max(1, len(text)//4)`，来源写为 `estimated_characters_div4`，同时设置 `tokens_estimated=true`。这是英文字符经验值，不是分词；中文、代码和多语言文本的误差可能很大。每个请求都保留来源，本级成功请求只要有一个估算值，汇总就显著标记，并在终端警告。估算会影响 TPOT 分母与吞吐分子，不改变已经观察到的 chunk 到达时间。

请求始终携带 `stream_options.include_usage=true`。服务端若拒绝此字段，按 HTTP 失败记录，不静默更改请求重试。`usage.prompt_tokens` 如服务端提供，则保存为 `prompt_tokens`；未提供就为 `null`，不通过字符数假装测出输入 token 数。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--base-url` / `--model` | 必填 | 服务根地址（不带 `/v1`）与实际注册模型名 |
| `--max-tokens` | 128 | 输出上限，不保证实际输出长度 |
| `--concurrency` | 1 8 32 | 请求在途上限 C，逐档扫描 |
| `--requests` | 128 | 每档总请求数 N；N < 任一 C 时 CLI 与直接调用均先报错，不发 HTTP |
| `--timeout` | 300 s | socket 操作超时，不是整条请求总时限 |
| `--prompt` / `--prompt-file` | 默认短英文 prompt | 两者互斥；后者读 UTF-8，JSON 保存完整原文与哈希 |
| `--test-id` / `--conditions-id` | 自动生成 | 实验与条件编号；默认条件哈希只覆盖客户端参数，硬件/引擎版本需另记 |
| `--output` | bench-results.json | 汇总和逐请求原始记录的 JSON |

默认 prompt 是 `Explain why the sky is blue in one short paragraph.`，**它不等于 512 输入 token**。同一模型、tokenizer 和聊天模板下固定原文才有固定输入的意义；实际长度核对 `request_records[].prompt_tokens`。温度固定为 0 也不保证输出长度不变，EOS 可以提前结束；用逐请求 `completion_tokens` 及其分布检查实际输出，不仅记录 `max_tokens`。

这是一个有限工作负载：一次提交 N 个任务，线程池最多运行 C 个请求，结束一个才可能补上一个。启动时尚未填满，最后排空时没有新请求补位，因此 `load_model=finite_workload`，`steady_state_verified=false`。旧字段 `concurrency` 仍保留，但它是请求上限；新增 `requested_concurrency`、`max_possible_inflight=min(C,N)`、客户端实际观测的 `observed_peak_inflight` 与时间加权 `observed_mean_inflight`。这些值也不等于服务端 batch。吞吐的分子是**成功请求的总输出 token**，分母是**整档 `wall_s`**，失败占用的时间也在分母内。不要用配置 C 直接作稳态 Little 定律反查。

输出契约从第一份基线就固定下来：

| 层 | 实有字段与口径 |
|---|---|
| 顶层 | `test_id`、`conditions_id`、`raw_path`、完整 `prompt`、`prompt_source`、`prompt_sha256`、`script_sha256`、客户端 `conditions` 与测量定义 |
| 每档 | 兼容旧汇总；新增 `request_records`、`total_completion_tokens`、实际输出 `completion_tokens` 分布、服务提供的 `prompt_tokens` 分布、成功请求总延迟 `total_s` 分布 |
| 每个请求 | `request_id`（随 X-Request-ID 发出）、可选 `server_request_id`/`response_id`、`ok`、`error_type`/`error`、token 数/来源、首末内容时间、所有内容时间、chunk 数/间隔、总延迟及可空 TPOT |
| 在途观测 | 请求的 `started_at_s`/`finished_at_s` 和 `inflight_events` 相对整档起点；不把线程池等待算作服务端排队 |
| 失败与空值 | `failed`、`timeouts`、含成功/失败判定耗时的 `all_request_total_s`；未定义 TPOT 计入 `tpot_undefined_requests`，不进入 TPOT 分布 |

TTFT、成功总延迟和实际长度分布每个成功请求一个样本；ITL 的样本是内容 chunk 间隔，同一请求内的间隔相关，不能把间隔数当成同样多的独立请求。分位数按 `(n−1)×q` 线性插值；空样本为 `null`，表格显示 `n/a`。客户端成功流不代表模型质量通过，后续仍需质量回归。CLI 退出码 0 表示所有请求成功，1 表示至少一个请求失败（仍保存记录），2 表示参数不合法。

### mock 验证能证明什么

[`mock_sse.py`](scripts/mock_sse.py) 是仓库内的完整标准库入口，Linux、macOS、Windows 都可用 Python 运行；它只绑定 `127.0.0.1`。默认端口 8400，`--port 0` 由系统选择临时端口，启动行会打印实际地址。它先发空 role chunk，默认等待 20 ms 后发内容，再按 10 ms 间隔发送；一个 `x` 是一个**模拟 token**，输出数量受请求的 `max_tokens` 限制。

下面从仓库根目录运行，无需 GPU、Docker 或额外依赖。终端一：

```bash
python3 docs/modules/part1-foundations/scripts/mock_sse.py --port 8400
```

终端二，20 请求只用于 C=1/4 的 mock 冒烟：

```bash
python3 docs/modules/part1-foundations/scripts/bench_openai.py \
  --base-url http://127.0.0.1:8400 --model mock --max-tokens 16 \
  --concurrency 1 4 --requests 20 \
  --test-id mock-E1-r1 --conditions-id mock-stdlib-v1 \
  --output mock-results.json
```

检查两档 `ok=20`、每档 `total_completion_tokens=320`、逐请求 `completion_tokens=16`、`content_chunks=16`、`itl_samples` 长度 15，以及 `tokens_estimated=false`。TTFT 应包含约 20 ms 的模拟首内容等待，ITL 与 TPOT 在 10 ms 量级，但线程、HTTP 和平台计时会扰动它们；不要断言每个间隔精确等于 10 ms。观测峰值只需不超过请求并发上限，短请求不保证填满 C。

想验证估算分支，给 mock 加 `--no-usage` 再跑；16 个 `x` 会估成 4 token，JSON 与终端都应显著标记，不能沿用原吞吐/TPOT。`--prompt-tokens` 只能设置模拟输入计数，默认不提供，不能当真实 tokenizer 的测量。

两份入口都支持 `--help`。回归测试从仓库根目录运行：

```bash
python3 -B -m unittest discover -s tests -p test_bench_openai.py -v
```

测试用临时端口或稳定 HTTP/SSE mock 边界，验证空角色块、usage、多 token chunk、单 token、失败/截断流与原始记录。mock 验证的是客户端这把尺子，不证明任何 GPU/NPU 性能；真实基线按第 5 章跑，未具备硬件时在报告中保留待测状态。

**延伸**：[vLLM 的 OpenAI 兼容服务器文档](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/)（`stream`、`stream_options` 等请求字段）、[WHATWG 的 Server-Sent Events 规范](https://html.spec.whatwg.org/multipage/server-sent-events.html)（`data:` 行与事件流解析）。

## 第 5 章 · 跑出第一份基线

**本章目标。** 在固定环境、模型和负载条件下做一次有限并发扫描；保存可追溯的原始请求记录，从曲线找探索性的拐点区间，并把已测与待测分开写进基线报告。

基线是优化前的测量快照，不是最高分。它要回答：哪些负载档位开始排队或失去吞吐增益，成功请求的尾部如何变化，当前证据更支持哪个瓶颈候选。后面的优化与它比较时，使用同一条件编号和同一测量口径。

### 先探路，再扫一条曲线

先按[快速上手](../../quickstart.md)确认客户端可用，再按第二部分第 8 章第 2 节部署 vLLM；SGLang 或支持同一流式接口的服务也可以。**没有已运行的服务，就先走第 4 章的 mock 路径，不必先完成其余理论章节。**

冒烟检查只验证接口、模型名、完整 SSE 和记录字段；20 请求只用于 C=1/4 的 mock 冒烟，不用于宣称 32/64 并发容量。真实服务的预热另做并记录数量，直到时延、编译与缓存状态符合你的测量条件；一两个请求成功不等于已进入稳态。

正式扫描示例从仓库根目录运行，模型名换成服务实际注册值：

```bash
MODEL_NAME="Qwen/Qwen3.5-9B"
python3 docs/modules/part1-foundations/scripts/bench_openai.py \
  --base-url http://127.0.0.1:8000 --model "$MODEL_NAME" \
  --max-tokens 128 --concurrency 1 8 32 64 --requests 1024 \
  --test-id a40-E2-r1 --conditions-id a40-env-v1 \
  --output bench-baseline-r1.json
```

默认配置是 A40 48GB + Qwen3.5-9B；24GB 卡在服务端把模型换成 `Qwen/Qwen3.5-4B`，启动参数加 `--max-model-len 8192`；上面的客户端命令只换 `--model`，命令形状不变。

1024 使总请求数大于每个并发上限，减少过短批次的影响；它不是 p99 准确或稳态已建立的保证。1/8/32/64 是起始档位，不预设哪档饱和。依据第一次曲线加密拐点附近的档位，并用 r2、r3 重复，保存每次原始 JSON；更完整的到达率与容量评估见第 11 章。

同一轮固定模型权重、tokenizer/聊天模板、prompt 原文、输出上限、采样参数、引擎版本和启动命令。默认短英文 prompt 并不等于 512 token；实际输入长度读 `request_records[].prompt_tokens`，没有 usage 时记未知。输出长度读逐请求 `completion_tokens` 及每档的同名分布，而不是把 `max_tokens` 当成实际输出。

### 报告里必须写什么

使用[基线报告模板](baseline-report-template.md)，从开始就填写 `test_id`、`conditions_id`、`raw_path` 和真实验证状态，约定见[实验记录](../../experiments.md)。默认条件哈希只覆盖客户端参数，不能替代软硬件版本清单。

| 记录 | 内容与依据 |
|---|---|
| 环境 | 日期、设备/显存、驱动、实际 CUDA toolkit、引擎/镜像版本、模型权重修订、完整启动命令 |
| 客户端条件 | JSON 内的完整 prompt 与哈希、`max_tokens`、每档 C/N、超时、预热方式、脚本哈希 |
| 请求追溯 | `levels[].test_id` 对应档位，`request_records[].request_id` 对应请求；成功与失败都保留 |
| 结果 | TTFT、chunk 级 ITL、可定义的 TPOT、成功总延迟、实际长度分布、总输出 token、成功/失败/超时、整档墙钟时间 |
| 在途与排队 | 请求上限 C、客户端观测峰值/均值；服务端 waiting/running、KV 与抢占指标另采，未观测就留空 |
| 结论 | 曲线变化、SLO 是否满足、瓶颈候选及证据、待验证的下一步 |

`nvidia-smi` 顶部的 CUDA Version 是**驱动支持的 CUDA 上限**，不是已安装 toolkit 版本。工具链可用 `nvcc --version` 核对，运行时与容器版本分别记录；没有 toolkit 时写未安装/未核实。引擎用包版本或镜像 digest、模型用权重 revision，不凭记忆填值。

请求 TTFT 和成功总延迟的分位数只统计成功请求，必须同时附成功数、失败原因与超时数；`all_request_total_s` 单列到成功结束或失败判定的耗时，不能与成功延迟混用。吞吐使用 `total_completion_tokens / wall_s`，分子排除失败的部分输出，分母仍包含它们占用的时间。遇到 token 估算或 TPOT 为 `null`，表格明确标记，不用 0 顶替。

### 拐点怎么读

先看相邻档位的吞吐增幅，再看 TTFT/TPOT、实际长度和失败率是否一起变化。如果吞吐增益变小而等待与尾延迟升高，就在两档之间加密测试；这只是瓶颈候选，需监控或对照实验确认。只报吞吐最高的一档会掩盖 SLO 代价，只跑 C=1 又看不到批处理与排队的变化。

结果表直接从自己的 JSON 填，不预填模拟硬件成绩：

| 请求 C | 观测峰值 | 成功 / N | TTFT p50/p95/p99（s） | ITL chunk p50/p95（s） | TPOT p50（s） | 总延迟 p95（s） | 输出 token/s |
|---:|---:|---|---|---|---|---|---|
| 1 | 待测 | /1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | 待测 | /1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 32 | 待测 | /1024 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 64 | 待测 | /1024 | 待测 | 待测 | 待测 | 待测 | 待测 |

每个档位保留条件与验证状态。满足 SLO 的档位是本次负载下的候选配置，不能把客户端 C 直接解释成服务端同时运行数或普适容量。饱和和失败档位保留以说明边界，单独标记，不作为正常服务配置推荐。

本脚本的完整窗口包含启动与排空，`observed_mean_inflight` 往往小于 C。**不能用 C / 请求吞吐反算稳态平均延迟，更不能用 p50 代替均值进行 Little 定律验证。**若另有独立采得的稳态窗口，才将该窗口的平均系统请求数、完成率与平均停留时间对齐。脚本保留在途轨迹，但没有自动筛选稳态窗口。

判断算力还是带宽时，先排除排队、冷启动、客户端与代理缓冲。仅在 token 数可靠、TPOT 有定义且内容 chunk 足够细时，将单请求 `1/TPOT` 与权重/KV 搬运的理论上限作粗略对照；差距由第 3 章 profiling 解释，不能凭一个时延数字确定瓶颈。

### 没有 GPU 时怎么办

完成第 4 章的标准库 mock 和回归测试后，报告状态写“mock 已验证，真实模型性能待测”。保存 mock JSON 用于检查客户端，不把其 token/s 或模拟 prompt token 数当作真实模型成绩。

真实环境缺失的字段写待测，补测任务包含设备型号/显存、版本组合、启动命令、固定 prompt、C/N 和验收指标。拿到能运行目标服务的环境后，另做预热，用正式扫描与重复运行补齐记录；不直接复用 20 请求冒烟作容量结论。

**延伸**：[GuideLLM：在真实部署上评估 LLM](https://developers.redhat.com/articles/2025/06/20/guidellm-evaluate-llm-deployments-real-world-inference)、[GuideLLM 仓库](https://github.com/vllm-project/guidellm)、[vLLM `bench serve` 文档](https://docs.vllm.ai/en/latest/benchmarking/cli/)。

## 本部分检查清单

- [ ] 手算过目标模型的 decode 算术强度，能说清为什么解码受带宽限制
- [ ] 能区分 TTFT、ITL、吞吐、排队深度，并知道为什么报 p95/p99
- [ ] 用 profiler 找到过一次最耗时的算子，并判断了瓶颈类型
- [ ] `bench_openai.py` 在你机器上跑通过，能解释输出的每一列
- [ ] 有一份基线报告（或明确标注了待补测）
