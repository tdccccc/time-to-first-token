# 专题 D：各章核心论文精读与数据口径档案 { .chapter-title }

<p class="chapter-summary">脱离实验条件、对照基线与 SLO 口径谈“快了几倍”没有意义；本档案汇总全书核心论文的机制精要、数字边界与延伸生态。</p>

系统工程论文中最容易误导人的，往往不是方法本身，而是从摘要里孤立摘出来的倍数——“吞吐提升 24 倍”“容量提升 525%”“最高加速 6.5 倍”。每一组数字背后，都绑着特定的年份基线、模型几何、负载长度分布与统计分母。

本档案将全书 23 章主线与专题涉及的里程碑论文统一收拢，按最新章节编号建立索引，把**核心机制、原文实验条件、数字口径差异与适用边界**一次摊开，方便查阅原文或撰写技术评审时准确引用。关于如何高效精读一篇系统论文的三遍法，可同步参考 [第 23 章](../chapters/chapter-23.md)。

---

## D.1 读论文数字的三条标尺与全书总索引

引用任何一篇系统论文的性能结果前，先核对三个问题：

1. **来源形态**：该数字来自真实 GPU 集群端到端实测、轨迹回放（Trace Replay），还是离散事件集群模拟？
2. **对照基线**：对比的是当年的朴素实现（如无连续批处理的 HF Transformers），还是经过充分调优的强基线（如 Orca / vLLM）？
3. **指标与分母**：测量的是单请求延迟、同延迟下的最大吞吐，还是给定尾延迟 SLO 约束下的有效请求容量（Goodput）？被拒请求或失败请求是否在分母内？

### 全书核心论文映射总表

| 论文 | 发表会议 / 年份 | 核心机制 | 本书对应位置 |
| --- | --- | --- | --- |
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) | OSDI 2022 | 迭代级连续批处理与选择性批处理 | [第 5 章](../chapters/chapter-05.md) |
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | SOSP 2023 | 虚拟分页 KV 管理、块表映射与写时复制 | [第 5 章](../chapters/chapter-05.md) |
| [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104) | NeurIPS 2024 | 基数树前缀缓存、缓存感知调度与结构化解码 | [第 6 章](../chapters/chapter-06.md) |
| [SARATHI](https://arxiv.org/abs/2308.16369) / [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | arXiv 2023 / OSDI 2024 | 分块 Prefill 与无停顿混批调度 | [第 6 章](../chapters/chapter-06.md) |
| [GPTQ](https://arxiv.org/abs/2210.17323) / [SmoothQuant](https://arxiv.org/abs/2211.10438) / [AWQ](https://arxiv.org/abs/2306.00978) | ICLR 2023 / ICML 2023 / MLSys 2024 | 二阶误差补偿、通道平滑与激活感知权重量化 | [第 8 章](../chapters/chapter-08.md) |
| [StreamingLLM](https://arxiv.org/abs/2309.17453) | ICLR 2024 | Attention Sink 现象与流式无限长滑窗 | [第 9 章](../chapters/chapter-09.md) |
| [Speculative Decoding](https://arxiv.org/abs/2211.17192) / [Speculative Sampling](https://arxiv.org/abs/2302.01318) | ICML 2023 / arXiv 2023 | 草稿-并行验证框架与无损拒绝采样 | [第 9 章](../chapters/chapter-09.md) |
| [EAGLE](https://arxiv.org/abs/2401.15077) / [EAGLE-3](https://arxiv.org/abs/2503.01840) | ICML 2024 / NeurIPS 2025 | 特征级自回归起草、多层特征融合与树形验证 | [第 9 章](../chapters/chapter-09.md) · [第 10 章](../chapters/chapter-10.md) |
| [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | OSDI 2024 | Prefill/Decode 阶段解耦与 Goodput 优化 | [第 11 章](../chapters/chapter-11.md) |
| [Splitwise](https://arxiv.org/abs/2311.18677) | ISCA 2024 | 按阶段匹配异构 GPU 的集群硬件设计 | [专题 B](cluster-pd-mooncake.md) |
| [Mooncake](https://arxiv.org/abs/2407.00079) | FAST 2025 | KV 中心化集群架构、分布式 KV 池与 Conductor | [专题 B](cluster-pd-mooncake.md) |
| [FlashAttention 1](https://arxiv.org/abs/2205.14135) / [2](https://arxiv.org/abs/2307.08691) / [3](https://arxiv.org/abs/2407.08608) | NeurIPS 2022 / ICLR 2024 / NeurIPS 2024 | IO 感知分块 Online Softmax、Split-Q 与 Hopper 异步流水 | [第 14 章](../chapters/chapter-14.md) · [第 16 章](../chapters/chapter-16.md) |

---

## D.2 推理引擎调度与显存管理（第 5–6 章）

### PagedAttention 与 vLLM（SOSP 2023 → 第 5 章）

围绕 vLLM 有两套广为流传的数字——**SOSP 2023 论文口径**与 **2023 年 6 月官方发布博客口径**，二者基线完全不同，绝不能混为一谈：

| 口径来源 | 对比基线与负载条件 | 核心数字 | 适用边界 |
| --- | --- | --- | --- |
| **SOSP 2023 论文** | OPT-13B/66B/175B、LLaMA-13B，GCP A100，ShareGPT/Alpaca 分布；对照 FasterTransformer 与 Orca | 旧系统仅 **20.4%–38.2%** KV 显存有效利用；同等延迟下吞吐 **2–4×**；Beam Search 显存省 **37.6%–66.3%**，较 Orca 吞吐最高 **2.3×** | 序列越长、模型越大、多路采样共享越多越显著；固定长度无共享场景收益收窄 |
| **vLLM 首发博客** | 同类模型对照当时未做连续批处理的 HF Transformers 与早期 TGI | KV 浪费降到 **4% 以下**；吞吐较 HF 最高 **24×**、较 TGI 最高 **3.5×**；并行采样显存省 **55%**、吞吐 **2.2×** | 4% 浪费依赖特定平均序列长度（短序列如 100 token、block=16 时末块内部碎片仍达 10.7%）；对照的是 2023 年旧基线 |

**阅读提示**：论文 §3 的碎片剖析、§4 的块表设计与 §5.2 的 `fork/append/free` 写时复制至今不过时；但论文基于 vLLM V0 架构，现代生产默认使用的 **vLLM V1**（统一调度器、默认开启分块 Prefill 与前缀缓存、取消 Swap 改重算）需结合 [vLLM V1 架构博客](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) 与 [Inside vLLM 剖析](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) 阅读。

### Orca（OSDI 2022 → 第 5 章）

- **机制核心**：打破传统请求级批处理（等待整批最慢请求结束才释放），提出**迭代级调度（Iteration-level Scheduling）**与针对注意力/线性层的**选择性批处理**，是今天全部连续批处理引擎的鼻祖。
- **数字口径与边界**：在 GPT-3 175B 上，对比 NVIDIA FasterTransformer，在归一化中位延迟相同时测得 **36.9×** 吞吐提升。这一数字建立在**非均匀合成长度分布**（输入 $U(32, 512)$、最大生成 $U(1, 128)$、泊松到达）之上；当所有请求长度完全一致（Homogeneous Trace）时，不存在早完成请求被阻塞的现象，优势会显著收窄。

### SGLang / RadixAttention（NeurIPS 2024 → 第 6 章）

- **机制核心**：前后端协同设计，后端用**基数树（Radix Tree）**组织跨请求的任意长度共享前缀 KV，配合 LRU 淘汰与缓存感知调度，以及基于压缩有限状态机的零开销结构化输出解码。
- **博客 vs 论文口径区分**：
  - [LMSYS 发布博客](https://lmsys.org/blog/2024-01-17-sglang/)在 Llama-7B（单张 A10G，TP=1）与 Mixtral-8×7B（8×A10G，TP=8）的前缀重任务（5-shot MMLU、ReAct Agent、JSON 解码、多轮对话、DSPy RAG）上，对照 Guidance 与早期 vLLM 测得最高 **5×** 吞吐；
  - NeurIPS 2024 正式论文扩展至 Llama-70B 与 LLaVA 多模态任务，对照 Guidance、vLLM、LMQL 测得最高 **6.4×** 吞吐、延迟最高降 **3.7×**。
- **边界**：收益强依赖于工作负载的前缀复用结构；无共享前缀时收益趋近于零，但基数树维护开销极低，因此默认开启通常不亏。

### SARATHI 与 Sarathi-Serve（OSDI 2024 → 第 6 章）

- **机制核心**：利用 Prefill 算力密集、Decode 带宽密集的互补性，把长 Prompt 切成近似等大的小块（Chunked Prefill），在每次迭代中拼入正在运行的 Decode 批，实现**无停顿调度（Stall-free Batching）**，消除长 Prefill 插入引发的 ITL 尖峰。
- **两篇论文的数字口径**：
  - **SARATHI**（arXiv 2308.16369）需严格区分“Decode 吞吐倍数”与“端到端倍数”：LLaMA-13B/A6000 为 **10× Decode 吞吐 / 1.33× 端到端**；LLaMA-33B/A100 为 **4.25× Decode / 1.25× 端到端**；GPT-3 流水线并行为 **6.29× 气泡缩减 / 1.91× 端到端**；
  - **Sarathi-Serve**（OSDI 2024）的所有倍数均带**严格尾延迟 SLO 约束**定语：对照当时的 vLLM，单 A100 跑 Mistral-7B 容量提升 **2.6×**、双 A100 跑 Yi-34B 提升 **3.7×**、流水线并行跑 Falcon-180B 提升 **5.6×**。

---

## D.3 量化、投机解码与长上下文淘汰（第 8–10 章）

### GPTQ、SmoothQuant 与 AWQ（第 8 章）

三篇代表论文瞄准的是低比特量化中的三个不同痛点：

| 方法 | 核心思想与代价位置 | 典型定位与适用边界 |
| --- | --- | --- |
| **GPTQ**（ICLR 2023） | 逐层逐列量化权重，利用逆海森矩阵（二阶导数）把当前列的舍入误差动态补偿到后续未量化列；离线校准耗时较长，推理零额外补偿开销 | W4A16 经典方案；极端低比特下对校准集分布有一定敏感性 |
| **SmoothQuant**（ICML 2023） | 引入逐通道平滑因子 $s$，等价变换 $(X / s)(s W)$，把激活中难以量化的离群尖峰迁移至权重端 | W8A8（INT8 矩阵乘）核心支柱，适合大批量 Compute-bound 加速 |
| **AWQ**（MLSys 2024） | 依据激活幅度而非权重本身大小识别约 1% 的显著权重通道，量化前放大保护、推理时融入缩放系数；无需反向传播，泛化稳健 | W4A16 生产主力，大幅压低 Memory-bound Decode 搬运量 |

!!! note "阅读经典综述的时间线提醒"
    [Lilian Weng 推理优化综述](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)写于 2023 年 1 月，非常适合建立“量什么、量几位、何时量”的顶层分类框架，但它早于 AWQ（2023 年 6 月）与 Hopper FP8 工程普及，最新方法矩阵需配合 [第 8 章](../chapters/chapter-08.md) 与 [NVIDIA 推理优化综述](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) 补齐。

### StreamingLLM（ICLR 2024 → 第 9 章）

- **机制核心**：揭示大模型对序列开头若干 token 分配极高注意力分数的 **Attention Sink（注意力汇）** 现象——即使开头 token 语义普通，它们也承担了 Softmax 归一化“泄洪槽”的角色。直接滑窗丢弃开头 token 会导致困惑度雪崩；只要永久钉住最初 4 个左右 Sink token 再拼接滑动窗口，无需微调即可稳定处理超长流式文本。
- **数字与边界**：在 Llama-2、MPT、Falcon、Pythia 上稳定外推至 **400 万 token 以上**，流式场景较滑动窗口重算基线最高快 **22.2×**。核心边界在于**它不是无损压缩**——滑出窗口的远距历史信息已被永久丢弃，无法回答超出窗口范围的精确细节检索题。

### 投机解码、EAGLE 与 EAGLE-3（第 9–10 章）

| 论文 / 来源 | 机制演进要点 | 原文代表数字与条件 | 适用边界 |
| --- | --- | --- | --- |
| **Leviathan et al.**（ICML 2023） | 提出 Draft-Verify 范式与无损拒绝采样：接受概率 $\min(1, p/q)$，拒绝后从 $\text{norm}(\max(0, p-q))$ 重采样 | T5-XXL 上 **2–3×** 加速 | 严格保持输出分布与目标模型一致；收益取决于接受率 $\alpha$ 与草稿开销比 |
| **Chen et al.**（2023） | 独立提出 Speculative Sampling 并在分布式大模型上验证 | Chinchilla 70B 上 **2–2.5×** 加速 | 验证了超大规模解码下并行验证的可行性 |
| **EAGLE**（ICML 2024） | 在倒数第二层**特征层**做自回归起草，并引入“提前一步的 token 序列”消除特征采样不确定性 | LLaMA2-Chat 70B 延迟加速 **2.7–3.5×**，吞吐约翻倍 | 轻量草稿头极大压低起草成本，但高并发算力饱和时增益消失 |
| **EAGLE-3**（NeurIPS 2025） | 摒弃特征预测约束改直接预测 token，结合 Training-Time Test、多层特征融合与树形起草验证 | 低 batch 下最高 **6.5×**，batch=1 较 EAGLE-2 快约 **1.4×**（约 8× EAGLE 训练数据） | 6.5× 是高接受率 + 树形多分支 + 低并发叠加的结果；高并发或接受率下滑时迅速回落 |

另注：[vLLM 投机解码工程博客](https://blog.vllm.ai/2024/10/17/spec-decode.html)给出的最高 **2.8×** 加速来自 Llama-3-70B、4×H100 的特定低 QPS 场景，与学术论文的测试脚本、并发区间不可直接跨组对比。

---

## D.4 PD 解耦与集群级 KV 服务（第 11 章与专题 B）

DistServe、Splitwise 与 Mooncake 依次回答了三个正交问题：**要不要拆、拆到什么卡上、拆完 KV 放哪里**。三篇论文的增益各有前提，**绝对不能把它们的倍数相乘**。

| 论文 | 核心问题与方案 | 关键数字与实验条件 | 必看口径边界 |
| --- | --- | --- | --- |
| **DistServe**（OSDI 2024） | 消除同机 P/D 干扰，独立配置 P/D 并行度与带宽感知放置 | 4 节点 32×A100（跨节点 25 Gbps）、OPT 13B–175B、90% SLO 下 Goodput 最高 **7.4×**，或 SLO 收紧 **12.6×** | 7.4× 是特定长短混合负载下的 Goodput 上界，对比未做精细 Chunked Prefill 调优的同机基线 |
| **Splitwise**（ISCA 2024） | 纠正 Decode 算力浪费，P 用高算力卡、D 用高性价比/低功耗卡 | 基于 DGX-A100/H100 性能表征 + 生产 Trace 事件驱动模拟：吞吐 **1.4×** 且降本 **20%**，或同成本同功耗吞吐 **2.35×** | 属于集群硬件规划级模拟结果，非单卡价格直比 |
| **Mooncake**（FAST 2025） | 以分布式 KV 池（DRAM/SSD）为调度枢纽，Conductor 全局路由 + 预测式提前拒绝 | 测试床：16 节点 × 8×A800（4×200 Gb/s RDMA），256-token 块，回放 Trace 前缀缓存率上界 **40%–66%**。<br>• **arXiv 摘要口径**：模拟满足 SLO 吞吐 **+525%**；Kimi 生产负载多处理 **+75%**。<br>• **FAST'25 正式版口径**：有效请求容量 **+498%**；生产 A800/H800 分别多处理 **+115% / +107%** | 两版论文指标定义不同不可混用；“有效请求容量”指同时满足 TTFT（30 s）与 TBT（最慢 10% 均值 100/200/300 ms）的请求占**含被拒请求在内全部请求**的比例，衡量的是高负载下保住 SLO 的服务能力，而非单请求快了 5 倍 |

---

## D.5 IO 感知注意力算子演进（第 14 章与第 16 章）

三代 FlashAttention 的演进堪称“紧贴 GPU 硬件微架构写算子”的教科书范例：

- **FlashAttention-1**（Dao 等，NeurIPS 2022，[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)）：通过 SRAM 分块与 Online Softmax，避免把 $N \times N$ 的注意力得分矩阵 $S, P$ 写回 HBM，将 HBM 访问从 $O(N^2)$ 降至 $O(N^2 d^2 / M)$（$M$ 为 SRAM 容量）。在 A100 上达到理论 FP16 峰值的 **25%–40%**（显著高于标准 PyTorch 注意力，但离调优 GEMM 的 80%–90% 仍有差距）。
- **FlashAttention-2**（Dao，ICLR 2024，[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)）：围绕三点逼近算力屋顶——减少非矩阵乘（Non-matmul）浮点开销、把外层并行维度改到序列长度 $Q$ 上（提升低 batch 长序列的 SM 占用率）、在 Block 内改为 **Split-Q** 切分 Warp 以彻底消除 Warp 间的 Shared Memory 归约通信。A100 上达到峰值 **50%–73%**，约为 FA1 的 **2×**。
- **FlashAttention-3**（Shah 等，NeurIPS 2024，[arXiv:2407.08608](https://arxiv.org/abs/2407.08608)）：专为 NVIDIA Hopper（H100，CC 9.0）硬件特性重写——利用 **TMA（Tensor Memory Accelerator）** 做硬件异步数据搬运，采用 **Warp Specialization（生产者/消费者 Warp 分工）** 将 WGMMA 矩阵乘与慢速 Softmax 指数运算完全流水重叠，并引入块级 FP8 量化与随机正交非相干处理抑制离群误差。H100 上 FP16 实测高达 **740 TFLOPS**（约 75% 峰值），FP8 逼近 **1.2 PFLOPS**。

---

## D.6 优质公开课程、技术博客与开源生态索引

除了精读一手论文，以下高校课程、工程系列与开源工具链是持续追踪推理工程前沿的绝佳入口：

### 推荐公开课程与长期专栏

- **[Stanford CS336：Language Modeling from Scratch](https://cs336.stanford.edu/spring2025/)**（Percy Liang、Tatsunori Hashimoto）：尤其推荐第 5 讲（GPU 架构与存储层级）、第 6 讲（Kernel 优化）、第 7–8 讲（分布式并行）与第 10 讲（推理系统）。
- **[MIT 6.5940：EfficientML.ai](https://hanlab.mit.edu/courses/2024-fall-65940)**（Song Han）：系统覆盖量化、剪枝、投机解码、KV 压缩与硬件协同设计（2023/2024 秋及后续课程录像均公开）。
- **[GPU MODE](https://www.youtube.com/@GPUMODE)**：面向实战内核开发者的深度讲座，涵盖 CUDA、Triton、CUTLASS、PyTorch 编译器与 vLLM 投机解码内核细节。
- **[vLLM Office Hours](https://www.youtube.com/@vllm-project)** 与 **[USENIX 会议录像](https://www.usenix.org/conferences)**（OSDI / NSDI / FAST）：引擎核心作者与论文一作的第一手架构分享。
- **工程长文标杆**：[Horace He — Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)、[Modal LLM Engineer's Almanac](https://modal.com/llm-almanac/summary)、[vLLM 官方博客](https://blog.vllm.ai) 与 [LMSYS 博客](https://lmsys.org/blog/)。

### 核心开源推理生态速查

| 开源项目 | 定位与典型用途 |
| --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) | 主流高吞吐服务引擎（PagedAttention、V1 架构、多硬件平台插件体系） |
| [SGLang](https://docs.sglang.io) | RadixAttention 前缀复用与高吞吐结构化生成引擎 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA 官方深度优化推理运行时（PyTorch 后端 + 高性能 CUDA/FP8 内核） |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) | 分布式 KV 缓存池（Mooncake Store）与 RDMA 聚合传输引擎（Transfer Engine） |
| [vLLM production-stack](https://github.com/vllm-project/production-stack) / [llm-d](https://llm-d.ai) | Kubernetes 推理编排、前缀感知路由与延迟预测式负载均衡 |
| [GuideLLM](https://github.com/vllm-project/guidellm) / [genai-perf](https://github.com/triton-inference-server/perf_analyzer) / [LLMPerf](https://github.com/ray-project/llmperf) | SLO 感知的恒定速率扫描与端到端流式压测工具 |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 标准化开源模型质量回归评测框架（量化与跨芯片精度验收必备） |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) / [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | 工业级 Attention 与矩阵乘 CUDA/CuTe 内核权威源码库 |
