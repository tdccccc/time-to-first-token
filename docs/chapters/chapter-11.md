# 第 11 章 生产架构：PD 解耦、扩缩容与成本路由 { .chapter-title }

<p class="chapter-summary">从 Roofline 瓶颈错配出发算清 Prefill/Decode 分池的干扰收益与传输税，建立以排队深度与 KV 水位为核心的 Kubernetes 弹性扩缩容体系，再把每百万 Token 的真实硬件成本写进多后端路由器与原子预算账本。</p>

把一台单机引擎调到极限之后，生产环境会立刻抛来三个集群级难题：第一，长 Prompt 涌入瞬间把正在流式输出的并发请求全部卡顿，单机调参已触及天花板；第二，流量突发时按传统 CPU 或 GPU 利用率配的自动扩缩容要么纹丝不动、要么疯狂误扩；第三，月底云账单来了，怎么把按小时计费的 GPU 租金折算成精确的每百万输入/输出 Token 成本，并在路由层把钱花在刀刃上？本章用同一套可复算的物理账本，把这三道生产关口一气打通。

## 11.1 为什么拆分 Prefill 与 Decode：干扰账与分块边界

### 痛点：同机混跑的“瓶颈错配”与“线头阻塞”

回到第 1 章的 [Roofline 框架](chapter-01.md)，Prefill 与 Decode 在同一张 GPU 上呈现截然相反的物理习性。以 **NVIDIA H100 80GB SXM**（FP16/BF16 Tensor Core 稠密峰值 **989.5 TFLOPS**，HBM3 带宽 **3.35 TB/s**，脊点 **$295.37\text{ FLOP/B}$**）运行 **Llama-3.1-8B（FP16）**（参数量 $N = 8.03\text{B}$，权重 $W = 16.06\text{ GB}$，每 token KV $k = 128\text{ KiB}$）为例，分别核算两个阶段：

- **1,000-token Prefill（极度计算受限）**：权重主导计算量 $F_P \approx 2 N L = 16.06\text{ TFLOP}$，读权重加写 KV 总搬运量约 $16.191\text{ GB}$，算术强度高达 $991.9\text{ FLOP/B}$（约为 H100 脊点的 **3.36 倍**）。跑满稠密峰值需 $16.23\text{ ms}$，按工程常见的 **50% MFU** 折算耗时约 **$32.46\text{ ms}$**（若 Prompt 增至 4,000 token，50% MFU 下仅权重计算就高达 **$129.84\text{ ms}$**）。
- **1,000-token 上下文下的单请求 Decode（极度带宽受限）**：每生成 1 个 token 的计算量仅 $16.06\text{ GFLOP}$（算术强度低至 $0.99\text{ FLOP/B}$），但仍需从 HBM 搬运 $16.191\text{ GB}$ 权重与 KV，带宽下界为 **$4.83\text{ ms}$**（单请求理论上限约 207 tok/s）。在此工作点上，H100 实际用到的算力仅为 $3.32\text{ TFLOPS}$，占 989.5 TFLOPS 峰值的 **$0.34\%$**！

把这两个习性水火不容的阶段关在同一张 GPU 里混跑，会同时触发两重损耗：

1. **线头阻塞（Head-of-Line Blocking）**：在两个流畅的 4.8 ms Decode 步之间，只要串行插进一条 1,000-token（32.5 ms）或 4,000-token（129.8 ms）的 Prefill，同卡上所有正在吐字的 Decode 请求立刻被硬生生“冻结”几十到上百毫秒，造成剧烈的 **ITL 尾延迟尖峰**；
2. **硬件资源错配**：Prefill 渴望 H100 级的超高 Tensor Core 算力，而 Decode 在中小并发下连 A100（312 TFLOPS / 2 TB/s，单步带宽下界 8.10 ms，算力利用率仅 $0.64\%$）的算力零头都吃不完。同机混跑迫使你为带宽受限的 Decode 支付顶级算力卡的溢价，且两阶段被迫共享同一套张量并行度与显存预算。

<figure class="book-figure">
  <img src="../../../assets/figures/pd.svg" alt="PD 解耦基础结构：Router 将请求先发往算力密集的 Prefill 池生成完整 KV，经网络或显存传输至带宽密集的 Decode 池逐 token 生成" loading="lazy">
  <figcaption><span class="book-figure__label">图 11-1</span>读图顺着请求流向看：Router 先将 Prompt 交给算力型 Prefill 池，生成整段 KV 后经显式传输送往带宽型 Decode 池。结论：PD 解耦用可度量的 KV 传输成本，换取两阶段彻底隔离与独立资源配比。</figcaption>
</figure>

### Chunked Prefill 能走多远？与物理拆池的边界

在直接拆机之前，同机路线有一道极强的防线——**Chunked Prefill（分块预填充）**（已在[第 5 章](chapter-05.md)与[第 10 章](chapter-10.md)介绍）。它把长 Prompt 切成小块，穿插到连续的 Decode 步中混合执行，将一次几十毫秒的大阻塞摊平为多次可控的小波动。

但 Chunked Prefill 也有清晰的物理天花板。沿用 H100（50% MFU）+ Llama-3.1-8B 核算 1,000-token Prompt 按不同分块单独执行的代价（每块步时取计算与搬运最大值 $\max\!\left(\frac{2 N \cdot c}{989.5\text{T} \times 0.5},\; \frac{W + k c}{3.35\text{T}}\right)$）：

| 最大分块大小（tokens） | 切分块数 | 单块步时估计（ms） | 总 Prefill 耗时（ms） | 单块算术强度（FLOP/B） | 所处 Roofline 区间 |
|---:|---:|---:|---:|---:|---|
| 1,000 | 1 | 32.46 | 32.46 | 991.9 | 稳居计算受限侧 |
| 500 | 2 | 16.23 | 32.46 | 498.0 | 计算受限侧 |
| 250 | 4 | 8.12 | 32.46 | 249.5 | 略低于峰值脊点，50% MFU 下仍由计算主导 |
| 125 | 8 | 4.80 | **38.39** | 124.9 | 跌入带宽受限侧，每块重复搬权重，总耗时抬升 |
| 63 | 16 | 4.80 | **76.74** | 63.0 | 严重带宽受限，总耗时暴涨至 2.36× |

计算项与搬运项在约 **148 token/块** 处相交：块切得比 148 还细，单步搬运 16.06 GB 权重的 **4.80 ms 地板**不再下降，块数却成倍增长，不仅拖垮 TTFT，还让大量重复权重读取挤占本就宝贵的 HBM 带宽。反过来，若块切得大（如 500 token），单步 16.23 ms 叠加同批 Decode 与调度开销，又会刺穿苛刻的 ITL p99 红线。

<figure class="book-figure">
  <img src="../../../assets/figures/pd-timeline-cluster.svg" alt="上半部对比同机混跑、Chunked 切块与 PD 解耦三轨时间线；下半部展示以共享分布式 KV Cache 池与 Transfer Engine 为中心的 Mooncake 集群架构" loading="lazy">
  <figcaption><span class="book-figure__label">图 11-2</span>先对照上半部三条执行时间线：同机混跑在长 Prefill 处产生明显 ITL 尖峰，Chunked 切块通过穿插平抑毛刺但仍争抢同卡资源，PD 解耦将两阶段彻底分轨并引入显式 KV 传输段；下半部进一步示意超大规模集群如何引入共享 KV 池。结论：Chunked Prefill 是中小规模同机的强基线，PD 物理拆池则是双严格 SLO 下的终极隔离解。</figcaption>
</figure>

!!! note "工程取舍：什么时候该从 Chunked Prefill 迈向 PD 解耦？"
    切忌把 PD 解耦当成“默认必选项”。当满足以下任一条件时，**同机 Chunked Prefill 往往更省钱更简单**：总 GPU 卡数少（如 $\le 4$ 卡）、节点间缺乏高速 RDMA 互联（仅 10GbE/拥塞以太网）、或业务以短输入（$< 1\text{K}$）为主。只有当**长上下文输入密集（4K–32K+）、TTFT 与 ITL p99 同时面临严苛 SLA、且集群规模足以按需异构配比 P/D 两池**时，物理 PD 解耦的收益才会彻底释放。

## 11.2 PD 解耦的传输税与两池配比

把 Prefill 和 Decode 拆到两套独立实例后，干扰消失了，但 Prefill 刚算好的整段 KV 缓存必须跨卡或跨节点搬给 Decode 实例——这就是解耦的**传输税（Transfer Tax）**。

### 精确核算 KV 传输税：互联带宽决定生死

先守住一条绝不可混的单位换算铁律：网络标称带宽用十进制 **Gb/s**（除以 8 才等于 **GB/s**），而 KV 体积按二进制 **MiB / GiB** 计（$1\text{ GiB} = 1.0737\text{ GB}$）。以 **Llama-3.1-8B（128 KiB/token）** 为例，对比不同上下文长度在三档主流互联与低速以太网下的理想线速传输耗时，以及传输耗时与 H100（50% MFU）Prefill 耗时之比 $t_{\text{transfer}} / t_{\text{prefill}}$：

| Prompt 长度 | KV 体积 | H100 Prefill（50% MFU） | 同机 NVLink 4<br>（双向 900 / 单向 450 GB/s） | 跨机 400G InfiniBand<br>（50 GB/s） | 普通 100GbE<br>（12.5 GB/s） | 10GbE TCP<br>（1.25 GB/s） |
|---|---:|---:|---:|---:|---:|---:|
| 1K（1,024） | 128 MiB | 33.24 ms | 0.30 ms（0.90%） | 2.68 ms（8.08%） | 10.74 ms（32.3%） | 107.4 ms（323%） |
| **4K（4,096）** | **512 MiB** | 132.96 ms | **1.19 ms**（0.90%） | **10.74 ms**（8.08%） | **42.95 ms**（32.3%） | 429.5 ms（323%） |
| **8K（8,192）** | **1 GiB** | 265.92 ms | **2.39 ms**（0.90%） | **21.47 ms**（8.08%） | **85.90 ms**（32.3%） | 859.0 ms（323%） |
| 32K（32,768） | 4 GiB | 1,063.68 ms | 9.54 ms（0.90%） | 85.90 ms（8.08%） | 343.60 ms（32.3%） | 3,436 ms（323%） |

（若模型换为 36 层的 **Qwen3-8B**，每 token KV 为 144 KiB，8K KV 达 1.125 GiB，上表各格耗时按 $144/128 = 1.125\times$ 放大。）

这张表把解耦的盈亏平衡公式写得明明白白：

$$\text{解耦净收益} = \text{消除的混跑干扰与排队} + \text{异构硬件与独立并行红利} - \text{进入关键路径的 KV 传输税}$$

- **同机 NVLink 4（单向 450 GB/s）与跨机 400G IB/RoCE RDMA（50 GB/s）**：传输税仅占 Prefill 计算的 **0.9%–8.1%**，配合**逐层流水线重叠**（第 $l$ 层刚算完即异步发出该层 KV，不必等全部 32 层算完），真正落入 TTFT 关键路径的只有最后几层的传输尾巴。省下的几十毫秒 ITL 阻塞与独立扩缩红利轻而易举盖过这笔税；
- **100GbE 及以下慢网**：4K KV 就要传 43 ms、8K 要传 86 ms，在多请求争用与协议开销下极易翻倍，传输税反而成了 TTFT 的最大杀手，此时强行跨机拆池得不偿失。

### DistServe、Splitwise 与分布式 KV 池演进

围绕“怎么拆、怎么配卡、怎么搬 KV”，学术与工业界形成了三代清晰演进：

- **DistServe（OSDI 2024）**：首次系统提出以 **Goodput（同时满足 TTFT 与 TPOT 双重 SLO 的有效请求速率）** 取代裸吞吐，将 Prefill 与 Decode 解耦至独立 GPU 池，允许两侧采用不同张量/流水并行策略，并结合集群互联拓扑就近放置两池以压低 KV 传输。
- **Splitwise（ISCA 2024）**：直击两阶段的硬件错配——Prefill 留在 H100 等高算力卡，Decode 下放至更高性价比或降功耗的带宽型 GPU，以“同采购预算与同功耗下的集群总吞吐”为核心优化目标。
- **以分布式 KV 池为枢纽（Mooncake / NIXL / LMCache）**：前两者仍把 KV 视作“P 算完发给 D、用完即扔”的一次性货物；在超长上下文与高前缀复用场景下，Mooncake 将集群节点闲置的 CPU DRAM 与 NVMe SSD 拼成全局共享 KV 池，由 **Conductor** 做全局前缀感知调度、**Transfer Engine** 经多网卡 RDMA 批量搬运（参见图 11-2 下半部）。关于大规模跨节点 KV 池的哈希寻址、热块复制、预测式提前拒绝与故障恢复细节，请参阅 **[专题 B：大规模集群 PD 与分布式 KV 池](../topics/cluster-pd-mooncake.md)**。

### 两池实例数配比直觉：绝不能拿单请求时长直接分卡

解耦上线前最常犯的架构错误，是用“单请求 TTFT vs 单请求生成总时长”直接拍板 P/D 卡数比（例如拍脑袋定个 1:4）。正确的起点是按**单请求消耗的 GPU 卡时（GPU-seconds）**结合连续到达率 $\lambda$ 与目标装载率 $\rho$（如 0.8）估算：

$$N_{\text{pool}} = \left\lceil \frac{\lambda \cdot t_{\text{gpu}}}{\rho} \right\rceil$$

最关键的直觉在于：**Decode 池的权重读取是被整个并发 Batch $B$ 共同摊销的！**设 P 阶段产出首字并复用，D 阶段需执行 $n = G - 1$ 步，第 $i$ 步读取累计历史 KV 长度 $L + i$，则在 H100（50% MFU Prefill，带宽主导 Decode）上运行 Llama-3.1-8B 时，每请求两池 GPU 卡时为：

$$t_P = \frac{2 N L}{C_{\text{H100}} \times 0.5}, \qquad t_{D,\text{gpu}}(B) = \frac{n \cdot \frac{W}{B} + k \left(n L + \frac{n(n-1)}{2}\right)}{\text{BW}_{\text{H100}}}$$

代入两种典型业务形状，立刻能看到 $P:D$ 配比的剧烈翻转：

| 业务负载形状 | P GPU 卡时 $t_P$ | D 卡时（$B=1$ 错算） | D 卡时（$B=32$ 摊销） | 卡数比 $P:D$（$B=1 \to B=32$） | $\lambda=100\text{ req/s}, \rho=0.8$ 配卡示例 |
|---|---:|---:|---:|---|---|
| **长答对话**（2000 in / 200 out） | 64.92 ms | 970.35 ms | **46.16 ms** | $1:14.95 \to \mathbf{1.41:1}$ | **9 张 P 卡 + 6 张 D 卡** |
| **RAG / 摘要**（4000 in / 20 out） | 129.84 ms | 94.07 ms | **5.83 ms** | $1.38:1 \to \mathbf{22.28:1}$ | **17 张 P 卡 + 1 张 D 卡**（按高可用配 $\ge 2$ 张） |

同一套“2000 in / 200 out”对话负载，如果不做 Batch 摊销，会荒谬地算成需要 122 张 D 卡配 9 张 P 卡；一旦按稳态 $B=32$ 摊销权重搬运，每请求在 D 池实际只占 46.16 ms 卡时，反而变成 **P 卡略多于 D 卡**！可见生产中绝无放之四海而皆准的静态配比，两池卡数必须紧跟真实业务的输入输出长度比、可达到的稳态并发 $B$ 以及物理 KV 容量上限动态调整。

## 11.3 Kubernetes 推理扩缩容原理

将单机引擎或 PD 两池搬上 Kubernetes（例如借助 vLLM `production-stack` Helm Chart 声明两套独立 Deployment 与 Router），真正的挑战不在把 Pod 拉起来，而在于**选对自动扩缩容（Autoscaling）的控制信号**。

### 为什么不能按 CPU 或 GPU 利用率做 HPA？

在传统 Web 微服务中，按 CPU 利用率配置 HPA（Horizontal Pod Autoscaler）屡试不爽；但在 LLM 推理集群里，CPU 与 GPU 利用率都会系统性失灵：

- **CPU 利用率与推理瓶颈毫无单调关系**：正向失灵——GPU 早已打满、请求排队几十条，但 vLLM 的 CPU 进程只做轻量的分词与 Python 调度，CPU 利用率长期趴在 15%，HPA 毫无察觉，队列越积越死；反向失灵——Prometheus 集中抓取、日志刷盘或一波突发长文本分词把 CPU 冲到 85%，GPU 却在空转，HPA 误判触发无意义的昂贵扩容。
- **GPU SM 利用率（`nvidia-smi` GPU-Util）同样失真**：NVIDIA 驱动上报的 GPU Util 衡量的是“过去采样窗口内任意时刻有 kernel 在跑的时间占比”，而不是算力饱和度。在带宽受限的 Decode 阶段，即便 `batch=1` 刚刚起步，GPU Util 也常常直接顶到 90%–100%；换言之，它在“轻载畅通（0 排队）”和“严重过载（排队 100 条、TTFT 飙升 30 秒）”之间几乎没有区分度！

真正与用户体验和系统瓶颈严格单调相关的控制信号，必须直接取自推理引擎内部暴露的业务级指标（第 7 章已详解监控体系）：

1. **主信号：`vllm:num_requests_waiting`（排队等待深度）**——直接反映已有多少请求因算力或 KV 槽位耗尽而被拦在门外；
2. **辅信号：`vllm:kv_cache_usage_perc`（KV 缓存水位）**——在排队还没彻底爆发前提前感知显存逼近抢占红线（如持续 $> 85\%$）；
3. **联合校验：TTFT 变化率**——区分“真容量不足（等待数持续攀升且 TTFT 同步上涨）”与“瞬时微突发（等待数冲高后几秒内回落、TTFT 几乎不动）”。

Kubernetes 原生 HPA 不直接读 Prometheus 指标，生产标准做法是通过 **KEDA（Kubernetes Event-driven Autoscaling）** 周期轮询 Prometheus 的引擎队列指标，自动生成并接管底层 HPA。

### 流量突发、冷启动提前量与排空算例

推理 Pod 的弹性扩缩容与普通无状态 Go/Java 服务最大的不同在于**高昂的冷启动时间**。拉起一个新的 **Llama-3.1-8B** GPU 副本，不仅要调度节点、拉取数 GB 容器镜像（生产务必预先预热节点镜像缓存），还要从存储读取 16 GB 权重（按 1 GB/s 读盘需 16 秒），再加上初始化显存池与捕获 CUDA Graph（约 15–30 秒）——整体冷启动窗口通常落在 **30–60 秒** 量级。

用一组具体数字算清突发流量下的扩容、排空与防抖账：

- **突发积压**：单副本稳态处理能力为 $2\text{ req/s}$，流量突然翻倍至 $\lambda = 4\text{ req/s}$。每秒净增 $4 - 2 = 2$ 条排队请求；经过 30 秒冷启动与观测窗口后，队列积压达到 **60 条**（此时新到队尾的请求若靠单副本消化需苦等 $60 \div 2 = 30\text{ 秒}$！）。
- **HPA 超配排空公式**：HPA 按比例计算期望副本数：

$$N_{\text{desired}} = \left\lceil N_{\text{current}} \times \frac{\text{Metric}_{\text{current}}}{\text{Metric}_{\text{target}}} \right\rceil$$

目标阈值 `threshold = 10`（每副本最多容忍 10 条等待，对应约 5 秒排队上限；稳态下用利特尔法则 $L = \lambda W$ 验算：2 个副本共 20 条等待 + 2 条运行 = 22 条在途，到达率 $4\text{ req/s}$ 时平均停留 $22 \div 4 = 5.5\text{ s}$，恰好是 5 秒排队 + 0.5 秒服务）。面对当前 1 副本、60 条等待：

$$N_{\text{desired}} = \left\lceil 1 \times \frac{60}{10} \right\rceil = \mathbf{6}\text{ 个副本}$$

- **快速排空与缩容冷却**：稳态其实只需 $4 \div 2 = 2$ 个副本，但 2 个副本刚好与到达率打平，60 条历史存量永远排不掉！KEDA 短暂超配至 6 个副本，总能力飙升至 $12\text{ req/s}$，净排空速率 $12 - 4 = 8\text{ req/s}$，只需 **$60 \div 8 = 7.5\text{ 秒}$** 就将积压彻底清零。排空之后绝不可立刻缩容，否则新副本刚热完身就被销毁、下一波小波动立刻重新引发雪崩；因此必须配置**非对称防抖策略——扩容快（如轮询 15 秒、短稳定窗）、缩容慢（冷却窗口 `cooldownPeriod` 设 300–360 秒，每次缩容限步长）**，让副本数平滑回落到 2–3 个稳态值。

在 vLLM `production-stack`（或标准 KEDA `ScaledObject`）中，基于排队深度与冷却窗口的核心声明清晰简洁：

```yaml
# 基于 KEDA + Prometheus 排队深度的推理弹性配置片段
keda:
  enabled: true
  minReplicaCount: 2            # 保留基础冗余，避免冷启动击穿低谷突发
  maxReplicaCount: 8
  pollingInterval: 15           # 15s 轮询，快速响应持续积压
  cooldownPeriod: 360           # 6 分钟缩容冷却，防止队列刚排空就剧烈抖动
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://kube-prometheus-stack-prometheus:9090
        metricName: vllm_num_requests_waiting
        query: avg(vllm:num_requests_waiting{model_name="llama-3.1-8b"})
        threshold: "10"         # 单副本目标队列上限，由 SLO 等待预算倒推
```

在 PD 解耦部署下，Prefill 池与 Decode 池各自挂载独立的 KEDA 策略：Prefill 池紧盯 `prefill` 队列等待数与 TTFT，Decode 池紧盯 `kv_cache_usage_perc` 水位与 ITL，实现“输入变长扩 P、输出变长扩 D”的精准独立弹性。

## 11.4 单位 Token 经济学与多后端路由器

技术优化做到最后，一切指标都要汇成财务总监和产品经理能看懂的一行数字：**每百万 Token 硬件成本**。

### 每百万 Token 硬件成本拆解：为何输出比输入贵 3–5 倍？

GPU 按小时固定计费，而推理收入与消耗按 Token 流动。连接这两者的核心桥梁是**全天有效利用率 $U$**（实际产出 Token 除以峰值吞吐乘以窗口总秒数）与**请求成功率 $S$**：

$$\text{每百万有效 Token 成本} = \frac{\text{GPU 小时价}}{P_{\text{peak}} \times 3600 \times U \times S} \times 10^6$$

利用率放在分母里，带来两个残酷而直接的商业推论：第一，**利用率翻倍与 GPU 租金砍半完全等价**——同样一块 2.00 美元/小时、峰值 $1,000\text{ tok/s}$ 的 GPU，利用率从 $10\%$（5.56 美元/百万 token）提到 $80\%$（0.69 美元/百万 token），成本直降 8 倍，且低利用率下有 $90\%$ 的租金都白白买单了空转！第二，**切忌用晚高峰 80% 利用率填月底预算表**，夜间低谷往往把 24 小时均值拉回 40%–60%；同时失败重试不仅压低分母 $S$，还会让 Prefill 重复计费。

更进一步，输入 Token 与输出 Token 的成本天差地别。我们以单张 **A100 80GB**（按示例综合租金 2.00 美元/小时，全天平均利用率 $U = 0.60$）运行 **Llama-3.1-8B（FP16）**、稳态拐点并发 $B = 32$ 时的两个工作点为例，用纯差分法把输入与输出边际单价彻底剥离出来：

| 工作点 | 输入 / 输出长度 | 稳态输出吞吐 | 等效请求速率 | 单请求 GPU 时 | 单请求硬件成本（美元，含 $U=0.60$） |
|---|---:|---:|---:|---:|---:|
| **A** | 512 / 128 | 1,300 tok/s | 10.16 req/s | 0.0985 s | $9.12 \times 10^{-5}$ |
| **B** | 2,048 / 128 | 480 tok/s | 3.75 req/s | 0.2667 s | $2.47 \times 10^{-4}$ |

两个工作点的输出长度完全相同（128 token），唯一差异是 B 比 A 多了 $2,048 - 512 = 1,536$ 个输入 token，带来 $1.557 \times 10^{-4}$ 美元的单请求成本增量。由此立刻解出两项边际单价：

$$\text{输入边际单价 } c_{\text{in}} = \frac{2.469 \times 10^{-4} - 9.117 \times 10^{-5}}{1536} \times 10^6 \approx \mathbf{0.101\text{ 美元 / M tokens}}$$

$$\text{输出边际单价 } c_{\text{out}} = \frac{9.117 \times 10^{-5} - 512 \times 1.014 \times 10^{-7}}{128} \times 10^6 \approx \mathbf{0.307\text{ 美元 / M tokens}}$$

我们可以立刻把这两个价格**反推回硬件物理规格**验算闭环：

- $c_{\text{in}}$ 对应 $1,536$ 输入 token 耗时 $0.168\text{ s}$，即等效 Prefill 吞吐 $9,132\text{ tok/s}$，折合算力 $9,132 \times 16.06\text{ GFLOP} \approx 146.7\text{ TFLOPS}$，恰好是 A100 312 TFLOPS 峰值的 **$47\%$ MFU**；
- $c_{\text{out}}$ 对应每输出 token 占 $0.331\text{ ms}$ GPU 时，32 路并发下每步约 $10.6\text{ ms}$，搬运 16.06 GB 权重 + 32 路 KV（约 2.6 GB）$\approx 18.7\text{ GB}$，等效 HBM 带宽 **$1.76\text{ TB/s}$**，精准贴近 A100 2 TB/s 的带宽天花板！

这就是商业大模型 API **输出 Token 价格普遍高于输入 3–5 倍**的底层物理铁律：输入 Token 享受高度并行计算（算力受限、单 token 占用 GPU 时间极短），而输出 Token 必须串行逐字生成并反复搬运巨量权重与增长的 KV（带宽受限）；在更低并发或更长上下文下，Batch 摊销变薄，输出相对输入的单价倍率还会进一步放大至 4–5 倍以上。

### 多后端路由器与原子令牌预算中间件

算出不同尺寸模型的精确单价之后（例如同集群内 8B 模型单价往往只有 70B 模型的 $1/5$ 到 $1/10$），生产降本的最后一公里不再发生在单张 GPU 内部，而是发生在推理集群前面的**预算网关 + 多后端智能路由器**。

要清晰区分**负载均衡器（Load Balancer）**与**推理路由器（Model Router）**：前者在同一模型的多个副本之间按健康状态与前缀亲和性分流；后者则在**不同能力与单价的模型之间**（如 8B 轻量集群 vs 70B 旗舰集群）做实时商业与质量决策。为了绝不因省钱酿成核心业务质量事故，路由决策必须严格遵守五级优先级：

1. **质量红线最高**：核心付费等级或 `critical=True` 的关键任务强制锁定高能力大模型，即使预算逼近红线也记账超支而不静默降级；
2. **故障安全保护**：账本不可读或单价缺失时默认走大模型并上报 `cost_unknown` 告警，拒绝静默降级；
3. **实时延迟与队列门控**：结合各后端实时队列深度与 p95 TTFT 过滤掉注定超时的池子，两端都无法满足 deadline 时果断快速拒绝（HTTP 503），避免把注定被客户端丢弃的请求送进 GPU 白算；
4. **成本封顶与按特征分流**：在满足预算与延迟前提下，根据 Prompt 长度、任务复杂度分类器（如分类、改写、简单提取）将简单请求导向 8B 轻量后端；
5. **受控超时熔断降级**：仅当主后端超时/过载且**剩余请求 deadline 仍大于备选后端 p95 TTFT 加安全余量**、同时余额充足时，才触发一次降级回退。

与路由器紧耦合的是**原子令牌预算中间件**。并发请求蜂拥而至时，若采取“先推理、后扣费”，几十个并发请求同时看到“还剩 0.01 美元”并全部放行，就会瞬间穿透租户配额。标准工业解法是经典的**两阶段原子记账（Reserve → Settle / Release）**：

1. **请求入站（`reserve`）**：在推理执行前，按 `prompt_tokens + max_tokens` 的**最坏成本**在共享账本（生产通常由 Redis + Lua 脚本实现原子 CAS）中先锁定额度；超限则直接按策略执行拒绝（429）、截断 `max_tokens` 或降级小模型；
2. **流式完成（`settle`）**：拿到推理引擎返回的真实 `usage`（含客户端中途断开时服务端已生成的实际 token）后，按实耗精准结算，自动退还 `预留 − 实耗` 差额；
3. **前置失败（`release`）**：若请求在尚未产生任何 token 前即被上游拦截或连接失败，全额释放预留额度。

<figure class="book-figure">
  <img src="../../../assets/figures/router-budget-flow.svg" alt="预算网关与多后端智能路由全链路：请求先经预算网关基于租户配额账本原子 reserve 最大预算，再由路由器按输入长度、任务复杂度和队列深度分流至 8B 或 70B 模型并带超时熔断降级，响应完成后回流精准 settle 退还多余差额" loading="lazy">
  <figcaption><span class="book-figure__label">图 11-3</span>按 ①→②→③→④ 顺时针阅读全链路：入口预算网关先依托 Redis/Lua 账本原子 <code>reserve</code> 锁住最坏开销，智能路由器综合长度、复杂度与队列分流并保留 70B→8B 超时熔断降级，最终在响应流式结束后按真实 usage 回到网关 <code>settle</code> 多退少补。结论：路由管“走哪更划算”，原子账本管“并发不透支”。</figcaption>
</figure>

下面这段完整的纯 Python 核心实现浓缩了**多后端路由决策、受控超时回退与原子 `reserve / settle / release` 预算账本**的关键结构：

```python
from dataclasses import dataclass
import threading
import time
from typing import Callable


@dataclass(frozen=True)
class BackendSpec:
    name: str
    c_in_per_m: float     # 美元 / 百万输入 token
    c_out_per_m: float    # 美元 / 百万输出 token
    ttft_p95_s: float     # 当前负载下的 p95 TTFT（秒）
    queue_depth: int      # 当前实时排队深度


@dataclass(frozen=True)
class InferenceRequest:
    tenant_id: str
    task: str
    prompt_tokens: int
    max_tokens: int
    ttft_budget_s: float
    deadline_ts: float
    critical: bool = False
    quality_flexible: bool = True


@dataclass(frozen=True)
class BudgetHold:
    tenant_id: str
    reserved_usd: float
    backend_name: str


class BudgetError(Exception):
    """预算超限或账本异常基类。"""


class ServiceRejected(Exception):
    """延迟不可达或过载快速拒绝（对应 HTTP 503 / 429）。"""


SIMPLE_TASKS = frozenset({"classify", "rewrite", "extract", "summarize_short"})


def estimate_cost_usd(be: BackendSpec, in_tokens: int, out_tokens: int) -> float:
    return (in_tokens * be.c_in_per_m + out_tokens * be.c_out_per_m) / 1_000_000.0


class BudgetLedger:
    """线程安全的两阶段原子预算账本（生产环境以 Redis + Lua 替换本地锁）。"""

    def __init__(self, per_request_cap: float, daily_cap: float) -> None:
        self.per_request_cap = per_request_cap
        self.daily_cap = daily_cap
        self._lock = threading.Lock()
        self._spent: dict[str, float] = {}

    def remaining(self, tenant_id: str) -> float:
        with self._lock:
            return self.daily_cap - self._spent.get(tenant_id, 0.0)

    def reserve(self, req: InferenceRequest, be: BackendSpec) -> BudgetHold:
        worst_usd = estimate_cost_usd(be, req.prompt_tokens, req.max_tokens)
        if not req.critical and worst_usd > self.per_request_cap:
            raise BudgetError(f"per_request_cap_exceeded:{worst_usd:.6f}")

        with self._lock:
            used = self._spent.get(req.tenant_id, 0.0)
            if not req.critical and used + worst_usd > self.daily_cap:
                raise BudgetError(f"daily_cap_exhausted:{used:.6f}")
            self._spent[req.tenant_id] = used + worst_usd

        return BudgetHold(
            tenant_id=req.tenant_id,
            reserved_usd=worst_usd,
            backend_name=be.name,
        )

    def settle(
        self,
        hold: BudgetHold,
        actual_in: int,
        actual_out: int,
        be: BackendSpec,
    ) -> float:
        actual_usd = estimate_cost_usd(be, actual_in, actual_out)
        delta = actual_usd - hold.reserved_usd
        with self._lock:
            self._spent[hold.tenant_id] = max(
                0.0, self._spent.get(hold.tenant_id, 0.0) + delta
            )
        return actual_usd

    def release(self, hold: BudgetHold) -> None:
        with self._lock:
            self._spent[hold.tenant_id] = max(
                0.0, self._spent.get(hold.tenant_id, 0.0) - hold.reserved_usd
            )


def choose_backend(
    req: InferenceRequest,
    small: BackendSpec,
    large: BackendSpec,
    ledger: BudgetLedger,
    max_queue_threshold: int = 32,
) -> tuple[BackendSpec, str]:
    """按 质量红线 -> 延迟与队列 -> 成本上限 -> 任务特征 优先级选择后端。"""
    if req.critical:
        return large, "quality_floor"

    small_ok = (
        small.ttft_p95_s <= req.ttft_budget_s
        and small.queue_depth < max_queue_threshold
    )
    large_ok = (
        large.ttft_p95_s <= req.ttft_budget_s
        and large.queue_depth < max_queue_threshold
    )

    if not large_ok:
        if req.quality_flexible and small_ok:
            return small, "latency_or_queue_shed"
        raise ServiceRejected("no_backend_meets_latency_slo")

    rem = ledger.remaining(req.tenant_id)
    large_worst = estimate_cost_usd(large, req.prompt_tokens, req.max_tokens)
    small_worst = estimate_cost_usd(small, req.prompt_tokens, req.max_tokens)

    if large_worst > min(rem, ledger.per_request_cap):
        if req.quality_flexible and small_worst <= min(rem, ledger.per_request_cap) and small_ok:
            return small, "budget_downgrade"
        raise BudgetError("insufficient_budget_for_large")

    if (
        req.quality_flexible
        and small_ok
        and req.task in SIMPLE_TASKS
        and req.prompt_tokens <= 2048
    ):
        return small, "simple_task_short_prompt"

    return large, "default_large"


def dispatch_with_budget(
    req: InferenceRequest,
    small: BackendSpec,
    large: BackendSpec,
    ledger: BudgetLedger,
    invoke: Callable[[BackendSpec, InferenceRequest, float], tuple[str, int, int]],
) -> tuple[str, str]:
    """完整闭环：路由 -> 原子预留 -> 执行与受控超时降级 -> 精准结算/释放。"""
    primary, reason = choose_backend(req, small, large, ledger)
    hold = ledger.reserve(req, primary)

    try:
        timeout_s = max(0.05, req.deadline_ts - time.monotonic())
        text, actual_in, actual_out = invoke(primary, req, timeout_s)
        ledger.settle(hold, actual_in, actual_out, primary)
        return text, reason
    except TimeoutError:
        ledger.release(hold)
        # 仅在允许降级、原走大模型且剩余时间足以完成小模型首字时触发一次降级回退
        remaining_s = req.deadline_ts - time.monotonic()
        if (
            primary.name == large.name
            and req.quality_flexible
            and remaining_s > small.ttft_p95_s + 0.05
        ):
            fb_hold = ledger.reserve(req, small)
            try:
                text, actual_in, actual_out = invoke(small, req, remaining_s)
                ledger.settle(fb_hold, actual_in, actual_out, small)
                return text, f"{reason}+fallback_small"
            except Exception:
                ledger.release(fb_hold)
                raise
        raise
    except Exception:
        ledger.release(hold)
        raise
```

最后，将生产网关与账本最容易踩坑的六种异常场景收口为标准应对策略表：

| 异常与边界场景 | 检测信号 | 推荐工程动作 |
|---|---|---|
| **流式输出中途客户端断连** | SSE 连接关闭但引擎已生成 $k$ 个 token | 绝不全额 `release`，必须按服务端已实际吐出的 $k$ 个 token 调用 `settle` 计费 |
| **前端重试（Retry）** | 客户端或网关发起重试请求 | 每次重试独立执行 `reserve` 与 `settle`；漏记重试会导致成本账本系统性低估云账单 |
| **价格表缺失或 `usage` 字段丢失** | 路由/中间件查不到单价或响应缺 token 统计 | 严禁按 `0` 记账！按已知最高单价保守计费并打 `cost_unknown` 告警，非关键请求降级 |
| **前缀缓存命中带来的成本差** | 服务端实际跳过 Prefill，但网关仍按全价预留 | 记账口径需与计费策略统一（按全价预留或按引擎回传的 `cached_tokens` 折算结算），并在月度对账说明 |
| **账本存储（Redis）不可用** | 锁超时或连接拒绝 | 触发熔断降级策略：非关键流量限流拒绝，关键请求带 `ledger_degraded` 标记放行并异步补账 |
| **账本统计与云账单偏差** | 月度对账偏差 $> 5\%$ | 按顺序核查：漏记重试与断连、分词器口径差、合同预留价 vs 按需价、以及磁盘/网络/负载均衡等非 GPU 成本 |

## 小结

本章完成了从“单机推理引擎”到“生产级集群与商业闭环”的最后一块拼图：

- **PD 解耦**用可控的 NVLink / RDMA KV 传输税，换取 Prefill 与 Decode 彻底的尾延迟隔离与异构硬件配比；两池实例数绝非拍脑袋的静态比例，而由输入输出比与稳态 Batch 摊销共同决定；
- **Kubernetes 弹性扩缩容**彻底告别失效的 CPU/GPU 利用率，转而以引擎原生的 `num_requests_waiting` 与 KV 水位经 KEDA 驱动，用短时超配快速排空积压、用长冷却窗口守护稳态；
- **单位 Token 经济学与智能路由**揭示了输出单价 3–5 倍于输入的物理根源，并以质量红线优先的多后端路由器叠加 `reserve / settle / release` 原子账本，守住线上服务的 SLA 与财务红线。

至此，本书第二篇“推理引擎与部署优化”全部收官。从下一篇开始，我们将潜入推理引擎之下，直面 GPU 硬件与底层算子的世界——[第 12 章](chapter-12.md)将从 GPU 执行模型出发，一步步手推 GEMM 的四版性能阶梯。
