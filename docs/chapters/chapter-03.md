# 第 3 章 推理度量与压测方法 { .chapter-title }

<p class="chapter-summary">把“快”拆成 TTFT、ITL/TPOT、吞吐与排队，掌握分位数分布、SSE 计时原理与拐点读法。</p>

前两章在纸面上算出了理论下界与模型账本，但服务真正上线后，用户和业务方只会问一个朴素的问题：“模型现在快不快，能扛多少并发？”

如果只给一个笼统的“平均响应 1.2 秒”，几乎等于什么都没说——两台同样平均 1.2 秒的服务，一台可能让用户干等 1 秒才吐出第一个字、随后飞快刷完，另一台可能 80 毫秒就吐出首字、但每吐十个字就莫名卡死半秒。两者的根因与调优动作南辕北辙。本章建立 LLM 推理的统一测量标尺：四个核心指标、一套尾部统计方法、一段流式压测核心代码，以及从并发曲线读出拐点与容量的完整方法。

## 3.1 从一条流式时间线读出四个数

把一条流式推理请求沿时间轴展开，才能看清时间究竟花在了哪几个串联阶段。

设客户端在 $t = 0$ 发出请求：请求在引擎队列里等待了 **30 ms**，随后经历 Prefill 与首个 token 采样耗时 **120 ms**（暂忽略网络回传），在 **$t = 150\text{ ms}$** 收到第 1 个内容 token；之后每隔 **25 ms** 收到 1 个新 token，共输出 $G = 100$ 个 token。

从这条时间线直接读出全部关键数字：

- **首 token 延迟（TTFT）**：$150\text{ ms}$（含排队 30 ms + Prefill 与首采样 120 ms）；
- **token 间延迟（ITL）**：相邻两个输出 token 的间隔均为 $25\text{ ms}$；
- **端到端总延迟**：$150 + (100 - 1) \times 25 = 2,625\text{ ms}$；
- **单请求端到端吞吐**：$100 / 2.625 \approx 38.1\text{ tok/s}$（仅看解码生成段速率则是 $1 / 0.025 = 40.0\text{ tok/s}$）。

<figure class="book-figure">
  <img src="../../../assets/figures/timeline.svg" alt="一次流式请求的时间线：排队、prefill、TTFT 与 ITL" loading="lazy">
  <figcaption><span class="book-figure__label">图 3-1</span>读图顺序：先看左半段，排队与 Prefill 串联组成 TTFT；再看右半段，Decode 阶段逐个 token 吐出，相邻到达间隔即 ITL。</figcaption>
</figure>

把单条时间线推广到整台服务，度量 LLM 推理性能必须同时盯着**四个正交指标**：

| 指标 | 英文缩写 | 定义与测量边界 | 常用单位 | 回答的核心工程问题 |
|---|---|---|---|---|
| 首 token 延迟 | **TTFT** | 从客户端发出请求到收到首个非空内容 chunk（包含排队、Prefill 与网络） | ms / s | 用户等多久才看到第一个字 |
| token 间延迟 | **ITL / TPOT** | ITL 为相邻输出 token 的逐次间隔；TPOT 为解码总跨度除以 $G - 1$ | ms | 流式吐字是否丝滑、有没有明显卡顿 |
| 系统吞吐 | **Throughput** | 测量窗口内成功请求的输出 token 总数除以窗口时长（另报 req/s） | output tok/s, req/s | 整张卡单位时间能产出多少内容 |
| 排队深度 | **Queue Depth** | 服务端等待调度的请求数（`waiting`）与正在批内执行的请求数（`running`） | 个（requests） | 系统离饱和与过载还有多远 |

### ITL 与 TPOT 的本质区分

工程报告中最容易混淆的是 **ITL** 和 **TPOT**。假设一条请求输出 $G$ 个 token，首个内容在 $t_{\text{first}}$ 到达，末个内容在 $t_{\text{last}}$ 到达，则：

$$
\text{TPOT} = \frac{t_{\text{last}} - t_{\text{first}}}{G - 1} \quad (G > 1)
$$

注意两点边界：**第一，TPOT 是单请求内部的平均解码步长，而 ITL 是每一个独立间隔的样本集合**——TPOT 会把中途某次长达数百毫秒的卡顿彻底抹平；**第二，当 $G = 1$（只输出一个 token）或只收到 1 个内容 chunk 时，没有相邻间隔，TPOT 必须记为未定义（`null`），绝不能填 `0`**，否则会把均值和分位数严重拉低。

此外，服务端吐出末个内容 token 后，往往还要附带 `usage` 统计包并关闭流，因此总延迟 $t_{\text{total}}$ 通常略大于 $t_{\text{last}}$，计算 TPOT 切勿用 $t_{\text{total}}$ 替代 $t_{\text{last}}$。

### 吞吐口径与 Little 定律边界

衡量整机吞吐时，必须把 **output tok/s**（生成速率）与 **req/s**（请求完成率）分开汇报，并将**输入吞吐与输出吞吐分开**——由于 Prefill 和 Decode 处于 Roofline 的不同区间，把输入 token 和输出 token 混成一个总 token/s 会掩盖真实瓶颈。窗口吞吐的分子只计**成功请求**的输出 token，但分母必须是**包含失败与超时在内的完整墙钟时长**。

在稳定运行的稳态系统中，系统平均在途请求数 $L$、请求完成率 $\lambda$（req/s）与平均端到端停留时间 $W$（s）满足经典的 **Little 定律**：

$$
L = \lambda W
$$

这条公式极适合做数据一致性体检，但有两个严格前提：$L$ 必须是**时间平均**在途数（有限批次压测含冷启动和排空阶段，实际平均在途往往小于配置并发上限 $C$），且 $W$ 必须是**算术平均**延迟而非中位数 p50。关于稳态队列与容量建模，第 7 章会进一步展开。

---

## 3.2 为什么平均值会撒谎：长尾分布与分位数

推理服务的延迟天生呈现**右偏长尾分布**：绝大多数请求很快，但只要有少数请求撞上长 Prefill 抢占、KV 换出或网络抖动，就会拖出长长的尾巴。只看均值几乎必然低估真实用户体验。

### 线性插值分位数（全书统一口径）

将 $n$ 个升序排列的样本记为 $x_0 \le x_1 \le \dots \le x_{n-1}$，对于分位点 $q \in [0, 1]$（如 p95 取 $q = 0.95$，p99 取 $q = 0.99$），本书及主流科学计算库（NumPy、PyTorch 默认口径）采用**线性插值法**定义秩 $r$ 与分位数 $\hat{x}_q$：

$$
r = (n - 1) \cdot q, \qquad \hat{x}_q = x_{\lfloor r \rfloor} + \bigl(r - \lfloor r \rfloor\bigr)\bigl(x_{\lceil r \rceil} - x_{\lfloor r \rfloor}\bigr)
$$

工程中还存在另一种 **nearest-rank（最近秩）**算法，取第 $\lceil n q \rceil$ 个样本。小样本下两种算法算出的 p99 可能直接差出一个极端值，因此压测报告务必标明所用分位数定义。

### 两个直击痛点的算例

**算例一：均值与 p99 完全相同，却藏住 2 倍长尾。** 对比两套各跑了 $n = 100$ 条请求的服务：

- **系统 A**：99 条请求的 TTFT 是 $100\text{ ms}$，最慢的 1 条是 **$200\text{ ms}$**；
- **系统 B**：全部 100 条请求的 TTFT 都是 **$101\text{ ms}$**。

算一下两套系统的指标：
- 均值：系统 A 为 $(99 \times 100 + 200) / 100 = 101\text{ ms}$，系统 B 也是 $101\text{ ms}$；
- 线性插值 p99：秩 $r = (100 - 1) \times 0.99 = 98.01$，落在索引 98（100 ms）与索引 99（200 ms）之间，系统 A 得出 $100 + 0.01 \times (200 - 100) = \mathbf{101\text{ ms}}$，系统 B 同样是 **$101\text{ ms}$**！

均值和 p99 竟然分毫不差，但系统 A 最倒霉那 1% 的用户多等了整整一倍时间。**想要看见真正的长尾，报告里必须并列给出 p50、p95、p99 与 max（最大值）。**

**算例二：TPOT 把一次 300 ms 的明显卡顿彻底抹平。** 一条输出 101 个 token 的请求包含 100 个 ITL 间隔：99 个间隔平稳落在 $25\text{ ms}$，但中间因为插队执行了一次长 Prefill，某一个间隔突然飙到 **$300\text{ ms}$**。如果只看平均步长：

$$
\text{TPOT} = \frac{99 \times 25 + 300}{100} = 27.75\text{ ms}
$$

纸面上看不过比 $25\text{ ms}$ 慢了 11%，仿佛毫无异常；可坐在屏幕前的用户却清清楚楚看到文字突然冻住 0.3 秒。这就是为什么监控流式体验必须把 **ITL 的 p95 和 max** 单独拎出来看。

!!! tip "核心直觉：小样本里谈 p99 没有统计意义"
    只要样本非空就能套公式算出一个数字，但小样本根本抓不住 1% 尾部。如果真实服务有 1% 概率出现慢请求，只测 100 条独立请求时，一条慢请求都没碰上的概率高达 $(1 - 0.01)^{100} \approx 36.6\%$；若只测 20 条，p95（秩 18.05）和 p99（秩 18.81）全都挤在最后两条样本之间，随机波动极大。因此 20 条请求只适合接口冒烟，正式容量评估每档至少应跑数百至上千请求，并重复运行看中位数与波动区间。

---

## 3.3 流式压测的客户端原理

要把上面四个指标准确采出来，压测客户端必须遵守三条协议与计时细则。

### 为什么必须使用 SSE 流式接口

OpenAI 兼容接口 `/v1/chat/completions` 在默认非流式模式下，服务端会等全部 token 生成完毕才一次性返回 HTTP 响应，客户端只能看到总延迟，TTFT 和 ITL 彻底丢失。因此压测必须同时带上：

- `"stream": True`：启用 Server-Sent Events（SSE）流式推送；
- `"stream_options": {"include_usage": True}`：要求服务端在流末尾附带精确的 `prompt_tokens` 与 `completion_tokens`，避免客户端靠字符数猜 token 造成偏差（中文、代码按 `len(text) // 4` 估算误差极大）。

### 精准打点 TTFT 与识别 chunk 合并陷阱

解析 SSE 文本流时，有三个细节极易写错，导致 TTFT 或 ITL 严重失真：

1. **跳过非内容事件**：SSE 流开头的第一个 JSON 块往往只包含 `{"role": "assistant", "content": ""}`，末尾则有纯 `usage` 块或注释心跳行（以 `:` 开头）。**客户端 TTFT 必须锚定在首个非空 `delta.content` 到达的时刻**，空字符串绝不触发打点（但包含空格的非空输出属于有效内容）。
2. **严格判定流完整**：只有读到标准结束标记 `data: [DONE]` 且至少收到一个内容块，才算请求成功；HTTP 错误、超时或中途截断的流不能把残缺内容计入成功吞吐。
3. **正视 chunk 合并与代理缓冲**：受网关缓冲、TCP 粘包或投机解码单步多输出影响，客户端收到的 1 个 SSE chunk 里可能打包了多个 token。因此客户端观测到的间隔本质上是 **chunk 级 ITL 近似**，绝不能把多 token 的 chunk 强行线性拆分成虚构的逐 token 时间戳。

下面是一段不到 40 行的核心 Python 示意代码，展示单次请求如何用单调时钟 `time.perf_counter()` 准确提取 TTFT、ITL 列表、TPOT 与精确 token 计数：

```python
import json
import time
import urllib.request


def measure_stream(url: str, model: str, prompt: str, max_tokens: int = 128) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    content_times: list[float] = []
    usage_tokens: int | None = None
    done = False

    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:  # 跳过空 role 块与空字符串，仅对非空内容打点
                    content_times.append(time.perf_counter() - start)
            if isinstance(chunk.get("usage"), dict):
                usage_tokens = chunk["usage"].get("completion_tokens")

    total_s = time.perf_counter() - start
    if not done or not content_times:
        raise RuntimeError("incomplete or empty SSE stream")
    ttft_s = content_times[0]
    itl_s = [b - a for a, b in zip(content_times, content_times[1:])]
    tokens = usage_tokens if usage_tokens is not None else len(content_times)
    tpot_s = (content_times[-1] - ttft_s) / (tokens - 1) if tokens > 1 and len(content_times) > 1 else None
    return {"ttft_s": ttft_s, "itl_s": itl_s, "tpot_s": tpot_s, "total_s": total_s, "tokens": tokens}
```

仓库中附带了仅依赖标准库、支持多档并发扫描、多行 SSE 事件、严格错误分类与逐请求 JSON 落盘的完整工具脚本，参见 [bench_openai.py](../modules/part1-foundations/scripts/bench_openai.py)。

---

## 3.4 如何读懂一份基线并发曲线

掌握了指标和测量工具，最后一步是把并发从低到高扫一遍，从表格里亲手读出系统的**性能拐点（Knee）**与 **SLO 有效容量（Goodput）**。

### A100 80GB + Llama-3.1-8B 并发扫描范例

以下是一组在 **NVIDIA A100 80GB SXM** 上运行 **Llama-3.1-8B（FP16）**、采用连续批处理与分块 Prefill、固定负载为 **输入 512 token / 输出 128 token** 时的典型基线扫描数据：

| 并发上限 $C$ | 输出吞吐（tok/s） | 请求吞吐（req/s） | TTFT p50 / p95（ms） | ITL p50 / p95（ms） | 排队深度 `waiting` p95 | 状态判读 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 92 | 0.72 | 52 / 58 | 10.5 / 12.1 | 0 | 单流带宽基线，GPU 算力严重空转 |
| 8 | 620 | 4.84 | 68 / 95 | 12.5 / 18.4 | 0 | 权重复用发力，吞吐增 **6.7×**，延迟几乎持平 |
| 32 | 1,685 | 13.16 | 145 / 290 | 18.0 / 38.0 | 2 | 逼近拐点前甜蜜区，吞吐再增 **2.7×** |
| 64 | 2,240 | 17.50 | 360 / 780 | 26.0 / 68.0 | 14 | 吞吐增幅放缓至 **1.33×**，排队与尾延迟抬头 |
| 128 | 2,395 | 18.71 | 1,250 / 2,480 | 44.0 / 135.0 | 68 | **进入过载区**：吞吐仅微涨 **7%**，TTFT p95 暴涨 **3.2×** |

### 三步定位拐点与有效容量

拿到这样一张表，按固定三步做工程判读：

1. **看吞吐边际增幅**：$C$ 从 1 提到 8，吞吐暴涨 6.7 倍；8 到 32 仍涨 2.7 倍；32 到 64 缩水到 1.33 倍；到了 64 到 128，并发翻倍换来的吞吐增幅却只剩区区 **7%**——说明显卡有效算力与显存带宽已逼近天花板，拐点就在 $C = 32 \sim 64$ 之间。
2. **看四指标联动恶化**：拐点之后，$C = 128$ 档位上新增的请求不再提升执行效率，而是全部积压在 `waiting` 队列（p95 从 14 飙至 68），直接把 TTFT p95 推上 2.48 秒，同时更密集的混批把 ITL p95 拖慢到 135 ms。
3. **用 SLO 划定有效容量**：**最大吞吐绝不等于可用容量。** 假设业务 SLO 要求 **TTFT p95 $\le 1,000\text{ ms}$ 且 ITL p95 $\le 80\text{ ms}$**，那么 $C = 64$（输出 2,240 tok/s）完全达标，而吞吐更高的 $C = 128$（2,395 tok/s）已全面违约——这套单卡部署在当前负载下的有效容量上限落在 64 并发附近。

### 决定压测结果可比性的 5 个核心变量

工程团队之间经常出现“同样 A100 跑 Llama-8B，为什么你的吞吐比我高一倍”的争论，十有八九是以下五个条件没有对齐：

1. **实际输入与输出长度分布**：`max_tokens=128` 只是上限，模型输出 EOS 会提前结束；不同聊天模板也会暗中追加数十个系统 token。务必记录服务端返回的实际 `prompt_tokens` 与 `completion_tokens` 分布。
2. **预热与稳态窗口**：服务刚启动的前十几个请求会触发 CUDA 上下文初始化、JIT 编译、cuDNN autotune 与 CUDA Graph 录制，必须先跑完预热再开始正式计时。
3. **前缀缓存冷热状态**：现代推理引擎默认开启自动前缀缓存（第 5、6 章详解）。用一模一样的 prompt 连压 1000 次，测出的是命中缓存后跳过 Prefill 的“虚假超低 TTFT”；评估真实冷 Prefill 必须在每条请求前注入唯一随机前缀或显式关闭缓存。
4. **并发到达模型**：**闭环压测**（固定 $C$ 个客户端，收到回复再发下一条）天然带背压，系统变慢时会自动降速；**开环压测**（按泊松分布或固定 QPS 持续注入请求）在超过服务能力后队列会无限堆积，更能暴露真实生产雪崩。两者数据不可混比。
5. **采样与输出协议**：关闭或开启流式、温度设置、是否开启多候选 `n > 1` 或结构化输出约束，都会改变采样耗时与实际输出长度。

现在，我们已经能从外部精准量出 TTFT、ITL、吞吐与排队，并定位拐点（更多压测规范与跨配置对照边界详见 [测量口径与对比分析方法](../toolkit/methodology.md)）。但如果发现在 $C = 1$ 时 ITL 实测 10.5 ms 距离理论搬运下界 8.0 ms 还有 2.5 ms 的差距，或者在 $C = 64$ 时某个阶段莫名卡顿，外部压测无法告诉你“机器内部到底是谁在拖后腿”。在 [第 4 章](chapter-04.md)，我们拿起显微镜——Profiling，把一次推理内部的时间线彻底切开。
