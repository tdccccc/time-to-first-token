# 第 5 章 vLLM：一条请求如何穿过推理引擎 { .chapter-title }

<p class="chapter-summary">推理引擎的核心职责不是“把模型跑起来”，而是通过 PagedAttention 消灭 KV 显存碎片、通过连续批处理与分块 Prefill 消除计算空转，把同一张 GPU 的显存与算力稳定转化为高并发吞吐。</p>

把权重加载进 GPU 并生成一段文本，与把它做成能持续承接高并发请求的在线服务，是完全不同的两件事。在 A100 80GB 上运行 FP16 精度的 Llama-3.1-8B 时，模型权重只占 16 GB，留给 KV 缓存（KV Cache）的显存池高达约 56 GB；但早期推理服务往往只能同时接住寥寥数十条请求。瓶颈不在权重，而在 KV 显存的分配方式与批处理调度方式。

本章沿着一条请求从进入 API 到吐出最后一个 token 的完整轨迹，拆解 vLLM V1 的进程结构、PagedAttention 显存账本、迭代级连续批处理、一次 `step()` 的内部运作以及线上部署的核心旋钮与排障逻辑。

---

## 5.1 为什么 `model.generate` 不是推理引擎

### 最小生成循环的四个缺口

概念上最简单的自回归生成循环只有短短几行：

```python
def naive_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    input_ids = tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token = sample(logits[:, -1, :])
        input_ids = append_token(input_ids, next_token)
    return tokenizer.decode(input_ids)
```

哪怕把这段循环包进一个 HTTP 接口，它也远远称不上推理引擎。除了每步重算全量历史 token 之外，面对并发流量时它会立刻暴露四个结构性缺口：

| 核心问题 | 对应职责 | 决定什么 |
|---|---|---|
| 请求由谁接收，文本与采样参数如何转为内部状态、如何流式回传？ | **输入输出管理** | CPU 侧分词/反分词是否阻塞 GPU |
| 哪些请求在排队、哪些正在执行、超时取消时如何回收资源？ | **请求编排** | 生命周期完整性与无泄漏清理 |
| 这一步算哪些请求、每个请求算多少 token（Prefill 还是 Decode）？ | **计算调度** | 吞吐、TTFT 与 TPOT 的平衡 |
| 算出的 KV 放在哪里、能否复用、显存吃紧时如何腾挪？ | **状态与内存管理** | 显存利用率与最大并发驻留数 |

模型本身只负责神经网络前向计算；**推理引擎负责组织这些计算**——决定为谁算、何时算、算多少 token，以及算出的状态存放在哪。

### vLLM V1 双进程架构与 EngineCore 四角色

早期引擎常把 HTTP 处理、分词、调度与模型执行塞在同一个 Python 进程里，导致 CPU 侧的 JSON 解析、Tokenizer 和 Detokenizer 频繁抢占调度循环，在高频 Decode 时制造出大量 GPU 空泡。vLLM V1 将引擎清晰拆分为两个独立进程，通过消息队列异步解耦：

- **API 进程（前端进程）**：面向客户端暴露 OpenAI 兼容接口，承担请求解析、Tokenizer、增量 Detokenizer 与 SSE 流式返回。这些纯 CPU 任务移出主循环后，不再拖慢 GPU 迭代节拍。
- **EngineCore 进程（引擎核心）**：只盯住 GPU 迭代，内部由四个边界清晰的角色协同工作：
  1. **调度器（Scheduler）**：每一步决定选中哪些请求、各分配多少 token 预算。
  2. **KV 管理器（KV Cache Manager）**：维护全局空闲物理块池，按需分配、共享与回收 KV 槽位。
  3. **模型执行器（Model Executor / Worker）**：将本步选中的 token 打包成一个扁平批（batch），驱动 GPU 执行一次前向计算。
  4. **采样器（Sampler）**：根据输出 logits 与各请求的 `temperature`、`top_p` 等参数抽取下一个 token 并写回请求状态。

由于 API 进程与 EngineCore 进程完全解耦，CPU 处理上一轮输出文本时，GPU 已经在跑下一轮前向。vLLM V1 相比 V0 在单机场景下显著的吞吐提升，很大一部分正是来自这种 CPU/GPU 重叠与调度开销压降，而并非底层 Attention Kernel 本身的变化。

<figure class="book-figure">
  <img src="../../../assets/figures/vllm-request-journey.svg" alt="vLLM V1 请求旅程：API 进程负责分词与流式返回，EngineCore 内由调度器、KV 管理器、模型执行器与采样器协同驱动状态机流转" loading="lazy">
  <figcaption><span class="book-figure__label">图 5-1</span>沿着箭头从左到右看一条请求如何穿越双进程边界：API 进程隔离 CPU 侧编解码开销，EngineCore 按 step 节拍驱动调度、KV 分配、前向计算与采样，并在满足停止条件时归还全部物理块。</figcaption>
</figure>

### 请求状态机与双路径视角

EngineCore 的基本时间节拍是**一步（step）**：每循环一次，执行一次模型前向，产出一批 token。一条请求在引擎内部沿状态机流转：

1. **WAITING**：请求经 API 进程分词后，携带采样参数与长度上限进入调度器的等待队列。
2. **RUNNING（Prefill → Decode）**：当 token 预算与空闲 KV 块同时满足时，请求被准入。首步（或分块后的若干步）完成 Prompt 前向并采样出第一个输出 token——**TTFT 到此定格**，涵盖了前端处理、排队与 Prefill 全部耗时；随后请求留在 `RUNNING` 队列，每步 Decode 一个新 token，经 API 进程流式推送给用户。
3. **FINISHED**：遇到 `EOS`、命中停止词、达到 `max_tokens` 或客户端主动取消，请求退出并立刻将占用的物理块全部归还空闲池。
4. **PREEMPTED（抢占）**：当已准入的请求持续生成导致 KV 池耗尽时，调度器会挑出部分 `RUNNING` 请求暂停执行、回收其 KV 块并退回 `WAITING` 队列，待显存宽裕时再重新计算。

理解推理引擎的演进，还可以把一条请求的旅程拆成**控制路径**与**数据路径**：

- **控制路径**传递请求状态与调度决策（谁在等、谁进批、分几个 token、是否抢占），流经 API 进程与调度器。
- **数据路径**传递张量与显存数据（`input_ids`、位置编码、块表、logits 与 KV 张量），流经模型执行器与 Attention 后端。

!!! tip "核心直觉"
    此后学到的每一项引擎机制，都可以用“改了哪条路径”来快速定位：PagedAttention 既重构了数据路径的 KV 存放形态，也改写了控制路径的显存准入账；连续批处理重塑了控制路径的组批边界；第 6 章的前缀缓存则同时改写数据路径的 KV 复用与控制路径的请求排序。

---

## 5.2 KV 显存账与 PagedAttention

### 一张 A100 80GB 上的 KV 显存账

一条请求的显存占用分为两笔：**模型权重**由全部并发请求共享，只加载一份；**KV 缓存**则随请求数量与上下文长度线性增长。高并发或长上下文下，先撞上显存墙的几乎总是 KV 缓存。

对于标准多头注意力（MHA）或分组查询注意力（GQA），单个 token 在所有层占用的 KV 字节数为：

$$
\text{KV}_{\text{per\_token}} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times d_{\text{head}} \times b_{\text{elem}}
$$

其中系数 $2$ 对应 K 和 V 两份张量，$b_{\text{elem}}$ 为单元素字节数（FP16/BF16 为 $2\text{ B}$）。对比三款代表性模型：

| 模型 | 架构与注意力几何 | FP16/BF16 权重 | 每 token KV | 2K 上下文 KV | 8K 上下文 KV | 32K 上下文 KV |
|---|---|---:|---:|---:|---:|---:|
| **Llama-3.1-8B** | 32 层，GQA（32Q / 8KV），$d=128$ | 16.0 GB | **128 KiB** | 256 MiB | **1.00 GiB** | 4.00 GiB |
| **Qwen3-8B** | 36 层，GQA（32Q / 8KV），$d=128$ | 16.38 GB | **144 KiB** | 288 MiB | **1.125 GiB** | 4.50 GiB |
| **LLaMA-13B**（经典对照） | 40 层，MHA（40Q / 40KV），$d=128$ | 26.0 GB | **800 KiB** | 1.56 GiB（1.68 GB） | **6.25 GiB** | 25.0 GiB |

以 **A100 80GB SXM + Llama-3.1-8B（FP16）** 为例：80 GB 显存扣除 16 GB 权重及约 8 GB 激活工作区、CUDA Graph 捕获与框架预留余量后，留给 KV 缓存的显存池约为 **56 GB（约 52.2 GiB）**。账面上看起来空间充裕——每条 8K 上下文请求仅需 1 GiB，足以装下超过 50 条；每条 500 token 的短对话仅需 62.5 MiB，足以装下数百条。

### 预留浪费与外部碎片：旧系统为什么装不下

既然账面空间充裕，为什么早期推理系统却装不下多少请求？根源不在 KV 理论体积，而在**连续整段预留**的分配方式：

1. **内部浪费（过度预留）**：生成长度事先未知，旧系统只能在请求到达时，按模型最大长度 $L_{\max}$ 一次性划走一整段连续显存。若 $L_{\max}=2048$，实际长度只有 512 token，就有 $(2048 - 512)/2048 = 75\%$ 的显存从分配到释放从未写入过任何数据；实际长度 400 token 时浪费高达 $80.5\%$。真实在线负载上，这类静态预留往往导致 **60%–80%** 的 KV 显存被白白闲置。
2. **外部碎片**：不同请求到达和结束的时间错落不一，反复分配与释放后，空闲显存被切成大量互不相连的小空洞。即使剩余空闲总量超过 20 GB，只要找不到一整块连续的 4 GiB 区域，一条新的 32K 请求依然无法准入。

简言之，过度预留是**“占着不用”**，外部碎片是**“有空用不上”**。

### PagedAttention：逻辑块、块表与非连续物理页

vLLM 借用操作系统虚拟内存的分页思想提出了 **PagedAttention**：把每个序列的 KV 缓存切分成固定大小的**逻辑块（logical block）**，默认每块容纳 $B = 16$ 个 token；逻辑上连续的块，通过该请求专属的**块表（block table）**映射到显存中任意位置的**物理页（physical block）**。最关键的变化是**按需分配**：请求到达时只为当前 Prompt 分配必需的物理块，Decode 阶段每写满一块才向空闲池申请下一块，彻底消除预先划地。

<figure class="book-figure">
  <img src="../../../assets/figures/kv-pages.svg" alt="PagedAttention 映射链：连续逻辑块 L0 至 L5 经每请求一份的块表翻译为显存中分散的物理页 P7、P2、P5、P0、P1、P4" loading="lazy">
  <figcaption><span class="book-figure__label">图 5-2</span>读图顺序为“上（逻辑块）→ 左下（块表）→ 右下（物理页）”：模型与调度器看到的是连续逻辑序列 L0–L5，显存里它们散落在不同物理页上；外部碎片彻底消失，内部浪费被压到最后一个未满页。</figcaption>
</figure>

#### 浪费上界与块大小折中

设序列实际长度为 $L$、块大小为 $B = 16$。所需物理块数为 $\lceil L / 16 \rceil$，内部浪费的槽位数为：

$$
\text{Waste}(L) = 16 \cdot \left\lceil \frac{L}{16} \right\rceil - L \le 15\text{ tokens}
$$

前面所有块都是 100% 写满的，只有最后一个块可能未满，因此**浪费的严格上界不到一个块（最多 15 个 token）**。带入几组典型长度验算：

- $L = 500$：需 $\lceil 500 / 16 \rceil = 32$ 块（512 槽位），浪费 12 个 token，占比 $12/512 = 2.34\%$；若末块填充均匀分布，平均浪费 $7.5$ 个 token，占比仅 $1.46\%$。
- $L = 2048$：恰好 128 块，浪费为 $0$。
- $L = 100$：需 7 块（112 槽位），浪费 12 个 token，占比 $10.71\%$。可见分页承诺的不是“任何请求都低于 4%”，而是**上界恒为一个块**——序列越长，浪费占比越小。

为什么不把块大小设成 $B = 1$ 来彻底清零内部浪费？这是一项典型的工程折中：块越小，块表条数越多、内存分配与寻址查表越频繁，且 Attention Kernel 每次读取的连续显存片段越短，GPU 访存合并（coalescing）效率越差；反之，若把块放大到 $B = 64$，$L = 100$ 时需 2 块（128 槽位），浪费 28 个 token（$21.9\%$），直接比 $B = 16$ 翻倍。**$B = 16$ 是在低浪费上界、小块表开销与良好 GPU 访存连续性之间取得的工程平衡点。**

#### 分页前后的并发容量跃升

把这笔账放到整卡批并发的尺度上，差距立现：

- **A100 80GB + Llama-3.1-8B（56 GB / 约 52.2 GiB KV 池）**：
  - 若旧系统按最大上下文 `32K`（每条 4 GiB）静态预留，整张卡只能同时接纳 $\lfloor 52.2 / 4 \rfloor = \mathbf{13}$ 条请求；即便把上限收紧到 `8K`（每条 1 GiB），也最多只能接纳 **52** 条。
  - 启用 PagedAttention 后，若线上请求实际平均长度为 **2K token**（256 MiB = 0.25 GiB），KV 池可同时驻留约 **208** 条请求（相对 32K 预留提升 **16 倍**）；若平均为 **500 token**（对齐到 512 槽位 = 64 MiB），显存足以容纳超过 **800** 条请求，并发上限转而由调度器的 `--max-num-seqs` 控制。
- **A100 80GB + LLaMA-13B（经典对照，80 GB 扣 26 GB 权重剩 54 GB KV 池）**：
  - 按 `2048` 最大长度预留（每条约 1.68 GB），最多并行 $\lfloor 54 / 1.68 \rfloor \approx \mathbf{32}$ 条请求。
  - 分页后按平均 500 token（对齐 512 槽位，约 419 MB）算，可并行 $\lfloor 54{,}000 / 419 \rfloor \approx \mathbf{128}$ 条请求，并发提升 **4 倍**。

!!! warning "常见误区：分页 ≠ 把 KV Swap 到 CPU"
    PagedAttention 只借用了操作系统虚拟内存的**非连续分页映射**，物理块依然全部位于 GPU HBM 显存内。它解决的是“显存利用率与碎片问题”，并不会把 80 GB 显存凭空变大。把 KV 换出到主机 CPU 内存（Swap）是另一套抢占回退机制，两者不可混为一谈。

---

## 5.3 连续批处理与分块 Prefill

分页解决了“显存里能同时装下多少请求”，但它本身不会让 GPU 在请求之间停止空转。如果调度器仍然按整批进出组织计算，省下来的显存依然换不成吞吐。

### 静态批处理的槽位空转账

传统深度学习服务采用**请求级静态批处理（static batching）**：把若干条请求凑成固定大小的 batch 一起送进 GPU，直到最慢的那条请求生成完毕，整批才能释放并换入下一批。

假设同一批里有 4 条请求，输出长度分别为 **10、50、100、200** 个 token：

- 整批必须跑满 200 步，总共占用 $4 \times 200 = 800$ 个“槽位·步”。
- 第 1 条请求在第 10 步就已生成完毕，却必须占着空槽等到第 200 步；4 条请求的空转总量高达：

    $$
    (200 - 10) + (200 - 50) + (200 - 100) + (200 - 200) = 440\text{ 槽位·步}
    $$

- 整批足足有 **$440 / 800 = 55\%$** 的批处理槽位在白白空转；更糟的是，原本 10 步就能返回的短请求被硬生生拖到第 200 步，等待时间膨胀了 **20 倍**。

### 迭代级连续批处理：解耦延迟与吞吐

受 Orca 提出的**迭代级调度（iteration-level scheduling）**启发，vLLM 采用**连续批处理（continuous batching）**：把调度粒度从“一整批请求”缩小到**单次前向迭代（one step）**。每跑完一步，遇到停止条件的请求立刻退出并返回，空出的槽位在下一步马上补入等待队列中的新请求。

<figure class="book-figure">
  <img src="../../../assets/figures/batching.svg" alt="静态批处理与连续批处理时间线对比：静态批处理中 A、B 结束后留下大量空槽等待 C；连续批处理在 A、B、C 结束瞬间分别换入 D、G、E、F" loading="lazy">
  <figcaption><span class="book-figure__label">图 5-3</span>上下对比两条时间线：静态批处理中槽位被最长请求 C 锁死；连续批处理做到“结束即换入”（A→D→G、B→E、C→F），显著提升吞吐并消除短请求的尾部拖累。</figcaption>
</figure>

回到刚才 10 / 50 / 100 / 200 的例子：连续批处理下，10 个 token 的请求第 10 步立刻返回，50 的第 50 步返回，100 的第 100 步返回。无论当前 batch 同时跑着 4 条还是 40 条请求，短请求的延迟只取决于自身长度，而系统总吞吐则由 batch 规模决定——**单请求延迟与系统吞吐由此解耦**。

落实到调度器里，连续批处理每一步都在权衡三条硬约束：

1. **Token 预算（`max_num_batched_tokens`）**：控制单步前向的计算量，防止一步算太久把 TPOT 顶穿。
2. **空闲物理块数**：保障本步所有待写入的 KV 都有落脚处。
3. **最大并发序列数（`max_num_seqs`）**：限制单步同时追踪的活跃请求数上限。

!!! tip "核心直觉：分页与连续批处理互为前提"
    如果没有分页，新请求想随时补位就必须找到一整块连续的大显存，只能干等老请求整段释放，连续批处理在显存上根本站不住；反之，如果没有连续批处理，分页腾出来的数倍显存也只能呆在空闲池里，无法随时转化为有效吞吐。按需分配与迭代级补位，本质上是同一枚硬币的两面。

### vLLM V1 的统一调度表示与 Chunked Prefill

在早期引擎实现中，Prefill 批与 Decode 批往往走两套割裂的代码路径：一旦插入一个 8,192 token 的长 Prompt Prefill，整张 GPU 会被这一步算力密集的前向独占几十到上百毫秒，所有正在流式吐字的 Decode 请求被迫卡顿，造成剧烈的 ITL 抖动。

vLLM V1 做了一项极其优雅的抽象统一：调度器不再区分“这一步是 Prefill 批还是 Decode 批”，而是统一输出一张字典：

```text
{ req_id: num_scheduled_tokens }
```

无论是 Prompt 里的输入 token 还是 Decode 生成的新 token，在调度器眼里都只是“本步要为该请求推进多少个 token”。在这一统一表示之上，**分块 Prefill（Chunked Prefill）**成为默认开启的自然能力：

- 设单步总预算 $B = 2048$ tokens，当前有 64 条活跃 Decode 请求。
- 调度器**优先保障 `RUNNING` 队列的 Decode**，为 64 条请求各分配 1 个 token（合计 64 token，仅占预算的 $3.125\%$）。
- 剩余 $2048 - 64 = 1984$ 个 token 预算全部切给等待队列里的长 Prompt；一条 8,192 token 的长输入会被自动拆成 $\lceil 8192 / 1984 \rceil = 5$ 个步长平滑消化。

这样一来，Decode 既不会被长 Prompt 堵死，还能“搭便车”借着 Prefill Chunk 把 GPU Tensor Core 算力填满，大幅拉高整机算力利用率。同时，vLLM V1 也默认开启了基于块哈希的**自动前缀缓存（Automatic Prefix Caching, APC）**：共享系统提示词的请求可以直接复用已对齐的 KV 块。块粒度哈希与 token 粒度基数树的深度对比，将在[第 6 章](chapter-06.md)展开。

---

## 5.4 一次 `step()` 的内部运作

前面三节分别立住了 KV 显存、分页映射与连续批处理三笔账；本节进入 EngineCore 的单步循环，看这三件事如何在代码层面咬合在一起。

### 调度器主循环与准入算术

下面这段精简伪代码勾勒出 vLLM V1 一次 `step()` 中**“先 `RUNNING`、后 `WAITING`，先算预算、再要物理块”**的核心决策流：

```python
def schedule_step(running, waiting, kv_manager, max_num_batched_tokens, max_num_seqs):
    scheduled_tokens = {}
    token_budget = max_num_batched_tokens

    # 1. 优先保障已在跑的 Decode 请求，稳定 token 间延迟（ITL）
    idx = 0
    while idx < len(running) and token_budget > 0:
        req = running[idx]
        num_new = min(req.num_decode_tokens(), token_budget)
        if not kv_manager.allocate_slots(req, num_new):
            # 物理块不足：抢占队尾优先级最低的请求并回收其全部 KV 块
            victim = running.pop()
            kv_manager.free(victim)
            victim.reset_for_recompute()
            waiting.appendleft(victim)
            if victim is req:
                break
            continue
        scheduled_tokens[req.req_id] = num_new
        token_budget -= num_new
        idx += 1

    # 2. 用剩余 token 预算接纳等待队列中的 Prefill（支持自动分块）
    while waiting and token_budget > 0 and len(running) < max_num_seqs:
        req = waiting[0]
        num_new = min(req.remaining_prefill_tokens(), token_budget)
        if not kv_manager.allocate_slots(req, num_new):
            break
        waiting.popleft()
        running.append(req)
        scheduled_tokens[req.req_id] = num_new
        token_budget -= num_new

    return scheduled_tokens
```

注意调度器与 KV 管理器的清晰职责边界：**调度器做决策，KV 管理器记账。** 调度器只关心“能不能给这个请求批下 $n$ 个 token 的槽位”，不关心分到的是哪几个物理页号；KV 管理器只管空闲块池与分配记账，不关心队列优先级。

准入判断的算术非常直观：

- **Prefill 阶段是一次性块申请**：一条 1,000 token 的 Prompt 需要 $\lceil 1000 / 16 \rceil = 63$ 个物理块。若空闲池里还剩 1,000 个块，不考虑分块时这一步最多准入 $\lfloor 1000 / 63 \rfloor = 15$ 条这样的请求（占 945 块，余 55 块不足第 16 条）。长 Prompt 挤压并发的底层机理就在这 63 个块里。
- **Decode 阶段是细水长流的块消耗**：一条在跑的请求每步只追加 1 个 token，只有每跨过 16 个 token 的边界时才需从空闲池新领 1 个块。整个池子处于“结束请求不断归还、活跃请求零星领取”的动态流动中。

### 块表的 $O(1)$ 寻址与写时复制

物理块散落在显存各处，Attention Kernel 如何快速找到任意历史 token 的 K 和 V？答案是纯粹的 $O(1)$ 数组索引——没有任何搜索或哈希：

对于序列中的第 $t$ 个 token（0 起始）和块大小 $B = 16$，其逻辑块号与块内槽位偏移分别为：

$$
\text{logical\_block} = \left\lfloor \frac{t}{16} \right\rfloor, \qquad \text{slot\_offset} = t \bmod 16
$$

验算两个位置：

- 长度 500 的序列最后一个 token（$t = 499$）：逻辑块号 $499 // 16 = 31$（$31 \times 16 = 496$），块内偏移 $499 \bmod 16 = 3$。Kernel 读取 `phys_id = block_table[31]`（例如 `1042`），直接访问物理页 `1042` 的第 3 号槽即可。
- 长度 1000 的序列最后一个 token（$t = 999$）：逻辑块号 $999 // 16 = 62$，块内偏移 $999 \bmod 16 = 7$。首尾 token 走完全一致的公式，没有任何分支特判。

实现上要分清**全局池**与**每请求表**：全局只有一张 `BlockPool` 管物理块归属，每个请求有一张逻辑到物理的映射表（送入 GPU 时打包成二维张量 `block_table`）。池子少一块，某个请求的块表里就必然多一块；两者守恒是内存无泄漏的第一不变量。同时，正因为访存从“线性连续读”变成了“查块表跳读”，PagedAttention 必须配套改写底层 Attention Kernel（如 Paged FlashAttention / FlashInfer）才能在读取 K/V 时实时完成间接寻址。

#### 引用计数与写时复制（Copy-on-Write）

块表还天然解锁了**零拷贝 KV 共享**。在 Parallel Sampling、Beam Search 或共享前缀场景中，多条候选序列拥有完全相同的前缀，它们只需把块表前若干项指向**同一批物理块**，并在物理块元数据上维护**引用计数（`ref_cnt`）**：

- 当某条分支需要往共享块（`ref_cnt > 1`）写入新生成的 token 时，KV 管理器触发**写时复制（Copy-on-Write）**：新分配一块物理页，拷贝已有内容，原块 `ref_cnt -= 1`，随后分支安全写入自己的私有副本，绝不污染其他候选路径。
- 以 Llama-3.1-8B（128 KiB/tok）跑宽度为 4 的 Beam Search、共享 100 token 前缀为例：不共享需存 $4 \times 100 = 400$ token；共享后只存 100 token（引用计数 4），立省 300 token = **37.5 MiB**；按 16-token 块对齐（7 块 = 112 槽位），省下 $3 \times 112 = 336$ 槽位 = **42.0 MiB**。随着前缀拉长与分支增多，这部分显存红利会成倍放大。

### vLLM V1 源码四文件导航

如果想在 vLLM 源码中对照本章机制，抓住下面四个核心文件就能理清整条主线（具体行号随版本演进，但四者的职责边界高度稳定）：

| 模块文件（`vllm/v1/` 相对路径） | 所属角色 | 核心职责与关键入口 |
|---|---|---|
| `core/sched/scheduler.py` | 调度器 | `Scheduler.schedule()`：维护 `running`/`waiting` 队列，按 token 预算产出 `{req_id: num_tokens}` |
| `core/kv_cache_manager.py` | KV 管理器 | `allocate_slots()` / `free()`：将 token 需求折算为块需求、处理前缀命中与显存记账 |
| `core/block_pool.py` | 物理块池 | `BlockPool`：管理空闲链表、`ref_cnt` 引用计数、LRU 淘汰与块级哈希映射 |
| `worker/block_table.py` | 块表管理 | `BlockTable`：维护各请求逻辑块到物理块号的二维映射张量，供 GPU Attention Kernel 读取 |

---

## 5.5 关键启动参数、抢占与排障逻辑

### 一条标准启动命令与三大核心旋钮

在 A100 80GB 上以标准 OpenAI 兼容模式启动 Llama-3.1-8B，典型命令如下：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

服务就绪后，通过 `/v1/chat/completions` 提供流式对话接口，通过 `/metrics` 暴露 Prometheus 监控指标（见[第 7 章](chapter-07.md)）。其中四个核心参数直接划定了引擎的资源边界：

| 启动参数 | 物理含义 | 调优取舍与风险 |
|---|---|---|
| `--gpu-memory-utilization` | 划归 vLLM 使用的显存比例（如 `0.90`） | 越高则 KV 池越大、并发越高；设太满（如 `0.98`）容易因 PyTorch 碎片或并发激活尖峰触发 OOM |
| `--max-model-len` | 单条请求允许的最大总长度（输入 + 输出） | 分页不会为每条请求预留该长度，但启动时要求 KV 池至少装得下 1 条满长度请求；按业务真实上限设（如 8K 或 32K），勿盲目沿用模型 128K 原生上限 |
| `--max-num-seqs` | 单步允许同时处于 `RUNNING` 状态的序列上限 | 控制最大 Decode 批大小与调度元数据规模；过小会浪费 KV 池，过大可能拉高单步 ITL |
| `--max-num-batched-tokens` | 单步前向的 token 预算（控制 Chunked Prefill 粒度） | 越大则 Prefill 吞吐越高，但单步耗时变长、Decode 抖动增大；调小可保护 TPOT/ITL |

### 抢占机制：重算（Recompute）与换出（Swap）

连续批处理允许请求按需扩张 KV，这带来了一个无法完全回避的边界情况：所有已准入的请求都在同时变长，某一步时空闲物理块耗尽，连一条 Decode 所需的 1 个新块都挤不出来。此时调度器必须执行**抢占（Preemption）**。历史上存在两条退路：

- **换出（Swap）**：通过 PCIe 把被抢占请求的 KV 块拷贝到主机 CPU 内存，待显存空出再拷回 GPU。
- **重算（Recompute）**：直接释放被抢占请求的全部 GPU 物理块，把已经生成的 token 拼接回它的原始 Prompt，退回等待队列；重新调度时像做一次更长的 Prefill 一样重算 KV。

!!! note "工程取舍：为什么 vLLM V1 默认选择 Recompute 而放弃 Swap？"
    PCIe Gen4 ×16 的双向有效带宽仅约 25–30 GB/s，把一条 8K 上下文的 1 GiB KV 换出再换回，仅搬运耗时就接近 70–80 ms，且需要额外维护庞大的 CPU 锁页内存池（pinned memory）与异步拷贝状态机。相比之下，在 A100（312 TFLOPS）上 Prefill 是高度算力密集的矩阵运算，重算 8K 上下文的耗时往往与 PCIe 往返相当甚至更快，且系统实现极为干净、零主机内存开销。因此 vLLM V1 移除了 `--swap-space`，将 **Recompute** 作为统一抢占路径。

需要牢记的是：**偶发排队与抢占是显存满载时的自我保护机制，并非引擎崩溃**；但如果在[第 7 章](chapter-07.md)的监控面板上观察到抢占计数持续攀升，说明准入并发或最大上下文已经超出单卡稳态容量，必须下调并发上限、开启量化或横向扩容。

### 显存、启动与端口三层排障逻辑

推理服务上线遇到异常，按生命周期切分为三层排查最快：

1. **启动阶段拦截（Profiling OOM / 架构不匹配）**：
   - vLLM 启动时会先跑一次假前向（memory profiling），测出峰值激活显存，再把剩余预算一次性划为 KV 块池（所以 `nvidia-smi` 启动即占满 90% 显存是正常现象，不是内存泄漏）。
   - 若启动报 KV 缓存空间不足，按**“先降 `--max-model-len` → 再微调 `--gpu-memory-utilization` → 最后降 `--max-num-seqs`”**的顺序收缩配置。
   - 若报模型架构未注册，核对 vLLM 版本与模型 `config.json` 中的 `architectures` 字段；端口冲突直接更换 `--port`；离线部署把模型名替换为本地权重目录路径即可。
2. **请求阶段拦截（HTTP 400）**：
   - 单条请求的 `prompt_tokens + max_tokens` 超过 `--max-model-len` 时，API 进程会直接拒绝请求，避免无效计算穿透到 EngineCore。
3. **运行阶段过载（延迟飙升与抢占抖动）**：
   - 此时检查 KV 使用率、排队深度与抢占计数三项指标（详见[第 7 章](chapter-07.md)），区分是长 Prompt 引发的算力排队还是长上下文引发的显存挤压。

---

**本章小结与延伸阅读**

vLLM 用两把钥匙打开了大模型高并发服务的大门：**PagedAttention** 用非连续分页把 KV 显存浪费锁死在最后一个块内，**迭代级连续批处理 + Chunked Prefill** 把省下的显存实时兑现为高吞吐且平滑的流式响应。关于 PagedAttention 论文与 Orca / vLLM V1 的详细数据口径对照，请参阅[专题 D：各章核心论文精读与数据口径](../topics/paper-notes.md)。

不过，在默认视角下，每条请求的 KV 在结束后仍会被直接丢弃。当线上业务存在大量重复的系统提示词、多轮对话历史或并行思维链分支时，如何跨请求复用已经算过的 KV？下一章[第 6 章 SGLang：RadixAttention 与前缀复用](chapter-06.md)将把匹配粒度一直推进到单个 token。
