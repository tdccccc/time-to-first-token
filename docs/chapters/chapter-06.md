# 第 6 章 SGLang：RadixAttention 与前缀复用 { .chapter-title }

<p class="chapter-summary">当大量请求共享系统提示词、多轮对话历史或并行思维链前缀时，SGLang 通过基数树（RadixAttention）将 KV 缓存的跨请求复用粒度推至单个 token，并与缓存感知调度、分块 Prefill 及结构化输出协同兑现为低 TTFT 与高吞吐。</p>

第 5 章的 vLLM 已经通过分页和连续批处理把单次请求内的显存浪费压到极低，但如果在每条请求结束时直接把 KV 全部销毁，另一笔巨大的浪费就会浮现出来：真实业务里，成百上千条请求往往顶着同一段冗长的 System Prompt、同一篇 RAG 检索文档，或在同一场多轮对话里不断把前几轮历史重复发给模型。如果每次都从头 Prefill、各自存一份私有 KV，GPU 的大部分算力与显存都会耗在重复劳动上。

本章先算清重复前缀这笔隐形账本，再沿着**“整段哈希 → Block 哈希 → 基数树”**的粒度演进路线拆解 RadixAttention 的数据结构与调度生命周期，最后把分块 Prefill 与 SGLang 的前后端协同设计拼成完整图景。

---

## 6.1 重复前缀的隐形账本

### 100 条共享请求省下多少显存与算力？

以本书主线模型 **Llama-3.1-8B（FP16，每 token KV 占用 128 KiB = 0.125 MiB）** 为例，考察一个典型的 Agent / 客服场景：100 条并发请求共享同一段 **1,000 token** 的 System Prompt，每条请求另有 **50 token** 独有用户输入并生成 **100 token** 输出（单请求总长 1,150 token）。

按有效 token 结算两笔账：

1. **KV 显存账**：
   - **完全不共享**：100 条请求各自存一份完整序列，共占 $100 \times 1{,}150 \times 0.125\text{ MiB} = \mathbf{14{,}375\text{ MiB}} \approx 14.04\text{ GiB}$，其中 $100 \times 1{,}000 \times 0.125 = 12{,}500\text{ MiB}$ 全是同一段 System Prompt 的 100 份重复副本，独有后缀只占 $1{,}875\text{ MiB}$。
   - **前缀复用后**：1,000 token 的公共前缀只存 1 份（$125\text{ MiB}$），加上 100 条独有后缀（$1{,}875\text{ MiB}$），总计仅需 **$2{,}000\text{ MiB} \approx 1.95\text{ GiB}$**——**直接省掉 $(14{,}375 - 2{,}000)/14{,}375 = \mathbf{86.1\%}$ 的 KV 显存**！
2. **Prefill 算力账**：
   - **完全不共享**：100 条请求每条都要对 1,050 个输入 token 做前向，总计计算 $100 \times 1{,}050 = 105{,}000$ 个 Prefill token。
   - **前缀复用后**：首条请求算完 1,050 个 token，后续 99 条请求直接跳过前 1,000 个 token，各自只算 50 个新 token，总计仅需 $1{,}050 + 99 \times 50 = 6{,}000$ 个 token——**直接省掉 $99{,}000$ 个 Prefill token（降幅高达 $\mathbf{94.3\%}$）**！

对于单条命中缓存的请求，其 Prefill token 数从 1,050 骤降到 50（减少 $95.2\%$）。不过要注意 TTFT 并非总能等比例下降 $95.2\%$：第一，TTFT 由排队时间与 Prefill 时间共同构成，设 Prefill 占 TTFT 的比例为 $p$，理论降幅上限约为 $95.2\% \times p$；第二，在 A100 80GB（FP16 脊点 156 FLOP/B）上，单条 50 token 的短 Prefill 会落回权重搬运主导的带宽受限区，整卡读一遍 16 GB 权重的固定地板（约 8 ms）会托住实际下限。

!!! note "模型架构差异"
    换到另一主线模型 **Qwen3-8B（144 KiB/tok）**，因为同样是纯 GQA Transformer，上述 86.1% 的 KV 节省比例与 94.3% 的 Prefill 节省比例完全不变。但如果底层是混合线性注意力模型（如部分层采用固定大小递推状态的架构），递推状态难以像 KV 块那样在任意前缀处直接零拷贝共享，实得显存节省会被每请求的固定状态底盘稀释，详见[专题 A：前沿长上下文与混合架构拆账](../topics/advanced-architectures.md)。

### 为什么只能复用“最长公共前缀”？

既然重复就能复用，为什么不能复用两段 Prompt 中相同的后缀或中间片段？根源在于大语言模型的两个核心数学性质：

- **因果注意力掩码（Causal Attention Mask）**：第 $i$ 个 token 的隐藏状态只能看到位置 $0 \dots i$ 的全部左侧上下文。经过第 1 层注意力后，该位置的隐藏状态就已经融合了整段左侧历史；随后投影出的各层 K、V 全部携带了左侧上下文的印记。
  - 例如请求 A 为 `["天气", "很", "好"]`，请求 B 为 `["心情", "很", "好"]`。尽管后缀 `["很", "好"]` 字面完全一致，但“很”字左侧的语境不同，其 K、V 数值截然不同，直接复用会破坏语义正确性。
- **位置编码（如 RoPE）**：旋转位置编码把绝对位置信息注入 Q、K 向量。即便同一段文本出现在不同起始偏移处，其 K 向量经过的旋转角度也不同，无法直接复用。

因此，**语义无损的可复用单位严格等价于“从位置 0 开始的最长公共前缀（Longest Common Prefix）”**。这给缓存索引结构提出了四项硬要求：

1. 快速检索任意输入序列的最长公共前缀；
2. 当新序列只匹配到某段前缀的中间位置时，能够低开销地分裂节点；
3. 公共前缀只保存一份物理 KV，分叉后的后缀各自独立读写；
4. 元数据保持紧凑，不能为毫无分叉的长提示词建出成千上万个冗余节点。

---

## 6.2 从整段哈希到 Block 哈希，再到基数树（Radix Tree）

### 三种前缀缓存方案横评

沿着匹配粒度由粗到细，工程界演化出三种方案：

- **整段哈希（Full-Prompt Hashing）**：把整条 Prompt 当作一个 key 做哈希。它只能命中一字不差的重复请求；在 6.1 节 100 条“同 System Prompt + 不同用户输入”的负载里，后面 99 条请求哈希全都不匹配，**命中率为 0%**。
- **Block 哈希（Block-Level Prefix Caching，vLLM APC 方案）**：天然复用 PagedAttention 的固定块（默认 `block_size = 16`）。每个块的哈希把父块哈希链式纳入计算，从而唯一代表“从序列起点到该块末尾”的完整前缀。请求到达时逐块查表，命中就复用整块。
  - 回到 6.1 节算例：1,000 token 的共享前缀按 16 token 分块，可凑出 62 个整块（992 token），**余下 8 个尾巴 token 凑不满一块，无法共享**。
  - 这 8 个尾巴被迫在 100 条请求里各存一份，额外消耗 $99 \times 8 \times 0.125 = 99\text{ MiB}$（总有效 KV 升至 $2{,}099\text{ MiB}$；再计入每条独有后缀 158 token 向上取整到 10 块 160 槽位的 25 MiB 对齐开销，物理占用 $2{,}124\text{ MiB}$）。
- **基数树（Radix Tree，SGLang RadixAttention 方案）**：一种**路径压缩前缀树（Compressed Trie）**。每条边不再只存单个 token，而是压缩存储一整段无分叉的连续 token 序列，只在真正发生分叉的位置创建节点。节点数量与请求分叉次数成正比，与 Prompt 长度无关，同时将匹配粒度一路压到最小页/单 token 级别。

三套方案的横向对比如下：

| 维度 | 整段哈希 | Block 哈希（vLLM APC） | 基数树（SGLang RadixAttention） |
|---|---|---|---|
| **匹配粒度** | 整条 Prompt | 固定物理块（默认 16 token） | 可低至 1 token（或按 `page_size` 对齐） |
| **未对齐尾部浪费** | 整段失效 | 最多 $B - 1$ 个 token（默认最多 15） | 0（`page_size=1` 时） |
| **6.1 算例有效 KV 占用** | 14,375 MiB（省 0%） | 2,099 MiB（省 85.4%） | **2,000 MiB（省 86.1%）** |
| **元数据与维护开销** | 极低（$O(1)$ 哈希表） | 低（块级哈希表 + 引用计数） | 中等（树节点查找、节点分裂与叶子 LRU） |
| **适用优势区** | 固定模板重放 | 块对齐的长前缀、极简维护 | 任意长度分叉、多轮对话、树状采样（Tree-of-Thought） |

### RadixTreeNode 结构与三步分裂演进

首先必须确立一个核心边界：**基数树是索引，KV 池才是数据。** 树节点里从不保存沉重的 K/V 浮点张量，只保存逻辑 token 片段与对应 GPU 物理槽位编号的映射：

```python
from dataclasses import dataclass, field


@dataclass
class RadixTreeNode:
    parent: "RadixTreeNode | None" = None
    children: dict[int, "RadixTreeNode"] = field(default_factory=dict)
    key: list[int] = field(default_factory=list)      # 本节点覆盖的连续 token 段
    value: list[int] = field(default_factory=list)    # 对应的物理 KV 槽位编号
    ref_count: int = 0                                # 正在使用该路径的活跃请求数
    last_used: float = 0.0                            # LRU 最近访问时间戳

    @property
    def length(self) -> int:
        return len(self.key)
```

用三条请求走一遍树的生长与分裂全过程（假设 `page_size = 1`）：

- 请求 A：`[101, 11, 12, 21, 22]`
- 请求 B：`[101, 11, 12, 31, 32]`
- 请求 C：`[101, 11, 99]`

**第 1 步：插入 A。** 树为空，A 的整段序列直接压成单个节点挂在 `root` 下：

```text
root
└── [101, 11, 12, 21, 22]   # A
```

**第 2 步：插入 B。** 沿首 token `101` 找到 A 节点，逐位比对得到最长公共前缀 `[101, 11, 12]`（长度 3 < 节点长度 5）。匹配停在节点中部，于位置 3 触发**分裂（split）**：公共前缀抽成新父节点，A 收缩为后缀 `[21, 22]`，B 的后缀 `[31, 32]` 作为兄弟节点挂入：

```text
root
└── [101, 11, 12]           # A/B 共享
    ├── [21, 22]            # A 独有
    └── [31, 32]            # B 独有
```

**第 3 步：插入 C。** 与 `[101, 11, 12]` 比对，公共前缀 `[101, 11]`（长度 2 < 3），于位置 2 再次分裂：

```text
root
└── [101, 11]               # A/B/C 共享
    ├── [12]                # A/B 共享
    │   ├── [21, 22]        # A 独有
    │   └── [31, 32]        # B 独有
    └── [99]                # C 独有
```

<figure class="book-figure">
  <img src="../../../assets/figures/radix-tree.svg" alt="基数树共享前缀示意：左侧为 root 经 [101,11] 分裂出 [12] 和 [99] 以及子叶 [21,22] 和 [31,32] 的树结构，右侧为 A、B、C 三条请求共享物理 KV 的颜色映射" loading="lazy">
  <figcaption><span class="book-figure__label">图 6-1</span>左树右序列对照阅读：同色 token 指向 GPU KV 池里的同一份物理槽位。节点分裂仅在 CPU 侧切片重组 <code>key</code>/<code>value</code> 指针，绝不复制显存中的 KV 数据。</figcaption>
</figure>

三次插入浓缩了基数树的全部精髓：**整段命中**（继续下探）、**部分命中**（停在节点中部）、**零拷贝分裂**（只切元数据切片，新父节点继承原 `ref_count`，物理 KV 原地不动）。

### `match_prefix` 核心逻辑、`page_size` 折中与叶子 LRU 淘汰

最长前缀匹配算法干净利落，下面这段代码完整呈现了“逐段下探 + 遇分叉即分裂”的主干：

```python
def common_prefix_len(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    return idx


def align_down(length: int, page_size: int) -> int:
    return (length // page_size) * page_size


def match_prefix(root, input_ids: list[int], page_size: int = 1):
    node = root
    prefix_len = 0
    matched_values: list[int] = []

    while prefix_len < len(input_ids):
        key_head = tuple(input_ids[prefix_len : prefix_len + page_size])
        child = node.children.get(key_head)
        if child is None:
            break

        match_len = common_prefix_len(child.key, input_ids[prefix_len:])
        match_len = align_down(match_len, page_size)
        if match_len == 0:
            break

        if match_len < child.length:
            # 停在节点中部：零拷贝分裂，返回新生成的前缀父节点
            child = child.split_at(match_len)
            matched_values.extend(child.value)
            prefix_len += match_len
            node = child
            break

        matched_values.extend(child.value)
        prefix_len += match_len
        node = child

    return prefix_len, node, matched_values
```

工程落地时有两个紧挨着的设计折中值得看透：

- **`page_size` 的对齐折中**：当 `page_size = 1` 时，共享尾巴能压到 0 浪费，但 Attention 后端（如 FlashInfer / FlashAttention）往往按页组织访存。若将 `page_size` 设为 16 或 64，树节点索引以“前 `page_size` 个 token 的元组”为键，匹配长度向下对齐到 `page_size` 的整数倍——换取更少的页表项与更高的 GPU 访存连续性，代价则是允许最多 `page_size - 1` 个尾部 token 无法复用。
- **叶子 LRU 淘汰策略**：当显存吃紧需要驱逐历史缓存时，基数树**严格从叶子节点开始淘汰**，按最小 `last_used` 挑选 **`ref_count == 0`** 的叶子摘除，并将其物理槽位归还空闲池。如果某叶子的摘除让它的父节点不再拥有任何子节点，该父节点会自然退化为新的候选叶子。这一设计保证了一个极佳的局部性性质：**最冷门的长尾独有后缀最先被回收，而高频共享的公共前缀（靠近根部）会被保护到最后。**

---

## 6.3 把树接进调度器：完整运行时生命周期

基数树本身只是一张静态索引。要让前缀复用在并发请求潮汐里安全运转，必须把树深度接入调度器，回答五个运行时问题：怎么找前缀、前向期间怎么防淘汰、未命中后缀怎么分配、算完后怎么入树、结束或取消怎么回收？

### 五阶段生命周期：`match → lock → allocate → insert → unlock/evict/free`

一条命中请求在 SGLang 运行时中严格走完以下五个阶段：

| 阶段 | 树与物理池动作 | 核心目的 | 关键工程边界 |
|---|---|---|---|
| **1. `match`** | 沿树检索最长前缀，必要时分裂节点，刷新沿途 `last_used` | 拿到可复用物理槽位列表 `matched_values` | 匹配长度向下对齐到 `page_size` |
| **2. `lock`** | 沿匹配节点到 `root` 整条路径 `ref_count += 1` | 锁住共享路径，防止前向计算期间被并发淘汰 | **加锁后必须二次校验可用空间** |
| **3. `allocate`** | 为未命中后缀 `extend_len = input_len - cached_len` 分配新物理页，写入请求私有 Page Table | 给新算出的 KV 准备显存落脚点 | 先做预算判断，分块 Prefill 仅兑现当前 chunk 所需页 |
| **4. `insert`** | 前向写完 KV 后，把新前缀与槽位登记进树，替换句柄并解锁旧句柄 | 将新计算成果开放给后续请求复用 | 若已被并发请求抢先插入，只更新时间戳并释放冗余页 |
| **5. `unlock / evict / free`** | 解锁路径 `ref_count -= 1`；未入树页立即归还空闲池；空间不足时按 LRU 淘汰 `ref_count == 0` 的叶子 | 安全回收显存、杜绝内存泄漏 | **正常结束与客户端取消共用同一清理路径** |

理解这条生命周期时，要清晰区分**两张表**和**一个归档时机**：

- **请求私有 Page Table vs 全局树索引**：请求的 Page Table 保存“本请求从第 0 个 token 到当前步的完整物理页列表”，直接喂给 Attention Kernel；树索引则只记录“已计算完毕、可供跨请求共享的前缀”。两者在命中前缀处指向同一批物理页；未命中后缀则先写入 Page Table，待前向完成后再登记进树。
- **Decode 阶段不逐 token 插树**：进入 Decode 后，每步只产出 1 个 token，如果每步都插树分裂会引入高昂的 CPU 元数据开销。因此引擎通常只在 Prefill 结束、请求完成或特定检查点时，才把稳定下来的序列批量归档进树。

### 加锁前后双检：堵住高并发淘汰竞态

!!! warning "工程关键细节：先查、再锁、再查"
    调度器估算可用显存时，口径是：

    $$
    \text{Available Pages} = \text{Free Pages} + \text{Evictable Tree Pages}\ (\text{ref\_count} = 0)
    $$

    这里藏着一个隐蔽的并发竞态：假设第一次检查时“空闲页 + 可淘汰页”刚好够准入当前请求；紧接着执行 **`lock`**，把这条请求命中的前缀路径全部 `ref_count += 1`——原本算在“可淘汰页”里的一批共享节点瞬间变成了**不可淘汰**，导致系统的实际可用空间突然缩水！如果不做复查直接放行，后续分配新页时会直接撞穿显存。因此 SGLang 的 `PrefillAdder` 在加锁前后严格执行**双检（Double Check）**：锁后空间不足则立即 `unlock` 并将请求放回等待队列。

### 缓存感知调度与完整性不变量

建好基数树后，等待队列里请求的排班顺序会剧烈左右全局命中率。设等待队列里有两条请求：请求 X 命中树上 1,000 token 的冷门前缀，请求 Y 几乎无命中。若按传统先来先服务先放行 Y，Y 分配显存触发的 LRU 淘汰很可能把 X 正要复用的 1,000 token 叶子直接驱逐，轮到 X 时被迫全部重算！

SGLang 为此引入了**缓存感知调度（Cache-Aware Scheduling / Longest Prefix Match 优先）**：按匹配前缀长度对等待队列排序，优先执行命中更长的请求，使同一批共享前缀的请求聚团消费，把缓存命中率推近负载上限。

!!! note "工程取舍：命中率 vs 公平性"
    按最长前缀长度排序本质上是拿**调度公平性**换**全局吞吐与命中率**。在持续涌入大量共享前缀请求的极端流量下，一条独一无二的新短请求可能会反复被插队而陷入长尾饥饿（starvation）。生产系统通常会配合等待超时权重或最大插队次数（aging），在提升命中率的同时兜住尾延迟底线。

最后，前缀复用虽然数学上严格等价，但工程上任何一次微小的错位都会导致模型胡言乱语或显存缓慢泄漏。成熟引擎在调试与回归中会守住三条**正确性不变量**：

1. **逐 token 精确匹配**：命中的每个 token ID 必须与树中路径严格一致。
2. **位置偏移无损恢复**：未命中后缀的 RoPE 位置必须从 `cached_len` 接着编号，绝不能重置为 0。
3. **物理页账本守恒**：任意时刻必须严格满足：

    $$
    \text{Free Pool Pages} + \text{Unique Pages in Tree \& Active Requests} = \text{Total Allocated Pages}
    $$

    一旦打破守恒，立即报错暴露泄漏或重复分配，而不是让 bug 潜伏数小时后才以随机 OOM 爆发。

---

## 6.4 补上另一半：分块 Prefill 与前后端协同设计

### 前缀复用省命中的，分块 Prefill 治跳不掉的

前缀复用再强，也救不了那些首次出现、完全未命中的长 Prompt。一条 8K 或 32K 的新文档进来，算力密集型的 Prefill 仍然必须完整算完；如果任由它独占 GPU 几十上百毫秒，正在流式输出的 Decode 请求就会集体卡顿。

来自 SARATHI 与 Sarathi-Serve 的**分块 Prefill（Chunked Prefill）**配合 **Decode-Maximal 批处理**补上了这另一半拼图：

- **机制**：把一个大 Prefill 切分成近似等大的 Chunk，每个 Chunk 搭上尽可能多的活跃 Decode 请求拼进同一个 Batch。以一条 **2,048 token** 的未命中 Prompt 为例：
  - `chunk_size = 2048`（不分块）：Decode 必须原地干等整段 2,048 token 算完。
  - `chunk_size = 1024`（分 2 块）：Decode 每步最多等约 $1/2$ 个 Prefill。
  - `chunk_size = 512`（分 4 块）：Decode 每步最多等约 $1/4$ 个 Prefill，ITL 尾延迟大幅收平。
- **双刃剑与取舍**：分块 Prefill 优化的是**全局 Decode ITL、整体算力利用率**（Decode 搭 Prefill 便车，不再空转算力），以及多机流水线并行下的微批均衡（消除级间气泡）；但代价是**这条长 Prefill 自己的首字延迟 TTFT 会被拉长到跨越 4 个 step**。选择 Chunk 大小（常见默认值在 2K–8K token 量级）的准则是：**先锁死业务能容忍的 ITL 上限，再在此约束下尽量把 Chunk 调大以摊薄 Kernel 启动开销。**

把两条主线合起来看：**前缀复用让命中的 Prefill 直接消失，分块 Prefill 让躲不掉的 Prefill 不再卡死 Decode。** 至于彻底在物理机器层面拆开两个阶段的 PD 解耦（Prefill-Decode Disaggregation），则是“改调度”与“改部署”的进一步取舍，我们将在[第 11 章](chapter-11.md)深入展开。

### 前端 DSL 与压缩有限状态机（Compressed FSM）

SGLang 的定位从来不只是“带基数树的 vLLM 替代品”，它的另一半核心竞争力在于**前后端协同设计**：

- **结构化前端 DSL**：SGLang 内嵌于 Python 的前端提供 `gen`（生成）、`select`（在候选中择优）、`fork`（并行分叉）与 `join`（汇合）原语。当一个多分支 Agent 调用 `fork(3)` 对同一篇长文档并行做摘要、实体抽取和情感打分时，前端会**主动通知运行时先将公共前缀入树，再并发下发三条分支**。共享结构不再靠后端“事后瞎猜”，而是由前端显式声明。
- **压缩有限状态机（Compressed FSM）加速结构化输出**：要求模型输出严格合法的 JSON（如 `{"title": "...", "score": ...}`）时，常规约束解码会逐 token 遮蔽非法词表，仍要一步步跑前向。但诸如 `{"title": "` 这类格式骨架在状态机里只有**唯一确定的转移路径**，模型根本没有选择余地。压缩 FSM 将这类确定性多字符边合并为一次跳转，让引擎直接跳过这些固定 token 的逐个采样，只在模型真正需要自由填值的位置才正常前向解码，从而显著拉升 JSON 结构化输出的吞吐。

### 源码路径对照与“前缀重/轻”双负载评估逻辑

SGLang 与 vLLM 在底层共享连续批处理与分页思想，核心差异落在前缀索引粒度（基数树 vs 块哈希）与前端协同上。阅读两大引擎源码时，可以按如下映射表对照切入：

| 职责 | SGLang（`python/sglang/srt/`） | vLLM V1（`vllm/v1/`） |
|---|---|---|
| **前缀索引与淘汰** | `mem_cache/radix_cache.py`（`RadixCache`, `TreeNode`, `match_prefix`, `split_at`, `evict`） | `core/block_pool.py`（`BlockPool`, `block_hash`, `ref_cnt`, LRU） |
| **KV 池与页表** | `mem_cache/memory_pool.py`（Token-to-KV Pool） | `core/kv_cache_manager.py` & `worker/block_table.py` |
| **调度策略** | `managers/schedule_policy.py`（LPM 缓存感知排序） | `core/sched/scheduler.py`（FCFS / 优先级调度） |
| **批组装与主循环** | `managers/schedule_batch.py`（`PrefillAdder` 双检）& `managers/scheduler.py` | `core/sched/scheduler.py`（统一 `{req_id: num_tokens}`） |

工程选型或做引擎横向评估时，记住一个核心方法论：**引擎差距是负载前缀共享率的函数，绝不是一个常数。** 一组干净、无偏的对比必须成对设置两种负载并严格控制变量：

1. **前缀重负载（量上限）**：例如 100 并发共享 1,000-token System Prompt、各 50 token 输入与 100 token 输出，重点观察 TTFT 中位数/p95 与整体输出吞吐。
2. **单轮无共享负载（量基线）**：100 条互不重叠的短 Prompt，校验引入树索引与缓存感知调度是否带来额外开销。

评估中务必警惕四类最容易污染结论的**混杂变量**：

- **标点或聊天模板差一个字节**：前缀匹配会从差异处立刻断链，造成命中率归零的假象；必须确保两边送入的逐字节文本与采样参数（`temperature`、`top_p`、`max_tokens`）完全一致。
- **冷树未预热**：第一轮请求仍在把公共前缀写进树里，计时前必须完成至少一轮完整热身，排除冷启动与 CUDA Graph 捕获抖动。
- **并发度选点单一**：并发极高时 TTFT 主要被排队时间淹没，前缀复用省下的 Prefill 时间会被稀释；应覆盖低、中、高多个并发档位。
- **块对齐巧合**：若测试前缀恰好是 16 token 的整数倍，vLLM 的 Block 哈希也能整块 100% 命中，无法区分块粒度与 token 粒度的尾巴差异；只有测试非 16 整数倍分叉或复杂树状多轮分支时，RadixAttention 的细粒度优势才会彻底显现。

---

**本章小结与延伸阅读**

RadixAttention 把“按请求私有的 KV 池”升级为“跨请求可共享的压缩前缀树”，用 `match → lock → allocate → insert → unlock/evict/free` 五步生命周期把多轮与共享前缀场景的 Prefill 算力与 KV 显存砍掉一个量级；而分块 Prefill 与前端压缩 FSM 则补上了未命中长输入与结构化解码的两块拼图。相关论文（SGLang、SARATHI、Sarathi-Serve）的原始数据口径请参阅[专题 D：各章核心论文精读与数据口径](../topics/paper-notes.md)。

引擎优化做得再漂亮，如果线上看不清队列深度、KV 水位和尾延迟的变化，就永远无法量化一次改动是变快还是变慢。下一章[第 7 章 可观测性与容量评估](chapter-07.md)将建立完整的指标体系与容量曲线方法论。
