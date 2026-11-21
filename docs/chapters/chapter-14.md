# 第 14 章 Fused Attention 深讲：Online Softmax 与分块实现 { .chapter-title }

<p class="chapter-summary">拆解标准 Attention 的二次显存搬运瓶颈，从零推导 Online Softmax 的精确重缩放恒等式与代码化简桥接，算清 A100/H100 片上 SRAM 与寄存器预算，并用完整 Triton 实现打通 Causal 跳块与 Roofline 定位。</p>

第 13 章末尾我们留下了悬念：Fused Softmax 之所以能把搬运量压成 1/3，靠的是把整行一次塞进片上寄存器；可在 Transformer 的自注意力（Self-Attention）里，不仅有长为 $N$ 的 Softmax，前后还夹着 $S = QK^T / \sqrt{d}$ 与 $O = PV$ 两次矩阵乘，中间矩阵尺寸随序列长度呈 $O(N^2)$ 膨胀。本章将彻底打开 **FlashAttention** 的核心引擎——从 IO 搬运账、Online Softmax 数学推导与代码化简，到 A100/H100 片上 SRAM 预算与完整 Triton 实现，一步不落地讲透 Fused Attention 为什么快、以及它的物理边界在哪里。

---

## 14.1 朴素 Attention 的 IO 瓶颈

标准缩放点积注意力的计算链路分为三步：

$$S = \frac{Q K^T}{\sqrt{d}}, \qquad P = \operatorname{softmax}(S), \qquad O = P V$$

朴素实现把这三步写成三个独立 Kernel，每次都把巨大的 $N \times N$ 中间矩阵 $S$ 与 $P$ 写回片外 HBM 再重新读出。这笔搬运账有多夸张？我们以 **NVIDIA A100 80GB SXM**（FP16 Tensor Core **312 TFLOPS**，HBM2e 带宽 **2 TB/s**，FP16 脊点 **156 FLOP/B**）上运行单头、单序列 FP16 Attention（序列长度 $N = 1024$、头维 $d = 64$）为例，并附 **H100 80GB SXM**（FP16 **989.5 TFLOPS**，HBM3 **3.35 TB/s**，FP16 脊点 **295 FLOP/B**）作为对照：

### 搬运膨胀 17 倍的显存账

**总计算量（两种实现完全相同）**：$QK^T$ 与 $PV$ 各为 $2 N^2 d$ FLOPs，合计：

$$\text{FLOPs} = 4 N^2 d = 4 \times 1024^2 \times 64 = 268{,}435{,}456\text{ FLOPs} \approx 0.268\text{ GFLOP}$$

**输入输出基础搬运量**：$Q, K, V, O$ 四个矩阵均为 $N \times d$ 的 FP16（每元素 2 字节），读写合计：

$$\text{IO}_{\text{fused}} = 4 \times (2 N d) = 8 N d = 8 \times 1024 \times 64 = 524{,}288\text{ B} = \mathbf{512\text{ KiB}}$$

**朴素实现的额外 $S/P$ 往返**：$S$ 与 $P$ 各为 $N \times N$ FP16（各占 2 MiB），各自写 HBM 一次、再读回一次：

$$\text{IO}_{S,P} = 2 \times 2 \times (2 N^2) = 8 N^2 = 8 \times 1024^2 = 8{,}388{,}608\text{ B} = \mathbf{8\text{ MiB}}$$

朴素版总搬运高达 $512\text{ KiB} + 8\text{ MiB} = \mathbf{8.5\text{ MiB}}$（$8{,}912{,}896\text{ B}$）——仅中间矩阵就把显存流量**膨胀了整整 17 倍**！当序列拉长到 $N = 8192$ 时，单头单层的 $S/P$ 往返更会暴增至 512 MiB，把显存容量与带宽双双打穿。

### 朴素版 vs Fused 版 Roofline 对照

| 指标 | 公式 / 口径 | 朴素实现（物化 $S, P$） | Fused Attention（片上融合） | 物理意义解读 |
|---|---|---|---|---|
| **计算量** | $4 N^2 d$ | 0.268 GFLOP | 0.268 GFLOP | 融合不省 FLOPs，算术总量完全一致 |
| **理想 HBM 流量** | 朴素 $8Nd + 8N^2$；Fused $8Nd$ | **8.5 MiB** | **512 KiB** | 搬运总量骤降 **17×** |
| **算术强度 $I$** | $\text{FLOPs} / \text{IO}$ | **30.1 FLOP/B** | **512 FLOP/B** | 跨过 A100(156) 与 H100(295) 脊点 |
| **A100 计算下界** | $\text{FLOPs} / 312\text{ TFLOPS}$ | $0.86\ \mu\text{s}$ | $0.86\ \mu\text{s}$ | A100 Tensor Core 理论极限 |
| **A100 搬运下界** | $\text{IO} / 2\text{ TB/s}$ | **$4.46\ \mu\text{s}$** | $0.26\ \mu\text{s}$ | 朴素版卡在带宽，Fused 卡在算力 |
| **A100 理论时间下界** | $\max(\text{计算}, \text{搬运})$ | $4.46\ \mu\text{s}$（带宽受限） | **$0.86\ \mu\text{s}$（计算受限）** | 理论加速下界比 **$5.2\times$** |
| **H100 理论时间下界** | 989.5 TFLOPS / 3.35 TB/s | $2.66\ \mu\text{s}$（带宽受限） | **$0.27\ \mu\text{s}$（计算受限）** | H100 脊点更陡，理论加速比达 **$9.8\times$** |

!!! tip "核心直觉：为什么搬运降了 17 倍，A100 理论加速下界却是 5.2 倍？"
    因为融合**切换了瓶颈区间**。朴素版困在带宽斜线上（$4.46\ \mu\text{s}$），Fused 版把搬运下界压到了 $0.26\ \mu\text{s}$，远低于 $0.86\ \mu\text{s}$ 的算力平线——时间不再由搬运决定，而是顶在了 Tensor Core 算力天花板上。到了算力暴涨、带宽增幅较缓的 H100 上，这一理论加速比进一步放大到 $9.8\times$。

**FlashAttention 的核心目标**由此一清二楚：沿序列维将 $Q, K, V$ 分块读入片上 SRAM，让 $S$ 与 $P$ 永远只活在片上 SRAM 和寄存器里——算完即用、用完即扔，绝不落盘 HBM。

---

## 14.2 Online Softmax 数学推导与代码化简桥接

想把 $S$ 和 $P$ 锁在片上，必须先跨过一道数学障碍：标准安全 Softmax 需要整行的全局最大值 $m$ 与全局指数和 $l$：

$$m = \max_{j} S_j, \qquad l = \sum_{j} e^{S_j - m}, \qquad O = \frac{1}{l} \sum_{j} e^{S_j - m} V_j$$

分块处理 $K, V$ 时，算完第 1 块根本不知道后面块里藏着多大的分数，全局 $m$ 和 $l$ 都是未知数。如果等扫完所有块再回头算 Softmax，就又得把整个 $S$ 写回 HBM。**Online Softmax** 用极其优雅的**换底重缩放（Rescaling）**恒等式彻底解开了这个死结。

### 运行状态不变式与精确更新推导

假设我们已经扫过若干 $K/V$ 块（记作已处理集合 $\mathcal{H}$），并在寄存器中维护三个运行量，满足严格不变式：

$$m = \max_{j \in \mathcal{H}} S_j, \qquad l = \sum_{j \in \mathcal{H}} e^{S_j - m}, \qquad \tilde{O} = \sum_{j \in \mathcal{H}} e^{S_j - m} V_j$$

注意 $\tilde{O}$ 是**尚未除以分母 $l$** 的加权分子。现在新到达一块 $K/V$（记作块 $b$），块内局部最大值为 $m_b = \max_{j \in b} S_j$，局部和为 $l_b = \sum_{j \in b} e^{S_j - m_b}$，局部加权分子为 $\tilde{O}_b = \sum_{j \in b} e^{S_j - m_b} V_j$。

新全局最大值顺理成章地更新为：

$$m' = \max(m, m_b)$$

要把旧基准 $m$ 下的历史状态与局部基准 $m_b$ 下的新块状态统一换算到共同新基准 $m'$ 下，只需利用一条初中指数恒等式：

$$e^{S_j - m'} = e^{S_j - m} \cdot e^{m - m'}$$

将历史项与新块项分别乘上各自的换底系数并相加，便得到 Online Softmax 的**精确更新公式**：

$$l' = l \cdot e^{m - m'} + l_b \cdot e^{m_b - m'}$$

$$\tilde{O}' = \tilde{O} \cdot e^{m - m'} + \tilde{O}_b \cdot e^{m_b - m'}$$

当所有 $K/V$ 块全部扫完时，只需在循环外做一次最终归一化 $O = \tilde{O} / l$。用数学归纳法即可看出：初始置 $m = -\infty,\ l = 0,\ \tilde{O} = 0$（代表空集 $\mathcal{H} = \varnothing$）；首块到来时 $m' = m_b$，历史缩放因子 $e^{-\infty - m_b} = 0$，精确落入首块状态；每步维持不变式，结束时与整行 Softmax 在数学上**精确相等**——它不是近似算法，只是改变了求和基准与数据落点。

### 从论文公式到 Kernel 代码的 2 行代数桥接

许多读者第一次对照论文公式读 FlashAttention 代码时都会困惑：论文里明明有 $e^{m - m'}$ 和 $e^{m_b - m'}$ 两个缩放因子，为什么代码里只算了一个 `alpha = exp(m_i - m_new)`，根本不见 `exp(m_b - m_new)`？

把上面更新式里的第二项展开两行，谜底瞬间揭晓：

$$l_b \cdot e^{m_b - m'} = \left(\sum_{j \in b} e^{S_j - m_b}\right) e^{m_b - m'} = \sum_{j \in b} e^{S_j - m'}$$

$$\tilde{O}_b \cdot e^{m_b - m'} = \left(\sum_{j \in b} e^{S_j - m_b} V_j\right) e^{m_b - m'} = \sum_{j \in b} e^{S_j - m'} V_j$$

原来，$e^{m_b - m'}$ 被直接**吸收进了块内指数计算**！只要在求完 $m' = \max(m, m_b)$ 之后，直接以新基准 $m'$ 计算当前块的未归一化权重矩阵：

$$P_{\text{block}} = \exp(S_{\text{block}} - m'), \qquad \alpha = \exp(m - m')$$

整套更新公式便严丝合缝地化简为代码里的两行核心赋值：

$$l' = l \cdot \alpha + \operatorname{rowsum}(P_{\text{block}}), \qquad \tilde{O}' = \tilde{O} \cdot \alpha + P_{\text{block}} V_{\text{block}}$$

由于 $m' \ge m_b \ge S_j$ 且 $m' \ge m$，不仅省掉了单独计算 $e^{m_b - m'}$ 与一次乘法，还保证指数输入 $S_j - m' \le 0$ 与 $m - m' \le 0$ 严格非正，永远不会发生上溢！

### 手算验算：证明精确等价与“漏乘 $\alpha$”的灾难

取极简一维算例：$S = [1, 2, 3, 4]$，$V = [10, 20, 30, 40]$（$d = 1$）。

**整行直接计算（基准真值）**：全局 $m = 4$，$\exp(S - 4) = [e^{-3}, e^{-2}, e^{-1}, 1] = [0.049787,\ 0.135335,\ 0.367879,\ 1.000000]$，分母 $l = 1.553002$。Softmax 权重为 $[0.032059,\ 0.087144,\ 0.236883,\ 0.643914]$，最终加权输出为：

$$O = 0.032059 \times 10 + 0.087144 \times 20 + 0.236883 \times 30 + 0.643914 \times 40 = \mathbf{34.926527}$$

**分两块 Online Softmax 计算**（块 1 为 $[1, 2]$、$[10, 20]$；块 2 为 $[3, 4]$、$[30, 40]$）：

1. **处理块 1**：$m_1 = 2$，$P_1 = [e^{-1}, 1] = [0.367879,\ 1]$，$l_1 = 1.367879$，$\tilde{O}_1 = 0.367879 \times 10 + 1 \times 20 = 23.678794$。
2. **处理块 2**：$m_b = 4 \Rightarrow m' = \max(2, 4) = 4$；历史缩放系数 $\alpha = e^{2 - 4} = 0.135335$；直接以 $m' = 4$ 算 $P_2 = [e^{3-4}, e^{4-4}] = [0.367879,\ 1]$，得 $\operatorname{rowsum}(P_2) = 1.367879$，$P_2 V_2 = 0.367879 \times 30 + 1 \times 40 = 51.036383$。
3. **重缩放合并并归一化**：将块 1 的运行状态乘上 $\alpha$ 后与块 2 累加：

$$l' = 1.367879 \times 0.135335 + 1.367879 = 0.185122 + 1.367879 = \mathbf{1.553002}$$

$$\tilde{O}' = 23.678794 \times 0.135335 + 51.036383 = 3.204576 + 51.036383 = \mathbf{54.240960}$$

最终做一次除法即可得到 $O = 54.240960 / 1.553002 = \mathbf{34.926527}$，与整行结果小数点后 6 位完全吻合！

如果在写代码时粗心给 `l_i` 乘了 `alpha`，却忘了给累加器 `acc`（即 $\tilde{O}$）乘 `alpha`，分子就会错算为 $23.678794 + 51.036383 = 74.715177$，最终输出变成 $48.110168$——整整偏离真值 **13.18**，远超浮点舍入误差！

!!! warning "定义域雷区：全 Mask 行触发 $\exp(-\infty - (-\infty)) = \text{NaN}$"
    如果某一行在当前块及此前所有块中没有任何合法 Key（全被 Mask 置为 $-\infty$），则 $m = -\infty$、$m_b = -\infty \Rightarrow m' = -\infty$，此时 $\alpha = \exp(-\infty - (-\infty)) = \exp(\text{NaN}) = \text{NaN}$！这不是精度问题，而是 IEEE 754 定义域灾难。Kernel 必须保证每条有效 Query 行至少有一个合法 Key 参与首轮更新（或在跳块/滑动窗口场景下对全掩码行做显式保护）。

---

## 14.3 SRAM 与寄存器预算决定分块大小

数学通了，接下来要过硬件资源这一关。把 $S, P$ 留驻片上的代价，是每个线程块（CTA）必须在有限的 SM 片上 **SRAM（Shared Memory）** 与 **寄存器文件**里同时安放所有活跃 Tile。

本节以 **A100 80GB SXM**（108 SM，每 SM **164 KB** Shared Memory，单 Block 上限 **163 KB**，64K 个 32-bit 寄存器/SM，单线程上限 255 寄存器）为主线测算，并列 **H100 80GB SXM**（132 SM，每 SM **228 KB** Shared Memory，单 Block 上限 **227 KB**，同样 64K 寄存器/SM）作为对照。

### 片上存储分工与单/双缓冲 SRAM 公式

设 Query 块行数为 $B_r$（代码中的 `BLOCK_M`），Key/Value 块列数为 $B_c$（代码中的 `BLOCK_N`），头维为 $d$，输入为 FP16（2 字节/元素）。一个 Program 的片上核心数据结构如下：

- **Q Tile（$B_r \times d$ FP16，$2 d B_r$ 字节）**：在进入 K/V 循环前仅从 HBM 读入一次。需要理清一个常被混淆的细节——Q 经 SRAM 加载缓冲进入 SM 后，在后续 K/V 迭代中主要**常驻于寄存器复用**；我们在保守估算单缓冲 SRAM 峰值时仍保留 Q 的加载槽位，同时在寄存器预算里也计入 FP16 `q` 的寄存器开销，以此为编译器留足安全余量。
- **K Tile 与 V Tile（各 $B_c \times d$ FP16，各 $2 d B_c$ 字节）**：每轮迭代各搬入一块，共 $4 d B_c$ 字节。
- **P Tile（$B_r \times B_c$ FP16，$2 B_r B_c$ 字节）**：作为第二次 `tl.dot(p, v)` 的左操作数暂存。
- **寄存器专属张量**：FP32 分数块 `qk`（$B_r \times B_c$）、FP32 输出累加器 `acc`（$B_r \times d$）、FP16 复用块 `q`（$B_r \times d$，折合 $B_r d / 2$ 个 32-bit 寄存器）以及运行向量 `m_i`、`l_i`（各 $B_r$ 个 FP32）。

由此得到单缓冲（`num_stages=1`）与双缓冲流水线（`num_stages=2`，额外预取下一对 K/V 块，增加 $4 d B_c$ 字节）的片上 SRAM 预算公式：

$$\text{SRAM}_{\text{single}} \approx 2d(B_r + 2B_c) + 2 B_r B_c, \qquad \text{SRAM}_{\text{double}} = \text{SRAM}_{\text{single}} + 4 d B_c$$

以及每个线程至少消耗的 32-bit 寄存器数下限（未计地址计算、Mask 与编译器临时变量）：

$$\text{regs/thread} \ge \frac{B_r B_c + 1.5 B_r d + 2 B_r}{32 \cdot \text{num\_warps}}$$

### $B_r \times B_c$ 资源预算与 SM 驻留全景表（$d = 64$，FP16）

按每 SM 同时满足 **SRAM 容量上限**与 **65,536 寄存器上限**取最小值，算清各分块组合在 A100 与 H100 上的真实驻留能力：

| $B_r \times B_c$ | `num_warps` | Q / K / V / P（KB） | 单 / 双缓冲 SRAM | 寄存器下限（每线程） | A100 驻留 Block（单 / 双） | H100 驻留 Block（单 / 双） | 瓶颈项与可行性评估 |
|---|---:|---|---|---:|---|---|---|
| **$64 \times 64$** | 4 | 8 / 8 / 8 / 8 | 32 / 48 KB | 81 | **5 / 3** | **6 / 4** | 稳健起点，双缓冲驻留充足 |
| **$128 \times 64$** | 8 | 16 / 8 / 8 / 16 | 48 / 64 KB | 81 | **3 / 2** | **3 / 3** | 黄金均衡点，高复用且无溢出 |
| **$64 \times 128$** | 4 | 8 / 16 / 16 / 16 | 56 / 88 KB | 113 | **2 / 1** | **4 / 2** | K/V 占比较大，A100 双缓冲掉至 1 块 |
| **$128 \times 128$** | 8 | 16 / 16 / 16 / 32 | 80 / 112 KB | 113 | **2 / 1** | **2 / 2** | A100 实用上限，P 块膨胀至 32 KB |
| **$256 \times 64$** | 8 | 32 / 8 / 8 / 32 | 80 / 96 KB | 162 | **1 / 1** | **1 / 1** | 寄存器率先撞墙（41.5K/块），仅驻 1 块 |
| **$256 \times 128$** | 8 | 32 / 16 / 16 / 64 | 128 / 160 KB | 226 | **1 / 1** | **1 / 1** | 逼近 255 寄存器硬顶，极易触发 Spill |
| **$128 \times 256$** | 8 | 16 / 32 / 32 / 64 | 144 / 208 KB | 177 | **1 / 超限** | **1 / 1** | 双缓冲 208 KB 突破 A100 163 KB 单块上限 |

注意两项关键变量联动：

1. **头维 $d = 128$ 必须重算**：Llama-3.1-8B 与 Qwen3-8B 的真实头维是 $d = 128$ 而非 64。此时 Q/K/V 体积全部翻倍——例如 $B_r = 128, B_c = 64$ 的单缓冲从 48 KB 涨至 **80 KB**，双缓冲涨至 **112 KB**，A100 上立刻从 2 块降为 1 块。换模型就得重算表，不能盲抄默认块参数。
2. **$P$ 块是唯一的二次项**：当 $B_r, B_c$ 同时翻倍（$64\times 64 \to 128\times 128$），Q/K/V 只翻 2 倍，而 $P$ 按 $B_r \cdot B_c$ 暴涨 **4 倍**（$8\text{ KB} \to 32\text{ KB}$），迅速吞噬片上空间。

!!! warning "常见误区：$B_r, B_c$ 绝不是越大越好"
    初学者常误以为把块开得越大、K/V 复用越高就一定越快。实际上过度放大 $B_r, B_c$ 会同时引发三重反噬：（1）$P$ 二次膨胀挤压双缓冲流水与 SM 驻留数；（2）单线程寄存器逼近 255 上限，触发昂贵的片外 Local Memory Spilling；（3）**压缩总并行度（CTA 数）**——对 $N = 1024$ 的单头注意力，$B_r = 64$ 仅有 $\lceil 1024/64 \rceil = 16$ 个 Program，连 A100 108 个 SM 的 15% 都填不满；若强行开到 $B_r = 256$，只剩 4 个 Program，超过 96% 的 SM 直接空转！工业级实现必须把 `batch × num_heads` 放进 Grid，长序列下还需沿 K 维切分（Split-K / FlashDecoding）才能喂饱整卡。

---

## 14.4 完整 Triton Fused Attention 实现与 Causal 块跳过

理清数学与片上资源后，把第 13 章的 2D 广播指针、Mask 保护与本节 Online Softmax 合龙，就能写出结构极其凝练的 Fused Attention 内核。

<figure class="book-figure">
  <img src="../../../assets/figures/flash-attention-online-softmax.svg" alt="FlashAttention 片上数据流、Online Softmax 递推环与 Causal Mask 4×4 分块方阵：左半展示 Q_i 载入一次常驻片上、K_j/V_j 逐块流入，S_ij 与 P_ij 仅存片上永不落盘 HBM，经 alpha 同步重缩放更新 l 与 O_tilde；右半展示 4×4 因果分块中严格下三角全保留、对角块逐元素掩码、严格上三角整块跳过省去约 50% 计算" loading="lazy">
  <figcaption><span class="book-figure__label">图 14-1</span>Fused Attention 的片上执行流与因果分块剪枝。左图：<code>Q_i</code> 载入一次常驻片上，<code>K_j/V_j</code> 沿序列维滑窗流入，中间矩阵 <code>S_ij</code> 与 <code>P_ij</code> 用完即弃、零落盘 HBM，下方状态环用 <code>alpha</code> 同步重缩放运行分母与分子；右图：4×4 因果掩码矩阵裂解为严格下三角（全保留）、主对角（逐元素掩码）与严格上三角（整块跳过）三类 Tile。</figcaption>
</figure>

### 完整单头 Triton Fused Attention 代码（含 Causal 整块跳过）

以下实现面向单头、单序列、末维连续的 `[N, D]` FP16 输入（$D$ 为 2 的幂且 $\ge 16$，如 64 或 128），支持 `IS_CAUSAL` 编译期开关与严格上三角整块跳过：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_attn_fwd(
    Q, K, V, Out,
    stride_qm, stride_kn, stride_vn, stride_om,
    N, sm_scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < N

    # 1. Q 块仅在循环外加载一次，整个 K/V 扫描期间常驻片上复用
    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )

    # 2. 初始化 Online Softmax 的三个运行状态（对应归纳法空集）
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    # 3. Causal 整块跳过：严格上三角区域（列号 >= 当前 Q 块最大行号 + 1）直接截断循环
    hi = tl.minimum((pid_m + 1) * BLOCK_M, N) if IS_CAUSAL else N

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # 关键 A：K 越界位置必须先填 0.0，绝不可填 -inf（否则 0 * (-inf) = NaN 污染 dot）
        k = tl.load(
            K + offs_n[:, None] * stride_kn + offs_d[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # 关键 B：在 dot 之后统一施加越界掩码与因果掩码（>= 保留主对角线）
        keep = mask_n[None, :]
        if IS_CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(keep, qk, -float("inf"))

        # 关键 C：Online Softmax 化简后的重缩放更新
        m_block = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            V + offs_n[:, None] * stride_vn + offs_d[None, :],
            mask=mask_n[:, None],
            other=0.0,
        )
        # 关键 D：alpha 必须同时缩放 l_i 与 acc；p 转回 FP16 进入 Tensor Core
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # 4. 最终归一化（对合法行恒有 l_i >= 1.0，钳底仅防空行除零）并写回 HBM
    l_i = tl.maximum(l_i, 1.0)
    acc = acc / l_i[:, None]
    tl.store(
        Out + offs_m[:, None] * stride_om + offs_d[None, :],
        acc.to(Out.dtype.element_ty),
        mask=mask_m[:, None],
    )


def fused_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    n, d = q.shape
    out = torch.empty_like(q)
    if n == 0:
        return out
    block_m, block_n = 64, 64
    grid = (triton.cdiv(n, block_m),)
    _fused_attn_fwd[grid](
        q, k, v, out,
        q.stride(0), k.stride(0), v.stride(0), out.stride(0),
        n, 1.0 / (d ** 0.5),
        D=d, BLOCK_M=block_m, BLOCK_N=block_n,
        IS_CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return out
```

### 逐行拆解五个最易踩坑的工程细节

1. **先 `other=0.0` 加载 $K$，再在 `tl.dot` 之后用 `tl.where` 置 $-\infty$**：若在 `tl.load` 时图省事把越界列直接设为 $-\infty$，矩阵乘里一旦遇到 $0 \times (-\infty)$ 就会立刻生成 `NaN`，并把整行 `qk` 全部毒死。先填 `0.0` 安全做完点积，再通过 `tl.where` 屏蔽无效列，是不可颠倒的顺序。
2. **`alpha` 同时乘 `l_i` 与 `acc`**：正如 14.2 节的手算所示，`l_i` 与 `acc` 共享同一套基准 $m_i$，漏乘任意一方都会直接撕裂分子分母比例。
3. **`p.to(v.dtype)` 切入 Tensor Core**：`p` 由 FP32 指数算出，但在执行 `tl.dot(p, v)` 前必须转回 FP16，才能走满 A100 的 **FP16×FP16→FP32** Tensor Core 路径（312 TFLOPS）。若把 `p` 留在 FP32，要么回退到 TF32（算力腰斩），要么退回 IEEE FP32（慢一个数量级）。这次 FP32→FP16 转换引入的单次舍入相对误差约 $2^{-11} \approx 4.9 \times 10^{-4}$，叠加累加次序差异，工程上与 FP32 参考或 PyTorch SDPA 比对时通常设定最大绝对误差容限 `1e-2`（一旦看到 `1e-1` 或 `NaN`，必是 Mask 或 Rescale 逻辑错误）。
4. **因果条件必须写 `>=` 而非 `>`**：第 $m$ 个 token 可以关注自身（$n = m$）。一旦误写成 `>`，第 0 行将没有任何合法 Key，立刻触发 $\exp(-\infty - (-\infty)) = \text{NaN}$ 崩盘。
5. **为何保留 `tl.exp` 与 `exp2` 优化伏笔**：Triton 后端在生成 NVIDIA PTX 时本身就会把 FP32 `tl.exp(x)` 乘以 $\log_2 e$ 后下沉为硬件快速指令 `ex2.approx.f32`。在 FlashAttention 2 等工业实现中，会进一步把 $\log_2 e$ 提前并入 `sm_scale` 常量中直接调用 `exp2`，减少内层乘法指令（详见第 16 章）。

### Causal 整块跳过：三类块拆解与 ~50% 计算节省

开启自回归因果掩码后，原本的 $N \times N$ 注意力网格天然分裂成三类 Tile（以 $N = 128,\ B_r = B_c = 64$ 组成的 $2 \times 2$ 网格为例）：

- **全保留块（严格下三角）**：如行 $[64, 127]$、列 $[0, 63]$，块内所有位置都满足 $m \ge n$，全部保留（工业版甚至会把这类块拆出独立循环，连 `tl.where` 比较指令都一并省掉）。
- **对角块（跨越主对角线）**：如行 $[0, 63]$、列 $[0, 63]$ 与行 $[64, 127]$、列 $[64, 127]$，块内约一半有效、一半被掩码，需要逐元素执行 `offs_m[:, None] >= offs_n[None, :]`。
- **全跳过块（严格上三角）**：如行 $[0, 63]$、列 $[64, 127]$，块内最大行号 63 仍小于最小列号 64，整块所有元素均为未来 token！通过把内层循环上界设为 `hi = min((pid_m + 1) * BLOCK_M, N)`，直接跳过整块加载与两次 `tl.dot`。

在整个 $N \times N$ 矩阵中，因果有效位置仅有 $\frac{N(N+1)}{2}$ 个（$N=128$ 时为 $8{,}256 / 16{,}384 = 50.4\%$）。随着 $N$ 增大，整块跳过直接削掉接近 **50%** 的总 FLOPs 与 K/V 显存读取，而且**由于循环仍从 `start_n = 0` 出发**，第 0 列对任意行都合法，每一行在第一轮便脱离 $-\infty$，天然杜绝了全掩码行的 `NaN` 风险。

### Roofline 交叉点与隐蔽的 L2 复用边界

最后把 Fused Attention 放回第 1 章的 Roofline 坐标系，看清两个常被忽略的真实边界：

**边界一：序列长度的 Roofline 交叉点**。在非 Causal 条件下，Fused Attention 的理想算术强度与头维 $d$ 无关，只与序列长度 $N$ 成线性正比：

$$I_{\text{fused}} = \frac{4 N^2 d}{8 N d} = \frac{N}{2}\text{ FLOP/B}$$

令 $I_{\text{fused}} = I_{\text{ridge}}$，即得从带宽受限跃入计算受限的临界序列长度 $N^* = 2 \cdot I_{\text{ridge}}$：

- **A100 80GB**（$I_{\text{ridge}} = 156$）：交叉点 $N^* = \mathbf{312}$。短于 312 token 时，Fused Attention 依然落在带宽斜线上！
- **H100 80GB**（$I_{\text{ridge}} = 295$）：交叉点推后至 $N^* = \mathbf{590}$。

**边界二：被隐藏的 L2 缓存复用前提**。上面算出的 $8Nd = 512\text{ KiB}$ HBM 流量并非无条件成立！单头 $N = 1024, B_r = 64$ 时，16 个 Q Block 各自都要扫一遍完整的 $K$ 和 $V$（共 256 KiB），发出的全局加载请求总计高达 $16 \times 256\text{ KiB} = 4\text{ MiB}$。之所以实际 HBM 流量能贴近 $512\text{ KiB}$，完全仰仗 $256\text{ KiB}$ 的 $K, V$ 能安稳留在 A100 的 **40 MB L2**（H100 50 MB）中被后续 Q Block 反复命中。一旦在超长上下文或高并发多头推理中工作集冲爆 L2，$K/V$ 每次重读都会穿透回 HBM，流量从 512 KiB 飙升至 $4.25\text{ MiB}$，有效算术强度暴跌到约 **60 FLOP/B**，瞬间回落到带宽受限区！

到这里，我们已经手写了 GEMM、Fused Softmax 与单头 Fused Attention。不过真实大模型推理中，权重多已压成 INT8/INT4，序列动辄多头并发甚至跨越万级长度。接下来两章，第 15 章将切入 **W8A8 与 W8A16 量化 Matmul 算子**（含 L2 Swizzle 优化），第 16 章则顺着本章留下的伏笔，深入导读工业级 **FlashAttention 2/3 与 CUTLASS** 是如何榨干每一分硬件吞吐的。

---

## 参考文献

1. Tri Dao 等, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*（NeurIPS 2022，[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)），IO 感知分块注意力与 Online Softmax 融合原论文。
2. Maxim Milakov & Natalia Gimelshein, *Online normalizer calculation for softmax*（2018，[arXiv:1805.02867](https://arxiv.org/abs/1805.02867)），Online Softmax 算法原典。
3. OpenAI Triton 官方教程 `06-fused-attention`（含多头批处理、Warp 流水与 Hopper TMA 演进路径）。
