# 硬件规格与模型账本速查表 { .chapter-title }

<p class="chapter-summary">全书唯一规格事实源：统一主流 GPU/NPU 峰值算力与脊点、核心模型权重与 KV 容量账本，以及最容易踩坑的单位换算。</p>

推理性能工程的第一步，永远是在打开终端之前先拿纸笔算出物理上下界。为了保证全书 23 章与 5 篇专题的所有算例、公式和图表口径完全自洽，本手册收敛了全站统一的硬件公开规格、代表性模型几何账本与常用速算公式。阅读 [第 1 章](../chapters/chapter-01.md)、[第 2 章](../chapters/chapter-02.md)、[第 8 章](../chapters/chapter-08.md) 或 [专题 A](../topics/advanced-architectures.md) 时，可随时折返本页查表核对。

---

## 一、主流加速卡核心规格表

全书以 **NVIDIA A100 80GB SXM** 为默认主线算例，以 **NVIDIA H100 80GB SXM** 作为 FP8 / FlashAttention-3 / 集群 PD 前沿对照，以 **NVIDIA A40 48GB** 作为常见单机卡对照，并以 **华为昇腾 Ascend 910B（64GB）** 作为国产芯片主线底座：

| 规格维度 | NVIDIA A100 80GB SXM<br>（全书默认主线） | NVIDIA H100 80GB SXM<br>（前沿对照） | NVIDIA A40 48GB PCIe<br>（常见单卡对照） | 华为昇腾 Ascend 910B<br>（64GB，国产主线） |
| --- | --- | --- | --- | --- |
| **架构 / 计算能力** | Ampere GA100 / CC 8.0 | Hopper GH100 / CC 9.0 | Ampere GA102 / CC 8.6 | 达芬奇分离架构（AIC 矩阵核 / AIV 向量核） |
| **显存容量与介质** | **80 GB** HBM2e | **80 GB** HBM3 | **48 GB** GDDR6 ECC（可见约 44.42 GiB） | **64 GB** HBM2e |
| **峰值显存带宽** | **2 TB/s**（精确 2,039 GB/s） | **3.35 TB/s**（3,350 GB/s） | **696 GB/s** | **1.2–1.6 TB/s**（随 910B2/B3 等子型号分档） |
| **FP16 / BF16 稠密算力** | **312 TFLOPS**（稀疏 624） | **989.5 TFLOPS**（稀疏 1,979） | **74.8 TFLOPS**（稀疏 149.7） | **280–376 TFLOPS**（如 910B3 约 313 / 910B2 约 353+） |
| **INT8 Tensor 稠密** | **624 TOPS** | **1,979 TOPS** | **149.7 TOPS** | **560–752 TOPS**（约为 FP16 的 2×） |
| **FP8 Tensor 稠密** | 不支持原生 W8A8（仅 W8A16） | **1,979 TFLOPS**（原生 W8A8） | 不支持原生 W8A8（仅 W8A16） | 硬件无原生 FP8 矩阵乘（主流走 INT8 W8A8 / W4A16） |
| **FP32 / TF32 稠密** | 19.5 / 156 TFLOPS | 67 / 494.5 TFLOPS | 37.4 / 37.4 TFLOPS | 向量核 FP32 辅助计算（矩阵乘主力走 FP16/BF16） |
| **FP16 Roofline 脊点** | **156 FLOP/B**（按 2,039 GB/s 为 153） | **295.4 FLOP/B**（FP8 达 591） | **107.5 FLOP/B**（FP32 为 53.7） | **约 220–260 FLOP/B**（依具体子型号带宽算力比） |
| **计算单元与片上缓存** | 108 SM；164 KB Shared/SM（单 Block 上限 163 KB）；40 MB L2；2048 线程/SM | 132 SM；228 KB Shared/SM（单 Block 上限 227 KB）；50 MB L2；2048 线程/SM | 84 SM；100 KB Shared/SM（单 Block 上限 99 KB）；6 MB L2；1536 线程/SM | 分离式片上存储：AIV 独享 **UB**，AIC 配备 **L1 / L0A / L0B / L0C**，软件显式 DMA 搬运 |
| **片间互联 / 软件栈** | NVLink 600 GB/s / CUDA + NCCL | NVLink 900 GB/s / CUDA + TMA + NCCL | PCIe Gen4（可选桥接） / CUDA | HCCS + RoCE / CANN + AscendC + HCCL |

!!! warning "查硬件表最容易踩的四个口径坑"
    1. **原生 FP8 W8A8 的硬门槛是 $\text{CC} \ge 8.9$**：仅 Ada Lovelace（L40S / RTX 4090，CC 8.9）、Hopper（H100/H800，CC 9.0）及 Blackwell 支持硬件原生 FP8 矩阵乘。A100（8.0）与 A40（8.6）配置 `--quantization fp8` 时，底层走的是 **W8A16 在线反量化**（如 Marlin 内核），能省一半权重带宽，但算力仍落在 FP16 天花板上。
    2. **稠密算力 vs 2:4 稀疏算力差 2 倍**：NVIDIA 官方规格表常把 2:4 结构化稀疏算力印在最显眼的位置（例如把 H100 FP16 写成 1,979 TFLOPS）。常规 LLM 权重未经 2:4 稀疏剪枝，Roofline 推导一律采用**稠密算力**。
    3. **H100 的 989.5 vs 494.5**：989.5 TFLOPS 是 H100 SXM 的 **FP16/BF16 稠密**峰值，494.5 TFLOPS 是 **TF32 稠密**峰值，两者不可混用。
    4. **A40 不是“缩小版 A100”**：A40 的 FP32 算力（37.4 TFLOPS）反而高于 A100（19.5 TFLOPS），但它的 L2 缓存仅 6 MB（不到 A100 的 1/6）、显存带宽低近 3 倍、每 SM 线程上限为 1536。算子 Tile 与占用率必须按目标卡重算。

---

## 二、主流模型配置与显存账本表

下表统一列出全书使用的五款代表模型（涵盖纯 GQA 稠密、经典 MHA、混合线性注意力与稀疏 MoE）。单卡并发估算均按 **A100 80GB SXM、FP16/BF16、显存预算 `0.9`（约 $67.0\text{ GiB}$）、扣减权重与 $2\text{ GiB}$ 激活/工作区、忽略前缀共享与 Block 尾碎片** 计算，用于快速把握量级（实际可用 KV 池大小以引擎启动日志输出为准）：

| 维度 | Llama-3.1-8B<br>（默认主线） | Qwen3-8B<br>（默认对照） | LLaMA-13B<br>（经典 MHA 对照） | Qwen3.5-9B<br>（混合 DeltaNet 对照） | Qwen3.6-35B-A3B<br>（MoE + 混合对照） |
| --- | --- | --- | --- | --- | --- |
| **总参数 / 激活参数** | 8.03B / 8.03B | 8.19B / 8.19B | 13.02B / 13.02B | LM 9.2B / 9.2B（跳过视觉塔） | **34.66B / ~2.95B**（A3B） |
| **16-bit 权重体积** | **16.06 GB**（14.96 GiB） | **16.38 GB**（15.26 GiB） | **26.03 GB**（24.24 GiB） | **18.39 GB**（17.13 GiB） | **69.32 GB**（64.56 GiB） |
| **8-bit / 4-bit 权重（约）** | ~8.0 GB / ~4.5 GB | ~8.2 GB / ~4.6 GB | ~13.0 GB / ~7.2 GB | ~9.2 GB / ~4.6 GB | ~34.7 GB / ~19.5 GB |
| **层数 / $d$ / FFN $f$** | 32 / 4096 / 14336 | 36 / 4096 / 12288 | 40 / 5120 / 13824 | 32（8 Full + 24 DeltaNet） | 40（10 Full + 30 线性；MoE 256 选 8） |
| **注意力结构（Q / KV / $d_h$）** | **GQA**（32 / 8 / 128） | **GQA**（32 / 8 / 128） | **MHA**（40 / 40 / 128） | **混合**（Full 层 16 / 4 / 256） | **混合**（10 Full + 30 线性） |
| **每 token KV（16-bit）** | **128 KiB** | **144 KiB** | **800 KiB**（GQA 的 6.25×） | **32 KiB**（仅 Full 层） | **20 KiB**（仅 Full 层） |
| **每请求固定状态** | 0 | 0 | 0 | **约 49 MiB**（48 MiB FP32 + 卷积） | **61.875 MiB**（60 MiB FP32 + 卷积） |
| **8K 单请求状态** | **1.00 GiB** | **1.125 GiB** | **6.25 GiB** | **305 MiB**（0.298 GiB） | **221.875 MiB**（0.217 GiB） |
| **32K 单请求状态** | **4.00 GiB** | **4.50 GiB** | **25.00 GiB**（结构外推） | **1.05 GiB** | **701.875 MiB**（0.685 GiB） |
| **A100 80GB 典型并发（8K / 32K）** | 约 **50 / 12 条** | 约 **44 / 11 条** | 约 **6 / 1 条** | 约 **160 / 45 条**（A40 约 70 / 20） | BF16 单卡装不下大并发；**FP8 单卡约 150 / 47 条** |

> 注：4-bit 权重列已粗略计入典型分组量化（如 Group Size = 128）所需的缩放因子与零点元数据，以及未量化的词表嵌入/输出头权重；若对比更前沿的 Kimi K3、DeepSeek V4-Flash 与 V4.1 Flash，请查阅 [专题 A：前沿长上下文与混合架构深度拆账](../topics/advanced-architectures.md)。

---

## 三、推理工程核心速算公式卡

遇到任何新卡或新模型，直接套用下面五条公式即可在两分钟内搭出完整物理账本：

1. **每 token 的 16-bit KV 字节数（MHA / GQA / MQA 通用）**：

$$
c_{\text{KV}} = 2 \times L_{\text{full}} \times n_{\text{kv}} \times d_h \times b_{\text{KV}}
$$

其中系数 $2$ 代表 K 与 V 两份，$L_{\text{full}}$ 为全注意力层数，$n_{\text{kv}}$ 为 KV 头数，$d_h$ 为每头维度，$b_{\text{KV}}$ 为单元素字节数（FP16/BF16 取 $2$，FP8 取 $1$）。

2. **单请求状态曲线与固定/线性交点（含混合线性注意力）**：

$$
M_{\text{state}}(H) = M_{\text{fixed}} + c_{\text{KV}} \cdot H, \qquad H^* = \frac{M_{\text{fixed}}}{c_{\text{KV}}}
$$

3. **显存容量并发上限**：

$$
B_{\max}(H) = \left\lfloor \frac{M_{\text{device}} \cdot u - M_{\text{weight}} - M_{\text{workspace}}}{M_{\text{state}}(H)} \right\rfloor
$$

其中 $u$ 为显存预算比例（如 vLLM 默认 `gpu-memory-utilization = 0.9`）。

4. **单请求 Decode 权重搬运下界与权重/KV 读写平衡批大小**：

$$
t_{\text{decode,min}} = \frac{M_{\text{weight}}}{\text{BW}_{\text{peak}}}, \qquad B^* = \frac{R_W}{c_{\text{KV}} \cdot H}
$$

5. **权重主导 Prefill 计算下界与 Roofline 脊点**：

$$
t_{\text{prefill,min}} = \frac{2 N_{\text{active}} L}{C_{\text{peak}}}, \qquad \text{Ridge Point} = \frac{C_{\text{peak}}}{\text{BW}_{\text{peak}}}
$$

---

## 四、关键单位换算备忘

跨文档对账时，一半以上的“差几倍”或“差 7%”乌龙都源于单位混用：

| 容易混淆的一对单位 | 精确关系与换算 | 书中统一约定与防坑提示 |
| --- | --- | --- |
| **GB vs GiB**（十进制 vs 二进制） | $1\text{ GB} = 10^9\text{ B}$，$1\text{ GiB} = 2^{30}\text{ B} \approx 1.0737\text{ GB}$ | 显存标称容量、模型权重体积与显存带宽用十进制 **GB、TB/s**；KV 缓存单价、块大小与请求状态用二进制 **KiB、MiB、GiB**；换算并发时统一折成 GiB |
| **Gb/s vs GB/s**（比特 vs 字节） | $1\text{ GB/s} = 8\text{ Gb/s}$ | 网卡与 RDMA 标称速率用 **Gb/s**（如 400 Gb/s IB = 50 GB/s）；算 KV 传输时间务必先除以 8，混一次直接错 8 倍 |
| **FLOPs vs FLOPS** | **FLOPs**（小写 s）= 浮点运算**总量**；**FLOPS**（全大写）= **每秒**浮点运算速率 | 算一次前向工作量用 TFLOPs / GFLOPs，描述 GPU 算力上限用 TFLOPS |
| **MAC（乘加）vs FLOP** | $1\text{ MAC} = 2\text{ FLOPs}$（一次乘法 + 一次加法） | 矩阵乘 $A_{m \times k} \times B_{k \times n}$ 计算量恒为 $2 m k n\text{ FLOPs}$；部分硬件文档按 MAC 报数，直接翻倍即可对齐 |
| **ITL / TPOT vs 单请求速率** | $\text{tok/s} = 1000 / \text{ms\_per\_token}$ | $20\text{ ms/token}$ 对应 $50\text{ tok/s}$，$10\text{ ms/token}$ 对应 $100\text{ tok/s}$；单请求速率不能替代含 TTFT 的端到端速率或系统总吞吐 |
