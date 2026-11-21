# 第 12 章 GPU 执行模型与 GEMM 四版优化阶梯 { .chapter-title }

<p class="chapter-summary">从 Thread、Warp、SM 与三层存储层级出发，沿朴素版、Shared Memory 分块、向量化与 Bank Conflict 消除、2×2 线程粗化四级阶梯，看清 GPU 算子性能优化的真实杠杆。</p>

第二篇打开了推理引擎的调度与显存管理，但模型前向里最耗时的矩阵乘法（GEMM）与注意力算子仍停留在“调用库函数”的黑盒层面。本章正式潜入芯片内部：先建立 GPU 执行模型与存储层级的清晰心智，再以一个 $4096 \times 4096$ FP32 GEMM 为统一标尺，沿四级单变量优化阶梯逐层拆解——看清同一份 137.4 GFLOP 的计算量，如何仅凭“换数据落点、改指令宽度、调线程粒度”就跨越一个数量级的性能鸿沟。

---

## 12.1 GPU 执行模型与三层存储层级

为什么第 1 章算出 8B 模型单请求解码读 16 GB 权重只要 8 ms，而写得不好的自定义算子却可能比理论下界慢上几十倍？答案藏在 GPU“用极简控制换极致并行”的硬件哲学里：GPU 把芯片面积大比例让给算术逻辑单元（ALU / Tensor Core），几乎没有 CPU 那样庞大的分支预测器与乱序执行引擎，而是靠**层级化线程组织、三层存储分工与 Warp 切换隐藏延迟**来维持吞吐。

### 线程层级：Thread、Warp、Block 与 Grid

CUDA 将并行计算按三层软件抽象组织，并映射到两层硬件实体上：

- **Thread（线程）**：最小编程单位，拥有私有寄存器，通常负责计算 1 个或一小块（如 $2\times 2$）输出元素。
- **Warp（线程束）**：**硬件实际调度的最小执行单位**，固定由 32 个连续线程组成。同一 Warp 内的 32 个线程采用 **SIMT（Single Instruction, Multiple Threads）** 锁步执行同一条指令、处理各自不同的数据，共享一套取指与译码硬件以节省面积和功耗。
- **Block（线程块 / CTA）**：一组协作线程（常见 128、256 或上限 1024 个，即 4、8 或 32 个 Warp）。整个 Block 被完整分派到单个 **SM（Streaming Multiprocessor，流式多处理器）** 上执行，Block 内线程共享同一块 Shared Memory，并可通过 `__syncthreads()` 廉价同步。
- **Grid（网格）**：一次 Kernel 启动的所有 Block 集合。不同 Block 之间无法直接共享普通 Shared Memory，也不存在全局同步屏障，跨 Block 通信只能绕道 Global Memory（HBM）。

!!! tip "核心直觉：Block 是“便宜协作”与“昂贵协作”的分界线"
    设计任何 GPU 算子时，每当需要两个线程交换数据或等待彼此，先问一句：它们在同一个 Block 内吗？Block 内走片上 Shared Memory 或 Warp Shuffle 只需几十个时钟周期；跨 Block 必须落回 HBM 甚至拆成两次 Kernel 启动，代价高出几个数量级。

由于 Warp 内 32 个线程锁步执行，当它们在 `if / else` 处走向不同分支时，硬件只能将两条路径**串行执行**——执行 `if` 分支时 `else` 路径上的线程被屏蔽空等，反之亦然，这就是**分支分化（Warp Divergence）**。需要注意：数组末尾 `if (i < n)` 这类边界保护仅影响最后一个不满 32 线程的边缘 Warp，全局代价微乎其微；真正致命的是大量 Warp 在内层循环里长期分化。

每个线程通过内置坐标确定职责。一维索引为 $i = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}$，二维矩阵索引为：

$$\text{row} = \text{blockIdx.y} \cdot \text{blockDim.y} + \text{threadIdx.y}, \qquad \text{col} = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}$$

以 $4096 \times 4096$ 输出矩阵、$16 \times 16$ 线程块为例，每维需要 $\lceil 4096 / 16 \rceil = 256$ 个 Block，Grid 共 $256 \times 256 = 65{,}536$ 个 Block、$16{,}777{,}216$ 个线程，恰好覆盖全部输出元素。若矩阵尺寸非整除（如 100 万元素配 256 线程/Block，需 $\lceil 10^6 / 256 \rceil = 3907$ 个 Block），尾块多出的 192 个线程必须由边界判断拦截，否则越界写会静默破坏相邻张量。

### 三层存储层级：容量、延迟与可见性

物理电路无法同时做到“容量大”又“延迟低”，因此 GPU 围绕 SM 构建了三层核心存储结构（以 **NVIDIA A100 80GB SXM** 为基准，括号内附 **H100 80GB SXM** 对照）：

| 存储层级 | 物理位置与容量（A100 / H100） | 访问延迟 | 作用域（可见性） | 算子设计中的核心角色 |
|---|---|---|---|---|
| **Register（寄存器）** | 片上，64K 个 32-bit / SM（单线程上限 255 个） | ~1 周期 | 仅当前线程私有 | 存放累加器、循环变量与高频标量 |
| **Shared Memory（共享内存）** | 片上 SRAM，164 KB / SM（H100 228 KB / SM），单 Block 上限 163 KB | ~20–30 周期 | 同一 Block 内全部线程 | 由程序员显式管理的片上高速缓存，承载复用 Tile |
| **Global Memory（HBM）** | 片外显存，80 GB HBM2e @ 2 TB/s（H100 HBM3 @ 3.35 TB/s）；片上 L2 40 MB（H100 50 MB） | ~400–600 周期 | 整个 Grid 所有线程 | 存放模型权重、KV 缓存与算子输入输出张量 |

写 GPU Kernel 的本质，就是在这个物理妥协里把数据摆到正确的位置：只读一次的数据直接流过寄存器用完即弃；会被多个线程反复读取的数据（如 GEMM 的子矩阵块、Attention 的 $K/V$ 块），必须先搬进 Shared Memory 充分榨干复用次数。

### 合并访存（Coalescing）与延迟隐藏（Occupancy）

把数据放在正确层级之外，还有两条决定实测带宽与算力利用率的关键机制：

1. **合并访存（Memory Coalescing）**：硬件从显存搬运数据的最小粒度不是 4 字节的 `float`，而是 32 字节扇区（Sector）组成的 128 字节缓存行。
    - **理想连续访问**：同一 Warp 的 32 个线程读取 32 个相邻的 FP32 元素，总计 128 字节，硬件合并为 **1 次 128 字节事务**即可取完，有效带宽利用率 100%。
    - **跨步非连续访问**：若 32 个线程读取的地址彼此相隔 32 个 `float`（例如按列遍历行优先矩阵），每个线程落在不同的 32 字节扇区，硬件被迫发出 32 个扇区读取，实际搬运 $32 \times 32 = 1024$ 字节，而算子真正用到的只有 128 字节——整整 **8 倍显存带宽白白浪费在无用字节上**。
2. **Warp 延迟隐藏与占用率（Occupancy）**：读一次 HBM 需要数百个周期，SM 并不靠大缓存硬扛，而是靠**零开销上下文切换**掩盖延迟。当 Warp A 发出全局内存读取并陷入等待时，SM 的 Warp 调度器在下一个周期直接切换给就绪的 Warp B 发射算术指令。**占用率（Occupancy）**即单个 SM 上同时驻留的活跃 Warp 数占硬件上限（A100 为 64 Warp / 2048 线程，共 108 个 SM；H100 为 64 Warp / SM，共 132 个 SM）的比例。驻留 Warp 是调度器手里的“牌”：牌太少，所有 Warp 都在等访存时 SM 只能空转；但牌也不是越多越好，一旦为了追求高占用率而强行压低单线程寄存器配额，导致变量溢出（Spill）到片外 Local Memory，性能反而雪崩。

---

## 12.2 GEMM 四版阶梯方法论

为了看清每一种优化技术究竟撬动了硬件的哪一根杠杆，我们固定同一个计算任务构造四版单变量递进阶梯：在 **NVIDIA A100 80GB SXM** 上执行 $M = N = K = 4096$ 的 **FP32 矩阵乘法** $C = A \times B$。

### 为什么选 $4096 \times 4096$ FP32 GEMM

先算这笔固定的工作量账：输出矩阵共有 $4096^2 = 16{,}777{,}216$ 个元素，每个元素执行 $K = 4096$ 次乘加（FMA，计 2 FLOP），全矩阵总计算量为：

$$\text{FLOPs} = 2 M N K = 2 \times 4096^3 = 137{,}438{,}953{,}472\text{ FLOP} \approx 137.4\text{ GFLOP}$$

选择 FP32 CUDA Core 而非 FP16 Tensor Core 作为入门阶梯，原因很务实：Tensor Core 引入了特殊的矩阵分片指令（`mma.sync` / `wgmma`）与复杂 Swizzle 布局（留待第 16 章 CUTLASS 展开），而 FP32 路径让我们能用最干净的代码孤立出 **Tiling、向量化、Bank Conflict 与线程粗化** 四项底层通用机制。

在 A100 80GB 上，FP32 CUDA Core 稠密峰值为 **19.5 TFLOPS**，HBM2e 带宽为 **2 TB/s**，因此 **FP32 脊点**为：

$$I_{\text{ridge, FP32}} = \frac{19.5\text{ TFLOPS}}{2\text{ TB/s}} = 9.75\text{ FLOP/B}$$

（对照：A100 的 FP16 Tensor Core 峰值 312 TFLOPS，对应 FP16 脊点为 **156 FLOP/B**；H100 的 FP16 峰值 989.5 TFLOPS、带宽 3.35 TB/s，FP16 脊点 295 FLOP/B。）

由此直接算出这个 $4096^3$ FP32 GEMM 的两个极端锚点：

- **算力理论下界**：$137.44\text{ GFLOP} \div 19.5\text{ TFLOPS} \approx \mathbf{7.05\text{ ms}}$。
- **朴素版无复用搬运锚点**：若每个线程的逻辑读取全部穿透到 HBM（共计 $549.8\text{ GB}$），耗时达 $549.8\text{ GB} \div 2\text{ TB/s} \approx \mathbf{274.9\text{ ms}}$，与算力下界相差 **39 倍**！

无论代码怎么改，137.4 GFLOP 的乘加一次都不会少；四级优化阶梯缩小的，正是这 39 倍的数据搬运、指令发射与延迟暴露鸿沟。

<figure class="book-figure">
  <img src="../../../assets/figures/cuda-ladder.svg" alt="GEMM 四版优化阶梯示意图：从朴素基线经 Shared Memory 分块、float4 向量化与 Bank Conflict 消除、到 2×2 线程粗化与占用率调优，逐步逼近 Roofline 上限" loading="lazy">
  <figcaption><span class="book-figure__label">图 12-1</span>四版 GEMM 单变量优化阶梯。每版只引入一类硬件改动：第 2 版用分块把算术强度拉升 32 倍；第 3 版压缩访存指令数并消除片上冲突；第 4 版通过线程粗化减少 Shared 读取并平滑占用率。柱高仅示意递进方向，实际收益须结合硬件计数器与计时共同验证。</figcaption>
</figure>

### 计时标尺与 `ncu` 瓶颈归因原则

测量单 Kernel 耗时绝不能用 CPU 端墙钟时间（其中混杂了驱动启动、队列排队与主机同步抖动），而应在热身（Warmup）稳定 GPU 时钟与 L2 状态后，用同一 Stream 上的 **CUDA Events**（`cudaEventRecord` + `cudaEventSynchronize` + `cudaEventElapsedTime`）对连续多次启动取均值。

定位“为什么慢”则依赖第 4 章介绍的 Nsight Compute（`ncu`）。查看 `SpeedOfLight`、`Occupancy` 与 `MemoryWorkloadAnalysis` 时需遵循两条铁律：

1. **SM% 与 DRAM% 只做初筛，不作唯一判决**：`sm__throughput` 汇总了 SM 内多条子流水（FP32、LSU、SFU 等）的最高值，访存单元（LSU）指令拥堵同样会推高 SM%。因此必须结合**实际 DRAM 读写字节数、L1/L2 命中率、生成指令数（`LDG` / `LDS`）**与 **SchedulerStats 中的活跃/就绪/停顿 Warp（Active / Eligible / Stall Reasons）**交叉印证。
2. **计时与 Profile 分开采**：`ncu` 为读取硬件计数器会多次重放（Replay）Kernel，其报告内的时间不可当作真实性能数字；真实耗时永远看独立的 CUDA Events 计时。

```bash
# 使用 ncu 采集 SpeedOfLight、Occupancy 与关键吞吐计数器（跳过校验与热身启动）
ncu --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
    --kernel-name regex:gemm --launch-skip 6 --launch-count 1 \
    ./gemm_bench --m 4096 --n 4096 --k 4096 --version tiled
```

---

## 12.3 第 1–2 版：从朴素版到 Shared Memory 分块（Tiling）

### 第 1 版：朴素 GEMM（算术强度仅 0.25 FLOP/B）

最直觉的写法是让每个线程独占一个输出元素 $C[\text{row}, \text{col}]$，沿 $K$ 维循环 4096 次，每次直接从 Global Memory 各读一个 `A` 元素与 `B` 元素做乘加：

```cpp
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc = fmaf(A[static_cast<size_t>(row) * K + k],
                   B[static_cast<size_t>(k) * N + col], acc);
    }
    C[static_cast<size_t>(row) * N + col] = acc;
}
```

朴素版只做对了一件事：累加器 `acc` 驻留在寄存器中，输出 $C$ 只写回一次。它的致命伤在于**零数据复用**：

- 每个线程循环 $K = 4096$ 次，发出 $2K = 8192$ 次全局内存读取（每次 4 字节，共 32 KB）。
- 全矩阵 $4096^2$ 个线程合计发起逻辑读取量 $4096^2 \times 8192 \times 4\text{ B} = 549.8\text{ GB}$。
- $A$ 的每一个元素被同一行的 $N = 4096$ 个线程各读一遍，$B$ 的每一个元素被同一列的 $M = 4096$ 个线程各读一遍，每字节数据仅支撑：

$$I_{\text{naive}} = \frac{2 M N K}{4 \cdot 2 M N K} = \mathbf{0.25\text{ FLOP/B}}$$

0.25 FLOP/B 远低于 A100 的 FP32 脊点 9.75 FLOP/B。即便 L1/L2 缓存能在相邻线程间兜住一部分重复访问，海量全局 Load 指令与高访存延迟依然让算术单元长期处于饥饿状态。

### 第 2 版：32×32 Shared Memory 分块（算术强度跃升 32 倍）

消除重复读取的药方是**分块（Tiling）**：将 $A$ 和 $B$ 沿 $K$ 维切成 $B \times B$（取 $B = 32$）的小方阵，每个 $32 \times 32$ 的 Block 协作把当前迭代所需的两块 Tile 搬进片上 Shared Memory（`As` 与 `Bs`），在片上完成 32 次乘加后再推进下一块。

```cpp
constexpr int TILE = 32;

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    for (int kt = 0; kt < (K + TILE - 1) / TILE; ++kt) {
        int a_col = kt * TILE + tx;
        int b_row = kt * TILE + ty;
        As[ty][tx] = (row < M && a_col < K)
            ? A[static_cast<size_t>(row) * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N)
            ? B[static_cast<size_t>(b_row) * N + col] : 0.0f;
        __syncthreads();  // 屏障 1：确保 Tile 全部写入 Shared 后再开始计算

        for (int k = 0; k < TILE; ++k) {
            acc = fmaf(As[ty][k], Bs[k][tx], acc);
        }
        __syncthreads();  // 屏障 2：确保本轮计算读完 Shared 后再覆写下一块
    }
    if (row < M && col < N) {
        C[static_cast<size_t>(row) * N + col] = acc;
    }
}
```

现在重新核算这笔搬运账。每个 Tile 里的任意一个元素从 HBM 搬进 Shared Memory 后，被同一 Block 内 32 个线程复用 32 次：

- 每个 Block 沿 $K$ 迭代 $4096 / 32 = 128$ 轮，每轮搬入 `As` 与 `Bs` 共 $2 \times 32 \times 32 \times 4\text{ B} = 8\text{ KiB}$，128 轮合计 1 MiB。
- Grid 共有 $(4096/32)^2 = 16{,}384$ 个 Block，全矩阵逻辑全局读取总量骤降至 $16{,}384 \text{ MiB} \approx \mathbf{17.18\text{ GB}}$，精准压缩至朴素版的 $1/32$！
- 通用算术强度公式与 Tile 边长 $B$ 成正比：

$$I_{\text{tiled}} = \frac{2 B^3}{2 \times 4 B^2} = \frac{B}{4}\text{ FLOP/B}$$

当 $B = 32$ 时，$I_{\text{tiled}} = \mathbf{8\text{ FLOP/B}}$，较朴素版跃升 **32 倍**！在无 L2 复用的保守模型下，可达性能上界从 0.5 TFLOPS 暴涨至 $\min(8 \times 2000, 19500) = \mathbf{16\text{ TFLOPS}}$（对应参考时间 **8.59 ms**），已经逼近 A100 的 FP32 脊点（9.75 FLOP/B）。

!!! warning "常见误区：少写任意一个 `__syncthreads()` 都会埋下竞态炸弹"
    循环体内的两次屏障缺一不可：漏掉第一个，跑得快的 Warp 会读到尚未写入的旧垃圾（写后读 RAW 冒险）；漏掉第二个，跑得快的 Warp 进入下一轮直接覆写 `As/Bs`，踩坏慢 Warp 正在读取的当前 Tile（读后写 WAR 冒险）。由于 Warp 调度具有不确定性，缺屏障的 Bug 往往时好时坏，极难排查。

---

## 12.4 第 3–4 版：`float4` 向量化、Bank Conflict 消除与 2×2 线程粗化

把算术强度抬到 8 FLOP/B 后，显存总流量大降，此时瓶颈会下沉到两个更隐蔽的层面：**单指令的访存吞吐效率**、**片上 Shared Memory 的 Bank 冲突**，以及 **1024 线程超大 Block 引发的 Occupancy 悬崖与 Shared 带宽瓶颈**。

### 第 3 版前半：`float4` 向量化访存——把 4 条指令压成 1 条

先分清两个常被混为一谈的概念：

- **合并访存（Coalescing）**解决的是“**跨线程**地址是否连续”——第 2 版的标量读已经满足合并（同 Warp 内 `tx` 连续，32 线程正好读 128 字节）。
- **向量化访存（Vectorization）**解决的是“**单线程单条指令**搬多少字节”。SM 每个周期的指令发射槽位有限，1 条 32-bit `LDG.E`（搬 4 字节）和 1 条 128-bit `LDG.E.128`（借助 `float4` 一次搬 16 字节）同样消耗 1 个发射槽。

改用 `float4` 搬运每个 $32 \times 32$ Tile 时，1024 个元素只需 **256 条** `float4` 加载指令（而非 1024 条标量加载），指令数直降为 **1/4**。对全矩阵 $16{,}384$ 个 Block × 128 轮 × 2 块 Tile 而言，总计直接省掉约 **32.2 亿条**全局 Load 指令！向量化不改变 17.18 GB 的总搬运字节数，也不改变算术强度，但显著释放了指令发射队列与 LSU 压力。

`float4` 要求首地址 16 字节对齐（`cudaMalloc` 保证基址对齐，且当行宽 $K, N$ 为 4 的倍数、行内列偏移为 4 的倍数时满足对齐）；遇到非 4 倍数的尾部或未对齐矩阵，需要像参考源码中的 `load_tile4` 那样优雅退回带掩码的标量加载。同时切记：`__syncthreads()` 绝不能放在 `if (tid < 256)` 内部，Block 内所有线程必须无条件到达同一个同步点。

### 第 3 版后半：32-Bank Conflict 的模算术与 `+1` Padding

视线转向片上 Shared Memory。为支撑同一 Warp 32 个线程的单周期并发读写，GPU 将 Shared Memory 划分为 **32 个独立存储体（Bank）**，每个 Bank 宽 4 字节（正好 1 个 `float`），连续的 `float` 按轮询方式交错分布：

$$\text{bank}(a) = a \bmod 32$$

同一 Warp 内的 32 个线程在单条指令中访问 Shared Memory 时，存在三种硬件行为：

1. **无冲突并行（最优）**：32 个线程落在 32 个不同 Bank，单周期全部完成。例如第 2 版读 `Bs[k][tx]`，线性地址 $32k + tx$，对应 $\text{bank} = tx \bmod 32$，完美零冲突。
2. **广播（Broadcast，同样 0 冲突）**：多个线程访问**完全相同的地址**，硬件只需读一次并广播给所有线程。例如第 2 版读 `As[ty][k]`，同一 Warp 内 `ty` 固定、仅 `tx` 变化，32 个线程读取同一个 float，零冲突。
3. **Bank Conflict（串行惩罚）**：同一 Warp 内多个线程访问**同一 Bank 的不同地址**，硬件被迫将它们拆成多轮串行响应。若 32 个线程全部撞在同一个 Bank，就会触发最恶劣的 **32 路 Bank Conflict**，Shared 延迟直接暴涨 32 倍！

什么时候会撞上 32 路冲突？假设为了某些算法需要把 `B` 的 Tile 转置存入 `__shared__ float BsT[32][32]`，计算时改为按列读 `BsT[tx][k]`：此时线性地址变为 $32 \cdot tx + k$，取模结果：

$$\text{bank}(32 \cdot tx + k) = (32 \cdot tx + k) \bmod 32 = k$$

结果与 `tx` 无关！32 个线程全部指向同一个 Bank $k$ 里的 32 个不同地址，读写两端双双触发 **32 路串行冲突**。

解法精巧得只需多花 3% 空间——**行尾 `+1` Padding**：把数组声明为 `__shared__ float BsT[32][33]`。多出的第 33 列不存任何有效数据，仅用来错开每行的 Bank 起始相位：

$$\text{bank}(33 \cdot tx + k) = (32 \cdot tx + tx + k) \bmod 32 = (tx + k) \bmod 32$$

固定任意 $k$，当 `tx` 遍历 $0 \sim 31$ 时，$(tx + k) \bmod 32$ 严格构成 $0 \sim 31$ 的一个置换——32 个线程精准散落到 32 个不同 Bank，32 路冲突瞬间归零！

<figure class="book-figure">
  <img src="../../../assets/figures/gpu-memory-tiling.svg" alt="GPU GEMM 分块滑动（Tiling）与 Shared Memory 32-Bank 错位图解：左半展示 A 行条带与 B 列条带沿 K 维逐块载入片上 As、Bs 复算 BLOCK 次使算术强度从 0.25 跃升至 8.0；右半对比 BsT[32][32] 按列读撞在同一 Bank 与 BsT[32][33] 加一列 Padding 零冲突并行" loading="lazy">
  <figcaption><span class="book-figure__label">图 12-2</span>左图：GEMM 的片上分块滑动链——将 <code>A</code>、<code>B</code> 子块沿 <code>K</code> 维搬入 Shared Memory 复算 32 次，算术强度从 0.25 跃升至 8.0 FLOP/B，累加器 <code>C_tile</code> 全程驻留寄存器。右图：32-Bank 模算术对照——<code>BsT[32][32]</code> 让同 Warp 32 线程齐撞同一 Bank，补 1 列变为 <code>BsT[32][33]</code> 后相位错开到 Bank 0..31，冲突彻底归零。</figcaption>
</figure>

### 第 4 版：2×2 线程粗化（Thread Coarsening）与 Occupancy 调优

回到第 2、3 版的资源结构，还潜伏着两个棘手缺陷：

1. **Occupancy 断崖**：$32 \times 32 = 1024$ 线程的 Block 粒度过粗。A100 每 SM 上限 2048 线程（最多 2 个 Block）和 65,536 个寄存器。一旦编译器为每个线程分配超过 32 个寄存器（如 40 个），单 SM 驻留 Block 数便从 2 个腰斩到 1 个，理论占用率瞬间从 **100% 跌至 50%**；且第 3 版搬运 `float4` 时，1024 个线程里只有前 256 个干活，其余 768 个线程空等。
2. **算术/Shared 读取比偏低**：每个线程只算 1 个输出元素，内层每执行 1 次 FMA 需要从 Shared Memory 读 2 次数据（`As[ty][k]` 与 `Bs[k][tx]`），Shared 带宽与 `LDS` 指令极易成为新的天花板。

**线程粗化（Thread Coarsening）**一石二鸟：维持 $32 \times 32$ 的 Tile 与 8 KiB Shared Memory 不变，将 Block 缩小为 $16 \times 16 = 256$ 线程（8 个 Warp），让**每个线程在寄存器中维护一个 $2 \times 2$ 累加器阵列** `acc[2][2]`，一次性算出 4 个相邻输出元素。以下是仓库完整基准 [gemm_bench.cu](../modules/part3-operators/examples/gemm_bench.cu) 中的第四版核心实现：

```cpp
// Tile 维持 32x32，Block 改为 16x16=256 线程，每线程负责 2x2 共 4 个输出
__global__ __launch_bounds__(256) void gemm_tiled_2x2(
    const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE * TILE];
    __shared__ float Bs[TILE * TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * 16 + tx;                 // 0..255，全部 256 线程满载参与 float4 搬运
    int r0 = blockIdx.y * TILE + 2 * ty;
    int c0 = blockIdx.x * TILE + 2 * tx;
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int kt = 0; kt < (K + TILE - 1) / TILE; ++kt) {
        load_tile4(A, As, M, K, blockIdx.y * TILE, kt * TILE, tid);
        load_tile4(B, Bs, K, N, kt * TILE, blockIdx.x * TILE, tid);
        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    acc[i][j] = fmaf(As[(2 * ty + i) * TILE + k],
                                     Bs[k * TILE + 2 * tx + j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (r0 + i < M && c0 + j < N) {
                C[static_cast<size_t>(r0 + i) * N + c0 + j] = acc[i][j];
            }
        }
    }
}
```

对照第 2、3 版，这几十行代码同时带来三重实质收益：

- **Shared Memory 访存减半**：内层每个 `k` 步，编译器把 `As` 的 2 个元素与 `Bs` 的 2 个元素载入寄存器后交叉完成 4 次 FMA——平均每次 FMA 只需 **1 次** Shared Load（原为 2 次），片上复用再升一级！工业级 GEMM（如第 16 章 CUTLASS）会将每线程微块进一步扩大到 $8 \times 8$，把 Shared 读取摊薄到每 FMA 仅 $0.25$ 次。
- **Occupancy 曲线平滑可控**：Block 降至 256 线程后，A100 每 SM 线程上限支持最多 8 个 Block。即便寄存器涨至 40 个/线程，仍可驻留 $\lfloor 65536 / (256 \times 40) \rfloor = 6$ 个 Block（1536 线程，75% 理论占用率），告别 100% $\to$ 50% 的悬崖。
- **全员搬运无空闲**：256 个线程与 `load_tile4` 所需的 256 次 `float4` 读入严丝合缝，消除了前版 75% 线程在搬运阶段的闲置。

!!! note "工程取舍：`__launch_bounds__` 与 Tile 尺寸的平衡"
    `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` 是向编译器声明块大小上限与期望驻留数的提示，源码里写 `__launch_bounds__(256)` 可让编译器放宽指令调度而不盲目压榨寄存器。切忌两类反模式：**一是为了抬高占用率而强行用过紧的 `minBlocks` 压低寄存器配额**，一旦触发 Register Spilling（溢写到片外 Local Memory），耗时立刻飙升；**二是为了抬高占用率而缩小 Tile**——将 `TILE` 从 32 缩到 16 会让算术强度直接腰斩（$8 \to 4\text{ FLOP/B}$），理论性能上限砍半，得不偿失。

### 四版 GEMM 优化阶梯全景对照

下表汇总四版 Kernel 在 $M = N = K = 4096$ FP32 GEMM（137.4 GFLOP，A100 80GB SXM 口径）下的理论账本与 `ncu` 关键归因指标：

| 版本（源码标识） | Block 线程 / Tile | 逻辑 HBM 读量 | 逻辑算术强度 | 无缓存参考上界（A100） | 核心优化杠杆 | `ncu` 重点验证指标 |
|---|---|---|---|---|---|---|
| **V1 朴素版** (`gemm_naive`) | $16 \times 16$ (256) / 无 | 549.8 GB | $0.25\text{ FLOP/B}$ | 0.50 TFLOPS (274.9 ms) | 基线：仅输出累加器驻留寄存器 | 高 DRAM 流量、极低 L1/L2 复用、长访存 Stall |
| **V2 分块版** (`gemm_tiled`) | $32 \times 32$ (1024) / $32 \times 32$ | 17.18 GB | $8.0\text{ FLOP/B}$ | 16.0 TFLOPS (8.59 ms) | Shared Memory 片上 32× 复用 | 全局请求与实际 DRAM 字节骤降、L1/TEX 请求大幅减少 |
| **V3 向量化版** (`gemm_vectorized`) | $32 \times 32$ (1024) / $32 \times 32$ | 17.18 GB | $8.0\text{ FLOP/B}$ | 16.0 TFLOPS (8.59 ms) | `float4` 128-bit 读压 4× Load 指令；Shared 布局防 Bank 冲突 | `LDG.E.128` 替代 4× `LDG.E`、Shared Bank Conflict 计数归零 |
| **V4 粗化版** (`gemm_tiled_2x2`) | $16 \times 16$ (256) / $32 \times 32$ | 17.18 GB | $8.0\text{ FLOP/B}$ | 16.0 TFLOPS (8.59 ms) | 每线程 $2\times 2$ 寄存器微块，Shared 读减半 + 消除 Occupancy 悬崖 | `LDS` 指令数减半、Active/Eligible Warps 提升、无 Local Spill |

完整可编译运行的四版源码（含固定种子初始化、全量/抽样 CPU 参考校验与 CUDA Events 计时框架）位于 [gemm_bench.cu](../modules/part3-operators/examples/gemm_bench.cu)。

回望这四级阶梯，每一步改动的物理机制都清晰可见：分块减少**显存搬运总量**，向量化与 Padding 扫清**访存指令与片上冲突**，线程粗化平衡**寄存器复用与 SM 延迟隐藏**。但你也一定感受到了手写 CUDA 的繁琐——每调一次 Tile 尺寸，线程映射、共享内存布局与同步边界就要全盘重写。下一章，我们引入 **Triton**，看看把这些繁琐细节交给编译器之后，算子开发会变成什么模样。

---

## 参考文献

1. NVIDIA, *CUDA C++ Programming Guide* & *Nsight Compute Profiling Guide*（硬件存储层级、Coalescing、Shared Memory Bank 与 `ncu` 指标定义权威文档）。
2. Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog*（2022），从朴素版到逼近 cuBLAS 的 10 级工程拆解。
3. 本书配套四版完整参考实现：[gemm_bench.cu](../modules/part3-operators/examples/gemm_bench.cu) 与[硬件规格与模型账本速查](../toolkit/hardware-models.md)。
