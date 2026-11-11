# M2 · 算子优化

> 状态：**内容进行中**——第 1–4 周为完整内容（加深版），第 5–6 周仍为大纲，见下文各周表格。
>
> 前置：M0（Roofline、profiling、压测脚本）与 M1 第 1–2 周（vLLM 服务已跑起来）。M2 的终点不是“会写 kernel”，而是“自己写的 kernel 跑进了推理引擎，端到端变快了”。
>
> 每节 30 分钟，每周 5 节，沿用 M1 的节奏。**每节开头标硬件门槛**：无卡可做 / 需 GPU / 需租卡，并给出无卡替代路径。代码示例分两种：能直接跑的（已按官方教程与源码核对过写法）和示意性的（标明“示意”二字，只讲结构不保证编译通过）——不要把示意代码当可运行代码。
>
> **本模块的深度标准**：优化做成阶梯，每一步都有实测数字，并且用 M0 的 Roofline 定位每一步卡在哪里；优化由 profiling（`ncu`）驱动，先定位瓶颈再改 kernel，而不是只看时间。原来的 5 周塞不下这些内容，M2 已扩为 6 周：第 3 周专讲 fused attention 的数学推导，第 4 周专讲量化 matmul 的量化方案，第 5 周专讲 FlashAttention 与 CUTLASS 的工业级实现，第 6 周收口到引擎接入。

## 为什么单开一个模块

M1 让你会用引擎、会调参数，但引擎内部的 attention、matmul、量化 kernel 对你还是黑盒。M2 的目标是打开这个盒子：先学会写 kernel，再把 kernel 接回引擎，最后用 M0 的压测脚本量出端到端的变化。现有的练习资源（GPU Puzzles、Triton Puzzles、LeetGPU）都在 kernel 层面收尾，**“接入引擎”这一步目前没有现成课程**，是本模块要补的关键一节。

## 第 1 周 · CUDA 优化阶梯：一个 GEMM，四次提速

> 硬件：**本周需要 GPU 与 CUDA 工具链（`nvcc`）**。租卡或本地卡均可；Colab 免费 GPU 能跑 GPU Puzzles，但阶梯实验建议在自己能装 `ncu` 的机器上做（`ncu` 在容器里可能需要 `sudo` 或 `CAP_SYS_ADMIN` 权限，跑不起来先查权限，见第 2 节）。
>
> 本周的主线是一个 4096×4096 的 FP32 矩阵乘法（记为 GEMM），从朴素版一路优化到接近硬件上限。每一节的结构都一样：**先用 `ncu` 读瓶颈 → 改 kernel → 再测 → 把点标到 Roofline 上**。不要跳过“读瓶颈”这一步——没有 profile 的优化是猜。

### 第 1 节 · GPU 执行模型与第一个 kernel

**目标**：能画出一次 kernel 启动的数据流向，并跑通第一个 kernel。

**怎么做**（30 分钟）：

1. 读 [GPU Puzzles](https://github.com/srush/GPU-Puzzles) 的前言：它用 Numba 把 Python 代码直接映射成 CUDA kernel，“看起来像 Python，实际就是 CUDA”。再看 [Stanford CS336 第 5 讲](https://www.youtube.com/watch?v=6OBtO9niT00)前 30 分钟（GPU 与显存层级）。
2. 记住三层结构：**thread**（最小执行单位）、**block**（一组线程，共享一块 shared memory，block 内可以同步）、**grid**（所有 block 的集合）。`blockIdx` 是第几个 block，`threadIdx` 是 block 里的第几个线程。
3. 记住显存层级：global memory（大、慢，所有线程可见）→ shared memory（小、快，block 内共享）→ register（最快，每个线程私有）。M0 的 Roofline 在这里落地：一次访存没进 shared memory，就等于又去 global memory 搬了一遍。
4. 做 Puzzle 1（Map）、Puzzle 2（Zip）、Puzzle 3（Guards）：一个线程负责一个元素；线程数多于元素数时加边界判断 `if i < size`。每个谜题的验收都是 `problem.check()`——把你的 kernel 输出与 NumPy 参考实现逐项对比。**以后你写的每个 kernel 都要配一个这样的参考实现**，这是本模块的铁律。
5. 环境注意：Numba 内置的 CUDA 目标已被标记为弃用（后续开发移到了 NVIDIA 的 `numba-cuda` 包），所以**以 GPU Puzzles 仓库自己锁定的 numba 版本与安装说明为准**，不要自行升级 numba。按仓库 README 的说明装好环境（Colab：把 GPU 运行时打开；本地：clone 仓库后用 Jupyter 跑）。

**产出**：Puzzle 1–3 通过 + 一张数据流向图（输入在 global memory → 按 block 分片 → 每个 thread 算一个元素 → 写回），并标出 threadIdx、blockIdx、blockDim 各自的含义。

### 第 2 节 · 搭测量台：计时、Roofline、第一次 ncu 读数

**目标**：有一套“先算、再测、再定位”的测量台，本周后面四节全部复用它。

**怎么做**（30 分钟，需 GPU）：

1. 写朴素版 GEMM（一个线程算一个输出元素，直接读 global memory）：

```cpp
__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}
```

2. 写计时 harness：用 CUDA Events（不要用 CPU 时钟量 GPU，中间隔着启动延迟）：

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start);
gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms = 0; cudaEventElapsedTime(&ms, start, stop);
```

3. **先算**：M=N=K=4096，FP32。计算量 = 2×4096³ ≈ 137.4 GFLOP。朴素版每个输出元素读 K 个 A 元素 + K 个 B 元素，搬运量 = 8×4096³ ≈ 550 GB，算术强度 = 0.25 FLOP/字节。以 A100 为例（FP32 峰值约 19.5 TFLOPS，HBM 带宽约 2 TB/s，数字以厂商公开规格为准）：Roofline 上界 = min(0.25 × 2000, 19500) = 500 GFLOP/s，4096³ 至少要跑 274 ms。**先把这个数写下来，再跑**——实测比它慢得多是正常的（访存还没合并），实测比它快是不可能的（说明你的计时或计算错了）。
4. **再读 ncu**：`ncu`（Nsight Compute）是定位 kernel 瓶颈的工具，先看 SpeedOfLight 一节的两个数：

```bash
ncu --section SpeedOfLight --section Occupancy --kernel-name regex:gemm -o gemm_v1 ./gemm_bench
```

也可以直接取两个指标：

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active --kernel-name regex:gemm ./gemm_bench
```

分类规则（两个数对照着看）：SM% 高（>60）、DRAM% 低（<40）→ 计算受限；SM% 低、DRAM% 高（>70）→ 访存受限；两个都低（<40）→ 延迟受限（等数据、等同步，不是算力也不是带宽的问题）。朴素版你应该看到 DRAM% 很高、SM% 很低——这是预期，说明瓶颈在访存，不在计算。
5. 注意：`ncu` 默认会 profile 所有 kernel 启动，用 `--launch-skip` 跳过热身、`--launch-count` 只采几次，避免采到初始化代码。容器里跑不起来时先查权限（`ncu` 可能需要 `sudo` 或 `CAP_SYS_ADMIN`），而不是怀疑工具。

**产出**：一份测量台记录——朴素版的实测耗时、实测算力（GFLOP/s = 137.4 / 秒数）、占 Roofline 上界的百分比、ncu 的 SM% 与 DRAM%，以及一句话瓶颈判断。

### 第 3 节 · 第一步：Shared memory 分块

**目标**：把重复读的数据搬进 shared memory，量出访存量下降带来的提速，并把点在 Roofline 上向右移。

**怎么做**（30 分钟，需 GPU）：

1. **先读 ncu**：第 2 节的结论是访存受限。为什么？朴素版里 A 的同一行被 N 个输出元素重复读，B 的同一列被 M 个输出元素重复读——每次重复都是一次 global memory 访问。
2. 改 kernel：分块（tiling）。把 A 的一块（BLOCK×BLOCK）与 B 的一块先搬进 shared memory，在 shared memory 里算部分和，再搬下一块：

```cpp
#define BLOCK 32
__global__ void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BLOCK][BLOCK];
    __shared__ float Bs[BLOCK][BLOCK];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BLOCK + ty;
    int col = blockIdx.x * BLOCK + tx;
    float acc = 0.f;
    for (int t = 0; t < (K + BLOCK - 1) / BLOCK; ++t) {
        int a_col = t * BLOCK + tx;
        int b_row = t * BLOCK + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < BLOCK; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
```

3. **先算**：分块后 A 的每个元素被读 N/BLOCK 次，B 的每个元素被读 M/BLOCK 次，搬运量 = 8×4096³/32 ≈ 17.2 GB，算术强度 = BLOCK/4 = 8 FLOP/字节。Roofline 上界 = min(8 × 2000, 19500) = 16 TFLOPS，4096³ 至少 8.6 ms——比朴素版的上界快了 32 倍。**这个 32 倍不是实测，是分块省掉的重复搬运算出来的**；实测能拿到多少，看 ncu。
4. **再测再读**：跑计时 + ncu。你应该看到 DRAM% 明显下降、SM% 上升。把两个版本的点都标到同一张 Roofline 图上：朴素版在左下（低算术强度、低算力），分块版向右上方移动。
5. 理解两个细节：`__syncthreads()` 必须成对出现（搬完同步一次、算完同步一次），否则有的线程会读到别的线程还没写完的 shared memory；`As[ty][k]` 是广播（同一 warp 的线程读同一地址，不冲突），`Bs[k][tx]` 是连续地址（不冲突）——这个布局是故意选的，下一节会故意写错一次给你看后果。

**产出**：分块版的实测耗时与算力、ncu 前后对比（SM% / DRAM% 各变了多少）、更新后的 Roofline 图（两个点）。

### 第 4 节 · 第二步：向量化访存与 bank conflict

**目标**：把访存从“每次 4 字节”改成“每次 16 字节”，并亲手制造一次 bank conflict、再修掉它。

**怎么做**（30 分钟，需 GPU）：

1. **先读 ncu**：分块版之后，瓶颈一般还在访存一侧（MemoryWorkloadAnalysis 一节），但换了一个形态：不是搬运总量，而是**访存指令数**——每个线程每次只读 4 字节，要发很多次 load 指令。
2. 改 kernel：向量化加载。把分块搬运时的标量读改成 `float4`（一次读 16 字节，4 个 float）：

```cpp
// 示意：分块搬运时，一次搬 4 个 float
float4 a4 = *reinterpret_cast<const float4*>(&A[row * K + a_col]);
// a4.x, a4.y, a4.z, a4.w 是连续的 4 个元素，写入 As[ty][tx*4 .. tx*4+3]
```

`float4` 要求地址 16 字节对齐（`cudaMalloc` 返回的地址天然对齐；行内偏移是 4 的倍数时也对齐）。改完再测：load 指令数降到约 1/4，耗时应该继续下降。
3. **故意制造一次 bank conflict**：把 `Bs[k][tx]` 的布局改成转置的 `Bs[tx][k]` 写法跑一遍，再改回来。shared memory 有 32 个 bank，连续地址落在不同 bank；转置写法让同一 warp 的 32 个线程同时读同一个 bank 的不同地址——32 路冲突，串行化 32 倍。用 ncu 的 Shared Memory 相关计数看变化（`--set full` 报告里找 Shared Memory / bank conflict 一节），再对照改回来的版本。
4. 记住结论：**访存合并解决的是 global memory 的合并问题，bank conflict 解决的是 shared memory 的并行问题**，两者都要对，缺一个都会在 ncu 里露出来。

**产出**：向量化前后的耗时对比 + 一次 bank conflict 的实测记录（改错版 vs 改回版的耗时与 ncu 计数），加一句话：你的 kernel 现在卡在哪。

### 第 5 节 · 第三步：Occupancy 调优 + 阶梯总表

**目标**：调 block 大小与寄存器用量，让 SM 上同时驻留的 warp 数上去，并把四步的结果汇总成一张阶梯表。

**怎么做**（30 分钟，需 GPU）：

1. **先读 ncu**：Occupancy 一节看两个数——理论占用率（受寄存器数、shared memory 用量、block 大小限制）与实测占用率（`sm__warps_active.avg.pct_of_peak_sustained_active`）。如果理论占用率本身就低，问题在资源用量；如果理论高、实测低，问题在延迟隐藏不足（等访存时没有别的 warp 可切）。
2. 调两个旋钮：block 大小（256 线程 vs 512 线程，改 `dim3 block` 的启动配置）与寄存器上限（`__launch_bounds__(256)` 告诉编译器每个 block 最多 256 线程，让它控制寄存器用量以换取更多驻留 block）。每改一次，用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 查理论值，再用 ncu 看实测值：

```cpp
int num_blocks = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, gemm_tiled, 256, 0);
```

3. 注意：占用率不是越高越好。分块版的 shared memory 用量（两块 32×32×4B = 8KB）本身就会限制驻留 block 数；盲目追高占用率而缩小 BLOCK，反而会把算术强度打回去。**调优的目标是实测耗时下降，不是某个指标好看。**
4. 写阶梯总表（本周的可交付物），每一行都要有实测数，没有实测的写“待补测”：

| 版本 | 算术强度 | 实测耗时 | 实测算力 | 占 Roofline 上界 | ncu SM% / DRAM% | 瓶颈判断 |
|---|---|---|---|---|---|---|
| 朴素版 | 0.25 | | | | / | 访存受限 |
| 分块版 | 8 | | | | / | |
| 向量化版 | 8 | | | | / | |
| Occupancy 调优版 | 8 | | | | / | |

5. 把本周的检查清单并进第 1 节的清单，加三条：“ncu 读了吗（先定位再改）”“Roofline 点标了吗（每改一步标一次）”“实测数填表了吗（没测的写待补测）”。

**产出**：这张阶梯总表 + 更新后的检查清单。本周结束时，你应该能讲清：GEMM 的每一步优化省的是什么（重复搬运 / 访存指令数 / 延迟隐藏），以及 ncu 的哪个数证明了它。

## 第 2 周 · Triton 入门：用 Python 写 kernel

> 硬件：**本周无需 GPU**。Triton Puzzles 用解释器跑；跑性能对比时再上 GPU。计时方法用 `triton.testing.do_bench`（见第 3 节），它与第 1 周的 CUDA Events 是同一回事：热身、多次重复、取中位数。

### 第 1 节 · Triton 语法与 block 编程模型

**目标**：写出第一个能跑的 Triton kernel，理解 block 编程模型与 CUDA 的区别。

**怎么做**（30 分钟，无 GPU 可读、可写；运行需 GPU 或留到第 2 节的解释器环境）：

1. 读 [Triton 官方教程](https://github.com/triton-lang/triton/tree/release/3.6.x/python/tutorials)的 `01-vector-add`：Triton 的核心思想是**按 block 编程**——你写的不是“一个线程做什么”，而是“一个程序处理一整块数据”。`tl.program_id(0)` 告诉你现在是第几个程序，`tl.arange(0, BLOCK)` 生成一块连续的下标。
2. 对照 CUDA：CUDA 里 `threadIdx.x` 是标量下标，Triton 里 `offsets` 是一整块向量下标；CUDA 里你要自己算 `blockIdx * blockDim + threadIdx`，Triton 里 `pid * BLOCK + tl.arange(0, BLOCK)` 是同一回事，只是从标量变成了向量。
3. 写出向量加法（官方教程 `01-vector-add` 的写法，`BLOCK` 是编译期常量）：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def add(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK=1024)
    return output
```

4. 注意三个细节：`mask` 处理尾巴（元素数不是 BLOCK 整数倍时，多余的位置不读写）；`BLOCK: tl.constexpr` 表示编译期常量，改它会重新编译；`grid` 是个 lambda，`triton.cdiv` 是向上取整除法。
5. 正确性检查：与 `x + y` 的 PyTorch 结果做 `torch.allclose` 对比——第 1 周的铁律继续有效。

**产出**：第一个能跑的 Triton kernel（或无卡时先写好代码、标注待 GPU 验证），加一页 CUDA vs Triton 对照笔记。

### 第 2 节 · 谜题 1–3：逐元素、归约、矩阵乘

**目标**：用解释器做完 Triton Puzzles 的前三道题，理解 Triton 里“访存模式决定一切”。

**怎么做**（30 分钟，**无需 GPU**）：

1. 打开 [Triton Puzzles](https://github.com/gpu-mode/Triton-Puzzles)（基于 [Triton-Viz](https://github.com/gpu-mode/triton-viz)，用 Triton 解释器跑，不需要 GPU）。按仓库 README 的说明启动：解释器模式由环境变量 `TRITON_INTERPRET=1` 开启，notebook 顶部有完整说明。轻量替代：[Triton-Puzzles-Lite](https://github.com/SiriusNEO/Triton-Puzzles-Lite)（最小环境配置即可玩）。
2. 谜题 1（逐元素操作）：把第 1 节的向量加法换成谜题形式做一遍，重点看解释器给出的访存可视化——每个程序读了哪块、写了哪块。
3. 谜题 2（归约）：一个 block 内把一整块数据加成一个数。关键是 `tl.sum(x, axis=0)`——归约是 block 内的操作，不需要跨程序通信；跨程序的归约要用原子操作或二次归约，留到第 3 周。
4. 谜题 3（矩阵乘）：用 `tl.dot(a, b)` 做分块矩阵乘。注意 `tl.dot` 要求输入是 2D block，且最小尺寸有限制（一般不小于 16×16）；`tl.load` 时用 `tl.make_block_ptr` 或手动算指针 + mask 两种写法，先会一种。
5. 每道题的验收标准与第 1 周相同：与 PyTorch 参考实现逐项对比，数值对上才算过。

**产出**：3 道谜题通过（或无 GPU 时解释器版通过），加一句话：Triton 里归约和矩阵乘的“分块”与 CUDA shared memory 分块是同一回事，只是编译器帮你管了。

### 第 3 节 · Fused softmax 与数值稳定性 + Autotune

**目标**：写一个 fused softmax，理解“融合省的是访存”，并学会用 autotune 与 `do_bench` 计时。

**怎么做**（30 分钟，无 GPU 可写，跑起来需 GPU 或解释器）：

1. 读 Triton 官方教程 `02-fused-softmax`：一行一个程序，整行一次性读进来，在寄存器里算完 softmax 再写回去——中间结果不经过 global memory，这就是“融合”的全部含义。
2. 写出 fused softmax（官方教程的写法）：

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_cols, BLOCK: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK=BLOCK)
    return y
```

3. 理解数值稳定性：`row - tl.max(row)` 这一步不是优化，是正确性——不减最大值，`tl.exp` 在大输入下会溢出成 inf。`other=-float('inf')` 让 mask 掉的位置不参与 max 与求和。
4. 正确性检查：与 `torch.softmax(x, dim=-1)` 做 `torch.allclose` 对比（注意浮点误差容限，`atol=1e-5` 左右）。
5. 计时用 `triton.testing.do_bench`（热身 + 多次重复取中位数，与第 1 周的 CUDA Events 对应）：

```python
ms = triton.testing.do_bench(
    lambda: softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK=BLOCK)
)
```

6. 加 autotune：给 kernel 套一层 `@triton.autotune`，让 Triton 自动扫配置：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}, num_warps=4),
        triton.Config({'BLOCK': 256}, num_warps=4),
        triton.Config({'BLOCK': 1024}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_cols, BLOCK: tl.constexpr):
    # 与上面第 3 节相同的 kernel 体；autotune 只改 BLOCK 与 num_warps 的取值
    row_idx = tl.program_id(0)
```

`key` 是“输入形状变化时重新调优”的依据；同一形状下 autotune 只在第一次调用时扫一遍，后面直接用最优配置。**调优是自动的，但“调什么”是你定的**：BLOCK 大小与 num_warps 就是你要理解的两个旋钮。

**产出**：一个 fused softmax（与 PyTorch 版数值对齐）+ 一份 autotune 扫出来的最优配置记录。注意：这个 softmax kernel 也是 M3 第 5 周移植案例的候选之一（与第 6 周的 SiluAndMul 二选一），先把它存好。

### 第 4 节 · 中文讲解对照

**目标**：把本周的概念用中文复述一遍，补上英文教程里没讲透的背景。

**怎么做**（30 分钟，无 GPU 可做）：

1. 读 [Datawhale《大模型算法与系统教程》Part 03 Triton 部分](https://github.com/datawhalechina/llm-algo-leetcode)（15 节，中文 Notebook，可运行）。它是目前最成体系的中文 Triton 教程：PyTorch → Triton 的递进设计与本模块一致。
2. 对照官方教程，把这三个概念用中文写下来：block 编程模型（程序 vs 线程）、mask 与越界（为什么尾巴要处理）、融合（省的是访存次数，不是计算量）。
3. 注意 Datawhale 教程的边界：它的 Part 04（CUDA 与系统优化）与 Part 05（综合项目）仍在建设中，且全书基于 NVIDIA——国产芯片的部分不在它的覆盖范围内，那是 M3 的事。

**产出**：一页中文复述笔记，能向别人讲清“为什么 Triton 比 CUDA 好写、又比 PyTorch 快”。

### 第 5 节 · 小结：Triton 什么时候够用，什么时候必须回 CUDA

**目标**：写出取舍判断，而不是“哪个更好”的站队结论。

**怎么做**（30 分钟，无 GPU 可做）：

1. 对照本周的三个 kernel（向量加法、fused softmax、矩阵乘），填这张表：

| 场景 | 选什么 | 原因 |
|---|---|---|
| 逐元素、归约、简单融合 | Triton | 代码短、编译器管访存与向量化，够快 |
| 需要精确控制 shared memory 布局、warp 级同步 | CUDA | Triton 抽象掉了这些，控制不到 |
| 要榨干最后一滴性能（比如 FlashAttention 3 级别） | CUDA / CUTLASS | 需要 warp specialization、手写流水线 |
| 快速验证一个融合想法 | Triton | 写起来快，解释器还能无卡跑 |

2. 记住判断标准：**瓶颈在访存模式与融合时，Triton 够用；瓶颈在硬件细节（bank conflict、流水线、异步拷贝）时，回 CUDA**。第 3–5 周会在这个判断上继续加码。
3. 把本周的检查清单并进第 1 周的清单：加两条——“mask 写对了吗（尾巴处理）”与“BLOCK 是不是 2 的幂（tl.arange 要求）”。

**产出**：一页取舍笔记 + 更新后的检查清单。

## 第 3 周 · Fused Attention 深讲：先推导，再写 kernel

> 硬件：本周需要 GPU 跑性能对比；无卡可用 LeetGPU 的免费 GPU 评测先做正确性验证；第 1–2 节的推导与计算无卡可做。

### 第 1 节 · Online softmax 的数学推导

**目标**：手推 online softmax 的 rescale 公式，并证明分块算出的 attention 与标准 attention 完全相等（不是近似）。

**怎么做**（30 分钟，无 GPU 可做）：

1. 先写标准 softmax：一行向量 `x`，`softmax(x)_i = exp(x_i − m) / Σ_j exp(x_j − m)`，其中 `m = max(x)`。减最大值是为了数值稳定，不减的话 `exp` 会溢出。
2. 关键问题：softmax 需要整行的最大值与求和，分块之后这两个量都还没算全，怎么办？答案是**维护两个运行量**（running statistics），每来一块更新一次。设已处理部分的运行最大值是 `m`、运行和是 `l = Σ exp(x_j − m)`。新来一块，算出块内最大值 `m_b` 与块内和 `l_b = Σ exp(x_j − m_b)`，更新规则是：

```text
m' = max(m, m_b)
l' = l · exp(m − m') + l_b · exp(m_b − m')
```

3. 把同样的思想搬到 attention 上。设已处理部分的未归一化累加器是 `Õ = Σ_j exp(S_j − m) · V_j`（`S` 是 QK^T 的一行），新来一块 K、V：

```text
α = exp(m_old − m_new)              # 旧状态的 rescale 系数
P_block = exp(S_block − m_new)      # 块内概率（未归一化）
l_new = l_old · α + rowsum(P_block)
Õ_new = Õ_old · α + P_block · V_block
```

最后 `O = Õ / l_new`。每一步的 invariant 是：`Õ = Σ_{已处理 j} exp(S_j − m_current) · V_j`，`l = Σ exp(S_j − m_current)`。`m` 变大时乘 `α` 恰好把旧状态 rescale 到新的基准下，invariant 保持成立。
4. 结论：**FlashAttention 是精确算法**。分块 + online softmax 改变的只是计算顺序与中间结果的存放位置（S、P 不再写回 HBM），最终输出与标准 attention 在数学上相等，差别只有浮点舍入。这一点后面第 5 周读 FA2/FA3 时还会用到：它们改的都是并行与流水线，不是数学。
5. 对照第 2 周的 fused softmax：那一节的 `row - tl.max(row)` 就是单块版的 online softmax——整行一次性读进来时，`m` 一步到位；分块时 `m` 是逐步长大的，`α` 就是为“长大”付的代价。

**产出**：一份推导笔记，写出 rescale 公式与 invariant，并用一句话回答“为什么分块不改变结果”。

> **论文选读（不读也能学完本周）**：FlashAttention 原论文（Dao 等，[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)，2022 年 5 月；链接保留，原文是可选的更深阅读，本周正文已包含它的核心内容）。
>
> - **核心思想**：attention 慢、吃显存，根源不是计算量，而是**访存**。标准 attention 把 N×N 的 S、P 矩阵写回 HBM 再读回，HBM 与片上 SRAM 之间的搬运才是瓶颈。论文提出的原则叫 **IO-aware**：算法设计要把“各级 GPU 显存之间的读写次数”算进去，而不是只算 FLOPs。
> - **核心逻辑链条**：① 用 tiling 把 Q、K、V 分块，一次只把一块读进 SRAM，在 SRAM 里算完再写回；② 用 online softmax 让 softmax 也能分块（就是本节推导的 rescale）；③ 由此 S、P 两个 N×N 中间矩阵**永远不落 HBM**，HBM 访问量从 O(N²) 降到 O(N)；④ 论文给出 IO 复杂度分析，证明在一定 SRAM 容量范围内这个做法是最优的。
> - **关键结论与数字**（论文摘要给出）：BERT-large 训练端到端提速 15%（序列长度 512，对比 MLPerf 1.1 训练速度纪录）；GPT-2 训练提速 3 倍（序列长度 1K）；Long-Range Arena 提速 2.4 倍（序列长度 1K–4K）。更长上下文还带来质量提升：GPT-2 困惑度好 0.7，长文档分类提升 6.4 个点；首次让 Transformer 在 Path-X（序列长度 16K，61.4% 准确率）与 Path-256（序列长度 64K，63.1% 准确率）上超过随机水平。
> - **为什么重要**：它把“算得快”的优化对象从 FLOPs 转向了访存，并且证明了精确 attention 也可以分块算——这正是 M2 整个模块的底层逻辑：先定位瓶颈（第 1 周的 ncu + Roofline），再按瓶颈改算法，而不是盲目堆算力。
> - **与本节的关系**：本节的推导就是论文第 2–3 节的核心机制；第 5 周会接着讲 FA2/FA3 在这条链上又改了什么（并行、流水线、FP8）。

**延伸**：[Triton 官方教程](https://github.com/triton-lang/triton/tree/release/3.6.x/python/tutorials)的 fused attention（`06-fused-attention`）。

### 第 2 节 · SRAM 容量约束如何决定 block 大小

**目标**：给定你的 GPU，能自己算出 BLOCK_M / BLOCK_N 的上限，而不是抄教程的默认值。

**怎么做**（30 分钟，无 GPU 可做）：

1. 列出一个 block 在 SRAM 里要放的东西（FP16，每元素 2 字节，head 维 `d`，Q 块 `Br×d`，K/V 块 `Bc×d`，S 块 `Br×Bc`）：

```text
SRAM ≈ Q + K + V + S
     = 2·d·Br + 2·d·Bc + 2·d·Bc + 2·Br·Bc
     = 2d(Br + 2Bc) + 2·Br·Bc   ≤ 每 block 可用的 shared memory
```

2. 手算两个例子（`d=64`，A100 每 SM shared memory 164KB，H100 228KB；每 block 的可用上限用 `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)` 查，以你所用 CUDA 版本的设备属性为准）：
   - `Br=128, Bc=64`：Q 16KB + K 8KB + V 8KB + S 16KB = **48KB**。
   - `Br=128, Bc=128`：Q 16KB + K 16KB + V 16KB + S 32KB = **80KB**。
   - 再算 `Br=256, Bc=64`：Q 32KB + K 8KB + V 8KB + S 32KB = 80KB——同样是 80KB，但 Q 块翻倍意味着一个程序算更多行，并行度结构变了。
3. 理解取舍：block 越大，Q/K/V 的重复读越少（K、V 被更多 Q 行复用），但 SRAM 占用越大、一个 SM 上能同时驻留的 block 越少。**分块参数是 SRAM 容量与并行度的折中，不是越大越好**。这就是第 1 周 occupancy 那一节在 attention 上的回响。
4. Triton 里的对应关系：你不直接管 SRAM，`BLOCK_M` / `BLOCK_N` 就是旋钮；配额超了编译或运行时会直接报错（shared memory 不足），这是最直接的反馈。把你算出的两组参数记下来，下一节各跑一遍。

**产出**：一张“block 大小 vs SRAM 占用”计算表（至少三组参数，含你 GPU 的上限），并标出你下一节要先试哪一组。

### 第 3 节 · 动手写 fused attention（单 head、单序列版）

**目标**：写出一个能跑的 fused attention，与 PyTorch 参考数值对齐。

**怎么做**（30 分钟，需 GPU）：

1. 读 Triton 官方教程 `06-fused-attention` 的结构：外层循环按 `BLOCK_M` 分 Q 行，内层循环按 `BLOCK_N` 扫 K/V，每步做一次 online softmax 更新。下面是单 head、单序列、无 causal mask 的版本（多 head、变长、causal 是它的扩展，第 5 节留作练习）：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd(Q, K, V, Out,
              stride_qm, stride_kn, stride_vn, stride_om,
              N, sm_scale,
              D: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < N
    q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :],
                mask=mask_m[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], -float('inf'), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        k = tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        qk = tl.where(mask_n[None, :], qk, -float('inf'))
        m_block = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(V + offs_n[:, None] * stride_vn + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(Out + offs_m[:, None] * stride_om + offs_d[None, :],
             acc, mask=mask_m[:, None])

def fused_attention(q, k, v):
    # q, k, v: [N, D]，fp16，D=64；单 head、单序列、无 causal mask 版
    N, D = q.shape
    out = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N, BLOCK_M),)
    _attn_fwd[grid](q, k, v, out,
                    q.stride(0), k.stride(0), v.stride(0), out.stride(0),
                    N, 1.0 / (D ** 0.5),
                    D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out
```

2. 逐行对照第 1 节的推导：`m_i` / `l_i` / `acc` 就是运行量；`alpha` 就是 rescale 系数；第一轮 `m_i` 是 `-inf`，`alpha = exp(-inf) = 0`，旧状态（全零）被正确清掉。
3. 正确性检查：与 PyTorch 参考对比（fp16 精度，最大绝对误差应在 1e-2 量级；大了先查 mask 与 rescale，不要先调容限）：

```python
def ref_attention(q, k, v):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    p = torch.softmax(q.float() @ k.float().T * scale, dim=-1)
    return (p @ v.float()).to(q.dtype)

torch.manual_seed(0)
q = torch.randn(128, 64, dtype=torch.float16, device='cuda')
k = torch.randn(128, 64, dtype=torch.float16, device='cuda')
v = torch.randn(128, 64, dtype=torch.float16, device='cuda')
out = fused_attention(q, k, v)
ref = ref_attention(q, k, v)
print('max abs err:', (out.float() - ref.float()).abs().max().item())
```

**产出**：一个能跑的 fused attention（数值对齐）+ 一份“代码行 vs 推导公式”的对照笔记。

### 第 4 节 · 验收：正确性 + 计时 + Roofline 定位

**目标**：给 fused attention 一个完整的验收：对不对、快不快、卡在哪。

**怎么做**（30 分钟，需 GPU）：

1. 计时：用 `triton.testing.do_bench` 测 fused attention 与 PyTorch 参考（`torch.nn.functional.scaled_dot_product_attention`，SDPA）在 N=1024、d=64 下的耗时。SDPA 本身就是融合实现，比它慢是正常的——先记下差距，下一节再分析。
2. **先算 Roofline**：标准 attention 计算量 = 4·N²·d（QK^T 占 2N²d，PV 占 2N²d）。Fused 版搬运量 = Q、K、V、O 各 N·d 个 fp16 元素 = 8·N·d 字节，算术强度 = 4N²d / 8Nd = N/2。N=1024 时 AI=512——理论上是计算受限。朴素版（物化 N×N 的 S 与 P）要多搬约 8N² 字节，N=1024 时是 fused 版的 16 倍。**这个 16 倍就是融合省掉的东西**，先算出来，再看实测。
3. LeetGPU 验收：上 [LeetGPU](https://leetgpu.com)（50+ 题，浏览器里跑，CUDA 12.8；题库见 [leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges)），做至少 3 道 attention / matmul 相关题。LeetGPU 的评测会给你正确性与耗时排名——把它当成“别人写的同一道题”的对照。
4. 把本周 kernel 的点标到 M0 的 Roofline 图上：横轴是你算的 AI，纵轴是实测算力。如果点远在 Roofline 下方，下一节（第 5 周读 FA2/FA3）会告诉你工业级实现是怎么把它顶上去的。

**产出**：一份实测记录（fused vs SDPA 耗时、Roofline 点、LeetGPU 3 道题通过）。

### 第 5 节 · 小结：Roofline 标图 + causal mask 练习

**目标**：收拢本周的结论，并把 causal mask 留作练习。

**怎么做**（30 分钟，推导与总结无卡可做）：

1. 更新 Roofline 图：把 fused attention 的点标上去，并写一句话——它离 Roofline 有多远、差在哪（大概率是 `tl.dot` 的利用率与 softmax 部分的非 matmul 开销，第 5 周会展开工业级实现用什么手段把它顶上去）。
2. 练习：给 kernel 加 causal mask。做法是在算 `qk` 之后加一行：`qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float('inf'))`。注意 mask 的形状是 `[BLOCK_M, BLOCK_N]`，与 `mask_n` 的那行 `tl.where` 可以合并。改完用同样的参考实现验证（参考里用 `torch.tril` 造 mask 或直接调带 `is_causal=True` 的 SDPA）。
3. 把本周的检查清单并进总清单，加两条：“rescale 的 invariant 写出来了吗”“分块参数是算出来的还是抄的”。

**产出**：更新后的 Roofline 图 + causal mask 版 kernel（数值对齐）。

## 第 4 周 · 量化 Matmul 深讲：量化方案本身

> 硬件：本周需要 GPU；无卡时先做第 1 节的手算与第 2–3 节的代码编写，实测标注“待补测”。

### 第 1 节 · Scale 怎么来：三种粒度与误差从哪来

**目标**：理解 per-tensor / per-channel / per-token 三种量化粒度的 scale 是怎么算出来的，以及误差从哪来。

**怎么做**（30 分钟，无 GPU 可做）：

1. 先定量化规则。对称 INT8 量化：给定一个 scale `s`，`w_q = round(w / s)`，钳制到 `[-127, 127]`，反量化 `w' = w_q · s`。scale 的取法是 `s = max|w| / 127`，量化误差的上界是 `|w − w'| ≤ s / 2`。**scale 越大，误差越大**——整个量化的精度问题就归结为“怎么让 scale 尽可能小”。
2. 三种粒度，scale 的“管辖范围”不同：
   - **per-tensor**：整个张量一个 scale，`s = max|整个张量| / 127`。最简单，但一个离群值会把整个张量的 scale 顶上去。
   - **per-channel**（权重常用）：每个输出通道一个 scale，`s_j = max|第 j 列| / 127`。离群值只影响它所在的通道。
   - **per-token**（激活常用）：每个 token（每行）一个 scale，`s_i = max|第 i 行| / 127`。激活的分布随输入变，所以 scale 要**运行时现算**（dynamic）；也可以事先用校准集定好（static），省了运行时开销，但分布一变就可能失配。
3. 手算一个例子，体会离群值的破坏力。一行权重 `[0.1, -0.08, 0.12, 10.0, 0.09]`：per-tensor 的 scale 是 `10 / 127 ≈ 0.079`，`0.1` 量化成 `round(0.1 / 0.079) = 1`，反量化得 `0.079`——21% 的误差。如果 10.0 在别的通道，per-channel 下这一行的 scale 是 `0.12 / 127 ≈ 0.0009`，误差上界 `0.0005`。**这就是为什么权重用量化几乎都用 per-channel：它把离群值关进一个通道里。**
4. 再算动态与静态的代价：per-token dynamic 每次前向都要对激活做一次规约（求每行最大值），多一次访存与同步；static 省了这一步，但要回答“校准集能不能代表真实输入”。推理引擎里常见的做法是**权重 per-channel static、激活 per-token dynamic**——权重分布固定，激活分布随输入变，各取所需。

**产出**：一份三种粒度的误差直觉 + 上面的手算（离群值例子 + 误差上界）。

### 第 2 节 · INT8 `tl.dot` 路径：int8 输入、int32 累加、epilogue 再乘 scale

**目标**：写出一个 INT8 matmul：int8 输入、`tl.dot` 得 int32 累加、epilogue 里再乘回 scale。

**怎么做**（30 分钟，需 GPU 做正确性验证；无卡先写代码）：

1. 先想清楚数据流：A（激活，per-token scale）与 B（权重，per-channel scale）都以 int8 存，`tl.dot` 做 int8×int8 → int32 累加；acc 是整数域的，**scale 不能在累加前乘**——必须等累加完，在 epilogue 里一次乘回：`C = acc_fp32 · s_a[:, None] · s_b[None, :]`。这一步在寄存器里完成，不额外访存。
2. 量化用 PyTorch 做（这部分是数据准备，不是 kernel 本身）：

```python
def quant_per_token(x):
    # x: [M, K] fp16 -> int8 + scales [M]
    amax = x.abs().amax(dim=1)
    scale = (amax / 127.0).clamp_min(1e-8)
    q = torch.round(x / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale

def quant_per_channel(w):
    # w: [K, N] fp16 -> int8 + scales [N]
    amax = w.abs().amax(dim=0)
    scale = (amax / 127.0).clamp_min(1e-8)
    q = torch.round(w / scale[None, :]).clamp(-127, 127).to(torch.int8)
    return q, scale
```

3. INT8 matmul kernel：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_int8_kernel(a_ptr, b_ptr, c_ptr,
                       M, N, K,
                       stride_am, stride_ak, stride_bk, stride_bn,
                       stride_cm, stride_cn,
                       scale_a_ptr, scale_b_ptr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bn
    sa = tl.load(scale_a_ptr + offs_m, mask=offs_m < M, other=0.0)
    sb = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c.to(tl.float16), mask=mask_c)

def matmul_int8(a_q, b_q, scale_a, scale_b):
    M, K = a_q.shape
    K2, N = b_q.shape
    assert K == K2
    c = torch.empty((M, N), dtype=torch.float16, device=a_q.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_int8_kernel[grid](a_q, b_q, c, M, N, K,
        a_q.stride(0), a_q.stride(1), b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1),
        scale_a.float(), scale_b.float(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c
```

4. 正确性检查：与 FP16 参考对比。量化有误差，`torch.allclose` 的绝对容限不再适用——改用**相对误差**，并把它记下来（这是本周的精度账本）：

```python
torch.manual_seed(0)
a = torch.randn(128, 256, dtype=torch.float16, device='cuda')
b = torch.randn(256, 128, dtype=torch.float16, device='cuda')
a_q, sa = quant_per_token(a)
b_q, sb = quant_per_channel(b)
c = matmul_int8(a_q, b_q, sa, sb)
ref = a @ b
rel_err = ((c.float() - ref.float()).abs() / ref.float().abs().clamp_min(1)).mean()
print('mean relative error:', rel_err.item())
```

预期相对误差在 1% 量级；如果大一个数量级，先查 scale 的粒度（是不是把 per-token 与 per-channel 弄反了）与 `tl.dot` 的累加类型（必须是 int32，int16 会溢出）。

**产出**：一个 INT8 matmul + 一份相对误差记录（数值与容限判断）。

### 第 3 节 · Fused dequant+matmul：dequant 融进 matmul，不单独物化 FP16 权重

**目标**：写出 fused dequant+matmul，并与“先 dequant 再 matmul”的两步基线做访存对比。

**怎么做**（30 分钟，需 GPU；无卡先写代码）：

1. 先想清楚两步基线的问题：权重以 int8 存，要先跑一个 dequant kernel 把它展开成 FP16（写回 HBM），再跑 FP16 matmul 把它读回来。权重每用一次就多一次“展开→写回→读回”的往返。**dequant 融合就是把展开放在寄存器里做**：读 int8 tile，乘 scale 转成 fp16，立刻喂给 `tl.dot`，FP16 权重永远不落 HBM。
2. Fused kernel（A 是 fp16 激活，B 是 int8 权重 + per-channel scale）：

```python
@triton.jit
def matmul_fused_dequant_kernel(a_ptr, b_ptr, c_ptr,
                                M, N, K,
                                stride_am, stride_ak, stride_bk, stride_bn,
                                stride_cm, stride_cn,
                                scale_b_ptr,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    sb = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b_int8 = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0)
        b = (b_int8.to(tl.float32) * sb[None, :]).to(tl.float16)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bn
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_c)
```

3. 关键点：`b_int8.to(tl.float32) * sb[None, :]` 是在寄存器里对每个元素做的——dequant 的代价是每个元素几次 ALU 操作，省的是 HBM 往返。**融合的收益取决于 matmul 本来卡在哪**：小 batch（decode 形状，M 很小）时 matmul 是访存受限，省搬运就是省时间；大 batch 时 matmul 是计算受限，dequant 的 ALU 开销占比上升，收益收窄。这个判断先写下来，下一节实测验证。
4. 两步基线（对照用，PyTorch 一行）：`b_fp16 = (b_q.float() * sb).to(torch.float16)`，再 `torch.matmul(a, b_fp16)`。用 `triton.testing.do_bench` 测两边在 M=32（decode 形状）与 M=1024（prefill 形状）下的耗时，填表。

**产出**：一个 fused dequant kernel + 一份两步基线 vs 融合版的访存对比表（M=32 与 M=1024 两档）。

### 第 4 节 · Autotune + 实测：BLOCK 配置扫一遍，`do_bench` 计时

**目标**：给第 2–3 节的两个 kernel 做 autotune，出一份配置-耗时对照表。

**怎么做**（30 分钟，需 GPU；无卡先写配置表框架，实测标注“待补测”）：

1. 给两个 kernel 套 `@triton.autotune`，`key` 取 `['M', 'N', 'K']`（形状一变就重调），配置覆盖 `BLOCK_M × BLOCK_N × BLOCK_K` 与 `num_warps` 的组合，至少扫 4 组，例如：`(64, 64, 32, warps=4)`、`(128, 64, 32, warps=4)`、`(64, 128, 32, warps=8)`、`(128, 128, 32, warps=8)`。
2. 测两档形状：M=32（decode）与 M=1024（prefill），N=K=4096。每组记三列：耗时、换算成的算力（TFLOPS）、与 FP16 Triton matmul 基线的比值。INT8 的理论峰值算力通常是 FP16 的 2 倍，但实测能到多少取决于 `tl.dot` 的 int8 路径与 epilogue 开销——先测再下结论。
3. 验证第 3 节的判断：fused dequant 在 M=32 时应该明显快于两步基线（省了权重展开的 HBM 往返），在 M=1024 时差距收窄。如果实测与判断不符，记下来——这是第 6 周“负面结果分析”的素材。

**产出**：一份配置-耗时对照表（两档形状 × 至少 4 组配置）+ 一句“判断是否成立”的结论。

### 第 5 节 · 小结：量化省的是什么（搬运量），代价是什么（精度），Roofline 点标图

**目标**：收拢本周的结论，把两个 kernel 的 Roofline 点标到图上。

**怎么做**（30 分钟，总结无卡可做）：

1. 算术强度对比：同一个 4096³ matmul，FP16 版搬运量是权重与激活各 2 字节/元素，INT8 版权重只剩 1 字节/元素——**量化首先省的是搬运量**，算术强度向右移；同时 int8 的 `tl.dot` 峰值算力更高，Roofline 的顶也抬高了。两边一起动，这就是量化提速的完整解释。
2. 代价：精度。第 2 节的相对误差就是代价的度量。把“相对误差”与“提速比”并排放进同一张表——**量化方案的取舍永远是这两个数的权衡**，不是单看速度。
3. 更新 Roofline 图：把 INT8 matmul 与 fused dequant 的点标上去，并写一句话——它们离 Roofline 有多远、差在哪。
4. 把本周的检查清单并进总清单，加两条：“scale 的粒度与误差上界能手算吗”“dequant 融合的收益在什么形状下成立、什么形状下收窄”。

**产出**：更新后的 Roofline 图 + 一张“相对误差 vs 提速比”对照表。

## 第 5 周 · FlashAttention 2/3 与 CUTLASS：读工业级实现（大纲）

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | FlashAttention 1→2→3 的工程演进：FA2 的并行与 work partitioning，FA3 的 warp specialization、异步拷贝与 FP8 | FlashAttention 论文：[FA1 arXiv:2205.14135](https://arxiv.org/abs/2205.14135)、[FA2 arXiv:2307.08691](https://arxiv.org/abs/2307.08691)、[FA3 arXiv:2407.08608](https://arxiv.org/abs/2407.08608)；[GPU MODE 讲座](https://github.com/gpu-mode/lectures) | 一份“论文做法 vs 工程做法”对照笔记 |
| 2 | 读 flash-attn 源码：`flash_fwd_kernel.h` 里的主循环长什么样 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)（`csrc/flash_attn/src/flash_fwd_kernel.h`、`flash_fwd_launch_template.h`） | 一张前向 kernel 的结构图 |
| 3 | CUTLASS 三级结构：threadblock / warp / thread + MMA pipeline | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)（`examples/00_basic_gemm/basic_gemm.cu`、`cutlass/gemm/device/gemm.h`） | 能画出三级分工与数据流向 |
| 4 | 动手改一个 CUTLASS GEMM：改 tile 配置（ThreadblockShape / WarpShape / InstructionShape），重编，测出变化 | 同上；第 1 周的计时与 ncu 方法 | 一份“改前改后”实测记录 |
| 5 | 小结：Triton 版 vs CUTLASS 版 attention 的性能与可维护性；什么时候自己写 kernel，什么时候直接用库 | 第 3–4 周的实测数据 | 一页对比表 + 取舍笔记 |

**硬件**：以读代码为主，无卡可做；第 4 节改配置实测需要 GPU（有 nvcc 的机器）。

## 第 6 周 · 收口：把自写 kernel 接入推理引擎（大纲）

> 注意：大纲原计划是“把第 3 周的 fused attention 包成 CustomOp”。核实 vLLM 的 `CustomOp` 源码（`vllm/model_executor/custom_op.py`）后，这个写法需要修正：vLLM 的 attention 走 Attention 后端，不走 CustomOp；CustomOp 的可替换面是激活、归一化这类算子。完整内容会按修正后的方案写——接入引擎的载体改为 Triton 版 SiluAndMul（与第 3 周的 fused attention 共享同一套“写 kernel → 验证 → 接入”流程），fused attention 仍在第 3 周做 standalone 验证。

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | vLLM 的 CustomOp 机制：注册、分发、启用三件事，及它的边界（attention 不走这条路） | [vLLM CustomOp 设计文档](https://docs.vllm.ai/en/latest/design/custom_op/)；vLLM 源码 `vllm/model_executor/custom_op.py` | 能讲清插件算子的注册与调用路径 + 边界 |
| 2 | 动手：写 Triton 版 SiluAndMul（参考 vLLM 自带的 Triton kernel 写法），standalone 正确性 + 微基准 | vLLM 源码 `vllm/model_executor/layers/activation.py`（内含 Triton 版实现可对照） | 可独立验证的 Triton kernel |
| 3 | 包成 CustomOp 子类并注册：`register_oot` 按类名替换，进程内验证替换真的发生了 | 同上；[vLLM 插件机制](https://github.com/vllm-project/vllm/blob/main/vllm/plugins/__init__.py)（`vllm.general_plugins` entry point） | 可加载的插件包 |
| 4 | 端到端验证：同一模型、同一评测集，接入前后的 TTFT / ITL / 吞吐（M0 的 `scripts/bench_openai.py`）+ profiler 确认 Triton kernel 真的跑了 | M0 的压测脚本与基线报告 | 一份接入前后对比表 |
| 5 | 小结：M2 毕业——一份“自写 kernel 端到端加速报告”（含负面结果分析：没变快的原因） | M0 第 3 节的 profiling 方法 | **M2 的可交付物** |

**硬件**：本周需要 GPU（租卡）。无卡时先完成第 1–3 节的机制学习与代码编写，端到端验证标注“待有卡后补测”。

## 本模块检查清单

- [ ] 有一张 GEMM 优化阶梯总表：四个版本各自的算术强度、实测耗时、实测算力、ncu 读数与瓶颈判断
- [ ] 能用 ncu 的两个数（SM% / DRAM%）判断一个 kernel 是计算受限、访存受限还是延迟受限
- [ ] 能手推 online softmax 的 rescale 公式，并按 SRAM 容量算出自己 GPU 的分块参数
- [ ] 写出过 fused attention 与 fused dequant matmul，数值都对齐过
- [ ] 改过一个 CUTLASS GEMM 的 tile 配置并测出了变化
- [ ] 能画出 vLLM 里一个 CustomOp 从注册到分发的完整链路
- [ ] 有一份“自写 kernel 接入引擎前后”的端到端对比报告（含负面结果）
