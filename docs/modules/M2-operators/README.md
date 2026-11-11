# M2 · 算子优化

> 状态：**内容进行中**——第 1–2 周为完整内容（加深版），第 3–6 周仍为大纲，见下文各周表格。
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

## 第 3 周 · Fused Attention 深讲：先推导，再写 kernel（大纲）

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | Online softmax 的数学推导：running max / running sum 如何让 softmax 分块算 | FlashAttention 论文（Dao 等，arXiv:2205.14135）；[Triton 官方教程](https://github.com/triton-lang/triton/tree/release/3.6.x/python/tutorials)（fused attention） | 一份推导笔记：rescale 公式与正确性证明思路 |
| 2 | SRAM 容量约束如何决定 block 大小：按你的 GPU 算出 BLOCK_M / BLOCK_N | 同上；`nvidia-smi` 与设备参数（每 SM 的 shared memory 大小） | 一张“block 大小 vs SRAM 占用”计算表 |
| 3 | 动手写 fused attention（单 head、单序列版） | 同上；[GPU MODE 讲座](https://github.com/gpu-mode/lectures)中 FlashAttention 相关章节 | 一个能跑的 fused attention，与 PyTorch 数值对齐 |
| 4 | 验收：正确性 + `do_bench` 计时 + Roofline 定位 | M0 的 Roofline 方法；[LeetGPU](https://leetgpu.com)（题库：[leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges)） | 至少 3 道 attention / matmul 相关题通过 + 一份实测记录 |
| 5 | 小结：把本周 kernel 的 Roofline 点标到 M0 的图上；causal mask 作为练习 | M0 的 Roofline 图 | 更新后的 Roofline 图 |

**硬件**：本周需要 GPU 跑性能对比；无卡可用 LeetGPU 的免费 GPU 评测先做正确性验证；推导部分无卡可做。

## 第 4 周 · 量化 Matmul 深讲：量化方案本身（大纲）

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | Scale 怎么来：per-tensor / per-channel / per-token 三种粒度，误差从哪来 | [GPU MODE 讲座](https://github.com/gpu-mode/lectures)中量化相关章节 | 一份三种粒度的误差直觉 + 手算 |
| 2 | INT8 `tl.dot` 路径：int8 输入、int32 累加、epilogue 再乘 scale | [Triton 官方教程](https://github.com/triton-lang/triton/tree/release/3.6.x/python/tutorials)（matmul） | 一个 INT8 matmul，与 FP16 参考数值对齐（容限内） |
| 3 | Fused dequant+matmul：dequant 融进 matmul，不单独物化 FP16 权重 | 同上 | 一个 fused dequant kernel + 与两步基线（先 dequant 再 matmul）的访存对比 |
| 4 | Autotune + 实测：BLOCK 配置扫一遍，`do_bench` 计时 | 第 2 周的 autotune 方法 | 一份配置-耗时对照表 |
| 5 | 小结：量化省的是什么（搬运量），代价是什么（精度），Roofline 点标图 | M0 的 Roofline 图 | 更新后的 Roofline 图 |

**硬件**：本周需要 GPU；无卡时先做第 1 节的手算与第 2–3 节的代码编写，实测标注“待补测”。

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
