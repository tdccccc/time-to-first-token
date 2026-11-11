# M2 · 算子优化

> 状态：**内容进行中**——第 1–2 周（CUDA、Triton）为完整内容，第 3–5 周仍为大纲，见下文各周表格。
>
> 前置：M0（Roofline、profiling、压测脚本）与 M1 第 1–2 周（vLLM 服务已跑起来）。M2 的终点不是“会写 kernel”，而是“自己写的 kernel 跑进了推理引擎，端到端变快了”。
>
> 每节 30 分钟，每周 5 节，沿用 M1 的节奏。**每节开头标硬件门槛**：无卡可做 / 需 GPU / 需租卡，并给出无卡替代路径。代码示例分两种：能直接跑的（已按官方教程核对过写法）和示意性的（标明“示意”二字，只讲结构不保证编译通过）——不要把示意代码当可运行代码。

## 为什么单开一个模块

M1 让你会用引擎、会调参数，但引擎内部的 attention、matmul、量化 kernel 对你还是黑盒。M2 的目标是打开这个盒子：先学会写 kernel，再把 kernel 接回引擎，最后用 M0 的压测脚本量出端到端的变化。现有的练习资源（GPU Puzzles、Triton Puzzles、LeetGPU）都在 kernel 层面收尾，**“接入引擎”这一步目前没有现成课程**，是本模块要补的关键一节。

## 第 1 周 · CUDA 基础：先理解 GPU 再写代码

> 硬件：需要一块 NVIDIA GPU（租卡或本地卡均可；Colab 免费 GPU 也能跑 GPU Puzzles）。无卡时先读题、看题解，不要跳过——第 2 周的 Triton 部分无需 GPU。

### 第 1 节 · GPU 执行模型：thread、block、grid

**目标**：能画出一次 kernel 启动的数据流向：数据从哪来、经过哪些线程、结果写回哪。

**怎么做**（30 分钟）：

1. 读 [GPU Puzzles](https://github.com/srush/GPU-Puzzles) 的前言：它用 Numba 把 Python 代码直接映射成 CUDA kernel，“看起来像 Python，实际就是 CUDA”。再看 [Stanford CS336 第 5 讲](https://www.youtube.com/watch?v=6OBtO9niT00)前 30 分钟（GPU 与显存层级）。
2. 记住三层结构：**thread**（最小执行单位，每个线程算一个或几个元素）、**block**（一组线程，共享一块 shared memory，block 内可以同步）、**grid**（所有 block 的集合）。一个 kernel 启动 = 一个 grid；`blockIdx` 告诉你现在是第几个 block，`threadIdx` 告诉你是 block 里的第几个线程。
3. 记住显存层级：global memory（大、慢，所有线程可见）→ shared memory（小、快，block 内共享）→ register（最快，每个线程私有）。M0 的 Roofline 在这里落地：一次访存没进 shared memory，就等于又去 global memory 搬了一遍。
4. 对照 M0 第 1 节的手算：decode 时每个 token 要把 7B 权重读一遍——这就是为什么“访存合并”和“分块”是本章唯二的主题。

**产出**：一张手画的数据流向图：输入向量在 global memory → 按 block 分片 → 每个 thread 算一个元素 → 写回 global memory，并标出 threadIdx、blockIdx、blockDim 各自的含义。

**延伸**：[NUMBA CUDA 指南](https://numba.readthedocs.io/en/stable/cuda/index.html)（GPU Puzzles 官方推荐的参考文档）。

### 第 2 节 · 第一个 kernel：向量加法与正确性检查

**目标**：跑通第一个 kernel，理解“正确性检查”是 kernel 开发的默认动作——先对，再快。

**怎么做**（30 分钟，需 GPU）：

1. 打开 [GPU Puzzles](https://github.com/srush/GPU-Puzzles) 的 notebook（推荐 Colab：把 GPU 运行时打开，`Runtime / Change runtime type` → `Hardware accelerator` 选 GPU；也可以本地 clone 仓库后用 Jupyter 跑，需要一块 NVIDIA 卡与 numba）。按仓库 README 的说明装好环境（`pip install` 一个 chalk 分支、下载 `lib.py` 与 `robot.png` 两个文件）。注意：Numba 内置的 CUDA 目标已被标记为弃用（后续开发移到了 NVIDIA 的 `numba-cuda` 包），所以**以 GPU Puzzles 仓库自己锁定的 numba 版本与安装说明为准**，不要自行升级 numba——本题集的价值在思路，不在追新版本。
2. 做 Puzzle 1（Map）：给向量每个位置加 10，一个线程负责一个位置。解法只有一行：

```python
def map_test(cuda):
    def call(out, a) -> None:
        i = cuda.threadIdx.x
        out[i] = a[i] + 10
    return call
```

3. 理解检查机制：每个谜题都有一个 `spec`（NumPy 写的参考实现）和 `problem.check()`——它把你的 kernel 输出与参考实现逐项对比，错了会把“你的结果”和“期望结果”都打印出来。**以后你写的每个 kernel 都要配一个这样的参考实现**，这是本模块的铁律。
4. 再做 Puzzle 2（Zip，两个向量相加）与 Puzzle 3（Guards，线程数多于元素数时要加边界判断 `if i < size`）。第 3 题的教训：多出来的线程如果不守边界，就会越界写坏内存——这是 GPU 编程最常见的 bug 形态。

**产出**：Puzzle 1–3 通过（`problem.check()` 全部显示通过），并能说清 `problem.show()` 里那张“每个线程访存次数”表在量什么。

**对照**：同一个向量加法，用原生 CUDA 写出来是这样（示意，只讲结构）：

```cpp
__global__ void vec_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
// 启动：vec_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_out, n);
```

Numba 版和 CUDA 版是同一回事：`threadIdx.x`、`blockIdx.x`、`blockDim.x` 逐一对应。看懂这个对应关系，后面读任何 CUDA 代码都不怕。

### 第 3 节 · 访存合并：为什么连续访存快

**目标**：亲手量出“连续访存”与“跨步访存”的速度差，并理解 GPU 的合并访存机制。

**怎么做**（30 分钟，需 GPU）：

1. 做 GPU Puzzles 的 Puzzle 4（Map 2D）与 Puzzle 5（Broadcast），理解二维索引与广播的访存模式。
2. 写一个对比实验：同一个向量加法 kernel，分别用“每个线程读连续地址”和“每个线程隔 32 个元素读一个地址”两种方式跑，计时对比。计时用 Numba 的同步 + 系统时钟：

```python
import time
from numba import cuda

start = time.perf_counter()
kernel[blocks, threads](d_out, d_a, d_b, n)
cuda.synchronize()  # 必须同步，否则只量到了“启动”而不是“跑完”
elapsed = time.perf_counter() - start
```

3. 理解原理：GPU 把相邻 32 个线程（一个 warp）的访存合并成一次事务。连续访问时 32 个线程一次搬完；跨步访问时 32 个线程要发 32 次事务，带宽直接除以 32。**这个数字不是理论，是你刚才量出来的。**
4. 把结论写回 M0 的 Roofline 语言：一次没合并的访存，等于把算术强度除以 32——一个本来算力受限的 kernel，可能因为访存模式直接掉进带宽受限区。

**产出**：一份两行对比表（连续 vs 跨步，各自的耗时与带宽），加一句话结论。

**无卡替代**：读 Puzzle 4–5 的题解，理解为什么二维索引要按行优先排布；计时实验留到有卡时补做，标注“待补测”。

### 第 4 节 · Shared memory 与分块：矩阵乘法

**目标**：理解分块（tiling）——把反复读的数据先搬进 shared memory，再算。

**怎么做**（30 分钟，需 GPU）：

1. 做 GPU Puzzles 的 Puzzle 6（Blocks：block 数量多于线程数，用 `blockIdx` 算全局索引）、Puzzle 7（Blocks 2D）与 Puzzle 8（Shared：第一次用 `cuda.shared.array` 与 `cuda.syncthreads()`）。
2. 理解 Puzzle 8 的关键规则：shared memory 的大小必须是**字面常量**（不能是变量），写完 shared memory 必须调 `cuda.syncthreads()`，否则有的线程会读到别的线程还没写完的数据。
3. 做 Puzzle 14（Matrix Multiply，矩阵乘法）：先写朴素版（每个线程算一个输出元素，直接读 global memory），再按分块思路改写——把 A 的一行块和 B 的一列块先搬进 shared memory，在 shared memory 里算部分和。对比两版的 `problem.show()` 访存计数：分块版的 global 读数应该明显下降。
4. 把分块与 M0 的 Roofline 连起来：分块没有减少总计算量，它减少的是**重复搬运**——A 的同一行被 B 的很多列复用，每复用一次，算术强度就涨一分。

**产出**：一份朴素版 vs 分块版矩阵乘法的访存对比（global 读数 + 耗时），加一句话：分块省的是什么。

### 第 5 节 · 占用率与 launch 开销 + 本周小结

**目标**：知道什么时候 kernel 启动本身就是瓶颈，并写出本周的检查清单。

**怎么做**（30 分钟，需 GPU；读的部分无卡可做）：

1. 用 M0 第 3 节的 profiling 方法（`torch.profiler` 或 `nsys`），对一次小模型推理做一次 timeline：数一数有多少次 kernel 启动，每次启动之间有多少空隙。
2. 理解两个概念：**占用率**（occupancy，SM 上同时驻留的 warp 数占最大值的比例——block 太大或 shared memory 用太多都会拉低它）和 **launch 开销**（一次 kernel 启动本身的固定成本，约数微秒；kernel 越小、启动越频繁，开销占比越高）。
3. 判断：解码阶段 batch=1 时，每个 kernel 只算一个 token，计算量极小——这时 launch 开销和访存就是全部成本。这就是为什么推理引擎要做 kernel 融合（第 2–3 周）与 CUDA Graph（M1 第 8 周会遇到）。
4. 写一份“CUDA 性能检查清单”，后面每节对照：

```text
□ 边界守好了吗（线程数 > 元素数时有没有 if 保护）
□ 访存连续吗（warp 内 32 个线程读的是连续地址吗）
□ 重复读的数据进 shared memory 了吗
□ 同步加了吗（写 shared memory 之后有没有 syncthreads）
□ kernel 是不是太小了（launch 开销占比是不是超过 10%）
□ 正确性检查跑了吗（先对，再快）
```

**产出**：这份检查清单 + 一份 profiling 笔记（一次推理里 launch 开销的占比）。

## 第 2 周 · Triton 入门：用 Python 写 kernel

> 硬件：**本周无需 GPU**。Triton Puzzles 用解释器跑；跑性能对比时再上 GPU。

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

**目标**：写一个 fused softmax，理解“融合省的是访存”，并学会用 autotune 扫配置。

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
5. 加 autotune：给 kernel 套一层 `@triton.autotune`，让 Triton 自动扫配置：

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

**产出**：一个 fused softmax（与 PyTorch 版数值对齐）+ 一份 autotune 扫出来的最优配置记录。

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

2. 记住判断标准：**瓶颈在访存模式与融合时，Triton 够用；瓶颈在硬件细节（bank conflict、流水线、异步拷贝）时，回 CUDA**。第 3–4 周会在这个判断上继续加码。
3. 把本周的检查清单并进第 1 周的清单：加两条——“mask 写对了吗（尾巴处理）”与“BLOCK 是不是 2 的幂（tl.arange 要求）”。

**产出**：一页取舍笔记 + 更新后的检查清单。

## 第 3 周 · Fused Attention 与量化 Matmul（大纲）

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | FlashAttention 的思路：分块 + 在线 softmax | [Triton 官方教程](https://github.com/triton-lang/triton/tree/release/3.6.x/python/tutorials)（fused attention） | 读懂分块 attention 的数据流 |
| 2 | 动手写 fused attention | 同上；[GPU MODE 讲座](https://github.com/gpu-mode/lectures)中 FlashAttention 相关章节 | 一个能跑的 fused attention，与 PyTorch 数值对齐 |
| 3 | 量化 matmul：INT8/FP8 的访存收益 | [GPU MODE 讲座](https://github.com/gpu-mode/lectures) | 一份量化前后访存量的手算 |
| 4 | 刷题验收 | [LeetGPU](https://leetgpu.com)（题库：[leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges)，提供免费 GPU 运行提交） | 至少 3 道 attention / matmul 相关题通过 |
| 5 | 小结：把本周 kernel 的 Roofline 点标到 M0 的图上 | M0 的 Roofline 图 | 更新后的 Roofline 图 |

**硬件**：本周需要 GPU 跑性能对比；无卡可用 LeetGPU 的免费 GPU 评测先做正确性验证。

## 第 4 周 · FlashAttention 与 CUTLASS：读工业级实现（大纲）

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | FlashAttention 2/3 的工程取舍 | [GPU MODE 讲座](https://github.com/gpu-mode/lectures) | 一份“论文做法 vs 工程做法”对照笔记 |
| 2 | CUTLASS / CuTe 的角色：模板化高性能 GEMM | [GPU MODE 讲座](https://github.com/gpu-mode/lectures) | 能讲清 CUTLASS 解决什么问题 |
| 3 | 读 vLLM / FlashInfer 里一个 attention kernel 的调用链 | vLLM 仓库源码；[vLLM CustomOp 设计文档](https://docs.vllm.ai/en/latest/design/custom_op/) | 一张调用链图：请求 → 调度 → kernel |
| 4 | 对比：Triton 版 vs CUTLASS 版 attention 的性能与可维护性 | 第 3 周的实测数据 | 一页对比表 |
| 5 | 小结：什么时候自己写 kernel，什么时候直接用库 | — | 取舍笔记 |

**硬件**：以读代码为主，无卡可做；有卡可复现第 3 周的对比。

## 第 5 周 · 收口：把自写 kernel 接入推理引擎（大纲）

> 注意：大纲原计划是“把第 3 周的 fused attention 包成 CustomOp”。核实 vLLM 的 `CustomOp` 源码（`vllm/model_executor/custom_op.py`）后，这个写法需要修正：vLLM 的 attention 走 Attention 后端，不走 CustomOp；CustomOp 的可替换面是激活、归一化这类算子。完整内容会在第 2 阶段按修正后的方案写——接入引擎的载体改为 Triton 版 SiluAndMul，fused attention 仍在第 3 周做 standalone 验证。

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | vLLM 的 CustomOp 机制：算子如何被替换 | [vLLM CustomOp 设计文档](https://docs.vllm.ai/en/latest/design/custom_op/) | 能讲清插件算子的注册与调用路径 |
| 2 | 动手：把第 3 周的 fused attention 包成 CustomOp | 同上 | 可加载的 CustomOp |
| 3 | 端到端验证：同一模型、同一评测集，接入前后的 TTFT / ITL / 吞吐 | M0 的 `scripts/bench_openai.py` 与基线报告 | 一份接入前后对比表 |
| 4 | 负面结果也要写下来：没变快的原因分析（launch 开销？访存没省下来？batch 太小？） | M0 第 3 节的 profiling 方法 | 一份原因分析 |
| 5 | 小结：M2 毕业——一份“自写 kernel 端到端加速报告” | — | **M2 的可交付物** |

**硬件**：本周需要 GPU（租卡）。无卡时先完成第 1–2 节的机制学习，端到端验证标注“待有卡后补测”。

## 本模块检查清单

- [ ] GPU Puzzles 核心谜题通过，能讲清访存合并与分块
- [ ] Triton Puzzles（解释器版）通过，写出过 fused attention
- [ ] LeetGPU 上至少 3 道相关题通过
- [ ] 能画出 vLLM 里一个 attention kernel 的调用链
- [ ] 有一份“自写 kernel 接入引擎前后”的端到端对比报告（含负面结果）
