# 第 13 章 Triton 编程模型：从向量加到 2D GEMM { .chapter-title }

<p class="chapter-summary">从 CUDA 的单线程视角跃迁到 Triton 的 Block 级张量视角，通过 1D Mask、Fused Softmax 访存账与带广播指针的最小 2D GEMM，建立编写现代 GPU 融合算子的完整直觉。</p>

上一章的四版 CUDA GEMM 暴露了一个残酷的工程现实：只要把 `TILE` 从 32 改成 64，或者想在矩阵乘末尾多融一个 `ReLU`，线程映射、Shared Memory 声明、Bank Padding 与 `__syncthreads()` 同步点几乎要推倒重写。在大模型推理迭代中，工程师真正需要快速验证的往往是“把两三个算子融成一趟 HBM 读写”或“定制带掩码/反量化的分块计算”。**Triton** 正是为打破这道生产力瓶颈而生——它把开发者的心智从“管好每一个线程”解放到“管好每一块张量”。

---

## 13.1 Triton 为什么改变算子开发方式

### 从“单线程视角”跃迁到“Block 级张量视角”

CUDA 和 Triton 的本质差异，在于你写的那段代码究竟代表谁在干活：

- **CUDA（SIMT 单线程视角）**：你写的代码描述**单个线程**如何计算标量坐标、搬运 1 个或 4 个元素、并在块内显式同步。你亲手操控 Shared Memory、Bank Padding 和每条向量 Load 指令。
- **Triton（SPMD Block 级张量视角）**：你写的代码描述**一整个 Program（等价于一个 CUDA Block / CTA）**如何加载一整块 Tile（如 $64 \times 32$ 的子矩阵）、执行块级张量运算并写回显存。至于这块数据具体切给哪几个 Warp、怎么合并访存、何时借 Shared Memory 做转置、如何做多级软件流水线（Software Pipelining），全部交由 Triton 编译器在编译期自动完成。

!!! tip "核心直觉：Triton 的“Python”不在 GPU 上跑"
    很多初学者担心“用 Python 写 Kernel 会不会很慢？”这是一个经典误解。`@triton.jit` 装饰的函数在首次调用时会被编译成 LLVM IR 再降至原生 **PTX / SASS 机器码**，执行时完全运行在 GPU SM 上，Python 解释器只出现在启动前的编译与参数分发路径上。

### Triton 语法与底层 CUDA / PTX 语义对照

把 Triton 原语逐行翻译回第 12 章的 CUDA 语义，就能一眼看穿编译器替你接管了什么、又把哪些旋钮留在了你手里：

| Triton 编程原语 | 编译后的底层 CUDA / PTX 语义 | 谁做决定 |
|---|---|---|
| `tl.program_id(axis)` | `blockIdx.x` / `blockIdx.y` / `blockIdx.z` | 开发者定 Grid 切分 |
| `pid * BLOCK + tl.arange(0, BLOCK)` | `blockIdx * blockDim + threadIdx` 叠加循环展开 | 编译器自动映射每个元素到具体线程 |
| `BLOCK` 与 `num_warps` | `blockDim = 32 × num_warps`，每线程分摊 `BLOCK / blockDim` 元素 | 开发者定粗细，编译器定分配 |
| `tl.load(ptr, mask=..., other=...)` | 带谓词（Predicated）的 `ld.global`，自动合并与向量化 | 编译器保证 Coalescing 与向量宽度 |
| `tl.store(ptr, val, mask=...)` | 带谓词的 `st.global` | 编译器生成防越界写指令 |
| `tl.max(x, axis)` / `tl.sum(x, axis)` | Warp 内 Shuffle 归约 + 跨 Warp Shared 暂存与 `bar.sync` | 编译器决定归约树与同步 |
| `tl.dot(a, b)` | Shared 暂存 + `ldmatrix` + Tensor Core `mma.sync` / `wgmma` | 编译器自动排布分片与软件流水 |
| `tl.constexpr` / `@triton.autotune` | 编译期模板特化 / 编译多份配置实测择优缓存 | 开发者定搜索空间，编译器自动特化 |

### 适用边界：何时用 Triton，何时回 CUDA / CUTLASS

Triton 与 CUDA/CUTLASS 不是替代关系，而是工程接力：

| 算子场景 | 推荐起手方案 | 工程依据 |
|---|---|---|
| 逐元素融合（RoPE、SiLU+Mul、RMSNorm、Softmax） | **Triton** | 核心收益来自减少 HBM 往返，编译器已能生成逼近带宽上限的合并访存 |
| 中等复杂度分块算子（Fused Attention、W8A8/W8A16 Matmul） | **Triton** | Block 级抽象可清晰表达 Online Softmax 与反量化流水，代码量仅为 CUDA 的 1/5 |
| 极致压榨单卡算力上限（手控 Warp Specialization、精细寄存器排布、定制指令调度） | **CUDA / CUTLASS** | 当 Profiling 证明瓶颈已下沉到编译器无法精准表达的指令级排布时，以验证过的 Triton 版作数值基准下沉重写 |

---

## 13.2 第一个 Kernel：1D 向量加法与 `mask` 边界保护

理解 Triton 的最快路径，是从一维向量加法 $z = x + y$ 拆开三个最核心的积木：`tl.program_id`、`tl.arange` 与 `mask`。

### 块切分算术与尾部越界保护

假设我们要对长度为 $N = 10{,}000$ 的一维向量做加法，设定每个 Program 一次处理 `BLOCK_SIZE = 1024` 个元素：

1. **Grid 尺寸**：需要启动的 Program 总数为向上取整除法 $\lceil 10{,}000 / 1024 \rceil = \mathbf{10}$ 个（在代码中用 `triton.cdiv(n_elements, BLOCK_SIZE)` 计算）。
2. **满块与尾块分工**：
    - `pid = 0..8` 共 9 个 Program，覆盖前 $9 \times 1024 = 9{,}216$ 个元素，全部有效。
    - `pid = 9` 是最后一个尾块，生成的下标区间为 $[9216, 10239]$：前 $10{,}000 - 9{,}216 = \mathbf{784}$ 个元素有效，末尾 **240 个 lane**（下标 $10{,}000 \sim 10{,}239$）已经越出张量边界！
3. **`mask` 双端拦截**：构造布尔向量 `mask = offsets < n_elements`，将其**同时**传入 `tl.load` 与 `tl.store`。尾部 240 个越界 lane 既不会向显存发起非法读取，也不会把垃圾值写进相邻张量的内存。

为什么 `BLOCK_SIZE` 必须是 2 的幂（如 256、512、1024，且上限为 $2^{20}$）？这是 `tl.arange` 的硬件编译硬约束：编译器需要按固定 2 的幂宽度对齐向量通道、切分 Warp 分工并排布 Shared Memory，非 2 的幂尺寸（如 1000）会直接在编译期报错。

### 完整 1D 向量加法实现

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 获取当前 Program 在 0 维网格上的编号（等价于 CUDA blockIdx.x）
    pid = tl.program_id(axis=0)
    # 2. 生成当前块负责的一整段连续下标（长度为 BLOCK_SIZE 的 1D 向量）
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 3. 生成越界保护掩码：拦截末尾超出 n_elements 的凑数 lane
    mask = offsets < n_elements
    # 4. 块级加载、计算与写回（load 与 store 必须对称使用 mask）
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def vec_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    # Grid 用 lambda 动态计算，方便后续直接接入 @triton.autotune 切换 BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

!!! warning "常见误区：Mask 必须“读、写对称”"
    新手最常犯的错是只给 `tl.load` 加 `mask`，却忘了给 `tl.store` 加 `mask`——结果尾块那 240 个未定义值直接覆写了紧邻 `output` 末尾的其他张量，GPU 往往不会立刻报 `illegal memory access`，而是让几十步之后的模型输出诡异发散。另外，首次调用 `vec_add` 会触发 JIT 编译，测速前务必先热身（Warmup）。

---

## 13.3 算子融合实战：Fused Softmax 的搬运账

向量加法只涉及一次读写，看不出融合威力；真正让 Triton 大放异彩的是 **Fused Softmax** 这类多阶段逐元素/归约算子。

### 朴素实现与融合版的 HBM 搬运账

对矩阵每一行执行数值稳定的 Softmax：

$$\operatorname{softmax}(x)_i = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}, \qquad m = \max_j x_j$$

先看为什么要先减去最大值 $m$：**平移不变性**保证分子分母同乘 $e^{-m}$ 不改变数学结果，但能彻底杜绝指数上溢。FP32 最大有限数约 $3.40 \times 10^{38}$，一旦 $x_i > \ln(3.40 \times 10^{38}) \approx \mathbf{88.72}$ 就会溢出为 `inf` 进而变成 `NaN`（FP16 更脆弱，最大仅 65504，超过 $\ln(65504) \approx \mathbf{11.09}$ 即溢出）。减去整行最大值后，所有指数输入 $\le 0$，输出严格落在 $(0, 1]$ 内。

再算显存搬运账。以一个 $4096 \times 4096$ 的 FP32 矩阵为例，单个张量占 $4096^2 \times 4\text{ B} = 67.1\text{ MB}$（64 MiB）：

| 实现方式 | 拆解后的 HBM 访问轮次 | 总读写次数 | $4096 \times 4096$ FP32 总流量 | 相对融合版倍数 |
|---|---|---|---|---|
| **朴素 PyTorch Eager** | `max`(1R) + `x - m`(1R,1W) + `exp`(1R,1W) + `sum`(1R) + `div`(1R,1W) | 5 读 + 3 写 = **8 遍** | **536.9 MB** | **4.0×** |
| **合并减法与指数（6 遍口径）** | `max`(1R) + `exp(x-m)`(1R,1W) + `sum`(1R) + `div`(1R,1W) | 4 读 + 2 写 = **6 遍** | **402.7 MB** | **3.0×** |
| **Triton Fused Softmax** | 整行读入片上完成 `max → sub → exp → sum → div` 后写回 | 1 读 + 1 写 = **2 遍** | **134.2 MB** | **1.0×** |

一次 `exp` 和一次除法都没少算，但融合把 3× 到 4× 的无谓 HBM 往返和多次 Kernel 启动开销一笔勾销。在 A100 2 TB/s 带宽下，仅理论搬运下界就从 0.268 ms 骤降到 0.067 ms。

### 整行驻留的 Fused Softmax 与 `other=-inf` 的双重职责

核心思路是**一行分配一个 Program**，用 `BLOCK_SIZE = triton.next_power_of_2(n_cols)` 将整行一次载入寄存器完成全部计算：

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["n_cols"],
)
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 关键：越界 lane 必须填 -inf，确保既不干扰 max，又让 exp(-inf) == 0 不污染分母
    row_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
    row = tl.load(row_ptrs, mask=mask, other=-float("inf"))

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    out_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(out_ptrs, softmax_output, mask=mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=block_size
    )
    return y
```

代码里有两处极具代表性的工程细节值得深记：

1. **`other=-float("inf")` 一石二鸟**：如果 `n_cols = 513`，`BLOCK_SIZE` 会向上补齐到 1024，尾部有 511 个无效 lane。在 `tl.load` 中填入 $-\infty$ 后，求 `tl.max` 时任何有限值都大于 $-\infty$；求 `tl.sum(tl.exp(...))` 时 $\exp(-\infty) = 0$，对分母贡献精确为零。如果漏写 `other`（或误填 `0.0`），当整行元素皆为负数时，`0.0` 会篡夺最大值且每个无效 lane 还会往分母里多加 1，结果彻底崩坏。
2. **`@triton.autotune` 与 `do_bench` 口径**：这里 `BLOCK_SIZE` 必须覆盖整行，不能交给 Autotune 随便缩减（Autotune 只看耗时不验正确性），因此只把 `num_warps` 放进搜索空间，以 `key=["n_cols"]` 按列宽缓存最优配置。计时推荐统一用 `triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")`：它会自动在每次重复前冲刷 L2（测得冷 L2 真实带宽），并用中位数过滤掉系统中断造成的偶发毛刺。

!!! note "边界预告：当一行放不进片上存储时怎么办？"
    上述写法要求整行都能塞进单个 Program 的寄存器。若列数升至数万（例如长上下文 Attention 的 $N = 32{,}768$ 行分数），单行不仅撑爆寄存器引发严重 Spilling，甚至会撞破 `tl.arange` 的 $2^{20}$ 长度天花板。届时必须把单行切块、动态维护运行最大值与分母——这正是第 14 章 **Online Softmax** 的出场时机。

---

## 13.4 从 1D 到 2D：最小 2D Triton GEMM 与广播指针

写完一维向量与按行 Softmax 后，迈向第 14 章 Fused Attention 和第 15 章量化 Matmul 的最后一块基石，是掌握 **2D Tile 切分、二维广播指针算术与 `tl.dot` 分块累加**。

### 二维网格切分与广播指针算术

考虑矩阵乘法 $C = A \times B$，其中 $A \in \mathbb{R}^{M \times K}$、$B \in \mathbb{R}^{K \times N}$、$C \in \mathbb{R}^{M \times N}$。我们将输出矩阵 $C$ 切成大小为 `BLOCK_M × BLOCK_N` 的二维网格，启动一个 2D Grid：

$$\text{grid} = \left(\left\lceil \frac{M}{\text{BLOCK\_M}} \right\rceil,\ \left\lceil \frac{N}{\text{BLOCK\_N}} \right\rceil\right)$$

每个 Program 认领一个坐标 `(pid_m, pid_n)`，负责算出输出子块 $C[\text{offs\_m}, \text{offs\_n}]$。为此，它需要沿内维 $K$ 每次迈步 `BLOCK_K`，读取一块 `[BLOCK_M, BLOCK_K]` 的 $A$ 子块与一块 `[BLOCK_K, BLOCK_N]` 的 $B$ 子块。

如何用一行表达式生成二维地址矩阵？答案是**张量广播（Broadcasting）**：

- 设行下标向量 `offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)`，形状为 `[BLOCK_M]`；
- 设内维下标向量 `offs_k = tl.arange(0, BLOCK_K)`，形状为 `[BLOCK_K]`；
- 用 `[:, None]` 把 `offs_m` 扩成列向量 `[BLOCK_M, 1]`，用 `[None, :]` 把 `offs_k` 扩成行向量 `[1, BLOCK_K]`，两者相加便自然广播成 `[BLOCK_M, BLOCK_K]` 的二维指针矩阵：

$$\text{a\_ptrs}[i, j] = \text{a\_ptr} + \text{offs\_m}[i] \cdot \text{stride\_am} + \text{offs\_k}[j] \cdot \text{stride\_ak}$$

同理，$B$ 子块的 `[BLOCK_K, BLOCK_N]` 二维指针矩阵为：

$$\text{b\_ptrs}[j, l] = \text{b\_ptr} + \text{offs\_k}[j] \cdot \text{stride\_bk} + \text{offs\_n}[l] \cdot \text{stride\_bn}$$

始终显式传入 `stride_am, stride_ak` 而不是把步长硬编码成 `K`，是 Triton 算子的黄金习惯：哪怕上游传入的是转置视图（`B.T`）或切片后的非连续张量，同一份 Kernel 也能一字不改地正确读取。

<figure class="book-figure">
  <img src="../../../assets/figures/triton-block-grid.svg" alt="Triton 编程模型图解：上半展示 N=10000、BLOCK=1024 的 10 个 1D Program 切分与尾块 pid=9 后 240 槽的 mask 拦截；下半展示列向量 offs_m[:, None]*stride_am 与行向量 offs_k[None, :]*stride_ak 广播相加生成 2D 内存地址网格" loading="lazy">
  <figcaption><span class="book-figure__label">图 13-1</span>Triton 的 Block 级坐标映射全景。上半回顾 13.2 节的 1D 切分与尾块 <code>mask</code> 拦截（<code>pid=9</code> 的前 784 槽有效、后 240 槽跳过）；下半拆解 2D GEMM 的核心广播算术——列向量与行向量相加直接生成 <code>[BLOCK_M, BLOCK_K]</code> 二维地址网格，沿 <code>K</code> 维每次步进 <code>BLOCK_K * stride_ak</code>。</figcaption>
</figure>

### `tl.dot` 的硬件契约与精度规则

在循环体内，核心乘加语句只剩一行 `acc = tl.dot(a, b, acc)`。使用它必须牢记三条硬件契约：

1. **维度下限 $\ge 16$**：`tl.dot` 在编译时会映射到 Tensor Core 的矩阵指令（如 Ampere 上 FP16 对应 `m16n8k16`，TF32 对应 `m16n8k8`）。因此 `BLOCK_M`、`BLOCK_N`、`BLOCK_K` 均须为 2 的幂且**至少为 16**，否则无法下沉到 Tensor Core 甚至直接编译失败。
2. **FP16 输入 + FP32 累加器**：即使输入 $A, B$ 是 FP16/BF16，累加器 `acc` 仍应初始化为 `tl.float32`，防止长 $K$ 维连加触发 FP16 溢位或严重舍入截断，循环结束后再 `.to(tl.float16)` 写回 $C$。
3. **FP32 输入默认走 TF32**：当 $A, B$ 为 FP32 且运行在 Ampere/Hopper 架构上时，`tl.dot` 默认 `input_precision="tf32"`（10 位尾数，A100 算力达 156 TFLOPS）。若需与 CPU/PyTorch 严格 IEEE FP32 比对，须显式传 `input_precision="ieee"`（或将 PyTorch 侧也开启 TF32 统一口径）。

### 完整最小 2D Triton GEMM 实现

下面这份不足 60 行的代码，就是从 1D 跨入 2D 分块算子的标准骨架——下一章的 Fused Attention 和第 15 章的量化 Matmul 全都在此骨架上生长：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 1. 定位当前 Program 负责的输出 Tile 坐标
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 2. 用广播构造初始 2D 指针矩阵：[BLOCK_M, BLOCK_K] 与 [BLOCK_K, BLOCK_N]
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # 3. 片上 FP32 累加器，防止 FP16 长内积累加溢出或精度损失
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 4. 沿 K 维分块滑窗累加
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        b_mask = k_mask[:, None] & (offs_n[None, :] < N)

        # 越界位置填 0.0，确保在矩阵乘中贡献精确为 0
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)

        # 指针沿 K 维步进一个 BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 5. 转换回输出精度，用 2D 掩码写回 C 子块
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4, num_stages=3,
    )
    return c
```

### 算术强度验算与迈向工业级 GEMM 的下一跳

用第 12 章学过的记账法验算这份 FP16 2D GEMM。每个 Program 计算 $2 B_M B_N K$ 次浮点运算，沿 $K$ 维逻辑读取 $2 (B_M K + K B_N)$ 字节（FP16 每元素 2 字节），因此块级逻辑算术强度为：

$$I_{\text{FP16}} = \frac{2 B_M B_N K}{2 K (B_M + B_N)} = \frac{B_M B_N}{B_M + B_N}\text{ FLOP/B}$$

取 $M = N = K = 1024$、`BLOCK_M = BLOCK_N = 64`、`BLOCK_K = 32`：

- **Grid 覆盖度**：$(1024/64) \times (1024/64) = 256$ 个 Program，足以覆盖 A100 的 108 个 SM（约 2.4 波）与 H100 的 132 个 SM（约 2.0 波）。
- **单块算术强度**：$\frac{64 \times 64}{64 + 64} = 32\text{ FLOP/B}$；若调大到 $128 \times 128$，单块算术强度翻倍至 **64 FLOP/B**。
- **L2 复用放大**：在实际执行中，多个共享同一行 $A$ 或同一列 $B$ 的 Program 同时驻留在 108 个 SM 上，借助 A100 40 MB L2 缓存进一步截留重复读取，把有效 HBM 算术强度推过 156 FLOP/B 的 FP16 脊点。

不过，上面这份最简版按 `(pid_m, pid_n)` 逐行发射 Program，在超大矩阵上会导致同一波次把整行 $B$ 冲出 L2。如何在 Program 调度顺序上引入 **Group-M L2 Swizzle** 提升缓存命中率，将在第 15 章结合量化 Matmul 一并揭晓。而现在，你已经手里握齐了 `tl.load/store` 掩码、Fused Softmax 与 2D `tl.dot` 全部工具——下一章，我们直接挑战大模型推理最核心的皇冠算子：**Fused Attention（FlashAttention）**。

---

## 参考文献

1. Philippe Tillet, H. T. Kung, David Cox, *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*（MAPL 2019），Tile 级编译抽象的设计原点。
2. OpenAI Triton Official Tutorials：`01-vector-add`、`02-fused-softmax`、`03-matrix-multiplication` 与语言参考手册（`triton.language`）。
3. Sasha Rush 等, *Triton Puzzles*（适合配合 `TRITON_INTERPRET=1` 可视化核对块级访存边界与广播形状）。
