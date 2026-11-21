# 第 15 章 量化 Matmul 算子原理：W8A8 与 W8A16 { .chapter-title }

<p class="chapter-summary">Scale 的粒度决定了反量化能否提到 K 循环外；W8A8 吃 Tensor Core 峰值翻倍，W8A16 吃权重搬运减半——两条算子路径在 Decode 与 Prefill 上各占主场。</p>

在[第 8 章](chapter-08.md)中，我们从算法视角讨论了量化误差、离群值与精度权衡；在[第 13 章](chapter-13.md)中，我们写出了 FP16 的 Triton 2D GEMM。生产部署中常见的是 FP8（如 Hopper 上的 W8A8 FP8）与 W4A16（如 AWQ / GPTQ），而算子原理教学从对称 INT8 起手，因为它免去了 4-bit 位打包与非对称零点的工程细节，却完整暴露了 **scale 粒度如何决定 Kernel 架构**以及**算力红利与带宽红利如何分化**的核心规律。一旦掌握了对称 INT8 的 W8A8 与 W8A16 两条路径，迁移到 FP8 或 W4A16 只是替换解包指令与硬件计算单元。

## 15.1 Scale 粒度如何决定 Kernel 结构

从算法直觉看，量化粒度越细（管辖的元素越少），单个离群值污染的范围越小，精度越高。但到了写 GPU Kernel 时，量化粒度不再只是精度超参——**它直接改写内层循环的代数结构**。

### 15.1.1 四种粒度的广播维度与循环位置

对称 INT8 量化将浮点张量线性映射到 $[-127, 127]$，零点固定为 $0$：

$$
s = \frac{\max |w|}{127}, \qquad w_q = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w}{s}\right), -127, 127\right), \qquad \hat{w} = w_q \cdot s
$$

由于四舍五入最多偏离半个格点，任意元素的绝对误差上界为 $|\hat{w} - w| \le s/2$。不同粒度的本质是决定“多少个元素共享同一个 $s$”。在矩阵乘 $C = A \cdot B$（其中激活 $A \in \mathbb{R}^{M \times K}$，权重 $B \in \mathbb{R}^{K \times N}$）中，决定 Kernel 架构的分水岭只有一个问题：**scale 是否依赖归约维下标 $k$？**

**情形一：scale 不依赖 $k$。** 例如激活使用 per-token scale $s_a[i]$（形状 $[M]$），权重使用 per-channel scale $s_b[j]$（形状 $[N]$）。此时两者对归约维 $k$ 都是常数，可以直接提到求和符号之外：

$$
\hat{C}_{i,j} = \sum_{k=0}^{K-1} \left(a_q[i, k] \cdot s_a[i]\right)\left(b_q[k, j] \cdot s_b[j]\right) = s_a[i] \cdot s_b[j] \cdot \sum_{k=0}^{K-1} a_q[i, k] \, b_q[k, j]
$$

举个极简算例验证：$s_a = 0.01$、$s_b = 0.02$，$a_q = [3, -5]$、$b_q = [7, 2]$。先反量化再乘加得到 $0.03 \times 0.14 + (-0.05) \times 0.04 = 0.0022$；先做纯整数点积再乘一次 scale 得到 $0.01 \times 0.02 \times (21 - 10) = 0.0022$。两者数学严格等价，但在 $K = 4096$ 的真实线性层里，后者让整个 $K$ 维归约停留在整数 Tensor Core 中，循环结束才做一次浮点缩放。

**情形二：scale 依赖 $k$。** 例如权重使用 per-group 量化，沿 $K$ 维每 $g$ 个元素（常见 $g = 128$）共享一个 scale $s_b[\lfloor k/g \rfloor, j]$（形状 $[K/g, N]$）。此时不同分组的缩放因子不同，无法提出最外层求和，必须在 $K$ 循环内按组加载 scale 并完成反量化或分组累加：

$$
\hat{C}_{i,j} = \sum_{p=0}^{K/g - 1} s_b[p, j] \sum_{k = p g}^{(p+1)g - 1} a[i, k] \, b_q[k, j]
$$

下表总结了四种常见粒度对 Kernel 结构与广播维度的直接影响：

| 粒度 | 典型对象 | Scale 形状 | 相对输出 $[M, N]$ 的广播方式 | 依赖的下标 | 能否提到 $K$ 循环外 |
|---|---|---|---|---|---|
| **per-tensor** | 权重或静态激活 | 标量 $(1,)$ | 全块标量广播 | 无 | 能（Epilogue 一次乘） |
| **per-channel** | 权重 $B[K, N]$ | $[N]$ | `s_b[None, :]` | 输出列 $j$ | 能（Epilogue 一次乘） |
| **per-token** | 动态激活 $A[M, K]$ | $[M]$ | `s_a[:, None]` | 输出行 $i$ | 能（Epilogue 一次乘） |
| **per-group** | 权重 $B[K, N]$ | $[K/g, N]$ | $K$ 循环内按组取 `s_b[g_idx, :]` | 归约组 $k/g$ 与列 $j$ | 不能（须在 $K$ 循环内处理） |

!!! tip "核心直觉：Scale 的成本不在字节数，而在出现的位置"
    以 Llama-3.1-8B 的 $4096 \times 4096$ 线性层为例，$N = 4096$ 的 FP32 per-channel scale 仅占 **$16\text{ KiB}$**，$M = 1024$ 的 per-token scale 仅占 **$4\text{ KiB}$**，对比 $16.78\text{ MB}$ 的 INT8 权重几乎可以忽略。真正影响算子效率的不是读取 scale 的字节数，而是两件事：一是 scale 落在 $K$ 循环外还是循环内；二是**动态激活量化的准备成本**——当 $M = 1024, K = 4096$ 时，单独跑一次动态 per-token 量化需读 $8.39\text{ MB}$ FP16 激活并写 $4.19\text{ MB}$ INT8 激活（共 $12.58\text{ MB}$，占同等 FP16 Matmul 总搬运量 $50.33\text{ MB}$ 的 **$25.0\%$**），因此生产引擎通常把动态量化融进上游 RMSNorm 或激活层的 Epilogue 中；而在 $M = 32$ 的 Decode 场景下，该搬运仅 $0.39\text{ MB}$（占 $1.2\%$），几乎无感。

### 15.1.2 为什么 Scale 必须在 FP32 中计算与保存

量化数据准备看似只有几行代码，却藏着一个极易引发静默 `NaN` 的浮点陷阱。

!!! warning "常见误区：在 FP16 下调用 `clamp_min(1e-8)` 仍会除零"
    IEEE 754 FP16 的最小正正规数是 $2^{-14} \approx 6.10 \times 10^{-5}$，最小正次正规数是 $2^{-24} \approx 5.96 \times 10^{-8}$。如果直接在 FP16 张量上计算 `(amax / 127.0).clamp_min(1e-8)`，由于 `1e-8` 小于最小正次正规数的一半，**FP16 会直接将其下溢舍入为 `0.0`**！一旦输入中出现全零行（如 padding token）或极小激活值，`x / scale` 会立刻变成 `0.0 / 0.0 = NaN` 并经矩阵乘污染整张输出张量。

因此，统计 `amax`、计算 `scale`、执行除法与舍入都必须先提升到 **FP32**，并且 `scale` 必须以 **FP32** 传给后续的 Matmul Kernel（绝不可再转回 FP16）。下面两段函数与本书完整可运行示例 [quant_matmul.py](../modules/part3-operators/examples/quant_matmul.py) 保持完全一致：

```python
import torch


def quant_per_token(x):
    # x: [M, K] FP16/BF16/FP32 -> INT8 激活 + FP32 scale [M]
    # 统计、scale、除法与舍入全程留在 FP32，防止 FP16 下溢除零
    xf = x.float()
    amax = xf.abs().amax(dim=1)
    scale = (amax / 127.0).clamp_min(1e-8)
    q = torch.round(xf / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def quant_per_channel(w):
    # w: [K, N] FP16/BF16/FP32 -> INT8 权重 + FP32 scale [N]
    wf = w.float()
    amax = wf.abs().amax(dim=0)
    scale = (amax / 127.0).clamp_min(1e-8)
    q = torch.round(wf / scale[None, :]).clamp(-127, 127).to(torch.int8)
    return q, scale
```

全零行在 FP32 下得到 `scale = 1e-8`（严格正且有限），`0.0 / 1e-8 = 0`，既无除零也无精度畸变；遇到本身含 `NaN`/`Inf` 的脏输入则应在调用前直接拒绝，不要将其混作量化舍入。

## 15.2 两条核心算子路径：W8A8 与 W8A16

确定了 scale 粒度后，量化 Matmul 在 GPU 上分化为两条结构截然不同的算子路径，如图 15-1 所示。

<figure class="book-figure">
  <img src="../../../assets/figures/quant-matmul-paths.svg" alt="量化 Matmul 双路径架构对比：左路为 W8A8 INT8 Tensor Core 累加后在 Epilogue 乘 scale，右路为 W8A16 在 K 循环内将 INT8 权重于寄存器反量化为 FP16 再进 FP16 Tensor Core" loading="lazy">
  <figcaption><span class="book-figure__label">图 15-1</span>对比两条路径的数据流与反量化位置：左路（路径 A，W8A8）让 INT8 激活与 INT8 权重直接进入 INT8 Tensor Core 累加到 INT32，循环结束在 Epilogue 统一乘 <code>s_a[:, None] * s_b[None, :]</code>；右路（路径 B，W8A16）保持 FP16 激活，在 K 循环内将 INT8 权重于寄存器现场反量化成 FP16 再进 FP16 Tensor Core。结论：路径 A 赚算力峰值翻倍与双侧搬运减半，路径 B 赚权重搬运减半且免去激活量化误差。</figcaption>
</figure>

### 15.2.1 路径 A：W8A8 INT8 `tl.dot` + Epilogue 反量化

在 W8A8 路径中，激活 $A_q$（per-token）与权重 $B_q$（per-channel）均以 INT8 存入 HBM 并直接送入 Tensor Core 执行 `int8 × int8 -> int32` 累加，循环结束后在 Epilogue 阶段一次性转为 FP32 并乘上 `sa[:, None] * sb[None, :]`。设计这条路径有三个关键硬件约束：

1. **累加器必须是 INT32**：INT8 绝对值上限是 $127$，单次乘法最大值 $127^2 = 16,129$。仅一个 `BLOCK_K = 32` 块的最坏局部和就是 $16,129 \times 32 = 516,128$，已达到 INT16 上限（$32,767$）的 **$15.8$ 倍**——第一个 $K$ 块就会溢出回绕成负数。而全长 $K = 4096$ 的最坏和为 $16,129 \times 4096 = 66,064,384$，仅占 INT32 上限（$2^{31}-1 \approx 2.15 \times 10^9$）的 $3.1\%$，余量超 $32$ 倍。
2. **对齐 Tensor Core 指令形状**：在 NVIDIA Ampere（A100，CC 8.0）上，INT8 矩阵乘指令为 PTX `mma.sync.aligned.m16n8k32` 一族。将 `BLOCK_M`、`BLOCK_N` 取 16 的倍数、`BLOCK_K` 取 32 的倍数（如 `64, 64, 32`），才能稳定落到 INT8 Tensor Core 上兑现 **624 TOPS** 稠密算力。
3. **正视 Epilogue 的执行单元差异**：每个输出元素在 Epilogue 执行 4 次操作（`int32 -> fp32`、乘 `sa`、乘 `sb`、`fp32 -> fp16`），相对总乘加量 $2MNK$ 的操作数占比为 $\frac{4MN}{2MNK} = \frac{2}{K}$，在 $K = 4096$ 时仅 $0.049\%$。但切忌把“操作数占比”等同于“时间占比”：Epilogue 运行在 CUDA Core 上（A100 FP32 峰值 $19.5\text{ TFLOPS}$），而主循环跑在 INT8 Tensor Core 上（$624\text{ TOPS}$），两者吞吐相差 **$32\times$**！按时间折算，Epilogue 在 $K = 4096$ 时占约 $1.6\%$ 仍可忽略，但若 $K$ 缩至 $256$ 以下，Epilogue 的耗时占比便会显著抬头。

### 15.2.2 路径 B：W8A16 Fused Dequant（寄存器内现场反量化）

当激活的动态范围过大、激活量化会伤及模型质量时，生产系统往往只量化权重、保留 FP16 激活（W8A16 / W4A16）。此时有两种实现方式：

- **两步基线（Unfused）**：先跑一个独立反量化 Kernel 把整块 INT8 权重展开成 FP16 写回 HBM，再调用常规 FP16 Matmul 把它读出来。以 $4096 \times 4096$ 权重为例，每个权重元素要过手 **5 字节**（读 INT8 $1\text{ B}$ + 写 FP16 $2\text{ B}$ + 读 FP16 $2\text{ B}$），仅权重搬运就高达 **$83.89\text{ MB}$**！
- **融合反量化（Fused Dequant）**：在 Matmul 的 $K$ 循环内直接从 HBM 读取 INT8 权重分块 `b_int8`，在**寄存器**里乘上已提前加载的 `sb` 转成 FP16，紧接着送入 FP16 `tl.dot`。反量化后的 FP16 权重从不落盘 HBM，生命周期仅限于当次循环迭代。

在 $M = 32, N = K = 4096$ 的 Decode 典型形状下，把激活与输出（共 $0.52\text{ MB}$）一起算进总搬运账：

| 方案（$M=32, N=K=4096$） | 权重搬运量 | 总搬运量（含激活与输出） | 相对两步基线省流量 | 相对直接 FP16 省流量 |
|---|---:|---:|---:|---:|
| 两步基线（Dequant + FP16 Matmul） | $83.89\text{ MB}$ | $84.41\text{ MB}$ | 基线 | $-147.7\%$（更慢） |
| 直接 FP16 Matmul（未量化权重） | $33.55\text{ MB}$ | $34.08\text{ MB}$ | $59.6\%$ | 基线 |
| **Fused W8A16（寄存器内反量化）** | **$16.78\text{ MB}$** | **$17.30\text{ MB}$** | **$79.5\%$** | **$49.2\%$** |

把三行放在一起才分得清两笔账：**融合**消灭的是两步基线的无意义 HBM 往返（省 $79.5\%$）；**量化**相对直接 FP16 省掉的是权重本身的一半体积（省 $49.2\%$）。

不过，“融合总是更快”并不成立。每个权重元素在寄存器中需要做 3 次 CUDA Core 操作（`int8 -> fp32`、乘 `sb`、`fp32 -> fp16`——先在 FP32 中算准再转 FP16 喂给 Tensor Core），并且被 $M$ 方向的每个程序块各重复展开一次。相对主循环 $2MNK$ 的计算量，反量化操作数占比为：

$$
\frac{3 \cdot \lceil M / \text{BLOCK\_M} \rceil \cdot K \cdot N}{2 M N K} = \frac{3 \cdot \lceil M / \text{BLOCK\_M} \rceil}{2 M}
$$

当 $M = 32, \text{BLOCK\_M} = 64$ 时，该比例为 $4.7\%$，但此时算子深陷带宽受限区，反量化指令完全可以与 HBM 读延迟重叠；当 $M = 1024, \text{BLOCK\_M} = 128$ 时，比例摊薄至 $1.2\%$，但此时算子已进入计算受限区，W8A16 依然走 FP16 Tensor Core（A100 上限仍是 $312\text{ TFLOPS}$），省下的字节换不成时间，反量化指令反而白占指令发射带宽。

### 15.2.3 两条路径的核心代码对照（关键 3 行差异）

去掉外层地址初始化与校验样板，对比两条 Triton 路径的主循环与 Epilogue，你会发现两者骨架高度一致，核心差异恰恰浓缩在 **3 处**（完整可运行的固定配置 Kernel、Host Wrapper、边界校验与基准测试入口见 [quant_matmul.py](../modules/part3-operators/examples/quant_matmul.py)）：

```python
import triton
import triton.language as tl


@triton.jit
def matmul_int8_core(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    scale_a_ptr, scale_b_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    # ... 省略 GROUP_M 坐标映射，见 15.3 节 ...
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # [差异 1] W8A8：INT32 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0)
        # [差异 2] W8A8：INT8 × INT8 直接进 INT8 Tensor Core
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # [差异 3] W8A8：K 循环外一次性乘双侧 FP32 scale
    sa = tl.load(scale_a_ptr + offs_m, mask=offs_m < M, other=0.0)
    sb = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def matmul_w8a16_core(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    scale_b_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    # per-channel scale 不依赖 k，循环外只读一次
    sb = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=1.0)

    # [差异 1] W8A16：FP32 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        b_int8 = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0)
        # [差异 2] W8A16：寄存器内先按 FP32 反量化再转 FP16 进 FP16 Tensor Core
        b = (b_int8.to(tl.float32) * sb[None, :]).to(tl.float16)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # [差异 3] W8A16：Epilogue 直接截断写回 FP16，无需再乘 scale
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

!!! warning "常见误区：指针步进、二维 Mask 与双层验收"
    1. **指针步进写错步长**：$K$ 循环内必须是 `b_ptrs += BLOCK_K * stride_bk`（行主序下 `stride_bk = N`）。如果误写成 `stride_bn`（$= 1$），指针每步只挪 `BLOCK_K` 个元素而不是 `BLOCK_K` 行，程序不报越界却悄无声息算出一片垃圾值。
    2. **漏掉 $M/N$ 方向 Mask**：方阵测试时 $M, N, K$ 都是分块的整倍数，只写 `offs_k < k_rem` 也能蒙混过关；一旦遇到 Decode 典型形状 $M = 32 < \text{BLOCK\_M} = 64$，缺少 `offs_m[:, None] < M` 会立刻越界读脏数据。
    3. **混淆 Kernel 误差与算法量化误差**：校验正确性必须拆成两层——先拿**同一份量化后的 `a_q, b_q, sa, sb`** 与高精度参考（如 CPU INT64 点积后乘 scale）比对，确认 Kernel 实现零 Bug；再拿输出与原始 FP16 矩阵乘比对，评估算法级量化误差。

## 15.3 L2 Cache Swizzle（`GROUP_M`）与 `@triton.autotune`

写对数值只是第一步。同样的分块大小，仅仅改变程序块的遍历顺序与按形状自动选配分块，往往就能把有效带宽和吞吐拉升一个台阶。

### 15.3.1 用 8×8 输出块看清 `GROUP_M` 如何拯救 L2 缓存

设想输出矩阵被切成了 **$8 \times 8$ 共 64 个输出块**，每一行块需要 $A$ 的 1 条行带，每一列块需要 $B$ 的 1 条列带。观察调度器依次发射前 9 个输出块（CTA）时 L2 缓存发生了什么：

- **默认行优先调度（`GROUP_M = 1`）**：前 9 个块是 `(0,0), (0,1), ..., (0,7), (1,0)`。为了算完这 9 个块，SM 必须从 HBM 加载 **$2$ 条 $A$ 行带 + 全部 $8$ 条 $B$ 列带 = $10$ 条数据带**。更糟的是，当第 0 行一路扫到 `(0,7)` 时，早先读入的 $B$ 第 0 列带大概率已被后续 $B$ 列带挤出 L2 缓存；等到第 1 行 `(1,0)` 开算，只能重新从 HBM 把 $B$ 第 0 列带再搬一遍——形成典型的 **L2 缓存颠簸（Cache Thrashing）**。
- **分组列优先调度（`GROUP_M = 3`，实际工程常取 `8`）**：每 `GROUP_M` 行划为一个水平组，组内按**列优先**顺序派发。此时前 9 个块变成 `(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)`——它们集中在 $3 \times 3$ 的方阵局部窗口里，只需加载 **$3$ 条 $A$ 行带 + $3$ 条 $B$ 列带 = $6$ 条数据带**！每条刚读进 L2 的 $B$ 列带都会立刻被同列的 3 个块连续复用，相对行优先直接减少 **$40\%$** 的 HBM 重复读取。

在 Triton 中，只需要 7 行整数映射就能把一维 `pid` 转换为 L2 友好的 `(pid_m, pid_n)`：

```python
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

注意一个边界直觉：当 Decode 形状 $M = 32 \le \text{BLOCK\_M} = 64$ 时，`num_pid_m = 1`，`group_size_m` 自动退化为 $1$，分组调度不起作用；`GROUP_M` 真正发威是在 $M$ 方向切出多个行块的大 batch 或 Prefill 阶段。

### 15.3.2 无冲突的 `@triton.autotune` 写法与计时要点

固定的 `BLOCK_M=64, BLOCK_N=64` 不可能在 Decode（$M=32$）和 Prefill（$M=1024$）上同时最优。以 **A100（108 个 SM）** 与 $N = 4096$ 为例：

- **Decode（$M = 32$）**：`64×64` 切出 $1 \times 64 = 64$ 个程序块，SM 覆盖率约 $59\%$；若盲目增大到 `128×128`，程序块骤减为 $1 \times 32 = 32$ 个，近 **$70\%$ 的 SM 彻底空转**！因此小 $M$ 偏爱小 tile 换并行度。
- **Prefill（$M = 1024$）**：`128×128` 仍能切出 $8 \times 32 = 256$ 个程序块（超过 108 个 SM 两倍以上），此时大 tile 显著提高寄存器与共享内存复用、压低 HBM 流量并摊薄反量化开销。

把分块搜索交给 `@triton.autotune` 时，**有一个极常见的 API 签名冲突**：`triton.Config` 提供的元参数（`BLOCK_M`、`BLOCK_N`、`BLOCK_K`、`GROUP_M`、`num_warps`、`num_stages`）会由自动调优器注入，**Host Wrapper 绝对不能再以关键字参数重复传入同名参数**，并且发射网格必须改写为接收 `META` 字典的 `lambda` 函数：

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8, num_stages=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_int8_autotuned_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    scale_a_ptr, scale_b_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    sa = tl.load(scale_a_ptr + offs_m, mask=offs_m < M, other=0.0)
    sb = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_int8_autotuned(a_q, b_q, scale_a, scale_b):
    M, K = a_q.shape
    N = b_q.shape[1]
    c = torch.empty((M, N), dtype=torch.float16, device=a_q.device)
    # grid 必须是接收 META 的 callable，且调用处不再重复传 BLOCK_* / num_warps
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    matmul_int8_autotuned_kernel[grid](
        a_q, b_q, c, M, N, K,
        a_q.stride(0), a_q.stride(1), b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1), scale_a.contiguous(), scale_b.contiguous(),
    )
    return c
```

!!! note "工程取舍：微基准计时的 L2 陷阱"
    使用 `triton.testing.do_bench(..., warmup=25, rep=100, return_mode="median")` 计时时（Triton 3.x 默认返回均值，建议显式指定 `median` 抗抖动），需先预热一次完成 JIT 与 Autotune。特别要注意评估“两步基线 vs Fused W8A16”时的 **L2 陷阱**：`do_bench` 只在两次完整调用**之间**清空 L2 缓存，清不掉同一次调用里“第一步写完 FP16 权重、第二步紧接着读它”的 L2 命中。A100 的 L2 高达 **40 MB**，刚好能兜住 $4096 \times 4096$ 的 $33.55\text{ MB}$ FP16 权重！因此仅看微基准耗时可能会低估融合收益，用 Nsight Compute（`ncu`）核对真实 DRAM 读字节数才不会误判。

## 15.4 Decode vs Prefill 收益拆账：A100 Roofline 定量分水岭

把 W8A8 与 W8A16 两条路径放回[第 1 章](chapter-01.md)的 Roofline 模型，就能把“何时该用哪条路径”算得一清二楚。

以 **NVIDIA A100 80GB SXM**（稠密 FP16 峰值 **$312\text{ TFLOPS}$**，稠密 INT8 峰值 **$624\text{ TOPS}$**，HBM2e 带宽 **$2\text{ TB/s}$**，FP16 脊点 **$156\text{ FLOP/B}$**，INT8 脊点 **$312\text{ FLOP/B}$**）上运行 Llama-3.1-8B 典型的 $N = K = 4096$ 线性层为例（按理想单次搬运模型，A100 40 MB L2 在小 $M$ 下接近该假设，不计 2:4 结构化稀疏）：

- 总计算量恒为 $\text{FLOPs} = 2 M N K$；
- 理想搬运字节数：
  - **FP16**：$B_{\text{FP16}} = 2(MK + KN + MN)$；
  - **W8A8**（输出 FP16，忽略 $16\text{ KiB}$ scale）：$B_{\text{W8A8}} = MK + KN + 2MN$；
  - **W8A16**（激活与输出 FP16，权重 INT8）：$B_{\text{W8A16}} = 2MK + KN + 2MN$。

代入从 Decode 到 Prefill 的五档典型 $M$，得到完整拆账表：

| 形状（$N=K=4096$） | 路径 | 理想搬运量 | 算术强度（FLOP/B） | 瓶颈区间 | A100 理论时间下限 | 相对 FP16 理论加速 | 收益来源 |
|---|---|---:|---:|---|---:|---:|---|
| **$M = 32$**（Decode） | FP16 | $34.08\text{ MB}$ | $31.5$ | 带宽受限 | $17.0\ \mu\text{s}$ | $1.00\times$ | 基线（权重占 $98.5\%$） |
| $M = 32$ | **W8A16** | $17.30\text{ MB}$ | $62.1$ | 带宽受限 | **$8.7\ \mu\text{s}$** | **$1.97\times$** | 权重搬运减半 |
| $M = 32$ | **W8A8** | $17.17\text{ MB}$ | $62.5$ | 带宽受限 | **$8.6\ \mu\text{s}$** | **$1.98\times$** | 权重搬运减半 |
| **$M = 128$**（大 batch Decode） | FP16 | $35.65\text{ MB}$ | $120.5$ | 带宽受限 | $17.8\ \mu\text{s}$ | $1.00\times$ | 接近 FP16 脊点 |
| $M = 128$ | **W8A16** | $18.87\text{ MB}$ | $227.6$ | **计算受限** | $13.8\ \mu\text{s}$ | $1.29\times$ | 先撞 $312\text{ TFLOPS}$ 顶板 |
| $M = 128$ | **W8A8** | $18.35\text{ MB}$ | $234.1$ | 带宽受限 | **$9.2\ \mu\text{s}$** | **$1.94\times$** | 吃带宽且未撞 $624\text{ TOPS}$ |
| **$M = 1024$**（Prefill） | FP16 | $50.33\text{ MB}$ | $682.7$ | 计算受限 | $110.1\ \mu\text{s}$ | $1.00\times$ | 算力打满 |
| $M = 1024$ | **W8A16** | $33.55\text{ MB}$ | $1024.0$ | 计算受限 | $110.1\ \mu\text{s}$ | **$1.00\times$** | 无算力增益 + 反量化开销 |
| $M = 1024$ | **W8A8** | $29.36\text{ MB}$ | $1170.3$ | 计算受限 | **$55.1\ \mu\text{s}$** | **$2.00\times$** | Tensor Core 峰值翻倍 |
| **$M = 4096$**（长 Prefill） | FP16 | $100.66\text{ MB}$ | $1365.3$ | 计算受限 | $440.5\ \mu\text{s}$ | $1.00\times$ | 算力打满 |
| $M = 4096$ | **W8A16** | $83.89\text{ MB}$ | $1638.4$ | 计算受限 | $440.5\ \mu\text{s}$ | **$1.00\times$** | 无算力增益 |
| $M = 4096$ | **W8A8** | $67.11\text{ MB}$ | $2048.0$ | 计算受限 | **$220.3\ \mu\text{s}$** | **$2.00\times$** | Tensor Core 峰值翻倍 |

### 15.4.1 三段分水岭与“为什么不是 4 倍”

从表中的数据可以直接推导出一条极其清晰的架构判断——**W8A8 的两笔收益（搬运减半 + 峰值翻倍）绝不会相乘变成 $4\times$**：

1. **纯带宽红利区（A100 上 $M \lesssim 84$）**：令 W8A16 的算术强度等于 A100 FP16 脊点 $156$，解 $\frac{2MNK}{2MK + KN + 2MN} = 156$ 得 $M \approx 84.4$。在此左侧，FP16、W8A16、W8A8 全部落在带宽斜线上，耗时由权重体积决定。此时 W8A16 与 W8A8 的搬运量仅差 $0.8\%$（$17.30\text{ MB}$ vs $17.17\text{ MB}$），理论加速双双逼近 **$2\times$**——而 W8A16 免去了激活量化误差与动态量化成本，是 Decode 小 batch 的绝对首选。
2. **过渡收窄区（A100 上 $84 < M < 169$）**：W8A16 因为搬运减半、算术强度翻倍，率先在 $M \approx 84$ 撞上 FP16 $312\text{ TFLOPS}$ 的算力顶板，而原始 FP16 要到 $M \approx 169$ 才撞上顶板。因此在这个夹缝里，W8A16 的理论加速从 $1.9\times$ 逐渐滑向 $1.0\times$；而 W8A8 凭借 $624\text{ TOPS}$ 的高顶板，依然稳守近 $2\times$ 提速。
3. **计算受限区（A100 上 $M \ge 169$）**：FP16 与 W8A16 同受 $312\text{ TFLOPS}$ 限制，W8A16 省下的权重字节彻底换不成时间，循环内的 CUDA Core 反量化反而让它持平或略慢于 FP16。此时唯有 **W8A8**（A100 INT8 $624\text{ TOPS}$，或 H100 原生 FP8 **$1,979\text{ TFLOPS}$**）能凭硬件峰值翻倍再拿走一笔近 $2\times$ 的计算红利（若未把动态激活量化融进上游算子，单独搬运 $12.58\text{ MB}$ 在 $M=1024$ 时约耗 $6.3\ \mu\text{s}$，会吃掉约 $11\%$ 的 W8A8 净收益）。

!!! note "工程取舍：换到 H100 上分水岭怎么漂移？"
    在 **NVIDIA H100 80GB SXM** 上，FP16 脊点从 A100 的 $156$ 跃升至 **$295\text{ FLOP/B}$**（$989.5\text{ TFLOPS} \div 3.35\text{ TB/s}$）。按同样公式反解，W8A16 的纯带宽红利区直接延伸到 **$M \approx 172$**！但反过来，H100 提供了原生硬件 **FP8 Tensor Core（$1,979\text{ TFLOPS}$）**，精度远好于 INT8 且免去了 INT32 累加的某些限制，因此在 Hopper 及后续架构上，W8A8 FP8 成为了兼顾 Prefill 与大 batch Decode 的主力生产方案。

**延伸阅读**：[Triton 矩阵乘官方教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)（`GROUP_M` Swizzle 与 Autotune 范式）、[Triton `tl.dot` API 文档](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)、[NVIDIA A100 Tensor Core Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf)、[PTX ISA `mma` 指令手册](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)。
