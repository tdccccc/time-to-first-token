# 第 17 章 把自定义算子接入推理引擎 { .chapter-title }

<p class="chapter-summary">单测快 2 倍的自定义 Kernel 接进 vLLM 后，端到端延迟却纹丝不动——这往往卡在“注册、启用、分发”三道静默失效的门，或撞上了 Amdahl 定律的份额天花板。打通接入链路并做好归因，算子优化才算闭环。</p>

经过[第 12–16 章](chapter-12.md)的训练，我们已经能自己写出高效、数值对齐的 Triton Kernel，也能读懂工业级 CUDA 库的流水线设计。但在真实推理工程里，把一个 standalone 单测跑得飞快的 Kernel 塞进 vLLM 等推理引擎时，最常见的结局却是：**服务顺利启动、输出语义完全正确、日志干干净净，可 TTFT、ITL 和吞吐一丝不动。**

比“报错崩溃”更折磨人的是**静默回退（Silent Fallback）**：你甚至不知道自己的 Kernel 到底有没有上过 GPU。本章用 vLLM 的 `CustomOp` 机制解剖自定义算子的完整接入链路，再以 Amdahl 定律拆解端到端收益归因，最后给出推理引擎三大扩展接口的全景地图。

## 17.1 为什么“单测更快”不等于“引擎生效”：vLLM `CustomOp` 的三道门

在 vLLM 中，模型前向里的逐元素算子（如 `SiluAndMul`、`RMSNorm`、`RotaryEmbedding`）并不是写死的函数调用，而是继承自 `vllm/model_executor/custom_op.py` 的 `CustomOp` 基类。一个自定义替换想真正跑到 GPU 上，必须依次打通三道门：**注册（Registration）→ 启用（Enablement）→ 分发（Dispatch）**，如图 17-1 所示。

<figure class="book-figure">
  <img src="../../../assets/figures/vllm-customop-dispatch.svg" alt="vLLM CustomOp 注册、启用与分发三道门流程图：启动时通过 entry point 注册进 op_registry_oot，层初始化时依据 compilation_config.custom_ops 判定启用，随后按平台分发到 forward_cuda 或 forward_oot，任意一环未命中则静默回退到 forward_native 或父类实现" loading="lazy">
  <figcaption><span class="book-figure__label">图 17-1</span>沿着从左到右的执行生命周期读图：进程启动时经插件入口完成“注册”（写入 <code>op_registry_oot</code>），层对象构造时依编译配置判定“启用”，随后按硬件平台完成“分发”并缓存为 <code>_forward_method</code>。结论：三道门任何一道未打通，引擎都会静默走回退路径而不抛错，必须逐门核验并用 Profiler 查实 GPU Kernel 名单。</figcaption>
</figure>

### 17.1.1 三道门的生命周期拆解

很多接入事故的根源在于把三件发生在不同时间点的事混为一谈：

| 环节 | 触发时机 | 内部机制（`vllm/model_executor/custom_op.py`） | 静默失效表现 | 核心排查手段 |
|---|---|---|---|---|
| **1. 注册（Registration）** | 引擎启动调用 `load_general_plugins()` 并 import 插件子模块时 | `@TargetOp.register_oot` 将子类按**被替换类的类名**写入 `op_registry_oot`；构造时 `CustomOp.__new__` 用 `cls.__name__` 查表替换实例类型 | 表中查无此类，全程构造原版类，无任何报错 | 检查 `op_registry_oot.get("SiluAndMul")` 是否为你的子类 |
| **2. 启用（Enablement）** | 模型层 `__init__` 构造、读取全局 `compilation_config` 时 | `CustomOp.default_on()` 检查 `compilation_config.custom_ops`（`all` / `none` 基底叠加 `+name` / `-name`） | 类已替换，但 `dispatch_forward` 落入未启用分支，静默执行 `forward_native` | 打印层实例的 `op.name` 与 `op._forward_method.__name__` |
| **3. 分发（Dispatch）** | 层 `__init__` 中一次性绑定 `self._forward_method`，每次 `forward()` 直接调用 | 按当前平台选择 `forward_cuda`（NVIDIA）、`forward_hip`（ROCm）、`forward_cpu`、`forward_tpu`、`forward_xpu` 或 `forward_oot`（树外硬件） | 覆写了错误平台方法，悄悄继承执行父类默认 CUDA 实现 | 抓取 Torch Profiler / Nsight Systems，核对 GPU Kernel 名单与调用次数 |

### 17.1.2 三个最隐蔽的静默失效陷阱

!!! warning "常见误区：最容易踩中的三大静默失效陷阱"
    1. **陷阱 1：安装了插件包，但注册装饰器从未执行**。vLLM 通过 Python 打包元数据中的 `vllm.general_plugins` 入口点发现插件，并在启动时调用入口函数 `register()`。如果仅安装了包却漏写入口点，或者 `register()` 没有 `import` 包含 `@SiluAndMul.register_oot` 的子模块，装饰器根本不会运行——一个“写好了但没被触发”的插件，从模型输出上看与不存在毫无差别。
    2. **陷阱 2：`torch.compile` / Inductor 默认 `custom_ops=none`，且开关名写错大小写**。启用 TorchInductor 图编译时，由于自定义算子对图编译器而言是无法拆解融合的黑盒，vLLM 默认将基底设为 `none`（全部关闭，必须用 `--compilation_config.custom_ops '["+名称"]'` 显式打开）。更隐蔽的坑在于**注册名变了**：树内默认 `SiluAndMul` 的属性 `name` 是小写 `"silu_and_mul"`，但被 `@SiluAndMul.register_oot` 替换后，装饰器会把子类的 `name` 改写为**被替换类的类名 `"SiluAndMul"`**！如果你凭印象传入 `"+silu_and_mul"`，开关匹配落空，引擎会直接退回 PyTorch `forward_native`——数值完全正确，速度悄悄掉档。
    3. **陷阱 3：在 NVIDIA GPU 上覆写了 `forward_oot` 而非 `forward_cuda`**。名称里的 `oot`（out-of-tree）极具迷惑性：`register_oot` 指的是“树外代码插件替换”，但 `dispatch_forward` 里的 `forward_oot` 只服务于**自报为树外硬件平台**的后端（例如[第 19 章](chapter-19.md)的昇腾 `vllm-ascend`）。在 NVIDIA CUDA 平台上，引擎始终路由到 `forward_cuda`；只写 `forward_oot` 意味着你写的代码永远不会被调用，父类内置的 C++ CUDA Kernel 照常接管。

## 17.2 端到端接入解剖：`SiluAndMul` 插件完整链路

为什么选 `SiluAndMul` 作为接入载体？在 Llama-3.1-8B 与 Qwen3-8B 的 SwiGLU MLP 中，门控投影与上投影合并输出形状为 `[M, 2d]` 的张量，紧接着由 `SiluAndMul` 完成：

$$
y = \operatorname{SiLU}(x_{[:d]}) \odot x_{[d:]}, \qquad \operatorname{SiLU}(u) = u \cdot \sigma(u)
$$

它严格落在 `CustomOp` 的**“无状态、单 `forward`”边界**内——不需要管理 Paged KV 缓存，也不需要离线重排权重。

先算清它的访存账（FP16，每元素 2 字节）：

- **PyTorch Eager 两步版（`forward_native`）**：第一步 `silu` 读 $2Md$、写 $2Md$；第二步乘法读中间结果 $2Md$、读后半段 $2Md$、写输出 $2Md$，合计 **$10Md$ 字节**。
- **融合单遍版（Triton 或 vLLM 内置 C++）**：一次读入完整输入 `[M, 2d]`（$4Md$ 字节），片上算完直接写出 `[M, d]`（$2Md$ 字节），合计 **$6Md$ 字节**——相对 Eager 省去 **$40\%$** 的 HBM 往返。

但有一个关键前提必须提前看透：**vLLM 默认的 `SiluAndMul.forward_cuda` 调的是树内 C++ 融合算子 `torch.ops._C.silu_and_mul`，它的访存本来就已经是 $6Md$ 字节！** 因此，自定义 Triton Kernel 相对 Eager 省的是访存遍数，而相对 vLLM 引擎基线争的只是向量化宽度、grid 映射与启动开销的常数项。微基准测试必须同时测齐 **Eager / vLLM C++ / Triton** 三条基线，缺一条都会产生错觉。

### 17.2.1 打包入口与延迟注册：`pyproject.toml` 与 `__init__.py`

完整可安装的示例包位于 [silu_plugin](../modules/part3-operators/examples/silu_plugin/pyproject.toml)。插件包只需三份小文件，第一环是声明 `vllm.general_plugins` 入口点：

```toml
# pyproject.toml
[project.entry-points."vllm.general_plugins"]
my_plugin = "my_plugin:register"
```

在包的入口 `__init__.py` 中，用 `register()` 延迟导入真正带装饰器的子模块，避免普通 `import my_plugin` 产生意外副作用：

```python
# my_plugin/__init__.py
def register() -> None:
    # vLLM 启动时调用此入口；Python 的模块导入缓存保证装饰器不会重复执行
    from . import silu_and_mul_plugin  # noqa: F401
```

### 17.2.2 Triton Kernel 核心实现与精度契约

第二环是编写语义严丝合缝的 Triton 算子（完整含多维/空张量测试与三基线微基准的源码见 [kernel.py](../modules/part3-operators/examples/silu_plugin/my_plugin/kernel.py)）：

```python
# my_plugin/kernel.py（核心 Kernel 与入口）
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(
    x_ptr, y_ptr, x_row_stride, y_row_stride, d, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < d
    x1 = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0)
    x2 = tl.load(x_ptr + row * x_row_stride + d + cols, mask=mask, other=0.0)
    x1_f32 = x1.to(tl.float32)
    # 先在 FP32 中计算 SiLU，再 cast 回输入精度与 x2 相乘，与树内语义对齐
    silu = (x1_f32 * tl.sigmoid(x1_f32)).to(x1.dtype)
    tl.store(y_ptr + row * y_row_stride + cols, silu * x2, mask=mask)


def silu_and_mul_triton(x: torch.Tensor, BLOCK: int = 1024) -> torch.Tensor:
    if x.ndim < 1 or x.shape[-1] % 2:
        raise ValueError("last dimension must be even")
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("expected CUDA FP16/BF16 input")
    if not x.is_contiguous():
        raise ValueError("contiguous input is required; do not hide a copy in the timing")
    if BLOCK <= 0 or BLOCK & (BLOCK - 1) or BLOCK > 2**20:
        raise ValueError("BLOCK must be a supported power of two")

    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    if x.numel() == 0:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)

    x_2d = x.reshape(-1, x.shape[-1])
    m, n = x_2d.shape
    d = n // 2
    y = torch.empty((m, d), dtype=x.dtype, device=x.device)
    grid = (m, triton.cdiv(d, BLOCK))
    _silu_and_mul_kernel[grid](x_2d, y, x_2d.stride(0), y.stride(0), d, BLOCK=BLOCK)
    return y.reshape(output_shape)
```

实现中有三处极易出错的“承重墙”细节：

1. **行跨度是整行 $2d$**：`x1` 的偏移是 `row * x_row_stride + cols`，`x2` 是 `row * x_row_stride + d + cols`。一旦把行跨度错写成 `d`，读到的就会是错行数据。
2. **精度转换顺序不可颠倒**：必须先转 FP32 算完 `x1_f32 * tl.sigmoid(x1_f32)`，**先截断回输入精度（FP16/BF16）再与 `x2` 相乘**，才能与 `SiluAndMul.forward_native` 保持逐语义一致。
3. **显式拒绝隐式拷贝**：Wrapper 不偷偷调用 `.contiguous()`，防止把一次隐藏的全局显存拷贝混进算子计时里；同时通过 `reshape(-1, x.shape[-1])` 兼容模型可能传入的 3D `[batch, seq, 2d]` 或 2D `[tokens, 2d]` 张量。

### 17.2.3 `@SiluAndMul.register_oot` 覆写与三层递进验真

第三环是用 `@SiluAndMul.register_oot` 继承并覆写目标类（注意必须装饰在被替换类 `SiluAndMul` 上，而非 `CustomOp` 上；子类继承 `SiluAndMul` 以保留父类的 `__init__` 与 `forward_native` 后备路径）：

```python
# my_plugin/silu_and_mul_plugin.py
import torch
from vllm.model_executor.layers.activation import SiluAndMul

from .kernel import silu_and_mul_triton


@SiluAndMul.register_oot
class TritonSiluAndMul(SiluAndMul):
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul_triton(x)
```

相比粗暴的 Monkey Patch，`register_oot` 走的是引擎官方注册表：既不篡改引擎源码文件、升级抗干扰能力更强，又完整保留了按配置开关优雅退回 `forward_native` 的保险丝。

把插件以可编辑模式安装（`pip install --no-deps --no-build-isolation -e docs/modules/part3-operators/examples/silu_plugin`）后，按**从便宜到昂贵的三层递进验真**即可彻底消灭“到底跑没跑”的悬念：

```python
# 第 1 层（裸进程查表）：确认装饰器已写入 op_registry_oot
import my_plugin
from vllm.model_executor.custom_op import op_registry_oot

my_plugin.register()
assert op_registry_oot.get("SiluAndMul") is my_plugin.silu_and_mul_plugin.TritonSiluAndMul

# 第 2 层（引擎配置初始化后查分发）：把“注册了”与“启用了”彻底分开
from vllm.model_executor.layers.activation import SiluAndMul
from my_plugin.silu_and_mul_plugin import TritonSiluAndMul

op = SiluAndMul()
assert type(op) is TritonSiluAndMul
assert op.name == "SiluAndMul"
assert op._forward_method.__name__ == "forward_cuda"
```

注意不要在 REPL 中反复手工调用 `register_oot` 注册同名类，否则 vLLM 会触发 `Duplicate op name` 断言。通过前两层后，再用严格的 A/B 启动配置做**第 3 层验证（Profiler 实锤）**：

```bash
# 基线 A：关闭示例插件，跑引擎默认内置 C++ 融合路径
VLLM_PLUGINS='' vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype float16 --max-num-seqs 32

# 实验 B：加载插件并按实际 op.name 显式开启（切勿在基线 A 中用 -SiluAndMul，那会退回 native）
VLLM_PLUGINS=my_plugin vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype float16 --max-num-seqs 32 \
  --compilation_config.custom_ops '["+SiluAndMul"]'
```

独立运行一次 Torch Profiler（不要与压测计时混跑，避免探针扭曲延迟），只要 GPU Kernel 名单里出现 `_silu_and_mul_kernel` 且调用次数严格等于 `层数 × 步数`（Llama-3.1-8B 每步前向恰好 32 次），替换即确凿生效。

## 17.3 端到端收益的 Amdahl 归因

假设三层验证全部绿灯，你的自定义 Kernel 稳稳地跑在 GPU 上，单测甚至比原生 Eager 快了 $2\times$，可端到端压测 TTFT 和 ITL 却几乎没有变化——这是算子工程师必经的清醒一课，背后的铁律就是 **Amdahl 定律（Amdahl's Law）**：

$$
\text{Speedup} = \frac{1}{(1 - p) + \frac{p}{s}}
$$

其中 $p \in [0, 1]$ 是目标算子在端到端总耗时中的占比，$s$ 是该算子自身的加速比。**决定端到端上限的从来不是 $s$ 有多惊艳，而是 $p$ 有多大。**

### 17.3.1 用 Llama-3.1-8B 算清 `SiluAndMul` 的真实占比

以 **Llama-3.1-8B（FP16 权重约 $16\text{ GB}$、32 层、`hidden = 4096`、`intermediate = 14336`）** 为例，手算一遍 `SiluAndMul` 的访存体量：

- **单层每 token 搬运量**：读 `[1, 2d]` + 写 `[1, d]` = $3 \times 14336 \times 2\text{ B} = 86,016\text{ B} = \mathbf{84\text{ KiB}}$。
- **32 层每 token 总搬运量**：$32 \times 84\text{ KiB} = 2,688\text{ KiB} \approx \mathbf{2.62\text{ MiB}}$（约 $2.75\text{ MB}$）。
- **对比 Decode 阶段权重搬运**：在 `batch = 1` 的 Decode 中，每生成一个 token 仅模型权重就要从 HBM 搬运约 **$16.06\text{ GB}$**（还不算历史 KV Cache 读取）。`SiluAndMul` 占总搬运量的比例仅约：

$$
\frac{2.75\text{ MB}}{16.06\text{ GB}} \approx 0.017\% \quad (\text{不足万分之二！})
$$

即便把 Decode 并发拉大到 `batch = 64`，全层 `SiluAndMul` 搬运总量也才约 $168\text{ MiB}$，占权重搬运仍仅 **$1.1\%$** 左右；而在 $N = 4096$ 的长 Prompt Prefill 阶段，虽然 `SiluAndMul` 总搬运量线性涨到 $10.5\text{ GiB}$，但 Prefill 的时间大头被庞大的线性投影 GEMM 与 $O(N^2)$ 注意力牢牢占据，`SiluAndMul` 的真实时间占比依然只有约 $1\%\text{–}3\%$。

把不同占比 $p$ 代入 Amdahl 公式，天花板瞬间显形：

| 算子端到端时间占比 $p$ | 单算子提速 $s = 1.5\times$ | 单算子提速 $s = 2.0\times$ | 算子耗时归零极限（$s \to \infty$） | 工程投入价值判断 |
|---:|---:|---:|---:|---|
| **$0.5\%$** | $1.0017\times$ | $1.0025\times$ | $1.0050\times$ | 彻底淹没在压测噪声中，仅具教学或验证意义 |
| **$1.0\%$** | $1.0033\times$ | $1.0050\times$ | $1.0101\times$ | 即使提速无穷大也最多省 $1\%$，不值重兵投入 |
| **$5.0\%$** | $1.0169\times$ | $1.0256\times$ | $1.0526\times$ | 需严格控制变量的 A/B/A 压测才能稳定观测 |
| **$20.0\%$** | $1.0714\times$ | $1.1111\times$ | $1.2500\times$ | 值得深度定制优化（如长上下文 Attention） |
| **$50.0\%$** | $1.2000\times$ | $1.3333\times$ | $2.0000\times$ | 核心热点（如量化 Linear GEMM），收益直接兑现到 SLA |

!!! tip "核心直觉：分清“完全没变化”与“接上了但没变快”"
    “完全没变化”可能是三道门断了、根本没走到你的代码；“接上了但没变快”则是 Profiler 确证你的 Kernel 已跑满 32 层，只是因为 $p < 1\%$ 且基线本就已是融合 C++ Kernel，Amdahl 上限本来就只有零点几个百分点。**前者是接入 Bug，后者是极其宝贵的系统归因结论**——它阻止你在占比 $1\%$ 的算子上白耗三周，把你及时拉回真正占 $70\%$ 耗时的 GEMM 与 Attention 战场上。

### 17.3.2 端到端不见效的五维排查清单

在生产实践中，一旦遇到“单测很快、端到端不见效甚至变慢”，按以下五个维度逐一排查，几乎覆盖全部根因：

| 排查维度 | 典型症状 | 根因机制 | 定位与化解手段 |
|---|---|---|---|
| **1. 未命中分发** | 延迟与基线零差异，Profiler 无自定义 Kernel 名 | 插件未注册、`custom_ops` 开关名错配、覆写错平台方法 | 查 `op_registry_oot`、打印 `op.name` / `_forward_method`、用 Profiler 核对调用次数 |
| **2. 份额过小 / 基线已融合** | Profiler 命中，但端到端差值落入抖动噪声内 | Amdahl 上限过低（如 Decode 激活层），或树内默认已是 C++ 融合实现 | 用 Profiler 算子时间饼图确认 $p$，只在 $p \ge 5\%\text{–}10\%$ 的热点上投入定制开发 |
| **3. 图捕获破坏融合** | 接入后反比 `torch.compile` 基线更慢 | 黑盒 CustomOp 阻断了 Inductor 跨算子融合（如把量化融进激活 Epilogue），或触发图断（Graph Break）破坏 CUDA Graph 捕获 | 检查 TorchCompile 融合图日志与 CUDA Graph 捕获状态；优先向算子上下游进一步做大融合 |
| **4. 额外格式转换** | 算子本身快，但前后突然冒出 `aten::copy_` 或 `transpose` | 自定义 Kernel 强依赖连续布局或特定 dtype，隐藏的 `.contiguous()` / 格式转换吞掉全部收益 | 在 Wrapper 禁止静默拷贝，或把权重转置提前到模型加载期一次性完成 |
| **5. 动态形状退化** | 离线固定形状快，线上真实流量 ITL 抖动甚至变慢 | Continuous Batching 下每步 `num_tokens` 动态变化，导致 Triton JIT 频繁重编或小形状下 `BLOCK` 尾块浪费严重 | 启动阶段按分桶形状预热、将 `autotune` 键离散化分桶，或为小/大 $M$ 分别派发不同分块策略 |

## 17.4 更广阔的扩展地图：从逐元素算子到 Attention 与量化 GEMM

至此，我们已经看清了 `CustomOp` 的能力与边界：它专为**无状态、单 `forward`** 的逐元素算子而生。那么，前面花费重墨的[第 14/16 章 FlashAttention](chapter-14.md) 和[第 15 章量化 Matmul](chapter-15.md)，在 vLLM 这类工业推理引擎里又该插在什么位置？

答案是推理引擎的**三层专业化扩展接口**——每一层都精准匹配对应算子的状态复杂度与生命周期：

| 引擎扩展接口 | 对应本书章节 | vLLM 核心抽象与代码位置 | 状态边界与核心职责 | 典型定制目标 |
|---|---|---|---|---|
| **逐元素 `CustomOp`** | 本章（[第 17 章](chapter-17.md)） | `CustomOp` / `@Op.register_oot`（`vllm/model_executor/custom_op.py`） | 无状态、单 `forward`，仅关心输入/输出张量语义与平台分发 | `SiluAndMul`、`RMSNorm`、`RotaryEmbedding` |
| **Attention Backend** | [第 14 章](chapter-14.md)、[第 16 章](chapter-16.md) | `AttentionBackend` + `AttentionImpl` + `AttentionMetadata`（`vllm/attention/backends/`） | 深度耦合 Paged KV 显存布局（`block_table`、`slot_mapping`）、Prefill/Decode 混合元数据与 Chunked Prefill 切分 | FlashAttention-2/3、FlashInfer、Triton Paged Attention、DeepSeek MLA 后端 |
| **Quantization `LinearMethod`** | [第 8 章](chapter-08.md)、[第 15 章](chapter-15.md) | `QuantizationConfig` + `LinearMethodBase`（`create_weights` / `process_weights_after_loading` / `apply`） | 管辖离线权重打包、硬件专属重排（如 Marlin / FP8 布局）、scale 参数注册，以及运行时按 $M$ 动态派发 W8A8 / W8A16 / W4A16 Kernel | FP8 W8A8 GEMM、SmoothQuant INT8、AWQ / GPTQ / Marlin W4A16 |

把这三层接口与前面五章拼在一起，第三篇的完整工程闭环正式落成：

1. 用 **GPU 执行模型与 Roofline**（[第 12 章](chapter-12.md)）诊断算子瓶颈；
2. 用 **Triton**（[第 13–15 章](chapter-13.md)）快速构建 Fused Attention 与量化 Matmul，或在极端热点下沉 **CUTLASS / FlashAttention**（[第 16 章](chapter-16.md)）；
3. 通过 **`CustomOp` / `AttentionBackend` / `LinearMethod`**（本章）将算子无缝接入推理引擎，用三层验证法确认生效，并用 **Amdahl 定律**客观归因端到端收益。

下一篇（[第 18 章](chapter-18.md)起），我们将带着这套“硬件微架构 + 算子原理 + 引擎扩展接口”的完整方法论，跨出 NVIDIA 生态，走进国产 AI 芯片（以昇腾 Ascend 910B 与 `vllm-ascend` 为主线）的适配实战——你会看到，刚才在 17.1 节留下的伏笔 `forward_oot`，正是跨芯片算子接入的起点。

**延伸阅读**：[vLLM CustomOp 官方设计文档](https://docs.vllm.ai/en/latest/design/custom_op/)、[`vllm/model_executor/custom_op.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/custom_op.py)、[`vllm/model_executor/layers/activation.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py)、[vLLM 插件系统文档](https://docs.vllm.ai/en/latest/design/plugin_system/)、[Amdahl 定律原论文（1967）](https://dl.acm.org/doi/10.1145/1465482.1465560)。
