# 第 20 章 CANN 分层与 AscendC 算子全链路 { .chapter-title }

<p class="chapter-summary">解剖 CANN 五层逻辑架构与两代算子编程抽象的演化权衡，通过向量加法算清 AscendC 三级硬件流水与双缓冲的访存账，再沿 RMSNorm 走通“定义 → 编译 → 三处注册 → 三次解析”的完整调用链。</p>

在昇腾上跑通 `vllm-ascend` 之后，工程师很快会碰到两类更深层的问题：第一类是日志跳出 `aclnnAddRmsNormBias not found`，模型退回慢路径却找不到该修哪个文件；第二类是 Profiler 指出了一个占比可观的瓶颈算子，需要判断是否值得重写、以及到底该怎么读懂芯片侧的 AscendC 实现。要彻底看透这两类问题，不能停留在 Python 插件层，必须下潜到昇腾底座 **CANN（Compute Architecture for Neural Networks）** 的内部架构与算子全链路。

## 20.1 CANN 五层逻辑架构与两条执行路径

很多初学者把 CANN 当成“华为版 CUDA 算子库”，这会让人在排障时完全找不到北。实际上，CANN 是一套横跨编译、算子库、运行时与集合通信的完整软件栈，官方逻辑架构自顶向下分为五层，最底部则依托宿主机驱动与固件：

### CANN 五层与通用软件栈的精确映射

| CANN 逻辑分层 | 核心组件 | 对应第 18 章通用软件栈 | 失效时的典型症状与首选排查点 |
|---|---|---|---|
| **计算语言 / 接口层** | AscendCL（`acl`、`aclrt`、`aclnn` 对外编程接口） | SDK / 运行时（对外 API 面） | 调用处报参数/流/显存错误，或 `aclnnXxx not found`；先查接口名与 CANN 版本 |
| **计算服务层** | 算子库 AOL（NN/BLAS/融合算子）、OPP 算子包目录、调优引擎 AOE、框架适配 | 算子库 + 框架桥接（`torch_npu` 承接 PyTorch 侧） | 算子能跑但慢或精度异常、自定义算子包检索不到；先查 OPP 路径与算子覆盖 |
| **计算编译层** | 图引擎 GE（Graph Engine）、图编译器、算子编译器（Ascend C / TBE 编译器） | SDK / 运行时（编译子系统） | 算子源码编不过、片上 UB 超限、tiling 越界；先查编译日志与切分参数 |
| **计算执行层** | Runtime、Graph Executor、集合通信库 HCCL、DVPP/AIPP | SDK / 运行时（执行子系统） | Kernel 无法下发、流与事件同步错乱、多卡通信卡死；先查 Runtime 与 HCCL 日志 |
| **计算基础层** | 共享虚拟内存（SVM）、设备虚拟化、主机-设备通信 | SDK / 运行时（内核交互面） | 内存映射失败、设备句柄异常；核对设备节点与容器共享内存 |
| *驱动与固件（五层之下）* | HDK（Driver / Firmware） | 驱动 / 固件 | `npu-smi info` 不可见或 Health 非 OK；排查宿主机驱动安装 |

把两张架构图对齐，就能明白为什么**“写一个算子”绝不只是单层操作**：算子源码在**计算编译层**被编译成 NPU 二进制；产物作为算子包安装进**计算服务层**的 OPP（Operator Package）检索目录；运行时由框架通过**计算语言层**的 `aclnn` C 接口调用，再经 `torch_npu` 与 `vllm-ascend` 接入模型。只写完 kernel 源码却漏掉安装注册，在推理引擎眼里这个算子就跟不存在一样。

!!! note "工程取舍：把五层当排查地图，而非严格 OSI 调用栈"
    CANN 的五层划分是按功能职责切分的**定位地图**，不是“每层只能调紧邻下一层”的严格协议栈。例如 AscendCL 作为统一对外接口被多方直接调用，图引擎 GE 也可以被单算子直调绕过。拿到报错时，先看关键词（`aclnn`、UB overflow、HCCL、SVM）落入哪一层的职责域，排查方向立刻清晰。

### Eager 单算子直调 vs 图引擎 / ACLGraph 路径

同样一份模型代码跑在 CANN 底座上，存在两条截然不同的执行路径：

- **单算子直调（Eager 路径）**：每个 PyTorch 算子经 `torch_npu` 或插件扩展直接翻译成一次 `aclnn` 调用，由 Runtime 逐个下发到 NPU 执行。优点是动态 shape 零阻力、调试极直观；缺点是在 Decode 这类小计算量阶段，host 侧频繁下发指令会留下大量调度空泡。
- **图模式路径（ACLGraph 捕获重放 / GE 整图编译）**：前者类似 CUDA Graph，将一连串静态 shape 的下发序列捕获成图并整图重放，大幅压缩 CPU 下发开销；后者依托 GE 或 `torchair` 进行整图优化、跨算子融合与内存静态复用。

两条路径的排障入口完全不同：单算子路径盯紧 `aclnn` 接口、OPP 搜索路径与单算子 Profiler 耗时；图模式路径则必须先核对图编译日志、静态分档 shape 与入图范围（有没有发生 graph break），确认异常究竟出在算子本身还是图捕获阶段。

## 20.2 两代算子抽象演化：TBE 与 AscendC 的控制权权衡

阅读昇腾算子库源码或文档时，会频繁看到 **TBE（Tensor Boost Engine）** 与 **AscendC** 两个名词。它们共用“定义 → 编译 → 注册 → 调用”同一条接入链路，本质差别在于**开发者和编译器之间如何划分控制权**。

| 链路环节 | TBE（上一代 DSL 抽象） | AscendC（当前主力 C++ 抽象） |
|---|---|---|
| **定义方式** | 用 Python DSL 描述“算什么”及粗粒度调度原语（另有更底层 TIK 路径） | 用 C++ 算子类显式编写 `Init`（地址/Buffer 规划）与 `Process`（搬运/计算/搬回） |
| **编译器职责** | 读 DSL 生成 IR，由编译 Pass 自动完成**双缓冲、流水同步、片上内存分配**并生成代码 | 将 C++ 编译为 NPU 指令，校验 API/对齐/UB 容量；**队列深度与同步节拍交还开发者** |
| **调用与落点** | 打包进 OPP，经 `aclnn` 或框架算子名调用 | 同样打包进 OPP，经 `aclnn` 或 `torch.ops` 调用 |
| **核心权衡** | 代码短小、上手快，但精细优化受限于 DSL 表达能力与编译器启发式策略 | 代码偏长、需要手动算清 UB 与同步账，但可把搬运粒度、双缓冲与核间切分捏在手里 |

这种演进背后的系统逻辑非常清晰：TBE 像高级自动挡，把调度、片上内存、异步同步三件麻烦事隐藏在编译器 Pass 背后；但在大语言模型时代，FlashAttention、Fused MoE 等复杂融合算子往往需要极致压榨每一字节片上存储与流水线重叠，自动调度常常力不从心。AscendC 把这三项控制权重新交还给工程师，代价则是写错队列深度、对齐边界或同步依赖会直接导致变慢甚至静默算错。

### 编译产物“四件套”与三类失败现场

无论采用哪一代抽象，算子工程编译出来的都不是一个孤零零的 `.o` 文件，而是一套完整的**算子包**，包含四类缺一不可的产物：

1. **按目标 SoC 编译的 Kernel 二进制**：绑定具体芯片型号（如 `Ascend910B`），跨代往往不通用。
2. **算子原型与信息库**：声明输入输出的 shape 推导规则、dtype 支持列表与张量 format 约束，供 OPP 检索注册。
3. **Host 侧 Tiling 实现**：运行时根据真实输入 shape、硬件核数与片上 UB 容量，动态算出切分块数、每块长度与尾块长度。
4. **Host 侧 `aclnn` 接口库**：对外暴露的 C 头文件（`op_api/include`）与动态链接库（`op_api/lib`）。

理解了四件套，算子开发的失败就能精准归为三大第一现场：**编译错**（语法、不支持 API、静态 UB 超限）看编译日志；**加载错**（`aclnn` 找不到、符号未定义、SoC 不匹配）查 `ASCEND_CUSTOM_OPP_PATH` 与 `LD_LIBRARY_PATH`；**运行错**（输出数值偏差、越界、卡死）查 `op_host` 的 tiling 边界与 `op_kernel` 的对齐及队列同步。

## 20.3 AscendC 硬件流水线模型剖析

从 CUDA 转向 AscendC 时，最大的思维跃迁不是语法，而是**把计算看作三条独立硬件单元之间的接力流水线**。我们用最干净的逐元素向量加法 `add_custom`（计算 $z = x + y$）来拆解这套模型——没有归约也没有矩阵乘，恰好把 AscendC 的核心骨架毫无遮掩地暴露出来。

<figure class="book-figure">
  <img src="../../../assets/figures/ascendc-pipeline.svg" alt="AscendC 三级硬件流水线与双缓冲示意图：展示 CopyIn(MTE2)、Compute(VEC)、CopyOut(MTE3) 三个独立硬件单元通过 TQue 队列接力重叠执行的过程" loading="lazy">
  <figcaption><span class="book-figure__label">图 20-1</span>横向对比单缓冲串行执行（$3n$）与 <code>BUFFER_NUM=2</code> 双缓冲流水重叠（$n+2$）。核心结论：MTE2、VEC、MTE3 是独立硬件流水，双缓冲通过倍增片上 UB 队列深度隐藏等待空泡，但不减少总搬运字节数。</figcaption>
</figure>

### 最小向量加法 `add_custom` 骨架精读

下面是一段面向单核、整除长度场景的 AscendC 向量加法精简阅读骨架（完整生产工程还需包含核间切分、系统 workspace 判空与尾块处理）：

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;      // 队列深度为 2：开启双缓冲（Double Buffer）
constexpr int32_t TILE_LENGTH = 1024;  // 每块处理的元素个数（实际工程由 host tiling 传入）

class KernelAdd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {
        // 1. 绑定全局内存（GM）张量视图
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ float*)z, totalLength);
        // 2. 通过 TPipe 在片上统一缓冲区（UB）为三个队列分配双缓冲空间
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        // 3. 计算分块总轮数（多核场景下每个核仅负责自身分片）
        tileCount = totalLength / TILE_LENGTH;
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < tileCount; ++i) {
            CopyIn(i);   // MTE2: GM -> UB
            Compute(i);  // VEC:  UB 向量计算
            CopyOut(i);  // MTE3: UB -> GM
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t i) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
        DataCopy(yLocal, yGm[i * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t i) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        Add(zLocal, xLocal, yLocal, TILE_LENGTH);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t i) {
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<float> xGm, yGm, zGm;
    int32_t tileCount;
};

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);  // 编译期固化：仅使用向量核（AIV）
    GET_TILING_DATA(tilingData, tiling);             // 反序列化 host 侧算好的切分参数
    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength);
    op.Process();
}
```

逐一拆解这段代码里的五个核心对象与入口约定：

- **入口五参数与核类型**：`x`、`y`、`z` 是输入输出在全局内存（GM，即 HBM）的裸地址；`workspace` 是跨核交换或暂存用的显存工作区；`tiling` 承载 host 侧预计算的切分结构体，kernel 通过 `GET_TILING_DATA` 直接读取而不必在片上重算。`KERNEL_TYPE_AIV_ONLY` 在编译期声明仅启用**向量核（AIV，AI Vector Core）**；若为矩阵乘算子则需声明使用**矩阵核（AIC，AI Cube Core）**或 CV 混合模式——核类型直接决定了该算子的硬件算力天花板。
- **`GlobalTensor<T>` 与 `LocalTensor<T>`**：前者封装全局显存 HBM 上的张量视图，后者封装片上**统一缓冲区（UB，Unified Buffer）**上的局部张量视图。NPU 的向量计算指令不能直接读写 HBM，数据必须先搬进 UB 才能算。
- **`TPipe`、`TQue<..., N>` 与 `TBuf`**：`TPipe` 是算子内部的片上资源大管家，统一分配 UB 内存与硬件同步事件；`TQue` 是连接上下游硬件单元的队列，既承载 `LocalTensor` 内存块，又用 `EnQue` / `DeQue` 表达严格的生产者-消费者硬件事件同步；若需要纯临时片上暂存区（不跨流水交接），则用 `TBuf` 申请。

### 三级独立硬件流水与双缓冲的数学账

这段代码看起来像一个平平无奇的 `for` 循环，但落到昇腾 AI Core 硬件上，三个函数分别驱动**三个彼此独立的物理执行单元**：

- `CopyIn` → **MTE2**（GM 到 UB 的专用 DMA 搬运引擎）；
- `Compute` → **VEC**（片上 SIMD 向量计算单元）；
- `CopyOut` → **MTE3**（UB 到 GM 的写回搬运引擎）。

当 `BUFFER_NUM = 1` 时，队列里同一时刻只放得下一块数据：第 $i$ 块没算完并释放，MTE2 就无法预取第 $i+1$ 块，三条流水被迫退化成严格串行。假设三阶段各耗 $1$ 个时间单位、共切分 $n$ 块，串行总耗时为 $3n$。

一旦把 `BUFFER_NUM` 设为 `2`（双缓冲），队列里可以同时挂着两块 buffer：当 VEC 正在计算第 $i$ 块时，MTE2 已经并行把第 $i+1$ 块搬进另一块空闲 buffer，同时 MTE3 正把第 $i-1$ 块的结果写回 HBM。第 1 块经历完整的 3 个单位后，此后每经过 1 个单位就产出 1 块，总耗时骤降为：

$$
T_{\text{pipe}} = 3 + (n - 1) = n + 2, \qquad \text{流水加速比} = \frac{3n}{n + 2}
$$

当 $n = 4$ 时，耗时从 $12$ 压到 $6$（$2\times$）；当 $n = 1024$ 时，耗时从 $3072$ 压到 $1026$，逼近三级流水的理论极值 $3\times$。

但不要被 $3\times$ 冲昏头脑——立刻拉出 Roofline 算一笔访存账。以 $N = 2^{20} = 1{,}048{,}576$ 个 FP32 元素为例：读入 $x, y$ 共 $8\,\text{MiB}$，写回 $z$ 共 $4\,\text{MiB}$，总搬运量 $12\,\text{MiB} = 12{,}582{,}912\,\text{B}$，仅做 $1{,}048{,}576$ 次浮点加法，算术强度只有：

$$
\text{AI} = \frac{1{,}048{,}576\;\text{FLOP}}{12{,}582{,}912\;\text{B}} \approx 0.083\;\text{FLOP/B}
$$

这是一个极其典型的**显存带宽受限**算子——实际中 `Compute` 的时间远小于 `CopyIn` 与 `CopyOut`。双缓冲真正消除的是阶段之间的“干等空泡”，让搬运尽量贴近 HBM 带宽上限，**它绝不会让 12 MiB 的总搬运量变少**；同时，双缓冲会让三个队列的片上 **UB 占用直接翻倍**（本例中 $3 \times 2 \times 1024 \times 4\,\text{B} = 24\,\text{KiB}$）。当中间张量很多、UB 告急时，只能在“缩小单块 `TILE_LENGTH`”与“降为单缓冲”之间做工程取舍。

### 分块粒度与 32 字节对齐尾巴

分块大小改变的是搬运次数而非总搬运量：对 $2^{20}$ 个 FP32 元素，每块 1024 个需搬运 1024 次，每块 4096 个只需 256 次。块越大，单次 DMA 事务越饱满、同步交互次数越少，但 UB 压力越大；块越小则相反。

真实生产中的输入长度往往无法被 `TILE_LENGTH` 整除。假设向量总长为 $10{,}000$、块长为 $1024$：向上取整需 10 块，前 9 块各搬 1024 个元素，最后 1 块只剩 $10{,}000 - 9 \times 1024 = 784$ 个元素。如果粗暴按 1024 搬运，就会越界读取并污染相邻显存。更细的约束在于硬件对齐：**UB 搬运指令 `DataCopy` 要求按 32 字节对齐**（784 个 FP32 占 $3136\,\text{B}$ 刚好是 32 的倍数；但若是 785 个 FP32 即 $3140\,\text{B}$ 则非对齐）。针对非整除与非对齐尾块，必须由 host 侧 tiling 将尾块有效长度传入 kernel，并在 kernel 中借助 `DataCopyPad` 按有效字节搬运或走专用尾块分支处理。

## 20.4 一个算子的完整链路：以 `RMSNorm` 为例

弄懂单个 kernel 怎么在 AI Core 上跑起来之后，最后一个关键拼图是：**从一段 C++ 源码到 vLLM 前向里真正命中它，中间到底要过几道手？** 以 Transformer 每层出现两次的 `RMSNorm` 为例，它在 `vllm-ascend` 里恰好同时拥有 CANN 内置路径（`torch_npu.npu_rms_norm`）与插件自定义融合路径（`npu_add_rms_norm_bias`），能把整条链路照得清清楚楚。

<figure class="book-figure">
  <img src="../../../assets/figures/cann-op-chain.svg" alt="CANN 分层与一个 AscendC 算子的四环全链路：定义、编译、三处注册与最终前向调用" loading="lazy">
  <figcaption><span class="book-figure__label">图 20-2</span>左列是 CANN 到推理引擎的软件分层，右列是自定义算子落地的四道工序。核心结论：编译通过只完成了前半程，CANN OPP、PyTorch 与 vLLM 三处注册缺一不可。</figcaption>
</figure>

### 四道工序与三处注册

| 链路环节 | 核心产物 | 代码与路径落点 | 工程核验要点 |
|---|---|---|---|
| **1. 定义** | 算子原型、tiling 切分、kernel 逻辑 | `op_host/<op>.cpp`（原型/shape 推导/`TilingFunc`）、`op_host/<op>_tiling.h`、`op_kernel/<op>.cpp` | 核对输入输出 dtype、eps 属性传参方式与边界 shape 推导 |
| **2. 编译** | 算子包四件套（kernel/信息库/tiling/aclnn 库） | 算子工程 `build.sh` → `build_out/` 或 `vendors/<vendor>/` | 确认目标 SoC 与当前硬件一致，产物含 `op_api/include` 与 `op_api/lib` |
| **3. 注册** | **三处独立登记**（CANN / PyTorch / vLLM） | 见下文三处注册拆解 | 三处分别核验，任何一处漏登都会在别处引发隐蔽故障 |
| **4. 调用** | 模型前向透明路由至底层 kernel | `RMSNorm` → `CustomOp` → `AscendRMSNorm.forward_oot` → `aclnn` → Runtime → Kernel | 结合 `msprof` / Profiler 核对设备时间线上的真实 kernel 名 |

整条链路里最容易踩坑、也最常被低估的是第 3 环——**三处注册**，它们各自管着一套独立的名字空间：

1. **CANN 侧注册（OPP 搜索路径）**：将编译好的算子包安装进 OPP 目录，或通过 `ASCEND_CUSTOM_OPP_PATH` 指定自定义算子包路径，并把 `op_api/lib` 加入 `LD_LIBRARY_PATH`。缺这一步，底层 `aclnnAddRmsNormBias` 根本找不到二进制实现。
2. **PyTorch 侧注册（`torch.ops._C_ascend`）**：`torch_npu` 的内置算子由自身的 `op_plugin` 注册；而 `vllm-ascend` 的自定义融合算子则通过 `csrc/torch_binding.cpp` 里的 `TORCH_LIBRARY_EXPAND` 与 `EXEC_NPU_CMD` 暴露为 `torch.ops._C_ascend.<name>`，在导入插件 C++ 扩展时触发登记。若扩展加载失败，插件通常会降级关闭 custom ops。
3. **vLLM 侧注册（`CustomOp.register_oot`）**：`vllm_ascend/utils.py::register_ascend_customop()` 将类名 `"RMSNorm"` 映射到 `AscendRMSNorm`。缺这一步，即便前两处都已就绪，模型前向依然会跳过插件去走 `forward_native`。

### 三次名称解析与极速验证命令

三处注册在运行时精确对应**三次名称解析**：

1. **Python 引擎层**：`RMSNorm` 类构造时经 `op_registry_oot` 解析为 `AscendRMSNorm`；
2. **PyTorch 分发层**：Dispatcher 依据 `PrivateUse1` 分发键，把 `torch.ops._C_ascend.npu_add_rms_norm_bias` 解析至 C++ 绑定函数；
3. **CANN 运行时层**：OPP 根据算子信息库将 `aclnnAddRmsNormBias` 解析至匹配当前 SoC 与 shape 的 NPU 二进制 kernel。

排查链路是否畅通，四条命令就能把三处注册逐一验明：

```bash
# 1. 验证宿主机驱动与设备
npu-smi info
# 2. 验证框架层 torch_npu 可用
python3 -c "import torch, torch_npu; print('torch_npu:', torch_npu.__version__)"
# 3. 验证 PyTorch 侧自定义算子注册（必须导入 C++ 扩展模块）
python3 -c "import torch, torch_npu, vllm_ascend.vllm_ascend_C; print(hasattr(torch.ops._C_ascend, 'npu_add_rms_norm_bias'))"
# 4. 验证当前 shell 是否生效自定义 OPP 路径
printenv ASCEND_CUSTOM_OPP_PATH
```

### 典型报错 → 锁定四环速查表

拿到异常现象后，先把它精准归到四环之一，再直奔对应文件或环境变量，避免盲目重装：

| 观察到的异常现象 | 故障落在哪一环 | 第一现场排查动作 |
|---|---|---|
| `aclnnXxx not found` | **注册（CANN 侧）** | 查算子包是否安装、`ASCEND_CUSTOM_OPP_PATH` 是否在当前 shell 生效、SoC 型号是否匹配 |
| `undefined symbol` 或 `.so` 加载失败 | **注册（CANN 侧）或版本咬合** | 查 `LD_LIBRARY_PATH` 是否包含 `op_api/lib`，核对 CANN 与 `torch_npu` 版本矩阵 |
| 编译期报 UB 超限、tiling 越界 | **定义 / 编译** | 核算 `InitBuffer` 总量（留意 `BUFFER_NUM=2` 翻倍）与单块 `TILE_LENGTH` |
| 能跑通但输出数值偏差或随机乱码 | **定义 / 调用** | 查 32 字节对齐、非整除尾块、`EnQue`/`DeQue` 同步配对、host tiling 边界假设 |
| 不报错但很慢，Profiler 显示零碎算子 | **注册（vLLM / PyTorch 侧）** | 查 `CustomOp.register_oot` 是否命中、custom ops 开关是否被降级禁用 |

### 何时值得手写 AscendC：Amdahl 判据

手写并维护一个高性能 AscendC 融合算子的人力与版本升级成本不低。动手之前，先用 **Amdahl 定律**冷静算清端到端收益天花板：设目标算子占单步前向总耗时的比例为 $p$，手写优化把它加速了 $s$ 倍，则端到端理论加速比为：

$$
S_{\text{e2e}} = \frac{1}{(1 - p) + p / s}
$$

| 算子原始耗时占比 $p$ | 单算子加速 $s = 2\times$ | 单算子加速 $s = 5\times$ | 单算子加速 $s = 10\times$ |
|---:|---:|---:|---:|
| **5%** | $1.026\times$ | $1.042\times$ | $1.047\times$ |
| **20%** | $1.111\times$ | $1.190\times$ | $1.220\times$ |
| **50%** | $1.333\times$ | $1.667\times$ | $1.818\times$ |

这张表给出了两条铁一般的工程准绳：

1. **占比低于 5% 的算子，写到极致也撬不动端到端**（即便提速 10 倍，整体收益不足 5%）；通常只有当单算子占比超过 **10%** 且确认为回退或明显低效时，才值得进入手写候选池。
2. **边际收益递减极快**：即便占比高达 20%，把算子从 5 倍死磕到 10 倍，端到端仅多出不到 3% 的收益。

因此，算子工程的正确节奏是先借助 CANN 内置库、统一算子层与图捕获把能拿的“大块收益”吃干净，下一章我们就用系统化的算子覆盖分析，把真正值得动刀的那 1–2 个热点算子精准揪出来。

## 本章小结

- **CANN 五层即排障地图**：算子源码在编译层变成包含四件套的算子包，落入服务层 OPP，经接口层 `aclnn` 调用；单算子 Eager 与图模式（ACLGraph/GE）拥有完全不同的排查入口。
- **AscendC 用控制权换极限**：相比 TBE 由编译器包办调度与双缓冲，AscendC 将 MTE2（CopyIn）、VEC（Compute）、MTE3（CopyOut）三级独立硬件流水与 `BUFFER_NUM=2` 双缓冲交由开发者掌控；双缓冲隐藏等待空泡（理论耗时 $3n \to n+2$），但以 UB 翻倍为代价，无法减少总访存字节，且须严守 32 字节对齐与尾块边界。
- **三处注册与 Amdahl 纪律**：自定义算子要在 vLLM 生效，必须同时打通 CANN OPP、PyTorch `_C_ascend` 与 vLLM `register_oot` 三处注册；是否值得手写 AscendC，永远先拿 $1/((1-p)+p/s)$ 算清收益上限。

**延伸阅读**：[昇腾 CANN 总体架构与官方文档](https://www.hiascend.com/document)、[Ascend C 算子开发与流水线编程指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.md)、[vLLM `custom_op.py` 树外分发源码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/custom_op.py)。
