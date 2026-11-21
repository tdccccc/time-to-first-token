# 第 19 章 昇腾部署架构与 vllm-ascend 插件机制 { .chapter-title }

<p class="chapter-summary">理顺昇腾推理环境的“版本咬合链”与三层自下而上诊断模型，解剖 <code>vllm-ascend</code> 平台插件如何将一次算子调用透明路由至 NPU，并建立与 CUDA 部署的逐项排障对照。</p>

有了[第 18 章](chapter-18.md)的五层软件栈地图，接下来我们把同一个大语言模型真正部署到昇腾 Ascend 910B 上。把一条 `vllm serve` 命令从 NVIDIA GPU 搬到昇腾 NPU，命令本身几乎一字不差，但真正让工程师第一次上手就频频碰壁的，是隐藏在命令之下的**版本咬合约束**、**三层启动诊断顺序**以及**树外（out-of-tree）平台插件的分发架构**。先跑通服务并看清插件全貌，下一章再钻进 CANN 与 AscendC 的底层算子链路。

## 19.1 为什么说昇腾环境是一条“版本咬合链”

在 NVIDIA GPU 上，多数开发者习惯先装驱动、再 `pip install vllm`，多数情况下就能顺利跑通。但在昇腾生态里，这套“松散安装”的经验几乎必然翻车——因为昇腾推理环境不是几份互不干涉的软件包，而是一条环环相扣、跨层校验 ABI 的**版本咬合链**。

### 五环成对约束

从宿主机内核态一直到用户态推理引擎，每一环都在编译或运行期调用下一环的头文件、动态库或内部扩展接口：

```text
宿主机 HDK（驱动 / 固件）
   │  内核态设备节点与驱动 ABI
   ▼
CANN Toolkit / Ops + NNAL（含 ATB 神经网络加速库）
   │  AscendCL / aclnn 动态库与环境变量
   ▼
PyTorch  ◀──强版本配对──▶  TorchNPU（torch_npu）
   │  PrivateUse1 分发键与 C++ 扩展符号
   ▼
vLLM     ◀──同版本同步──▶  vllm-ascend（+ 可选 Triton-Ascend 等）
```

这条链上有三组最容易咬断的**硬配对约束**：

1. **PyTorch 与 TorchNPU 强配对**：`torch_npu` 深入绑定 PyTorch 的底层 C++ Dispatcher 与存储结构，必须精确匹配次版本号（例如 PyTorch `2.10.0` 对应 TorchNPU `2.10.0.post7`），单升 PyTorch 会立即触发符号未定义或 `import` 失败。
2. **vLLM 与 `vllm-ascend` 同版本锁死**：`vllm-ascend` 实现的是 vLLM 内部的 `Platform`、`WorkerBase`、`AttentionBackend` 等接口，随 vLLM 主干演进极快，两者必须保持版本对齐（例如 `vllm-ascend v0.30.0` 对应一套验证通过的组合：Python `3.12`、CANN `9.2.1`、NNAL `9.2.1`、PyTorch `2.10.0`、TorchNPU `2.10.0.post7`、vLLM `0.30.0`；可选依赖如 `Triton-Ascend 3.6.0.post2` 亦需核对适用芯片型号）。
3. **CANN 与宿主机 HDK 兼容区间**：CANN 运行时依赖宿主机驱动提供的设备接口与共享内存机制，升级 CANN 大版本时必须核对宿主机 HDK 是否在官方兼容矩阵内。

### 容器镜像锁什么、不锁什么

工程实践中，强烈建议优先采用官方预构建容器镜像（如 `quay.io/ascend/vllm-ascend:<tag>`），而非在裸机上从零逐包编译。容器镜像一次性锁死了 **CANN、NNAL、PyTorch/TorchNPU、vLLM 与 `vllm-ascend`** 整个用户态软件栈，省去大量编译与路径冲突。

但必须牢记一条边界：**容器镜像只锁用户态，绝不锁宿主机驱动、固件与设备节点**。以昇腾 Atlas A2（910B）单卡容器启动为例，必须把宿主机设备节点、管理通道与驱动动态库显式映射进容器：

```bash
export IMAGE=quay.io/ascend/vllm-ascend:v0.30.0
docker run --rm --name vllm-ascend --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -p 8000:8000 -it "$IMAGE" bash
```

不同昇腾硬件形态的挂载项存在明显差异：例如 Atlas A3 采用双 DIE（双裸片）设计，单张物理卡需映射两个 `/dev/davinci*` 设备节点；而 300I Duo 等推理卡的设备节点与驱动路径也有专属清单。切忌把某一款卡的 `docker run` 脚本机械照搬到所有机型上。

!!! warning "常见误区"
    遇到 `ImportError` 或插件激活失败时，最糟糕的反应是随手执行 `pip install -U vllm` 或只升级 `torch_npu`。这只会把整条咬合链彻底拧断。正确动作是立刻用 `pip show vllm vllm-ascend torch torch_npu` 冻结当前版本快照，再对照 `vllm-ascend` 官方 README 的兼容矩阵整组对齐。

## 19.2 三层自下而上诊断模型

环境装好后，不要直接甩一个 8B 或 32B 模型上去赌运气。遵循[第 18 章](chapter-18.md)“自下而上”的排障铁律，用**三层最小探针**在 2 分钟内把驱动、框架与插件彻底验明正身。

### 三层探针逐级验证

1. **第一层：驱动与设备可见性（`npu-smi info`）**
   在容器内执行 `npu-smi info`，核对三项信息：预期卡号全部列出、每张卡的 `Health` 状态为 `OK`、HBM 容量与当前占用（`HBM-Usage`）正常。如果这一层看不到卡或健康状态异常，立刻停下来查宿主机 HDK 与容器 `--device` 挂载，向上排查毫无意义。

2. **第二层：框架设备识别与基础算子调度（`torch.npu`）**
   用一段 10 行 Python 脚本验证 `torch_npu` 能正常加载、识别设备，并在 NPU 上真正跑通一次张量加法与同步：

```python
import torch
import torch_npu  # 显式导入以向 PyTorch 注册 PrivateUse1/NPU 后端

assert torch.npu.is_available(), "未检测到可用 Ascend NPU，请检查 CANN 环境变量与驱动"
device_count = torch.npu.device_count()
x = torch.randn(2, 2, device="npu:0", dtype=torch.float16)
y = x + x
torch.npu.synchronize()
print(f"设备: {y.device}, 可见卡数: {device_count}, 结果均值: {y.float().mean().item():.4f}")
```

3. **第三层：推理引擎平台插件激活（vLLM 启动日志）**
   仅仅 `import vllm_ascend` 成功，**绝不等于** vLLM 已经启用了昇腾平台。必须启动一次最小模型（或冒烟命令），在启动日志中亲眼核对三条关键信号：
   - 发现插件组：`Available plugins for group vllm.platform_plugins`
   - 列出昇腾条目：`- ascend -> vllm_ascend:register`
   - 平台激活生效：`Platform plugin ascend is activated`

   如果机器上同时安装了多个厂商插件，还必须检查环境变量 `VLLM_PLUGINS`——该变量用于白名单过滤允许加载的插件名，配置错误会把 `ascend` 直接屏蔽掉。

### 2³ = 8 种组合中最常见的 4 类典型故障

三层探针每层只有“通过 / 失败”两种结果，共 $2^3 = 8$ 种状态，但工程中 90% 的故障高度集中在以下 4 类典型组合：

| `npu-smi`（驱动层） | `torch.npu`（框架层） | 插件激活（引擎层） | 故障精准定位 | 高频根因与首选处理动作 |
|---|---|---|---|---|
| 失败 | 失败 | 失败 | **驱动 / 固件与容器挂载层** | 宿主机 HDK 异常，或 `docker run` 漏挂 `/dev/davinci*`、`/dev/devmm_svm`、驱动库目录 |
| 通过 | 失败 | 失败 | **SDK 环境或框架配对层** | 当前 shell 未 `source` CANN/NNAL 的 `set_env.sh` 环境变量脚本，或 PyTorch 与 `torch_npu` 版本不配对 |
| 通过 | 通过 | 失败 | **平台插件注册层** | `vllm` 与 `vllm-ascend` 版本不一致、`setup.py` 入口点缺失，或被 `VLLM_PLUGINS` 环境变量过滤 |
| 通过 | 通过 | 通过（但跑崩/极慢） | **图捕获、显存或算子库层** | 图模式捕获异常（先关图跑 Eager 定位）、HBM OOM（调小并发/长度）或关键算子静默回退（转第 20/21 章） |

<figure class="book-figure">
  <img src="../../../assets/figures/vllm-ascend-plugin.svg" alt="vllm-ascend 平台插件架构：展示 Python 入口点注册、NPUPlatform 接管引擎核心组件以及 CustomOp 算子路由至昇腾实现的完整链路" loading="lazy">
  <figcaption><span class="book-figure__label">图 19-1</span>从左上角插件入口点出发，向右追踪平台级接管（Worker、ModelRunner、Attention、通信、编译）与算子级分发（CustomOp 查表路由到 <code>forward_oot</code>）。核心结论：插件不是零散补丁，而是接管芯片相关全部执行面的平行架构层。</figcaption>
</figure>

## 19.3 `vllm-ascend` 平台插件架构剖析

理解 `vllm-ascend` 的源码组织，能把排障效率提升一个量级：今后无论是查注意力后端走哪条路、查某算子有没有芯片实现，还是改图捕获配置，都不必在整个仓库盲搜，而是直接按图索骥。

### 顶层模块职责与引擎 Hook 地图

`vllm-ascend` 靠打包配置 `setup.py` 里的两组 Python 入口点（entry points）接入 vLLM：

- **`vllm.platform_plugins`**：声明 `ascend = vllm_ascend:register`。vLLM 启动时调用 `register()`，拿到类路径 `vllm_ascend.platform.NPUPlatform` 并实例化，正式确立当前硬件平台。
- **`vllm.general_plugins`**：注册平台之外的扩展能力（如 KV Connector、自定义模型加载器、服务级 Profiling 钩子等），在各个 Worker 进程中加载生效。

`NPUPlatform` 继承自 vLLM 的 `Platform` 基类，几个类属性直接定义了昇腾后端的身份契约：`_enum = PlatformEnum.OOT`（树外平台）、`device_type = "npu"`、`device_control_env_var = "ASCEND_RT_VISIBLE_DEVICES"`，以及 `dispatch_key = "PrivateUse1"`（PyTorch 为非原生第三方加速卡保留的算子分发键）。

整个插件按职责清晰划分为以下顶层模块（以 v0.30.0 结构为例，具体文件名随版本微调）：

| 插件目录 / 文件 | 对接的 vLLM 核心接口 | 核心实现类与职责 |
|---|---|---|
| `platform.py` | `Platform` 基类 | `NPUPlatform`：设备识别、环境变量控制、向引擎声明该用哪个注意力/通信/编译类 |
| `worker/` | `WorkerBase`、`GPUModelRunner` | `NPUWorker` 与 `NPUModelRunner`：设备初始化、权重加载、组批、KV 缓存与图捕获编排 |
| `ops/` | `CustomOp.register_oot` | 芯片侧逐算子实现（含 `layernorm.py`、`activation.py` 及 `ops/triton/`） |
| `attention/` | `AttentionBackend` | `AscendAttentionBackend`（Dense）、`AscendMLABackend`（MLA）、`AscendFABackend`（FA3）等，负责注意力计算与 KV 布局 |
| `quantization/` | 量化配置与线性层接口 | W8A8、W4A16 等昇腾侧量化加载与推理算子接入 |
| `distributed/` | `CommunicatorBase` | `NPUCommunicator`：封装 HCCL 集合通信原语与多卡并行协同 |
| `compilation/` | `get_compile_backend`、图封装接口 | `AscendCompiler` 与 `ACLGraphWrapper`：图编译与 ACLGraph 静态图捕获重放 |
| `_cann_ops_custom/` | 自定义 C++/AscendC 扩展 | 随插件编译打包的昇腾自定义融合算子（如 `npu_add_rms_norm_bias`） |

此外，仓库中通常还有一个 `patch/` 目录（分为 Worker 启动前的全局 platform patch 与每个进程内的 worker patch）。**它不是标准 Hook，而是标准插件接口尚未覆盖之处的定向补丁**——如果在源码里看到某个 vLLM 核心类的行为和上游不同却找不到注册点，去 `patch/` 按类名检索往往能立刻水落石出。

### 追踪一次 `RMSNorm` 的路由全景

模型代码里写的依然是标准 vLLM 的 `RMSNorm(hidden_size, eps)`，一行针对昇腾的分支都没有。那么运行前向时，它究竟是如何精准落到昇腾芯片实现上的？整个过程跨越四站：

1. **模型构造算子**：Llama 或 Qwen3 的 Decoder Layer 调用 `RMSNorm(...)`；`RMSNorm` 继承自 vLLM 的 `CustomOp`。
2. **`CustomOp.__new__` 查树外注册表**：在初始化阶段，`vllm_ascend/utils.py::register_ascend_customop()` 已经把 `"RMSNorm": AscendRMSNorm`、`"SiluAndMul": AscendSiluAndMul` 等映射通过 `CustomOp.register_oot()` 登记进全局表 `op_registry_oot`。于是实例化 `RMSNorm` 时，`__new__` 截获类名并直接返回 `AscendRMSNorm` 实例。
3. **绑定树外前向方法**：`CustomOp.__init__` 检查当前平台，因为 `NPUPlatform` 的 `_enum` 为 `PlatformEnum.OOT`（`is_out_of_tree()` 为真），自动将前向入口绑定到 `forward_oot`。
4. **执行昇腾实现**：前向触发 `vllm_ascend/ops/layernorm.py::AscendRMSNorm.forward_oot`。它根据是否存在残差输入及自定义算子开关，选择调用融合自定义扩展 `torch.ops._C_ascend.npu_add_rms_norm_bias`，或 CANN 内置接口 `torch_npu.npu_rms_norm`。

!!! tip "核心直觉：未登记算子的隐性回退"
    这套查表机制同时揭示了“静默回退”的根源：如果某个模型用到的 `CustomOp` 没有被注册进 `op_registry_oot`，`CustomOp` 基类的默认 `forward_oot` 会自动退回到 `forward_native`——即在 NPU 上把融合算子拆成一堆零碎的 PyTorch 原生操作逐个下发。程序不报错、输出也能看，但性能会大打折扣。这也正是[第 21 章](chapter-21.md)算子覆盖分析要重点排查的对象。

注意力的路由则独立于 `CustomOp`：它由平台级方法 `NPUPlatform.get_attn_backend_cls()` 统一决策，根据模型是否使用 MLA、稀疏注意力或 FA3 特性，直接把整套计算与分页 KV 布局指向 `attention/` 下的对应后端。

### 多进程 `spawn`、ACLGraph 图捕获与常规部署要点

把上述插件机制落回一条生产可用的部署命令，有三个工程细节值得牢记：

- **强制 `spawn` 多进程模式**：导出 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。Linux 默认的 `fork` 会把主进程已经初始化的 NPU/ACL 设备上下文一并复制给子进程，极易引发运行时死锁或显存状态损坏；`spawn` 启动干净的 Python 子进程，彻底规避上下文污染。
- **先 Eager 通畅，再开 ACLGraph 图模式**：`vllm-ascend` 通过 ACLGraph（及图优化开关如 NPUGraph Ex）实现类似 CUDA Graph 的捕获与重放，以消除 Decode 阶段的 Python 下发开销。排查精度异常或启动卡死时，先关闭图模式跑通 Eager 确认算子与权重无误，再按版本文档开启静态图捕获。
- **规范的冒烟与显存观测口径**：启动后按“非流式校验语义 → 流式校验逐 chunk 输出”两步冒烟，同时区分 `npu-smi info` 的设备整卡占用（`HBM-Usage`）与进程内的框架分配峰值（`torch.npu.max_memory_allocated()`）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 以 Llama-3.1-8B（FP16 权重约 16 GB，在 910B 64 GB 单卡上预留充裕 KV 空间）为例启动服务
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype float16 \
  --max-model-len 8192 \
  --max-num-seqs 32

# 流式冒烟：-N 禁用 curl 本地缓冲，temperature=0 锁定贪心解码便于跨芯片肉眼初检
curl -sS -N http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"Explain why the sky is blue.","max_tokens":64,"temperature":0,"stream":true}'
```

## 19.4 NVIDIA CUDA 与昇腾 Ascend 部署差异速查表

把第二篇在 NVIDIA GPU 上的部署经验与本章的昇腾流程并排对照，你会发现**模型与业务语义完全一致，差异全部收敛在模型之下的运行栈**。日常排障时，按这张表从上往下核对即可：

| 对比维度 | NVIDIA CUDA 部署路径 | 昇腾 Ascend 部署路径 | 工程排查优先级 |
|---|---|---|---|
| **设备管理与显存观测** | `nvidia-smi`、`torch.cuda.max_memory_allocated()` | `npu-smi info`（HBM-Usage）、`torch.npu.max_memory_allocated()` | 区分整卡设备占用与框架分配器峰值，不混用口径 |
| **核心环境变量** | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES`、`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`VLLM_PLUGINS` | 选卡失效查可见性变量；多卡/多进程卡死先核 `spawn` |
| **软件安装依赖链** | 宿主机驱动 + CUDA/cuDNN → PyTorch → vLLM | 宿主机 HDK + CANN/NNAL → PyTorch + TorchNPU → vLLM + `vllm-ascend` | 按兼容矩阵整组核验，容器不锁宿主机 HDK |
| **引擎平台加载** | 引擎内置树内（in-tree）平台，原生识别 CUDA | 树外平台（`PlatformEnum.OOT`），经 `vllm.platform_plugins` 注册激活 | 启动日志必须出现 `Platform plugin ascend is activated` |
| **图捕获与编译优化** | CUDA Graph（`--compilation-config` 控制捕获模式） | ACLGraph 与 NPUGraph Ex（经 `compilation/` 封装控制） | 启动卡捕获或输出漂移时，先回退 Eager 隔离变量 |
| **多卡集合通信** | NCCL（NVLink / PCIe / IB） | HCCL（HCCS / PCIe / RoCE），由 `NPUCommunicator` 接入 | 多卡建链失败或扩展性差时，查 HCCL 拓扑与网络配置 |
| **算子覆盖与兜底** | 内置 CUDA / Triton 融合算子 + `CustomOp` | CANN 内置 + `register_ascend_customop` / `_cann_ops_custom`，缺项退回 `forward_native` | 未命中 `op_registry_oot` 会静默走慢路径，需做覆盖审计 |
| **性能与精度调试工具** | `nsys`、`ncu`、PyTorch Profiler | `msprof`、`msprobe`、`torch_npu.profiler`、服务级 Profiling | 不同工具指标与采集方式各异，按阶段（Prefill/Decode）分采 |

归根结底，排查一次昇腾部署异常的黄金顺序只有五步：**版本咬合链 → 三层探针（含平台激活）→ 图模式隔离 → 算子覆盖回退 → 显存与并发调优**。前两步没站稳，后面所有的性能调参都只是空中楼阁。

## 本章小结

- **版本是一条咬合链**：HDK → CANN/NNAL → PyTorch/TorchNPU → vLLM/`vllm-ascend` 存在严格的成对 ABI 约束；容器镜像锁用户态不锁宿主机驱动，升级须整组核对兼容表。
- **三层探针精准定位**：按 `npu-smi info` → `torch.npu` 最小计算 → vLLM 日志激活平台插件的顺序自下而上排查，可瞬间定位绝大多数环境与加载故障。
- **插件接管整套芯片面**：`vllm-ascend` 通过 `vllm.platform_plugins` 注入 `NPUPlatform`，以平台级 Hook 接管 Worker、Attention、通信与图捕获，以 `CustomOp.register_oot` 将模型里的 `RMSNorm` 等算子透明路由到 `forward_oot`。

**延伸阅读**：[vllm-ascend 官方安装与快速上手文档](https://docs.vllm.ai/projects/ascend/en/latest/)、[vLLM 硬件插件架构设计博客（Introducing vLLM Hardware Plugin）](https://blog.vllm.ai/2025/05/12/hardware-plugin.html)、[昇腾社区 CANN 与 TorchNPU 兼容矩阵](https://www.hiascend.com/document)。
