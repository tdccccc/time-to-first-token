# 第 10 章 TensorRT-LLM 与三引擎架构横评 { .chapter-title }

<p class="chapter-summary">破除“TensorRT-LLM 是静态编译、vLLM 是纯 Eager”的过时标签，拆解当前 TensorRT-LLM PyTorch 后端的执行循环与三本预算账，并以八维架构横评与业务选型矩阵回答 vLLM V1、SGLang 与 TensorRT-LLM 究竟该怎么选。</p>

第 5 章和第 6 章分别拆解了 [vLLM](chapter-05.md) 的 PagedAttention 与连续批处理、以及 [SGLang](chapter-06.md) 的 RadixAttention 前缀树与重叠调度。在数据中心 NVIDIA GPU 生态中，还有第三根核心支柱——**TensorRT-LLM**。过去很长一段时间里，工程师对它贴着一个刻板标签：“性能猛，但要离线编译成静态 `.engine`，难调试、难换模型”。今天这套印象已经彻底过时。本章将先厘清现代推理引擎的架构趋同，再深入 TensorRT-LLM PyTorch 后端的执行循环、预算管理与核心特性，最后把三大引擎摆在同一张工程天平上做全景横评。

## 10.1 破除“编译式 vs Eager”迷思：从静态 Engine 到 PyTorch 后端

### 痛点：旧静态 Engine 流程为何走到了尽头？

早期 TensorRT-LLM 承袭了经典 TensorRT 的离线编译范式：用户必须先通过转换脚本把 Hugging Face 权重转成中间格式，再用 `trtllm-build` 指定 `min / opt / max` 形状区间与并行度，离线编译出与特定 GPU 架构（SM 版本）、CUDA 与 TensorRT 版本强绑定的二进制 `.engine` 制品，最后交给 C++ 运行时加载执行。

这套范式在模型架构相对稳定的时代确实能榨出极高的算子融合效率，但在 LLM 每月都在迭代新架构（GQA、MoE、MLA、混合线性注意力、原生多模态）的现实里，它暴露出三个致命摩擦：

- **构建与迁移沉重**：换一张卡（例如 A100 换 H100）、改一次并行度或扩一次 `max_seq_len`，整个 `.engine` 就要推倒重编；
- **新模型适配慢**：每接入一个新模型都要用 TensorRT Graph API 重写一遍静态网络定义，很难做到 Day-0 支持；
- **黑盒排障门槛高**：动态控制流（如复杂投机树、混合调度分支）极难表达，二次开发成本远高于 PyTorch 生态。

因此，TensorRT-LLM 官方已完成关键架构演进——**以原生 PyTorch 后端作为 `trtllm-serve` 的默认执行路径，并逐步淘汰旧 TensorRT Engine 流程**。新版直接读取标准 Hugging Face Checkpoint，用模块化 Python 代码定义模型结构，底层继续直接调用 NVIDIA 深度调优的 TensorRT-LLM / CUTLASS / FlashInfer / FlashAttention 自定义算子库，兼得 PyTorch 的开发敏捷性与硬件级算子性能。

### 现代引擎的殊途同归：加载期准备 + 运行期动态调度

与此同时，另一边的 vLLM 与 SGLang 也早已不是“纯 Python Eager 逐行发射 kernel”的模样：

- **vLLM V1** 默认在启动阶段运行 `torch.compile` 完成跨算子图融合与代码生成，并针对常见 batch 档位捕获**分段 CUDA Graph（Piecewise CUDA Graph）**，把动态长度的 Attention 留作轻量跳板、把前后线性层与归一化锁进零发射开销的 GPU 执行图；
- **SGLang** 同样在服务初始化阶段按预设 batch 阶梯预捕获 CUDA Graph，并结合 FlashInfer / Triton 融合核与重叠调度（Zero-Overhead Batch Scheduler）隐藏 CPU 侧开销；
- **TensorRT-LLM PyTorch 后端** 在加载期执行图优化融合、JIT / 预编译 kernel 选择与 CUDA Graph 捕获（含 `torch.compile` 与分段 Graph 路径），同时在未覆盖形状或特殊算子处保留无缝的 Eager 回退能力。

<figure class="book-figure">
  <img src="../../../assets/figures/trtllm-pipeline.svg" alt="TensorRT-LLM PyTorch 后端流程：上方为加载与预热阶段，经权重配置、图优化融合、kernel 选择含 JIT 与执行图捕获形成执行计划；下方为运行时五步循环，按三预算动态调度并复用执行计划" loading="lazy">
  <figcaption><span class="book-figure__label">图 10-1</span>先看上方“加载与预热”流水线——直接从 HF Checkpoint 出发，经图融合、kernel 选择与执行图捕获生成执行计划；再看下方“运行时循环”——每步按实时请求动态调度并复用已捕获的图。结论：今天三大引擎都在加载期做编译与图捕获、在运行期做动态调度，“编译式 vs Eager”的二分法已成历史。</figcaption>
</figure>

把旧静态 Engine 与现代 PyTorch 执行后端放在一起对照，工程差异一目了然：

| 对比维度 | 历史 TensorRT Engine 流程（已逐步淘汰） | 现代 PyTorch 执行后端（TensorRT-LLM / vLLM V1 / SGLang 共同范式） |
|---|---|---|
| **模型加载方式** | 离线转换权重 + `trtllm-build` 生成 `.engine` 制品 | 直接加载标准 Hugging Face 权重，自动初始化自定义 kernel 与 KV 池 |
| **算子融合与 kernel** | 离线构建期做 tactic 搜索、插件注入与静态融合 | 加载期结合手写 CUTLASS / FlashInfer / Triton 融合核、`torch.compile` 与 JIT |
| **CUDA Graph 捕获** | 绑定在静态执行引擎运行时配置中 | 启动预热期按 batch 阶梯捕获（或分段捕获），运行期自动匹配、padding 或回退 |
| **动态形状与批处理** | 严格受限于构建期声明的 `min/opt/max` 轮廓 | 调度器每步按真实到达请求动态组批，自由混排 Prefill 与 Decode |
| **缓存与变更成本** | 换卡、升版本、改并行度均需重新构建完整制品 | 权重即改即载；编译/JIT 缓存按模型与环境哈希自动复用于二次启动 |
| **新模型适配速度** | 慢（需静态图 API 逐层重构与验证） | 快（标准 `nn.Module` 模块化编写，极易对接社区前沿架构） |

!!! tip "核心直觉"
    评估三大引擎时，不要再从名字里猜“谁编译、谁解释执行”。今天真正拉开性能差距的是四件实打实的工程细节：**核心 GEMM/Attention 融合核的硬件压榨深度、CUDA Graph 的分段覆盖率、低精度（FP8/FP4/INT4）算子库成熟度，以及 CPU 调度循环与 GPU 计算的重叠效率**。

## 10.2 PyExecutor 循环与三本预算账

拆开 TensorRT-LLM PyTorch 后端的引擎盖，真正驱动推理吞吐的核心引擎是 **`PyExecutor`**。它的设计逻辑与第 5 章 vLLM V1 的步进执行引擎高度呼应，每一轮迭代都在做同一套五步闭环，由四个分工清晰的部件协同完成。

### 四大核心部件的五步执行循环

在 `PyExecutor` 的主循环中，每一次前向步进严格按以下五个阶段流转（对应图 10-1 下半框）：

1. **取请求（Fetch）**：从异步请求队列中拉取新到达的客户端请求，连同采样参数与停止条件一并注册到活跃池；
2. **调度（Schedule — `Scheduler` 拍板）**：综合当前运行中的生成序列（Generation）与等待中的预填充请求（Context），在三本硬预算约束下决定“这一步跑哪些序列、每条跑几个 token”；
3. **准备 KV（Prepare KV — `KVCacheManager` 记账）**：为新进入的序列分配分页物理 KV 块，尝试在前缀基数树中匹配可复用块，同时回收上一轮刚吐出 EOS 或达上限请求的物理块；
4. **执行（Execute — `ModelEngine` 干活）**：将本步所有被选中的 token（无论是 Prefill 片段还是 Decode 单字）打平拼接（Ragged / Packed Batch），匹配最接近的预捕获 CUDA Graph（或进入分段/Eager 路径）完成一次 GPU 前向；
5. **采样与推进（Sample — `Sampler` 收尾）**：对每条序列最后一个位置的 logits 执行温度缩放、top-p/top-k 过滤与采样，将新 token 写回序列状态，并判定是否满足终止条件。

这四个角色不是四段跨进程的异步流水，而是同处一个 tight loop 里的职责切分：调度器管算力与并发配额，KV 管理器管显存分页账本，模型引擎管 GPU 算子与执行图，采样器管概率抽取与生命周期收尾。

### 三本预算账的互相挤占关系

无论引擎内核写得多精巧，线上服务的显存安全、吞吐上限与尾延迟最终都落在三个核心预算参数的拉锯上：

- **`max_batch_size`（并发预算）**：同一时刻允许处于活跃状态的序列总数上界；
- **`max_seq_len`（单条长度预算）**：单条请求允许的最大总长度（Prompt + Completion）；
- **`max_num_tokens`（单步 token 算力与延迟预算）**：单次前向步进里，所有 Context（Prefill）token 与 Generation（Decode）token **共享的 token 总配额池**。

它们不是三本互不相干的独立账，而是彼此耦合、互相挤占。我们以 **Llama-3.1-8B（FP16，权重 16.06 GB，每 token KV 128 KiB）** 部署在单张 **A100 80GB**（按 90% 显存上限约 72 GB，扣除权重、激活工作区与 CUDA Graph 显存后，实际留给 KV 池的物理容量设为 **48 GiB**）为例，把三本账彻底算透：

#### 1. 显存账：`max_batch_size` × `max_seq_len` vs 物理 KV 池

48 GiB 的物理 KV 池总共能容纳：

$$\frac{48\text{ GiB}}{128\text{ KiB/token}} = 393,216\text{ tokens}$$

按每条 4,096 token 折算，物理上可装下 **96 条**满载序列的 KV。现在看参数如何互动：

- 若配置 `max_batch_size = 32`、`max_seq_len = 4096`，则活跃序列全满时的最坏 KV 需求仅 $32 \times 4096 \times 128\text{ KiB} = \mathbf{16\text{ GiB}}$。虽然显存池里还空着 32 GiB，但只要 `max_batch_size = 32` 锁死，调度器绝不会放行第 33 条活跃请求——多出来的 32 GiB 会自然留作历史前缀块缓存池（供跨请求复用），提升命中率却无法进一步推高并发吞吐；
- 反之，若贸然将 `max_batch_size` 调到 128、`max_seq_len = 8192`，最坏并发 KV 需求飙升至 $128 \times 8192 \times 128\text{ KiB} = \mathbf{128\text{ GiB}} \gg 48\text{ GiB}$。一旦线上同时涌入几十条长上下文请求，物理 KV 块迅速耗尽，调度器被迫频繁触发抢占（Preemption）与换出/重算，尾延迟骤然失控。

#### 2. 算力与延迟账：`max_num_tokens` 的零和博弈

设定单步总预算 `max_num_tokens = 8192`，新到达一批长度为 1,024 token 的 Prompt：

- **空载场景**：池中无运行中 Decode 序列，一步恰好可装入 $8192 \div 1024 = 8$ 条整块 Prefill，单步 GEMM 矩阵规模饱满，Tensor Core 算力利用率拉满；
- **混跑场景（已有 32 条序列正在 Decode）**：TensorRT-LLM 的调度器默认优先保障正在生成的请求（**Generation-First**），因为 Decode 请求停滞一步就会直接拉高用户的 ITL，而新到的 Prefill 多等一步仅小幅增加 TTFT。32 条 Decode 先划走 32 个名额，剩余 Prefill 可用槽位为：

$$8192 - 32 = 8160\text{ tokens}$$

这时是否开启分块预填充（Chunked Context）立刻显出分野：

- **关闭分块**：$\lfloor 8160 / 1024 \rfloor = 7$ 条（占用 7,168 token），白白浪费 992 个算力槽位，第 8 条 Prompt 被整条推迟到下一步；
- **开启分块**：先塞入 7 条完整 Prompt + 第 8 条的前 992 个 token，正好吃满 8,192 预算，第 8 条剩余的 32 个 token 留到下一步顺手收尾。

然而，`max_num_tokens` 也绝非越大越好！把它设到 16,384 或 32,768，固然能让单步吞下更多 Prefill 提高吞吐，但单步前向时间随之拉长——同批里搭顺风车的 32 条 Decode 请求，全都要陪着这个巨型 Prefill 步一起等待，ITL p99 立刻向上跳变。**`max_num_tokens` 的本质，正是 TTFT/吞吐 与 ITL 尾延迟之间的精确交换比**。

遇到性能或容量异常时，可以按下面这张诊断表迅速定位卡在哪本账：

| 线上典型症状 | 卡在哪本账 | 根因排查与调参方向 |
|---|---|---|
| GPU 算力与 KV 显存双低，但请求持续排队 | `max_batch_size` 过小 | 并发名额成为人为天花板；核算 KV 池地板后适当放大 `max_batch_size` |
| 长 Prompt 请求直接报错 400 或中途异常截断 | `max_seq_len` 不足 | 单条总长度（输入 + 输出）突破上限；结合显存与 RoPE 上限上调 |
| 并发不高，但 ITL p99 周期性出现几十毫秒毛刺 | `max_num_tokens` 过大 | 单步混入过多 Prefill token 拖慢整步；收紧 `max_num_tokens` 并确保开启 Chunked Context |
| 新请求 TTFT 长期居高不下，但正在生成的 ITL 很顺滑 | `max_num_tokens` 过小或 Decode 占满名额 | 检查 `max_num_tokens` 是否过紧导致每步留给 Prefill 的余量太少，或评估 PD 解耦 |
| 并发拉高后吞吐不升反降、伴随大量重算 | KV 池容量不足 | 活跃序列实际总长度超出物理 KV 池；缩减 `max_batch_size`、开启 FP8 KV Cache 或扩卡 |

## 10.3 执行路径细节：量化、In-flight Batching、KV 复用与 EAGLE-3

作为硬件原厂打造的推理引擎，TensorRT-LLM 的核心竞争力并不在于发明了另一套基础理论，而在于把前几章介绍的每一项关键优化，以极高密度的底层融合算子和工程接口落到了 NVIDIA 硬件上。

### 量化后端：按 GPU 架构代际走正确路径

量化在 TensorRT-LLM 中与 GPU 计算能力（Compute Capability）严格绑定（量化数学原理详见[第 8 章](chapter-08.md)，低精度 Matmul 算子拆解详见[第 15 章](chapter-15.md)）：

- **Hopper（H100，CC 9.0）与 Blackwell（B200，CC 10.0）的原生 FP8 / FP4 路线**：
  - H100 配备原生 FP8 Tensor Core（稠密峰值 1,979 TFLOPS，较 FP16 翻倍）。Llama-3.1-8B 从 FP16 转为 **FP8 W8A8** 后，权重从 16.06 GB 缩半至约 8.03 GB，单请求 Decode 权重搬运地板直接从 4.80 ms 砍到 2.40 ms；在计算受限的长 Prompt Prefill 阶段，还能借原生 FP8 GEMM 吞吐翻倍显著压低 TTFT；
  - Blackwell 架构进一步提供原生 **NVFP4（4-bit 块缩放浮点）** 硬件指令，在保持显著优于传统 INT4 的数值保真度的同时，把 8B 权重压缩至约 4 GB，并解锁更高的低精度算力峰值；
  - 权重获取既可直接加载预量化 Hugging Face Checkpoint（如 `nvidia/Llama-3.1-8B-Instruct-FP8`），也可使用 **NVIDIA Model Optimizer（ModelOpt）** 对私有微调权重做离线 PTQ 标定与导出。
- **Ampere（A100，CC 8.0）的务实路线**：
  - A100 不具备原生 FP8 W8A8 Tensor Core 计算单元（仅支持 FP8 KV 存储与软件反量化）。在 A100 上应避免走低效的软件模拟 FP8 计算，转而选择原生高吞吐的 **INT8 SmoothQuant（W8A8，624 TOPS）** 或 **W4A16 AWQ / GPTQ**（权重 4-bit 压至 ~4 GB，激活保持 FP16，经 CUTLASS / Marlin 融合反量化核直击 Decode 带宽墙）。

### In-flight Batching、基数树 KV 复用与 PD 传输

- **In-flight Batching（在线迭代级动态批处理）**：这是 TensorRT-LLM 对 Continuous Batching 与 Chunked Context 的统一实现名称。请求在任意生成步完成即立刻腾出 `max_batch_size` 槽位与 KV 物理块，新请求立刻切块插入下一步，消灭静态批处理的尾部“陪跑气泡”。
- **基于基数树的 KV 块复用（KV Block Reuse）**：TensorRT-LLM 的 KV 缓存管理器默认开启跨请求块复用（`kv_cache_config.enable_block_reuse: true`），底层采用基数树（Radix Tree）索引分页块，并支持部分块复用（`enable_partial_reuse`）与带优先级的 LRU 淘汰。工程上需牢记**块对齐边界**：若两段长 System Prompt 在第 1 个 token 就存在差异（如插入了实时时间戳），后续所有块都无法命中前缀哈希。
- **内置解耦传输层（`cache_transceiver_config`）**：面对跨实例 PD 解耦（将在[第 11 章](chapter-11.md)详述），TensorRT-LLM 原生集成基于 **NIXL（底层自动封装 NVLink / UCX / RDMA）** 的 KV 收发器，支持 `trtllm-serve disaggregated` 一键拉起 Context 与 Generation 编排服务，并可实现逐层计算与 KV 发送的流水线重叠。

### EAGLE-3 投机解码工程集成

在 PyTorch 后端中，TensorRT-LLM 已将投机解码收口为声明式配置，重点推荐特性完备、接受率最高的 **EAGLE-3**（以及面向原生支持模型的 MTP），而逐步弱化早期的 N-gram 路径。通过 `trtllm-serve` 的 YAML 配置文件，即可在一份声明里把三预算、KV 复用与 EAGLE-3 投机解码统一管起来：

```yaml
# trtllm-serve --extra_llm_api_options config.yaml 的典型生产配置结构
backend: pytorch
max_batch_size: 32
max_seq_len: 8192
max_num_tokens: 8192

kv_cache_config:
  enable_block_reuse: true
  enable_partial_reuse: true
  free_gpu_memory_fraction: 0.88

cuda_graph_config:
  enable_padding: true
  max_batch_size: 32

# 低并发延迟敏感场景开启 EAGLE-3；高满载吞吐集群建议关闭以避免算力争抢
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 4
  speculative_model_dir: yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
```

!!! warning "常见误区：EAGLE-3 上线前的三项硬核对"
    第一，Draft 权重必须与 Target 基座版本、词表与 Chat Template 严格匹配，错配版本会直接导致接受率趋近于零；第二，TensorRT-LLM 当前的投机解码配置作用于整实例级别，不支持逐请求动态关闭，高并发集群务必参考[第 9 章](chapter-09.md)的 QPS 交叉点原则，切勿在满载高吞吐实例上盲目常开；第三，压测对比三引擎前务必关闭或对齐 `enable_block_reuse`，防止跨轮次前缀命中污染 TTFT 基准。

## 10.4 三引擎深度架构与选型总表

把 vLLM V1、SGLang 与 TensorRT-LLM 并排放在一起，很容易陷入“社区跑分谁高了 5%”的短期口水战。引擎版本每两周都在迭代，单个 benchmark 数字转瞬即逝，但三者的**架构哲学、生态重心与工程取舍**却有清晰的长期脉络。

### 八维深度架构横评：vLLM V1 vs SGLang vs TensorRT-LLM

| 对比维度 | vLLM V1 | SGLang | TensorRT-LLM（PyTorch 后端） |
|---|---|---|---|
| **1. 架构演进与执行后端** | V1 彻底重构多进程引擎，解耦前端 API 与核心调度器，以 Python + PyTorch 为主轴，极度强调模块化与可扩展性 | 轻量级高并发 Python 引擎，深度围绕“语言/程序级调用 + 极致低 CPU 调度开销”设计，端到端架构紧凑 | 已从静态 `.engine` 全面转向原生 PyTorch 后端（`PyExecutor`），保留 NVIDIA 底层 C++/CUDA 高性能算子与执行计划内核 |
| **2. 调度机制与批处理** | 统一 Chunked Prefill 调度（不再割裂 P/D 两阶段逻辑），支持优先级与抢占恢复，强调大流量稳健性 | **Zero-Overhead 重叠调度器**将 CPU 调度与 GPU 前向异步重叠，单步调度开销极低，高并发小模型优势突出 | **In-flight Batching**（Generation 优先保护 ITL），严格围绕 `max_batch_size / max_seq_len / max_num_tokens` 三预算闭环控制 |
| **3. KV 缓存与前缀复用** | **PagedAttention** 分页块表 + 自动前缀缓存（APC，哈希块复用），支持多级 KV 卸载连接器 | **RadixAttention**（基数树全局管理所有会话 KV，含树状多分支共享与缓存感知调度，天然契合复杂 Agent 工作流） | 分页块管理器 + **基数树 KV 块复用**（支持部分块复用与优先级 LRU），默认开启跨请求共享 |
| **4. 编译与 CUDA Graph** | 默认启用 `torch.compile` 算子融合 + **分段 CUDA Graph**，在动态 Attention 间隙无缝拼合静态图 | 预捕获多档位 CUDA Graph，支持 `torch.compile` 与 FlashInfer 融合图路径，捕获速度快、冷启动轻盈 | 结合自定义融合核、JIT、`torch.compile` 与 CUDA Graph（含 padding 与分段模式），面向 NVIDIA 架构精细调优 |
| **5. 量化与算子生态** | 量化生态最全（FP8 / AWQ / GPTQ / Marlin / GGUF / BitsAndBytes），同时兼容 FlashAttention / FlashInfer / Triton | 深度集成 **FlashInfer** 与 Triton/CUTLASS，原生 FP8、DeepSeek MLA/FP8 块量化与 MoE Fused 算子迭代极快 | **NVIDIA 官方标杆**：Hopper FP8、Blackwell NVFP4、CUTLASS/cuBLASLt GEMM 与 TensorRT 专属融合核常年领先 |
| **6. 投机解码、结构化与 PD 解耦** | 全家桶支持 Draft/N-gram/Medusa/EAGLE/MTP；支持 XGrammar/Outlines；多款 KV Connector（NIXL/Mooncake/LMCache） | **结构化输出（压缩有限状态机 FSM / XGrammar 零开销跳转）业内标杆**；EAGLE-2/3 与 MTP 成熟；原生 Bootstrap + Mooncake/NIXL PD 解耦 | EAGLE-3 与 MTP 性能极致；集成引导解码；内置基于 **NIXL** 的 `cache_transceiver` 解耦编排与计算-传输重叠 |
| **7. 冷启动、硬件与模型支持广度** | **模型与硬件广度第一**：新开源模型常年 Day-0 支持，覆盖 NVIDIA / AMD ROCm / TPU / 昇腾（`vllm-ascend`）等异构芯片 | **前沿大模型（尤其 MoE/MLA 如 DeepSeek、Qwen3）跟进极快**，启动轻快；支持 NVIDIA / AMD / 部分国产 NPU 插件 | **仅限官方 NVIDIA GPU 矩阵**（Ampere 至 Blackwell）；主流旗舰模型覆盖稳健，但冷门/超新架构适配节奏慢于前两者 |
| **8. 典型工程定位** | 企业级通用推理底座与多模型中台“默认安全牌” | 高共享前缀 Agent、结构化输出、前沿 MoE/MLA 性能先锋 | 纯 NVIDIA 旗舰卡（H100/B200）榨取极限单卡延迟与低精度吞吐的“硬件专精派” |

!!! note "工程取舍：基准横评必须守住“四同一”铁律"
    做内部选型压测时，任何跨引擎对比都必须严格对齐四个前提：**同一硬件与驱动/CUDA 栈、同一权重文件/量化路径/Chat Template、同一精确输入输出 token 分布与采样参数、同一延迟分位数（p50/p95/p99）与稳态有效吞吐口径**。常见“伪对比”包括：用 FP8 的 TensorRT-LLM 对比 FP16 的 vLLM、未清空前缀缓存导致第二轮 TTFT 虚低、或者只比 `batch=1` 忽略高并发下的 p99 拐点。

### 按业务场景快速决策选型指南

回到真实业务架构评审，你可以直接按以下七类典型场景快速收口选型：

| 业务场景特征 | 首选推荐 | 备选方案 | 核心决策理由 |
|---|---|---|---|
| **多模型中台 / 频繁上线新开源模型 / 企业级标准平台** | **vLLM V1** | SGLang | 模型覆盖最广、Day-0 响应快、K8s `production-stack` 与 Prometheus 生态成熟，团队上手与排障成本最低 |
| **多轮 Agent、Tree-of-Thought、复杂工具调用、重度 JSON/Schema 结构化输出** | **SGLang** | vLLM V1 | RadixAttention 缓存感知调度 + 压缩 FSM 零开销结构化解码，高复用前缀与严格格式输出下综合吞吐优势最明显 |
| **纯 H100 / B200 旗舰集群、核心单一主力模型、压榨极限 FP8 / NVFP4 延迟与吞吐** | **TensorRT-LLM** | SGLang | NVIDIA 原厂针对 Hopper/Blackwell 深度打磨的 FP8/NVFP4 算子与执行计划，在大流量稳定模型上能把硬件压到极致 |
| **DeepSeek MLA / 超大稀疏 MoE（如 671B）大规模专家并行（EP）部署** | **SGLang** | TensorRT-LLM / vLLM V1 | SGLang 与 TensorRT-LLM 在 MLA 吸收、FP8 分块量化、DeepEP 通信重叠方面推进最快，需紧跟最新 release 核对 |
| **多云异构硬件（含 AMD GPU 或昇腾等国产 NPU 混合纳管）** | **vLLM V1** | SGLang | vLLM 通过统一插件接口（如第 19 章的 `vllm-ascend`）屏蔽多芯片差异，TensorRT-LLM 强绑定 NVIDIA 生态无法跨厂 |
| **超大规模跨节点 PD 解耦 + 分布式共享 KV 池集群** | **SGLang / vLLM V1**（配 Mooncake/NIXL） | TensorRT-LLM（配 NIXL） | 三者均已支持 NIXL/RDMA 解耦；若需集成跨机 DRAM/SSD 共享 KV 池，SGLang 与 vLLM 的社区连接器生态最丰富 |
| **极低并发交互式应用（代码补全、实时语音对话），强需求低 TPOT** | **TensorRT-LLM / SGLang**（开 EAGLE-3） | vLLM V1（开 EAGLE-3） | 优先对比三者在目标模型上 EAGLE-3 + CUDA Graph 的端到端 TPOT p95 与冷启动耗时 |

## 小结

现代 LLM 推理引擎已经走出“编译式 vs 解释执行”的早期分水岭，全面汇流到**加载期算子融合 + CUDA Graph 捕获 + 运行期迭代级动态调度**的统一架构范式。TensorRT-LLM 借 PyTorch 后端彻底卸下了静态编译的包袱，通过 `PyExecutor` 与三本预算账（并发、单条长度、单步 token 池）精确驾驭 GPU 资源；而在 vLLM V1、SGLang 与 TensorRT-LLM 之间做技术选型，核心永远是“先过硬件与模型兼容红线，再按前缀复用、结构化输出与低精度硬件特化匹配场景，最后以同口径 SLO 下的有效吞吐（Goodput）一锤定音”。

单机引擎的潜力挖尽之后，面对更大规模的并发与更苛刻的 TTFT / ITL 双重 SLA，我们必须把视野拉升到集群与生产架构层。[下一章](chapter-11.md)我们将系统展开 **PD 解耦、Kubernetes 弹性扩缩容、单位 Token 经济学与多后端成本路由**。
