#!/usr/bin/env python3
"""Prepare public sources and assemble the book site for MkDocs.

With the one-file-per-chapter architecture:
  docs/index.md             -> site-src/index.md
  docs/chapters/chapter-01..23.md -> annotated with hover glossary links
  docs/topics/*.md          -> annotated with hover glossary links
  docs/toolkit/*.md         -> staged directly
  docs/parts/part-1..5.md   -> generated with chapter summaries and section counts
  docs/glossary.md          -> generated from TERM_GLOSSARY
  docs/progress.md          -> generated for the 23 main chapters
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site-src"

PART_TITLES = {
    1: "性能直觉与度量方法",
    2: "推理引擎与部署优化",
    3: "GPU 算子优化与引擎接入",
    4: "国产芯片适配原理与方法",
    5: "综合分析与方法收口",
}
PART_CN = {1: "第一篇", 2: "第二篇", 3: "第三篇", 4: "第四篇", 5: "第五篇"}

PART_LEDES = {
    1: "先把“为什么慢”和“怎么量”想透：Roofline 三区间、模型显存与算力账本、延迟分布与 Profiling 诊断。",
    2: "打开主流推理引擎与生产架构：vLLM、SGLang、监控与容量评估、量化、投机解码、TensorRT-LLM 横评与 PD/路由。",
    3: "下潜到硬件与内核：GPU 执行模型与四版 GEMM、Triton、FlashAttention、量化 Matmul、CUTLASS 与自定义算子接入。",
    4: "跳出 CUDA 生态，以华为昇腾为主线系统掌握国产芯片软件栈、插件机制、CANN/AscendC、覆盖归因与算子迁移。",
    5: "用配对精度回归、五步差异归因、完整跨芯片分析范例与系统论文读法完成全书方法论收口。",
}

PART_INTROS = {
    1: """本篇为全书建立统一的物理直觉与测量标尺：第 1 章用 Roofline 分清计算、带宽与启动开销三区间；第 2 章算清参数、KV 缓存、MoE 与线性注意力的显存和算力账；第 3 章把“快”拆成 TTFT、ITL、吞吐、排队与长尾分布；第 4 章用 Profiling 双泳道把时间落到具体算子与空白。

全书默认以 **NVIDIA A100 80GB / H100** 与 **Llama-3.1-8B / Qwen3-8B** 作为主线教学算例，完整规格随时可查[硬件规格与模型账本速查](../toolkit/hardware-models.md)。""",
    2: """本篇拆解现代 LLM 推理引擎如何组织显存、调度和计算，并梳理生产环境的核心优化杠杆。

第 5–6 章沿请求旅程精讲 vLLM（分页与连续批处理）与 SGLang（基数树前缀复用）；第 7 章串联监控指标与容量拐点；第 8–9 章深入量化、投机解码与 KV 淘汰的收益边界；第 10 章横向对比 vLLM、SGLang 与 TensorRT-LLM；第 11 章落地 PD 解耦、队列扩缩容与成本路由。大规模集群 PD 详见[专题 B](../topics/cluster-pd-mooncake.md)。""",
    3: """本篇打开推理引擎内部的算子黑盒：第 12 章从 GPU 执行模型切入四版 GEMM 优化阶梯；第 13 章建立 Triton Block 编程心智并写透最小 2D GEMM；第 14–15 章手推并拆解 Fused Attention 与量化 Matmul；第 16 章导读 FlashAttention 2/3 与 CUTLASS/CuTe 工业实现；第 17 章讲清自定义算子接入 vLLM 的注册、启用与分发全链路。""",
    4: """本篇以华为昇腾 Ascend 为主线，按“先画地图 → 再看部署与插件 → 再钻 CANN/AscendC 底层 → 查覆盖与回退 → 做跨芯片迁移”的顺读顺序推进。

第 18 章建立四面墙 × 五层软件栈坐标系；第 19 章拆解昇腾环境链与 `vllm-ascend` 插件架构；第 20 章走完 AscendC 硬件流水与算子四环；第 21 章讲透算子覆盖率与四类回退定位；第 22 章系统梳理从 CUDA/Triton 到 Ascend 的六类语义差异。其他国产平台生态见[专题 E](../topics/domestic-ecosystems.md)。""",
    5: """本篇将全书的度量、引擎、算子与国产芯片方法合为一体：围绕“同一模型在不同芯片上差异从何而来”，精讲配对精度回归、五步归因闭环与一篇完整的跨芯片对比分析范例，最后用系统论文三遍阅读法为持续演进收尾。""",
}

# (chapter number, part number, title)
CHAPTERS: list[tuple[int, int, str]] = [
    (1, 1, "推理为什么慢：Roofline 与三区间直觉"),
    (2, 1, "模型的显存与计算账本"),
    (3, 1, "推理度量与压测方法"),
    (4, 1, "Profiling 方法论：看时间花在哪"),
    (5, 2, "vLLM：一条请求如何穿过推理引擎"),
    (6, 2, "SGLang：RadixAttention 与前缀复用"),
    (7, 2, "可观测性与容量评估"),
    (8, 2, "量化原理与取舍"),
    (9, 2, "投机解码与长上下文 KV 淘汰"),
    (10, 2, "TensorRT-LLM 与三引擎架构横评"),
    (11, 2, "生产架构：PD 解耦、扩缩容与成本路由"),
    (12, 3, "GPU 执行模型与 GEMM 四版优化阶梯"),
    (13, 3, "Triton 编程模型：从向量加到 2D GEMM"),
    (14, 3, "Fused Attention 深讲：Online Softmax 与分块实现"),
    (15, 3, "量化 Matmul 算子原理：W8A8 与 W8A16"),
    (16, 3, "工业级实现导读：FlashAttention 2/3 与 CUTLASS"),
    (17, 3, "把自定义算子接入推理引擎"),
    (18, 4, "国产芯片全景与软件栈地图"),
    (19, 4, "昇腾部署架构与 vllm-ascend 插件机制"),
    (20, 4, "CANN 分层与 AscendC 算子全链路"),
    (21, 4, "算子覆盖分析与回退定位方法"),
    (22, 4, "跨芯片算子迁移：从 CUDA / Triton 到 Ascend"),
    (23, 5, "跨芯片精度回归、性能归因与分析报告范式"),
]

CHAPTER_SUMMARIES: dict[int, str] = {
    1: "用算术强度、硬件脊点与计算/带宽/启动三区间，判断一次推理或融合算子究竟卡在哪。",
    2: "拆解权重、KV 缓存（MHA/GQA/MLA）、MoE 稀疏激活与混合线性注意力的显存与计算账。",
    3: "把“快”拆成 TTFT、ITL/TPOT、吞吐与排队，掌握分位数分布、SSE 计时原理与拐点读法。",
    4: "按 torch.profiler → nsys → ncu 三层粒度读懂 CPU/GPU 双泳道，定位热点与时间线空白。",
    5: "沿一条请求的完整旅程，拆解 vLLM V1 双进程架构、PagedAttention、连续批处理与调度抢占。",
    6: "从整段哈希到 Block 哈希再到基数树，走完 RadixAttention 前缀复用、调度生命周期与分块 Prefill。",
    7: "先定义指标再看看板：掌握 Prometheus 三类指标、速率扫描曲线、Little 定律与过载抢占边界。",
    8: "看清低比特如何省搬运与提算力，用微型算例拆解离群值、SmoothQuant、AWQ/GPTQ 与硬件边界。",
    9: "先建立草稿并行验证直觉与无损拒绝采样原理，再看清 QPS 交叉点与 Attention Sink 淘汰机制。",
    10: "厘清 TensorRT-LLM PyExecutor 与三预算账，用一张完整架构总表横评 vLLM、SGLang 与 TRT-LLM。",
    11: "算清 PD 解耦传输税与 K8s 排队深度扩缩容逻辑，建立单位 Token 经济学与多后端预算路由。",
    12: "从 Thread/Warp/SM 存储层级出发，走完 GEMM 朴素、Shared 分块、向量化防 Bank 冲突与线程粗化四版阶梯。",
    13: "建立 Triton Block 级编程思维，打通 1D Mask、Fused Softmax 与最小 2D GEMM 广播指针。",
    14: "推导 Online Softmax 精确更新与代码化简，算清 SRAM/寄存器上限，拆解完整实现与 Causal 跳块。",
    15: "看懂 Scale 粒度如何决定 Kernel 结构，对比 W8A8 与 W8A16 Fused Dequant 双路径及 L2 Swizzle。",
    16: "图解 FlashAttention 2 的 Split-Q 与 cp.async 流水、FA3 Hopper TMA/Warp Specialization 与 CUTLASS/CuTe。",
    17: "走通 vLLM CustomOp 注册、启用、分发三道门，规避静默失效，并用 Amdahl 归因与扩展地图收口。",
    18: "以四面墙与五层软件栈建立国产芯片坐标系，对比平台插件与 FlagGems 统一算子层，附主流平台实填表。",
    19: "理顺昇腾版本咬合链、驱动→框架→插件三层诊断模型与 vllm-ascend 模块架构，对比 CUDA 关键差异。",
    20: "打通 CANN 五层映射、TBE/AscendC 抽象、MTE2/VEC/MTE3 双缓冲流水线，沿 RMSNorm 走完算子四环。",
    21: "区分 CPU 回退、图外执行、精度降级与静默回退，用三口径覆盖率、四象限法则与示范 Profiler 清单定位瓶颈。",
    22: "对照 GPU 与 Ascend 存储流水架构，逐条拆解六类 CUDA 假设失效点与小引擎/Triton 迁移思路。",
    23: "掌握配对精度检验、五步归因闭环、一篇完整的跨芯片对比分析范例，以及系统论文三遍阅读法。",
}

TERM_GLOSSARY = [
    {"id": "roofline", "aliases": ("Roofline", "Roofline 模型"), "en": "Roofline model", "zh": "屋顶线模型", "desc": "用峰值算力、峰值显存带宽和算术强度估算 kernel 性能上限的模型"},
    {"id": "arithmetic_intensity", "aliases": ("算术强度", "arithmetic intensity"), "en": "arithmetic intensity", "zh": "算术强度", "desc": "每搬运 1 字节数据所执行的浮点运算次数（FLOP/Byte）"},
    {"id": "ridge_point", "aliases": ("脊点", "ridge point"), "en": "ridge point", "zh": "脊点", "desc": "硬件峰值算力与峰值显存带宽之比，划分带宽受限区与计算受限区"},
    {"id": "memory_bound", "aliases": ("显存带宽受限", "memory-bound", "memory bound"), "en": "memory-bound", "zh": "显存带宽受限", "desc": "运行时间主要受显存数据搬运速度限制的状态"},
    {"id": "compute_bound", "aliases": ("计算受限", "compute-bound", "compute bound"), "en": "compute-bound", "zh": "计算受限", "desc": "运行时间主要受算力单元吞吐限制的状态"},
    {"id": "ttft", "aliases": ("TTFT",), "en": "Time to First Token", "zh": "首 token 延迟", "desc": "从发出请求到收到首个非空生成内容的时间，包含排队与 prefill"},
    {"id": "itl", "aliases": ("ITL",), "en": "Inter-Token Latency", "zh": "token 间延迟", "desc": "同一请求相邻两个输出 token 之间的时间间隔，决定流式输出是否卡顿"},
    {"id": "tpot", "aliases": ("TPOT",), "en": "Time Per Output Token", "zh": "每输出 token 平均时间", "desc": "解码阶段总跨度除以输出 token 数减一，衡量平均生成速度"},
    {"id": "prefill", "aliases": ("prefill", "Prefill", "预填充"), "en": "prefill", "zh": "预填充", "desc": "并行处理输入 prompt 并写入初始 KV 缓存的前向阶段"},
    {"id": "decode", "aliases": ("decode", "Decode", "解码阶段"), "en": "autoregressive decoding", "zh": "解码", "desc": "逐 token 自回归生成输出、并每步复用历史 KV 缓存的阶段"},
    {"id": "kv_cache", "aliases": ("KV 缓存", "KV Cache", "KV cache"), "en": "Key-Value Cache", "zh": "键值缓存", "desc": "保存历史 token 注意力 K/V 向量、避免 decode 阶段重复计算的显存区域"},
    {"id": "mha", "aliases": ("MHA",), "en": "Multi-Head Attention", "zh": "多头注意力", "desc": "每个 query 头拥有独立 K/V 头的标准注意力结构"},
    {"id": "gqa", "aliases": ("GQA",), "en": "Grouped-Query Attention", "zh": "分组查询注意力", "desc": "多个 query 头共享一组 K/V 头，以成倍降低 KV 显存占用与访存量"},
    {"id": "mla", "aliases": ("MLA",), "en": "Multi-head Latent Attention", "zh": "多头潜变量注意力", "desc": "将 K/V 压缩到低维潜变量缓存，并利用矩阵结合律在查询侧吸收上投影的注意力结构"},
    {"id": "moe", "aliases": ("MoE",), "en": "Mixture of Experts", "zh": "混合专家", "desc": "每个 token 经路由仅激活少数专家子网络，实现总参数容量与单步计算量脱钩"},
    {"id": "paged_attention", "aliases": ("PagedAttention", "Paged Attention"), "en": "Paged Attention", "zh": "分页注意力", "desc": "将 KV 缓存切为固定大小 block，通过块表映射到非连续物理显存的注意力管理机制"},
    {"id": "block_table", "aliases": ("block table",), "en": "block table", "zh": "块表", "desc": "记录序列逻辑 KV block 到物理显存 block 映射的每请求索引数组"},
    {"id": "continuous_batching", "aliases": ("连续批处理", "continuous batching", "Continuous Batching"), "en": "continuous batching", "zh": "连续批处理", "desc": "以单次迭代为粒度动态组批，已完成请求立即退出、等待请求即时补入的调度机制"},
    {"id": "chunked_prefill", "aliases": ("chunked prefill", "Chunked Prefill", "分块 prefill", "分块 Prefill"), "en": "chunked prefill", "zh": "分块预填充", "desc": "将长 prompt 的 prefill 切为定长小块，与 decode 请求混批以消除解码卡顿"},
    {"id": "prefix_caching", "aliases": ("prefix caching", "Prefix Caching", "前缀缓存"), "en": "prefix caching", "zh": "前缀缓存", "desc": "跨请求复用相同 prompt 前缀已计算出的 KV 缓存，跳过重复 prefill"},
    {"id": "radix_attention", "aliases": ("RadixAttention", "Radix Attention"), "en": "Radix Attention", "zh": "基数树注意力", "desc": "用压缩前缀树（Radix Tree）按 token 粒度管理、共享与 LRU 淘汰 KV 缓存的机制"},
    {"id": "quantization", "aliases": ("量化", "quantization"), "en": "quantization", "zh": "量化", "desc": "用更低位宽表示权重或激活，以降低显存占用与带宽搬运量", "exclude_prefixes": ("向", "预", "标")},
    {"id": "awq", "aliases": ("AWQ",), "en": "Activation-aware Weight Quantization", "zh": "激活感知权重量化", "desc": "依据激活幅值分布保护关键权重通道的低比特权重量化方法"},
    {"id": "gptq", "aliases": ("GPTQ",), "en": "Generative Pre-trained Transformer Quantization", "zh": "GPT 权重量化", "desc": "基于近似二阶海森信息逐列补偿量化误差的训练后权重量化方法"},
    {"id": "smoothquant", "aliases": ("SmoothQuant",), "en": "SmoothQuant", "zh": "SmoothQuant 平滑量化", "desc": "通过等价通道缩放将激活离群值转移到权重侧，使 W8A8 量化更平滑的方法"},
    {"id": "fp8", "aliases": ("FP8",), "en": "8-bit Floating Point", "zh": "8 位浮点", "desc": "使用 8 位浮点表示权重或激活的低精度格式，Hopper 及更新架构提供原生 Tensor Core 加速"},
    {"id": "int4", "aliases": ("INT4",), "en": "4-bit Integer", "zh": "4 位整数", "desc": "每权重仅占 4 bit 的极低位宽格式，常以 W4A16 形式大幅缓解显存与带宽压力"},
    {"id": "speculative_decoding", "aliases": ("投机解码", "speculative decoding"), "en": "speculative decoding", "zh": "投机解码", "desc": "先由低成本草稿模型预测多步候选 token，再由目标大模型单次并行验证的无损加速方法"},
    {"id": "kv_eviction", "aliases": ("KV 淘汰", "KV eviction", "KV cache eviction"), "en": "KV cache eviction", "zh": "KV 缓存淘汰", "desc": "按策略丢弃部分历史 KV（如保留 Attention Sink 与滑窗），以控制超长上下文显存"},
    {"id": "pd_disaggregation", "aliases": ("PD 解耦", "prefill-decode disaggregation", "解耦式服务"), "en": "prefill-decode disaggregation", "zh": "预填充-解码解耦", "desc": "将算力密集的 Prefill 与带宽密集的 Decode 拆分到独立计算池，消除相互干扰"},
    {"id": "goodput", "aliases": ("goodput", "有效吞吐"), "en": "goodput", "zh": "有效吞吐", "desc": "同时满足延迟（TTFT/ITL）SLO 与正确性约束的有效请求吞吐"},
    {"id": "hpa", "aliases": ("HPA",), "en": "Horizontal Pod Autoscaler", "zh": "水平 Pod 自动扩缩容", "desc": "Kubernetes 中根据队列或业务指标自动增减副本数的控制器"},
    {"id": "keda", "aliases": ("KEDA",), "en": "Kubernetes Event-driven Autoscaling", "zh": "Kubernetes 事件驱动扩缩容", "desc": "将 Prometheus 队列深度等外部指标接入 Kubernetes 自动扩缩容的组件"},
    {"id": "tensorrt_llm", "aliases": ("TensorRT-LLM",), "en": "TensorRT-LLM", "zh": "TensorRT-LLM 推理引擎", "desc": "NVIDIA 面向数据中心 GPU 的高性能 LLM 推理系统，当前默认采用 PyTorch 执行后端"},
    {"id": "cuda", "aliases": ("CUDA",), "en": "Compute Unified Device Architecture", "zh": "统一计算设备架构", "desc": "NVIDIA 的 GPU 并行计算平台与编程模型"},
    {"id": "cuda_graph", "aliases": ("CUDA Graph", "CUDA 图"), "en": "CUDA Graph", "zh": "CUDA 图", "desc": "将一连串 GPU kernel 启动录制成静态图单次重放，显著降低 CPU 发射开销"},
    {"id": "triton", "aliases": ("Triton",), "en": "Triton", "zh": "Triton GPU 编程语言", "desc": "以 Block 级张量抽象用 Python 编写高性能 GPU kernel 的语言与编译器"},
    {"id": "shared_memory", "aliases": ("shared memory", "Shared Memory", "共享内存"), "en": "shared memory", "zh": "共享内存", "desc": "同一线程块（SM）内可见、带宽极高、延迟仅数十周期的可编程片上存储"},
    {"id": "warp", "aliases": ("warp", "Warp"), "en": "warp", "zh": "线程束", "desc": "NVIDIA GPU 中以 32 个线程为一组按 SIMT 方式同步调度的最小硬件执行单元"},
    {"id": "occupancy", "aliases": ("occupancy", "Occupancy", "占用率"), "en": "occupancy", "zh": "占用率", "desc": "SM 上实际驻留活跃 warp 数占硬件最大支持 warp 数的比例"},
    {"id": "bank_conflict", "aliases": ("bank conflict", "Bank Conflict", "存储体冲突"), "en": "bank conflict", "zh": "存储体冲突", "desc": "同一 warp 内多个线程同时访问同一 shared memory bank 的不同地址而导致的串行化"},
    {"id": "gemm", "aliases": ("GEMM",), "en": "General Matrix Multiplication", "zh": "通用矩阵乘法", "desc": "Transformer 投影层、MLP 与输出头的核心基础计算算子"},
    {"id": "flash_attention", "aliases": ("FlashAttention",), "en": "FlashAttention", "zh": "FlashAttention", "desc": "通过片上分块与在线 Softmax 消除 N×N 中间矩阵 HBM 读写的 IO 感知精确注意力算法"},
    {"id": "online_softmax", "aliases": ("online softmax", "Online Softmax", "在线 softmax"), "en": "online softmax", "zh": "在线 Softmax", "desc": "分块扫描时动态维护运行最大值与归一化分母、与整行计算精确等价的数值稳定算法"},
    {"id": "cutlass", "aliases": ("CUTLASS",), "en": "CUDA Templates for Linear Algebra Subroutines", "zh": "CUTLASS 模板库", "desc": "NVIDIA 开源的可组合高性能 GEMM 与流水线 C++/CuTe 模板库"},
    {"id": "cann", "aliases": ("CANN",), "en": "Compute Architecture for Neural Networks", "zh": "昇腾神经网络计算架构", "desc": "华为昇腾 NPU 的分层异构计算软件栈与算子开发平台"},
    {"id": "ascendc", "aliases": ("AscendC", "Ascend C"), "en": "Ascend C", "zh": "AscendC 算子编程语言", "desc": "面向华为昇腾 NPU 的 C++ 自定义算子编程接口，显式管理片上 UB 与硬件流水"},
    {"id": "npu", "aliases": ("NPU",), "en": "Neural Processing Unit", "zh": "神经网络处理器", "desc": "专为矩阵与向量张量计算设计的专用人工智能加速处理器"},
    {"id": "vllm_ascend", "aliases": ("vllm-ascend",), "en": "vLLM Ascend plugin", "zh": "vLLM 昇腾插件", "desc": "将昇腾 NPU 平台、算子库与通信后端无缝挂载到 vLLM 主干的官方硬件插件"},
    {"id": "operator_coverage", "aliases": ("算子覆盖", "operator coverage"), "en": "operator coverage", "zh": "算子覆盖", "desc": "模型推理所需算子在目标硬件上由原生或专用实现承接的比例与清单"},
    {"id": "tensor_parallelism", "aliases": ("张量并行", "tensor parallelism"), "en": "tensor parallelism", "zh": "张量并行", "desc": "将单层权重矩阵切分到多张加速卡上并行计算并通过集合通信汇总的并行策略"},
    {"id": "mfu", "aliases": ("MFU",), "en": "Model FLOPs Utilization", "zh": "模型算力利用率", "desc": "实际有效模型计算吞吐占硬件理论峰值算力的比例"},
    {"id": "slo", "aliases": ("SLO",), "en": "Service Level Objective", "zh": "服务等级目标", "desc": "对服务延迟分位数（如 TTFT/ITL p95）与可用性设定的可度量目标"},
    {"id": "qps", "aliases": ("QPS",), "en": "Queries Per Second", "zh": "每秒请求数", "desc": "系统每秒到达或完成的请求数量"},
    {"id": "ncu", "aliases": ("ncu",), "en": "NVIDIA Nsight Compute", "zh": "Nsight Compute", "desc": "采集单个 GPU kernel 硬件计数器、指令与访存瓶颈的微观性能分析器"},
    {"id": "nsys", "aliases": ("nsys",), "en": "NVIDIA Nsight Systems", "zh": "Nsight Systems", "desc": "采集 CPU、GPU、内存拷贝与跨进程同步全景时间线的系统级分析器"},
    {"id": "tbe", "aliases": ("TBE",), "en": "Tensor Boost Engine", "zh": "张量加速引擎", "desc": "昇腾 CANN 上一代基于 Python DSL 描述计算与自动调度的算子开发框架"},
    {"id": "rope", "aliases": ("RoPE",), "en": "Rotary Position Embedding", "zh": "旋转位置编码", "desc": "通过旋转 query/key 向量注入相对位置信息的经典位置编码方法"},
]

FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
CONFLICT_RE = re.compile(r"^(?:<{7}|>{7})(?:\s|$)")

INLINE_PROTECT_RE = re.compile(
    r"(`[^`]*`|\$\$[^$]*\$\$|\$(?!\s)[^$\n]+?(?<!\s)\$|!?\[[^\]]*\]\([^)]*\)|<[^>]+>|https?://\S+)"
)

NON_DEFINITION_PREFIXES = (
    "如果", "若", "例如", "比如", "见", "可", "当", "在", "不", "只",
    "按", "默认", "下文", "本节", "第", "以及", "或", "并", "但",
)


@dataclass
class Chapter:
    num: int
    part: int
    title: str
    sections: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return f"chapter-{self.num:02d}"


@dataclass
class Part:
    num: int
    title: str
    chapters: list[Chapter] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return f"part-{self.num}"


def write_staged(rel: str, text: str) -> Path:
    path = SITE / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def protect_inline(text: str) -> tuple[str, list[str]]:
    store: list[str] = []

    def keep(match: re.Match[str]) -> str:
        store.append(match.group(0))
        return f"\x00{len(store) - 1}\x00"

    return INLINE_PROTECT_RE.sub(keep, text), store


def restore_inline(text: str, store: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], text)


def inside_parentheses(text: str, pos: int) -> bool:
    depth = 0
    for ch in text[:pos]:
        if ch in "（(":
            depth += 1
        elif ch in "）)":
            depth = max(0, depth - 1)
    return depth > 0


def looks_like_definition(text: str, end: int) -> bool:
    match = re.match(r"\s*[（(]([^（）()]{1,180})[）)]", text[end:])
    if match is None:
        return False
    content = match.group(1).strip()
    if not content or content.startswith(NON_DEFINITION_PREFIXES):
        return False
    if re.search(r"[A-Za-z]", content) or re.search(r"[，,：:]", content):
        return True
    return len(content) >= 3


def excluded_match(text: str, match: re.Match[str], entry: dict[str, object]) -> bool:
    for prefix in entry.get("exclude_prefixes", ()):  # type: ignore[union-attr]
        if text[: match.start()].endswith(prefix):
            return True
    for suffix in entry.get("exclude_suffixes", ()):  # type: ignore[union-attr]
        if text[match.end():].startswith(suffix):
            return True
    return False


def annotatable_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(("#", "|", "<", ">", "---", "```", "~~~", "$$", "!!!", "???")):
        return False
    if line.startswith(("    ", "\t")):
        return False
    return True


def glossary_annotation(term: str, entry: dict[str, str], glossary_rel: str = "../glossary.md") -> str:
    en = entry["en"]
    zh = entry["zh"]
    desc = entry["desc"]
    key = term.lower().replace("-", " ").strip()
    en_key = en.lower().replace("-", " ").strip()
    zh_key = zh.lower().strip()
    if en_key == key:
        inside = f"{zh}：{desc}" if zh_key != key else desc
    elif zh_key == key:
        inside = f"{en}：{desc}"
    else:
        inside = f"{en}，{zh}：{desc}"
    tooltip = html.escape(inside, quote=True)
    return f'[{term}]({glossary_rel}#{entry["id"]}){{ .term-ref title="{tooltip}" }}'


class TermAnnotator:
    """Annotate the first prose occurrence of each known term in a document once."""

    def __init__(self, glossary_rel: str = "../glossary.md") -> None:
        self.glossary_rel = glossary_rel
        self.seen: set[str] = set()
        self.entries: list[tuple[dict[str, str], list[re.Pattern[str]]]] = []
        for entry in TERM_GLOSSARY:
            patterns = []
            for alias in dict.fromkeys(entry["aliases"]):
                if (
                    not re.search(r"[A-Za-z]", alias)
                    and len(alias) < 3
                    and not entry.get("exclude_prefixes")
                    and not entry.get("exclude_suffixes")
                ):
                    continue
                escaped = re.escape(alias)
                if re.search(r"[A-Za-z]", alias):
                    escaped = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
                patterns.append(re.compile(escaped, re.IGNORECASE))
            self.entries.append((entry, patterns))

    def annotate_line(self, line: str) -> str:
        protected, store = protect_inline(line)
        defined_here: set[str] = set()
        for entry, patterns in self.entries:
            if entry["id"] in self.seen:
                continue
            for pattern in patterns:
                for match in pattern.finditer(protected):
                    if excluded_match(protected, match, entry):
                        continue
                    if looks_like_definition(protected, match.end()):
                        defined_here.add(entry["id"])
                        break
                if entry["id"] in defined_here:
                    break
        while True:
            best: tuple[int, int, dict[str, str], re.Match[str]] | None = None
            for entry, patterns in self.entries:
                if entry["id"] in self.seen:
                    continue
                for pattern in patterns:
                    for match in pattern.finditer(protected):
                        if excluded_match(protected, match, entry):
                            continue
                        candidate = (match.start(), -(match.end() - match.start()), entry, match)
                        if best is None or candidate[:2] < best[:2]:
                            best = candidate
                        break
            if best is None:
                break
            _, _, entry, match = best
            self.seen.add(entry["id"])
            if (
                entry["id"] in defined_here
                or inside_parentheses(protected, match.start())
                or looks_like_definition(protected, match.end())
            ):
                continue
            replacement = glossary_annotation(match.group(0), entry, self.glossary_rel)
            store.append(replacement)
            end = match.end()
            if protected[end : end + 1] == " ":
                end += 1
            protected = protected[: match.start()] + f"\x00{len(store) - 1}\x00" + protected[end:]
        return restore_inline(protected, store)

    def annotate_lines(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        fence: str | None = None
        in_math = False
        html_depth = 0
        for line in lines:
            stripped = line.strip()
            fence_match = re.match(r"^(```+|~~~+)", stripped)
            if fence is not None:
                if fence_match and fence_match.group(1)[0] == fence:
                    fence = None
                out.append(line)
                continue
            if fence_match:
                fence = fence_match.group(1)[0]
                out.append(line)
                continue
            if stripped == "$$":
                in_math = not in_math
                out.append(line)
                continue
            if in_math:
                out.append(line)
                continue
            if html_depth > 0:
                opens = len(re.findall(r"<(?:div|figure|ol|section|table)\b", line, re.IGNORECASE))
                closes = len(re.findall(r"</(?:div|figure|ol|section|table)>", line, re.IGNORECASE))
                html_depth = max(0, html_depth + opens - closes)
                out.append(line)
                continue
            if re.match(r"^\s*<(?:div|figure|ol|section|table)\b", line, re.IGNORECASE):
                opens = len(re.findall(r"<(?:div|figure|ol|section|table)\b", line, re.IGNORECASE))
                closes = len(re.findall(r"</(?:div|figure|ol|section|table)>", line, re.IGNORECASE))
                html_depth = max(0, opens - closes)
                out.append(line)
                continue
            if not annotatable_line(line):
                out.append(line)
                continue
            out.append(self.annotate_line(line))
        return out


def validate_markdown(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    problems: list[str] = []
    fence: str | None = None
    for i, line in enumerate(lines, 1):
        if CONFLICT_RE.match(line):
            problems.append(f"{path}:{i}: unresolved merge conflict marker")
        m = FENCE_RE.match(line)
        if m:
            if fence is None:
                fence = m.group(1)[0]
                if line.strip() == "```":
                    problems.append(f"{path}:{i}: code fence without language")
            elif fence == m.group(1)[0]:
                fence = None
            continue
        if fence is not None:
            continue
        if line.lstrip().startswith("|") and (i == 1 or not lines[i - 2].lstrip().startswith("|")):
            if i >= len(lines) or not re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i]):
                problems.append(f"{path}:{i}: table without separator row")
    if fence is not None:
        problems.append(f"{path}: unclosed code fence")
    return problems


def extract_h2_sections(lines: list[str]) -> list[str]:
    sections: list[str] = []
    fence: str | None = None
    for line in lines:
        m = FENCE_RE.match(line)
        if m:
            fence = None if fence == m.group(1)[0] else (m.group(1)[0] if fence is None else fence)
            continue
        if fence is not None:
            continue
        hm = HEADING_RE.match(line)
        if hm and len(hm.group(1)) == 2 and hm.group(2).strip() != "参考文献":
            sections.append(hm.group(2).strip())
    return sections


def render_glossary() -> str:
    out = [
        "# 术语表",
        "",
        "正文首次出现的专业术语自动链接至本页；将鼠标悬停在带虚线下划线的术语上即可查看精简解释。",
        "",
        "[硬件规格与模型速查](toolkit/hardware-models.md) · [测量口径与分析方法](toolkit/methodology.md)",
        "",
    ]
    for entry in TERM_GLOSSARY:
        label = entry["zh"] if entry["zh"] == entry["en"] else f'{entry["zh"]}（{entry["en"]}）'
        aliases = "、".join(f"`{alias}`" for alias in dict.fromkeys(entry["aliases"]))
        out += [f'## {label} {{ #{entry["id"]} }}', "", entry["desc"], "", f"常见写法：{aliases}。", ""]
    return "\n".join(out)


def build_part_page(part: Part) -> None:
    out: list[str] = [
        "---",
        f"title: {PART_CN[part.num]} {part.title}",
        "---",
        "",
        '<div class="part-cover" markdown>',
        "",
        f'<div class="part-kicker">{PART_CN[part.num]}</div>',
        "",
        f'# {PART_CN[part.num]} {part.title} {{ .part-title }}',
        "",
        f'<div class="chapter-meta"><span>{len(part.chapters)} 章</span></div>',
        "",
        "</div>",
        "",
        f'<p class="part-lede">{PART_LEDES[part.num]}</p>',
        "",
        "## 本篇导览",
        "",
        *PART_INTROS[part.num].splitlines(),
        "",
        "## 本篇章节",
        "",
        '<div class="grid cards chapter-cards" markdown>',
        "",
    ]
    for ch in part.chapters:
        sec_label = f"{len(ch.sections)} 节" if ch.sections else "精讲"
        summary = CHAPTER_SUMMARIES.get(ch.num, "")
        out += [
            f"- **第 {ch.num} 章 · {ch.title}**",
            "",
            "    ---",
            "",
            f'    <span class="chapter-card-summary">{html.escape(summary)}</span>',
            "",
            f"    {sec_label} · [阅读本章](../chapters/{ch.slug}.md)",
            "",
        ]
    out += ["</div>", ""]
    write_staged(f"docs/parts/{part.slug}.md", "\n".join(out))


def build_progress_page(parts: dict[int, Part]) -> None:
    total = len(CHAPTERS)
    out = [
        "# 阅读进度 { .part-title }",
        "",
        '<p class="progress-lede">勾选已读完的章节。进度保存在当前浏览器本地，支持导出 JSON 或复制同步链接跨设备恢复。</p>',
        "",
        f'<div class="progress-summary"><div class="progress-track"><span id="book-progress-fill"></span></div><span id="book-progress-text">0 / {total} 章</span></div>',
        "",
        f'<div class="book-progress" data-total="{total}">',
    ]
    for part in parts.values():
        out.append('  <section class="progress-part">')
        out.append(f'    <h2 id="part-{part.num}">{PART_CN[part.num]} {part.title}</h2>')
        for ch in part.chapters:
            out.append(
                f'    <label class="progress-row"><input type="checkbox" data-chapter="{ch.num}">'
                f'<span class="progress-num">{ch.num:02d}</span>'
                f'<span class="progress-title">第 {ch.num} 章 {html.escape(ch.title)}</span>'
                f'<a class="progress-link" href="../chapters/{ch.slug}/">阅读</a></label>'
            )
        out.append("  </section>")
    out += [
        "</div>",
        "",
        '<div class="progress-actions">',
        '  <button type="button" id="book-progress-export" class="md-button">导出 JSON</button>',
        '  <label class="md-button" for="book-progress-import">导入 JSON</label>',
        '  <input type="file" id="book-progress-import" accept="application/json" hidden>',
        '  <button type="button" id="book-progress-link" class="md-button">复制同步链接</button>',
        '  <button type="button" id="book-progress-reset" class="md-button">重置</button>',
        "</div>",
        '<p class="progress-feedback" id="book-progress-feedback" aria-live="polite"></p>',
    ]
    write_staged("docs/progress.md", "\n".join(out))


def main() -> None:
    parts: dict[int, Part] = {num: Part(num, title) for num, title in PART_TITLES.items()}
    problems: list[str] = []

    # 1. Stage homepage docs/index.md -> site-src/index.md and drop repo README from docs_dir
    index_src = SITE / "docs/index.md"
    if not index_src.is_file():
        raise SystemExit(f"missing homepage source: {index_src}")
    problems.extend(validate_markdown(index_src))
    (SITE / "index.md").write_text(index_src.read_text(encoding="utf-8"), encoding="utf-8")
    index_src.unlink()
    staged_readme = SITE / "README.md"
    if staged_readme.is_file():
        staged_readme.unlink()

    # 2. Validate and annotate all 23 chapters
    for num, part_num, title in CHAPTERS:
        ch = Chapter(num, part_num, title)
        path = SITE / f"docs/chapters/{ch.slug}.md"
        if not path.is_file():
            raise SystemExit(f"missing chapter source: {path}")
        problems.extend(validate_markdown(path))
        lines = path.read_text(encoding="utf-8").splitlines()
        ch.sections = extract_h2_sections(lines)
        annotated = "\n".join(TermAnnotator("../glossary.md").annotate_lines(lines)).rstrip() + "\n"
        path.write_text(annotated, encoding="utf-8")
        parts[part_num].chapters.append(ch)

    # 3. Validate and annotate topics
    topics_dir = SITE / "docs/topics"
    if topics_dir.is_dir():
        for path in sorted(topics_dir.glob("*.md")):
            problems.extend(validate_markdown(path))
            lines = path.read_text(encoding="utf-8").splitlines()
            annotated = "\n".join(TermAnnotator("../glossary.md").annotate_lines(lines)).rstrip() + "\n"
            path.write_text(annotated, encoding="utf-8")

    # 4. Validate toolkit pages
    toolkit_dir = SITE / "docs/toolkit"
    if toolkit_dir.is_dir():
        for path in sorted(toolkit_dir.glob("*.md")):
            problems.extend(validate_markdown(path))

    if problems:
        raise SystemExit("Markdown validation failed:\n" + "\n".join(problems))

    # 5. Generate part landing pages, glossary, progress page, and redirect shim
    for part in parts.values():
        build_part_page(part)
    write_staged("docs/glossary.md", render_glossary())
    build_progress_page(parts)
    write_staged(
        "progress/index.html",
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=../docs/progress/"><title>阅读进度</title>',
    )
    print(f"Built {len(CHAPTERS)} chapters across {len(parts)} parts.")


if __name__ == "__main__":
    main()
