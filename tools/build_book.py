#!/usr/bin/env python3
"""Build the chapter-based book site from the repo's Markdown sources.

The site is organised as 5 parts / 31 globally numbered chapters:

  Part 1  (ch 1-5)   基础与度量      <- docs/modules/M0-foundations/README.md
  Part 2  (ch 6-18)  推理框架        <- README.md
  Part 3  (ch 19-24) 算子优化        <- docs/modules/M2-operators/README.md
  Part 4  (ch 25-30) 国产芯片适配    <- docs/modules/M3-domestic-chips/README.md
  Part 5  (ch 31)    综合项目        <- docs/modules/M4-capstone/README.md

Chapters are written to docs/chapters/, part pages to docs/parts/, back
matter to docs/backmatter/. Old week/day/minute framing is rewritten to
chapter/section references; code blocks are left untouched.
"""
from __future__ import annotations

import html
import posixpath
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site-src"

PART_TITLES = {
    1: "用算术强度与脊点判断瓶颈在算力、带宽还是启动开销，再手算 batch、prefill 与 KV 如何改写上限。",
    2: "TTFT、ITL、吞吐、排队深度各量一件事；只报均值会藏住长尾，分位数还要配样本量。",
    3: "先分清 CPU 与 GPU 两条时间线：是 kernel 在跑、在等数据，还是在等 CPU 发下一步。",
    4: "把首 token 延迟、token 间延迟与吞吐固化成脚本：条件固定，数字才可比。",
    5: "第一份基线要回答三件事：拐点在哪、长尾多长、瓶颈更像算力还是带宽。",
}
PART_CN = {1: "第一部分", 2: "第二部分", 3: "第三部分", 4: "第四部分", 5: "第五部分"}
PART_LEDES = {
    1: "先把“快”与“慢”算清楚：Roofline、四个度量、profiling、压测脚本与第一份基线。",
    2: "围绕一个 OpenAI 兼容的推理服务，从心智模型、vLLM/SGLang，到量化、投机解码、解耦、单位经济学与发表。",
    3: "打开推理引擎的盒子：CUDA/Triton、fused attention、量化 matmul、FlashAttention/CUTLASS，最后把自写 kernel 接回引擎。",
    4: "同一模型、同一评测集，在国产芯片上跑通、压测、归因，写出一份可复现的对比报告。",
    5: "把前面所有工具合到一份跨芯片推理优化报告里。",
}

PART_INTROS = {
    1: """本部分先把“度量”补齐：Roofline 判断一个 kernel 受算力还是带宽限制，四个指标定义 TTFT、ITL、吞吐与排队深度，profiling 定位时间花在哪个算子上，最后用一套可复现的压测脚本跑出第一份基线。

本部分不需要 GPU 就能读完前三章；第 4、5 章需要一台跑着推理服务的机器。读完这五章，你手里会有：一张 Roofline 图、一张指标对照表、一套 profiling 读数方法、一份基线报告。""",
    2: """本部分围绕一个不断长大的 OpenAI 兼容推理服务展开：先建立心智模型与模型架构的账本，再部署 vLLM、SGLang，做度量、压测、量化、投机解码、解耦与单位经济学，最后把结果写成一份可复现的 benchmark。

13 章中，第 6–11 章是主线；第 12–17 章可以按需跳读；第 15 章（大规模集群 PD）与第 18 章（TensorRT-LLM）是选读。

### 适合谁

熟悉 Python、了解 Transformer、会用命令行的工程师；不需要 serving、Kubernetes 或 CUDA 经验。已经做过 serving 的读者可以从第 8 章开始，把前面当查漏补缺。

### 需要准备

7–8B 模型约需一块 24GB 显卡；少数实验需要 H100。vLLM、SGLang、Prometheus/Grafana、Python 3.10+。""",
    3: """本部分打开推理引擎的盒子：先用 CUDA 与 Triton 写 kernel，再读 fused attention 与量化 matmul 的推导，最后读 FlashAttention/CUTLASS 并把自己写的 kernel 接回 vLLM。

前置：第二部分第 6–8 章（vLLM 服务已跑通）；需要 GPU 与 CUDA 工具链，部分实验需要 `ncu`。

6 章：第 19–21 章写 kernel，第 22–23 章读工业实现，第 24 章接回引擎。""",
    4: """本部分以 Ascend 为主线：先画国产芯片的软件栈地图，再读 CANN/AscendC，跑通同一模型，做算子覆盖分析、移植案例与精度回归，最后写一份跨芯片对比报告初稿。

前置：第二部分（vLLM/SGLang 已跑通）；需要 Ascend NPU 或云上试用环境；无卡时按各章的替代路径做阅读与配置。""",
    5: """把前四部分的工具合到同一模型、同一评测集上：两套硬件、同一负载、同一指标，写出一份可复现的跨芯片推理优化报告。""",
}

# (chapter number, part number, title, source module, source index, optional)
CHAPTERS = [
    (1, 1, "Roofline：先算，再优化", "m0", 0, False),
    (2, 1, "推理度量：四个数，一个分布", "m0", 1, False),
    (3, 1, "Profiling：看时间花在哪", "m0", 2, False),
    (4, 1, "写压测脚本：bench_openai.py", "m0", 3, False),
    (5, 1, "跑出第一份基线", "m0", 4, False),
    (6, 2, "心智模型", "m1", 0, False),
    (7, 2, "模型架构", "m1", 1, False),
    (8, 2, "vLLM：先部署，再读内部实现", "m1", 2, False),
    (9, 2, "度量基础设施", "m1", 3, False),
    (10, 2, "SGLang、RadixAttention 与批处理内部", "m1", 4, False),
    (11, 2, "压测到 1000 并发", "m1", 5, False),
    (12, 2, "量化取舍", "m1", 6, False),
    (13, 2, "投机解码与 KV 淘汰", "m1", 7, False),
    (14, 2, "解耦式服务与 Kubernetes", "m1", 8, False),
    (15, 2, "大规模集群 PD 优化", "m1", 9, True),
    (16, 2, "单位经济学与路由器", "m1", 10, False),
    (17, 2, "发表，然后养成阅读习惯", "m1", 11, False),
    (18, 2, "TensorRT-LLM：编译式推理引擎", "m1", 12, True),
    (19, 3, "CUDA 优化阶梯：一个 GEMM，四次提速", "m2", 0, False),
    (20, 3, "Triton 入门：用 Python 写 kernel", "m2", 1, False),
    (21, 3, "Fused Attention 深讲：先推导，再写 kernel", "m2", 2, False),
    (22, 3, "量化 Matmul 深讲：量化方案本身", "m2", 3, False),
    (23, 3, "FlashAttention 2/3 与 CUTLASS：读工业级实现", "m2", 4, False),
    (24, 3, "收口：把自写 kernel 接入推理引擎", "m2", 5, False),
    (25, 4, "芯片与软件栈地图：先画地图，再谈适配", "m3", 0, False),
    (26, 4, "CANN 与 AscendC 深讲：算子从定义到编译到调用", "m3", 1, False),
    (27, 4, "同一模型在 Ascend 上跑通 + vllm-ascend 插件架构", "m3", 2, False),
    (28, 4, "算子覆盖分析", "m3", 3, False),
    (29, 4, "移植案例：把 Triton kernel 搬到 Ascend", "m3", 4, False),
    (30, 4, "精度回归与调优报告", "m3", 5, False),
    (31, 5, "跨芯片推理优化报告", "m4", 0, False),
]


CHAPTER_SUMMARIES = {
    1: "先用算术强度和脊点判断一次推理受算力、显存带宽还是启动开销限制；再用手算例子理解 batch、prefill 与 KV 怎样改变结论。",
    2: "TTFT、ITL、吞吐、排队深度量的是四件不同的事；只报平均值会藏住长尾。",
    3: "Profiling 先问时间花在哪个 kernel、哪类访存、哪次启动上，再谈优化。",
    4: "压测脚本的价值不在跑得快，而在每次跑的条件都一样、结果能复现。",
    5: "第一份基线要能回答：拐点在哪、p99 多长、瓶颈在算力还是带宽。",
    6: "一个 kernel 慢，先分清在等计算、等显存搬运还是等启动；三种区间对应三套动作。",
    7: "MHA、GQA、MLA、MoE 改变的是每 token 的 KV 账本，优化清单要按模型重算。",
    8: "vLLM 把 KV 缓存切成可分页的块，再用调度器把空闲 block 分给随时进出的请求；先看浪费怎么消失，再看吞吐从哪里来。",
    9: "度量基础设施要先定义每个面板在量什么，再谈 Grafana 好不好看。",
    10: "SGLang 的 RadixAttention 把前缀复用做成一棵树；连续批处理让 batch 不再是固定集合。",
    11: "压测要扫出整条吞吐—延迟曲线；拐点决定容量，1000 并发只是观测边界。硬件不够时，写清饱和段与待补测条件。",
    12: "量化省的是搬运量，代价落在精度；先看质量表，再看加速比。",
    13: "投机解码的收益由接受率决定；交叉点比低 QPS 的加速比更有用。",
    14: "解耦的前提是传输账算得过来：prefill 与 decode 的瓶颈不同，才值得分开。",
    15: "大规模 PD 的难点在 KV 池、传输和容错，不在把两个服务拆开。",
    16: "单位经济学最终都落到每百万 token 成本；利用率比单价更影响成本。",
    17: "发表 benchmark 是把条件、命令、分位数写全；阅读习惯是让输入持续更新。",
    18: "TensorRT-LLM 用编译换极限性能，也换来了架构锁定；选型看边界条件。",
    19: "一个 GEMM 的四次提速：分块、向量化、占用率，最后都回到 Roofline 验证。",
    20: "Triton 够用时写 Triton，需要精确控制 warp 与流水线时回 CUDA/CUTLASS。",
    21: "Fused attention 的核心是 online softmax 与 SRAM 容量约束，不是把公式抄一遍。",
    22: "量化 matmul 的关键在 scale 粒度与 dequant 融合，省搬运而不是省 FLOPs。",
    23: "读工业实现先画骨架：主循环、tile 加载、rescale；特化分支是后面的事。",
    24: "把自写 kernel 接回引擎，才算完成从 kernel 到端到端收益的闭环。",
    25: "国产芯片适配先画软件栈地图，再问模型能不能跑、算子覆盖到哪。",
    26: "CANN/AscendC 的算子链路是定义、编译、调用；读懂最小闭环比背 API 有用。",
    27: "同一模型在 Ascend 上跑通，难点在版本矩阵、插件 hook 与踩坑记录。",
    28: "算子覆盖分析要拿 profiler 的真实清单逐条对照，不靠猜。",
    29: "移植 kernel 先找语义等价的最小闭环，再处理内存层级与同步差异。",
    30: "精度回归与性能对比要同评测集、同负载、同版本；差异要归因到算子。",
    31: "综合项目的重点是同一套条件、可追溯的数据和明确的证据等级：把同一模型在两套硬件上的差异做两遍并写清楚。",
}

CHAPTER_FIGURES = {
    1: ("roofline.svg", "Roofline 示意图：横轴为算术强度，纵轴为可达性能；斜线是带宽上限，横线是算力上限，交点是脊点；decode 点在斜线一侧，prefill 点越过脊点", "先看斜线、横线与交点（本算例的脊点是 156 FLOP/字节），再找 decode 与 prefill 示例点。结论：decode 在带宽斜线一侧，增加算力不改第一瓶颈；prefill 越过脊点后更受算力限制。"),
    2: ("timeline.svg", "一次流式请求的时间线：请求到达、排队、prefill、TTFT、逐 token decode 与 ITL", "先看 TTFT 在哪里结束，再看其后的 token 间隔如何组成 ITL。结论：排队和 prefill 都计入 TTFT，不能和 decode 的 step time 混成一个数。"),
    3: ("profiler-trace.svg", "CPU 与 GPU 双泳道 trace 示意：CPU 侧依次是 Python 调度、连续 kernel 启动与 synchronize，GPU 侧是若干短 kernel、一段空白、一个长 kernel 与更多短 kernel；空白与 CPU 的同步等待对齐", "先对齐 CPU 与 GPU 两条泳道，看 GPU 空白下方 CPU 在做什么；再区分发射跟不上、同步等待与拷贝。结论：GPU 时间线上的空白不是没有成本，它和 kernel 一样要归因。"),
    7: ("model-ledgers.svg", "五个模型的总参数与每 8K 请求全局 KV 对照：两列均为对数轴，同一行是一个模型；左列总参数，右列每 8K 请求的全局 KV，固定状态另计", "先看左右两列的对数轴，同一行对应同一个模型；再比较两列的位置，参数规模与 KV 账并不同向。结论：总参数管容量与放置，KV 结构管长上下文，两张账要分别算。"),
    6: ("kernel-anatomy.svg", "kernel 时间账：计算、显存搬运、启动与同步三项取最大，并指向不同优化动作", "先看三项时间哪一项最大，再看底部对应的动作。结论：判断瓶颈在计算、带宽还是启动，优化清单才选得对。"),
    8: ("kv-pages.svg", "PagedAttention 的块表映射：左侧逻辑块 L0–L5，经中间块表映射到右侧非连续物理页", "先看左侧逻辑块和中间块表，再沿映射看右侧物理页；相邻逻辑块可以落在不相邻的物理页。结论：按需分页把浪费压到最后一个未满 block，代价是 attention 按块表 gather。"),
    10: ("batching.svg", "静态批处理与连续批处理的时间线对比：请求结束后的空槽与新请求换入", "先对齐两条时间轴看空白与换入点：静态批处理等最慢请求，连续批处理在请求结束时释放槽位。结论：它提高 GPU 占用率，不改变单请求计算速度。"),
    11: ("throughput-knee.svg", "吞吐与 p95 延迟随并发变化的示意曲线，标出拐点与饱和段", "先看吞吐何时停止线性增长，再对照 p95 何时陡增。结论：拐点是容量结论的起点；1000 并发只是观测边界，不是考核指标。"),
    12: ("quant.svg", "FP16、FP8、INT4 的位数、权重体积与 scale 示意", "先比较三种格式的位数和权重体积，再看 scale 与误差说明。结论：位宽降低直接减少搬运量，但误差、离群值和反量化开销决定它能用到哪里。"),
    13: ("spec.svg", "投机解码：draft 模型先猜多个 token，target 模型一次验证并接受前缀", "先看 draft 猜了几个，再看 target 验证后接受几个、在哪里截断。结论：收益由接受率和额外计算成本共同决定，低 QPS 的加速比不能外推。"),
    14: ("pd.svg", "解耦式服务：请求进入 prefill 池，产出的 KV 经传输层进入 decode 池", "先看 prefill 与 decode 两池，再沿 KV 传输箭头看成本项。结论：拆分的收益要扣掉 KV 传输、两池空闲与运维成本，不是拆开就免费。"),
    15: ("pd.svg", "大规模 PD 的抽象示意：prefill 池、decode 池与 KV 传输；未展开跨节点拓扑", "这张图只抽象两池与 KV 传输，不代表跨节点拓扑。先看 KV 路径，再回到正文核对跨机带宽、池配比和容错；结论必须落在你自己的传输账上。"),
    19: ("cuda-ladder.svg", "CUDA 优化阶梯：朴素版、shared memory 分块、向量化访存、占用率调优", "先看横轴四个版本，再看每一步只改哪个变量；柱高是示意，不是实测。结论：每一步都要由 ncu 读数和 Roofline 点验证，不能只看总时间。"),
    22: ("quant.svg", "量化 matmul 的 scale 粒度、反量化与矩阵乘融合示意", "先看权重与 scale 的粒度，再看 dequant 是否融进 matmul。结论：量化省的是权重搬运，精度损失与反量化开销决定净收益。"),
    25: ("chip-stack.svg", "国产芯片软件栈：驱动与固件、SDK/运行时、算子库、框架插件、推理引擎", "从下往上看五层软件栈，再检查同一模型经过哪些层。结论：跑通和性能问题都要按层定位，不能只归因到芯片本身。"),
    27: ("chip-stack.svg", "vllm-ascend 插件在芯片软件栈中的位置：接入引擎与算子库之间", "先找插件层，再看它向上接 vLLM、向下接 CANN 与算子实现。结论：版本矩阵、插件 hook 和算子覆盖要分别验证，不能互相替代。"),
}

TERM_GLOSSARY = [
    {"id": "roofline", "aliases": ("Roofline", "Roofline 模型"), "en": "Roofline model", "zh": "屋顶线模型", "desc": "用峰值算力、峰值显存带宽和算术强度估算 kernel 性能上限的模型"},
    {"id": "arithmetic_intensity", "aliases": ("算术强度", "arithmetic intensity"), "en": "arithmetic intensity", "zh": "算术强度", "desc": "每搬运 1 字节数据所执行的浮点运算次数"},
    {"id": "ridge_point", "aliases": ("脊点", "ridge point"), "en": "ridge point", "zh": "脊点", "desc": "硬件峰值算力与峰值显存带宽之比，用来判断计算受限还是带宽受限"},
    {"id": "memory_bound", "aliases": ("显存带宽受限", "memory-bound", "memory bound"), "en": "memory-bound", "zh": "显存带宽受限", "desc": "运行时间主要受显存带宽而不是计算能力限制的状态"},
    {"id": "compute_bound", "aliases": ("计算受限", "compute-bound", "compute bound"), "en": "compute-bound", "zh": "计算受限", "desc": "运行时间主要受计算能力而不是显存带宽限制的状态"},
    {"id": "ttft", "aliases": ("TTFT",), "en": "Time to First Token", "zh": "首 token 延迟", "desc": "从请求发出到客户端收到第一个输出 token 的时间"},
    {"id": "itl", "aliases": ("ITL",), "en": "Inter-Token Latency", "zh": "token 间延迟", "desc": "同一请求相邻输出 token 之间的平均时间"},
    {"id": "tpot", "aliases": ("TPOT",), "en": "Time Per Output Token", "zh": "每输出 token 时间", "desc": "decode 阶段平均生成一个输出 token 所需的时间"},
    {"id": "prefill", "aliases": ("prefill", "Prefill", "预填充"), "en": "prefill", "zh": "预填充", "desc": "在生成前一次性处理输入 prompt，并写入 KV 缓存的前向阶段"},
    {"id": "decode", "aliases": ("decode", "Decode", "解码阶段", "解码请求"), "en": "autoregressive decoding", "zh": "解码", "desc": "逐 token 生成输出、并复用已有 KV 缓存的自回归阶段"},
    {"id": "kv_cache", "aliases": ("KV 缓存", "KV Cache", "KV cache"), "en": "Key-Value Cache", "zh": "键值缓存", "desc": "保存历史 token 的注意力 K/V 向量、避免 decode 时重复计算的显存区域"},
    {"id": "mha", "aliases": ("MHA",), "en": "Multi-Head Attention", "zh": "多头注意力", "desc": "每个 query 头拥有独立 K/V 头的标准注意力结构"},
    {"id": "gqa", "aliases": ("GQA",), "en": "Grouped-Query Attention", "zh": "分组查询注意力", "desc": "多个 query 头共享一组 K/V 头，以降低 KV 缓存和访存开销"},
    {"id": "mla", "aliases": ("MLA",), "en": "Multi-head Latent Attention", "zh": "多头潜变量注意力", "desc": "先把 K/V 压缩到低维潜变量、再按需展开的注意力结构"},
    {"id": "moe", "aliases": ("MoE",), "en": "Mixture of Experts", "zh": "混合专家", "desc": "每个 token 只激活少数专家网络，以较少计算获得更大参数容量"},
    {"id": "paged_attention", "aliases": ("PagedAttention", "Paged Attention"), "en": "Paged Attention", "zh": "分页注意力", "desc": "把 KV 缓存切成固定大小 block，并通过块表映射到非连续显存的注意力实现"},
    {"id": "block_table", "aliases": ("block table",), "en": "block table", "zh": "块表", "desc": "记录逻辑 KV block 到物理显存 block 映射的每请求索引表"},
    {"id": "continuous_batching", "aliases": ("连续批处理", "continuous batching", "Continuous Batching"), "en": "continuous batching", "zh": "连续批处理", "desc": "每次模型迭代都重新组批，请求完成即退出、等待请求可立即补入的调度方式"},
    {"id": "chunked_prefill", "aliases": ("chunked prefill", "Chunked Prefill", "分块 prefill"), "en": "chunked prefill", "zh": "分块预填充", "desc": "把长 prompt 的 prefill 切成小块，与 decode 请求交错执行的调度策略"},
    {"id": "prefix_caching", "aliases": ("prefix caching", "Prefix Caching", "前缀缓存"), "en": "prefix caching", "zh": "前缀缓存", "desc": "复用多个请求相同前缀已计算出的 KV 缓存，避免重复 prefill"},
    {"id": "radix_attention", "aliases": ("RadixAttention", "Radix Attention"), "en": "Radix Attention", "zh": "基数树注意力", "desc": "用基数树管理可共享前缀的 KV 缓存，以 token 序列为键做复用与淘汰"},
    {"id": "quantization", "aliases": ("量化", "quantization"), "en": "quantization", "zh": "量化", "desc": "用更低位宽表示权重或激活，以降低显存占用和搬运量", "exclude_prefixes": ("向", "预")},
    {"id": "awq", "aliases": ("AWQ",), "en": "Activation-aware Weight Quantization", "zh": "激活感知权重量化", "desc": "依据激活分布保护重要权重通道的低比特权重量化方法"},
    {"id": "gptq", "aliases": ("GPTQ",), "en": "Generative Pre-trained Transformer Quantization", "zh": "GPT 权重量化", "desc": "利用二阶信息逐层最小化重建误差的训练后权重量化方法"},
    {"id": "fp8", "aliases": ("FP8",), "en": "8-bit Floating Point", "zh": "8 位浮点", "desc": "使用 8 位浮点表示权重或激活的低精度格式"},
    {"id": "int4", "aliases": ("INT4",), "en": "4-bit Integer", "zh": "4 位整数", "desc": "使用 4 位整数表示权重、配合 scale 反量化的极低位宽格式"},
    {"id": "speculative_decoding", "aliases": ("投机解码", "speculative decoding"), "en": "speculative decoding", "zh": "投机解码", "desc": "先用草稿模型生成多个候选 token，再由目标模型一次验证和接受"},
    {"id": "draft_model", "aliases": ("草稿模型", "draft model"), "en": "draft model", "zh": "草稿模型", "desc": "投机解码中快速生成候选 token 的小模型"},
    {"id": "target_model", "aliases": ("target model",), "en": "target model", "zh": "目标模型", "desc": "投机解码中负责验证候选 token 的原始大模型"},
    {"id": "acceptance_rate", "aliases": ("接受率", "acceptance rate"), "en": "acceptance rate", "zh": "接受率", "desc": "草稿 token 被目标模型验证通过的比例"},
    {"id": "kv_eviction", "aliases": ("KV 淘汰", "KV eviction", "KV cache eviction"), "en": "KV cache eviction", "zh": "KV 缓存淘汰", "desc": "按策略丢弃部分历史 KV，以降低长上下文显存与读取量"},
    {"id": "pd_disaggregation", "aliases": ("PD 解耦", "prefill-decode disaggregation", "解耦式服务"), "en": "prefill-decode disaggregation", "zh": "预填充-解码解耦", "desc": "把 prefill 与 decode 放到不同实例或硬件池，分别优化两种瓶颈"},
    {"id": "goodput", "aliases": ("goodput", "有效吞吐"), "en": "goodput", "zh": "有效吞吐", "desc": "满足延迟或正确性约束的成功请求吞吐"},
    {"id": "hpa", "aliases": ("HPA",), "en": "Horizontal Pod Autoscaler", "zh": "水平 Pod 自动扩缩容", "desc": "Kubernetes 中按指标自动调整 Pod 副本数的控制器"},
    {"id": "keda", "aliases": ("KEDA",), "en": "Kubernetes Event-driven Autoscaling", "zh": "Kubernetes 事件驱动扩缩容", "desc": "把事件或外部指标接入 HPA 的扩缩容组件"},
    {"id": "router", "aliases": ("router", "Router", "路由器"), "en": "router", "zh": "路由器", "desc": "在多后端或多模型之间按成本、延迟或质量选择目标服务的组件"},
    {"id": "tensorrt_llm", "aliases": ("TensorRT-LLM",), "en": "TensorRT for Large Language Models", "zh": "TensorRT 大语言模型推理引擎", "desc": "NVIDIA 的编译式 LLM 推理引擎，在加载时生成执行计划并做图优化"},
    {"id": "cuda", "aliases": ("CUDA",), "en": "Compute Unified Device Architecture", "zh": "统一计算设备架构", "desc": "NVIDIA 的 GPU 并行计算平台与编程模型"},
    {"id": "cuda_graph", "aliases": ("CUDA Graph", "CUDA 图"), "en": "CUDA Graph", "zh": "CUDA 图", "desc": "把多次 kernel 启动录制成一张图，以减少重复启动开销"},
    {"id": "triton", "aliases": ("Triton",), "en": "Triton", "zh": "Triton GPU 编程语言", "desc": "用 Python 风格语法编写 GPU kernel 的语言与编译器"},
    {"id": "shared_memory", "aliases": ("shared memory", "Shared Memory", "共享内存"), "en": "shared memory", "zh": "共享内存", "desc": "同一线程块内可见、延迟远低于显存的可编程片上存储"},
    {"id": "warp", "aliases": ("warp", "Warp"), "en": "warp", "zh": "线程束", "desc": "GPU 上一次调度执行的 32 个线程组成的硬件执行单位"},
    {"id": "occupancy", "aliases": ("occupancy", "Occupancy", "占用率"), "en": "occupancy", "zh": "占用率", "desc": "SM 上实际驻留 warp 数占硬件上限的比例"},
    {"id": "bank_conflict", "aliases": ("bank conflict", "Bank Conflict", "存储体冲突"), "en": "bank conflict", "zh": "存储体冲突", "desc": "多个线程访问同一 shared memory bank 的不同地址而被迫串行"},
    {"id": "gemm", "aliases": ("GEMM",), "en": "General Matrix Multiplication", "zh": "通用矩阵乘法", "desc": "注意力投影、MLP 和大多数线性层的基础计算算子"},
    {"id": "fused_attention", "aliases": ("fused attention", "Fused Attention", "融合注意力"), "en": "fused attention", "zh": "融合注意力", "desc": "把 attention 的多个步骤融进一个 kernel，避免中间矩阵写入显存"},
    {"id": "flash_attention", "aliases": ("FlashAttention",), "en": "FlashAttention", "zh": "闪存注意力", "desc": "通过分块与在线 softmax 减少 attention 中间矩阵读写的高效实现"},
    {"id": "online_softmax", "aliases": ("online softmax", "Online Softmax", "在线 softmax"), "en": "online softmax", "zh": "在线 softmax", "desc": "分块计算 softmax 时维护运行最大值与归一化因子的数值稳定算法"},
    {"id": "cutlass", "aliases": ("CUTLASS",), "en": "CUDA Templates for Linear Algebra Subroutines", "zh": "CUDA 线性代数模板库", "desc": "NVIDIA 提供的可组合 GEMM 与流水线 kernel 模板库"},
    {"id": "cann", "aliases": ("CANN",), "en": "Compute Architecture for Neural Networks", "zh": "昇腾神经网络计算架构", "desc": "华为昇腾 NPU 的软件栈与算子开发平台"},
    {"id": "ascendc", "aliases": ("AscendC", "Ascend C"), "en": "Ascend C", "zh": "昇腾 C 算子编程语言", "desc": "用于在昇腾 NPU 上编写自定义算子的 C++ 编程接口"},
    {"id": "npu", "aliases": ("NPU",), "en": "Neural Processing Unit", "zh": "神经网络处理器", "desc": "面向神经网络张量计算加速的专用处理器"},
    {"id": "vllm_ascend", "aliases": ("vllm-ascend",), "en": "vLLM Ascend plugin", "zh": "vLLM 昇腾插件", "desc": "把昇腾算子与运行时接入 vLLM 的硬件后端插件"},
    {"id": "operator_coverage", "aliases": ("算子覆盖", "operator coverage"), "en": "operator coverage", "zh": "算子覆盖", "desc": "模型所需算子中，目标硬件已有可用实现的比例与清单"},
    {"id": "tensor_parallelism", "aliases": ("张量并行", "tensor parallelism"), "en": "tensor parallelism", "zh": "张量并行", "desc": "把单个矩阵或张量切分到多张卡上协同计算"},
    {"id": "mfu", "aliases": ("MFU",), "en": "Model FLOPs Utilization", "zh": "模型浮点算力利用率", "desc": "实际模型计算吞吐占硬件峰值算力的比例"},
    {"id": "slo", "aliases": ("SLO",), "en": "Service Level Objective", "zh": "服务等级目标", "desc": "对延迟、可用性或错误率设定的可度量目标"},
    {"id": "qps", "aliases": ("QPS",), "en": "Queries Per Second", "zh": "每秒查询数", "desc": "系统每秒处理的请求数"},
    {"id": "ncu", "aliases": ("ncu",), "en": "NVIDIA Nsight Compute", "zh": "NVIDIA 内核性能分析器", "desc": "采集单个 GPU kernel 硬件计数器与瓶颈指标的工具"},
    {"id": "nsys", "aliases": ("nsys",), "en": "NVIDIA Nsight Systems", "zh": "NVIDIA 系统级性能分析器", "desc": "采集 CPU、GPU、拷贝与同步时间线的系统级 profiler"},
    {"id": "tbe", "aliases": ("TBE",), "en": "Tensor Boost Engine", "zh": "张量加速引擎", "desc": "昇腾 CANN 中较早的算子开发框架，用 DSL 描述算子计算"},
    {"id": "rope", "aliases": ("RoPE",), "en": "Rotary Position Embedding", "zh": "旋转位置编码", "desc": "通过旋转 query/key 表示注入相对位置信息的编码方法"},
    {"id": "rag", "aliases": ("RAG",), "en": "Retrieval-Augmented Generation", "zh": "检索增强生成", "desc": "先从外部知识库检索相关文档，再把结果交给模型生成"},
]

CHAPTER_FIGURE_PLACEMENT = {
    1: (3, 6),    # 1.3 after the ridge-point cross-over is solved
    2: (None, 1), # chapter opens with the metric question; figure before the table
    3: (2, 1),    # 3.2 CPU/GPU trace lanes
    7: (5, 1),    # 7.5 five-model ledger comparison
    6: (1, 1),    # 6.1 three bottleneck buckets
    8: (1, 1),    # 8.1 PagedAttention block mapping
    10: (4, 1),   # 10.4 continuous batching
    11: (4, 1),   # 11.4 throughput knee
    12: (1, 1),   # 12.1 weight volume by precision
    13: (2, 1),   # 13.2 proposer/scorer/verifier pipeline
    14: (1, 1),   # 14.1 prefill/decode separation
    15: (2, 1),   # 15.2 Mooncake request path
    19: (2, 1),   # 19.2 measurement bench and optimization ladder
    22: (3, 1),   # 22.3 fused dequant+matmul
    25: (2, 1),   # 25.2 software stack layers
    27: (3, 1),   # 27.3 plugin position in the stack
}

M1_WEEK_TO_CH = {
    "1": 6, "1续": 7, "2": 8, "3": 9, "4": 10, "5": 11, "6": 12,
    "7": 13, "8": 14, "8续": 15, "9": 16, "10": 17, "11": 18,
}
M2_WEEK_TO_CH = {str(i + 1): 19 + i for i in range(6)}
M3_WEEK_TO_CH = {str(i + 1): 25 + i for i in range(6)}
MODULE_WEEKS = {"m1": M1_WEEK_TO_CH, "m2": M2_WEEK_TO_CH, "m3": M3_WEEK_TO_CH, "m0": M1_WEEK_TO_CH, "m4": M1_WEEK_TO_CH}

FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
LINK_RE = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)(\))")


@dataclass
class Heading:
    level: int
    title: str
    line: int


@dataclass
class Chapter:
    num: int
    part: int
    title: str
    source: str
    index: int
    optional: bool = False
    sections: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return f"chapter-{self.num:02d}"

    @property
    def link(self) -> str:
        return f"docs/chapters/{self.slug}.md"


@dataclass
class Part:
    num: int
    title: str
    chapters: list[Chapter] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return f"part-{self.num}"

    @property
    def link(self) -> str:
        return f"docs/parts/{self.slug}.md"


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def write(rel: str, text: str) -> Path:
    path = SITE / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def scan(lines: list[str]) -> list[Heading]:
    out: list[Heading] = []
    fence: str | None = None
    for i, line in enumerate(lines):
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        m = HEADING_RE.match(line)
        if m:
            out.append(Heading(len(m.group(1)), m.group(2).strip(), i))
    return out


def section_end(hs: list[Heading], idx: int, total: int) -> int:
    level = hs[idx].level
    for other in hs[idx + 1:]:
        if other.level <= level:
            return other.line
    return total


def take(lines: list[str], hs: list[Heading], idx: int) -> list[str]:
    return lines[hs[idx].line:section_end(hs, idx, len(lines))]


def strip_blank(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def find_h2(lines: list[str], hs: list[Heading], title: str) -> int:
    for i, h in enumerate(hs):
        if h.level == 2 and h.title == title:
            return i
    raise SystemExit(f"missing h2: {title!r}")


def clean_ws(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"（\s*）", "", text)
    text = re.sub(r"，\s*，", "，", text)
    text = re.sub(r"。。", "。", text)
    text = re.sub(r"\s+([，。；：！？、])", r"\1", text)
    return text


def protect(text: str) -> tuple[str, list[str]]:
    store: list[str] = []

    def keep(match: re.Match[str]) -> str:
        store.append(match.group(0))
        return f"\x00{len(store) - 1}\x00"

    text = re.sub(r"\x60[^\x60]*\x60", keep, text)
    text = re.sub(r"\]\([^)]+\)", keep, text)
    text = re.sub(r"<https?://[^>]+>", keep, text)
    text = re.sub(r"https?://\S+", keep, text)
    return text, store


def restore(text: str, store: list[str]) -> str:
    def back(match: re.Match[str]) -> str:
        return store[int(match.group(1))]
    return re.sub(r"\x00(\d+)\x00", back, text)




KEYLINE_RE = re.compile(r"^\*\*(本章目标|本章产出|本章主线|阅读材料|目标|怎么做|产出|阅读锚点|硬件门槛|状态|小结|验收|产出物)([。：:])?\*\*")
BRIEF_RE = re.compile(r"^\*\*(本章目标|本章产出)[。：:]?\*\*\s*(.+)$")
KEYLINE_KIND = {
    "本章目标": "goal", "目标": "goal", "本章主线": "goal",
    "本章产出": "output", "产出": "output", "产出物": "output",
    "阅读材料": "read", "阅读锚点": "read",
    "怎么做": "how",
    "硬件门槛": "note", "状态": "note",
    "小结": "summary", "验收": "summary",
}

REFERENCE_LABEL_RE = re.compile(r"^\s*>?\s*(?:\*\*(阅读锚点|阅读材料|延伸阅读|延伸|阅读|参考)(?:[。：:]?)\*\*|(阅读锚点|阅读材料|延伸阅读|延伸|阅读|参考)[：:])(?:[。：:]?)\s*(.*)$")
HOW_LABEL_RE = re.compile(r"^\*\*怎么做\*\*(?:（[^）]*）|\([^)]*\))?[：:]?\s*(.*)$")
OUTPUT_LABEL_RE = re.compile(r"^\*\*(?:本章产出|产出|产出物)([。：:]?)\*\*(?:[。：:]?)\s*.*$")
BUFFER_LABEL_RE = re.compile(r"^\*\*缓冲\*\*.*$")


def chapter_goal(lines: list[str]) -> tuple[list[str], list[str]]:
    """Extract the chapter-level goal as plain prose."""
    goals: list[str] = []
    kept: list[str] = []
    for line in lines:
        match = BRIEF_RE.match(line)
        if match:
            if match.group(1) == "本章目标":
                goals.append(match.group(2).strip())
            continue
        kept.append(line)
    return goals, kept


def insert_figure_after_paragraphs(
    lines: list[str], figure: list[str], after_paragraphs: int = 1
) -> list[str]:
    """Insert a figure after the Nth prose block in a section."""
    if not figure or after_paragraphs <= 0:
        return lines
    out: list[str] = []
    prose_seen = 0
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            out.append(lines[i])
            i += 1
            continue
        first = lines[i]
        stripped = first.lstrip()

        # Keep fenced blocks together and do not count them as prose.
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            out.append(lines[i])
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith(fence):
                out.append(lines[i])
                i += 1
            if i < len(lines):
                out.append(lines[i])
                i += 1
            continue

        block: list[str] = []
        while i < len(lines) and lines[i].strip():
            block.append(lines[i])
            i += 1

        is_prose = not (
            stripped.startswith(("#", "|", "<", ">", "---"))
            or re.match(r"^(?:[-*+]\s|\d+\.\s)", stripped)
            or KEYLINE_RE.match(first)
        )
        out.extend(block)
        if is_prose:
            prose_seen += 1
            if prose_seen == after_paragraphs:
                out += ["", *figure, ""]
    if prose_seen < after_paragraphs:
        out += ["", *figure, ""]
    return out


def chapter_figure(chapter: Chapter) -> list[str]:
    figure = CHAPTER_FIGURES.get(chapter.num)
    if not figure:
        return []
    filename, alt, caption = figure
    return [
        '<figure class="book-figure">',
        f'<img src="../../../assets/figures/{filename}" alt="{html.escape(alt)}" loading="lazy">',
        f'<figcaption><span class="book-figure__label">图 {chapter.num}-1</span>{html.escape(caption)}</figcaption>',
        "</figure>",
    ]


def split_long_paragraph(line: str) -> list[str]:
    if len(line) <= 280 or line.startswith(("|", "#", ">", "- ", "* ", "+ ", "    ", "<", "!")):
        return [line]
    if re.match(r"^\d+\.\s", line):
        return [line]
    protected, store = protect(line)
    sentences = [part.strip() for part in re.split(r"(?<=[。！？])", protected) if part.strip()]
    if len(sentences) <= 1:
        return [line]
    groups: list[str] = []
    buf = ""
    for sentence in sentences:
        if buf and len(buf) + len(sentence) > 260:
            groups.append(buf)
            buf = sentence
        else:
            buf += sentence
        if len(buf) >= 190:
            groups.append(buf)
            buf = ""
    if buf:
        groups.append(buf)
    if len(groups) <= 1:
        return [line]
    return [restore(group, store) for group in groups]


INLINE_PROTECT_RE = re.compile(
    r"(`[^`]*`|!?\[[^\]]*\]\([^)]*\)|<[^>]+>|https?://\S+)"
)


def protect_inline(text: str) -> tuple[str, list[str]]:
    store: list[str] = []

    def keep(match: re.Match[str]) -> str:
        store.append(match.group(0))
        return f"\x00{len(store) - 1}\x00"

    return INLINE_PROTECT_RE.sub(keep, text), store


def inside_parentheses(text: str, pos: int) -> bool:
    depth = 0
    for ch in text[:pos]:
        if ch in "（(":
            depth += 1
        elif ch in "）)":
            depth = max(0, depth - 1)
    return depth > 0


NON_DEFINITION_PREFIXES = (
    "如果", "若", "例如", "比如", "见", "可", "当", "在", "不", "只",
    "按", "默认", "下文", "本节", "第", "以及", "或", "并", "但",
)


def looks_like_definition(text: str, end: int) -> bool:
    match = re.match(r"\s*[（(]([^（）()]{1,180})[）)]", text[end:])
    if match is None:
        return False
    content = match.group(1).strip()
    if not content or content.startswith(NON_DEFINITION_PREFIXES):
        return False
    if re.search(r"[A-Za-z]", content):
        return True
    if re.search(r"[，,：:]", content):
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
    if stripped.startswith(("#", "|", "<", ">", "---", "```", "~~~")):
        return False
    if line.startswith(("    ", "\t")):
        return False
    return True


def glossary_annotation(term: str, entry: dict[str, str]) -> str:
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
    return f'{term}<span class="term-def">（{inside}）</span>'


class TermAnnotator:
    """Annotate the first prose use of each known term in a chapter once."""

    def __init__(self) -> None:
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
                    escaped = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
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
            after = protected[match.end():]
            if (
                entry["id"] in defined_here
                or inside_parentheses(protected, match.start())
                or looks_like_definition(protected, match.end())
            ):
                continue
            replacement = glossary_annotation(match.group(0), entry)
            store.append(replacement)
            end = match.end()
            if protected[end:end + 1] == " ":
                end += 1
            protected = protected[: match.start()] + f"\x00{len(store) - 1}\x00" + protected[end:]
        return restore(protected, store)

    def annotate_lines(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        fence: str | None = None
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


def format_blocks(lines: list[str], references: list[str] | None = None) -> list[str]:
    """Split walls of text, wrap labels in callouts, and keep code/tables separate."""
    out: list[str] = []
    i = 0

    def blank() -> None:
        if out and out[-1] != "":
            out.append("")

    while i < len(lines):
        line = lines[i]
        ref_match = REFERENCE_LABEL_RE.match(line)
        if ref_match:
            label = ref_match.group(1) or ref_match.group(2)
            content = ref_match.group(3).strip()
            if (
                label != "阅读材料"
                and content
                and not content.startswith("本章不读新材料")
                and references is not None
                and re.search(r"https?://|\[[^\]]+\]\([^)]+\)", content)
            ):
                references.append(content)
            i += 1
            continue

        how_match = HOW_LABEL_RE.match(line)
        if how_match:
            rest = how_match.group(1).strip()
            if rest:
                out.extend(split_long_paragraph(rest))
                out.append("")
            i += 1
            continue

        if OUTPUT_LABEL_RE.match(line) or BUFFER_LABEL_RE.match(line):
            i += 1
            continue

        if line.startswith("```"):
            blank()
            raw_info = line.strip()[3:].strip()
            lang = raw_info.split()[0] if raw_info else ""
            lang = re.sub(r"[^A-Za-z0-9_+-]", "", lang)
            data_attr = f' data-lang="{html.escape(lang, quote=True)}"' if lang else ""
            out += [f'<div class="code-block"{data_attr} markdown>', ""]
            if line.strip() == "```":
                out.append("```text")
            else:
                out.append(line)
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                out.append(lines[i])
                i += 1
            if i < len(lines):
                out.append(lines[i])
                i += 1
            out += ["", "</div>"]
            blank()
            continue

        if line.lstrip().startswith("|"):
            blank()
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                out.append(lines[i])
                i += 1
            blank()
            continue

        match = KEYLINE_RE.match(line)
        if match:
            kind = KEYLINE_KIND.get(match.group(1), "note")
            if kind == "goal":
                rest = line[match.end():].lstrip("。：: ")
                if rest:
                    blank()
                    out.extend(split_long_paragraph(rest))
                    blank()
                i += 1
                continue
            blank()
            out += [f'<div class="keyline keyline--{kind}" markdown>', "", line, "", "</div>"]
            blank()
            i += 1
            continue

        if line.startswith("#"):
            blank()
            out.append(line)
            blank()
            i += 1
            continue

        if line.startswith(">"):
            blank()
            while i < len(lines) and lines[i].startswith(">"):
                out.append(lines[i])
                i += 1
            blank()
            continue

        if line.startswith(("- ", "* ", "+ ")) or re.match(r"^\d+\.\s", line):
            blank()
            while i < len(lines) and (
                lines[i].startswith(("- ", "* ", "+ ")) or re.match(r"^\d+\.\s", lines[i])
            ):
                out.append(lines[i])
                i += 1
            blank()
            continue

        if not line.strip():
            blank()
            i += 1
            continue

        for paragraph in split_long_paragraph(line):
            out.append(paragraph)
            out.append("")
        i += 1

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.splitlines()


def validate_markdown(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    problems: list[str] = []
    fence: str | None = None
    for i, line in enumerate(lines):
        if line.startswith("```"):
            if fence is None:
                fence = line.strip()
                if fence == "```":
                    problems.append(f"{path}:{i + 1}: code fence without language")
            else:
                fence = None
        if line.lstrip().startswith("|") and (i == 0 or not lines[i - 1].lstrip().startswith("|")):
            if i + 1 >= len(lines) or not re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]):
                problems.append(f"{path}:{i + 1}: table without separator row")
    if fence is not None:
        problems.append(f"{path}: unclosed code fence")
    return problems


def map_week(text: str, module: str) -> str:
    current = MODULE_WEEKS[module]
    prefixes = {"m0": M1_WEEK_TO_CH, "m1": M1_WEEK_TO_CH, "m2": M2_WEEK_TO_CH, "m3": M3_WEEK_TO_CH, "m4": M1_WEEK_TO_CH}

    def repl(match: re.Match[str]) -> str:
        mod = match.group(1)
        n = match.group(2)
        cont = match.group(3) or ""
        key = n + ("续" if cont else "")
        table = current if not mod else MODULE_WEEKS[{"0": "m0", "1": "m1", "2": "m2", "3": "m3"}[mod]]
        num = table.get(key)
        return f"第 {num} 章" if num else match.group(0)

    # module-prefixed weeks
    text = re.sub(r"M([0-3])\s*第\s*(\d+)\s*周(（续）)?", repl, text)
    # bare weeks, longest labels first
    for key in sorted(current, key=len, reverse=True):
        cont = "（续）" if key.endswith("续") else ""
        n = key[:-1] if cont else key
        text = re.sub(rf"第\s*{re.escape(n)}\s*周{re.escape(cont)}", f"第 {current[key]} 章", text)
    # week ranges such as 第 1–6 周
    def range_repl(match: re.Match[str]) -> str:
        a = current.get(match.group(1))
        b = current.get(match.group(2))
        return f"第 {a}–{b} 章" if a and b else match.group(0)
    text = re.sub(r"第\s*(\d+)\s*[–-]\s*(\d+)\s*周", range_repl, text)
    return text


def transform_text(text: str, module: str, chapter: Chapter) -> str:
    text, store = protect(text)
    text = text.replace("上周五", "上一章第 5 节").replace("下周一", "下一章第 1 节")
    # Weekday references become numbered sections.
    for i, day in enumerate(("一", "二", "三", "四", "五"), 1):
        text = text.replace(f"周{day}", f"{chapter.num}.{i}")
    text = text.replace("本周", "本章").replace("下周", "下一章").replace("上周", "上一章")
    text = text.replace("每周", "每章").replace("这周", "本章")
    text = map_week(text, module)
    # Module-prefixed week ranges and section references (e.g. M0 第 1 节).
    def module_range(match: re.Match[str]) -> str:
        table = MODULE_WEEKS[{"0": "m0", "1": "m1", "2": "m2", "3": "m3"}[match.group(1)]]
        a = table.get(match.group(2))
        b = table.get(match.group(3))
        return f"第 {a}–{b} 章" if a and b else match.group(0)

    text = re.sub(r"M([0-3])\s*第\s*(\d+)\s*[–-]\s*(\d+)\s*周", module_range, text)
    text = re.sub(r"M0\s*第\s*(\d+)\s*节", lambda m: f"第 {m.group(1)} 章", text)
    # Chapter + section references, then bare section references.
    text = re.sub(r"第\s*(\d+)\s*章第\s*(\d+)\s*节", lambda m: f"第 {m.group(1)}.{m.group(2)} 节", text)
    if module == "m0":
        text = re.sub(r"第\s*(\d+)\s*节", lambda m: f"第 {m.group(1)} 章", text)
    else:
        text = re.sub(r"第\s*(\d+)\s*节", lambda m: f"{chapter.num}.{m.group(1)}", text)
    # Module labels become parts.
    text = text.replace("M0–M4", "第一部分到第五部分")
    text = re.sub(
        r"(?<![A-Za-z0-9])M([0-4])(?![A-Za-z0-9])",
        lambda m: {"0": "第一部分", "1": "第二部分", "2": "第三部分",
                   "3": "第四部分", "4": "第五部分"}[m.group(1)],
        text,
    )
    text = text.replace("模块地图", "全书结构")
    if module == "m4":
        text = text.replace("这个模块", "这个项目").replace("逐周教程", "正文教程")
    text = text.replace("本模块", "本部分").replace("该模块", "该部分")
    text = text.replace("这个模块", "这个部分")
    text = text.replace("模块", "部分")
    # Chapter-intro labels.
    text = text.replace("**重点。**", "**本章目标。**")
    text = text.replace("**形式。**", "**阅读材料。**")
    text = text.replace("**产出。**", "**本章产出。**")
    text = text.replace("**本周主线**", "**本章主线**")
    # Week wording.
    text = text.replace("后面九周", "后面的章节").replace("后面五周", "后面的章节")
    text = text.replace("每一周", "每一章").replace("这一周", "这一章")
    text = text.replace("上一周", "上一章").replace("下一周", "下一章")
    text = text.replace("这一续周", "这一章").replace("续周", "章")
    text = text.replace("6 周分工", "章节分工").replace("10 周路线图", "第二部分")
    text = text.replace("11 周主线", "第二部分").replace("每周", "每章")
    # Course wording.
    text = text.replace("55 节核心课", "13 章核心内容").replace("核心课", "核心章节")
    text = text.replace("节课", "节").replace("这一课", "这一节")
    text = text.replace("课程", "本书")
    # Time framing.
    text = text.replace("（60 分钟，", "（")
    text = text.replace("（60 分钟）", "")
    text = text.replace("60 分钟/节", "每节")
    text = text.replace("每节 60 分钟", "每节")
    text = text.replace("5 节 × 60 分钟", "5 节")
    text = text.replace("共 300 分钟（5 小时）", "")
    text = text.replace("300 分钟（5 小时）", "")
    text = text.replace("前 30 分钟", "前半部分").replace("后 30 分钟", "后半部分")
    text = text.replace("30 分钟读一篇论文", "读一篇论文")
    text = text.replace("这 60 分钟用来干什么", "这一章要解决什么")
    text = text.replace("这一节的 60 分钟", "这一节的时间")
    text = text.replace("租卡的 60 分钟", "租卡的上机时间")
    text = text.replace("后面几周", "后面的章节").replace("后面的周", "后面的章节")
    text = text.replace("模块总时长", "")
    text = re.sub(r"每天\s*60\s*分钟", "", text)
    text = re.sub(r"每天", "", text)
    text = re.sub(r"(?<![\d.])55\s*小时", "", text)
    text = re.sub(r"(?<![\d.])5\s*小时", "", text)
    text = re.sub(r"(?<![\d.])11\s*周", "第二部分", text)
    text = re.sub(r"(?<![\d.])10\s*周", "第二部分", text)
    text = re.sub(r"(?<![\d.])6\s*周", "本部分", text)
    text = re.sub(r"(?<![\d.])5\s*周", "本部分", text)
    text = re.sub(r"(?<![\d.])1\s*周", "第一部分", text)
    text = re.sub(r"(\d+)\s*周", lambda m: f"{m.group(1)} 章", text)
    text = re.sub(r"(?<![\d.])55\s*节", "13 章", text)
    text = re.sub(r"(?<![\d.])30\s*节", "6 章", text)
    text = re.sub(r"(?<![\d.])10\s*节", "可选章节", text)
    text = text.replace("| 周 | 主题 | 产出 |", "| 阶段 | 主题 | 产出 |")
    text = text.replace("第三部分/第四部分", "第三、第四部分")
    text = text.replace("整周", "整章").replace("两周", "两章").replace("三周", "三章")
    text = text.replace("四周", "四章").replace("五周", "五章").replace("六周", "六章")
    text = text.replace("七周", "七章").replace("八周", "八章").replace("九周", "九章").replace("十周", "十章")
    text = text.replace("每一课", "每一节").replace("这一课", "这一节")
    text = text.replace("全周", "全章")
    text = text.replace("那一周", "那一章").replace("隔着一周", "隔着一章")
    text = text.replace("过一周", "过一段时间").replace("那几周", "那几章")
    text = text.replace("数周起步", "工作量很大").replace("读一小时", "读一遍")
    text = text.replace("第一次真正派上用场的周", "第一次真正派上用场")
    text = text.replace("本章花 60 分钟把抓取修好", "本章把抓取修好")
    text = text.replace("先算一笔时间总账：本部分 × 5 节 = 1800 分钟 = 30 小时。本章 5 节是 300 分钟、。这 30 小时的账要这么读：", "这五章的账要这么读：")
    text = text.replace("那才是这 30 小时真正留下的东西", "那才是这份报告真正留下的东西")
    text = text.replace("先算一笔总账：本部分 × 5 节 = 1800 分钟 = 30 小时。本章 5 节是 300 分钟、。报告就是这 30 小时的收口。", "这五章的材料最后收进一份报告。")
    text = text.replace("这五章的材料最后收进一份报告。把第 19–24 章的东西收进一份报告：", "把第 19–24 章的东西收进一份报告：")
    text = text.replace("纯文字的周", "纯文字的章节")
    text = text.replace("缓冲日", "复习").replace("学习日", "阅读日")
    text = re.sub(r"(\d+\.\d)([\u4e00-\u9fff])", r"\1 \2", text)
    text = re.sub(r"(\d+\.\d+) 直在", r"\1 一直在", text)
    text = re.sub(r"([\u4e00-\u9fff])(\d+\.\d)", r"\1 \2", text)
    text = re.sub(r"本\s*(\d+\.\d+)", r"\1", text)
    text = re.sub(r"(第\s*\d+\s*章)(\d)", r"\1 \2", text)
    text = text.replace("第一部分 的", "第一部分的").replace("第二部分 的", "第二部分的")
    text = text.replace("第三部分 的", "第三部分的").replace("第四部分 的", "第四部分的")
    text = text.replace("第五部分 的", "第五部分的")
    text = text.replace("每节，每章", "每章").replace("每节，", "每节")
    # Remove spaces introduced when module references become part/chapter references.
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(第\s*[一二三四五]部分)", r"\1", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(第\s*[一二三四五](?:、第?\s*[一二三四五])*部分)", r"\1", text)
    text = re.sub(r"(第\s*[一二三四五]部分)\s+(?=[\u4e00-\u9fff、，。；：）])", r"\1", text)
    text = re.sub(r"(第\s*\d+\s*章)\s+(?=[\u4e00-\u9fff、，。；：）])", r"\1", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(第\s*\d+\s*章)", r"\1", text)
    text = clean_ws(text)
    return restore(text, store)


def transform_title(title: str, module: str, chapter: Chapter) -> str:
    title = re.sub(r"^第\s*\d+\s*周(（续）)?\s*[：·]\s*", "", title)
    title = re.sub(r"^第\s*\d+\s*节\s*[·：]\s*", "", title)
    title = re.sub(r"^周[一二三四五]\s*[·：]\s*", "", title)
    title = re.sub(r"（60 分钟）", "", title)
    title = title.replace("论文精读（原文均为选读）", "论文选读")
    title = title.replace("建议的 6 周安排", "工作阶段")
    title = transform_text(title, module, chapter)
    return clean_ws(title).strip()


def map_repo_path(path: str) -> str:
    table = {
        "README.md": "docs/parts/part-2.md",
        "docs/modules/M0-foundations/README.md": "docs/parts/part-1.md",
        "docs/modules/M2-operators/README.md": "docs/parts/part-3.md",
        "docs/modules/M3-domestic-chips/README.md": "docs/parts/part-4.md",
        "docs/modules/M4-capstone/README.md": "docs/chapters/chapter-31.md",
        "docs/MODULES.md": "docs/MODULES.md",
        "progress.md": "docs/progress.md",
        "tracker": "docs/progress.md",
        "tracker/index.html": "docs/progress.md",
    }
    return table.get(path, path)


def rewrite_links(text: str, source_dir: str, target_dir: str) -> str:
    out: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            out.append(line)
            continue
        if fence is not None:
            out.append(line)
            continue

        def repl(match: re.Match[str]) -> str:
            url = match.group(2)
            if url.startswith(("#", "http://", "https://", "mailto:", "data:", "//", "/")):
                return match.group(0)
            path, sep, frag = url.partition("#")
            if not path:
                return match.group(0)
            trailing = path.endswith("/")
            resolved = posixpath.normpath(posixpath.join(source_dir, path))
            if resolved == ".." or resolved.startswith("../"):
                return match.group(0)
            mapped = map_repo_path(resolved)
            if trailing and not mapped.endswith((".md", ".html")):
                mapped = posixpath.join(mapped, "index.html")
            rel = posixpath.relpath(mapped, target_dir)
            if sep:
                rel += "#" + frag
            return match.group(1) + rel + match.group(3)

        out.append(LINK_RE.sub(repl, line))
    return "\n".join(out)


@dataclass
class Source:
    module: str
    path: Path
    lines: list[str]
    hs: list[Heading]
    h1: Heading


def load_source(module: str) -> Source:
    path = {
        "m0": SITE / "docs/modules/M0-foundations/README.md",
        "m1": SITE / "README.md",
        "m2": SITE / "docs/modules/M2-operators/README.md",
        "m3": SITE / "docs/modules/M3-domestic-chips/README.md",
        "m4": SITE / "docs/modules/M4-capstone/README.md",
    }[module]
    lines = read_lines(path)
    hs = scan(lines)
    h1 = next(h for h in hs if h.level == 1)
    return Source(module, path, lines, hs, h1)


def source_chapter_blocks(src: Source) -> list[list[str]]:
    blocks: list[list[str]] = []
    if src.module == "m0":
        picks = [h for h in src.hs if h.level == 2 and re.match(r"第\s*\d+\s*(?:节|章)", h.title)]
    elif src.module == "m1":
        picks = [h for h in src.hs if h.level == 3 and re.match(r"第\s*\d+\s*(?:周|章)", h.title)]
    elif src.module in ("m2", "m3"):
        picks = [h for h in src.hs if h.level == 2 and re.match(r"第\s*\d+\s*(?:周|章)", h.title)]
    else:
        chunks = [src.lines[src.h1.line:]]
        return chunks
    for i, h in enumerate(picks):
        idx = src.hs.index(h)
        end = section_end(src.hs, idx, len(src.lines))
        blocks.append(src.lines[h.line:end])
    return blocks


def source_pre(src: Source) -> list[str]:
    if src.module == "m4":
        return []
    if src.module == "m0":
        first = next(h for h in src.hs if h.level == 2 and re.match(r"第\s*\d+\s*(?:节|章)", h.title))
        return src.lines[src.h1.line + 1:first.line]
    if src.module == "m1":
        first = next(h for h in src.hs if h.level == 3 and re.match(r"第\s*\d+\s*(?:周|章)", h.title))
        return src.lines[src.h1.line + 1:first.line]
    first = next(h for h in src.hs if h.level == 2 and re.match(r"第\s*\d+\s*(?:周|章)", h.title))
    return src.lines[src.h1.line + 1:first.line]


def source_post(src: Source) -> list[str]:
    if src.module == "m4":
        return []
    if src.module == "m0":
        last = [h for h in src.hs if h.level == 2 and re.match(r"第\s*\d+\s*(?:节|章)", h.title)][-1]
    elif src.module == "m1":
        last = [h for h in src.hs if h.level == 3 and re.match(r"第\s*\d+\s*(?:周|章)", h.title)][-1]
    else:
        last = [h for h in src.hs if h.level == 2 and re.match(r"第\s*\d+\s*(?:周|章)", h.title)][-1]
    end = section_end(src.hs, src.hs.index(last), len(src.lines))
    return src.lines[end:]


def transform_lines(lines: list[str], module: str, chapter: Chapter) -> list[str]:
    out: list[str] = []
    fence: str | None = None
    for line in lines:
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            out.append(line)
            continue
        if fence is not None:
            out.append(line)
            continue
        out.append(transform_text(line, module, chapter))
    return out


def chapter_is_section(h: Heading, module: str) -> bool:
    if module == "m0":
        return h.level == 3
    if module == "m1":
        return h.level == 4
    if module == "m2":
        return h.level == 3 and re.match(r"第\s*\d+\s*节", h.title) is not None
    if module == "m3":
        return h.level == 3
    if module == "m4":
        return h.level == 2
    return False


def section_title(title: str, module: str, chapter: Chapter) -> str:
    title = re.sub(r"^第\s*\d+\s*节\s*[·：]\s*", "", title)
    title = re.sub(r"^周[一二三四五]\s*[·：]\s*", "", title)
    title = re.sub(r"（60 分钟）", "", title)
    if title.startswith("论文精读"):
        title = "论文选读"
    if title.startswith("本周论文精读"):
        title = "论文选读"
    if title.startswith("建议的 6 周安排"):
        title = "工作阶段"
    return transform_title(title, module, chapter)


def subhead_level(h: Heading, module: str) -> int:
    if module == "m0":
        return 2 if h.level == 3 else 3
    if module == "m1":
        return 3
    if module in ("m2", "m3"):
        return 3 if h.level == 3 else 4
    if module == "m4":
        return 3 if h.level == 3 else 4
    return h.level


def render_chapter(chapter: Chapter, raw: list[str], module: str) -> None:
    hs = scan(raw)
    if not hs:
        return
    body = raw[hs[0].line + 1:]
    body_hs = scan(body)
    sections = [h for h in body_hs if chapter_is_section(h, module)]
    section_titles = [section_title(h.title, module, chapter) for h in sections]
    chapter.sections = section_titles
    first_section_line = sections[0].line if sections else len(body)
    intro = body[:first_section_line]
    source_dir = {
        "m0": "docs/modules/M0-foundations",
        "m1": ".",
        "m2": "docs/modules/M2-operators",
        "m3": "docs/modules/M3-domestic-chips",
        "m4": "docs/modules/M4-capstone",
    }[module]
    summary = CHAPTER_SUMMARIES.get(chapter.num, "")
    out: list[str] = [
        f'# 第 {chapter.num} 章 {chapter.title} {{ .chapter-title }}',
        "",
    ]
    if summary:
        out += [f'<p class="chapter-summary">{html.escape(summary)}</p>', ""]
    intro_lines = transform_lines(intro, module, chapter) if intro else []
    goals, intro_lines = chapter_goal(intro_lines)
    if goals:
        goal_text = "；".join(html.escape(goal) for goal in goals)
        out += [
            f'<p class="chapter-goal"><strong>本章目标</strong>：{goal_text}</p>',
            "",
        ]
    figure_lines = chapter_figure(chapter)
    placement = CHAPTER_FIGURE_PLACEMENT.get(chapter.num, (None, 1) if not sections else (1, 1))
    target_section, figure_after = placement
    if figure_lines and target_section is None:
        intro_lines = insert_figure_after_paragraphs(intro_lines, figure_lines, figure_after)
        figure_lines = []
    references: list[str] = []
    if intro_lines:
        out += format_blocks(intro_lines, references)
        out.append("")
    sec = 0
    for i, h in enumerate(body_hs):
        end = body_hs[i + 1].line if i + 1 < len(body_hs) else len(body)
        chunk = body[h.line:end]
        if chapter_is_section(h, module):
            sec += 1
            title = section_titles[sec - 1]
            out.append(f"## {chapter.num}.{sec} {title}")
            out.append("")
            section_lines = transform_lines(chunk[1:], module, chapter)
            if figure_lines and target_section == sec:
                section_lines = insert_figure_after_paragraphs(
                    section_lines, figure_lines, figure_after
                )
                figure_lines = []
            out += format_blocks(section_lines, references)
            out.append("")
        else:
            level = subhead_level(h, module)
            title = transform_title(h.title, module, chapter)
            out.append("#" * level + " " + title)
            out.append("")
            out += format_blocks(transform_lines(chunk[1:], module, chapter), references)
            out.append("")
    if figure_lines:
        out += ["", *figure_lines, ""]
    out = TermAnnotator().annotate_lines(out)
    unique_refs: list[str] = []
    seen_refs: set[str] = set()
    for ref in references:
        normalized = re.sub(r"\s+", " ", ref).strip()
        if normalized and normalized not in seen_refs:
            seen_refs.add(normalized)
            unique_refs.append(normalized)
    if unique_refs:
        out += ["## 参考文献", ""]
        out += [f"- {ref}" for ref in unique_refs]
        out.append("")
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = rewrite_links(text, source_dir, "docs/chapters")
    if chapter.num == 17:
        text = override_chapter_17(text)
    text = re.sub(r"^\*\*(?:怎么做|本章产出|产出|产出物|阅读材料|缓冲)[^\n]*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    write(f"docs/chapters/{chapter.slug}.md", text)


def build_sources_and_chapters() -> dict[str, Source]:
    sources = {m: load_source(m) for m in ("m0", "m1", "m2", "m3", "m4")}
    blocks: dict[str, list[list[str]]] = {m: source_chapter_blocks(s) for m, s in sources.items()}
    for num, part_num, title, module, index, optional in CHAPTERS:
        if index >= len(blocks[module]):
            raise SystemExit(f"missing source block: {module} #{index} (ch {num})")
        ch = Chapter(num, part_num, title, module, index, optional)
        ch.raw = blocks[module][index]
        CH_BY_NUM[num] = ch
    return sources


CH_BY_NUM: dict[int, Chapter] = {}
PARTS: dict[int, Part] = {i: Part(i, PART_TITLES[i]) for i in PART_TITLES}


def build_parts() -> None:
    for num, part_num, title, module, index, optional in CHAPTERS:
        PARTS[part_num].chapters.append(CH_BY_NUM[num])


def m1_block(src: Source, title: str) -> list[str]:
    idx = find_h2(src.lines, src.hs, title)
    return take(src.lines, src.hs, idx)


def transform_fragment(lines: list[str], module: str, chapter: Chapter, title: str | None = None) -> list[str]:
    hs = scan(lines)
    out: list[str] = []
    start = 0
    if hs and hs[0].line == 0 and title is not None:
        out.append("## " + title)
        out.append("")
        start = 1
    out.extend(transform_lines(lines[start:], module, chapter))
    return out


def build_part_page(part: Part, sources: dict[str, Source]) -> None:
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
        "## 本部分导览",
        "",
    ]
    out += PART_INTROS[part.num].splitlines()
    out += ["", "## 本部分章节", "", '<div class="grid cards chapter-cards" markdown>', ""]
    for ch in part.chapters:
        count = len(ch.sections)
        meta = f"{count} 节" if count else "专题"
        if ch.optional:
            meta += " · 选读"
        out += [
            f"- **第 {ch.num} 章 · {ch.title}**",
            "",
            "    ---",
            "",
            f"    {meta} · {PART_CN[part.num]}",
            "",
            f"    [阅读本章](../chapters/{ch.slug}.md)",
            "",
        ]
    out += ["</div>", ""]

    post = strip_blank(source_post(sources["m0" if part.num == 1 else "m2" if part.num == 3 else "m3"])) if part.num in (1, 3, 4) else []
    if post:
        out += ["## 本部分检查清单", ""]
        post_text = "\n".join(post)
        post_text = post_text.replace("## 本模块检查清单", "")
        module = "m0" if part.num == 1 else "m2" if part.num == 3 else "m3"
        ch = CH_BY_NUM[1 if part.num == 1 else 19 if part.num == 3 else 25]
        post_text = rewrite_links(transform_text(post_text, module, ch), "docs/modules/M0-foundations" if part.num == 1 else "docs/modules/M2-operators" if part.num == 3 else "docs/modules/M3-domestic-chips", "docs/parts")
        out.append(post_text)
        out.append("")

    write(f"docs/parts/{part.slug}.md", "\n".join(out))


def render_appendix(src: Source, h2_title: str, filename: str, new_title: str | None = None,
                    notice: str | None = None, relevel_body: bool = True) -> list[str]:
    idx = find_h2(src.lines, src.hs, h2_title)
    block = take(src.lines, src.hs, idx)
    body = block[1:]
    out: list[str] = [f"# {new_title or h2_title}", ""]
    if notice:
        out += [notice, ""]
    fence: str | None = None
    for line in body:
        m = FENCE_RE.match(line)
        if m:
            marker = m.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            out.append(line)
            continue
        if fence is None and relevel_body:
            hm = HEADING_RE.match(line)
            if hm:
                level = max(1, len(hm.group(1)) - 1)
                line = "#" * level + " " + transform_title(hm.group(2), "m1", CH_BY_NUM[6])
            else:
                line = transform_text(line, "m1", CH_BY_NUM[6])
        out.append(line)
    text = rewrite_links("\n".join(out), ".", "docs/backmatter")
    write(f"docs/backmatter/{filename}", text)
    return [new_title or h2_title, filename]


def build_backmatter(src: Source) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    entries.append(("跟踪进度", "tracking.md"))
    write("docs/backmatter/tracking.md", """# 跟踪进度

阅读进度现在集中在[阅读进度](../progress.md)页：

- 勾选已经读完的章节，进度自动保存在当前浏览器；
- 点“导出 JSON”，把文件带到另一台设备后“导入 JSON”；
- 点“复制同步链接”，进度会编码在 URL 里，打开链接即可恢复；
- 清空浏览器数据前，先导出或复制链接。

## 用 GitHub issue 跟踪

Fork 本仓库，新建一个 issue，把根目录 [progress.md](https://github.com/tdccccc/time-to-first-token/blob/main/progress.md) 里的章节清单复制进去，在 issue 里打勾。GitHub 会把任务列表状态保存在你的账号里，换设备登录也能看到。

## 为什么不做账号系统

站点是纯静态的，没有后端，也不收集账号。阅读进度是本地的，同步链接与 JSON 导出覆盖了跨设备需求；如果要做多人排行榜，才需要后端与身份系统。""")
    render_appendix(src, "发表 benchmark 之前", "benchmark.md")
    render_appendix(src, "决策点", "decisions.md")
    render_appendix(src, "参考资料", "references.md")
    col: list[str] = ["# 来源、贡献与许可", ""]
    for title in ("关于来源的说明", "参与贡献", "许可与出处"):
        idx = find_h2(src.lines, src.hs, title)
        block = take(src.lines, src.hs, idx)
        col.append("## " + transform_title(title, "m1", CH_BY_NUM[6]))
        col.append("")
        for line in block[1:]:
            col.append(transform_text(line, "m1", CH_BY_NUM[6]))
        col.append("")
    write("docs/backmatter/colophon.md", rewrite_links("\n".join(col), ".", "docs/backmatter"))
    entries.append(("来源、贡献与许可", "colophon.md"))
    return entries


def build_catalog(parts: dict[int, Part]) -> None:
    out = ["# 全书目录", "", "5 部分 · 31 章。左侧目录用于跳转，这一页适合一次看完。", ""]
    for part in parts.values():
        out += [f"## {PART_CN[part.num]} {part.title}", "", f"[本部分导览](parts/{part.slug}.md)", ""]
        for ch in part.chapters:
            opt = " · 选读" if ch.optional else ""
            out += [f"### 第 {ch.num} 章 {ch.title}{opt}", ""]
            out.append(f"[阅读本章](chapters/{ch.slug}.md)")
            out.append("")
    write("docs/catalog.md", "\n".join(out))


def build_structure_page(parts: dict[int, Part]) -> None:
    out = ["# 全书结构", "", "全书按主题分为五部分，共 31 章；章内小节按编号排列。", "", "| 部分 | 章节 | 主题 |", "|---|---|---|"]
    for part in parts.values():
        nums = f"{part.chapters[0].num}–{part.chapters[-1].num}" if len(part.chapters) > 1 else str(part.chapters[0].num)
        out.append(f"| [{PART_CN[part.num]} {part.title}](parts/{part.slug}.md) | {nums} | {PART_LEDES[part.num]} |")
    out += ["", "## 章节", ""]
    for part in parts.values():
        out += [f"### {PART_CN[part.num]} {part.title}", ""]
        for ch in part.chapters:
            opt = "（选读）" if ch.optional else ""
            out.append(f"- [第 {ch.num} 章 {ch.title}{opt}](chapters/{ch.slug}.md)")
        out.append("")
    write("docs/MODULES.md", "\n".join(out))


def build_progress_page(parts: dict[int, Part]) -> None:
    out = [
        "# 阅读进度 { .part-title }",
        "",
        '<p class="progress-lede">勾选已经读完的章节。进度只保存在当前浏览器；可以导出 JSON，或生成同步链接带到另一台设备。</p>',
        "",
        '<div class="progress-summary"><div class="progress-track"><span id="book-progress-fill"></span></div><span id="book-progress-text">0 / 31 章</span></div>',
        "",
        '<div class="book-progress" data-total="31">',
    ]
    for part in parts.values():
        out.append('  <section class="progress-part">')
        out.append(f'    <h2>{PART_CN[part.num]} {part.title}</h2>')
        for ch in part.chapters:
            opt = '<span class="progress-opt">选读</span>' if ch.optional else ""
            out.append(
                f'    <label class="progress-row"><input type="checkbox" data-chapter="{ch.num}">'
                f'<span class="progress-num">{ch.num:02d}</span>'
                f'<span class="progress-title">第 {ch.num} 章 {ch.title}{opt}</span>'
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
        '  <span id="book-progress-feedback" class="progress-feedback" aria-live="polite"></span>',
        "</div>",
    ]
    write("docs/progress.md", "\n".join(out))
    write("progress/index.html", """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=../docs/progress/">
  <link rel="canonical" href="../docs/progress/">
  <title>阅读进度 · 边做边学 LLM 推理服务</title>
</head>
<body><p>正在前往<a href="../docs/progress/">阅读进度</a>…</p></body>
</html>""")


def cleanup_sources() -> None:
    for rel in [
        "README.md",
        "docs/modules/M0-foundations/README.md",
        "docs/modules/M2-operators/README.md",
        "docs/modules/M3-domestic-chips/README.md",
        "docs/modules/M4-capstone/README.md",
    ]:
        p = SITE / rel
        if p.exists():
            p.unlink()
    for rel in ["docs/modules/M2-operators", "docs/modules/M4-capstone"]:
        p = SITE / rel
        if p.exists() and not any(p.iterdir()):
            p.rmdir()


def override_chapter_17(text: str) -> str:
    text = text.replace("其他周的论文", "其他章的论文")
    text = text.replace("日历上每 17.1 次的研究时间", "固定的研究节奏")
    text = text.replace("每 17.1 次", "固定节奏")
    text = text.replace("前九章", "前面的章节")
    text = text.replace("不是第 17 章的“下一章”", "不是第 18 章")
    text = text.replace("本章时间紧", "本章篇幅紧")
    text = text.replace("第一部分 脚本", "第一部分的脚本")
    start = text.find("## 17.3 ")
    end = text.find("## 17.5 ")
    if start < 0 or end < 0:
        return text
    body = """## 17.3 建立阅读习惯

阅读习惯不是“多读”，而是一个可运转的机制：固定的来源、一个不空的队列、一段固定的阅读时间。三样齐了，读得多是自然结果；缺一样，就会退回到“有空才读”。

**来源。** 固定四个：arXiv cs.DC、MLSys、OSDI、NeurIPS 效率方向。每次从这四个来源里选，不要漫无目的地刷首页。来源固定，比较才有基准；来源太多，选择本身就会消耗掉阅读的意愿。

**漏斗。** 选论文走三步：扫标题 → 读摘要 → 进队列。标题里出现推理、服务、延迟、吞吐、KV、调度、量化、投机解码这类词，留下；纯训练、纯理论、与推理系统无关的，划掉。标题过关的读摘要，摘要里找三句话：问题是什么、方法是什么、数字是多少；三句话里有一句与你的服务无关，就划掉。最后剩下三篇进队列，每篇记一行：标题、来源、为什么选它。漏斗把“选论文”从一个漫无目的的动作，变成一个可以当场做完的动作。

**队列。** 队列保持三篇，按来源分开：一篇 arXiv cs.DC，一篇 MLSys，一篇 OSDI 或 NeurIPS。读完一篇，从队列顶部取走，再按漏斗补一篇。每篇记一个入队日期：躺了太久还没读的，划掉比留着强。划掉时写一句原因，那句话就是你选篇标准的一部分。

**记录。** 一个文件，一篇一行，四个字段：日期、标题、来源、一句话总结。按阅读顺序往下写，不按日期、不按项目分文件。文件要能被搜索：半年后遇到 KV 淘汰的问题，搜“KV”，过去读过的相关论文会跳出来。记录的价值不在数量，在能被翻出来用。

**节奏。** 把阅读放进固定日程，而不是等“有空”。固定的时段不必很长，但要在日历上占位、可重复；到了时间就做漏斗、读一篇、写记录。忙的时候读队列里最短的一篇，不要跳过；跳过一次，下一次启动的成本会更高。

## 17.4 第一次完整练习：读一篇论文

**目标**：按“只读摘要、方法图和实验，不读全文”的方法读完一篇论文，并留下一段读后记录。你要能说出：先看什么、后看什么、什么可以跳过、读完留下什么。

**怎么做**：

从队列里取最上面的一篇，先用标题与摘要确认它仍然值得读；不是就换下一篇，不纠结。

读的顺序固定：

1. **摘要**：找三句话——问题、方法、数字。三句话找齐，论文的骨架就有了；任何一句找不到，说明摘要没写清楚，或者它不是你现在要读的论文，换下一篇。
2. **方法图**：找到“输入—处理—输出”的主线，先不管符号和细节。方法图看不懂，先看图的标题与图注，再回到正文里对应的段落。
3. **实验**：只看三样：对比对象、指标、数字。问自己：它和谁比、在什么负载下、赢了多少；这三个问题回答不了，实验部分对你就没有信息。
4. **合上论文，写一段记录**：一句话问题、一句话方法、一句话数字，再加一句“它和我的服务有什么关系”。写不出来，说明还没读进去；回去补看摘要或方法图。

第一遍不要读引言、相关工作、证明和附录。它们回答的是“细读时该看什么”，不是“这篇论文值不值得读”。只有当你确认这篇论文与你的服务直接相关，才值得第二遍细读。

**产出**：一条读后记录，四个字段：标题、来源、问题—方法—数字各一句、与当前服务的关系。

"""
    return text[:start] + body + text[end:]
def main() -> None:
    sources = build_sources_and_chapters()
    build_parts()
    for num in sorted(CH_BY_NUM):
        ch = CH_BY_NUM[num]
        render_chapter(ch, ch.raw, ch.source)
    for part in PARTS.values():
        build_part_page(part, sources)
    build_backmatter(sources["m1"])
    build_catalog(PARTS)
    build_structure_page(PARTS)
    build_progress_page(PARTS)
    shutil.copyfile(ROOT / "tools" / "homepage.md", SITE / "index.md")
    cleanup_sources()
    print("built 5 parts / 31 chapters:", ", ".join(f"P{p.num}={len(p.chapters)}" for p in PARTS.values()))


if __name__ == "__main__":
    main()