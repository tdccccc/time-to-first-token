# M3 · 国产芯片适配

> 状态：**大纲**。本模块的完整内容尚未写出，先把学习路径、资源与每节的产出定下来。
>
> 前置：M0（度量、profiling、压测脚本）与 M1（vLLM/SGLang 已跑通）。M3 的终点是：**同一模型、同一评测集，在 NVIDIA 与至少一种国产芯片上都跑通、压测，并写出一份可复现的对比报告**。
>
> **诚实声明**：国产芯片适配进展极快，任何写死的支持矩阵都会很快过时。本模块不写静态结论，只给方法论和可复现的验证脚本；凡是涉及具体芯片、具体版本的实测结果，都按「某年某月某版本实测」的格式记录，由学习者自己验证后填写。**本仓库目前没有任何一条在国产芯片上的实测记录**，不要把大纲里的“预期”当成已验证的事实。

## 为什么单开一个模块

调研结论：国产芯片适配**没有任何动手的学习材料**，只有框架插件、一张支持状态追踪表和零散厂商文档。M3 要补的就是这一块：从“知道有这些芯片”到“真的在上面跑起模型、量出数字”。

## 课程安排（5 周，每周 5 节、每节 30 分钟，沿用 M1 的节奏）

### 第 1 周 · 芯片与软件栈地图

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | 国产芯片全景：昇腾、寒武纪、摩尔线程、海光、昆仑芯、天数智芯、壁仞、燧原、沐曦 | [awesome-china-gpu](https://github.com/bytetora/awesome-china-gpu/blob/HEAD/README.md)（国产芯片推理适配进度追踪表） | 一张芯片清单，标出各自的软件栈 |
| 2 | 软件栈分层：驱动 → SDK → 算子库 → 框架插件 → 推理引擎 | 同上；[vLLM CustomOp 与硬件插件机制](https://docs.vllm.ai/en/latest/design/custom_op/) | 一张分层图 |
| 3 | 插件机制：vLLM 是怎么做到一块代码跑多家芯片的 | [vLLM CustomOp 设计文档](https://docs.vllm.ai/en/latest/design/custom_op/) | 能讲清插件算子的替换点 |
| 4 | 统一算子层：FlagGems / FlagCX 的思路 | [vllm-plugin-FL 文档](https://docs.flagos.io/projects/vllm-plugin-FL/en/latest/overview/features.html)；[FlagTree](https://github.com/flagos-ai/FlagTree) | 一页笔记：统一层解决了什么、没解决什么 |
| 5 | 小结：画一张带日期的软件栈地图，**所有支持状态都标注核实日期** | — | 带日期的地图（过期即作废） |

**硬件**：本周不需要任何硬件，纯阅读与整理。

### 第 2 周 · 同一模型在 Ascend 上跑通

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | 环境：CANN + torch_npu + vllm-ascend 的安装链 | [vllm-ascend](https://github.com/vllm-project/vllm-ascend)（vLLM 官方 Ascend 插件） | 一份安装记录（含版本号） |
| 2 | 部署：华为官方的 vLLM Ascend 部署流程 | [华为 vLLM Ascend 部署与调优（PPT）](https://www.hiascend.com/dam/asset/view/8344df3aad8d40a89d17801757f134c5.pdf) | 同一模型在 Ascend 上跑起来的服务 |
| 3 | 冒烟：同一个 prompt、同一输出长度，确认能出结果 | M0 的 `scripts/bench_openai.py` | 一次成功的端到端调用记录 |
| 4 | 踩坑记录：把安装与部署中遇到的每个问题和解法写下来 | — | 踩坑清单 |
| 5 | 小结：Ascend 与 NVIDIA 的部署差异对照表 | — | 一页差异表 |

**硬件**：**本周需要 Ascend NPU 卡或云上试用环境**，这是本模块最硬的约束。无卡时的替代路径：先读文档与安装脚本、跑通 Docker 镜像的拉取与配置检查（不启动推理），把“待有卡后补测”写进报告；不要编造运行结果。

### 第 3 周 · 算子覆盖分析

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | 导出算子清单：用 PyTorch Profiler 导出一次推理的全部算子 | M0 第 3 节的 profiling 方法 | 一份算子清单 |
| 2 | 逐条对照：哪些算子在 CANN/DTK 里有对应实现，哪些会回退到 CPU 或低效实现 | CANN / 目标芯片官方文档 | 一份算子覆盖矩阵 |
| 3 | 迁移 benchmark 怎么做 | [TritonAscendBench](https://github.com/Gxj230958/TritonAscendBench)（评测生产级 Triton kernel 从 GPU 迁移到 Ascend 的可用性） | 理解迁移评测的设计 |
| 4 | 动手：对清单里最耗时的 3 个算子，查它们的芯片侧实现状态 | 同上 | 一份“Top 3 算子覆盖分析” |
| 5 | 小结：覆盖矩阵的结论——瓶颈在算子覆盖，还是在调度/显存 | — | 一页结论 |

**硬件**：第 1 节需要能跑 profiling 的环境（NVIDIA 卡即可）；第 2–5 节的对照工作无卡可做。

### 第 4 周 · 移植

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | 教学 demo：把一个小推理引擎移植到 Ascend 是怎么做的 | [nano-vllm-ascend](https://github.com/linzm1007/nano-vllm-ascend)（教学用 nano-vllm 移植到 Ascend 的 demo） | 读懂移植的关键改动点 |
| 2 | Triton kernel 迁移：从 GPU 到 Ascend 要改什么 | [TritonAscendBench](https://github.com/Gxj230958/TritonAscendBench) | 一份改动点清单 |
| 3 | 统一算子库：FlagGems 如何让一份 kernel 跑多家芯片 | [vllm-plugin-FL 文档](https://docs.flagos.io/projects/vllm-plugin-FL/en/latest/overview/features.html) | 一页机制笔记 |
| 4 | 动手：选一个小 kernel（或小模型），按第 2–3 节的方法走一遍移植流程 | 同上 | 一份移植记录（含失败的尝试） |
| 5 | 小结：移植的通用检查清单 | — | 检查清单 |

**硬件**：第 1–3 节无卡可做；第 4 节动手需要目标芯片环境，无卡时先完成流程梳理，标注“待有卡后补测”。

### 第 5 周 · 精度回归与调优报告

| 节 | 内容 | 资源 | 产出 |
|---|---|---|---|
| 1 | 精度回归：同一评测集，在 NVIDIA 与 Ascend 上各跑一遍，对比输出质量 | M0 的基线报告模板 | 一份精度对比表 |
| 2 | 性能对比：TTFT / ITL / 吞吐，用 M0 的压测脚本在两边各跑三级并发 | M0 的 `scripts/bench_openai.py` | 一份性能对比表 |
| 3 | 差异分析：数字不同的原因——算子覆盖？显存带宽？调度？量化实现？ | 第 3 周的覆盖矩阵 | 一份原因分析 |
| 4 | 写报告：按「某年某月某版本实测」的格式，把环境、版本、命令、结果写全 | M0 的基线报告模板 | **M3 的可交付物：跨芯片对比报告（初稿）** |
| 5 | 小结：这份报告里哪些结论会随版本过期、哪些方法论不会 | — | 一页“保质期”说明 |

**硬件**：本周需要两边（NVIDIA + 至少一种国产芯片）的可运行环境；缺一边就先写好报告框架与方法论，实测部分标注“待补测”。

## 其他芯片的切入点

本模块以 Ascend 为主线，因为它的生态最成熟、资料最多。其他芯片的切入点供学完主线后扩展（状态以 [awesome-china-gpu](https://github.com/bytetora/awesome-china-gpu/blob/HEAD/README.md) 的最新追踪为准，不要照抄这里的版本信息）：

- **摩尔线程 MUSA**：vllm-musa 插件，另有 llama.cpp 官方支持
- **海光 DCU**：DTK（基于 ROCm），vLLM 容器镜像，迁移成本相对低
- **昆仑芯 XPU**：vllm-kunlun 插件
- **天数智芯 Iluvatar**：DeepSparkInference / vllm-iluvatar
- **壁仞 MetaX**：vllm-metax 插件
- **多芯片统一接入**：[vllm-plugin-FL](https://docs.flagos.io/projects/vllm-plugin-FL/en/latest/overview/features.html)（FlagGems 统一算子库 + FlagCX 统一通信库）

## 本模块检查清单

- [ ] 能画出国产芯片软件栈的分层图，并说清插件机制
- [ ] 同一模型在 Ascend 上跑通过（或明确标注了待补测的原因）
- [ ] 有一份算子覆盖矩阵，指出 Top 3 耗时算子的芯片侧状态
- [ ] 走过一次小规模移植流程并留下了记录
- [ ] 有一份按「某年某月某版本实测」格式写的跨芯片对比报告（初稿）
