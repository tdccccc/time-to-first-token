# A40 + Qwen3.5 试点配置（内部事实表）

本文件是第 1/5/8 章改写的唯一数字来源。所有正文数字先在这里算清，再写进章节；估算值必须标“估算/算例”，不能写成实测。

## 编辑策略

- **范围**：第 1、5、8 章 + 环境准备 + quickstart/experiments 的配置说明。不要动第 14/15/18 章的多卡/FP8 结论。
- **双轨**：Llama-3.1-8B 保留为“最简 GQA 教学算例”；Qwen3.5-9B 是默认运行模型；Qwen3.5-4B 是 24GB/快速迭代配置。
- **保留对照**：A100/H100 的数字不要删，作为对照列/对照句保留；A40 是单卡默认算例。
- **风格**：中文正文、章节编号、标题、链接、术语表标注保持不变；只改事实、命令和算例。
- **不运行**：不要跑 site-build、git 写操作或大范围重排；由主会话统一验证。

## A40 公开规格（本机实测可见值）

| 项 | A40 | A100 80GB（对照） |
|---|---|---|
| 显存 | 48 GB GDDR6 ECC；可见 44.42 GiB | 80 GB |
| 带宽 | 696 GB/s | ~2,039 GB/s（书里常写 ~2 TB/s） |
| SM | 84 | 108 |
| FP16 Tensor 稠密 | 74.8 TFLOPS（稀疏 149.7） | 312 TFLOPS |
| FP32 / TF32 稠密 | 37.4 / 37.4 TFLOPS | 19.5 / 156 TFLOPS |
| INT8 Tensor 稠密 | 149.7 TOPS | 624 TOPS |
| shared / L2 / 线程每 SM | 100 KiB / 6 MiB / 1536 | 164 KiB / 40 MiB / 2048 |
| compute capability | 8.6 | 8.0 |
| Roofline 脊点（FP16 稠密） | 74.8e12 ÷ 696e9 ≈ **107.5 FLOP/B** | 312e12 ÷ 2,039e9 ≈ **153**（书里取整 156） |

- A40 单 block shared 上限：99 KiB（101,376 B opt-in）。
- FP8 W8A8 需要 compute capability ≥ 8.9；A40 是 8.6，A100 是 8.0，二者都走 FP8 W8A16（Marlin）/ AWQ / GPTQ，不能写“A40 原生 FP8 W8A8”。
- A40 是 GA102（84 SM、6 MiB L2、1536 线程/SM），不是“小一号 A100”；第 19–24 章的 tile/occupancy 要按 A40 重算。

## Qwen3.5-9B（本地 `~/models/Qwen3.5-9B`）

- HF 名：`Qwen/Qwen3.5-9B`；架构 `Qwen3_5ForConditionalGeneration`；文本服务加 `--language-model-only` 可跳过视觉塔。
- 32 层 = 8 层 full attention + 24 层 Gated DeltaNet（`full_attention_interval: 4`）。
- full attention：16 个 Q 头、4 个 KV 头、head_dim 256。
- LM 参数约 9.2B；LM 权重 **18.394 GB = 17.13 GiB**；视觉塔 0.912 GB；总计 19.306 GB。
- 每 token KV：`2 × 8 层 × 4 KV 头 × 256 头维 × 2 字节 = 32,768 B = 32 KiB`。
- 每请求固定状态：24 层 × 2 MiB FP32 = **48 MiB**；卷积槽位约 1.1–1.5 MiB；合计约 **49 MiB/请求**。
- 状态账：
  - 500 token：16 MiB KV + 49 MiB = **约 65 MiB**；
  - 8K：256 MiB + 49 MiB = **约 305 MiB**；
  - 32K：1 GiB + 49 MiB = **约 1.05 GiB**；
  - 128K：4 GiB + 49 MiB = **约 4.05 GiB**；
  - 256K：8 GiB + 49 MiB = **约 8.05 GiB**。
- A40 0.9 预算：`44.42 × 0.9 ≈ 40 GiB`；减 17.13 GiB 权重、约 2 GiB 工作区，KV 池约 **20.8 GiB**。
- 并发估算（0.9 预算、只算 KV+固定状态、忽略 block 对齐/激活/前缀缓存）：
  - 500 token：约 **330** 条；
  - 8K：约 **70** 条；
  - 32K：约 **20** 条；
  - 128K：约 **5** 条。
- 单请求 decode 权重搬运下界：`18.394 GB ÷ 696 GB/s = 26.4 ms/token ≈ 38 token/s`；计算下界 `18.394 GFLOP ÷ 74.8 TFLOPS = 0.246 ms`。
- 权重主导 prefill 下界：512 token `2 × 9.2e9 × 512 = 9.42 TFLOP ÷ 74.8 TFLOPS ≈ 126 ms`；8K `150.7 TFLOP ÷ 74.8 TFLOPS ≈ 2.01 s`。
- 原生上下文 262,144 token。

## Qwen3.5-4B（本地 `~/models/Qwen3.5-4B`）

- 与 9B 相同的注意力几何：32 层 = 8 full + 24 linear，4 KV 头、head_dim 256 → **同样 32 KiB/token、约 49 MiB 固定状态**。
- LM 参数约 4.33B；LM 权重 **8.653 GB = 8.06 GiB**；视觉塔 0.667 GB；总计 9.320 GB。
- A40 0.9 预算：KV 池约 **29.9 GiB**；500 token 约 **470** 条；8K 约 **100** 条；32K 约 **29** 条。
- 单请求 decode：`8.653 GB ÷ 696 GB/s = 12.4 ms/token ≈ 80 token/s`；prefill 512 `4.43 TFLOP ÷ 74.8 ≈ 59 ms`；8K `70.9 TFLOP ÷ 74.8 ≈ 0.95 s`。
- 24GB 卡（可见 22.35 GiB，0.9 = 20.1 GiB）：减 8.06 GiB 权重、约 2 GiB 工作区，KV 池约 **10 GiB**；8K 约 **33** 条；32K 约 **9** 条。

## 后续章节派生数字

- **Ch11**：640 token 请求 = 20 MiB KV + 49 MiB 状态 ≈ **69 MiB**；A40 20.8 GiB 池约 **308** 条；2560 token = 80 MiB + 49 ≈ **129 MiB**，约 **165** 条。11.4/11.5 的 640-token 常驻约 300、8K 约 70。
- **Ch12**：Qwen3.5-9B BF16 18.394 GB、FP8 约 9.20 GB、INT4 约 4.60 GB；Qwen3.5-4B BF16 8.653 GB、FP8 约 4.33 GB、INT4 约 2.16 GB。A40 的 `--quantization fp8` 只能走 W8A16（Marlin），W8A8 需 CC≥8.9。
- **Ch18**：A40 的默认量化对比是 FP16/BF16 18.39 GB vs W4A16 约 4.60 GB；TensorRT-LLM 对 Qwen3.5 的支持以当前模型列表为准，不支持就写“不支持/未验证”。
- **Ch19**：A40 FP32 脊点 = 37.4e12 ÷ 696e9 ≈ **53.7 FLOP/B**；AI=8 的参考线 = min(8×696, 37,400) ≈ **5.57 TFLOPS**，137.4 GFLOP ÷ 5.57 ≈ **24.7 ms**；越脊点的 BLOCK 阈值约 214.8 → **256**。
- **Ch21**：A40 fused attention N=1024、d=64 的计算下界 = 0.268 GFLOP ÷ 74.8 TFLOPS ≈ **3.58 µs**；朴素版 8.5 MiB 搬运下界 ≈ **12.8 µs**；AI=N/2 与脊点 107 的交点在 **N≈214**；N=256 时 AI=128，A40 已计算受限，A100（脊点 156）仍带宽受限。
- **Ch14/15**：单张 A40 不能同时驻留 P/D，两章 H100/A100 的 PD/集群数字只作架构与账本练习。
- **Ch16**：Qwen3.5-9B 单请求 decode 权重搬运下界 = 18.394 GB ÷ 696 GB/s ≈ **26.4 ms/token**；A100 2 TB/s 约 **9.2 ms/token**；同吞吐所需卡数按带宽比约 **2.9 倍**。

## 数字书写约定

- 权重/带宽用十进制 GB、GB/s；KV/状态用二进制 KiB/MiB/GiB；两者不混。
- 所有并发数标“估算”，所有性能数标“下界/算例”；不要写成实测。
- `gpu-memory-utilization 0.9` 是基线；实际 vLLM 还会按 profiling、block 对齐、激活峰值和 `max_num_seqs` 调整，正文要提示“以启动日志的 KV cache size 为准”。
- 不要把 Qwen3.5 的固定递推状态写成可由 block table 共享；PagedAttention 只共享 full attention 的 KV block。
