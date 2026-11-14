---
name: Progress tracker
about: 按章跟踪 5 部分 31 章的阅读进度。Open this in your own fork.
title: '阅读进度: [your name or handle]'
labels: progress
---

> **在网站上**：站点里的[阅读进度](https://tdccccc.github.io/time-to-first-token/docs/progress/)可以勾选、导出 JSON、生成同步链接。这个 issue 模板是给 GitHub 用的静态副本；网页上的勾选不会写回仓库，issue 里的勾选由 GitHub 保存到你的账号。

论文是选读：正文对应章节已有详细总结，读不读论文原文不影响章节进度。

Started on:

## 第一部分 基础与度量

- [ ] 第 1 章 Roofline：先算，再优化
- [ ] 第 2 章 推理度量：四个数，一个分布
- [ ] 第 3 章 Profiling：看时间花在哪
- [ ] 第 4 章 写压测脚本：bench_openai.py
- [ ] 第 5 章 跑出第一份基线

## 第二部分 推理框架

- [ ] 第 6 章 心智模型
- [ ] 第 7 章 模型架构
- [ ] 第 8 章 vLLM：先部署，再读内部实现
- [ ] 第 9 章 度量基础设施
- [ ] 第 10 章 SGLang、RadixAttention 与批处理内部
- [ ] 第 11 章 压测到 1000 并发
- [ ] 第 12 章 量化取舍
- [ ] 第 13 章 投机解码与 KV 淘汰
- [ ] 第 14 章 解耦式服务与 Kubernetes
- [ ] 第 15 章 大规模集群 PD 优化（选读）
- [ ] 第 16 章 单位经济学与路由器
- [ ] 第 17 章 发表，然后养成阅读习惯
- [ ] 第 18 章 TensorRT-LLM：编译式推理引擎（选读）

## 第三部分 算子优化

- [ ] 第 19 章 CUDA 优化阶梯：一个 GEMM，四次提速
- [ ] 第 20 章 Triton 入门：用 Python 写 kernel
- [ ] 第 21 章 Fused Attention 深讲：先推导，再写 kernel
- [ ] 第 22 章 量化 Matmul 深讲：量化方案本身
- [ ] 第 23 章 FlashAttention 2/3 与 CUTLASS：读工业级实现
- [ ] 第 24 章 收口：把自写 kernel 接入推理引擎

## 第四部分 国产芯片适配

- [ ] 第 25 章 芯片与软件栈地图：先画地图，再谈适配
- [ ] 第 26 章 CANN 与 AscendC 深讲：算子从定义到编译到调用
- [ ] 第 27 章 同一模型在 Ascend 上跑通 + vllm-ascend 插件架构
- [ ] 第 28 章 算子覆盖分析
- [ ] 第 29 章 移植案例：把 Triton kernel 搬到 Ascend
- [ ] 第 30 章 精度回归与调优报告

## 第五部分 综合项目

- [ ] 第 31 章 跨芯片推理优化报告
