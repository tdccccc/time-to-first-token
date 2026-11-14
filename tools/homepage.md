# 边做边学 LLM 推理服务 { .home-title }

<p align="center">
  <img src="assets/banner.png" alt="边做边学 LLM 推理服务：交付一个真实的推理服务">
</p>

从 Roofline 到跨芯片对比：一本写给工程师的 LLM 推理基础设施书。全书 **5 部分、31 章**，围绕一个不断长大的 OpenAI 兼容推理服务展开。

[从第 1 章开始读](docs/chapters/chapter-01.md){ .md-button .md-button--primary }
[全书目录](docs/catalog.md){ .md-button }
[阅读进度](docs/progress.md){ .md-button }

## 全书结构

<div class="grid cards part-cards" markdown>

- **第一部分 基础与度量**

    ---

    第 1–5 章：Roofline、四个度量、profiling、压测脚本、第一份基线。

    [本部分导览](docs/parts/part-1.md) · [第 1 章](docs/chapters/chapter-01.md)

- **第二部分 推理框架**

    ---

    第 6–18 章：心智模型、vLLM、SGLang、量化、投机解码、解耦、路由器、发表。

    [本部分导览](docs/parts/part-2.md) · [第 6 章](docs/chapters/chapter-06.md)

- **第三部分 算子优化**

    ---

    第 19–24 章：CUDA/Triton、fused attention、量化 matmul、CUTLASS、接回引擎。

    [本部分导览](docs/parts/part-3.md) · [第 19 章](docs/chapters/chapter-19.md)

- **第四部分 国产芯片适配**

    ---

    第 25–30 章：软件栈地图、CANN/AscendC、跑通、算子覆盖、移植、精度报告。

    [本部分导览](docs/parts/part-4.md) · [第 25 章](docs/chapters/chapter-25.md)

- **第五部分 综合项目**

    ---

    第 31 章：同一模型、同一评测集，在两套硬件上端到端部署、压测、归因、写作。

    [本部分导览](docs/parts/part-5.md) · [第 31 章](docs/chapters/chapter-31.md)

</div>

## 这本书适合谁

熟悉 Python、了解 Transformer、会用命令行的工程师。不需要 serving、Kubernetes 或 CUDA 经验；也不要求你先成为 ML 研究员，只需要真的想把一套推理服务跑起来。

## 怎么读

按“部分”推进，章内小节按编号读：

- **第 1–5 章**补度量与基线，不需要 GPU 也能读完前三章。
- **第 6–11 章**是推理主线：心智模型、模型架构、vLLM、SGLang、度量、压测。
- **第 12–17 章**按需跳读：量化、投机解码、解耦、集群 PD、单位经济学、发表。
- **第 18 章**（TensorRT-LLM）与 **第 15 章**（大规模集群 PD）是选读。
- **第 19–24 章**写算子，**第 25–30 章**做国产芯片，**第 31 章**收口成跨芯片报告。

不必严格按顺序：第三部分可以跟第 12 章并行，第四部分最好放在第二部分之后。每一章的产出比“读完”重要。

## 这个站点怎么用

- **左侧是全书目录**：部分、章、节都可以折叠；当前章的小节目录也合并在这里。
- **右侧是笔记栏**：笔记默认自动保存在浏览器；配置 GitHub token 后可一键提交到本仓库的 `notes/` 目录。
- **顶部搜索**：搜正文、代码和标题。
- **每页底部**：上一章 / 下一章，按书的顺序走。

!!! warning "笔记的隐私边界"
    本仓库是公开的，同步到 `notes/` 的笔记也会公开。不要把 token、密码、内部地址写进笔记。只想自己看，请在“同步设置”里改用私有仓库。

## 配套

- [阅读进度](docs/progress.md)：勾选已读章节，导出或生成同步链接。
- [全书目录](docs/catalog.md)：一次看完 5 部分 31 章。
- [来源、贡献与许可](docs/backmatter/colophon.md)：上游项目、授权与贡献方式。
