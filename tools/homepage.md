# 边做边学 LLM 推理服务

<p align="center">
  <img src="assets/banner.png" alt="边做边学 LLM 推理服务：交付一个真实的推理服务">
</p>

一份给工程师的中文学习计划：11 周、每天 60 分钟，边做边学，最后交付一个 OpenAI 兼容的推理服务。主线之外还有 M0 基础与度量、M2 算子优化、M3 国产芯片适配和 M4 综合项目。

[从 M0 开始](docs/modules/M0-foundations/index.md){ .md-button .md-button--primary }
[进入 M1 路线图](docs/m1/index.md){ .md-button }
[课程总目录](docs/catalog.md){ .md-button }

## 五个模块

<div class="grid cards" markdown>

-   **M0 · 基础与度量**

    ---

    Roofline、TTFT/ITL/吞吐、profiling、可复现的压测脚本。

    **1 周 · 5 节** · ✅ 已上线

    [开始 M0](docs/modules/M0-foundations/index.md)

-   **M1 · 11 周路线图**

    ---

    vLLM → SGLang → 量化 → 投机解码 → 解耦部署 → 路由器 → 发表；另有 2 处可选加深。

    **11 周 · 55 节核心 + 10 节可选** · ✅ 已上线

    [进入 M1](docs/m1/index.md)

-   **M2 · 算子优化**

    ---

    CUDA 优化阶梯 → Triton → fused attention / 量化 matmul → FlashAttention/CUTLASS → 接入引擎。

    **6 周 · 30 节** · ✅ 已上线

    [进入 M2](docs/modules/M2-operators/index.md)

-   **M3 · 国产芯片适配**

    ---

    软件栈地图 → CANN/AscendC → Ascend 跑通 → 算子覆盖 → 移植 → 精度回归。

    **6 周 · 30 节** · ✅ 已上线

    [进入 M3](docs/modules/M3-domestic-chips/index.md)

-   **M4 · 综合项目**

    ---

    同一模型、同一评测集，在两套硬件上端到端部署、压测、归因、写作。

    **6 周** · 🚧 占位页

    [查看 M4](docs/modules/M4-capstone/README.md)

</div>

## 推荐路径

```text
M0（1 周，无卡可做）
 → M1（11 周主线 + 第 8 周（续）与第 11 周可选加深）
 → M2 / M3（各 6 周，可与 M1 后半段并行）
 → M4（毕业设计）
```

不必严格按顺序：M2 的量化 matmul 可以跟 M1 第 6 周并行，M3 最好放在 M1 之后。每一节的产出比“读完”重要。

## 这个站点怎么用

- **左侧是全书目录**：模块、周、节都可以折叠；当前页面的小节目录也合并在这里。
- **右侧是笔记栏**：笔记默认自动保存在浏览器；配置 GitHub token 后可一键提交到本仓库的 `notes/` 目录。
- **顶部搜索**：搜课程正文、代码和标题。
- **每页底部**：上一页 / 下一页，按书的顺序走。

!!! warning "笔记的隐私边界"
    本仓库是公开的，同步到 `notes/` 的笔记也会公开。不要把 token、密码、内部地址写进笔记。只想自己看，请在“同步设置”里改用私有仓库。

??? note "这个 fork 相对上游做了什么"

    - **中文版**：主路线图翻译成中文，英文原文见[上游英文原文](docs/original-README.md)。
    - **M0–M4 模块**：新增基础与度量、算子优化、国产芯片适配和综合项目，见[模块地图](docs/MODULES.md)。
    - **课本密度**：M0/M1/M2/M3 按每节 60 分钟重写，每节都有产出、硬件门槛和无卡替代路径。
    - **两处可选加深**：第 8 周（续）大规模集群 PD 优化、第 11 周 TensorRT-LLM；论文一律标为选读，正文给出总结。
    - **站点版**：把 1MB 级的 Markdown 拆成每周/每节页面，左侧书籍目录、右侧笔记，内容仍以仓库 Markdown 为唯一事实来源。

## 开始之前

先读[模块地图](docs/MODULES.md)和 [M1 路线图总览](docs/m1/index.md)的时间投入；如果你已经有 serving 经验，可以从 M1 第 2 周开始，把 M0 当查漏补缺。第一次部署前，先确认你有一张能租到的 GPU，以及一个能存代码和结果的仓库。
