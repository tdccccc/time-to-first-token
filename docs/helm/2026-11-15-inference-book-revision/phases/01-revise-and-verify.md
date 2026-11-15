# P1 — revise-and-verify

goal_ref: ../goal.md
created: 2026-11-15T17:22:25+08:00
updated: 2026-11-15T18:46:53+08:00
revision: 2

## Outcome

当前 31 章站点的技术纠错、快速动手路径和表达改进已落在权威源稿中，生成结果与测试一致，未验证的实机内容没有被升级为实测结论。

## Assumptions

- 当前工作树无用户未提交改动；严格构建基线已实际通过（2026-11-15，构建耗时 1.45 秒）。
- 本轮不依赖 GPU/NPU 环境。无卡可验证接口解析、统计脚手架、数学护栏、源码/页面一致性；硬件能力和性能依旧以引用与待验证状态呈现。
- 章节编号保持稳定，新增入口和附录不占用原章节 ID。

## Approach

四个互不重叠的源稿模块并行修改，主代理同时改善生成器与阅读入口。子代理不构建网站、不提交、不继续委派；所有模块完成后统一审查、生成与回归。

## Chunks

### Chunk 1 — 修正压测行为与基础章节

- change: fail-fast 拒绝请求数不足的并发扫描，保留原始请求记录；提供完整 mock SSE 入口，说明 chunk/token 与有限工作负载的边界。
- kind: bug fix
- first signal: 新增标准库 unittest，原脚本因缺少并发校验/请求记录而按预期失败。
- checks: 同一用例 Green；角色空 chunk、usage、失败/截断流、有限工作负载与 CLI 的 SSE 回归通过。
- [x] accepted on observed evidence

### Chunk 2 — 修正推理框架事实与关联算例

- change: 修正 H100 FP16/TF32/稀疏口径和所有依赖数字，纠正重叠下界、投机采样和统计、当前后端二分、PD 路径与过载测量表述；压缩重复定义并保持原引用。
- kind: non-behavioral
- first signal: n/a
- checks: 源稿差异与关联公式复算；旧错误表述检索；章节块计数和引用保持有效。
- [x] accepted on observed evidence

### Chunk 3 — 修正算子数值、入口与验收

- change: FP32 scale 护栏、完整 INT8 host wrapper、四版 GEMM 参考入口、插件裸进程显式注册；不把 DRAM%/occupancy 当因果证据。
- kind: bug fix
- first signal: FP16 1e-8 下溢的最小复现；插件注册示例的导入不触发验证按预期失败。未具备 GPU 栈时用标准库/静态提取核对，实机验证保留待测。
- checks: 护栏与完整性检查 Green；Python 片段语法检查；实际可用工具的编译检查，未运行项目明确记录。
- [x] accepted on observed evidence

### Chunk 4 — 修正国产迁移与综合项目闭环

- change: MetaX 名称、迁移设计定位、版本/平台适用边界；配对质量回归和优化后重测；算例证据统一 D，未验证归因改为候选。
- kind: non-behavioral
- first signal: n/a
- checks: 源稿差异与证据表核对；参考与状态一致；报告复用和实测/示意边界明确。
- [x] accepted on observed evidence

### Chunk 5 — 改善首页、快速入口与术语生成

- change: 首屏先写成果、三路径与设备门槛；新增 CPU mock 快速上手和统一实验约定；章节起步卡；术语链接取代长括号，保持代码和已有定义不变，统一标题。
- kind: behavior
- first signal: 新增生成器测试，原 glossary_annotation 返回长 span 而非术语链接，按预期 Red。
- checks: 生成器测试 Green；31 章及进度 ID、标题与锚点一致；术语附录与新入口都注册导航。
- [x] accepted on observed evidence

### Chunk 6 — 整合构建与发布边界

- change: 统一生成，内部 Helm 文档和缓存不进入 staging/公开产物；为公开构建加入内容回归检查。
- kind: behavior
- first signal: 新增 staging/构建一致性回归，对原复制内部计划的行为按预期 Red；在无需网络的范围测试。
- checks: python3 -B -m unittest discover -s tests -v；bash site-build.sh；git diff --check；生成结构/链接/证据状态与最终差异人工复核。
- [x] accepted on observed evidence

## Phase verification

- 所有代理结束写入之后，由 lead 单独运行全部 unittest 与严格构建。
- 新首页、快速上手、统一实验约定、术语表和 31 章页面均存在；章节编号、进度 data-chapter 和旧 URL 不变。
- 生成站点不包含 docs/helm/ 或 __pycache__/；现有笔记/进度功能的源码不改。
- 最终 diff 无冲突标记、无编造实测成绩，待 GPU/NPU 核验的内容仍明确标注。

## Abort / reshape triggers

- 如遇正在写入的用户改动或代理写域交叉，先停止覆盖并协调。
- 如缺少硬件导致某验证不可执行，保留未验证状态，不安装大型栈或用 mock 数据替代。
- 如生成器重排破坏章节编号、引用或现有进度协议，先修正再构建，不发布带缺口的产物。

## Accepted evidence

- lead 在所有代理释放写域后运行完整回归：Python 3.14.4 的 51 tests 全部 Green；现有 MkDocs Python 3.11.15 的同 51 tests 全部 Green。最后一轮融合文案修订后再次运行 51 tests、严格构建、产物检查与 diff 空白检查，均通过。
- 完整 mock CLI、异常 SSE、并发 fail-fast、逐请求记录和可空 TPOT 均由实际回归验证；数学/静态算子检查不声称是设备验证。
- lead 独立复算 H100 脊点 295.373134、1000-token prefill 32.460839 ms、P/D 卡时与整数配卡 9:6、17:1；几何条件 alpha 与累计 beta 分开。
- bash site-build.sh 严格构建通过；tools/check_site.py 检查 31 章、标题、起步卡、术语锚点、全部本地链接、笔记 URL 键和进度 ID，通过。真实 HTML 中紧凑术语链接已解析为 a.term-ref，而非原样 Markdown。
- git diff --check 通过；LICENSE、原英文、notes/progress JS 和笔记 meta 模板与 HEAD 一致。内部 Helm 文档/缓存未进入 staging 或公开产物。
- test-first 例外与补偿：旧 staging 内联于整站脚本，与并行写入的源稿耦合，未为旧 cp -r 行为并行执行整站泄露 Red。旧风险经源码核对；新 helper 缺失的预期 Red、临时项目隔离/拒绝重定向删除的 Green 和真实产物无泄露检查共同验证目标边界。
- 未执行项：缺 torch/triton/torch_npu、NVCC、NPU 环境，本轮未做 CUDA/NPU JIT、实际引擎分发、任务精度、PD/RDMA 或 GPU/NPU 性能。目标范围只接受完整参考与诚实待测标记，不将缺失项勾为硬件通过。
- 未提交、未推送、未部署；本地站点和权威源稿已完成本轮修订。
