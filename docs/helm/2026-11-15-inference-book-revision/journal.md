# 工作日志

## 2026-11-15 — L1 adjust

- evidence: 多个继承根上下文的新编辑代理在初始化时失败，未留下源稿改动；基线工作树检查仍只有计划与主代理新增测试。已成功工作的 benchmark-fix 保持其 M0 写域。
- change: 恢复此前成功的 inference-review 与 operator-chip-review；前者独占根 README，后者依次处理 M2 与 M3/M4。lead 继续独占生成器、导航、首页、统一约定和计划，不重试覆盖半成品。
- disposition: 保持 31 章与进度 ID，不改原英文、许可证或远端。暂存器改为可隔离测试的 helper，避免在代理写入期间运行整站 Red 构建；旧 cp -r 的泄露风险已由源码核对，新的临时项目夹具验证公开源稿保留、Helm/缓存排除和重定向删除拒绝。
- next: 所有源稿写域释放后统一生成；lead 审查实际差异和验证结果，不以代理“完成”直接勾验收。

## 2026-11-15 — note

- evidence: 原站点严格构建基线通过，MkDocs 构建耗时 1.45 秒。生成器新增测试先观察到 8 tests / 11 个预期断言失败，修订后同 8 tests 全部 Green；暂存 helper 缺失先产生 5 个预期失败，实现后同 5 tests 全部 Green。
- change: 新增快速上手和统一实验约定；导航、章节起步字段与术语链接改为保留短主句、提供可展开前置与附录。图解同步删除“重叠低于 max”和引擎 eager/编译二分。
- disposition: CPU mock 与数学/语法检查可以验收方法，不可替代 GPU/NPU 编译、路径和性能。实机内容继续待验证，客户端首次基线和正式稳态容量区分。
- next: 核对 M0 的最终 JSON/CLI 契约，整合各模块后跑全部回归和产物链接/编号检查。

## 2026-11-15 — note

- evidence: benchmark-fix 与 operator-chip-review 已释放全部写域。lead 复读客户端/mock/模板、完整算子参考源码与相关差异；独立复跑 M0 20 tests、M2/M3/M4 14 tests 全部 Green，生成器/暂存/链接工具的 17 tests 也全部 Green。尚未在修改后的源稿上执行统一网站构建。
- change: lead 整合时补齐 M3/M4 从仓库根目录执行的客户端路径、迁移 candidate 的显式导入，以及有限窗口/单指标不能直接归因的说明；修正 M2 服务部署前置只在第 24 章需要。新增产物 checker 检查笔记 URL 键、31 个进度 ID 和内部锚点。
- disposition: 保留原英文、许可证与笔记/进度 JS；仅图片文字与内容/构建源稿改动。CUDA/NPU/引擎参考仍未实机验证，语法/契约检查不等同于设备通过。
- next: 当前只有 inference-review 仍写根 README；等其释放后，lead 复核最终差异/关键公式，执行完整 unittest（主 Python 及现有 MkDocs Python）、bash site-build.sh、python3 -B tools/check_site.py 与 git diff --check。全部通过后按 REFERENCE.md 完成阶段/目标；未推送或部署。

## 2026-11-15 — note

- evidence: 全部写域释放，lead 已审查最终源码/文案差异并修复接口尾项、计时终点、参考入口与融合条件化表述；全仓 51 tests 在 Python 3.14/3.11 各通过，最后一轮回归/严格构建/产物检查/diff 空白检查再次通过。
- change: P1 的六个 chunk 与七项成功标准按观察结果验收，阶段和目标均完成；private artifacts 不进入公开产物。
- disposition: 保留原英文、许可证、原章节 URL、笔记键与进度协议；完整参考 GPU/NPU 实机项目仍为待验证，不编造结果。没有 Git 提交或远端发布。
- next: 用户可阅读新的快速上手与统一实验约定；后续如授权发布，提交/推送后由 GitHub Pages CI 更新线上。实机实验需另取得匹配设备与软件栈。

## 2026-11-15 — note

- evidence: 用户在 P1 验收后明确要求“提交，然后推送发布”。现有 Git 身份与 GitHub 登录均可用，origin 为 tdccccc/time-to-first-token，main 与最新 origin/main 无分歧。
- change: Commits 授权记录于 Constraints；本次是范围明确的交付动作，直接执行而不重开已验收的 P1 或新增规划阶段。
- disposition: 提交已验收的源稿、参考入口、测试、构建与计划日志；不包含 site/、site-src/ 或缓存，不使用 force push。
- next: 正常推送 origin/main，检查该提交的 GitHub Pages 工作流及线上新首页、快速上手与实验约定。
