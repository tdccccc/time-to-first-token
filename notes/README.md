# 笔记

这个目录由站点右侧的“笔记”面板同步写入：每个页面一个 Markdown 文件，路径与站点页面一一对应。

- **公开性**：本仓库是公开仓库，提交到这里的笔记也会公开。不要写 token、密码、内部地址或其他敏感信息。
- **同步方式**：在笔记面板的“同步设置”里填入细粒度 GitHub token，权限只需本仓库的 **Contents: Read and write**。token 只保存在当前浏览器的 localStorage，不会提交到仓库。
- **没有 token 时**：笔记仍然会自动保存在浏览器本地，可以随时导出为 `.md`。
- **文件命名**：站点页面 `docs/chapters/chapter-06/` 对应 `notes/docs/chapters/chapter-06.md`；`docs/parts/part-2/` 对应 `notes/docs/parts/part-2.md`；`docs/progress/` 对应 `notes/docs/progress.md`；首页对应 `notes/index.md`。
- **旧版笔记**：之前按周页面保存的 `notes/docs/m1/week-01.md` 等文件仍然留在仓库里，但新站点不会自动读取；需要保留的，按新的章节路径复制一份。
- **不要手改同步格式**：笔记文件内容就是笔记正文，面板按纯 Markdown 读写；直接编辑文件也可以，下次打开页面会按需载入。

如果只想给自己看，建议把笔记同步到另一个**私有**仓库：在“同步设置”里把仓库改成 `你的用户名/私有仓库名`，并给 token 对应权限。
