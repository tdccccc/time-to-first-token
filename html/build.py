#!/usr/bin/env python3
"""把学习计划的 Markdown 内容转成一份 HTML 版（放在 html/ 目录）。

用法：
    python3 html/build.py

说明：
- 源文件是仓库里的 md（README、MODULES、progress、各模块 README 与报告模板），
  md 是唯一事实来源；HTML 只是生成物，改内容请改 md，再运行本脚本重新生成。
- 内部的 .md 链接会改写成对应的 .html；其他相对链接（.py、图片等）会改写成
  从 html/ 出发的正确相对路径；外链不动。
- 需要 python-markdown 与 pygments；脚本会优先用同目录下的 venv
  （../.venv-html-build），没有就现场建一个并安装依赖。

输出：
- html/index.html            目录页
- html/README.html 等 9 个内容页
"""
import html as html_mod
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HTML_DIR = REPO_ROOT / "html"
VENV_DIR = REPO_ROOT.parent / ".venv-html-build"


def ensure_markdown():
    """保证能 import markdown 与 pygments，否则用 venv（没有就建一个）。"""
    try:
        import markdown  # noqa: F401
        import pygments  # noqa: F401
        return
    except ImportError:
        pass
    venv_python = VENV_DIR / "bin" / "python"
    if venv_python.exists() and os.path.abspath(sys.executable) != os.path.abspath(str(venv_python)):
        os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    if not venv_python.exists():
        print("正在创建 venv 并安装 markdown/pygments ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "--quiet", "markdown", "pygments"])
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


ensure_markdown()

import markdown  # noqa: E402
from pygments.formatters import HtmlFormatter  # noqa: E402

# 源文件 -> (输出文件名, 导航名, 一句话介绍)
PAGES = [
    ("README.md", "README.html", "主路线图", "主路线图：10 周核心 + 第 8 周（续）与第 11 周两处可选加深"),
    ("docs/MODULES.md", "MODULES.html", "模块地图", "模块地图：M0–M4 五个模块分别是什么、在哪、什么状态"),
    ("progress.md", "progress.html", "学习进度", "学习进度：60 节课的打勾清单"),
    ("docs/modules/M0-foundations/README.md", "M0-foundations.html", "M0 · 基础与度量", "M0：基础与度量（5 节）+ benchmark 脚本"),
    ("docs/modules/M0-foundations/baseline-report-template.md", "M0-baseline-report-template.html", "M0 · 基线报告模板", "M0：基线报告模板"),
    ("docs/modules/M2-operators/README.md", "M2-operators.html", "M2 · 算子优化", "M2：算子优化（6 周 × 5 节）"),
    ("docs/modules/M3-domestic-chips/README.md", "M3-domestic-chips.html", "M3 · 国产芯片适配", "M3：国产芯片适配（6 周 × 5 节）"),
    ("docs/modules/M3-domestic-chips/cross-chip-report-template.md", "M3-cross-chip-report-template.html", "M3 · 跨芯片报告模板", "M3：跨芯片对比报告模板"),
    ("docs/original-README.md", "original-README.html", "上游英文原文", "上游英文原文（存档，未翻译）"),
]

# 输出文件名 -> 源文件（用于改写 .md 内部链接）
MD_TO_HTML = {src: out for src, out, _, _ in PAGES}


def html_name_for_md_target(target_path: Path) -> str | None:
    """给一个仓库内的 .md 路径找对应的 html 输出名；不在映射里返回 None。"""
    try:
        rel = target_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None
    return MD_TO_HTML.get(rel)


def rewrite_relative_links(html_text: str, src_path: Path) -> str:
    """改写生成 HTML 里的相对链接，使其在 html/ 目录下仍然正确。

    - 相对 .md 链接 -> 对应的 .html（在映射里的才改写）
    - 其他相对链接（.py、图片等）-> 从 html/ 出发的相对路径
    - 外链（http/https/# /mailto:）不动
    """
    src_dir = src_path.parent

    def repl(match):
        attr, href = match.group(1), match.group(2)
        if href.startswith(("http://", "https://", "#", "mailto:", "/", "data:")):
            return match.group(0)
        path_part, sep, anchor = href.partition("#")
        if not path_part:
            return match.group(0)
        target = (src_dir / path_part).resolve()
        if path_part.endswith(".md"):
            name = html_name_for_md_target(target)
            if name is None:
                return match.group(0)
            new_href = name + (sep + anchor if sep else "")
            return f'{attr}="{new_href}"'
        try:
            rel = os.path.relpath(target, HTML_DIR)
        except ValueError:
            return match.group(0)
        return f'{attr}="{rel}{sep + anchor if sep else ""}"'

    return re.sub(r'(href|src)="([^"]+)"', repl, html_text)


PYGMENTS_CSS = HtmlFormatter(style="default").get_style_defs(".codehilite")

CSS = (
    PYGMENTS_CSS
    + """
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;color:#1c1c1c;line-height:1.75;background:#fff}
a{color:#1a66cc;text-decoration:none}
a:hover{text-decoration:underline}
header.site{background:#f7f7f4;border-bottom:1px solid #e3e3de;padding:14px 0 0}
header.site .inner{max-width:880px;margin:0 auto;padding:0 20px}
header.site h1{font-size:20px;margin:0 0 10px}
header.site h1 a{color:#1c1c1c}
nav{display:flex;flex-wrap:wrap;gap:4px 16px;padding-bottom:12px;font-size:14px}
nav a{color:#555}
nav a.current{color:#1a66cc;font-weight:600}
main{max-width:880px;margin:0 auto;padding:8px 20px 60px}
h1{font-size:28px;line-height:1.3;border-bottom:1px solid #e8e8e3;padding-bottom:10px}
h2{font-size:22px;border-bottom:1px solid #eee;margin-top:36px}
h3{font-size:18px;margin-top:28px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em;background:#f4f4f1;padding:1px 5px;border-radius:4px}
pre{overflow-x:auto;background:#f7f7f4;padding:14px 16px;border-radius:6px;line-height:1.55}
pre code{background:none;padding:0}
.codehilite pre{background:#f7f7f4}
table{border-collapse:collapse;margin:16px 0;display:block;overflow-x:auto}
th,td{border:1px solid #ddd;padding:8px 12px;text-align:left;font-size:15px}
th{background:#f7f7f4}
blockquote{border-left:4px solid #ddd;color:#555;margin:16px 0;padding:4px 16px}
img{max-width:100%}
hr{border:none;border-top:1px solid #e8e8e3;margin:32px 0}
footer{max-width:880px;margin:0 auto;padding:0 20px 40px;color:#888;font-size:13px}
ul,ol{padding-left:24px}
li{margin:4px 0}
"""
)


def nav_html(current: str) -> str:
    items = [("index.html", "目录")]
    items += [(out, name) for _, out, name, _ in PAGES]
    items.append(("../index.html", "进度打卡 ↗"))
    parts = []
    for href, label in items:
        cls = ' class="current"' if href == current else ""
        parts.append(f'<a href="{href}"{cls}>{label}</a>')
    return "<nav>" + "".join(parts) + "</nav>"


def page_shell(title: str, body_html: str, current: str, src_rel: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_mod.escape(title)} · AI Infra 学习计划</title>
<style>{CSS}</style>
</head>
<body>
<header class="site"><div class="inner">
<h1><a href="index.html">AI Infra 学习计划 · HTML 版</a></h1>
{nav_html(current)}
</div></header>
<main>
{body_html}
</main>
<footer>本页由 Markdown 生成，源文件：<a href="../{src_rel}">{src_rel}</a>。请勿直接改 HTML，改 md 后运行 <code>python3 html/build.py</code> 重新生成。</footer>
</body>
</html>
"""


def first_h1(md_text: str, fallback: str) -> str:
    for line in md_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def main():
    HTML_DIR.mkdir(exist_ok=True)
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"],
        extension_configs={"codehilite": {"css_class": "codehilite", "guess_lang": False}},
    )
    built = []
    for src_rel, out_name, nav_name, desc in PAGES:
        src_path = REPO_ROOT / src_rel
        text = src_path.read_text(encoding="utf-8")
        md.reset()
        body = md.convert(text)
        body = rewrite_relative_links(body, src_path)
        title = first_h1(text, nav_name)
        out_path = HTML_DIR / out_name
        out_path.write_text(page_shell(title, body, out_name, src_rel), encoding="utf-8")
        built.append((out_name, desc))

    # 目录页
    items = []
    for (_, out, name, desc) in PAGES:
        items.append(f'<li><a href="{out}"><strong>{html_mod.escape(name)}</strong></a>——{html_mod.escape(desc)}</li>')
    index_body = f"""<h1>AI Infra 学习计划 · HTML 版</h1>
<p>一份 10 周、每天 60 分钟的中文路线图：边做边学 LLM 推理服务。50 节核心课，外加第 8 周（续）大规模集群 PD 优化与第 11 周 TensorRT-LLM 两处可选加深，另有 M0–M4 五个模块。内容与仓库里的 Markdown 完全一致，只是换成了能在浏览器里直接读的样子。</p>
<ul>
{''.join(items)}
</ul>
<h2>怎么用</h2>
<ul>
<li>按顺序读：先看<a href="README.html">主路线图</a>，再按模块读 M0、M2、M3。</li>
<li>打卡进度用根目录的<a href="../index.html">进度打卡页</a>（浏览器本地保存）。</li>
<li>内容改动请改 md，改完运行 <code>python3 html/build.py</code> 重新生成本目录。</li>
</ul>
"""
    index_path = HTML_DIR / "index.html"
    index_path.write_text(
        page_shell("目录", index_body, "index.html", "html/build.py").replace(
            "本页由 Markdown 生成，源文件：<a href=\"../html/build.py\">html/build.py</a>。",
            "本页由 <a href=\"build.py\">html/build.py</a> 生成",
        ),
        encoding="utf-8",
    )
    print(f"已生成 {len(built)} 个内容页 + 目录页：")
    for out_name, desc in built:
        print(f"  html/{out_name}  ({desc})")
    print("  html/index.html  (目录页)")


if __name__ == "__main__":
    main()
