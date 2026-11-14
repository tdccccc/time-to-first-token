#!/usr/bin/env python3
"""Generate the book-style MkDocs source tree from the repo's Markdown.

site-build.sh stages the source files into site-src/ and then runs this script:

- README.md                         -> docs/m1/ overview + 13 week pages + appendices
- docs/modules/M0-foundations/README.md -> overview + 5 section pages
- docs/modules/M2-operators/README.md   -> overview + 6 week pages
- docs/modules/M3-domestic-chips/README.md -> overview + 6 week pages
- docs/MODULES.md links are rewritten for the split layout
- docs/catalog.md and index.md are generated

The Markdown files in the repo stay the single source of truth; this script only
produces the staged tree under site-src/ (gitignored).
"""
from __future__ import annotations

import posixpath
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site-src"

SPLIT_MODULES = {"M0-foundations", "M2-operators", "M3-domestic-chips"}

FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
LINK_RE = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)(\))")


@dataclass
class Heading:
    level: int
    title: str
    line: int


@dataclass
class Page:
    title: str
    link: str
    sections: list[str] = field(default_factory=list)


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


def section_end(headings: list[Heading], idx: int, total: int) -> int:
    level = headings[idx].level
    for other in headings[idx + 1:]:
        if other.level <= level:
            return other.line
    return total


def take(lines: list[str], headings: list[Heading], idx: int) -> list[str]:
    return lines[headings[idx].line:section_end(headings, idx, len(lines))]


def strip_blank(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def relevel(lines: list[str], base_level: int) -> list[str]:
    """Shift ATX headings so the first level becomes H1."""
    delta = base_level - 1
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
        if fence is None:
            hm = HEADING_RE.match(line)
            if hm:
                level = max(1, len(hm.group(1)) - delta)
                line = "#" * level + " " + hm.group(2)
        out.append(line)
    return out


def map_repo_path(path: str) -> str:
    if path == "README.md":
        return "docs/m1/index.md"
    m = re.fullmatch(r"docs/modules/([^/]+)/README\.md", path)
    if m and m.group(1) in SPLIT_MODULES:
        return f"docs/modules/{m.group(1)}/index.md"
    if path in ("tracker", "tracker/index.html"):
        return "tracker/index.html"
    return path


def rewrite_links(text: str, source_dir: str, target_dir: str) -> str:
    """Rewrite local links for a page that moved to target_dir."""
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


def find_heading(headings: list[Heading], title: str, level: int = 2) -> int:
    for i, h in enumerate(headings):
        if h.level == level and h.title == title:
            return i
    raise SystemExit(f"missing heading: {title!r} (level {level})")


def week_is_optional(title: str) -> bool:
    return "第 8 周（续）" in title or title.startswith("第 11 周")


def lesson_count(sections: list[str]) -> int:
    return len([s for s in sections if "论文精读" not in s])


def build_m1() -> dict:
    src = SITE / "README.md"
    lines = read_lines(src)
    hs = scan(lines)
    h1 = next(h for h in hs if h.level == 1)
    h2s = [h for h in hs if h.level == 2]
    route_i = find_heading(hs, "路线图")
    tracking_i = find_heading(hs, "跟踪进度")
    route = hs[route_i]
    tracking = hs[tracking_i]
    first_h2 = h2s[0]

    intro = strip_blank(lines[h1.line + 1:first_h2.line])
    pre_route = lines[first_h2.line:route.line]
    week_hs = [h for h in hs if h.level == 3 and route.line < h.line < tracking.line]
    if not week_hs:
        raise SystemExit("M1: no week headings found")

    weeks: list[Page] = []
    for i, h in enumerate(week_hs):
        end = week_hs[i + 1].line if i + 1 < len(week_hs) else tracking.line
        raw = lines[h.line:end]
        sections = [x.title for x in scan(raw) if x.level == 4]
        block = relevel(raw, 3)
        fname = f"week-{i + 1:02d}.md"
        write(f"docs/m1/{fname}", rewrite_links("\n".join(block), ".", "docs/m1"))
        weeks.append(Page(h.title, fname, sections))

    intro_r = rewrite_links("\n".join(intro), ".", "docs/m1")
    pre_r = rewrite_links("\n".join(pre_route), ".", "docs/m1")

    rows = []
    for w in weeks:
        kind = "可选加深" if week_is_optional(w.title) else "主线"
        n = lesson_count(w.sections)
        rows.append(f"| [{w.title}]({w.link}) | {n} | {kind} |")

    overview = [
        "# M1 · 11 周推理框架路线图",
        "",
        "> [← 模块地图](../MODULES.md) · [课程总目录](../catalog.md)",
        "",
        intro_r,
        "",
        pre_r,
        "",
        "## 路线图",
        "",
        "| 周 | 节数 | 类型 |",
        "|---|---:|---|",
        *rows,
        "",
        "## 附录",
        "",
        "- [跟踪进度的三种方式](tracking.md)",
        "- [发表 benchmark 之前](benchmark-checklist.md)",
        "- [决策点](decision-points.md)",
        "- [参考资料](references.md)",
        "- [来源、贡献与许可](colophon.md)",
    ]
    write("docs/m1/index.md", "\n".join(overview))

    appendix: list[Page] = []

    def appendix_page(source_title: str, fname: str, new_title: str | None = None,
                      notice: str | None = None) -> None:
        idx = find_heading(hs, source_title)
        block = relevel(take(lines, hs, idx), 2)
        if new_title:
            block[0] = "# " + new_title
        if notice:
            block = block[:1] + ["", notice, ""] + block[1:]
        write(f"docs/m1/{fname}", rewrite_links("\n".join(block), ".", "docs/m1"))
        appendix.append(Page(new_title or source_title, fname))

    appendix_page(
        "跟踪进度", "tracking.md", "跟踪进度的三种方式",
        "> **在网站上**：网页里的勾选不会同步到 GitHub。要保存进度，用[进度打卡页](../../tracker/index.html)（浏览器本地保存），或按下面的方式在 GitHub issue 里勾选。",
    )
    appendix_page("发表 benchmark 之前", "benchmark-checklist.md")
    appendix_page("决策点", "decision-points.md")
    appendix_page("参考资料", "references.md")

    colophon: list[str] = ["# 来源、贡献与许可", ""]
    for title in ("关于来源的说明", "参与贡献", "许可与出处"):
        idx = find_heading(hs, title)
        colophon += strip_blank(take(lines, hs, idx)) + [""]
    write("docs/m1/colophon.md", rewrite_links("\n".join(colophon), ".", "docs/m1"))
    appendix.append(Page("来源、贡献与许可", "colophon.md"))

    src.unlink()
    return {"weeks": weeks, "appendix": appendix}


def build_module(slug: str, module_title: str) -> dict:
    src = SITE / f"docs/modules/{slug}/README.md"
    lines = read_lines(src)
    hs = scan(lines)
    h1 = next(h for h in hs if h.level == 1)
    weeks = [h for h in hs if h.level == 2 and re.match(r"第\s*\d+\s*周", h.title)]
    if not weeks:
        raise SystemExit(f"{slug}: no week headings found")

    first = weeks[0]
    pre = strip_blank(lines[h1.line + 1:first.line])
    last_idx = hs.index(weeks[-1])
    post = strip_blank(lines[section_end(hs, last_idx, len(lines)):])

    pages: list[Page] = []
    for i, h in enumerate(weeks):
        end = weeks[i + 1].line if i + 1 < len(weeks) else section_end(hs, hs.index(weeks[-1]), len(lines))
        raw = lines[h.line:end]
        sections = [
            x.title for x in scan(raw)
            if x.level == 3 and re.match(r"第\s*\d+\s*节", x.title)
        ]
        block = relevel(raw, 2)
        fname = f"week-{i + 1:02d}.md"
        write(f"docs/modules/{slug}/{fname}", "\n".join(block))
        pages.append(Page(h.title, fname, sections))

    rows = [f"| [{w.title}]({w.link}) | {len(w.sections)} |" for w in pages]
    pre_text = "\n".join(pre)
    has_plan = any(
        x.level == 2 and x.title.startswith("课程安排") for x in scan(pre)
    )
    overview = [
        lines[h1.line],
        "",
        "> [← 模块地图](../../MODULES.md)",
        "",
        pre_text,
        "",
    ]
    if not has_plan:
        overview += ["## 课程安排", ""]
    overview += [
        "| 周 | 节数 |",
        "|---|---:|",
        *rows,
        "",
        "\n".join(post),
    ]
    write(f"docs/modules/{slug}/index.md", "\n".join(overview))
    src.unlink()
    return {"pages": pages, "title": module_title}


def build_m0() -> dict:
    src = SITE / "docs/modules/M0-foundations/README.md"
    lines = read_lines(src)
    hs = scan(lines)
    h1 = next(h for h in hs if h.level == 1)
    sections = [h for h in hs if h.level == 2 and re.match(r"第\s*\d+\s*节", h.title)]
    if not sections:
        raise SystemExit("M0: no section headings found")

    pre = strip_blank(lines[h1.line + 1:sections[0].line])
    last_idx = hs.index(sections[-1])
    post = strip_blank(lines[section_end(hs, last_idx, len(lines)):])

    pages: list[Page] = []
    for i, h in enumerate(sections):
        end = sections[i + 1].line if i + 1 < len(sections) else section_end(hs, last_idx, len(lines))
        raw = lines[h.line:end]
        subs = [x.title for x in scan(raw) if x.level == 3]
        block = relevel(raw, 2)
        fname = f"section-{i + 1}.md"
        write(f"docs/modules/M0-foundations/{fname}", "\n".join(block))
        pages.append(Page(h.title, fname, subs))

    rows = [f"| [{s.title}]({s.link}) |" for s in pages]
    overview = [
        lines[h1.line],
        "",
        "> [← 模块地图](../../MODULES.md)",
        "",
        "\n".join(pre),
        "",
        "## 课程安排",
        "",
        "| 节 |",
        "|---|",
        *rows,
        "",
        "\n".join(post),
    ]
    write("docs/modules/M0-foundations/index.md", "\n".join(overview))
    src.unlink()
    return {"pages": pages, "title": "M0 · 基础与度量"}


def rewrite_modules_page() -> None:
    path = SITE / "docs/MODULES.md"
    text = path.read_text(encoding="utf-8")
    path.write_text(rewrite_links(text, "docs", "docs") + "\n", encoding="utf-8")


def build_catalog(m0: dict, m1: dict, m2: dict, m3: dict) -> None:
    out: list[str] = [
        "# 课程总目录",
        "",
        "全部模块与页面，按学习顺序排列。左侧目录用于跳转，这一页适合一次看完。",
        "",
        "## M0 · 基础与度量",
        "",
        "1 周，5 节。无卡可做。",
        "",
    ]
    for page in m0["pages"]:
        out.append(f"- [{page.title}](modules/M0-foundations/{page.link})")
    out += ["", "## M1 · 11 周推理框架路线图", "", "55 节核心 + 10 节可选加深。", ""]
    for page in m1["weeks"]:
        kind = " · 可选" if week_is_optional(page.title) else ""
        out.append(f"### [{page.title}](m1/{page.link}){kind}")
        out.append("")
        for sec in page.sections:
            out.append(f"- {sec}")
        out.append("")
    out += ["### 附录", ""]
    for page in m1["appendix"]:
        out.append(f"- [{page.title}](m1/{page.link})")
    out += ["", "## M2 · 算子优化", "", "6 周，30 节。", ""]
    for page in m2["pages"]:
        out.append(f"### [{page.title}](modules/M2-operators/{page.link})")
        out.append("")
        for sec in page.sections:
            out.append(f"- {sec}")
        out.append("")
    out += ["## M3 · 国产芯片适配", "", "6 周，30 节。", ""]
    for page in m3["pages"]:
        out.append(f"### [{page.title}](modules/M3-domestic-chips/{page.link})")
        out.append("")
        for sec in page.sections:
            out.append(f"- {sec}")
        out.append("")
    out += [
        "## M4 · 综合项目",
        "",
        "- [跨芯片推理优化报告（占位页）](modules/M4-capstone/README.md)",
        "",
    ]
    write("docs/catalog.md", "\n".join(out))


def main() -> None:
    m0 = build_m0()
    m1 = build_m1()
    m2 = build_module("M2-operators", "M2 · 算子优化")
    m3 = build_module("M3-domestic-chips", "M3 · 国产芯片适配")
    rewrite_modules_page()
    build_catalog(m0, m1, m2, m3)
    shutil.copyfile(ROOT / "tools" / "homepage.md", SITE / "index.md")
    print(f"generated: M0 {len(m0['pages'])} sections, M1 {len(m1['weeks'])} weeks + "
          f"{len(m1['appendix'])} appendix, M2 {len(m2['pages'])}, M3 {len(m3['pages'])}, M4 1")


if __name__ == "__main__":
    main()
