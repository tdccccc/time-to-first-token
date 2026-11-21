#!/usr/bin/env python3
"""Check the generated book's IDs, titles, local links, figures and publication boundary.

This is an offline artifact check, not a browser interaction or device test.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent


class Page(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.references: list[str] = []
        self.chapters: list[int] = []
        self.heading: list[str] = []
        self.in_heading = False
        self.note_key: str | None = None
        self.figure_count = 0
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "meta" and attrs.get("name") == "notes-page":
            self.note_key = attrs.get("content")
        for name in ("href", "src"):
            if attrs.get(name):
                self.references.append(attrs[name])
        if tag == "input" and attrs.get("data-chapter"):
            self.chapters.append(int(attrs["data-chapter"]))
        if tag == "figure":
            self.figure_count += 1
        if tag == "h1":
            self.in_heading = True

    def handle_endtag(self, tag):
        if tag == "h1":
            self.in_heading = False

    def handle_data(self, data):
        if self.in_heading:
            self.heading.append(data)

    @property
    def title(self):
        return " ".join("".join(self.heading).replace("¶", "").split())


def check_links(site: Path, prefix: str = "/time-to-first-token/") -> list[str]:
    site = site.resolve()
    pages = {path.resolve(): Page(path.read_text(encoding="utf-8")) for path in site.rglob("*.html")}
    problems = []
    for path, page in pages.items():
        for reference in page.references:
            url = urlsplit(reference)
            if url.scheme or url.netloc or reference.startswith("//"):
                continue
            relative = unquote(url.path)
            if relative.startswith("/"):
                if relative == prefix.rstrip("/"):
                    relative = ""
                elif relative.startswith(prefix):
                    relative = relative[len(prefix):]
                else:
                    relative = relative.lstrip("/")
                target = site / relative
            else:
                target = path if not relative else path.parent / relative
            target = target.resolve()
            try:
                target.relative_to(site)
            except ValueError:
                problems.append(f"{path.relative_to(site)}: local link escapes site: {reference}")
                continue
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                problems.append(f"{path.relative_to(site)}: missing local target: {reference}")
                continue
            fragment = unquote(url.fragment)
            if fragment and target in pages and fragment not in pages[target].ids:
                problems.append(f"{path.relative_to(site)}: missing local anchor: {reference}")
    return problems


def validate_site(root: Path = ROOT) -> list[str]:
    root = root.resolve()
    site = root / "site"
    spec = importlib.util.spec_from_file_location("book_artifact_contract", root / "tools/build_book.py")
    book = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = book
    spec.loader.exec_module(book)
    problems = []

    for number, _part, title in book.CHAPTERS:
        path = site / f"docs/chapters/chapter-{number:02d}/index.html"
        if not path.is_file():
            problems.append(f"missing chapter {number}: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        page = Page(text)
        if page.title != f"第 {number} 章 {title}":
            problems.append(f"chapter {number}: title {page.title!r} != expected '第 {number} 章 {title}'")
        if page.note_key != f"docs/chapters/chapter-{number:02d}/":
            problems.append(f"chapter {number}: note key mismatch: {page.note_key!r}")
        if page.figure_count < 1:
            problems.append(f"chapter {number}: missing visual figure")
        if 'class="term-def"' in text:
            problems.append(f"chapter {number}: obsolete long automatic term annotation")

    required_pages = (
        "docs/topics/advanced-architectures/index.html",
        "docs/topics/cluster-pd-mooncake/index.html",
        "docs/topics/edge-runtimes/index.html",
        "docs/topics/paper-notes/index.html",
        "docs/topics/domestic-ecosystems/index.html",
        "docs/toolkit/hardware-models/index.html",
        "docs/toolkit/methodology/index.html",
        "docs/toolkit/tools-and-configs/index.html",
        "docs/glossary/index.html",
        "docs/progress/index.html",
        "docs/colophon/index.html",
    )
    for rel in required_pages:
        if not (site / rel).is_file():
            problems.append(f"missing public page: {rel}")

    progress = site / "docs/progress/index.html"
    if progress.is_file():
        expected_ids = [row[0] for row in book.CHAPTERS]
        if Page(progress.read_text(encoding="utf-8")).chapters != expected_ids:
            problems.append("reading progress chapter IDs mismatch")

    glossary = site / "docs/glossary/index.html"
    if glossary.is_file():
        anchors = Page(glossary.read_text(encoding="utf-8")).ids
        missing = {entry["id"] for entry in book.TERM_GLOSSARY} - anchors
        if missing:
            problems.append(f"missing glossary anchors: {sorted(missing)}")

    for tree in (root / "site-src", site):
        if (tree / "docs/helm").exists():
            problems.append(f"internal Helm documents leaked into {tree.name}")
        if any(tree.rglob("*.pyc")) or any(tree.rglob("__pycache__")):
            problems.append(f"Python caches leaked into {tree.name}")

    config = (root / "mkdocs.yml").read_text(encoding="utf-8")
    match = re.search(r"^site_url:\s*(\S+)", config, re.MULTILINE)
    prefix = urlsplit(match.group(1)).path if match else "/"
    problems.extend(check_links(site, prefix))
    return problems


def main() -> None:
    problems = validate_site()
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        raise SystemExit(1)
    print("PASS: 23 chapters (with figures), 5 topics, toolkit, glossary anchors, local links and public-only artifacts.")


if __name__ == "__main__":
    main()
