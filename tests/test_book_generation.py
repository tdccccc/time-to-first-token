"""Regression checks for compact glossary annotations and stable book IDs.

Only Python's standard library is required; these tests never rebuild the site
or touch another agent's source files.
"""
import importlib.util
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("book_generation_under_test", ROOT / "tools/build_book.py")
BOOK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOOK
SPEC.loader.exec_module(BOOK)


class BookGenerationTests(unittest.TestCase):
    def entry(self, key):
        return next(item for item in BOOK.TERM_GLOSSARY if item["id"] == key)

    def test_compact_annotations_link_to_glossary(self):
        annotation = BOOK.glossary_annotation("TTFT", self.entry("ttft"))
        self.assertIn("[TTFT](../glossary.md#ttft)", annotation)
        self.assertNotIn("term-def", annotation)
        self.assertNotIn("<span", annotation)
        self.assertIn("title=", annotation)

    def test_annotates_only_first_prose_use(self):
        output = "\n".join(BOOK.TermAnnotator().annotate_lines([
            "TTFT 决定首次反馈。", "比较 TTFT 时要固定条件。"
        ]))
        self.assertEqual(output.count("[TTFT](../glossary.md#ttft)"), 1)
        self.assertIn("比较 TTFT 时要固定条件。", output)

    def test_fenced_code_and_inline_code_are_unchanged(self):
        lines = ["```python", "print('TTFT')", "```", "`TTFT` 是字段名。"]
        self.assertEqual(BOOK.TermAnnotator().annotate_lines(lines), lines)

    def test_existing_definitions_are_not_repeated(self):
        line = "TTFT（首 token 延迟）量的是第一次反馈。"
        self.assertEqual(BOOK.TermAnnotator().annotate_lines([line]), [line])

    def test_flashattention_is_not_translated_as_storage_flash(self):
        self.assertNotEqual(self.entry("flash_attention")["zh"], "闪存注意力")

    def test_glossary_has_one_explicit_anchor_per_entry(self):
        self.assertTrue(hasattr(BOOK, "render_glossary"), "missing glossary page renderer")
        output = BOOK.render_glossary()
        anchors = re.findall(r"\{\s*#([a-z0-9_]+)\s*\}", output)
        expected = [item["id"] for item in BOOK.TERM_GLOSSARY]
        self.assertEqual(anchors, expected)
        self.assertEqual(len(anchors), len(set(anchors)))
        self.assertNotIn("闪存注意力", output)

    def test_guidance_covers_every_existing_chapter(self):
        self.assertTrue(hasattr(BOOK, "CHAPTER_GUIDANCE"), "missing chapter starting guidance")
        self.assertEqual(set(BOOK.CHAPTER_GUIDANCE), set(range(1, 32)))
        for number, fields in BOOK.CHAPTER_GUIDANCE.items():
            with self.subTest(chapter=number):
                self.assertEqual(set(fields), {"environment", "prerequisite", "deliverable"})
                self.assertTrue(all(value.strip() for value in fields.values()))

    def test_revised_titles_keep_chapter_ids(self):
        titles = {row[0]: row[2] for row in BOOK.CHAPTERS}
        self.assertEqual(sorted(titles), list(range(1, 32)))
        expected = {
            6: "推理瓶颈判断：计算、带宽与启动开销",
            9: "推理服务监控：Prometheus 与 Grafana",
            11: "容量评估：并发曲线、SLO 与抢占边界",
            18: "TensorRT-LLM：执行后端与三引擎对比",
            19: "GEMM 的四版实现：优化、计时与归因",
            29: "迁移前检查与实验设计：从 Triton 到 Ascend",
        }
        for number, title in expected.items():
            with self.subTest(chapter=number):
                self.assertEqual(titles[number], title)
                chapter = BOOK.Chapter(number, 1, title, "part1", 0)
                self.assertEqual(chapter.slug, f"chapter-{number:02d}")


if __name__ == "__main__":
    unittest.main()
