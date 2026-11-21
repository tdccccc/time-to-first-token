"""Regression checks for compact glossary annotations, math preservation, and 23-chapter IDs."""
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

    def test_fenced_code_inline_code_and_math_are_unchanged(self):
        lines = [
            "```python",
            "print('TTFT')",
            "```",
            "`TTFT` 是字段名。",
            "公式 $TTFT + ITL$ 不应被插入链接。",
            "$$",
            "TTFT = T_{queue} + T_{prefill}",
            "$$",
        ]
        self.assertEqual(BOOK.TermAnnotator().annotate_lines(lines), lines)

    def test_existing_definitions_are_not_repeated(self):
        line = "TTFT（首 token 延迟）量的是第一次反馈。"
        self.assertEqual(BOOK.TermAnnotator().annotate_lines([line]), [line])

    def test_scalarization_is_not_annotated_as_quantization(self):
        line = "不对齐可能触发编译器标量化处理。"
        self.assertEqual(BOOK.TermAnnotator().annotate_lines([line]), [line])

    def test_flashattention_is_not_translated_as_storage_flash(self):
        self.assertNotEqual(self.entry("flash_attention")["zh"], "闪存注意力")

    def test_glossary_has_one_explicit_anchor_per_entry(self):
        output = BOOK.render_glossary()
        anchors = re.findall(r"\{\s*#([a-z0-9_]+)\s*\}", output)
        expected = [item["id"] for item in BOOK.TERM_GLOSSARY]
        self.assertEqual(anchors, expected)
        self.assertEqual(len(anchors), len(set(anchors)))
        self.assertNotIn("闪存注意力", output)

    def test_twenty_three_canonical_chapters_and_summaries(self):
        titles = {row[0]: row[2] for row in BOOK.CHAPTERS}
        self.assertEqual(sorted(titles), list(range(1, 24)))
        self.assertEqual(set(BOOK.CHAPTER_SUMMARIES), set(range(1, 24)))
        self.assertTrue(all(v.strip() for v in BOOK.CHAPTER_SUMMARIES.values()))
        self.assertEqual(titles[1], "推理为什么慢：Roofline 与三区间直觉")
        self.assertEqual(titles[5], "vLLM：一条请求如何穿过推理引擎")
        self.assertEqual(titles[12], "GPU 执行模型与 GEMM 四版优化阶梯")
        self.assertEqual(titles[19], "昇腾部署架构与 vllm-ascend 插件机制")
        self.assertEqual(titles[23], "跨芯片精度回归、性能归因与分析报告范式")


if __name__ == "__main__":
    unittest.main()
