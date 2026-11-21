"""Offline checks exercise local links without fetching external sites."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("site_validation_under_test", ROOT / "tools/check_site.py")
CHECK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECK
SPEC.loader.exec_module(CHECK)


class SiteValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dsh-book-validation-")
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.cleanup_fixture)
        (self.root / "docs/page").mkdir(parents=True)
        (self.root / "assets").mkdir()
        (self.root / "docs/page/index.html").write_text('<h1 id="target">Page</h1><div id="首字"></div>', encoding="utf-8")
        (self.root / "assets/ok.svg").write_text("<svg/>", encoding="utf-8")

    def cleanup_fixture(self):
        target = Path(self.temp.name).resolve()
        self.assertEqual(target, self.root)
        self.assertTrue(target.name.startswith("dsh-book-validation-"))
        self.assertEqual(target.parent, Path(tempfile.gettempdir()).resolve())
        self.temp.cleanup()

    def index(self, content):
        (self.root / "index.html").write_text(content, encoding="utf-8")

    def test_valid_directory_assets_and_anchors(self):
        self.index('<h1 id="home">Home</h1><a href="docs/page/#target">page</a>'
                   '<a href="/time-to-first-token/docs/page/#target">absolute</a>'
                   '<a href="docs/page/#%E9%A6%96%E5%AD%97">unicode</a>'
                   '<img src="assets/ok.svg"><a href="#home">home</a>'
                   '<a href="https://example.invalid/unfetched">external</a>')
        self.assertEqual(CHECK.check_links(self.root), [])

    def test_missing_target_and_anchor_are_reported(self):
        self.index('<a href="missing/">missing</a><a href="docs/page/#missing">anchor</a>')
        problems = CHECK.check_links(self.root)
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("missing local target" in problem for problem in problems))
        self.assertTrue(any("missing local anchor" in problem for problem in problems))

    def test_links_cannot_escape_generated_site(self):
        self.index('<a href="../escape.html">outside</a>')
        problems = CHECK.check_links(self.root)
        self.assertEqual(len(problems), 1)
        self.assertIn("escapes site", problems[0])

    def test_title_and_progress_ids_keep_semantic_text(self):
        page = CHECK.Page('<div class="md-sidebar md-sidebar--secondary"></div>'
                          '<h1>第 6 章 <em>推理瓶颈</em><a>¶</a></h1>'
                          '<input type="checkbox" data-chapter="6">')
        self.assertEqual(page.title, "第 6 章 推理瓶颈")
        self.assertEqual(page.chapters, [6])
        self.assertTrue(page.has_toc_sidebar)
        self.assertFalse(page.has_notes_panel)


if __name__ == "__main__":
    unittest.main()
