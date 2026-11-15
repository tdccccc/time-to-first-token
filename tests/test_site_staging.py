"""Staging must exclude private plans and only reset verified build outputs."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools/stage_site.py"


class SiteStagingTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(HELPER.is_file(), "missing safe, testable site staging helper")
        spec = importlib.util.spec_from_file_location("site_staging_under_test", HELPER)
        self.stage = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.stage
        spec.loader.exec_module(self.stage)
        self.temp = tempfile.TemporaryDirectory(prefix="dsh-book-staging-")
        self.addCleanup(self.cleanup_fixture)
        self.root = Path(self.temp.name).resolve()
        (self.root / "README.md").write_text("# fixture\n", encoding="utf-8")
        (self.root / "mkdocs.yml").write_text("site_name: fixture\n", encoding="utf-8")
        for directory in ("docs/modules", "docs/helm/private", "assets", "tracker", "tools"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        (self.root / "tools/build_book.py").write_text("# fixture\n", encoding="utf-8")
        (self.root / "docs/modules/example.md").write_text("# example\n", encoding="utf-8")
        (self.root / "docs/helm/private/goal.md").write_text("private plan\n", encoding="utf-8")
        (self.root / "assets/fixture.svg").write_text("<svg/>\n", encoding="utf-8")
        (self.root / "tracker/index.html").write_text("fixture\n", encoding="utf-8")
        cache = self.root / "docs/modules/__pycache__"
        cache.mkdir()
        (cache / "fixture.pyc").write_bytes(b"cache")

    def cleanup_fixture(self):
        # The target is the fresh temporary project fixture, never the checkout.
        target = Path(self.temp.name).resolve()
        self.assertEqual(target, self.root)
        self.assertEqual(target.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(target.name.startswith("dsh-book-staging-"))
        self.temp.cleanup()

    def test_staging_copies_public_sources_but_not_plans_or_caches(self):
        output = self.stage.stage_site(self.root)
        self.assertEqual(output, self.root / "site-src")
        self.assertTrue((output / "README.md").is_file())
        self.assertTrue((output / "docs/modules/example.md").is_file())
        self.assertTrue((output / "assets/fixture.svg").is_file())
        self.assertTrue((output / "tracker/index.html").is_file())
        self.assertFalse((output / "docs/helm").exists())
        self.assertFalse((output / "docs/modules/__pycache__").exists())

    def test_repeat_staging_resets_only_staging(self):
        output = self.stage.stage_site(self.root)
        stale = output / "stale.html"
        stale.write_text("old output\n", encoding="utf-8")
        self.stage.stage_site(self.root)
        self.assertFalse(stale.exists())
        self.assertTrue((self.root / "docs/modules/example.md").is_file())
        self.assertTrue((self.root / "docs/helm/private/goal.md").is_file())

    def test_clean_site_preserves_sources(self):
        output = self.root / "site"
        output.mkdir()
        (output / "old.html").write_text("old output\n", encoding="utf-8")
        self.assertEqual(self.stage.clean_site(self.root), output)
        self.assertTrue(output.is_dir())
        self.assertEqual(list(output.iterdir()), [])
        self.assertTrue((self.root / "docs/modules/example.md").is_file())

    def test_symlink_output_is_rejected_before_deletion(self):
        outside = self.root / "not-a-build-output"
        outside.mkdir()
        sentinel = outside / "keep.txt"
        sentinel.write_text("keep\n", encoding="utf-8")
        (self.root / "site").symlink_to(outside, target_is_directory=True)
        with self.assertRaises((ValueError, RuntimeError)):
            self.stage.clean_site(self.root)
        self.assertTrue(sentinel.is_file())

    def test_non_project_is_rejected_before_reset(self):
        incomplete = self.root / "incomplete"
        incomplete.mkdir()
        output = incomplete / "site-src"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("keep\n", encoding="utf-8")
        with self.assertRaises((ValueError, RuntimeError)):
            self.stage.stage_site(incomplete)
        self.assertTrue(sentinel.is_file())


if __name__ == "__main__":
    unittest.main()
