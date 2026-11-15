#!/usr/bin/env python3
"""Prepare only public sources, and reset only verified generated directories.

The shell build entry delegates destructive operations here so a symlink or
an incomplete project cannot redirect cleanup away from the checkout.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _project_root(root: Path) -> Path:
    root = Path(root).resolve()
    required_files = ("README.md", "mkdocs.yml", "tools/build_book.py")
    required_directories = ("docs", "assets", "tracker")
    missing = [name for name in required_files if not (root / name).is_file()]
    missing += [name for name in required_directories if not (root / name).is_dir()]
    if missing:
        raise ValueError(f"Not a complete book checkout: {root}; missing {', '.join(missing)}")
    return root


def _reset_output(root: Path, name: str) -> Path:
    if name not in {"site-src", "site"}:
        raise ValueError(f"Not a generated directory: {name!r}")
    target = root / name
    # Check the resolved absolute target immediately before any removal.
    if target.is_symlink() or target.resolve() != target or target.parent != root:
        raise ValueError(f"Refusing redirected output directory: {target}")
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"Expected a generated directory, not a file: {target}")
        shutil.rmtree(target)
    target.mkdir()
    return target


def stage_site(root: Path = ROOT) -> Path:
    root = _project_root(root)
    output = _reset_output(root, "site-src")
    shutil.copy2(root / "README.md", output / "README.md")

    def ignore(directory: str, names: list[str]) -> set[str]:
        excluded = set(shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")(directory, names))
        if Path(directory) == root / "docs":
            excluded.add("helm")
        return excluded

    for name in ("docs", "assets", "tracker"):
        shutil.copytree(root / name, output / name, ignore=ignore, symlinks=True)
    return output


def clean_site(root: Path = ROOT) -> Path:
    return _reset_output(_project_root(root), "site")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "clean-site"), nargs="?", default="stage")
    args = parser.parse_args()
    operation = stage_site if args.action == "stage" else clean_site
    print(f"Verified generated directory: {operation()}")


if __name__ == "__main__":
    main()
