#!/usr/bin/env bash
# Build (or preview) the MkDocs site from the repo's Markdown.
#
# MkDocs forbids docs_dir = repo root, so assemble a staging tree first
# (same layout as the repo, so internal links need no changes), then build.
#
# Usage:
#   bash site-build.sh        # build into site/
#   bash site-build.sh serve  # build staging tree, then serve on localhost:8000
#
# Needs the dependencies from requirements.txt:
#   python3 -m venv .venv-mkdocs
#   .venv-mkdocs/bin/pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"

# Prefer mkdocs from PATH (CI: pip install); fall back to the local venv.
MKDOCS="${MKDOCS:-mkdocs}"
if ! command -v "$MKDOCS" >/dev/null 2>&1 && [[ -x .venv-mkdocs/bin/mkdocs ]]; then
  MKDOCS=.venv-mkdocs/bin/mkdocs
fi

rm -rf site-src
mkdir -p site-src
cp README.md progress.md site-src/
cp -r docs assets tracker site-src/

if [[ "${1:-build}" == "serve" ]]; then
  # Note: edits to README.md / docs/ need a re-run of this script to reach site-src/.
  exec "$MKDOCS" serve -f mkdocs.yml
fi

rm -rf site
"$MKDOCS" build -f mkdocs.yml
