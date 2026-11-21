#!/usr/bin/env bash
# Build (or preview) 《LLM 推理工程：原理与方法》 MkDocs site from Markdown sources.
#
# Usage:
#   bash site-build.sh        # build into site/
#   bash site-build.sh serve  # build staging tree, then serve on localhost:8010
set -euo pipefail
cd "$(dirname "$0")"

MKDOCS="${MKDOCS:-mkdocs}"
if ! command -v "$MKDOCS" >/dev/null 2>&1 && [[ -x .venv-mkdocs/bin/mkdocs ]]; then
  MKDOCS=.venv-mkdocs/bin/mkdocs
fi
PYTHON="${PYTHON:-python3}"

"$PYTHON" tools/stage_site.py stage
"$PYTHON" tools/build_book.py

if [[ "${1:-build}" == "serve" ]]; then
  exec "$MKDOCS" serve -f mkdocs.yml -a "${MKDOCS_ADDR:-127.0.0.1:8010}"
fi

"$PYTHON" tools/stage_site.py clean-site
"$MKDOCS" build -f mkdocs.yml
