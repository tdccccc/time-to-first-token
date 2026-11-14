#!/usr/bin/env bash
# Build the MkDocs site from the repo's Markdown (single source of truth).
# MkDocs forbids docs_dir = repo root, so assemble a staging tree first
# (same layout as the repo, so internal links need no changes), then build.
# Needs `mkdocs` and `mkdocs-material` on PATH:
#   pip install mkdocs mkdocs-material
set -euo pipefail
cd "$(dirname "$0")"
rm -rf site-src site
mkdir -p site-src
cp README.md progress.md site-src/
cp -r docs assets tracker site-src/
mkdocs build -f mkdocs.yml
