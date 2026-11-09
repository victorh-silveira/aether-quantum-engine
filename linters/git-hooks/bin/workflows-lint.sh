#!/usr/bin/env bash
set -euo pipefail

if command -v actionlint >/dev/null 2>&1; then
  actionlint .github/workflows/ci.yml
else
  echo '[workflows] actionlint ausente; job CI - Workflows valida no GitHub'
fi
